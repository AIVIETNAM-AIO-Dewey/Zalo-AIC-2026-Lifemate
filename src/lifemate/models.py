"""Local deterministic inference. Model imports are lazy for CPU-only audits."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import wave


def seed_everything(seed):
    # Set before importing torch / creating a CUDA context.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def versions():
    packages = ("torch", "transformers", "numpy", "accelerate", "safetensors")
    return {name: importlib.metadata.version(name) for name in packages}


class PhoWhisper:
    def __init__(self, model_id, device, revision=None):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.torch = torch
        self.model_id, self.device = str(model_id), device
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, revision=revision,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            attn_implementation="eager",
        ).to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        # Include loaded commit and config, not merely the user-facing repo name.
        self.identity = hashlib.sha256(json.dumps({
            "model": self.model_id, "config": self.model.config.to_dict(),
            "revision": revision, "generation": "vi/transcribe/greedy/max384/v1",
            "transformers": importlib.metadata.version("transformers"),
        }, sort_keys=True, default=str).encode()).hexdigest()

    def transcribe(self, path, remaining_seconds):
        import numpy as np
        with wave.open(str(path)) as audio:
            fmt = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
            if fmt != (16000, 1, 2):
                raise ValueError(f"Expected 16kHz mono PCM16, received {fmt}: {path}")
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if len(samples) > 30 * 16000:
            raise ValueError("Audio exceeds 30 seconds; implement explicit chunking before using this baseline")
        batch = self.processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        features = batch.input_features.to(device=self.device, dtype=self.model.dtype)
        attention_mask = batch.attention_mask.to(self.device)
        # generation_config may carry legacy forced_decoder_ids; language/task are explicit.
        with self.torch.inference_mode():
            output = self.model.generate(
                features, attention_mask=attention_mask, language="vi", task="transcribe", do_sample=False,
                num_beams=1, max_new_tokens=min(384, self.model.config.max_target_positions-8),
                max_time=max(0.01, remaining_seconds),
                return_timestamps=False,
            )
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()


class QwenAgentModel:
    def __init__(self, model_id, device, revision=None, context_limit=16384, max_new_tokens=384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not 1 <= context_limit <= 16384:
            raise ValueError("Competition requires total context <= 16384 tokens")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.model_id, self.device = str(model_id), device
        self.context_limit, self.max_new_tokens = context_limit, max_new_tokens
        self.params = sum(p.numel() for p in self.model.parameters())
        if self.params >= 5_000_000_000:
            raise ValueError(f"LLM must have <5B parameters, got {self.params}")

    def generate(self, messages, tools, remaining_seconds):
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_tokens = inputs["input_ids"].shape[-1]
        # Fail explicitly; silently dropping tool results/history can change the task.
        if input_tokens + self.max_new_tokens > self.context_limit:
            raise ValueError(f"Context limit: {input_tokens} input + {self.max_new_tokens} reserved output > {self.context_limit}")
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, num_beams=1,
                max_new_tokens=self.max_new_tokens, max_time=max(0.01, remaining_seconds),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, input_tokens:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), {
            "input_tokens": input_tokens, "output_tokens": len(generated),
            "seconds": time.perf_counter() - started,
            "output_cap_reached": len(generated) >= self.max_new_tokens,
        }
