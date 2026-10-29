"""Run PhoWhisper -> Qwen3 no-thinking -> official ToolEnv, locally/offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .agent import run_agent
from .data import audio_path, audio_turns, dump_json, load_catalog, load_runtime, read_rows, select_rows
from .models import PhoWhisper, QwenAgentModel, seed_everything, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True, help="A NEW run directory")
    parser.add_argument("--mode", choices=("audio", "oracle"), default="audio")
    parser.add_argument("--asr-only", action="store_true")
    parser.add_argument("--asr-model", default="vinai/PhoWhisper-medium")
    parser.add_argument("--asr-revision")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--llm-revision")
    parser.add_argument("--asr-device", default="cuda:0")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--asr-cache", type=Path, help="Optional diagnostic cache; cached timing is NOT live end-to-end timing")
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--time-budget", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "oracle" and args.split != "train":
        parser.error("Oracle mode is TRAIN ONLY; public/private test must use audio")
    if args.time_budget <= 0 or args.time_budget > 30:
        parser.error("Time budget must be >0 and <=30 seconds")
    if args.max_new_tokens < 1 or args.max_calls < 1:
        parser.error("max-new-tokens and max-calls must be positive")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new name to preserve runs")
    rows = select_rows(read_rows(args.data_root, args.split), args.ids, args.limit)
    if not rows:
        parser.error("No samples selected")
    args.output.mkdir(parents=True)
    seed_everything(args.seed)
    runtime = load_runtime(args.data_root, args.split)
    catalog = load_catalog(args.data_root, args.split)
    asr = PhoWhisper(args.asr_model, args.asr_device, args.asr_revision) if args.mode == "audio" else None
    llm = None if args.asr_only else QwenAgentModel(
        args.llm_model, args.llm_device, args.llm_revision, args.context_limit, args.max_new_tokens)
    if asr and llm and asr.params + llm.params >= 6_000_000_000:
        raise ValueError("System must have <6B parameters in total")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"versions": versions(), "sample_ids": [r["id"] for r in rows],
                   "asr_parameters": asr.params if asr else None,
                   "llm_parameters": llm.params if llm else None,
                   "asr_commit": getattr(asr.model.config, "_commit_hash", None) if asr else None,
                   "llm_commit": getattr(llm.model.config, "_commit_hash", None) if llm else None,
                   "decoding": "greedy; enable_thinking=False",
                   "latency_note": "Local measurement only; no A30 verification. max_time may overrun by a generation step."})
    dump_json(args.output / "run_config.json", config)
    cache = json.loads(args.asr_cache.read_text(encoding="utf-8")) if args.asr_cache else {}
    observed = {}
    with (args.output / "submission.jsonl").open("w", encoding="utf-8") as submission, \
         (args.output / "diagnostics.jsonl").open("w", encoding="utf-8") as diagnostics:
        for index, row in enumerate(rows, 1):
            started = time.perf_counter()
            deadline = started + args.time_budget
            # One and only one ToolEnv for this sample, even if inference fails.
            env = runtime.ToolEnv(args.split, row["id"], root=args.data_root)
            text_by_audio, cache_hits, asr_trace = {}, 0, []
            detail = {}
            try:
                for turn in audio_turns(row):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("Time budget exhausted during ASR/history")
                    if args.mode == "oracle":
                        text_by_audio[turn["audio"]] = turn["audio_text"]
                        continue
                    path = audio_path(args.data_root, args.split, turn["audio"])
                    audio_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                    key = hashlib.sha256((asr.identity + audio_hash).encode()).hexdigest()
                    cached = key in cache
                    if cached:
                        record = cache[key]
                        cache_hits += 1
                    else:
                        asr_started = time.perf_counter()
                        text = asr.transcribe(path, deadline - asr_started)
                        record = {"text": text, "seconds": time.perf_counter()-asr_started,
                                  "audio_sha256": audio_hash, "model_identity": asr.identity}
                        # Never persist an ASR output if generation ran out of time.
                        if time.perf_counter() >= deadline:
                            raise TimeoutError("ASR exceeded sample time budget; output not cached")
                        cache[key] = record
                    observed[f"{args.split}/{turn['audio']}"] = record
                    text_by_audio[turn["audio"]] = record["text"]
                    asr_trace.append({"audio": turn["audio"], "cached": cached, **record})
                if args.asr_only:
                    detail = {"stop_reason": "asr_only", "trace": []}
                else:
                    # Explicit whitelist: labels, transcripts and all other train-only fields never enter the agent.
                    agent_row = {k: row[k] for k in ("id", "audio", "context", "history")}
                    _, detail = run_agent(agent_row, text_by_audio, env, catalog, llm, deadline, args.max_calls)
            except Exception as error:
                detail.update({"stop_reason": "error", "error": f"{type(error).__name__}: {error}"})
            elapsed = time.perf_counter() - started
            if elapsed > args.time_budget:
                detail["stop_reason"] = "timeout"
            result = {"id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
            if not args.asr_only:
                submission.write(json.dumps(result, ensure_ascii=False) + "\n")
                submission.flush()
            detail.update({"id": row["id"], "seconds": elapsed,
                           "mode": args.mode, "cache_hits": cache_hits,
                           "live_audio_end_to_end": args.mode == "audio" and cache_hits == 0 and not args.asr_only,
                           "transcripts": text_by_audio, "asr": asr_trace})
            diagnostics.write(json.dumps(detail, ensure_ascii=False) + "\n")
            diagnostics.flush()
            dump_json(args.output / "asr_cache.json", cache)
            dump_json(args.output / "transcripts.json", observed)
            print(f"{index}/{len(rows)} {row['id']}: {detail['stop_reason']}, {len(result['tool_calls'])} calls, {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
