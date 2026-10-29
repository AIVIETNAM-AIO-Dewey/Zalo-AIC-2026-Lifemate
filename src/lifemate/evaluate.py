"""Train/dev diagnostics. This is NOT the organisers' fuzzy-call scorer."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import statistics
import unicodedata

from .data import audio_turns, category, dump_json, load_runtime, read_rows, select_rows


def normalise_asr(text):
    # A declared diagnostic convention: NFC, lowercase, punctuation -> spaces.
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def exact_call_f1(predicted, gold):
    # Multiset: duplicate calls cannot match the same reference twice.
    key = lambda c: json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    p, g = Counter(map(key, predicted)), Counter(map(key, gold))
    if not gold:
        return float(not predicted)
    return 2 * sum((p & g).values()) / (len(predicted) + len(gold))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    if config.get("split") != "train":
        parser.error("Labels may be evaluated only on train; public/private tests are not used for tuning")
    rows = select_rows(read_rows(args.data_root, "train"), args.ids, args.limit)
    if not args.ids and args.limit is None:
        selected = set(config["sample_ids"])
        rows = [r for r in rows if r["id"] in selected]
    runtime = load_runtime(args.data_root, "train")
    predictions = {}
    for line in (args.run / "submission.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] in predictions:
            raise ValueError(f"Duplicate prediction: {row['id']}")
        predictions[row["id"]] = row
    diagnostics = {d["id"]: d for d in (json.loads(l) for l in (args.run / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines())}
    results, word_errors, char_errors, words, chars, transcribed, reference_turns = [], 0, 0, 0, 0, 0, 0
    for row in rows:
        prediction, detail = predictions.get(row["id"]), diagnostics.get(row["id"], {})
        texts = detail.get("transcripts", {})
        for turn in audio_turns(row):
            reference_turns += 1
            if turn["audio"] not in texts:
                continue
            reference, hypothesis = normalise_asr(turn["audio_text"]), normalise_asr(texts[turn["audio"]])
            word_errors += edit_distance(reference.split(), hypothesis.split())
            char_errors += edit_distance(reference, hypothesis)
            words += len(reference.split())
            chars += len(reference)
            transcribed += 1
        if config.get("asr_only"):
            continue
        if prediction is None:
            results.append({"id": row["id"], "category": category(row), "missing_prediction": True,
                            "state_match": False, "exact_call_f1": 0.0, "state_gated_exact_f1": 0.0})
            continue
        env = runtime.ToolEnv("train", row["id"], root=args.data_root)
        expected = env.get_state()  # evaluation harness only; never passed into the agent
        expected.update(copy.deepcopy(row["label"]["final_state"]))
        schema_errors = []
        for call in prediction["tool_calls"]:
            check = env.validate(call["name"], call["arguments"])
            if not check["ok"]:
                schema_errors.append(check["error"])
            env.call(call["name"], call["arguments"])
        hash_reproducible = env.get_state_hash() == prediction["state_hash"]
        state_match = hash_reproducible and env.get_state_hash() == runtime.state_hash(expected)
        f1 = exact_call_f1(prediction["tool_calls"], row["label"]["tool_calls"])
        failed = detail.get("stop_reason") in ("timeout", "error", "repair_limit", "call_limit")
        # Empty-gold rule is special: an empty submission alone gets 1.
        gated = (float(not prediction["tool_calls"]) if not row["label"]["tool_calls"] else f1 * state_match)
        results.append({"id": row["id"], "category": category(row), "state_match": state_match,
                        "hash_reproducible": hash_reproducible, "exact_call_f1": f1,
                        "state_gated_exact_f1": gated if not failed else 0.0,
                        "schema_errors": schema_errors, "stop_reason": detail.get("stop_reason"),
                        "seconds": detail.get("seconds")})
    seconds = sorted(d["seconds"] for d in diagnostics.values() if d.get("seconds") is not None)
    summary = {
        "notice": "Local exact-call diagnostic, NOT official score; fuzzy x-match is not reproduced.",
        "mode": config.get("mode"), "asr_only": config.get("asr_only", False),
        "samples": len(rows), "state_match_rate": statistics.mean(r["state_match"] for r in results) if results else None,
        "mean_exact_call_f1": statistics.mean(r["exact_call_f1"] for r in results) if results else None,
        "state_gated_exact_f1_percent": 100*statistics.mean(r["state_gated_exact_f1"] for r in results) if results else None,
        "by_category": {cat: {"samples": sum(r["category"] == cat for r in results),
                              "state_match_rate": statistics.mean(r["state_match"] for r in results if r["category"] == cat),
                              "state_gated_exact_f1_percent": 100*statistics.mean(r["state_gated_exact_f1"] for r in results if r["category"] == cat)}
                        for cat in sorted({r["category"] for r in results})},
        "asr": {"turns_measured": transcribed, "reference_turns": reference_turns,
                "coverage": transcribed/reference_turns if reference_turns else 0,
                "wer": word_errors/words if words else None, "cer_including_spaces": char_errors/chars if chars else None,
                "normalisation": "NFC lowercase; punctuation->spaces; whitespace collapsed; Vietnamese syllable tokens for WER"},
        "stop_reasons": dict(Counter(d.get("stop_reason", "unknown") for d in diagnostics.values())),
        "latency_seconds": {"median": statistics.median(seconds), "p95": seconds[min(len(seconds)-1, int(len(seconds)*0.95))], "max": max(seconds)} if seconds else None,
        "latency_note": "Kaggle/local timing only. Oracle/cached-ASR runs are NOT live audio end-to-end timing; see each diagnostic live_audio_end_to_end field.",
    }
    dump_json(args.run / "evaluation.json", summary)
    dump_json(args.run / "per_sample_evaluation.json", results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
