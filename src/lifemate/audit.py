"""Reproducible metadata/audio audit and conversation-level train/dev manifests."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import wave

from .data import audio_path, audio_turns, category, dump_json, load_catalog, load_runtime, make_split, read_rows


def describe(root, split):
    rows = read_rows(root, split)
    catalog = load_catalog(root, split)
    durations, formats, failures, hashes = [], Counter(), [], Counter()
    pairs = []
    for row in rows:
        for turn in audio_turns(row):
            path = audio_path(root, split, turn["audio"])
            try:
                with wave.open(str(path)) as audio:
                    duration = audio.getnframes() / audio.getframerate()
                    durations.append(duration)
                    formats[f"{audio.getframerate()}Hz/{audio.getnchannels()}ch/{audio.getsampwidth()*8}bit"] += 1
                hashes[hashlib.sha256(path.read_bytes()).hexdigest()] += 1
                pairs.append({"sample_id": row["id"], "audio": turn["audio"],
                              "text": turn.get("audio_text"), "seconds": duration})
            except (OSError, wave.Error) as error:
                failures.append({"sample_id": row["id"], "audio": turn["audio"], "error": str(error)})
    ordered = sorted(durations)
    report = {
        "samples": len(rows), "audio_turns": len(pairs),
        "audio_files": len(list((Path(root) / split / "audio").rglob("*.wav"))),
        "hours": sum(durations) / 3600,
        "duration_seconds": {"median": statistics.median(durations),
                             "p95": ordered[min(len(ordered)-1, int(len(ordered)*0.95))],
                             "max": max(durations)},
        "formats": dict(formats), "audio_errors": failures,
        "duplicate_audio_copies": sum(n - 1 for n in hashes.values()),
        "history_user_turns": dict(sorted(Counter(len(audio_turns(r))-1 for r in rows).items())),
        "catalog_apps": len(catalog["apps"]),
        "catalog_tools": sum(len(a["tools"]) for a in catalog["apps"]),
        "mean_installed_tools_including_sys": statistics.mean(
            len({t for a in r["context"]["installed_apps"].values() for t in a["tools"]} | {"sys.ask_missing_info"}) for r in rows),
        "jsonl_sha256": hashlib.sha256((Path(root) / split / f"{split}.jsonl").read_bytes()).hexdigest(),
    }
    if split == "train":
        report["categories"] = dict(Counter(category(r) for r in rows))
        report["gold_call_counts"] = dict(sorted(Counter(len(r["label"]["tool_calls"]) for r in rows).items()))
        report["unique_gold_tool_sequences"] = len({tuple(c["name"] for c in r["label"]["tool_calls"]) for r in rows})
        report["unique_audio_texts"] = len({p["text"] for p in pairs})
    return report, pairs


def replay_gold(root):
    """TRAIN ONLY diagnostic. Agent inference never calls get_state()."""
    runtime = load_runtime(root, "train")
    errors, mismatches, noops = [], [], []
    rows = read_rows(root, "train")
    for row in rows:
        env = runtime.ToolEnv("train", row["id"], root=root)
        expected = env.get_state()
        expected.update(copy.deepcopy(row["label"]["final_state"]))
        for index, call in enumerate(row["label"]["tool_calls"]):
            result = env.call(call["name"], call["arguments"])
            if not result.get("ok"):
                errors.append({"id": row["id"], "call": index, "result": result})
            if result.get("done") is False:
                noops.append({"id": row["id"], "call": index, "result": result})
        if env.get_state_hash() != runtime.state_hash(expected):
            mismatches.append(row["id"])
    return {"samples": len(rows), "schema_errors": errors, "state_hash_mismatches": mismatches, "done_false": noops}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    reports = {}
    for split in ("train", "public_test"):
        reports[split], pairs = describe(args.data_root, split)
        if split == "train":
            train_ids, dev_ids = make_split(read_rows(args.data_root, split), args.seed)
            dump_json(args.output / "train_ids.json", train_ids)
            dump_json(args.output / "dev_ids.json", dev_ids)
            rng = random.Random(args.seed + 1)
            pilot_ids = []
            for kind in ("execute", "empty", "missing_info"):
                candidates = sorted(r["id"] for r in read_rows(args.data_root, split) if r["id"] in set(dev_ids) and category(r) == kind)
                pilot_ids.extend(rng.sample(candidates, min(16, len(candidates))))
            dump_json(args.output / "pilot_ids.json", sorted(pilot_ids))
            for name, ids in (("asr_train", set(train_ids)), ("asr_dev", set(dev_ids))):
                args.output.mkdir(parents=True, exist_ok=True)
                with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for pair in pairs:
                        if pair["sample_id"] in ids:
                            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            reports["split"] = {"seed": args.seed, "train_samples": len(train_ids), "dev_samples": len(dev_ids),
                                "pilot_samples": len(pilot_ids), "pilot_note": "Balanced debugging subset of dev, NOT a representative leaderboard estimate",
                                "grouping": "normalised identical latest audio_text; entire conversation kept together",
                                "limitations": "Not speaker-disjoint or template-disjoint; duplicate history utterances may cross splits.",
                                "categories": {name: dict(Counter(category(r) for r in read_rows(args.data_root, "train") if r["id"] in set(ids)))
                                               for name, ids in (("train", train_ids), ("dev", dev_ids))}}
    reports["gold_replay"] = replay_gold(args.data_root)
    train_tools = {t["name"] for a in load_catalog(args.data_root, "train")["apps"] for t in a["tools"]}
    public_tools = {t["name"] for a in load_catalog(args.data_root, "public_test")["apps"] for t in a["tools"]}
    reports["catalog_difference"] = {"train_only": sorted(train_tools-public_tools), "public_only": sorted(public_tools-train_tools)}
    dump_json(args.output / "audit.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
