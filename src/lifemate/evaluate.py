"""Train/dev scoring from the published LifeMate rules, using BTC runtime replay.

The server scorer is not shipped. Tokenisation uses NFC/lowercase whitespace
tokens with multiplicity; punctuation is preserved and accents are not removed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import statistics
import unicodedata

from .data import audio_turns, catalog_map, category, dump_json, load_catalog, load_runtime, read_rows, select_rows


SCORER_VERSION = "lifemate-published-rules-v1"
RULES_URL = "https://challenge.zalo.ai/portal/lifemate"


def fuzzy_tokens(text):
    return unicodedata.normalize("NFC", text.lower()).split()


def arguments_match(predicted, gold, schema):
    """Only fields explicitly marked x-match:fuzzy receive text normalisation.

    Key presence and optional arguments remain significant; no default values
    are inserted. Array order remains significant, unlike the call-list order.
    """
    if schema.get("x-match") == "fuzzy":
        if not isinstance(predicted, str) or not isinstance(gold, str):
            return False
        p, g = fuzzy_tokens(predicted), fuzzy_tokens(gold)
        if not p or not g:
            return p == g
        overlap = sum((Counter(p) & Counter(g)).values())
        # Strictly greater than 0.6, without floating-point boundary rounding.
        return 10 * overlap > 3 * (len(p) + len(g))
    if isinstance(predicted, dict) and isinstance(gold, dict):
        if predicted.keys() != gold.keys():
            return False
        props = schema.get("properties", {})
        return all(arguments_match(predicted[k], gold[k], props.get(k, {})) for k in gold)
    if isinstance(predicted, list) and isinstance(gold, list):
        return len(predicted) == len(gold) and all(
            arguments_match(p, g, schema.get("items", {})) for p, g in zip(predicted, gold))
    if isinstance(predicted, (dict, list)) or isinstance(gold, (dict, list)):
        return False
    # JSON booleans must not compare equal to numbers (True == 1 in Python).
    return predicted == gold and isinstance(predicted, bool) == isinstance(gold, bool)


def matched_call_count(predicted, gold, specs, valid_predictions=None):
    """Maximum one-to-one matching; duplicates/rejected calls stay in precision.

    Greedy matching can undercount when fuzzy candidates overlap. An augmenting
    path can reassign an earlier match so the result does not depend on order.
    """
    valid_predictions = [True] * len(predicted) if valid_predictions is None else valid_predictions
    if len(valid_predictions) != len(predicted):
        raise ValueError("One validity flag is required per submitted call")
    edges = []
    for call, valid in zip(predicted, valid_predictions):
        spec = specs.get(call["name"])
        edges.append([i for i, ref in enumerate(gold)
                      if valid and spec is not None and call["name"] == ref["name"]
                      and arguments_match(call["arguments"], ref["arguments"], spec["parameters"])])
    owners = {}

    def augment(index, seen):
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in owners or augment(owners[target], seen):
                owners[target] = index
                return True
        return False

    return sum(augment(i, set()) for i in range(len(predicted)))


def score_calls(predicted, gold, specs, state_match, valid_predictions=None):
    """Published score. Agent stop reasons and elapsed time are diagnostics only."""
    matched = matched_call_count(predicted, gold, specs, valid_predictions)
    f1 = 2 * matched / (len(predicted) + len(gold)) if gold else float(not predicted)
    score = f1 if not gold or state_match else 0.0
    return {"matched_calls": matched, "unmatched_calls": len(predicted) - matched,
            "call_f1": f1, "score": score}


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
    parser.add_argument("--output", type=Path, help="Write re-scored reports to a NEW directory; preserve the original run")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output = args.output if args.output is not None else args.run
    if args.output is not None and output.exists():
        parser.error("Output directory already exists; choose a new directory")
    config = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    if config.get("split") != "train":
        parser.error("Labels may be evaluated only on train; public/private tests are not used for tuning")
    rows = select_rows(read_rows(args.data_root, "train"), args.ids, args.limit)
    if not args.ids and args.limit is None:
        selected = set(config["sample_ids"])
        rows = [r for r in rows if r["id"] in selected]
    runtime = load_runtime(args.data_root, "train")
    specs = catalog_map(load_catalog(args.data_root, "train"))
    predictions = {}
    for line in (args.run / "submission.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] in predictions:
            raise ValueError(f"Duplicate prediction: {row['id']}")
        predictions[row["id"]] = row
    diagnostic_path = args.run / "diagnostics.jsonl"
    diagnostics = {d["id"]: d for d in (json.loads(l) for l in diagnostic_path.read_text(encoding="utf-8").splitlines())} if diagnostic_path.exists() else {}
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
                            "state_match": False, "exact_call_f1": 0.0, "state_gated_exact_f1": 0.0,
                            "matched_calls": 0, "unmatched_calls": 0, "call_f1": 0.0, "score": 0.0})
            continue
        env = runtime.ToolEnv("train", row["id"], root=args.data_root)
        expected = env.get_state()  # evaluation harness only; never passed into the agent
        expected.update(copy.deepcopy(row["label"]["final_state"]))
        schema_errors, valid_predictions = [], []
        for call in prediction["tool_calls"]:
            check = env.validate(call["name"], call["arguments"])
            valid_predictions.append(check["ok"])
            if not check["ok"]:
                schema_errors.append(check["error"])
            env.call(call["name"], call["arguments"])
        hash_reproducible = env.get_state_hash() == prediction["state_hash"]
        # The organisers replay the calls and their result is final. Keep a
        # submitted-hash mismatch visible, but use replay for the state gate.
        state_match = env.get_state_hash() == runtime.state_hash(expected)
        f1 = exact_call_f1(prediction["tool_calls"], row["label"]["tool_calls"])
        # Empty-gold rule is special: an empty submission alone gets 1.
        gated = (float(not prediction["tool_calls"]) if not row["label"]["tool_calls"] else f1 * state_match)
        scored = score_calls(prediction["tool_calls"], row["label"]["tool_calls"], specs, state_match, valid_predictions)
        results.append({"id": row["id"], "category": category(row), "state_match": state_match,
                        "hash_reproducible": hash_reproducible, "exact_call_f1": f1,
                        "state_gated_exact_f1": gated, **scored,
                        "schema_errors": schema_errors, "stop_reason": detail.get("stop_reason"),
                        "seconds": detail.get("seconds")})
    selected_ids = {row["id"] for row in rows}
    selected_diagnostics = [d for key, d in diagnostics.items() if key in selected_ids]
    seconds = sorted(d["seconds"] for d in selected_diagnostics if d.get("seconds") is not None)
    summary = {
        "notice": "Local implementation of the published BTC rules; not an official server score. Tokeniser details are not published.",
        "scorer_version": SCORER_VERSION, "rules_url": RULES_URL,
        "scoring": {"fuzzy_fields": "catalog x-match:fuzzy", "threshold": "> 0.6",
                    "tokenisation": "NFC(lowercase(text)).split(); multiset tokens; preserve punctuation and accents",
                    "matching": "maximum one-to-one; call order ignored; rejected calls unmatched",
                    "state": "ordered BTC runtime replay is authoritative; submitted hash mismatch reported separately",
                    "stop_reason_affects_score": False},
        "mode": config.get("mode"), "asr_only": config.get("asr_only", False),
        "samples": len(rows), "state_match_rate": statistics.mean(r["state_match"] for r in results) if results else None,
        "score_percent": 100*statistics.mean(r["score"] for r in results) if results else None,
        "mean_call_f1": statistics.mean(r["call_f1"] for r in results) if results else None,
        "mean_exact_call_f1": statistics.mean(r["exact_call_f1"] for r in results) if results else None,
        "state_gated_exact_f1_percent": 100*statistics.mean(r["state_gated_exact_f1"] for r in results) if results else None,
        "by_category": {cat: {"samples": sum(r["category"] == cat for r in results),
                              "state_match_rate": statistics.mean(r["state_match"] for r in results if r["category"] == cat),
                              "score_percent": 100*statistics.mean(r["score"] for r in results if r["category"] == cat),
                              "state_gated_exact_f1_percent": 100*statistics.mean(r["state_gated_exact_f1"] for r in results if r["category"] == cat)}
                        for cat in sorted({r["category"] for r in results})},
        "asr": {"turns_measured": transcribed, "reference_turns": reference_turns,
                "coverage": transcribed/reference_turns if reference_turns else 0,
                "wer": word_errors/words if words else None, "cer_including_spaces": char_errors/chars if chars else None,
                "normalisation": "NFC lowercase; punctuation->spaces; whitespace collapsed; Vietnamese syllable tokens for WER"},
        "stop_reasons": dict(Counter(d.get("stop_reason", "unknown") for d in selected_diagnostics)),
        "hash_mismatches": sum(not r.get("hash_reproducible", False) for r in results),
        "latency_over_30_seconds": sum(s > 30 for s in seconds),
        "latency_seconds": {"median": statistics.median(seconds), "p95": seconds[min(len(seconds)-1, int(len(seconds)*0.95))], "max": max(seconds)} if seconds else None,
        "latency_note": "Kaggle/local timing only. Oracle/cached-ASR runs are NOT live audio end-to-end timing; see each diagnostic live_audio_end_to_end field.",
    }
    dump_json(output / "evaluation.json", summary)
    dump_json(output / "per_sample_evaluation.json", results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
