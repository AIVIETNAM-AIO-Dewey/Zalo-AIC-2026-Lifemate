"""Build a text SFT dataset from the two LifeMate exports and the fixed BTC split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile
import unicodedata
from zipfile import ZipFile

from .agent import make_messages, parse_action, tools_for
from .data import audio_turns, catalog_map, load_catalog, load_runtime, read_rows
from .replay_drafts import canonical, digest, replay_draft


REVIEWED_EXPORT = "reviewed"
DRAFT_EXPORT = "drafts"
BTC = "btc"
SYNTHETIC = "synthetic"
MODEL_ID = "Qwen/Qwen3-4B"
TOKENIZER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
CONTEXT_LIMIT = 16384
RESERVED_OUTPUT = 384
SEED = 42
CONTROL_TOKENS = re.compile(r"<\|[^\s<>]+\|>|</?(?:tool_call|tool_response|think)>")
REFERENT = re.compile(r"(?<![\w.-]){}(?![\w.-])")


def sha256_file(path):
    hash_ = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def normal_text(value):
    value = unicodedata.normalize("NFC", value).casefold()
    return " ".join(re.findall(r"\w+", value))


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def zip_rows(archive, member, rejected):
    for line_number, line in enumerate(archive.read(member).decode("utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected object")
                canonical(row)
            except ValueError:
                rejected.append({
                    "id": None, "reason": "malformed_archive_jsonl",
                    **source_path(archive.filename, member, line_number),
                })
                continue
            yield line_number, row


def source_path(archive, member, line):
    return {"archive": Path(archive).name, "member": member, "line": line}


def read_reviewed(path):
    """Select exact versions approved in review exports, never intermediate drafts."""
    inventory, votes, candidates = {}, defaultdict(set), defaultdict(list)
    raw_count = 0
    rejected = []
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            filename = PurePosixPath(name).name.lower()
            if filename in {"codex_review.jsonl", "review_decisions.jsonl"}:
                for _, row in zip_rows(archive, name, rejected):
                    if row.get("draft_sha256"):
                        votes[(row.get("id"), row["draft_sha256"])].add(
                            row.get("classification", row.get("decision"))
                        )
            elif filename == "usable_auto.jsonl" or (name.startswith("Data/") and filename.endswith(".jsonl")):
                for line, row in zip_rows(archive, name, rejected):
                    if "draft_version" not in row:
                        continue
                    origin = source_path(path, name, line)
                    key = (row["id"], digest(row))
                    inventory[key] = origin
                    if filename == "usable_auto.jsonl":
                        raw_count += 1
                        candidates[row["id"]].append((row, key, origin))

    selected = []
    for episode_id, records in sorted(candidates.items()):
        eligible = [
            record for record in records
            if votes[record[1]] & {"approve", "usable_auto"}
            and not votes[record[1]] & {"reject", "needs_fix", "needs_review", "fix"}
        ]
        if not eligible:
            rejected.append({"id": episode_id, "reason": "no_uncontested_hash_matched_review", **records[0][2]})
            continue
        # For the same episode ID, prefer the highest numbered review run.
        def priority(record):
            member = record[2]["member"]
            runs = re.findall(r"run(\d+)", member)
            return int(runs[-1]) if runs else 0, member

        row, key, origin = max(eligible, key=priority)
        signatures = {digest({k: v for k, v in vrow.items() if k not in {"lineage_group", "review_flags"}})
                      for vrow, _, _ in eligible}
        if len(signatures) > 1:
            rejected.append({"id": episode_id, "reason": "conflicting_approved_episode_versions", **origin})
            continue
        selected.append((row, origin, REVIEWED_EXPORT))

    selected_ids = {r["id"] for r, _, _ in selected}
    for (episode_id, _), origin in sorted(inventory.items()):
        if episode_id not in selected_ids and episode_id not in candidates:
            rejected.append({"id": episode_id, "reason": "no_usable_auto_export", **origin})
    return selected, rejected, {"usable_export_rows": raw_count, "distinct_usable_versions": len({
        key for records in candidates.values() for _, key, _ in records
    }), "distinct_usable_ids": len(candidates), "selected_approved_ids": len(selected)}


def read_drafts(path, rejected):
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".jsonl"):
                continue
            for line, row in zip_rows(archive, name, rejected):
                if "draft_version" not in row:
                    raise ValueError(f"Unexpected non-draft row in {name}:{line}")
                yield row, source_path(path, name, line), DRAFT_EXPORT


def precise_changes(before, after, path=""):
    """Check appended indices individually while retaining conservative deletion checks."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                changes.append(child)
            else:
                changes.extend(precise_changes(before[key], after[key], child))
        return changes
    if isinstance(before, list) and isinstance(after, list) and len(before) <= len(after):
        changes = [f"{path}/{i}" for i in range(len(before), len(after))]
        for i, old in enumerate(before):
            changes.extend(precise_changes(old, after[i], f"{path}/{i}"))
        return changes
    return [] if canonical(before) == canonical(after) else [path]


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def screen_calls(draft, replay, specs):
    """Narrow mechanical checks. These do not constitute a semantic review."""
    input_ = draft["input"]
    history = replay["history"]
    context = input_["context"]
    available = {t for app in context["installed_apps"].values() for t in app["tools"]}
    visible = list(strings([input_["user_text"], context]))
    for message in history:
        if message["role"] == "user":
            visible.append(message["text"])
        else:
            visible.extend(strings(message.get("tool_results", [])))
    for step in replay["trajectory"]:
        call = step["action"]
        name, arguments = call["name"], call["arguments"]
        if name == "sys.ask_missing_info":
            tool = arguments["tool_name"]
            if tool not in available or not set(arguments["params"]) <= specs[tool]["parameters"]["properties"].keys():
                return "invalid_clarification_target"
        for key, prop in specs[name]["parameters"]["properties"].items():
            value = arguments.get(key)
            if prop.get("x-ref") and isinstance(value, str) and value:
                pattern = REFERENT.pattern.format(re.escape(value))
                if not any(re.search(pattern, s) for s in visible):
                    return f"reference_absent_from_visible_prefix:{name}.{key}"
        visible.extend(strings(step["observation"]))

    # Independently verify a narrowly defined, unambiguous relative-time pattern.
    text = input_["user_text"].casefold()
    delay = re.findall(r"\b(?:(\d+)\s*(?:giờ|tiếng)\s*(?:(\d+)\s*phút)?|(\d+)\s*phút)\s+nữa\b", text)
    if len(delay) == 1 and len(replay["trajectory"]) == 1:
        action = replay["trajectory"][0]["action"]
        if action["name"] in {"alarm.create", "reminders.create"}:
            hours, minutes, minutes_only = delay[0]
            now = datetime.fromisoformat(context["current_time"])
            expected = now + timedelta(hours=int(hours or 0), minutes=int(minutes or minutes_only or 0))
            actual = datetime.fromisoformat(action["arguments"]["time"])
            if expected != actual:
                return "relative_time_mismatch"
    return None


def synthetic_episode(draft, origin, export, runtime, catalog, specs):
    replay = replay_draft(draft, runtime, catalog)
    errors = list(replay["errors"])
    scope_refined = False
    if errors == ["Changes outside draft allowed_mutation_paths"]:
        before = {k: v for k, v in replay["after_history_snapshot"].items() if k != "_meta"}
        allowed = draft["expected"]["allowed_mutation_paths"]
        actual = precise_changes(before, replay["final_state"])
        if all(any(p == a or p.startswith(a + "/") for a in allowed) for p in actual):
            errors.clear()
            scope_refined = True
    if errors:
        return None, "; ".join(errors)
    issue = screen_calls(draft, replay, specs)
    if issue:
        return None, issue
    row = {"audio": "current", "context": draft["input"]["context"], "history": []}
    transcripts = {"current": draft["input"]["user_text"]}
    for i, message in enumerate(replay["history"]):
        if message["role"] == "user":
            audio = f"history_{i}"
            transcripts[audio] = message["text"]
            row["history"].append({"role": "user", "audio": audio})
        else:
            row["history"].append(message)
    return {
        "id": draft["id"], "source": SYNTHETIC,
        "topic": "TASKS_NOTES" if draft["topic_id"] == "TASK_NOTES" else draft["topic_id"],
        "review_tier": "prior_codex_review" if export == REVIEWED_EXPORT else "replay_screened",
        "group": "synthetic:" + draft["lineage_group"],
        "request": transcripts["current"],
        "messages": make_messages(row, transcripts), "tools": tools_for(row, catalog),
        "steps": [(s["action"], s["observation"]) for s in replay["trajectory"]],
        "origin": {**origin, "draft_sha256": digest(draft)},
        "outcome": draft["expected"]["outcome"], "append_scope_refined": scope_refined,
        "warnings": replay["warnings"],
    }, None


def btc_episode(row, split, root, runtime, catalog):
    env = runtime.ToolEnv("train", row["id"], root=root)
    label = row["label"]
    expected = env.get_state()
    expected.update(label["final_state"])
    steps = []
    for call in label["tool_calls"]:
        observation = env.call(call["name"], call["arguments"])
        if observation.get("ok") is not True:
            raise ValueError(f"BTC gold call rejected for {row['id']}")
        steps.append((call, observation))
    if env.get_state_hash() != runtime.state_hash(expected):
        raise ValueError(f"BTC gold final-state mismatch for {row['id']}")
    transcripts = {turn["audio"]: turn["audio_text"] for turn in audio_turns(row)}
    calls = label["tool_calls"]
    return {
        "id": row["id"], "source": BTC, "topic": "BTC", "review_tier": "official_gold",
        "group": "btc:" + digest(normal_text(row["audio_text"])),
        "request": row["audio_text"], "messages": make_messages(row, transcripts),
        "tools": tools_for(row, catalog), "steps": steps,
        "origin": {"btc_id": row["id"], "split_ids": split},
        "outcome": "ask_missing_info" if calls and calls[-1]["name"] == "sys.ask_missing_info"
                   else "execute" if calls else "no_action",
        "gold_observation_exact": [obs for _, obs in steps] == label["tool_results"],
    }


def render_decisions(episode, tokenizer):
    """Render the actual inference prefix and supervise only the action plus EOS."""
    messages = list(episode["messages"])
    tools = episode["tools"]
    payloads = [messages, tools, episode["steps"]]
    if any(CONTROL_TOKENS.search(s) for s in strings(payloads)):
        raise ValueError("reserved_chat_control_token_in_source")
    steps = list(episode["steps"])
    if episode["outcome"] != "ask_missing_info":
        steps.append((None, None))
    decisions = []
    for i, (call, observation) in enumerate(steps, 1):
        target = {"role": "assistant", "content": "DONE"} if call is None else {
            "role": "assistant", "tool_calls": [{"type": "function", "function": call}]
        }
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        whole = tokenizer.apply_chat_template(
            messages + [target], tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        if not whole.startswith(prompt) or not whole.endswith(tokenizer.eos_token + "\n"):
            raise ValueError("qwen_chat_template_boundary_mismatch")
        completion = whole[len(prompt):-1]
        decoded = completion[:-len(tokenizer.eos_token)]
        if parse_action(decoded) != call:
            raise ValueError("completion_does_not_parse_to_gold_action")
        # Match the tokenization that TRL applies to standard prompt/completion rows.
        p_ids = tokenizer(prompt)["input_ids"]
        full_ids = tokenizer(prompt + completion)["input_ids"]
        target_length = len(full_ids) - len(p_ids)
        if full_ids[:len(p_ids)] != p_ids or target_length < 2 or full_ids[-1] != tokenizer.eos_token_id:
            raise ValueError("completion_mask_boundary_mismatch")
        if len(p_ids) + RESERVED_OUTPUT > CONTEXT_LIMIT or target_length >= RESERVED_OUTPUT:
            raise ValueError("outside_deployment_token_budget")
        decisions.append({
            "id": f"{episode['id']}:d{i}", "episode_id": episode["id"],
            "split_group": episode["group"], "source": episode["source"],
            "topic": episode["topic"], "review_tier": episode["review_tier"],
            "prompt": prompt, "completion": completion, "prompt_tokens": len(p_ids),
            "completion_tokens": target_length,
        })
        if call is not None:
            # Same representation as run_agent. Observations only enter later prefixes.
            messages.extend([
                {"role": "assistant", "content": decoded},
                {"role": "tool", "name": call["name"], "content": json.dumps(
                    observation, ensure_ascii=False, separators=(",", ":")
                )},
            ])
    return decisions


def split_synthetic(episodes, dev_texts):
    """Keep lineages and normalized repeated requests in the same partition."""
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for e in episodes:
        g = find(e["group"])
        t = find("text:" + normal_text(e["request"]))
        if t != g:
            parents[t] = g
    reserved = {find("text:" + text) for text in dev_texts}
    groups = defaultdict(list)
    excluded = []
    for e in episodes:
        group = find(e["group"])
        if group in reserved:
            excluded.append(e)
        else:
            e["group"] = group
            groups[group].append(e)
    strata = defaultdict(list)
    for group, members in groups.items():
        # Use a single stratum for mixed-lineage groups to preserve isolation.
        strata[tuple(sorted({m["topic"] for m in members}))].append(group)
    validation = set()
    rng = random.Random(SEED)
    for keys in strata.values():
        keys.sort()
        rng.shuffle(keys)
        if len(keys) > 1:
            validation.update(keys[:max(1, round(len(keys) * 0.10))])
    for group, members in groups.items():
        for e in members:
            e["partition"] = "synthetic_validation" if group in validation else "train"
    return [e for members in groups.values() for e in members], excluded


def build(args, output):
    from transformers import AutoTokenizer

    root = args.data_root
    runtime = load_runtime(root, "train")
    catalog = load_catalog(root, "train")
    specs = catalog_map(catalog)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tokenizer.eos_token != "<|im_end|>":
        raise ValueError("Expected the Qwen3-4B tokenizer and <|im_end|> EOS")
    rows = read_rows(root, "train")
    by_id = {r["id"]: r for r in rows}
    train_ids, dev_ids = [json.loads((args.split_dir / f"{p}_ids.json").read_text()) for p in ("train", "dev")]
    if set(train_ids) & set(dev_ids) or sorted(train_ids + dev_ids) != sorted(by_id):
        raise ValueError("Fixed BTC train/dev IDs must be disjoint and cover the train release")

    reviewed, rejected, archive_stats = read_reviewed(args.reviewed_zip)
    drafts = list(read_drafts(args.drafts_zip, rejected))
    seen_ids, fingerprints, synthetic = set(), set(), []
    for draft, origin, tier in reviewed + drafts:
        eid = draft["id"]
        if eid in seen_ids:
            rejected.append({"id": eid, "reason": "duplicate_episode_id", **origin})
            continue
        seen_ids.add(eid)
        episode, issue = synthetic_episode(draft, origin, tier, runtime, catalog, specs)
        if issue:
            rejected.append({"id": eid, "reason": issue, **origin})
            continue
        fingerprint = digest([episode["messages"], episode["tools"], episode["steps"]])
        if fingerprint in fingerprints:
            rejected.append({"id": eid, "reason": "duplicate_training_episode", **origin})
            continue
        fingerprints.add(fingerprint)
        synthetic.append(episode)
    synthetic, overlaps = split_synthetic(synthetic, {normal_text(by_id[i]["audio_text"]) for i in dev_ids})
    rejected.extend({"id": e["id"], "reason": "overlap_with_btc_dev_lineage_or_request", **e["origin"]} for e in overlaps)
    episodes = synthetic
    for partition, ids in (("train", train_ids), ("validation", dev_ids)):
        for eid in ids:
            episode = btc_episode(by_id[eid], partition, root, runtime, catalog)
            episode["partition"] = partition
            episodes.append(episode)

    parts = ("train", "validation", "synthetic_validation")
    outputs = {part: [] for part in parts}
    provenance = []
    for episode in episodes:
        try:
            decisions = render_decisions(episode, tokenizer)
        except ValueError as error:
            rejected.append({"id": episode["id"], "reason": str(error), **episode["origin"]})
            continue
        partition = episode["partition"]
        outputs[partition].extend(decisions)
        provenance.append({
            key: episode[key] for key in ("id", "source", "topic", "review_tier", "group", "partition", "origin", "outcome")
        } | {
            "decisions": len(decisions),
            "append_scope_refined": episode.get("append_scope_refined", False),
            "gold_observation_exact": episode.get("gold_observation_exact"),
        })
    for part in parts:
        data = outputs[part]
        if not data or len({r["id"] for r in data}) != len(data):
            raise ValueError(f"Empty partition or duplicated decision IDs: {part}")
        if part == "train":
            random.Random(SEED).shuffle(data)
        write_jsonl(output / f"{part}.jsonl", data)
    write_jsonl(output / "provenance.jsonl", provenance)
    write_jsonl(output / "excluded.jsonl", rejected)
    train = [e for e in provenance if e["partition"] == "train"]
    validation = [e for e in provenance if e["partition"] != "train"]
    if {e["id"] for e in train} & {e["id"] for e in validation} or {e["group"] for e in train} & {e["group"] for e in validation}:
        raise ValueError("Train/validation episode or lineage overlap")
    manifest = {
        "format": "lifemate-qwen3-text-prompt-completion-v1",
        "model": MODEL_ID, "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_files": {p.name: sha256_file(p) for p in sorted(args.tokenizer.glob("*")) if p.is_file()},
        "enable_thinking": False, "completion_only_loss": True, "eos_token": tokenizer.eos_token,
        "context_limit": CONTEXT_LIMIT, "reserved_output_tokens": RESERVED_OUTPUT, "seed": SEED,
        "builder_sha256": sha256_file(__file__),
        "runtime_sha256": digest([
            [str(p.relative_to(root / "train/zaic_tools")), sha256_file(p)]
            for p in sorted((root / "train/zaic_tools").rglob("*.py"))
        ]),
        "sources": {Path(p).name: sha256_file(p) for p in [
            args.reviewed_zip, args.drafts_zip, root / "train/train.jsonl", root / "train/train_tools_catalog.json",
            args.split_dir / "train_ids.json", args.split_dir / "dev_ids.json",
        ]},
        "archives": {REVIEWED_EXPORT: archive_stats, DRAFT_EXPORT: {"draft_rows": len(drafts)}},
        "partitions": {},
        "excluded_by_reason": dict(Counter(e["reason"] for e in rejected)),
        "excluded_unique_ids": len({e["id"] for e in rejected if e["id"] is not None} - {e["id"] for e in provenance}),
        "semantic_review": {
            "prior_codex_review": "Hash-matched pre-existing usable_auto decisions; replayed again locally.",
            "replay_screened": "Replay and mechanical grounding checks; not individually approved by a human reviewer.",
        },
    }
    for part in parts:
        members = [e for e in provenance if e["partition"] == part]
        data = outputs[part]
        manifest["partitions"][part] = {
            "episodes": len(members), "decisions": len(data),
            "episodes_by_review_tier": dict(Counter(e["review_tier"] for e in members)),
            "decisions_by_review_tier": dict(Counter(e["review_tier"] for e in data)),
            "episodes_by_topic": dict(Counter(e["topic"] for e in members)),
            "episodes_by_outcome": dict(Counter(e["outcome"] for e in members)),
            "append_scope_refined": sum(e["append_scope_refined"] for e in members),
            "max_prompt_tokens": max(r["prompt_tokens"] for r in data),
            "max_completion_tokens": max(r["completion_tokens"] for r in data),
            "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in data),
        }
    manifest["files"] = {p.name: sha256_file(p) for p in sorted(output.glob("*.jsonl"))}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-zip", type=Path, required=True)
    parser.add_argument("--drafts-zip", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local pinned Qwen3-4B tokenizer snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("reports/data_audit"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-sft-", dir=args.output.parent))
    try:
        manifest = build(args, staging)
        staging.rename(args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(manifest["partitions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
