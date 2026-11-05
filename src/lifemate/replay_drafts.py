"""CPU-only replay of cloud-generated synthetic drafts using the BTC runtime.

No BTC sample state is loaded or modified. History is built in a preparation
Session; latest-request calls run in a NEW Session starting after that history.
Passing automatic checks NEVER approves an episode for SFT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from .data import catalog_map, load_catalog, load_runtime


POINTER = re.compile(r"^(?:/(?:[^~/]|~[01])*)*$")
REASONS = {
    "execute": {"completed", "lookup_complete", "condition_false"},
    "ask_missing_info": {"awaiting_info"},
    "no_action": {"condition_false", "unsupported_capability", "user_cancelled", "answer_in_history", "no_new_request"},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pointer_get(value, pointer):
    if not isinstance(pointer, str) or not POINTER.fullmatch(pointer):
        raise ValueError("Invalid JSON Pointer")
    for token in pointer.split("/")[1:] if pointer else []:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("Array pointer must use a nonnegative integer index")
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise ValueError("Pointer traverses a scalar")
    return value


def changed_paths(before, after, path=""):
    """Raw JSON differences, including fields excluded by BTC state_hash."""
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(before.keys() | after.keys()):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                paths.append(child)
            else:
                paths.extend(changed_paths(before[key], after[key], child))
        return paths
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [path]  # Append/remove permissions must cover the collection.
        return [p for i in range(len(before)) for p in changed_paths(before[i], after[i], f"{path}/{i}")]
    return [] if canonical(before) == canonical(after) else [path]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_draft(draft, specs):
    """Draft envelope checks, distinct from BTC's dynamic call validation."""
    require(isinstance(draft, dict), "Each line must be a JSON object")
    required = {"draft_version", "id", "topic_id", "variant", "lineage_group", "input", "environment_draft", "history_plan", "proposed_tool_calls", "expected", "verification_status", "review_flags"}
    require(set(draft) == required, f"Draft fields differ: missing={sorted(required - set(draft))}, extra={sorted(set(draft) - required)}")
    require(draft["draft_version"] == "1.0.0", "Unsupported draft_version")
    for key in ("id", "topic_id", "variant", "lineage_group"):
        require(nonempty(draft[key]), f"{key} must be a nonempty string")
    require(draft["verification_status"] == "not_run", "Draft verification_status must remain not_run")
    require(isinstance(draft["review_flags"], list) and all(isinstance(x, str) for x in draft["review_flags"]), "review_flags must be a string array")
    input_ = draft["input"]
    require(isinstance(input_, dict) and set(input_) == {"user_text", "context"}, "input must contain user_text and context only")
    require(nonempty(input_["user_text"]), "user_text must be nonempty")
    context = input_["context"]
    require(isinstance(context, dict) and set(context) == {"current_time", "timezone", "location", "user_profile", "installed_apps"}, "Unexpected context fields; do not expose hidden state in context")
    require(nonempty(context["timezone"]) and isinstance(context["user_profile"], dict), "Invalid timezone/user_profile")
    now = datetime.fromisoformat(context["current_time"].replace("Z", "+00:00"))
    require(now.utcoffset() is not None, "current_time must include a timezone offset")
    installed = context["installed_apps"]
    require(isinstance(installed, dict), "installed_apps must be an object")
    available = {"sys.ask_missing_info"}
    for app, entry in installed.items():
        require(isinstance(entry, dict) and set(entry) == {"tools"}, f"{app}: installed_apps may only expose tools")
        names = entry["tools"]
        require(isinstance(names, list) and all(isinstance(n, str) for n in names), f"{app}: tools must be strings")
        require(len(names) == len(set(names)), f"{app}: duplicate tool names")
        require(all(n.startswith(app + ".") and n in specs for n in names), f"{app}: tool absent from the train catalog or wrong app")
        available.update(names)
    env = draft["environment_draft"]
    require(isinstance(env, dict) and set(env) == {"state_boundary", "initial_state"}, "Invalid environment_draft fields")
    require(env["state_boundary"] == "before_history", "Draft snapshot must be before_history")
    snapshot = env["initial_state"]
    require(isinstance(snapshot, dict) and isinstance(snapshot.get("_meta"), dict), "Snapshot needs an object _meta")
    require(set(snapshot) - {"_meta", "sys"} == set(installed) - {"sys"}, "Snapshot app keys must match installed_apps")
    require(all(isinstance(state, dict) for state in snapshot.values()), "Each app state must be an object")
    meta = snapshot["_meta"]
    require(meta.get("current_time") == context["current_time"], "Snapshot/context current_time mismatch")
    names = meta.get("tools")
    require(isinstance(names, list) and all(isinstance(n, str) for n in names), "_meta.tools must be a string array")
    require(len(names) == len(set(names)) and set(names) == available, "_meta.tools must match installed_apps plus sys.ask_missing_info")

    history = draft["history_plan"]
    require(isinstance(history, list), "history_plan must be an array")
    plans = []
    for turn in history:
        require(isinstance(turn, dict) and set(turn) == {"user_text", "proposed_tool_calls"}, "History plan must not contain invented observations/tool_results")
        require(nonempty(turn["user_text"]), "History user_text must be nonempty")
        plans.append(turn["proposed_tool_calls"])
    plans.append(draft["proposed_tool_calls"])
    require(len(draft["proposed_tool_calls"]) <= 6, "Pilot permits at most six latest-request calls")
    for plan in plans:
        require(isinstance(plan, list), "proposed_tool_calls must be an array")
        for i, call in enumerate(plan):
            require(isinstance(call, dict) and set(call) == {"name", "arguments"}, "Each call requires name and arguments only")
            require(nonempty(call["name"]) and isinstance(call["arguments"], dict), "Invalid call name/arguments")
            require(call["name"] in available, f"Unavailable tool: {call['name']}")
            props = specs[call["name"]]["parameters"].get("properties", {})
            # Extra dataset-quality constraint: BTC may ignore unknown arguments.
            require(not (call["arguments"].keys() - props.keys()), f"Unknown arguments for {call['name']}: {sorted(call['arguments'].keys() - props.keys())}")
            require(call["name"] != "sys.ask_missing_info" or i == len(plan) - 1, "Calls cannot follow sys.ask_missing_info")
    expected = draft["expected"]
    require(isinstance(expected, dict) and set(expected) == {"outcome", "terminal_reason", "subgoals", "state_assertions", "allowed_mutation_paths"}, "Invalid expected fields")
    calls = draft["proposed_tool_calls"]
    actual = "no_action" if not calls else "ask_missing_info" if calls[-1]["name"] == "sys.ask_missing_info" else "execute"
    require(expected["outcome"] == actual, "Outcome contradicts latest-request call list")
    require(expected["terminal_reason"] in REASONS[actual], "terminal_reason contradicts outcome")
    paths = expected["allowed_mutation_paths"]
    require(isinstance(paths, list) and all(isinstance(p, str) and POINTER.fullmatch(p) and p for p in paths), "allowed_mutation_paths must be non-root JSON Pointers")
    require(isinstance(expected["state_assertions"], list), "state_assertions must be an array")
    for assertion in expected["state_assertions"]:
        require(isinstance(assertion, dict), "Invalid assertion")
        require({"path", "operator"} <= assertion.keys() <= {"path", "operator", "value"}, "Invalid assertion fields")
        require(isinstance(assertion["path"], str) and POINTER.fullmatch(assertion["path"]), "Invalid assertion pointer")
        require(assertion["operator"] in {"eq", "unchanged", "exists"}, "Unknown assertion operator")
        require(assertion["operator"] != "eq" or "value" in assertion, "eq assertion needs value")
    goals = expected["subgoals"]
    require(isinstance(goals, list) and bool(goals), "At least one subgoal is required")
    ids = []
    for goal in goals:
        require(isinstance(goal, dict) and set(goal) == {"id", "instruction", "depends_on", "expected_status", "tools", "missing_arguments"}, "Invalid subgoal fields")
        require(nonempty(goal["id"]) and nonempty(goal["instruction"]), "Invalid subgoal ID/instruction")
        require(goal["expected_status"] in {"completed", "awaiting_info", "skipped_condition_false", "unsupported", "cancelled"}, "Invalid expected subgoal status")
        for field in ("depends_on", "tools", "missing_arguments"):
            require(isinstance(goal[field], list) and all(nonempty(s) for s in goal[field]), f"{field} must be a string array")
        ids.append(goal["id"])
    require(len(ids) == len(set(ids)), "Duplicate subgoal IDs")
    edges = {g["id"]: g["depends_on"] for g in goals}
    visited, active = set(), set()

    def visit(node):
        require(node in edges, f"Unknown subgoal dependency: {node}")
        require(node not in active, "Subgoal dependency cycle")
        if node in visited:
            return
        active.add(node)
        for parent in edges[node]:
            visit(parent)
        active.remove(node)
        visited.add(node)

    for node in edges:
        visit(node)
    return snapshot


def replay_draft(draft, runtime, catalog):
    report = {"id": draft.get("id") if isinstance(draft, dict) else None,
              "automatic_status": "fail", "errors": [], "warnings": [],
              "checks": {}, "accepted_for_sft": False,
              "causal_valid": "not_run", "semantic_valid": "not_run", "call_policy_valid": "not_run",
              "review_status": "blocked", "history": [], "trajectory": []}
    try:
        report["draft_sha256"] = digest(draft)
        specs = catalog_map(catalog)
        snapshot = validate_draft(draft, specs)
        report["checks"]["draft_structure"] = "pass"
        report["user_text"] = draft["input"]["user_text"]
        report["expected"] = copy.deepcopy(draft["expected"])
        report["context"] = copy.deepcopy(draft["input"]["context"])
        report["warnings"].extend(draft["review_flags"])
        if any(not value for key, value in snapshot.items() if key not in {"_meta", "sys"}):
            report["warnings"].append("empty_app_state_needs_review")
        # Build completed history in a preparation session. No source file writes.
        history_env = runtime.Session(draft["id"] + ":history", copy.deepcopy(snapshot), catalog)

        def execute(env, call):
            check = env.validate(call["name"], call["arguments"])
            require(check["ok"], f"{call['name']}: {check['error']}")
            result = env.call(call["name"], call["arguments"])
            require(result.get("ok") is True, f"Runtime rejected {call['name']}: {result}")
            canonical(result)  # No non-JSON/nonfinite observations in reports.
            if result.get("done") is False:
                report["warnings"].append(f"done_false:{call['name']}:{result.get('message', '')}")
            if any(isinstance(value, list) and not value for value in result.values()):
                report["warnings"].append(f"empty_result_list:{call['name']}")
            return result

        for turn in draft["history_plan"]:
            results = [execute(history_env, call) for call in turn["proposed_tool_calls"]]
            report["history"].extend([
                {"role": "user", "text": turn["user_text"]},
                {"role": "assistant", "text": "", "tool_calls": copy.deepcopy(turn["proposed_tool_calls"]), "tool_results": results},
            ])
        after_history = {"_meta": copy.deepcopy(snapshot["_meta"]), **history_env.get_state()}
        report["after_history_snapshot"] = after_history
        # Latest request gets its own session, so historical calls are not logged
        # as current calls, and historical writes are not counted as new effects.
        env = runtime.Session(draft["id"], copy.deepcopy(after_history), catalog)
        before = env.get_state()
        report["initial_state_sha256"] = digest(after_history)
        for step, call in enumerate(draft["proposed_tool_calls"], 1):
            observation = execute(env, call)
            report["trajectory"].append({"step": step, "action": copy.deepcopy(call),
                                         "observation": observation, "state_hash_after": env.get_state_hash()})
        require(env.get_tool_calls() == draft["proposed_tool_calls"], "Latest-request log contains unexpected calls")
        report["checks"]["runtime_replay"] = "pass"
        after = env.get_state()
        report["final_state"] = after
        report["final_state_hash"] = env.get_state_hash()
        report["assertion_results"] = []
        for assertion in draft["expected"]["state_assertions"]:
            item = copy.deepcopy(assertion)
            try:
                actual = pointer_get(after, assertion["path"])
                item["actual"] = actual
                expected = pointer_get(before, assertion["path"]) if assertion["operator"] == "unchanged" else assertion.get("value")
                item["pass"] = assertion["operator"] == "exists" or canonical(actual) == canonical(expected)
            except (KeyError, IndexError, ValueError, TypeError) as error:
                item.update({"pass": False, "error": str(error)})
            report["assertion_results"].append(item)
        failures = [a for a in report["assertion_results"] if not a["pass"]]
        report["checks"]["draft_state_assertions"] = "fail" if failures else "pass"
        if failures:
            report["errors"].append("Draft state assertions failed")
        if not report["assertion_results"]:
            report["warnings"].append("no_state_assertions_provided")
        changes = changed_paths(before, after)
        allowed = draft["expected"]["allowed_mutation_paths"]
        outside = [path for path in changes if not any(path == p or path.startswith(p + "/") for p in allowed)]
        report["actual_mutated_paths"] = changes
        report["unexpected_mutated_paths"] = outside
        report["checks"]["draft_mutation_scope"] = "fail" if outside else "pass"
        if outside:
            report["errors"].append("Changes outside draft allowed_mutation_paths")
        if not report["errors"]:
            report["automatic_status"] = "pass"
            report["review_status"] = "needs_review"
        report["warnings"].append("expected_assertions_and_allowed_paths_are_untrusted_draft_labels")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as error:
        report["errors"].append(str(error))
    return report


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def run_file(input_path, output, data_root):
    input_path, output, data_root = Path(input_path), Path(output), Path(data_root)
    require(not output.exists(), "Output directory already exists; choose a new directory")
    payload = input_path.read_bytes()
    entries = []
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            draft = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            canonical(draft)  # reject NaN/Infinity, including overflowed floats
            entries.append((line_number, draft, None))
        except (ValueError, TypeError) as error:
            entries.append((line_number, None, f"Invalid JSON: {error}"))
    require(bool(entries), "No nonempty JSONL lines")
    counts = Counter(d["id"] for _, d, _ in entries if isinstance(d, dict) and isinstance(d.get("id"), str))
    runtime = load_runtime(data_root, "train")
    catalog = load_catalog(data_root, "train")
    reports = []
    for line_number, draft, error in entries:
        duplicate = isinstance(draft, dict) and isinstance(draft.get("id"), str) and counts[draft["id"]] > 1
        if error or duplicate:
            report = {"id": draft.get("id") if isinstance(draft, dict) else None,
                      "automatic_status": "fail", "errors": [error or "Duplicate episode ID in input file"],
                      "warnings": [], "review_status": "blocked", "accepted_for_sft": False}
        else:
            report = replay_draft(draft, runtime, catalog)
        report["line_number"] = line_number
        reports.append(report)
    package = (data_root / "train/zaic_tools").resolve()
    manifest = [[str(p.relative_to(package)), hashlib.sha256(p.read_bytes()).hexdigest()] for p in sorted(package.rglob("*.py"))]
    summary = {"input": str(input_path.resolve()), "input_sha256": hashlib.sha256(payload).hexdigest(),
               "validator_version": "draft_replay_v1", "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "catalog_sha256": hashlib.sha256((data_root / "train/train_tools_catalog.json").read_bytes()).hexdigest(),
               "runtime_sha256": digest(manifest), "records": len(reports),
               "automatic_pass": sum(r["automatic_status"] == "pass" for r in reports),
               "automatic_fail": sum(r["automatic_status"] == "fail" for r in reports),
               "accepted_for_sft": 0,
               "notice": "Automatic checks compare against untrusted draft expectations. Semantic, causal and call-policy approval are pending; this is not an SFT exporter or official scorer."}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in reports), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decisions = [{"id": r["id"], "line_number": r["line_number"], "draft_sha256": r.get("draft_sha256"),
                  "decision": "pending" if r["automatic_status"] == "pass" else "fix_or_reject",
                  "reviewer": None, "intent_correct": None, "visible_evidence_sufficient": None,
                  "call_policy_correct": None, "time_entities_and_subgoals_correct": None,
                  "terminal_correct": None, "notes": ""} for r in reports]
    (output / "review_decisions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in decisions), encoding="utf-8")
    cards = ["# Review synthetic drafts", "", summary["notice"], "",
             "Reviewer may inspect states here; hidden state must never be included in agent prompts.", "",
             "For each automatic-pass record: check user intent, sufficient visible evidence for IDs/conditions, necessary read/write calls, dates/amounts/negation, subgoal dependencies and stopping. Record a decision and evidence in review_decisions.jsonl.", ""]
    for report in reports:
        cards.extend([f"## Line {report['line_number']} — {report.get('id')}", "",
                      f"Automatic: {report['automatic_status']}; human review: {report['review_status']}; accepted_for_sft: false.", ""])
        evidence = {k: report[k] for k in ("user_text", "context", "history", "trajectory", "expected", "assertion_results", "actual_mutated_paths", "unexpected_mutated_paths", "errors", "warnings") if k in report}
        cards.extend(["```json", json.dumps(evidence, ensure_ascii=False, indent=2), "```", ""])
    (output / "review.md").write_text("\n".join(cards), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory, normally under ignored outputs/")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    summary = run_file(args.input, args.output, args.data_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if summary["automatic_fail"] else 0)


if __name__ == "__main__":
    main()
