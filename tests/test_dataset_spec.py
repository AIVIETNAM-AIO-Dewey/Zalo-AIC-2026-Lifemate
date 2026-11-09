"""Validate the dataset contract and self-created fixtures, not model quality.

jsonschema is an optional specification dependency; runtime fixtures require the
unmodified train package in data/. No BTC transcript or environment is copied.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

try:
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError
except ImportError:
    Draft202012Validator = None

from lifemate.agent import SYSTEM
from lifemate.data import load_runtime

ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value):
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(value, pointer):
    for token in pointer.split("/")[1:] if pointer else []:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def changed_paths(before, after, path=""):
    if type(before) is not type(after):
        return [path]
    if isinstance(before, dict) and before.keys() == after.keys():
        return [p for key in before for p in changed_paths(before[key], after[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))]
    if isinstance(before, list) and len(before) == len(after):
        return [p for i in range(len(before)) for p in changed_paths(before[i], after[i], f"{path}/{i}")]
    return [] if before == after else [path]


@unittest.skipIf(Draft202012Validator is None, "Install requirements-schema.txt to check dataset schemas")
class DatasetSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validators = {}
        for kind in ("master", "sft"):
            schema = json.loads((ROOT / f"schemas/{kind}.schema.json").read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            cls.validators[kind] = Draft202012Validator(schema, format_checker=FormatChecker())
        cls.masters = read_jsonl(ROOT / "schemas/examples/master.jsonl")
        cls.sft = read_jsonl(ROOT / "schemas/examples/sft.jsonl")

    def test_all_examples_match_draft_2020_12_schemas(self):
        for kind, rows in (("master", self.masters), ("sft", self.sft)):
            for row in rows:
                with self.subTest(kind=kind, id=row["id"]):
                    self.validators[kind].validate(row)

    def test_hidden_environment_is_rejected_in_input_context(self):
        row = copy.deepcopy(self.masters[0])
        row["input"]["context"]["installed_apps"]["smart_home"]["state"] = {"SECRET": True}
        with self.assertRaises(ValidationError):
            self.validators["master"].validate(row)

    def test_outcome_and_terminal_cannot_contradict_trajectory(self):
        row = copy.deepcopy(self.masters[0])
        row["target"]["outcome"] = "no_action"
        row["terminal"]["reason"] = "user_cancelled"
        with self.assertRaises(ValidationError):
            self.validators["master"].validate(row)
        row = copy.deepcopy(next(m for m in self.masters if m["target"]["outcome"] == "ask_missing_info"))
        row["terminal"]["signal"] = "DONE"
        with self.assertRaises(ValidationError):
            self.validators["master"].validate(row)

    def test_completion_requires_one_call_or_explicit_done(self):
        for completion in ([], [{"role": "assistant", "content": "[]"}], self.sft[0]["completion"] * 2):
            row = copy.deepcopy(self.sft[0])
            row["completion"] = completion
            with self.subTest(completion=completion), self.assertRaises(ValidationError):
                self.validators["sft"].validate(row)

    def test_fixtures_are_examples_with_pending_semantic_review(self):
        for row in self.masters:
            self.assertEqual(row["metadata"]["source"], "manual_fixture")
            self.assertEqual(row["metadata"]["partition"], "example")
            self.assertEqual(row["metadata"]["review"]["status"], "needs_review")
            self.assertEqual(row["verification"]["semantic_valid"], "not_run")
            self.assertEqual(row["verification"]["call_policy_valid"], "not_run")

    def test_sft_prefixes_have_only_past_observations_and_match_baseline_payload(self):
        for master in self.masters:
            input_ = master["input"]
            payload = {"context": input_["context"], "history": input_["conversation_history"], "latest_request": input_["user_text"]}
            prefix = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
            rows = [s for s in self.sft if s["episode_id"] == master["id"]]
            expected_size = len(master["trajectory"]) + (master["terminal"]["signal"] == "DONE")
            self.assertEqual(len(rows), expected_size)
            for index, step in enumerate(master["trajectory"]):
                row = rows[index]
                expected = {"role": "assistant", "tool_calls": [{"type": "function", "function": step["action"]}]}
                self.assertEqual(row["prompt"], prefix)
                self.assertEqual(row["tools"], master["tools"])
                self.assertEqual(row["completion"], [expected])
                self.assertEqual(row["decision_index"], index + 1)
                prefix = prefix + [expected, {"role": "tool", "name": step["action"]["name"], "content": json.dumps(step["observation"], ensure_ascii=False, separators=(",", ":"))}]
            if master["terminal"]["signal"] == "DONE":
                self.assertEqual(rows[-1]["prompt"], prefix)
                self.assertEqual(rows[-1]["completion"], [{"role": "assistant", "content": "DONE"}])

    def test_fixture_argument_references_are_visible_and_precede_actions(self):
        for master in self.masters:
            available = {t for app in master["input"]["context"]["installed_apps"].values() for t in app["tools"]} | {"sys.ask_missing_info"}
            self.assertEqual(available, {t["function"]["name"] for t in master["tools"]})
            for index, step in enumerate(master["trajectory"], 1):
                self.assertEqual(step["step"], index)
                self.assertTrue(all(0 < dep < index for dep in step["depends_on_steps"]))
                self.assertEqual(set(step["action"]["arguments"]), {s["argument"] for s in step["argument_sources"]})
                for source in step["argument_sources"]:
                    self.assertTrue(source["ref"].startswith(("/input/", "/tools/", "/trajectory/")))
                    resolved = resolve(master, source["ref"])
                    if source["source"] == "observation":
                        self.assertLess(source["source_step"], index)
                        self.assertTrue(source["ref"].startswith(f"/trajectory/{source['source_step'] - 1}/observation/"))
                    if source["source"] in ("history", "observation"):
                        self.assertEqual(resolved, step["action"]["arguments"][source["argument"]])

    def test_pilot_quotas_and_tool_names_match_train_catalog(self):
        config = json.loads((ROOT / "configs/synthetic_pilot.json").read_text())
        self.assertEqual(sum(f["accepted_episode_budget"] for f in config["families"]), config["accepted_episode_budget"])
        catalog = json.loads((ROOT / config["source_catalog"]).read_text())
        names = {t["name"] for a in catalog["apps"] for t in a["tools"]}
        for family in config["families"]:
            self.assertTrue(set(family["focus_tools"] + family["support_tools"]) <= names)
        self.assertAlmostEqual(config["scratch_train_fraction"] + config["scratch_validation_fraction"], 1)

    @unittest.skipUnless((ROOT / "data/train/zaic_tools/__init__.py").is_file(), "BTC train runtime required")
    def test_fixture_replay_hashes_observations_goals_and_mutation_scope(self):
        runtime = load_runtime(ROOT / "data", "train")
        for master in self.masters:
            with self.subTest(id=master["id"]):
                environment = master["environment"]
                snapshot = environment["initial_state"]
                self.assertEqual(canonical_hash(snapshot), environment["initial_state_sha256"])
                catalog_path = ROOT / environment["catalog_ref"]
                self.assertEqual(hashlib.sha256(catalog_path.read_bytes()).hexdigest(), environment["catalog_sha256"])
                package = ROOT / environment["runtime_ref"]
                manifest = [[str(p.relative_to(package)), hashlib.sha256(p.read_bytes()).hexdigest()] for p in sorted(package.rglob("*.py"))]
                self.assertEqual(canonical_hash(manifest), environment["runtime_sha256"])
                catalog = json.loads(catalog_path.read_text())
                # Fresh Session starts at the snapshot AFTER history. No historical calls are executed here.
                session = runtime.Session(master["id"], copy.deepcopy(snapshot), catalog)
                before = session.get_state()
                for step in master["trajectory"]:
                    call = step["action"]
                    self.assertTrue(session.validate(call["name"], call["arguments"])["ok"])
                    self.assertEqual(session.call(call["name"], call["arguments"]), step["observation"])
                    self.assertEqual(session.get_state_hash(), step["state_hash_after"])
                self.assertEqual(session.get_tool_calls(), [s["action"] for s in master["trajectory"]])
                after = session.get_state()
                for assertion in master["target"]["state_assertions"]:
                    value = resolve(after, assertion["path"])
                    if assertion["operator"] == "eq":
                        self.assertEqual(value, assertion["value"])
                    elif assertion["operator"] == "unchanged":
                        self.assertEqual(value, resolve(before, assertion["path"]))
                for path in changed_paths(before, after):
                    self.assertTrue(any(path == allowed or path.startswith(allowed + "/") for allowed in master["target"]["allowed_mutation_paths"]), path)


if __name__ == "__main__":
    unittest.main()
