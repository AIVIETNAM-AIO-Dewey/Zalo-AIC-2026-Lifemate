"""Regression tests for published LifeMate scoring, not model quality."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unicodedata
import unittest

from lifemate.data import load_runtime, read_rows
from lifemate.evaluate import arguments_match, matched_call_count, score_calls


SPECS = {"send": {"parameters": {"type": "object", "properties": {
    "id": {"type": "string"}, "content": {"type": "string", "x-match": "fuzzy"}}}}}


def call(text, **kwargs):
    return {"name": "send", "arguments": {"id": "c1", "content": text, **kwargs}}


class CallScoringTests(unittest.TestCase):
    def test_fuzzy_normalises_case_and_unicode_composition(self):
        self.assertEqual(matched_call_count([call("ĐÓNG TIỀN MẠNG")],
                         [call(unicodedata.normalize("NFD", "đóng tiền mạng"))], SPECS), 1)

    def test_threshold_is_strictly_greater_than_point_six(self):
        self.assertEqual(matched_call_count([call("a b c d e")], [call("a b c x y")], SPECS), 0)
        self.assertEqual(matched_call_count([call("a b c d")], [call("a b c x y")], SPECS), 1)

    def test_tokens_keep_multiplicity_punctuation_and_accents(self):
        for predicted, gold in [("a a a a b", "a b"), ("đóng.", "đóng"), ("đóng", "dong")]:
            with self.subTest(predicted=predicted):
                self.assertEqual(matched_call_count([call(predicted)], [call(gold)], SPECS), 0)

    def test_non_fuzzy_fields_and_key_presence_are_exact(self):
        for predicted in [call("a b", id="C1"), call("a b", repeat="once"),
                          {"name": "send", "arguments": {"content": "a b"}}]:
            self.assertEqual(matched_call_count([predicted], [call("a b")], SPECS), 0)
        self.assertFalse(arguments_match(True, 1, {}))
        self.assertFalse(arguments_match("A", "a", {}))
        self.assertFalse(arguments_match(["a", "b"], ["b", "a"], {}))

    def test_nested_fuzzy_is_driven_by_schema(self):
        schema = {"items": {"properties": {"note": {"x-match": "fuzzy"}, "id": {}}}}
        self.assertTrue(arguments_match([{"note": "A B C", "id": 1}],
                                        [{"note": "a b", "id": 1}], schema))
        self.assertFalse(arguments_match([{"note": "A B C", "id": 2}],
                                         [{"note": "a b", "id": 1}], schema))

    def test_maximum_matching_reassigns_ambiguous_fuzzy_candidate(self):
        predicted = [call("a b c"), call("a b")]
        gold = [call("a b"), call("b c")]
        for p in [predicted, predicted[::-1]]:
            for g in [gold, gold[::-1]]:
                self.assertEqual(matched_call_count(p, g, SPECS), 2)

    def test_duplicate_and_rejected_calls_reduce_precision(self):
        c = call("a b")
        result = score_calls([c, c], [c], SPECS, True)
        self.assertEqual(result["matched_calls"], 1)
        self.assertEqual(result["unmatched_calls"], 1)
        self.assertAlmostEqual(result["score"], 2/3)
        result = score_calls([c, c], [c, c], SPECS, True, [True, False])
        self.assertEqual(result["matched_calls"], 1)
        self.assertEqual(result["score"], 0.5)

    def test_unknown_tool_is_unmatched(self):
        c = {"name": "unknown", "arguments": {}}
        self.assertEqual(matched_call_count([c], [c], SPECS), 0)

    def test_missing_info_uses_the_same_matching_rules(self):
        specs = {"sys.ask_missing_info": {"parameters": {"properties": {}}}}
        gold = {"name": "sys.ask_missing_info", "arguments": {
            "tool_name": "phone.call", "params": ["contact_id"]}}
        self.assertEqual(score_calls([gold], [gold], specs, True)["score"], 1)
        wrong = copy.deepcopy(gold)
        wrong["arguments"]["params"] = ["phone_number"]
        self.assertEqual(score_calls([wrong], [gold], specs, True)["score"], 0)

    def test_state_gate_and_empty_gold_special_case(self):
        c = call("a b")
        self.assertEqual(score_calls([c], [c], SPECS, False)["score"], 0)
        self.assertEqual(score_calls([], [c], SPECS, True)["score"], 0)
        self.assertEqual(score_calls([], [], SPECS, False)["score"], 1)
        self.assertEqual(score_calls([c], [], SPECS, True)["score"], 0)


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("LIFEMATE_TEST_DATA_ROOT", ROOT / "data"))


@unittest.skipUnless((DATA / "train/train.jsonl").is_file(), "BTC train data not installed")
class ReplayScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_rows(DATA, "train")
        cls.runtime = load_runtime(DATA, "train")

    def evaluate(self, row, calls, stop_reason=None, bad_hash=False, missing=False):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            env = self.runtime.ToolEnv("train", row["id"], root=DATA)
            for c in calls:
                env.call(c["name"], c["arguments"])
            prediction = {"id": row["id"], "tool_calls": env.get_tool_calls(),
                          "state_hash": "wrong" if bad_hash else env.get_state_hash()}
            (run / "run_config.json").write_text(json.dumps({"split": "train", "mode": "oracle", "sample_ids": [row["id"]]}))
            (run / "submission.jsonl").write_text("" if missing else json.dumps(prediction) + "\n")
            if stop_reason:
                (run / "diagnostics.jsonl").write_text(json.dumps({"id": row["id"], "stop_reason": stop_reason, "seconds": 60}) + "\n")
            output = run / "rescored"
            process = subprocess.run([sys.executable, "-m", "lifemate.evaluate", "--data-root", str(DATA),
                                      "--run", str(run), "--output", str(output)],
                                     env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse((run / "evaluation.json").exists())
            return json.loads((output / "evaluation.json").read_text()), json.loads((output / "per_sample_evaluation.json").read_text())[0]

    def test_gold_calls_score_full_without_diagnostics(self):
        row = next(r for r in self.rows if r["label"]["tool_calls"])
        report, detail = self.evaluate(row, row["label"]["tool_calls"])
        self.assertEqual(report["score_percent"], 100)
        self.assertEqual(detail["score"], 1)

    def test_agent_stop_reasons_do_not_change_submission_score(self):
        row = next(r for r in self.rows if r["label"]["tool_calls"])
        for reason in ("repair_limit", "call_limit", "error", "timeout"):
            with self.subTest(reason=reason):
                report, _ = self.evaluate(row, row["label"]["tool_calls"], reason)
                self.assertEqual(report["score_percent"], 100)
                self.assertEqual(report["latency_over_30_seconds"], 1)

    def test_empty_gold_keeps_credit_despite_repair_limit(self):
        row = next(r for r in self.rows if not r["label"]["tool_calls"])
        report, _ = self.evaluate(row, [], "repair_limit")
        self.assertEqual(report["score_percent"], 100)

    def test_replay_is_authoritative_and_hash_mismatch_is_reported(self):
        row = next(r for r in self.rows if r["label"]["tool_calls"])
        report, detail = self.evaluate(row, row["label"]["tool_calls"], bad_hash=True)
        self.assertEqual(report["score_percent"], 100)
        self.assertTrue(detail["state_match"])
        self.assertFalse(detail["hash_reproducible"])
        self.assertEqual(report["hash_mismatches"], 1)

    def test_rejected_call_is_replayed_and_penalised_as_unmatched(self):
        row = next(r for r in self.rows if r["label"]["tool_calls"])
        gold = row["label"]["tool_calls"]
        calls = copy.deepcopy(gold) + [{"name": "not_installed.no_such_tool", "arguments": {}}]
        report, detail = self.evaluate(row, calls)
        self.assertEqual(len(detail["schema_errors"]), 1)
        self.assertEqual(detail["unmatched_calls"], 1)
        self.assertAlmostEqual(report["score_percent"], 100 * 2 * len(gold) / (2 * len(gold) + 1))

    def test_missing_prediction_is_zero_even_for_empty_gold(self):
        row = next(r for r in self.rows if not r["label"]["tool_calls"])
        report, detail = self.evaluate(row, [], missing=True)
        self.assertEqual(report["score_percent"], 0)
        self.assertTrue(detail["missing_prediction"])


if __name__ == "__main__":
    unittest.main()
