"""Synthetic-only replay regression checks. No cloud, GPU or model outputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from lifemate.data import load_catalog, load_runtime
from lifemate.replay_drafts import replay_draft, run_file


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@unittest.skipUnless((DATA / "train/zaic_tools/__init__.py").is_file(), "BTC train runtime required")
class DraftReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_runtime(DATA, "train")
        cls.catalog = load_catalog(DATA, "train")
        cls.examples = [json.loads(line) for line in (ROOT / "schemas/examples/cloud_drafts_review_demo.jsonl").read_text().splitlines()]

    def draft(self, index=0):
        return copy.deepcopy(self.examples[index])

    def replay(self, draft):
        return replay_draft(draft, self.runtime, self.catalog)

    def test_good_fixture_runs_and_input_is_not_mutated(self):
        draft = self.draft()
        before = copy.deepcopy(draft)
        result = self.replay(draft)
        self.assertEqual(result["automatic_status"], "pass", result["errors"])
        self.assertEqual(result["final_state"]["smart_home"]["devices"][0]["state"]["power"], "on")
        self.assertEqual(draft, before)
        self.assertFalse(result["accepted_for_sft"])

    def test_schema_valid_wrong_id_is_detected_by_state_assertion(self):
        result = self.replay(self.draft(1))
        self.assertEqual(result["checks"]["runtime_replay"], "pass")
        self.assertEqual(result["checks"]["draft_state_assertions"], "fail")
        self.assertEqual(result["automatic_status"], "fail")
        self.assertTrue(result["trajectory"][1]["observation"]["ok"])
        self.assertFalse(result["trajectory"][1]["observation"]["done"])

    def test_history_writes_are_not_replayed_or_counted_as_new_mutations(self):
        result = self.replay(self.draft(2))
        self.assertEqual(result["automatic_status"], "pass", result["errors"])
        self.assertEqual(len(result["history"]), 4)
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(len(result["after_history_snapshot"]["alarm"]["alarms"]), 1)
        self.assertEqual(len(result["final_state"]["alarm"]["alarms"]), 1)
        self.assertEqual(result["trajectory"][0]["action"]["name"], "smart_home.light_set")
        self.assertEqual(result["unexpected_mutated_paths"], [])

    def test_unrequested_extra_action_fails_mutation_scope(self):
        result = self.replay(self.draft(3))
        self.assertEqual(result["checks"]["draft_state_assertions"], "pass")
        self.assertEqual(result["checks"]["draft_mutation_scope"], "fail")
        self.assertIn("/alarm/alarms", result["unexpected_mutated_paths"])

    def test_wrong_semantics_can_pass_automatic_checks_but_never_approve(self):
        result = self.replay(self.draft(4))
        self.assertEqual(result["automatic_status"], "pass")
        self.assertEqual(result["review_status"], "needs_review")
        self.assertEqual(result["semantic_valid"], "not_run")
        self.assertEqual(result["causal_valid"], "not_run")
        self.assertEqual(result["call_policy_valid"], "not_run")
        self.assertFalse(result["accepted_for_sft"])

    def test_free_text_mutation_is_checked_even_if_btc_hash_ignores_it(self):
        draft = self.draft(2)
        draft["input"]["context"]["installed_apps"]["alarm"]["tools"].append("alarm.update")
        draft["environment_draft"]["initial_state"]["_meta"]["tools"].append("alarm.update")
        draft["proposed_tool_calls"] = [{"name": "alarm.update", "arguments": {"alarm_id": "alm_01", "label": "not requested"}}]
        draft["expected"]["state_assertions"] = [{"path": "", "operator": "unchanged"}]
        draft["expected"]["allowed_mutation_paths"] = []
        result = self.replay(draft)
        before = {k: v for k, v in result["after_history_snapshot"].items() if k != "_meta"}
        self.assertEqual(self.runtime.state_hash(before), result["final_state_hash"])
        self.assertEqual(result["automatic_status"], "fail")
        self.assertIn("/alarm/alarms/0/label", result["unexpected_mutated_paths"])

    def test_hidden_state_in_installed_app_context_is_rejected(self):
        draft = self.draft()
        draft["input"]["context"]["installed_apps"]["smart_home"]["state"] = {}
        self.assertEqual(self.replay(draft)["automatic_status"], "fail")

    def test_snapshot_and_context_tool_whitelists_must_match(self):
        draft = self.draft()
        draft["environment_draft"]["initial_state"]["_meta"]["tools"].append("alarm.create")
        self.assertEqual(self.replay(draft)["automatic_status"], "fail")

    def test_unavailable_tool_and_unknown_arguments_are_rejected(self):
        for change in ("unavailable", "unknown_argument"):
            draft = self.draft()
            if change == "unavailable":
                draft["proposed_tool_calls"][0]["name"] = "banking.get_balance"
            else:
                draft["proposed_tool_calls"][0]["arguments"]["invented_field"] = "x"
            with self.subTest(change=change):
                self.assertEqual(self.replay(draft)["automatic_status"], "fail")

    def test_dynamic_btc_required_argument_check_runs(self):
        draft = self.draft()
        del draft["proposed_tool_calls"][1]["arguments"]["device_id"]
        result = self.replay(draft)
        self.assertEqual(result["automatic_status"], "fail")
        self.assertTrue(any("device_id" in error for error in result["errors"]))

    def test_empty_read_result_is_flagged_without_blanket_rejection(self):
        draft = self.draft()
        draft["proposed_tool_calls"] = [{"name": "smart_home.get_status", "arguments": {"room": "phòng không có thiết bị"}}]
        draft["expected"]["terminal_reason"] = "lookup_complete"
        draft["expected"]["state_assertions"] = [{"path": "", "operator": "unchanged"}]
        draft["expected"]["allowed_mutation_paths"] = []
        result = self.replay(draft)
        self.assertEqual(result["automatic_status"], "pass")
        self.assertTrue(any(w.startswith("empty_result_list:") for w in result["warnings"]))

    def test_no_action_cannot_contain_calls(self):
        draft = self.draft()
        draft["expected"]["outcome"] = "no_action"
        draft["expected"]["terminal_reason"] = "user_cancelled"
        self.assertEqual(self.replay(draft)["automatic_status"], "fail")

    def test_cyclic_subgoals_are_rejected(self):
        draft = self.draft()
        draft["expected"]["subgoals"][0]["depends_on"] = ["g1"]
        self.assertEqual(self.replay(draft)["automatic_status"], "fail")

    def test_malformed_inputs_return_failed_reports(self):
        for draft in (None, [], {"id": "x"}):
            with self.subTest(draft=draft):
                self.assertEqual(self.replay(draft)["automatic_status"], "fail")

    def test_cli_reports_bad_json_duplicates_and_finite_numbers(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            input_path = folder / "drafts.jsonl"
            first = json.dumps(self.draft(), ensure_ascii=False)
            input_path.write_text(first + "\n" + first + '\n{"id":"a","id":"b"}\n{"x":NaN}\n{"x":1e999}\nnot JSON\n', encoding="utf-8")
            output = folder / "replay"
            summary = run_file(input_path, output, DATA)
            self.assertEqual(summary["records"], 6)
            self.assertEqual(summary["automatic_fail"], 6)
            self.assertEqual(summary["accepted_for_sft"], 0)
            self.assertTrue((output / "review.md").is_file())
            with self.assertRaises(ValueError):
                run_file(input_path, output, DATA)

    def test_demo_batch_has_three_automatic_passes_and_no_train_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "replay"
            summary = run_file(ROOT / "schemas/examples/cloud_drafts_review_demo.jsonl", output, DATA)
            self.assertEqual(summary["automatic_pass"], 3)
            self.assertEqual(summary["automatic_fail"], 2)
            self.assertEqual(summary["accepted_for_sft"], 0)
            reviews = [json.loads(line) for line in (output / "review_decisions.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["decision"] == "pending" for r in reviews), 3)


if __name__ == "__main__":
    unittest.main()
