"""CPU integration checks for data isolation and irreversible tool sequencing.

Scripted outputs test plumbing only; they say nothing about model accuracy.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from lifemate.agent import make_messages, parse_action, run_agent, tools_for
from lifemate.data import audio_turns, load_catalog, load_runtime, make_split, read_rows
from lifemate.evaluate import exact_call_f1

ROOT = Path(__file__).resolve().parents[1] / "data"


class ScriptedModel:
    def __init__(self, calls):
        self.outputs = iter([f"<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>" for c in calls] + ["DONE"])

    def generate(self, messages, tools, remaining_seconds):
        return next(self.outputs), {"input_tokens": 0, "output_tokens": 0}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_rows(ROOT, "train")
        cls.runtime = load_runtime(ROOT, "train")
        cls.catalog = load_catalog(ROOT, "train")

    def test_prompt_whitelist_and_history_are_preserved(self):
        row = next(r for r in self.rows if r["history"])
        transcripts = {t["audio"]: "ASR_ONLY" for t in audio_turns(row)}
        prompt = json.loads(make_messages(row, transcripts)[1]["content"])
        self.assertEqual(set(prompt), {"context", "history", "latest_request"})
        self.assertEqual(prompt["latest_request"], "ASR_ONLY")
        self.assertEqual(len(prompt["history"]), len(row["history"]))
        self.assertTrue(all(h["text"] == "ASR_ONLY" for h in prompt["history"] if h["role"] == "user"))
        self.assertNotIn('"audio_text"', json.dumps(prompt))
        self.assertNotIn('"final_state"', json.dumps(prompt))

    def test_catalog_is_dynamic_and_limited_to_installed_tools(self):
        row = self.rows[0]
        actual = {t["function"]["name"] for t in tools_for(row, self.catalog)}
        expected = {t for a in row["context"]["installed_apps"].values() for t in a["tools"]} | {"sys.ask_missing_info"}
        self.assertEqual(actual, expected)
        public_catalog = load_catalog(ROOT, "public_test")
        public_row = next(r for r in read_rows(ROOT, "public_test") if "ecommerce" in r["context"]["installed_apps"])
        self.assertIn("ecommerce.search", {t["function"]["name"] for t in tools_for(public_row, public_catalog)})

    def test_calls_are_executed_once_and_hash_matches_gold_replay(self):
        row = next(r for r in self.rows if r["history"] and len(r["label"]["tool_calls"]) >= 2 and all(c["name"] != "sys.ask_missing_info" for c in r["label"]["tool_calls"]))
        calls = row["label"]["tool_calls"]
        env = self.runtime.ToolEnv("train", row["id"], root=ROOT)
        texts = {t["audio"]: t["audio_text"] for t in audio_turns(row)}
        result, details = run_agent(row, texts, env, self.catalog, ScriptedModel(calls), time.perf_counter()+30)
        self.assertEqual(result["tool_calls"], calls)  # no replay of historical calls
        reference = self.runtime.ToolEnv("train", row["id"], root=ROOT)
        for call in calls:
            reference.call(call["name"], call["arguments"])
        self.assertEqual(result["state_hash"], reference.get_state_hash())
        self.assertEqual(details["stop_reason"], "done")

    def test_invalid_schema_is_repaired_before_any_execution(self):
        row = self.rows[0]
        env = self.runtime.ToolEnv("train", row["id"], root=ROOT)
        texts = {t["audio"]: t["audio_text"] for t in audio_turns(row)}
        invalid = {"name": "banking.transfer", "arguments": {}}
        result, details = run_agent(row, texts, env, self.catalog, ScriptedModel([invalid]), time.perf_counter()+30)
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(details["repairs"], 1)
        self.assertEqual(details["stop_reason"], "done")

    def test_missing_information_stops_after_recorded_system_call(self):
        row = next(r for r in self.rows if any(c["name"] == "sys.ask_missing_info" for c in r["label"]["tool_calls"]))
        env = self.runtime.ToolEnv("train", row["id"], root=ROOT)
        texts = {t["audio"]: t["audio_text"] for t in audio_turns(row)}
        calls = row["label"]["tool_calls"]
        result, detail = run_agent(row, texts, env, self.catalog, ScriptedModel(calls), time.perf_counter()+30)
        first_ask = next(i for i, c in enumerate(calls) if c["name"] == "sys.ask_missing_info")
        self.assertEqual(result["tool_calls"], calls[:first_ask+1])
        self.assertEqual(detail["stop_reason"], "missing_info")

    def test_parser_rejects_parallel_or_incomplete_outputs(self):
        block = '<tool_call>{"name":"alarm.list","arguments":{}}</tool_call>'
        self.assertEqual(parse_action(block)["name"], "alarm.list")
        for output in (block+block, block+'<tool_call>{', '<tool_call>{"name":'):
            with self.assertRaises(ValueError):
                parse_action(output)

    def test_split_is_deterministic_and_keeps_identical_requests_together(self):
        train, dev = make_split(self.rows)
        self.assertEqual((train, dev), make_split(self.rows))
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(len(train)+len(dev), len(self.rows))
        # Every row contains its complete history; no audio-level random split.
        current_texts = {r["audio_text"] for r in self.rows if r["id"] in set(train)}
        self.assertFalse(current_texts & {r["audio_text"] for r in self.rows if r["id"] in set(dev)})

    def test_f1_matches_duplicate_calls_only_once(self):
        call = {"name": "alarm.list", "arguments": {}}
        self.assertAlmostEqual(exact_call_f1([call, call], [call]), 2/3)
        self.assertEqual(exact_call_f1([], []), 1)
        self.assertEqual(exact_call_f1([call], []), 0)

    def test_evaluator_self_check_uses_runtime_replay(self):
        selected = [next(r for r in self.rows if (
            not r["label"]["tool_calls"] if kind == "empty" else
            any(c["name"] == "sys.ask_missing_info" for c in r["label"]["tool_calls"]) if kind == "missing" else
            r["label"]["tool_calls"] and all(c["name"] != "sys.ask_missing_info" for c in r["label"]["tool_calls"])
        )) for kind in ("empty", "missing", "execute")]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            predictions, details = [], []
            for row in selected:
                env = self.runtime.ToolEnv("train", row["id"], root=ROOT)
                for call in row["label"]["tool_calls"]:
                    env.call(call["name"], call["arguments"])
                predictions.append({"id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()})
                details.append({"id": row["id"], "stop_reason": "done", "mode": "oracle", "seconds": 0.01,
                                "transcripts": {t["audio"]: t["audio_text"] for t in audio_turns(row)}})
            (run / "run_config.json").write_text(json.dumps({"split": "train", "mode": "oracle", "sample_ids": [r["id"] for r in selected]}))
            for filename, values in (("submission.jsonl", predictions), ("diagnostics.jsonl", details)):
                (run / filename).write_text("".join(json.dumps(v, ensure_ascii=False)+"\n" for v in values))
            result = subprocess.run([sys.executable, "-m", "lifemate.evaluate", "--data-root", str(ROOT), "--run", str(run)],
                                    env=dict(os.environ, PYTHONPATH=str(ROOT.parent / "src")), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((run / "evaluation.json").read_text())
            self.assertEqual(report["state_match_rate"], 1)
            self.assertEqual(report["state_gated_exact_f1_percent"], 100)
            self.assertEqual(report["asr"]["coverage"], 1)


if __name__ == "__main__":
    unittest.main()
