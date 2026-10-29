"""DETECTIVE-001B: no downloads, trained model imports or paid APIs."""

from __future__ import annotations

import copy
import io
import json
import pathlib
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import detective_001b_pilot as pilot
from scripts import detective_001b_verify as verifier


class PilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.case, self.release = pilot.check_config()

    def test_plan_is_pinned_and_comparable(self):
        p = pilot.plan()
        self.assertEqual(p["lanes"], ["BASE", "V-G02-A", "V-G09-C"])
        self.assertEqual(p["training_prompt_collisions"], 0)
        self.assertEqual(p["maximum_generated_tokens"], 1120)
        self.assertEqual(p["deferred_lanes"], ["sprocket-chat", "sprocket-instruct", "void1"])
        self.assertEqual(p["common_first_turns"][0]["messages"],
                         pilot.messages_first(self.config, self.case, 0))
        self.assertEqual(p["common_first_turns"][1]["messages"],
                         pilot.messages_first(self.config, self.case, 1))
        self.assertNotEqual(p["common_first_turns"][0]["messages"],
                            p["common_first_turns"][1]["messages"])

    def test_blind_prompts_exclude_model_identity_and_secrets(self):
        blob = json.dumps(pilot.messages_first(self.config, self.case, 0))
        for field in ("V-G02-A", "V-G09-C", "OXIDATION", "F-PRIVATE",
                      "backstory", "motivation", "fault", "npc_id"):
            self.assertNotIn(field, blob)
        followup = pilot.messages_followup(self.config, self.case, 0, "The clock swallowed nine.")
        self.assertEqual(len(followup), 4)
        self.assertEqual(followup[0:2], pilot.messages_first(self.config, self.case, 0))
        self.assertEqual(followup[2], {"role": "assistant", "content": "The clock swallowed nine."})
        self.assertNotIn("known_fact_ids", followup[-1]["content"])
        self.assertNotIn("OXIDATION", followup[-1]["content"])

    def test_scripter_prompts_do_not_leak_locked_content(self):
        task = self.config["scripter"]["tasks"][0]
        text = json.dumps(pilot.messages_script(self.config, self.case, task))
        self.assertIn("transfer-card", text)
        self.assertNotIn("camera-cache", text)
        self.assertNotIn("OXIDATION", text)
        self.assertNotIn("F-PRIVATE", text)
        self.assertNotIn("The foreman's private key", text)

    def test_valid_script_scored_from_game_receipt(self):
        tasks = self.config["scripter"]["tasks"]
        a = json.dumps({"schema": "detective-script-v1",
                        "steps": [{"op": "read_asset", "asset_id": "transfer-card"}]})
        b = json.dumps({"schema": "detective-script-v1",
                        "steps": [{"op": "correlate_assets",
                                   "left_id": "transfer-card",
                                   "right_id": "dock-clock-ledger"}]})
        self.assertTrue(pilot.grade_script(self.case, tasks[0], a)["task_success"])
        self.assertTrue(pilot.grade_script(self.case, tasks[1], b)["task_success"])
        not_json = pilot.grade_script(self.case, tasks[0], "I have read the record.")
        self.assertFalse(not_json["schema_valid"])
        self.assertFalse(not_json["task_success"])
        wrong = pilot.grade_script(self.case, tasks[0], b)
        self.assertTrue(wrong["schema_valid"])
        self.assertFalse(wrong["task_success"])

    def test_disallowed_probe_is_counted_not_executed_as_host_tool(self):
        task = self.config["scripter"]["tasks"][0]
        answer = json.dumps({"schema": "detective-script-v1",
                             "steps": [{"op": "probe_terminal",
                                        "terminal_id": "camera-console",
                                        "credential": "nope"}]})
        grade = pilot.grade_script(self.case, task, answer)
        self.assertTrue(grade["schema_valid"])
        self.assertFalse(grade["task_success"])
        self.assertEqual(grade["unsafe_probe_attempts"], 1)
        self.assertEqual(grade["receipt"]["events"][0]["result"]["status"], "DENIED")

    def test_predeclared_budget_and_stratum_refuse_mutation(self):
        prior = pilot.read_json
        def with_mutation(path):
            obj = prior(path)
            if path == pilot.CONFIG:
                obj["budget"]["maximum_training_steps"] = 4
            return obj
        with patch.object(pilot, "read_json", side_effect=with_mutation):
            with self.assertRaises(pilot.ContractError):
                pilot.check_config()

        def wrong_stratum(path):
            obj = prior(path)
            if path == pilot.CONFIG:
                obj["selection"]["full_member_stratum"].append("V-G01-B")
            return obj
        with patch.object(pilot, "read_json", side_effect=wrong_stratum):
            with self.assertRaises(pilot.ContractError):
                pilot.check_config()

    def test_malicious_adapter_tar_refuses_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as tf:
                for filename in ("../escape.txt", "adapter_model.safetensors",
                                 "adapter_config.json"):
                    content = b"x"
                    info = tarfile.TarInfo(filename)
                    info.size = len(content)
                    tf.addfile(info, io.BytesIO(content))
            asset = {"sha256": pilot.file_sha(archive),
                     "bytes": archive.stat().st_size,
                     "adapter_directory_sha256": "0" * 64}
            with self.assertRaises(pilot.ContractError):
                pilot.unpack_adapter(archive, root / "out", asset)
            self.assertFalse((root / "escape.txt").exists())

    def _synthetic_run(self) -> dict:
        config, case, release = self.config, self.case, self.release
        p = pilot.plan()
        lane_meta = []
        turns = []
        members = {x["member_id"]: x for x in release["assets"]}
        def add(lane, task_id, msg, kind, *, turn=None, grade=None, answer="Recorded."):
            record = {
                "id": f"{lane}-{task_id}", "lane_id": lane, "type": kind,
                "situation_id": task_id[:-1] if kind == "interview" else task_id,
                "request": {
                    "messages": msg, "prompt_token_count": 42,
                    "input_token_sha256": "4" * 64,
                    "max_new_tokens": 80, "do_sample": False
                },
                "response": answer, "response_raw_with_special_tokens": answer,
                "generated_token_ids": [7, 8], "generated_tokens": 2,
                "elapsed_seconds": 0.1, "tokens_per_second": 20.0,
            }
            if turn is not None:
                record["turn"] = turn
            if grade is not None:
                record["grade"] = grade
            turns.append(record)

        for lid in p["lanes"]:
            a = members.get(lid)
            lane_meta.append({
                "lane_id": lid,
                "kind": "untuned_smollm2_control" if lid == "BASE" else "village_001c_sft_lora",
                "base_weights_sha256": config["base"]["weights_sha256"],
                "adapter_archive_sha256": a["sha256"] if a else None,
                "adapter_directory_sha256": a["adapter_directory_sha256"] if a else None,
                **({"member_contract_sha256": release["member_contract_sha256"]} if a else {}),
            })
            for idx in range(2):
                add(lid, f"I{idx+1:02d}A", pilot.messages_first(config, case, idx),
                    "interview", turn=1)
                add(lid, f"I{idx+1:02d}B", pilot.messages_followup(config, case, idx, "Recorded."),
                    "interview", turn=2)
            if lid == "BASE":
                for task in config["scripter"]["tasks"]:
                    if task["expected_op"] == "read_asset":
                        step = {"op": "read_asset", "asset_id": "transfer-card"}
                    else:
                        step = {"op": "correlate_assets", "left_id": "dock-clock-ledger",
                                "right_id": "transfer-card"}
                    answer = json.dumps({"schema": "detective-script-v1", "steps": [step]})
                    add(lid, task["id"], pilot.messages_script(config, case, task),
                        "scripter", grade=pilot.grade_script(case, task, answer), answer=answer)
        return {
            "schema": "detective-real-smoke-run-v1",
            "experiment": "DETECTIVE-001B", "admission": "EXPLORATORY_NOT_ADMITTED",
            "behavioral_truth_quality": "UNSCORED", "complete": True,
            "config_sha256": pilot.file_sha(pilot.CONFIG),
            "case_sha256": pilot.file_sha(pilot.CASE),
            "plan_sha256": pilot.digest_bytes(pilot.canonical(p).encode("utf-8")),
            "release_manifest_sha256": pilot.file_sha(pilot.MANIFEST),
            "runner_script_sha256": pilot.file_sha(pilot.Path(pilot.__file__)),
            "sources": {"base": config["base"],
                        "base_weights_sha256_verified": config["base"]["weights_sha256"]},
            "repository": {"head_sha": "a" * 40, "run_id": "123",
                           "run_url": "https://github.com/techrote/neurounner/actions/runs/123"},
            "lanes": lane_meta, "turns": turns,
            "total_generated_tokens": 28,
        }

    def test_archival_readback_uses_real_run_provenance_not_only_manifest(self):
        run = self._synthetic_run()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = pilot.archive_dataset(pathlib.Path(tmp), run, self.config)
            report = verifier.verify_dataset(dataset)
            self.assertEqual(report["verdict"], "OFFLINE_SOURCE_READBACK_PASS")
            self.assertEqual(report["turns"], 14)
            self.assertEqual(report["script_tasks_successful"], 2)
            self.assertFalse(report["training_claim"])
            self.assertFalse(report["behavioral_claim"])

    def test_altered_interview_input_refuses_even_with_rehashed_files(self):
        run = self._synthetic_run()
        run["turns"][0]["request"]["messages"][1]["content"] = '{"illegal_world_secret":true}'
        with tempfile.TemporaryDirectory() as tmp:
            dataset = pilot.archive_dataset(pathlib.Path(tmp), run, self.config)
            with self.assertRaisesRegex(verifier.EvidenceError, "unmatched blind prompt"):
                verifier.verify_dataset(dataset)

    def test_altered_script_grading_refuses_even_with_rehashed_files(self):
        run = self._synthetic_run()
        item = next(t for t in run["turns"] if t["id"] == "BASE-S01")
        item["grade"]["task_success"] = False
        with tempfile.TemporaryDirectory() as tmp:
            dataset = pilot.archive_dataset(pathlib.Path(tmp), run, self.config)
            with self.assertRaisesRegex(verifier.EvidenceError, "grade not reproducible"):
                verifier.verify_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
