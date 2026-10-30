"""Offline DETECTIVE-003D source-native Sprocket bridge tests."""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile
import unittest

from scripts import detective_001b3_sprocket as b3
from scripts import detective_003a_session as session_api
from scripts import detective_003c_scripter_bridge as bridge
from scripts import detective_003d_sprocket_live as live
from scripts import detective_003d_verify as verify


class SprocketLiveBridgeTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.case, self.b3_spec = live.load_contract()
        self.session, self.request = live.build_request(self.cfg, self.case)

    def native(self, name, arguments):
        return b3.tool_call_text({
            "name": name,
            "arguments": arguments,
        }) + "<|end|>"

    def test_exact_source_authority_is_root_instruct_not_chat(self):
        self.assertEqual(
            self.cfg["model"]["weight_sha256"],
            "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
        )
        self.assertEqual(self.cfg["model"]["weight_file"], "model.safetensors")
        self.assertEqual(
            self.cfg["model"]["chat_checkpoint_forbidden"],
            "chat/model.safetensors",
        )
        self.assertEqual(self.cfg["model"], self.b3_spec["model"])
        self.assertEqual(
            self.cfg["publisher"]["commit"],
            "a003767e70d766bc14b40c89792543ee8641c8bc",
        )
        self.assertEqual(
            self.cfg["authority"]["b3_native_wire"]["git_blob_sha1"],
            "214c16301075113b51123db4afeca550b4226fb9",
        )
        self.assertEqual(
            self.cfg["authority"]["bridge_module"]["git_blob_sha1"],
            "3d97971a07f19cadf4f2c88ddfe105919331d5ad",
        )

    def test_game_request_exposes_correct_player_known_credential_but_not_hidden_asset(self):
        view = self.request["view"]
        blob = json.dumps(view, sort_keys=True)
        self.assertIn("camera-console", blob)
        self.assertIn("OXIDATION", blob)
        self.assertNotIn("camera-cache", blob)
        self.assertNotIn("F-CAMERA", blob)
        self.assertNotIn("V-G02-A", blob)
        self.assertNotIn("V-G09-C", blob)

    def test_publisher_prompt_is_source_native_and_identity_blind(self):
        turns, prompt = live.render_native_turns(self.cfg, self.request)
        self.assertEqual(turns[0]["role"], "system")
        self.assertTrue(turns[0]["content"].startswith(
            "You have access to these tools:\n"
        ))
        self.assertEqual(turns[1]["role"], "user")
        self.assertTrue(prompt.endswith("<|assistant|>"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn("camera-console", prompt)
        self.assertIn("OXIDATION", prompt)
        self.assertNotIn("camera-cache", prompt)
        self.assertNotIn("F-CAMERA", prompt)
        self.assertNotIn("V-G02-A", prompt)
        self.assertNotIn("V-G09-C", prompt)

    def test_valid_exact_probe_normalizes_to_one_step_detective_program(self):
        raw = self.native(
            "probe_terminal",
            {"terminal_id": "camera-console", "credential": "OXIDATION"},
        )
        norm = live.normalize_native_output(raw, self.b3_spec)
        self.assertEqual(norm["native_parse_status"], "parsed")
        self.assertEqual(norm["normalization_status"], "normalized")
        self.assertEqual(norm["program"], {
            "schema": "detective-script-v1",
            "steps": [{
                "op": "probe_terminal",
                "terminal_id": "camera-console",
                "credential": "OXIDATION",
            }],
        })
        self.assertTrue(live.exact_probe_selected(self.cfg, norm))

    def test_valid_native_but_unadmitted_read_file_is_rejected_without_repair(self):
        raw = self.native(
            "read_file",
            {"path": "assets/case_file.txt"},
        )
        norm = live.normalize_native_output(raw, self.b3_spec)
        self.assertEqual(norm["native_parse_status"], "parsed")
        self.assertEqual(norm["normalization_status"], "rejected")
        self.assertIsNone(norm["program"])
        self.assertFalse(live.exact_probe_selected(self.cfg, norm))

    def test_prose_and_malformed_native_output_are_rejected(self):
        for raw in (
            "I would inspect the console.",
            "<|tool_call|>{bad json}<|end|>",
            '<|tool_call|>{"name":"probe_terminal","arguments":{}}<|end|>',
        ):
            with self.subTest(raw=raw):
                norm = live.normalize_native_output(raw, self.b3_spec)
                if raw.startswith("<|tool_call|>") and '"arguments":{}' in raw:
                    self.assertEqual(norm["native_parse_status"], "parsed")
                    self.assertEqual(norm["normalization_status"], "rejected")
                else:
                    self.assertEqual(norm["native_parse_status"], "rejected")
                    self.assertEqual(norm["normalization_status"], "not_attempted")

    def test_correct_normalized_probe_executes_through_003c_and_grants_camera(self):
        norm = live.normalize_native_output(
            self.native(
                "probe_terminal",
                {"terminal_id": "camera-console", "credential": "OXIDATION"},
            ),
            self.b3_spec,
        )
        before = copy.deepcopy(self.session)
        tx = bridge.execute_scripter_transaction(
            self.case,
            self.session,
            self.request,
            norm["program_text"],
        )
        self.assertEqual(tx["status"], "executed")
        self.assertEqual(
            tx["execution_receipt"]["events"][0]["result"]["status"],
            "GRANTED",
        )
        self.assertIn("camera-cache", tx["visible_asset_ids_after"])
        self.assertEqual(self.session, before)

        applied = bridge.apply_transaction_evidence(
            self.case,
            self.session,
            self.request,
            tx,
            "V-G09-C",
            "camera-cache",
        )
        prompt = session_api.build_prompt(
            self.case,
            applied,
            "V-G09-C",
            "What does the newly presented archive excerpt show?",
        )
        blob = json.dumps(prompt["prompt"], sort_keys=True)
        self.assertIn("scanned twice", blob)
        for forbidden in ("camera-cache", "F-CAMERA", "camera-console", "OXIDATION"):
            self.assertNotIn(forbidden, blob)

    def test_normalized_wrong_tool_executes_but_does_not_force_grant(self):
        norm = live.normalize_native_output(
            self.native(
                "read_asset",
                {"asset_id": "transfer-card"},
            ),
            self.b3_spec,
        )
        self.assertEqual(norm["normalization_status"], "normalized")
        tx = bridge.execute_scripter_transaction(
            self.case, self.session, self.request, norm["program_text"]
        )
        self.assertEqual(tx["status"], "executed")
        self.assertNotIn("camera-cache", tx["visible_asset_ids_after"])
        self.assertFalse(live.exact_probe_selected(self.cfg, norm))

    def _synthetic_result(self, raw_output):
        norm = live.normalize_native_output(raw_output, self.b3_spec)
        transaction = None
        session_after = self.session
        witness_prompt = None
        camera_visible = False
        if norm["normalization_status"] == "normalized":
            transaction = bridge.execute_scripter_transaction(
                self.case, self.session, self.request, norm["program_text"]
            )
            camera_visible = (
                "camera-cache" in transaction["visible_asset_ids_after"]
            )
            if camera_visible:
                session_after = bridge.apply_transaction_evidence(
                    self.case,
                    self.session,
                    self.request,
                    transaction,
                    "V-G09-C",
                    "camera-cache",
                )
                witness_prompt = session_api.build_prompt(
                    self.case,
                    session_after,
                    "V-G09-C",
                    "What does the newly presented archive excerpt show?",
                )
        if norm["normalization_status"] != "normalized":
            outcome = "native_or_normalization_rejected"
        elif camera_visible:
            outcome = "normalized_executed_grant_applied"
        else:
            outcome = "normalized_executed_no_grant"
        turns, prompt_text = live.render_native_turns(self.cfg, self.request)
        exact = live.exact_probe_selected(self.cfg, norm)
        model_files = {
            "config.json": {"sha256": "0" * 64, "bytes": 1},
            "tokenizer.json": {
                "sha256": self.cfg["model"]["tokenizer_sha256"], "bytes": 1
            },
            "tokenizer_config.json": {
                "sha256": self.cfg["model"]["tokenizer_config_sha256"], "bytes": 1
            },
            "special_tokens_map.json": {
                "sha256": self.cfg["model"]["special_tokens_map_sha256"], "bytes": 1
            },
            "model.safetensors": {
                "sha256": self.cfg["model"]["weight_sha256"],
                "bytes": self.cfg["model"]["weight_bytes"],
            },
        }
        ids = [6, 123, 5]
        return {
            "schema": "detective-003d-sprocket-live-run-v1",
            "experiment": "DETECTIVE-003D",
            "issue": 146,
            "status": "integration_pass",
            "outcome": outcome,
            "repository": {
                "head_sha": "a" * 40,
                "run_id": "12345",
                "run_url": "https://github.com/techrote/neurounner/actions/runs/12345",
            },
            "source": {
                "publisher_checkout": {
                    "commit": self.cfg["publisher"]["commit"],
                    "git_blobs": {
                        self.cfg["publisher"]["sft_path"]:
                            self.cfg["publisher"]["sft_git_blob"],
                        self.cfg["publisher"]["harvest_path"]:
                            self.cfg["publisher"]["harvest_git_blob"],
                    },
                },
                "model_files": model_files,
            },
            "request_receipt": self.request,
            "publisher_turns": turns,
            "publisher_prompt_text": prompt_text,
            "prompt_token_count": 120,
            "prompt_token_sha256": "1" * 64,
            "generation": {
                "generated_token_ids": ids,
                "generated_tokens": len(ids),
                "raw_output_with_special_tokens": raw_output,
                "generation_seconds": 0.1,
                "completion_tokens_per_second": 30.0,
                "hit_max_new_tokens": False,
            },
            "normalization": norm,
            "exact_probe_selected": exact,
            "transaction": transaction,
            "camera_cache_visible_after_transaction": camera_visible,
            "evidence_application": {
                "applied": witness_prompt is not None,
                "selected_asset_id": "camera-cache" if witness_prompt else None,
                "witness_id": "V-G09-C" if witness_prompt else None,
            },
            "final_session": session_after,
            "witness_prompt_after_application": witness_prompt,
            "case_sha256": hashlib.sha256(
                live.canonical(self.case).encode("utf-8")
            ).hexdigest(),
            "environment": {
                "platform": "test",
                "python": "3.12",
                "cpu": "test CPU",
                "cpu_count": 4,
                "torch": "2.14.1+cpu",
                "transformers": "4.57.6",
                "torch_threads": 4,
                "torch_interop_threads": 1,
                "cuda_available": False,
                "model_load_seconds": 0.2,
                "rss_after_load_kib": 100000,
                "peak_rss_kib": 120000,
            },
            "decision": {
                "correct_tool_selection_required": False,
                "exact_probe_selected": exact,
                "access_granted": camera_visible,
                "evidence_applied": witness_prompt is not None,
                "host_execution": False,
                "training_admitted": False,
                "body_selection_made": False,
                "relevance_mapping_accessed": False,
                "learner_final_accessed": False,
                "population_expansion_admitted": False,
            },
        }

    def test_synthetic_correct_probe_archive_replays_end_to_end(self):
        raw = self.native(
            "probe_terminal",
            {"terminal_id": "camera-console", "credential": "OXIDATION"},
        )
        result = self._synthetic_result(raw)
        self.assertEqual(result["outcome"], "normalized_executed_grant_applied")
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            dataset = live.archive_dataset(root, result, self.cfg)
            checked = verify.verify(dataset)
            self.assertEqual(
                checked["verdict"],
                "DETECTIVE_003D_SPROCKET_LIVE_READBACK_PASS",
            )
            self.assertTrue(checked["exact_probe_selected"])
            self.assertTrue(checked["access_granted"])
            self.assertTrue(checked["evidence_applied"])

    def test_synthetic_native_failure_archive_is_still_integration_pass(self):
        result = self._synthetic_result(
            self.native("read_file", {"path": "assets/case_file.txt"})
        )
        self.assertEqual(result["outcome"], "native_or_normalization_rejected")
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            dataset = live.archive_dataset(root, result, self.cfg)
            checked = verify.verify(dataset)
            self.assertEqual(
                checked["outcome"], "native_or_normalization_rejected"
            )
            self.assertFalse(checked["exact_probe_selected"])
            self.assertFalse(checked["access_granted"])
            self.assertFalse(checked["evidence_applied"])

    def test_synthetic_valid_wrong_tool_archive_is_preserved(self):
        result = self._synthetic_result(
            self.native("read_asset", {"asset_id": "transfer-card"})
        )
        self.assertEqual(result["outcome"], "normalized_executed_no_grant")
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            dataset = live.archive_dataset(root, result, self.cfg)
            checked = verify.verify(dataset)
            self.assertEqual(
                checked["outcome"], "normalized_executed_no_grant"
            )
            self.assertFalse(checked["access_granted"])

    def test_tampered_raw_output_refuses_replay_even_if_inventory_rehashed(self):
        result = self._synthetic_result(
            self.native(
                "probe_terminal",
                {"terminal_id": "camera-console", "credential": "OXIDATION"},
            )
        )
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            dataset = live.archive_dataset(root, result, self.cfg)
            run_path = dataset / "raw/run.json"
            doc = json.loads(run_path.read_text())
            doc["generation"]["raw_output_with_special_tokens"] = (
                "I would inspect the console."
            )
            live.write_json(run_path, doc)
            (dataset / "provenance/hashes.sha256").write_text(
                live._hash_inventory(dataset)
            )
            with self.assertRaises(verify.VerifyError):
                verify.verify(dataset)

    def test_no_training_or_selection_authority(self):
        self.assertEqual(self.cfg["runtime"]["optimizer_steps"], 0)
        self.assertEqual(self.cfg["runtime"]["paid_api_calls"], 0)
        self.assertFalse(self.cfg["acceptance"]["correct_tool_selection_required"])
        self.assertFalse(self.cfg["acceptance"]["access_grant_required"])
        self.assertFalse(self.cfg["acceptance"]["evidence_application_required"])
        self.assertFalse(self.cfg["acceptance"]["host_execution"])
        self.assertFalse(self.cfg["acceptance"]["training_admitted"])
        self.assertFalse(self.cfg["acceptance"]["body_selection_made"])
        self.assertFalse(self.cfg["acceptance"]["relevance_mapping_accessed"])
        self.assertFalse(self.cfg["acceptance"]["learner_final_accessed"])
        self.assertFalse(self.cfg["acceptance"]["population_expansion_admitted"])


if __name__ == "__main__":
    unittest.main()
