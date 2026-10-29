"""Offline DETECTIVE-003B live-session integration tests."""
from __future__ import annotations

import io
import json
import pathlib
import tarfile
import tempfile
import unittest

from scripts import detective_003a_session as session_api
from scripts import detective_003b_live_session as live
from scripts import detective_003b_verify as verify

ROOT = pathlib.Path(__file__).resolve().parents[1]


class LiveSessionTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.case, self.c2, self.c3 = live.load_contract()

    def test_exact_authority_and_no_training_expansion_boundary(self):
        self.assertEqual(self.cfg["issue"], 142)
        self.assertEqual(
            self.cfg["authority"]["session_module"]["git_blob_sha1"],
            "73801ea59ce0f6160acc7027bd6aeaa085950236",
        )
        self.assertEqual(
            self.cfg["authority"]["case"]["git_blob_sha1"],
            "bbe5e0b3991a8296033376147bf830b05d33d4bb",
        )
        self.assertEqual(
            self.cfg["witnesses"]["V-G02-A"]["directory_sha256"],
            "a3185187e81634e11079aa6eb58ee87b495dac260fc3b2dd17120eed8a9d2ae3",
        )
        self.assertEqual(
            self.cfg["witnesses"]["V-G09-C"]["directory_sha256"],
            "0c52050a0466b580129f72e9daea9079d8cdd9151bbb71171f101d130da7d23c",
        )
        self.assertEqual(self.cfg["runtime"]["optimizer_steps"], 0)
        self.assertEqual(self.cfg["runtime"]["paid_api_calls"], 0)
        self.assertFalse(self.cfg["acceptance"]["relevance_mapping_accessed"])
        self.assertFalse(self.cfg["acceptance"]["learner_final_accessed"])
        self.assertFalse(self.cfg["acceptance"]["body_selection_made"])
        self.assertFalse(self.cfg["acceptance"]["population_expansion_admitted"])

    def test_release_lineage_is_exact_c3_g02_and_frozen_c2_g09(self):
        c3_asset = self.c3["assets"][0]
        c2_asset = next(
            x for x in self.c2["assets"] if x["member_id"] == "V-G09-C"
        )
        self.assertEqual(
            c3_asset["sha256"],
            self.cfg["witnesses"]["V-G02-A"]["archive_sha256"],
        )
        self.assertEqual(
            c2_asset["sha256"],
            self.cfg["witnesses"]["V-G09-C"]["archive_sha256"],
        )
        self.assertEqual(self.c3["frozen_control"]["optimizer_steps"], 0)

    def test_literal_time_extractor_single_zero_same_repeat_and_ambiguous(self):
        one = live.extract_time_annotation("I remember 02:05.", self.cfg)
        self.assertEqual(one["status"], "single_time")
        self.assertEqual(one["distinct_times"], ["02:05"])
        self.assertEqual(one["annotations"], [{
            "proposition_key": "handoff-time",
            "value": "02:05",
            "epistemic_class": "statement",
        }])
        repeated = live.extract_time_annotation(
            "02:05, yes, still 02:05.", self.cfg
        )
        self.assertEqual(repeated["status"], "single_time")
        self.assertEqual(repeated["literal_times"], ["02:05", "02:05"])
        none = live.extract_time_annotation("No exact time.", self.cfg)
        self.assertEqual(none["status"], "no_time")
        self.assertEqual(none["annotations"], [])
        ambiguous = live.extract_time_annotation(
            "I remember 02:05 but the receipt says 02:14.", self.cfg
        )
        self.assertEqual(ambiguous["status"], "ambiguous_times")
        self.assertEqual(ambiguous["annotations"], [])
        self.assertFalse(ambiguous["canonical_truth_accessed"])
        self.assertFalse(ambiguous["scorer_accessed"])

    def test_literal_time_extractor_rejects_invalid_clock_values(self):
        result = live.extract_time_annotation(
            "25:99 03:14 3:14 24:00", self.cfg
        )
        self.assertEqual(result["distinct_times"], ["03:14"])
        self.assertEqual(result["status"], "single_time")

    def test_renderer_is_identity_blind_and_preserves_section_order(self):
        session = session_api.start_session(
            self.case,
            self.cfg["session"]["participants"],
            self.cfg["session"]["session_key"],
        )
        receipt = session_api.build_prompt(
            self.case,
            session,
            "V-G02-A",
            self.cfg["session"]["sequence"][0]["question"],
        )
        rendered = live.render_session_prompt(receipt["prompt"], self.cfg)
        positions = [
            rendered.index("Admitted memories:"),
            rendered.index("Presented evidence:"),
            rendered.index("Prior same-witness dialogue:"),
            rendered.index("Player question:"),
        ]
        self.assertEqual(positions, sorted(positions))
        for forbidden in (
            "V-G02-A", "V-G09-C", "F-SWAP", "transfer-card",
            "OXIDATION", "claim_id", "epistemic_class",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_challenge_policy_selects_only_prior_g02_handoff_claims(self):
        session = session_api.start_session(
            self.case,
            self.cfg["session"]["participants"],
            self.cfg["session"]["session_key"],
        )
        receipt = session_api.build_prompt(
            self.case, session, "V-G02-A", "What time?"
        )
        session = session_api.record_turn(
            self.case,
            session,
            receipt,
            "02:05.",
            [{
                "proposition_key": "handoff-time",
                "value": "02:05",
                "epistemic_class": "statement",
            }],
        )
        receipt = session_api.build_prompt(
            self.case, session, "V-G09-C", "What time?"
        )
        session = session_api.record_turn(
            self.case,
            session,
            receipt,
            "02:09.",
            [{
                "proposition_key": "handoff-time",
                "value": "02:09",
                "epistemic_class": "statement",
            }],
        )
        self.assertEqual(
            live.challenge_ids_for_g02(session, self.case, self.cfg),
            ["C0001"],
        )

    def _write_adapter_archive(self, root: pathlib.Path):
        src = root / "src"
        src.mkdir()
        (src / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA"}) + "\n"
        )
        (src / "adapter_model.safetensors").write_bytes(b"synthetic-adapter")
        archive = root / "adapter.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            for p in sorted(src.iterdir()):
                tf.add(p, arcname=p.name)
        return archive, live.directory_hash(src)

    def test_safe_unpack_accepts_bounded_archive_and_checks_directory_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            archive, dir_hash = self._write_adapter_archive(root)
            dest = root / "dest"
            result = live.safe_unpack_adapter(
                archive,
                dest,
                expected_sha256=live.file_sha(archive),
                expected_bytes=archive.stat().st_size,
                expected_directory_sha256=dir_hash,
            )
            self.assertEqual(result["directory_sha256"], dir_hash)
            self.assertEqual(
                set(result["files"]),
                {"adapter_config.json", "adapter_model.safetensors"},
            )

    def test_safe_unpack_refuses_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            archive = root / "bad.tar.gz"
            payload = b"nope"
            with tarfile.open(archive, "w:gz") as tf:
                info = tarfile.TarInfo("../escape.txt")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            with self.assertRaises(live.LiveSessionError):
                live.safe_unpack_adapter(
                    archive,
                    root / "dest",
                    expected_sha256=live.file_sha(archive),
                    expected_bytes=archive.stat().st_size,
                    expected_directory_sha256="0" * 64,
                )

    def _synthetic_result(self):
        cfg = self.cfg
        case = self.case
        session = session_api.start_session(
            case, cfg["session"]["participants"], cfg["session"]["session_key"]
        )
        turns = []
        evidence_actions = []
        responses = {
            "L01": "I remember 02:05.",
            "L02": "I say 02:09.",
            "L04": "The physical receipt changes my account to 02:14.",
        }
        total = 0
        for step in cfg["session"]["sequence"]:
            if step["action"] == "evidence":
                challenged = live.challenge_ids_for_g02(session, case, cfg)
                session = session_api.present_evidence(
                    case,
                    session,
                    step["npc_id"],
                    step["asset_id"],
                    challenged_claim_ids=challenged,
                )
                evidence_actions.append({
                    "step_id": step["id"],
                    "npc_id": step["npc_id"],
                    "asset_id": step["asset_id"],
                    "challenge_policy": step["challenge_policy"],
                    "challenged_claim_ids": challenged,
                    "session_sha256_after": session["sha256"],
                })
                continue
            receipt = session_api.build_prompt(
                case,
                session,
                step["npc_id"],
                step["question"],
                include_history=step["include_history"],
            )
            rendered = live.render_session_prompt(receipt["prompt"], cfg)
            response = responses[step["id"]]
            extraction = live.extract_time_annotation(response, cfg)
            before = session["sha256"]
            session = session_api.record_turn(
                case, session, receipt, response, extraction["annotations"]
            )
            witness = cfg["witnesses"][step["npc_id"]]
            ids = [101 + len(turns), 201 + len(turns)]
            total += len(ids)
            turns.append({
                "step_id": step["id"],
                "npc_id": step["npc_id"],
                "adapter_name": witness["adapter_name"],
                "adapter_directory_sha256": witness["directory_sha256"],
                "session_sha256_before": before,
                "prompt_receipt": receipt,
                "rendered_user_prompt": rendered,
                "messages": [
                    {"role": "system", "content": cfg["prompt_renderer"]["system"]},
                    {"role": "user", "content": rendered},
                ],
                "prompt_token_count": 120,
                "prompt_token_sha256": str(len(turns) + 1) * 64,
                "generated_token_ids": ids,
                "generated_tokens": len(ids),
                "generation_seconds": 0.1,
                "completion_tokens_per_second": 20.0,
                "hit_max_new_tokens": False,
                "response": response,
                "response_with_special_tokens": response,
                "annotation": extraction,
                "session_sha256_after": session["sha256"],
            })
        derived = session_api.replay_session(case, session)
        csha = live.case_sha(case)
        return {
            "schema": "detective-003b-live-session-run-v1",
            "experiment": "DETECTIVE-003B",
            "issue": 142,
            "status": "integration_pass",
            "repository": {
                "head_sha": "a" * 40,
                "run_id": "12345",
                "run_url": "https://github.com/techrote/neurounner/actions/runs/12345",
            },
            "config_sha256": live.file_sha(live.CONFIG),
            "case_file_sha256": live.file_sha(live.CASE),
            "case_before_sha256": csha,
            "case_after_sha256": csha,
            "source": {
                "g02_adapter": {
                    "archive_sha256": cfg["witnesses"]["V-G02-A"]["archive_sha256"],
                    "archive_bytes": cfg["witnesses"]["V-G02-A"]["archive_bytes"],
                    "directory_sha256": cfg["witnesses"]["V-G02-A"]["directory_sha256"],
                    "files": ["adapter_config.json", "adapter_model.safetensors"],
                },
                "g09_adapter": {
                    "archive_sha256": cfg["witnesses"]["V-G09-C"]["archive_sha256"],
                    "archive_bytes": cfg["witnesses"]["V-G09-C"]["archive_bytes"],
                    "directory_sha256": cfg["witnesses"]["V-G09-C"]["directory_sha256"],
                    "files": ["adapter_config.json", "adapter_model.safetensors"],
                },
                "base_weight": {
                    "path": "/tmp/model.safetensors",
                    "sha256": cfg["base_model"]["weight_sha256"],
                    "bytes": 269060552,
                },
            },
            "environment": {
                "platform": "test",
                "python": "3.12",
                "cpu": "test CPU",
                "cpu_count": 4,
                "torch": "2.14.1+cpu",
                "transformers": "4.57.6",
                "peft": "0.21.2",
                "torch_threads": 4,
                "torch_interop_threads": 1,
                "cuda_available": False,
                "load_seconds": 0.2,
                "rss_after_load_kib": 100000,
                "peak_rss_after_load_kib": 120000,
            },
            "model_turns": turns,
            "evidence_actions": evidence_actions,
            "final_session": session,
            "final_derived": derived,
            "mandatory_public_routes_intact": True,
            "optimizer_steps": 0,
            "paid_api_calls": 0,
            "relevance_mapping_accessed": False,
            "learner_final_accessed": False,
            "body_selection_made": False,
            "population_expansion_admitted": False,
            "total_generated_tokens": total,
            "wall_seconds": 0.3,
            "peak_rss_kib": 120000,
            "decision": {
                "integration_pass": True,
                "claim_count": len(derived["claims"]),
                "contradiction_count": len(derived["contradictions"]),
                "minimum_claim_count_required": 0,
                "minimum_contradiction_count_required": 0,
                "training_admitted": False,
                "body_selection_made": False,
                "population_expansion_admitted": False,
            },
        }

    def test_synthetic_live_session_archive_replays_end_to_end(self):
        result = self._synthetic_result()
        self.assertEqual(len(result["final_derived"]["claims"]), 3)
        self.assertEqual(len(result["final_derived"]["contradictions"]), 3)
        self.assertEqual(
            result["final_derived"]["claims"][0]["challenged_by_event_ids"],
            ["E0003"],
        )
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            dataset = live.archive_dataset(root, result, self.cfg)
            checked = verify.verify(dataset)
            self.assertEqual(
                checked["verdict"],
                "DETECTIVE_003B_LIVE_SESSION_READBACK_PASS",
            )
            self.assertEqual(checked["model_turns"], 3)
            self.assertEqual(checked["claim_count"], 3)
            self.assertEqual(checked["contradiction_count"], 3)
            self.assertTrue(checked["case_unchanged"])

    def test_ambiguous_response_creates_no_claim_and_is_valid(self):
        result = self._synthetic_result()
        # Rebuild a smaller session to directly prove no minimum claim contract.
        session = session_api.start_session(
            self.case,
            self.cfg["session"]["participants"],
            "ambiguous-fixture",
        )
        receipt = session_api.build_prompt(
            self.case, session, "V-G02-A", "What time?"
        )
        extraction = live.extract_time_annotation(
            "Maybe 02:05, maybe 02:14.", self.cfg
        )
        self.assertEqual(extraction["annotations"], [])
        session = session_api.record_turn(
            self.case, session, receipt,
            "Maybe 02:05, maybe 02:14.", extraction["annotations"]
        )
        self.assertEqual(
            session_api.replay_session(self.case, session)["claims"], []
        )
        self.assertEqual(self.cfg["acceptance"]["minimum_claim_count"], 0)

    def test_archive_tamper_is_detected_by_reconstruction(self):
        result = self._synthetic_result()
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            dataset = live.archive_dataset(root, result, self.cfg)
            run_path = dataset / "raw/run.json"
            doc = json.loads(run_path.read_text())
            doc["model_turns"][0]["response"] = "02:55."
            live.write_json(run_path, doc)
            # Rehash inventory so semantic reconstruction, not only file hash,
            # must reject the mutation.
            (dataset / "provenance/hashes.sha256").write_text(
                live._hash_inventory(dataset)
            )
            with self.assertRaises(verify.VerifyError):
                verify.verify(dataset)

    def test_source_has_no_learner_final_or_relevance_packet_dependency(self):
        cfg_blob = live.CONFIG.read_text().casefold()
        source = pathlib.Path(
            ROOT / "scripts/detective_003b_live_session.py"
        ).read_text().casefold()
        for forbidden in (
            "final-suite.json",
            "relevance-packet.json",
            "identity_mapping",
        ):
            self.assertNotIn(forbidden, cfg_blob)
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
