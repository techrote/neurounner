"""Offline source-only sentinels for the public-synthetic post-SGD cold/KV witness.

No frozen model, remote tokenizer, heldout, or optimizer executed by these tests.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_poststep_cold_replay as gate  # noqa: E402


class ColdPostStepSourceChecks(unittest.TestCase):
    def report(self):
        return {
            "phase": "producer",
            "status": gate.STATUS,
            "diagnostic_optimizer_steps": 1,
            "train_authorized": False,
            "comparative_training_performed": False,
            "heldouts_opened": False,
            "full_72_conformance_pass": False,
            "independent_scientific_review": False,
            "model_repo": gate.EXPECTED_MODEL,
            "revision": gate.EXPECTED_REV,
            "weights_sha256": gate.EXPECTED_WEIGHTS,
            "sample": {"real_eos_id": 258,
                       "supervised_after_shift": 21, "total_units": 31},
            "source": {"head": "ab" * 20},
            "threads_effective_after_load": 4,
            "producer_pid": 1234,
            "active_vs_disabled_max_abs": 0.0001,
            "active_logits_sha256": "1" * 64,
            "frozen_backbone_sha256": "2" * 64,
            "adapter_sha256": "3" * 64,
            "adapter_metadata_sha256": "4" * 64,
        }

    def test_one_synthetic_step_envelope_passes_without_training_admission(self):
        gate.validate_producer(self.report())
        self.assertEqual(gate.STATUS,
                         "POSTSTEP_COLD_DIAGNOSTIC_ONLY__SCIENTIFIC_72_BLOCKED")
        self.assertEqual(len(gate.EXPECTED_BLOBS), 6)

    def test_failed_and_forged_producer_records_refused(self):
        def replace(path, value):
            def change(record):
                cursor = record
                for p in path[:-1]:
                    cursor = cursor[p]
                cursor[path[-1]] = value
            return change

        cases = (
            ("two_steps", replace(("diagnostic_optimizer_steps",), 2)),
            ("bool_as_step", replace(("diagnostic_optimizer_steps",), True)),
            ("unauthorized_training", replace(("train_authorized",), True)),
            ("comparative_claim", replace(("comparative_training_performed",), True)),
            ("heldout_access", replace(("heldouts_opened",), True)),
            ("false_admission", replace(("full_72_conformance_pass",), True)),
            ("false_review", replace(("independent_scientific_review",), True)),
            ("wrong_checkpoint", replace(("weights_sha256",), "f" * 64)),
            ("wrong_variant", replace(("model_repo",), "kandivault/sprocket-500m")),
            ("old_text_eos", replace(("sample", "real_eos_id"), 62)),
            ("no_shifted_eos", replace(("sample", "supervised_after_shift"), 20)),
            ("wrong_units", replace(("sample", "total_units"), 30)),
            ("old_thread_metadata", replace(("threads_effective_after_load",), 2)),
            ("bool_threads", replace(("threads_effective_after_load",), True)),
            ("no_effect", replace(("active_vs_disabled_max_abs",), 0.0)),
            ("nonfinite_effect", replace(("active_vs_disabled_max_abs",), math.inf)),
            ("invalid_digest", replace(("adapter_sha256",), "corrupted")),
            ("no_original_source", replace(("source", "head"), "")),
            ("wrong_phase", replace(("phase",), "consumer")),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                record = copy.deepcopy(self.report())
                mutate(record)
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    gate.validate_producer(record)

    def test_original_immutable_source_and_four_model_gate(self):
        with mock.patch.dict(
            "os.environ",
            {"NCR_EXPECTED_HEAD": "test-immutable-head",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        ), mock.patch.object(
            gate.subprocess, "check_output",
            side_effect=lambda command, **kw: (
                "test-immutable-head\n" if command[-2:] == ["rev-parse", "HEAD"]
                else gate.EXPECTED_BLOBS[Path(command[-1]).relative_to(gate.ROOT).as_posix()] + "\n"),
        ):
            result = gate.source_gate()
            self.assertEqual(result["head"], "test-immutable-head")
            self.assertEqual(result["original_source_git_blobs"],
                             gate.EXPECTED_BLOBS)
        self.assertIn("scripts/ncr_001_void_nca_learning_gate.py",
                      gate.EXPECTED_BLOBS)
        self.assertIn("scripts/ncr_001_void1.py", gate.EXPECTED_BLOBS)

    def test_reject_offline_or_source_drift(self):
        with mock.patch.dict(
            "os.environ",
            {"NCR_EXPECTED_HEAD": "test-immutable-head",
             "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "1"},
        ), mock.patch.object(
            gate.subprocess, "check_output",
            side_effect=lambda command, **kw: (
                "test-immutable-head\n" if command[-2:] == ["rev-parse", "HEAD"]
                else gate.EXPECTED_BLOBS[Path(command[-1]).relative_to(gate.ROOT).as_posix()] + "\n"),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                gate.source_gate()
        with mock.patch.dict(
            "os.environ",
            {"NCR_EXPECTED_HEAD": "wrong-head", "HF_HUB_OFFLINE": "1",
             "TRANSFORMERS_OFFLINE": "1"},
        ), mock.patch.object(
            gate.subprocess, "check_output", return_value="actual-head\n",
        ):
            with self.assertRaisesRegex(RuntimeError, "nonliteral checkout"):
                gate.source_gate()


if __name__ == "__main__":
    unittest.main()
