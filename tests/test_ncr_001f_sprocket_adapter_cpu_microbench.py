"""NCR-001F adapter-only CPU prepass sentinels. No real LM or training."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import ncr_001f_sprocket_adapter_cpu_microbench as m


class SprocketAdapterOnlyCPUTests(unittest.TestCase):
    def test_exact_source_and_distinct_unexecuted_variants(self):
        source = m.pins.audit()
        self.assertEqual(set(source["lanes"]),
                         {"sprocket_chat", "sprocket_instruct"})
        self.assertNotEqual(
            source["lanes"]["sprocket_chat"]["weight_sha256"],
            source["lanes"]["sprocket_instruct"]["weight_sha256"])
        self.assertNotEqual(
            source["lanes"]["sprocket_chat"]["weight_path"],
            source["lanes"]["sprocket_instruct"]["weight_path"])
        self.assertFalse(source["train_authorized"])

    def test_actual_pytorch_module_shapes_not_just_arithmetic(self):
        nca, mlp = m.make_modules()
        self.assertEqual(nca.trainable_parameters(), 658904)
        self.assertEqual(mlp.trainable_parameters(), 658432)
        self.assertEqual(nca.steps_per_tap, 3)
        self.assertEqual(nca.tap_count, mlp.tap_count)
        self.assertEqual(nca.field_size, 256)

    def test_original_zero_readout_three_tap_forward_and_request_reset(self):
        nca, mlp = m.make_modules()
        first = m.synthetic_hidden(1)
        second = m.synthetic_hidden(16)
        a = m.three_taps("nca", nca, first)
        m.three_taps("nca", nca, second)
        b = m.three_taps("nca", nca, first)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, first[-1]))
        self.assertTrue(torch.equal(m.three_taps("mlp", mlp, first),
                                    first[-1]))
        self.assertFalse(any("field" in name for name, _ in nca.named_buffers()))

    def test_reject_unpreregistered_geometry_and_budget(self):
        for lengths, warmups, repeats in (
            ((2048,), 1, 2),
            ((0,), 1, 2),
            ((1, 1), 1, 2),
            ((1,), 0, 2),
            ((1,), 1, 1),
            ((1,), 1, True),
        ):
            with self.subTest(lengths=lengths, warmups=warmups, repeats=repeats):
                with self.assertRaisesRegex(ValueError, "REFUSED"):
                    m.validate_budget(lengths, warmups, repeats)

    def test_reject_altered_primary_step_count(self):
        nca, mlp = m.make_modules()
        nca.steps_per_tap = 2
        with patch.object(m, "PersistentDepthNCA", return_value=nca):
            with patch.object(m, "GatedMLPControl", return_value=mlp):
                with self.assertRaisesRegex(ValueError, "REFUSED"):
                    m.make_modules()

    def test_reject_training_flag_or_missing_variant_before_microbench(self):
        original = m.pins.audit()
        for mutation in ("train", "lane"):
            forged = copy.deepcopy(original)
            if mutation == "train":
                forged["train_authorized"] = True
            else:
                del forged["lanes"]["sprocket_instruct"]
            with self.subTest(mutation=mutation):
                with patch.object(m.pins, "audit", return_value=forged):
                    with self.assertRaisesRegex(ValueError, "REFUSED"):
                        m.run((1,), 1, 2)

    def test_validation_is_outside_timed_adapter_samples(self):
        nca, mlp = m.make_modules()
        actual_three_taps = m.three_taps
        calls = []

        def watched(kind, module, depths, *, validate=True):
            calls.append((kind, validate))
            return actual_three_taps(kind, module, depths, validate=validate)

        with patch.object(m, "three_taps", side_effect=watched):
            row = m.measured_case(nca, mlp, length=1, warmups=1, repeats=2)
        self.assertEqual(calls, [
            ("nca", True), ("mlp", True),
            ("nca", False), ("mlp", False),
            ("mlp", False), ("nca", False),
            ("nca", True), ("mlp", True),
        ])
        self.assertIs(row["timed_interval_includes_noop_or_finite_checks"], False)
        self.assertIs(row["warmup_and_postflight_noop_state_checks"], True)

    def test_corrupt_noop_is_still_refused_before_measurement(self):
        nca, _ = m.make_modules()

        class BadMLP:
            def step(self, hidden, *, tap_index):
                return hidden + 1

        with self.assertRaisesRegex(ValueError, "no-op parity"):
            m.measured_case(nca, BadMLP(), length=1, warmups=1, repeats=2)

    def test_bounded_cpu_samples_do_not_promote_science(self):
        report = m.run((1,), 1, 2)
        self.assertEqual(report["status"], m.STATUS)
        self.assertEqual(len(report["synthetic_cases"]), 1)
        for series in report["synthetic_cases"][0][
                "raw_forward_three_tap_ms"].values():
            self.assertEqual(len(series), 2)
            self.assertTrue(all(sample > 0 for sample in series))
        for key in ("frozen_model_weights_downloaded",
                    "frozen_model_inference_executed", "tokenizers_executed",
                    "heldouts_opened", "train_authorized",
                    "full_72_conformance_pass",
                    "actual_matched_frozen_host_budget_pass",
                    "training_performed"):
            self.assertIs(report[key], False, key)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["model_free_shared_geometry"][
            "nca_over_mlp_mac_ratio"], 1.20625)


if __name__ == "__main__":
    unittest.main()
