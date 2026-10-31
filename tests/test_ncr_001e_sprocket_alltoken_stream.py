"""NCR-001E original-source B1 2048 every-token streaming KV refusal tests.

All 12 tests are offline: no frozen checkpoint download, tokenizer execution,
optimizer, training or sealed held-out evaluation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001e_sprocket_alltoken_stream as stream


class AllTokenSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "experiments/ncr-001/protocol.json").read_text())
        cls.pins = json.loads(
            (ROOT / "experiments/ncr-001/source-pins.json").read_text())

    def test_original_four_lane_model_source_and_distinct_sprockets(self):
        self.assertFalse(self.protocol["train_authorized"])
        self.assertEqual(set(self.protocol["models"]), stream.FOUR_LANES)
        self.assertEqual(set(self.pins["models"]), stream.FOUR_LANES)
        for lane in ("chat", "instruct"):
            spec = stream.fixed_contract(lane, self.protocol, self.pins)
            self.assertEqual(spec["weight_path"],
                             stream.source.LANE_WEIGHTS[lane][0])
        self.assertNotEqual(stream.source.LANE_WEIGHTS["chat"][1],
                            stream.source.LANE_WEIGHTS["instruct"][1])

    def test_refuse_open_training_and_missing_four_model_lanes(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["train_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.fixed_contract("chat", protocol, self.pins)
        pins = copy.deepcopy(self.pins)
        del pins["models"]["void1"]
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.fixed_contract("chat", self.protocol, pins)

    def test_refuse_sibling_checkpoint_and_wrong_integration(self):
        pins = copy.deepcopy(self.pins)
        pins["models"]["sprocket_chat"][
            "weight_declared_lfs_sha256_not_downloaded"
        ] = stream.source.LANE_WEIGHTS["instruct"][1]
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            stream.fixed_contract("chat", self.protocol, pins)
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.fixed_contract("void1", self.protocol, self.pins)

    def test_refuse_relaxed_cache_tolerances(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["conformance"]["full_vs_cached_prefill_atol"] = 0.001
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.fixed_contract("chat", protocol, self.pins)
        protocol = copy.deepcopy(self.protocol)
        protocol["conformance"]["full_vs_cached_prefill_rtol"] = 0.001
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.fixed_contract("instruct", protocol, self.pins)

    def test_refuse_negative_out_of_bounds_and_bool_indices(self):
        for bad in (-1, stream.CONTEXT, True, False, 1.5, "512"):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                    RuntimeError, "REFUSED"):
                stream.check_position(bad)
        for good in (0, 1, 511, 512, 1024, 2047):
            stream.check_position(good)

    def test_comparison_accepts_exact_causal_reference(self):
        x = torch.ones((1, 1, 8), dtype=torch.float32)
        self.assertEqual(stream.compare_logits(
            "exact", x, x.clone(), exact=True, atol=0, rtol=0), 0)
        self.assertEqual(stream.compare_logits("numeric", x, x.clone()), 0)

    def test_detects_forged_cache_logits_beyond_preregistered_gate(self):
        left = torch.ones((1, 1, 8), dtype=torch.float32)
        right = left.clone()
        right[:, :, 0] = 2
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.compare_logits("cache_negative", left, right)

    def test_detects_future_suffix_noncausal_drift_even_if_small(self):
        left = torch.ones((1, 6, 8), dtype=torch.float32)
        right = left.clone()
        right[:, 2, 5] += 1e-5
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.compare_logits("strict_future", left, right,
                                  exact=True, atol=0, rtol=0)

    def test_detects_nonnumeric_or_nonfinite_logits(self):
        x = torch.ones((1, 1, 8), dtype=torch.float32)
        y = torch.ones((1, 2, 8), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.compare_logits("shape", x, y)
        bad = x.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.compare_logits("nonfinite", x, bad)

    def test_disallows_nonprotocol_exact_comparison_tolerance(self):
        x = torch.ones((1, 1, 8), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.compare_logits("forged", x, x, exact=True,
                                  atol=1e-5, rtol=1e-5)
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            stream.compare_logits("forged", x, x,
                                  atol=0.01, rtol=0.01)

    def test_locked_context_and_original_runtime_source_identity(self):
        self.assertEqual(stream.CONTEXT, 2048)
        self.assertEqual(stream.TOL, 0.0005)
        self.assertEqual(stream.PREFIXES,
                         (1, 2, 16, 64, 128, 256, 512, 1024, 1536, 2048))
        self.assertEqual(stream.gitblob(
            (ROOT / "experiments/ncr-001/protocol.json").read_bytes()),
            stream.PROTOCOL_BLOB)
        self.assertEqual(stream.gitblob(
            (ROOT / "experiments/ncr-001/source-pins.json").read_bytes()),
            stream.PINS_BLOB)
        self.assertEqual(stream.gitblob(
            (ROOT / "scripts/ncr_001_postlease_sprocket.py").read_bytes()),
            stream.SOURCE_POSTLEASE_BLOB)

    def test_no_optimizer_or_heldout_and_honest_every_token_scope(self):
        original = (ROOT / "scripts/ncr_001e_sprocket_alltoken_stream.py").read_text()
        self.assertNotIn("optimizer.step(", original)
        self.assertNotIn(".backward(", original)
        self.assertNotIn("train_dataset", original)
        self.assertIn('"train_authorized": False', original)
        self.assertIn('"full_72_conformance_pass": False', original)
        self.assertIn('"all_prefixes_run_without_cache": False', original)
        self.assertIn('"all_token_streaming_cache_observation_only": True', original)


if __name__ == "__main__":
    unittest.main()
