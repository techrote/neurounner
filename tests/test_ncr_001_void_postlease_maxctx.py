"""No-model source/negative regression for post-lease void.1 full-context host."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_postlease_maxctx as long
import ncr_001_void_eos_postlease as original


class PostleaseVoidMaxContextSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "experiments/ncr-001/protocol.json").read_text())
        cls.pins = json.loads(
            (ROOT / "experiments/ncr-001/source-pins.json").read_text())

    def test_locked_four_model_protocol_and_source(self):
        self.assertTrue(long.check_plan(self.protocol))
        self.assertEqual(original.check_contract(self.protocol, self.pins)
                         ["weight_bytes"], 360622744)
        long.check_source_blobs()
        self.assertFalse(self.protocol["train_authorized"])

    def test_pre_registered_numerical_bound_must_not_move(self):
        for key in ("full_vs_cached_prefill_atol",
                    "full_vs_cached_prefill_rtol"):
            bad = copy.deepcopy(self.protocol)
            bad["conformance"][key] = 0.0006
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                long.check_plan(bad)

    def test_fourth_model_and_no_train_lock_cannot_be_bypassed(self):
        bad = copy.deepcopy(self.protocol)
        bad["train_authorized"] = True
        with self.assertRaises(RuntimeError):
            long.check_plan(bad)
        bad = copy.deepcopy(self.protocol)
        del bad["models"]["sprocket_chat"]
        with self.assertRaises(RuntimeError):
            long.check_plan(bad)

    def test_void_native_max_context_cannot_be_redefined(self):
        bad = copy.deepcopy(self.protocol)
        bad["models"]["void1"]["expected_config"]["max_position_embeddings"] = 4096
        with self.assertRaises(RuntimeError):
            long.check_plan(bad)

    def test_exact_physical_probe_is_utf8_but_never_heldout(self):
        self.assertGreater(len(long.SOURCE.encode("utf-8")), 2048)
        self.assertTrue(any(ord(ch) > 127 for ch in long.SOURCE))
        self.assertEqual(long.LENGTHS, (512, 1024, 2048))
        self.assertEqual(long.TAIL, 8)
        self.assertEqual((long.GENERATE_PREFIX, long.GENERATE_UNITS), (512, 4))
        self.assertNotIn("experiments/ncr-001/fixtures/", long.SOURCE)

    def test_causal_eos_reference_rejects_fake_bytes(self):
        good = {
            "input_ids": [120, 60, 101, 111, 115, 62, 258],
            "attention_mask": [1] * 7,
            "labels": [-100, 60, 101, 111, 115, 62, 258],
            "causal_shift_supervised_units": 6,
        }
        self.assertEqual(
            original.validate_causal_eos(good, "x", "<eos>")["real_eos_id"], 258)
        bad = copy.deepcopy(good)
        bad["labels"][-1] = 62
        with self.assertRaises(RuntimeError):
            original.validate_causal_eos(bad, "x", "<eos>")

    def test_diagnostic_has_no_training_api_or_holdout_io(self):
        source = (ROOT / "scripts/ncr_001_void_postlease_maxctx.py").read_text()
        ast.parse(source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("train_authorized\": True", source)
        self.assertNotIn("fixtures/natural-text-holdout", source)
        self.assertNotIn("fixtures/void-byte-probes", source)
        self.assertIn("full_72_conformance_pass", source)
        self.assertIn("original.run()", source)

    def test_generation_scores_must_compare_nonfinite_masks(self):
        source = (ROOT / "scripts/ncr_001_void_postlease_maxctx.py").read_text()
        self.assertIn("torch.isfinite(left)", source)
        self.assertIn("torch.equal(finite, torch.isfinite(right))", source)
        self.assertIn("min_new_tokens=GENERATE_UNITS", source)
        self.assertIn("eos_token_id=258", source)
        self.assertIn("attention_mask=mask[:, :GENERATE_PREFIX]", source)


if __name__ == "__main__":
    unittest.main()
