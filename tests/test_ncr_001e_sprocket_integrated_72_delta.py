"""NCR-001E source-bound original dual-Sprocket conformance delta tests.

Read exact original PR #203 Git ZIP/log custody and separately re-audited #194
report. All mutated cases are IN-MEMORY adversarial fixtures; neither original
archive nor model weights nor any sealed heldout are modified or executed.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001e_sprocket_integrated_72_delta as gate


class SourceMatchedSprocketDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("NCR_ORIGINAL_DIAGNOSTIC_REPORT")
        if not source:
            raise RuntimeError("Exact-source #194 diagnostic must be re-audited first")
        cls.diag = json.loads(Path(source).read_text(encoding="utf-8"))
        gate.authoritative_source_gate()
        cls.originals = gate.original_masked_receipts()

    def changed(self, action, *, originals=False):
        a = copy.deepcopy(self.diag)
        b = copy.deepcopy(self.originals)
        action(b if originals else a)
        with self.assertRaisesRegex(ValueError, "NCR001_SPROCKET_INTEGRATED_72_REFUSED"):
            gate.compose(a, b)

    def test_two_distinct_original_model_lanes_remain_blocked(self):
        r = gate.compose(self.diag, self.originals)
        self.assertEqual(set(r["lanes"]), {"sprocket_chat", "sprocket_instruct"})
        self.assertNotEqual(r["lanes"]["sprocket_chat"]["frozen_weight_sha256"],
                            r["lanes"]["sprocket_instruct"]["frozen_weight_sha256"])
        self.assertFalse(r["full_72_conformance_pass"])
        self.assertFalse(r["train_authorized"])
        self.assertEqual(r["new_optimizer_steps"], 0)
        self.assertFalse(r["generated_tool_memory_semantics_admitted"])

    def test_true_original_2048_numeric_and_cold_receipts_bound_to_each_variant(self):
        r = gate.compose(self.diag, self.originals)
        for lane in r["lanes"].values():
            observed = lane["observed_bounded"]
            self.assertEqual(observed["masked_batch_2048_physical_positions"], 2)
            self.assertEqual(observed["short_row_true_tokens"], 1536)
            self.assertEqual(observed["actual_one_token_kv_increments"], 4)
            self.assertTrue(observed["cold_new_python_same_runner_last_four_logits_exact"])
            self.assertTrue(all(0 <= x <= gate.CACHE_ATOL for x
                                in observed["masked_left_right_cache_max_abs"].values()))
            self.assertFalse(lane["full_72_per_model_gate"])

    def test_reject_unauthorized_scientific_or_training_claims(self):
        for field in ("train_authorized", "full_72_conformance_pass",
                      "independent_scientific_admission",
                      "comparative_nca_mlp_training", "heldouts_opened"):
            with self.subTest(field=field):
                self.changed(lambda d: d.__setitem__(field, True))

    def test_reject_diagnostic_source_or_run_drift(self):
        self.changed(lambda d: d.__setitem__("physical_source_head", "0"*40))
        self.changed(lambda d: d.__setitem__("physical_source_run", 1))
        self.changed(lambda d: d.__setitem__("original_diagnostic_steps_per_lane", 2))

    def test_reject_chat_root_weight_substitution(self):
        root = gate.SOURCE_MODELS["instruct"][2]
        self.changed(lambda d: d["lanes"]["sprocket_chat"].__setitem__(
            "original_checkpoint_sha256", root))
        self.changed(lambda d: d["lanes"].__setitem__("sprocket_instruct", {}))

    def test_reject_forged_synthetic_learning_or_finite_loss(self):
        self.changed(lambda d: d["lanes"]["sprocket_chat"].__setitem__(
            "nonzero_adapter_gradient_after_warmup", "FAIL"))
        self.changed(lambda d: d["lanes"]["sprocket_instruct"].__setitem__(
            "one_in_memory_nca_only_diagnostic_step", False))
        self.changed(lambda d: d["lanes"]["sprocket_instruct"].__setitem__(
            "postwarmup_synthetic_supervised_loss", float("nan")))

    def test_reject_forged_full_72_promotion_for_either_lane(self):
        self.changed(lambda d: d["lanes"]["sprocket_chat"].__setitem__(
            "full_72_per_model_gate", True))
        self.changed(lambda d: d["lanes"]["sprocket_instruct"].__setitem__(
            "tool_memory_semantics_qualified", True))

    def test_reject_2048_mask_or_cache_mutations(self):
        self.changed(lambda d: d["chat"]["producer"]["checks"].__setitem__(
            "left_mask_corruption_sentinel", False), originals=True)
        self.changed(lambda d: d["instruct"]["producer"]["checks"][
            "right_2048_full_vs_four_real_cached"].__setitem__(
            "max_abs", 0.3), originals=True)

    def test_reject_changed_cold_process_and_original_negative_flags(self):
        self.changed(lambda d: d["chat"]["consumer"]["checks"][
            "left_cold_new_python_same_model_2048"].__setitem__(
            "max_abs", 0.001), originals=True)
        self.changed(lambda d: d["chat"]["consumer"]["runner"].__setitem__(
            "pid", d["chat"]["producer"]["runner"]["pid"]), originals=True)
        self.changed(lambda d: d["instruct"]["producer"].__setitem__(
            "right_pad_tail_is_synthetic_gap_not_publisher_wire", False),
            originals=True)

    def test_reject_truncated_old_lane_history_and_missing_originals(self):
        self.changed(lambda d: d["historical_formal_pretraining_adjudications"][
            "sprocket_chat"].__setitem__("head", "0"*40))
        self.changed(lambda d: d.pop("chat"), originals=True)

    def test_permanent_original_source_decision_is_blob_pinned_and_reproduced(self):
        raw = gate.PUBLISHED_REPORT.read_bytes()
        self.assertEqual(gate.gitblob(raw), gate.PUBLISHED_REPORT_GIT_BLOB)
        committed = json.loads(raw)
        regenerated = gate.compose(self.diag, self.originals)
        self.assertEqual(committed, regenerated)
        self.assertFalse(committed["full_72_conformance_pass"])
        self.assertFalse(committed["train_authorized"])
        self.assertNotEqual(gate.gitblob(raw + b" "), gate.PUBLISHED_REPORT_GIT_BLOB)

    def test_no_model_or_comparative_optimizer_invocation_in_delta(self):
        script = (ROOT / "scripts/ncr_001e_sprocket_integrated_72_delta.py").read_text()
        self.assertNotIn("optimizer.step(", script)
        self.assertNotIn(".backward(", script)
        self.assertIn('"full_72_conformance_pass": False', script)
        self.assertIn('"actual_compute_budget_qualified": False', script)


if __name__ == "__main__":
    unittest.main()
