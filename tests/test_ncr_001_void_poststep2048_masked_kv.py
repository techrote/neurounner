"""Source-only sentinels for post-update void.1 B=2 masked incremental KV."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_poststep2048_masked_kv as gate


def fake_producer():
    return {
        "source": {"head": "a"*40, "original_source_git_blobs": {}},
        "producer_pid": 111,
        "runner": {"run_id": "123"},
    }


def side(side: str):
    return {
        "direction": side,
        "batch_size": 2,
        "width": 2048,
        "long_valid_units": 2048,
        "short_valid_units": 1536,
        "synthetic_pad_units": 512,
        "tail_real_incremental_units_per_row": 64,
        "genuine_B2_incremental_steps": 64,
        "pad_id": 256,
        "explicit_attention_mask": True,
        "explicit_mask_derived_position_ids": True,
        "right_layout_is_synthetic_mask_gap": side == "right",
        "full_vs_cached": {
            "max_abs": 1e-5, "within_protocol_tolerance": True,
            "atol": 5e-4, "rtol": 5e-4, "exact": False,
        },
        "batch_vs_single_full_max_abs": 1e-5,
        "batch_vs_single_cached_max_abs": 2e-5,
        "row_swap_full": {
            "max_abs": 1e-5, "within_protocol_tolerance": True,
            "atol": 5e-4, "rtol": 5e-4, "exact": False,
        },
        "row_swap_cached": {
            "max_abs": 2e-5, "within_protocol_tolerance": True,
            "atol": 5e-4, "rtol": 5e-4, "exact": False,
        },
        "trace_checkpoints": [
            {"step": 0}, {"step": 31}, {"step": 63},
        ],
    }


def fake_consumer():
    p = fake_producer()
    return {
        "schema": "ncr001-void-poststep2048-masked-kv-v1",
        "status": gate.STATUS,
        "model_repo": gate.base.MODEL,
        "revision": gate.base.REV,
        "weights_sha256": gate.base.WEIGHTS,
        "tokenizer_sha256": gate.base.TOKENIZER,
        "frozen_backbone_sha256": gate.base.FROZEN,
        "source": p["source"],
        "masked_consumer_source": {
            "head": "a"*40, "immutable_dependency_blobs": gate.EXPECTED_BLOBS},
        "producer_pid": 111,
        "consumer_pid": 222,
        "same_runner_second_python": True,
        "poststep_adapter_reload_exact": True,
        "threads_effective_after_load": 4,
        "masked_incremental_kv": {
            "left": side("left"), "right": side("right")},
        "frozen_backbone_unchanged": True,
        "train_authorized": False,
        "comparative_training_performed": False,
        "heldouts_opened": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "actual_compute_budget_qualified": False,
        "generated_eos_skill_qualified": False,
        "historical_train002b_eos_effect_measured": False,
    }


class MaskedIncrementalKVSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "experiments/ncr-001/protocol.json").read_text())

    def test_locked_schedule_and_protocol(self):
        gate.check_plan(self.protocol)
        self.assertEqual(
            (gate.LONG, gate.SHORT, gate.TAIL, gate.BATCH, gate.PAD, gate.EOS),
            (2048, 1536, 64, 2, 256, 258))
        self.assertFalse(self.protocol["train_authorized"])
        self.assertEqual(
            self.protocol["conformance"]["full_vs_cached_prefill_atol"], 5e-4)

    def test_positive_consumer_boundary(self):
        p = fake_producer()
        gate.validate_consumer(fake_consumer(), p)

    def test_both_mask_directions_and_tail_coverage_are_mandatory(self):
        mutations = (
            lambda x: x["masked_incremental_kv"].pop("left"),
            lambda x: x["masked_incremental_kv"]["right"].__setitem__(
                "batch_size", 1),
            lambda x: x["masked_incremental_kv"]["left"].__setitem__(
                "tail_real_incremental_units_per_row", 4),
            lambda x: x["masked_incremental_kv"]["left"].__setitem__(
                "genuine_B2_incremental_steps", 2048),
            lambda x: x["masked_incremental_kv"]["right"].__setitem__(
                "explicit_attention_mask", False),
            lambda x: x["masked_incremental_kv"]["right"].__setitem__(
                "explicit_mask_derived_position_ids", False),
            lambda x: x["masked_incremental_kv"]["right"].__setitem__(
                "right_layout_is_synthetic_mask_gap", False),
        )
        for mutation in mutations:
            row=fake_consumer()
            mutation(row)
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                gate.validate_consumer(row, fake_producer())

    def test_cached_batch_single_and_row_swap_numeric_gates_are_fail_closed(self):
        for path, value in (
            (("left","full_vs_cached","max_abs"), .1),
            (("right","full_vs_cached","within_protocol_tolerance"), False),
            (("left","batch_vs_single_cached_max_abs"), .1),
            (("right","batch_vs_single_full_max_abs"), .1),
            (("left","row_swap_cached","max_abs"), .1),
            (("right","row_swap_cached","within_protocol_tolerance"), False),
        ):
            row=fake_consumer()
            target=row["masked_incremental_kv"][path[0]]
            if len(path)==3:
                target[path[1]][path[2]]=value
            else:
                target[path[1]]=value
            with self.subTest(path=path), self.assertRaisesRegex(
                    RuntimeError,"REFUSED"):
                gate.validate_consumer(row,fake_producer())

    def test_process_source_and_frozen_identity_cannot_be_substituted(self):
        mutations=(
            lambda x: x.__setitem__("model_repo","kandivault/sprocket-500m"),
            lambda x: x.__setitem__("frozen_backbone_sha256","f"*64),
            lambda x: x.__setitem__("consumer_pid",111),
            lambda x: x.__setitem__("threads_effective_after_load",2),
            lambda x: x.__setitem__("same_runner_second_python",False),
            lambda x: x.__setitem__("poststep_adapter_reload_exact",False),
            lambda x: x["masked_consumer_source"].__setitem__("head","b"*40),
        )
        for mutation in mutations:
            row=fake_consumer(); mutation(row)
            with self.assertRaisesRegex(RuntimeError,"REFUSED"):
                gate.validate_consumer(row,fake_producer())

    def test_false_science_training_quality_or_historical_effect_refused(self):
        for key in (
            "train_authorized","comparative_training_performed","heldouts_opened",
            "full_72_conformance_pass","independent_scientific_review",
            "actual_compute_budget_qualified","generated_eos_skill_qualified",
            "historical_train002b_eos_effect_measured",
        ):
            row=fake_consumer(); row[key]=True
            with self.subTest(key=key), self.assertRaisesRegex(
                    RuntimeError,"REFUSED"):
                gate.validate_consumer(row,fake_producer())

    def test_protocol_eos_context_or_tolerance_drift_refused(self):
        mutations=(
            lambda x: x.__setitem__("train_authorized",True),
            lambda x: x["models"]["void1"]["tokenizer_reference"]["ids"]
                .__setitem__("eos",62),
            lambda x: x["models"]["void1"]["expected_config"]
                .__setitem__("max_position_embeddings",4096),
            lambda x: x["conformance"].__setitem__(
                "full_vs_cached_prefill_atol",.001),
        )
        for mutation in mutations:
            p=copy.deepcopy(self.protocol); mutation(p)
            with self.assertRaisesRegex(RuntimeError,"REFUSED"):
                gate.check_plan(p)

    def test_source_uses_explicit_position_ids_genuine_past_and_right_gap_disclosure(self):
        src=(ROOT/"scripts/ncr_001_void_poststep2048_masked_kv.py").read_text()
        ast.parse(src)
        self.assertIn("position_ids=positions",src)
        self.assertIn("past_key_values=past",src)
        self.assertIn("for step, index in enumerate(range(LONG - TAIL, LONG))",src)
        self.assertIn('for side in ("left", "right")',src)
        self.assertIn("right_layout_is_synthetic_mask_gap",src)
        self.assertIn("batch_vs_single_cached_max_abs",src)
        self.assertIn("row_swap_cached",src)
        self.assertNotIn("optimizer.step(",src)
        self.assertNotIn("fixtures/void-byte-probes",src)
        self.assertNotIn("fixtures/natural-text-holdout",src)

    def test_trace_checkpoints_must_cover_first_middle_last_tail_steps(self):
        row=fake_consumer()
        row["masked_incremental_kv"]["left"]["trace_checkpoints"]=[
            {"step":0},{"step":1},{"step":63}]
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            gate.validate_consumer(row,fake_producer())


if __name__=="__main__":
    unittest.main()
