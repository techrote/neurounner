"""Source-only sentinels for post-update void.1 2048 physical diagnostic."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_poststep2048 as gate


def fake_producer():
    grads = {k: 1.0 for k in (
        "readout.weight", "observe.weight", "perception.weight",
        "rule.weight", "input_norm.weight")}
    return {
        "schema": "ncr001-void-poststep2048-v1",
        "status": gate.PRODUCER_STATUS, "phase": "producer",
        "model_repo": gate.MODEL, "revision": gate.REV,
        "weights_sha256": gate.WEIGHTS, "tokenizer_sha256": gate.TOKENIZER,
        "frozen_backbone_sha256": gate.FROZEN,
        "diagnostic_optimizer_steps": 1, "train_authorized": False,
        "comparative_training_performed": False, "heldouts_opened": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "threads_effective_after_load": 4, "producer_pid": 123,
        "adapter_sha256": "a"*64, "adapter_metadata_sha256": "b"*64,
        "short_active_logits_sha256": "c"*64,
        "short_active_vs_disabled_max_abs": 0.25,
        "source": {"head": "d"*40, "immutable_dependency_blobs": gate.EXPECTED_BLOBS},
        "runner": {"run_id": "12345"},
        "public_learning_sample": {
            "sample_kind": "synthetic_public_diagnostic_only", "heldout": False,
            "total_units": 31, "supervised_after_shift": 21,
            "real_eos_id": 258, "ids_sha256": "1"*64,
            "labels_sha256": "2"*64, "mask_sha256": "3"*64,
        },
        "learning_summary": {
            "diagnostic_optimizer_steps": 1,
            "checks": {
                "native_shifted_byte_eos_loss": True,
                "initial_zero_adapter_parity": True,
                "readout_gradient_first_backward": True,
                "post_step_upstream_gradients": True,
                "three_distinct_depth_states": True,
                "backbone_unchanged_after_one_step": True,
                "disabled_source_logits_unchanged": True,
                "no_cross_model_comparison": True,
            },
            "initial_loss": 3.0, "post_warmup_loss": 2.0,
            "post_warmup_required_gradient_l1": grads,
            "backbone_before_sha256": gate.FROZEN,
            "backbone_after_sha256": gate.FROZEN,
        },
    }


def fake_consumer():
    p = fake_producer()
    masked = {}
    for side in ("left", "right"):
        masked[side] = {
            "direction": side, "batch_size": 2, "width": 2048,
            "short_valid_units": 1536, "synthetic_pad_units": 512,
            "pad_id": 256, "explicit_attention_mask": True,
            "batch_vs_identical_single_within_tolerance": True,
            "max_batch_vs_single_abs": 1e-5,
            "row_order_swap_within_tolerance": True,
            "max_row_swap_abs": 2e-5,
        }
    return {
        "schema": "ncr001-void-poststep2048-v1",
        "status": gate.CONSUMER_STATUS, "phase": "consumer",
        "source": p["source"], "model_repo": gate.MODEL, "revision": gate.REV,
        "weights_sha256": gate.WEIGHTS, "tokenizer_sha256": gate.TOKENIZER,
        "frozen_backbone_sha256": gate.FROZEN,
        "producer_pid": 123, "consumer_pid": 456,
        "threads_effective_after_load": 4, "same_runner_second_python": True,
        "poststep_adapter_reload_exact": True,
        "poststep_2048_all_position_cache": {
            "context_units": 2048, "incremental_positions": 2048,
            "max_abs": 1e-5, "within_protocol_tolerance": True,
            "all_positions_compared": True,
            "full_logits_sha256": "4"*64, "incremental_logits_sha256": "5"*64,
        },
        "poststep_masked_batch": masked,
        "poststep_2048_suffix": {
            "context_units": 2048, "mutated_suffix_units": 64,
            "earlier_logits_exact": True, "earlier_max_abs": 0.0,
        },
        "poststep_native_generation": {
            "prefix_units": 2044, "new_units": 4, "total_units": 2048,
            "eos_id": 258, "cache_on_off_token_ids_equal": True,
            "cache_on_off_scores_within_tolerance": True,
            "max_score_abs": 1e-5, "generated_ids": [101,110,32,109],
        },
        "frozen_backbone_unchanged_after_long_diagnostic": True,
        "train_authorized": False, "comparative_training_performed": False,
        "heldouts_opened": False, "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "actual_compute_budget_qualified": False,
        "generated_eos_skill_qualified": False,
        "historical_train002b_eos_effect_measured": False,
    }


class Poststep2048SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "experiments/ncr-001/protocol.json").read_text())

    def test_exact_authorized_schedule_and_no_train_protocol(self):
        gate.check_plan(self.protocol)
        self.assertEqual(gate.CONTEXT, gate.INCREMENTAL_POSITIONS)
        self.assertEqual(gate.CONTEXT, 2048)
        self.assertEqual((gate.BATCH_SIZE, gate.BATCH_SHORT_UNITS), (2,1536))
        self.assertEqual((gate.GENERATE_PREFIX, gate.GENERATE_UNITS), (2044,4))
        self.assertEqual(gate.GENERATE_PREFIX + gate.GENERATE_UNITS, 2048)
        self.assertEqual(gate.SUFFIX_MUTATION_UNITS, 64)
        self.assertEqual((gate.PAD_ID, gate.EOS_ID), (256,258))
        self.assertFalse(self.protocol["train_authorized"])

    def test_source_is_public_synthetic_and_long_enough(self):
        raw = gate.LONG_SOURCE.encode("utf-8")
        self.assertGreaterEqual(len(raw), gate.CONTEXT + gate.BATCH_SHORT_UNITS)
        self.assertTrue(any(b >= 128 for b in raw))
        self.assertNotIn("fixtures/", gate.LONG_SOURCE)

    def test_producer_and_consumer_positive_boundary(self):
        p = fake_producer()
        c = fake_consumer()
        gate.validate_producer(p)
        gate.validate_consumer(c, p)

    def test_producer_false_training_or_mechanics_refused(self):
        mutations = (
            lambda x: x.__setitem__("diagnostic_optimizer_steps", True),
            lambda x: x.__setitem__("diagnostic_optimizer_steps", 2),
            lambda x: x.__setitem__("train_authorized", True),
            lambda x: x.__setitem__("comparative_training_performed", True),
            lambda x: x.__setitem__("heldouts_opened", True),
            lambda x: x.__setitem__("full_72_conformance_pass", True),
            lambda x: x["public_learning_sample"].__setitem__("real_eos_id", 62),
            lambda x: x["public_learning_sample"].__setitem__("total_units", 2048),
            lambda x: x["learning_summary"].__setitem__("diagnostic_optimizer_steps", True),
            lambda x: x["learning_summary"].__setitem__("post_warmup_loss", math.inf),
            lambda x: x["learning_summary"]["post_warmup_required_gradient_l1"].__setitem__(
                "rule.weight", 0.0),
            lambda x: x.__setitem__("frozen_backbone_sha256", "f"*64),
        )
        for mutate in mutations:
            row = copy.deepcopy(fake_producer())
            mutate(row)
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                gate.validate_producer(row)

    def test_2048_cache_cannot_be_relabelled_from_short_tail(self):
        for field, value in (
            ("context_units", 31), ("incremental_positions", 4),
            ("incremental_positions", 2047), ("max_abs", 0.1),
            ("all_positions_compared", False),
        ):
            c = fake_consumer()
            c["poststep_2048_all_position_cache"][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    RuntimeError, "REFUSED"):
                gate.validate_consumer(c, fake_producer())

    def test_masked_batch_requires_b2_both_directions_explicit_mask(self):
        for mutation in (
            lambda x: x["poststep_masked_batch"].pop("left"),
            lambda x: x["poststep_masked_batch"]["right"].__setitem__("batch_size", 1),
            lambda x: x["poststep_masked_batch"]["left"].__setitem__(
                "explicit_attention_mask", False),
            lambda x: x["poststep_masked_batch"]["left"].__setitem__(
                "short_valid_units", 2048),
            lambda x: x["poststep_masked_batch"]["right"].__setitem__(
                "max_batch_vs_single_abs", 0.1),
            lambda x: x["poststep_masked_batch"]["left"].__setitem__(
                "row_order_swap_within_tolerance", False),
        ):
            c = fake_consumer()
            mutation(c)
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                gate.validate_consumer(c, fake_producer())

    def test_suffix_and_native_generation_cannot_be_downgraded(self):
        variants = (
            ("suffix_units", lambda x: x["poststep_2048_suffix"].__setitem__(
                "mutated_suffix_units", 1)),
            ("suffix_exact", lambda x: x["poststep_2048_suffix"].__setitem__(
                "earlier_logits_exact", False)),
            ("generate_prefix", lambda x: x["poststep_native_generation"].__setitem__(
                "prefix_units", 512)),
            ("generate_total", lambda x: x["poststep_native_generation"].__setitem__(
                "total_units", 2049)),
            ("cache_ids", lambda x: x["poststep_native_generation"].__setitem__(
                "cache_on_off_token_ids_equal", False)),
            ("score", lambda x: x["poststep_native_generation"].__setitem__(
                "max_score_abs", .5)),
            ("eos", lambda x: x["poststep_native_generation"].__setitem__(
                "eos_id", 62)),
        )
        for name, mutation in variants:
            c = fake_consumer()
            mutation(c)
            with self.subTest(name=name), self.assertRaisesRegex(
                    RuntimeError, "REFUSED"):
                gate.validate_consumer(c, fake_producer())

    def test_scientific_admission_and_historical_effect_remain_false(self):
        for key in (
            "train_authorized", "comparative_training_performed",
            "heldouts_opened", "full_72_conformance_pass",
            "independent_scientific_review", "actual_compute_budget_qualified",
            "generated_eos_skill_qualified", "historical_train002b_eos_effect_measured",
        ):
            c = fake_consumer()
            c[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(
                    RuntimeError, "REFUSED"):
                gate.validate_consumer(c, fake_producer())

    def test_source_contains_real_all_position_stream_and_no_heldout_or_mlp_optimizer(self):
        source = (ROOT / "scripts/ncr_001_void_poststep2048.py").read_text()
        ast.parse(source)
        self.assertIn("for pos in range(CONTEXT)", source)
        self.assertIn('"incremental_positions": INCREMENTAL_POSITIONS', source)
        self.assertIn('for side in ("left", "right")', source)
        self.assertIn("SUFFIX_MUTATION_UNITS", source)
        self.assertIn("wrapper.generate", source)
        self.assertIn("learning.one_step_diagnostic", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("mlp.parameters()", source)
        self.assertNotIn("fixtures/natural-text-holdout", source)
        self.assertNotIn("fixtures/void-byte-probes", source)

    def test_protocol_or_tolerance_drift_refused(self):
        for mutation in (
            lambda x: x.__setitem__("train_authorized", True),
            lambda x: x["models"]["void1"]["tokenizer_reference"]["ids"].__setitem__(
                "eos", 62),
            lambda x: x["models"]["void1"]["expected_config"].__setitem__(
                "max_position_embeddings", 4096),
            lambda x: x["conformance"].__setitem__(
                "full_vs_cached_prefill_atol", .001),
        ):
            p = copy.deepcopy(self.protocol)
            mutation(p)
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                gate.check_plan(p)


if __name__ == "__main__":
    unittest.main()
