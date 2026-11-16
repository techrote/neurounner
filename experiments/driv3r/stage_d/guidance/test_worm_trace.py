"""Pure stdlib tests for hypothetical line-worm guidance-response custody."""
import hashlib
import struct
import unittest
from copy import deepcopy

try:
    from .worm_trace import audit_synthetic_worm_pair
except ImportError:
    from worm_trace import audit_synthetic_worm_pair


def move(i, x, y, pen=True):
    return {"decision_index": i, "action": "MOVE", "to_xy": [x, y], "pen_down": pen}


def stop(i):
    return {"decision_index": i, "action": "STOP", "to_xy": [], "pen_down": False}


class WormTraceTests(unittest.TestCase):
    def setUp(self):
        raw = struct.pack("<12H", *([0, 0, 1000, 65535] * 3))
        self.kw = dict(guidance_u16le=raw, width=4, height=3,
                       origin_xy=[0, 0], guidance_sha256=hashlib.sha256(raw).hexdigest(),
                       initial_xy=[1, 1], initial_world_sha256="a" * 64,
                       scenario_sha256="b" * 64, source_revision="c" * 40,
                       seed=570, max_decisions=8,
                       free_trace=[move(1, 0, 1), move(2, 0, 0), stop(3)],
                       soft_trace=[move(1, 2, 1), move(2, 3, 1), stop(3)])

    def run_audit(self, **updates):
        args = deepcopy(self.kw)
        args.update(updates)
        return audit_synthetic_worm_pair(**args)

    def test_posthoc_preference_and_zero_revealed_field_for_free(self):
        r = self.run_audit()
        self.assertGreater(r["posthoc_soft_minus_free_mean_reference_guidance_u16"], 0)
        self.assertEqual(r["arm_visible_guidance_sha256"]["free"], hashlib.sha256(bytes(24)).hexdigest())
        self.assertEqual(r["arm_visible_guidance_sha256"]["soft"], self.kw["guidance_sha256"])
        self.assertFalse(r["model_inference_performed"])
        self.assertFalse(r["native_receipts_verified"])
        self.assertEqual(r["native_undo_executions"], 0)

    def test_fixed_input_is_reproducible_and_copied(self):
        old = deepcopy(self.kw)
        self.assertEqual(self.run_audit(), self.run_audit())
        self.assertEqual(old, self.kw)

    def test_turn_direction_reversal_and_pen_up_are_descriptive_only(self):
        trace = [move(1, 2, 1), move(2, 2, 2, False), move(3, 2, 1), stop(4)]
        a = self.run_audit(free_trace=trace)["arms"]["free"]
        self.assertEqual((a["right_turns"], a["reversals"]), (1, 1))
        self.assertEqual(a["revisits"], 1)
        self.assertEqual(a["brush_lifted_moves"], 1)
        self.assertEqual(a["pen_down_intent_moves"], 2)

    def test_early_stop_retained_not_padding_for_budget(self):
        a = self.run_audit(soft_trace=[stop(1)])["arms"]["soft"]
        self.assertTrue(a["terminal_stop"])
        self.assertEqual(a["movement_cells"], 0)
        self.assertIsNone(a["mean_reference_guidance_all_moves_u16"])
        self.assertIsNone(self.run_audit(soft_trace=[stop(1)])["posthoc_soft_minus_free_mean_reference_guidance_u16"])

    def test_malformed_world_hash_field_and_scalar_rejected(self):
        for override in (dict(initial_world_sha256="x" * 64),
                         dict(scenario_sha256=""),
                         dict(guidance_u16le=b"\0" * 24),
                         dict(width=True),
                         dict(origin_xy=[2**31 - 1, 0]),
                         dict(seed=True)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.run_audit(**override)

    def test_teleports_oob_and_malformed_records_rejected(self):
        cases = [[move(1, 3, 1)], [move(1, 1, 0), move(2, 1, -1)],
                 [dict(move(1, 2, 1), pen_down=1)],
                 [dict(move(1, 2, 1), decision_index=True)],
                 [dict(move(1, 2, 1), extra="critic_keep")],
                 [stop(1), move(2, 2, 1)], [move(1, 2, 1), stop(3)],
                 [dict(stop(1), to_xy=[1, 1])]]
        for trace in cases:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                self.run_audit(soft_trace=trace)

    def test_same_path_arms_report_no_difference(self):
        r = self.run_audit(soft_trace=deepcopy(self.kw["free_trace"]))
        self.assertEqual(r["posthoc_soft_minus_free_mean_reference_guidance_u16"], 0)
        self.assertEqual(r["arms"]["free"], r["arms"]["soft"])

    def test_truncated_trace_not_silently_accepted_as_stopped(self):
        a = self.run_audit(soft_trace=[move(1, 2, 1)])["arms"]["soft"]
        self.assertFalse(a["terminal_stop"])
        self.assertTrue(a["truncated_before_budget_without_stop"])

    def test_lifted_motion_contact_does_not_imply_paint_intent_contact(self):
        lifted = [move(1, 2, 1, False), move(2, 3, 1, False), stop(3)]
        r = self.run_audit(soft_trace=lifted)
        a = r["arms"]["soft"]
        self.assertGreater(a["mean_reference_guidance_all_moves_u16"], 0)
        self.assertEqual(a["high_reference_contact_moves"], 1)
        self.assertEqual(a["pen_down_intent_reference"]["unique_target_cells"], 0)
        self.assertIsNone(a["pen_down_intent_reference"]["mean_all_pen_down_moves_u16"])
        self.assertIsNone(r["posthoc_soft_minus_free_mean_reference_pen_down_intent_u16"])
        self.assertFalse(r["matched_pen_down_intent_counts"])

    def test_pen_down_unique_cells_are_not_cursor_unique_cells(self):
        trace = [move(1, 2, 1, False), move(2, 3, 1),
                 move(3, 2, 1), move(4, 3, 1), stop(5)]
        a = self.run_audit(soft_trace=trace)["arms"]["soft"]
        p = a["pen_down_intent_reference"]
        self.assertEqual(a["movement_cells"], 4)
        self.assertEqual(a["pen_down_intent_moves"], 3)
        self.assertEqual(p["unique_target_cells"], 2)
        self.assertEqual(p["revisited_target_moves"], 1)
        self.assertEqual(p["high_contact_moves"], 2)
        self.assertEqual(p["high_contact_unique_cells"], 1)
        self.assertEqual(p["mean_all_pen_down_moves_u16"], (65535 + 1000 + 65535) / 3)
        self.assertEqual(p["mean_unique_pen_down_target_cells_u16"], (65535 + 1000) / 2)

    def test_first_pen_contact_after_lift_counts_even_if_cursor_revisits(self):
        trace = [move(1, 2, 1, False), move(2, 1, 1), stop(3)]
        a = self.run_audit(free_trace=trace)["arms"]["free"]
        self.assertEqual(a["revisits"], 1)
        self.assertEqual(a["pen_down_intent_reference"]["unique_target_cells"], 1)
        self.assertEqual(a["pen_down_intent_reference"]["revisited_target_moves"], 0)

    def test_paired_ink_preference_and_cardinality_are_explicit(self):
        r = self.run_audit()
        self.assertEqual(r["protocol"], "cybersand.driv3r.d1b-worm-trace-audit/2")
        self.assertTrue(r["matched_movement_counts"])
        self.assertTrue(r["matched_pen_down_intent_counts"])
        self.assertEqual(
            r["posthoc_soft_minus_free_mean_reference_pen_down_intent_u16"],
            r["posthoc_soft_minus_free_mean_reference_guidance_u16"])


if __name__ == "__main__":
    unittest.main()
