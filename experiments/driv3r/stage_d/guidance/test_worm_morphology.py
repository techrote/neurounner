"""Hostile and shape fixtures for synthetic, non-native pen intention."""
import unittest
try:
    from .worm_morphology import analyze_validated_trace, compare_validated_traces
except ImportError:
    from worm_morphology import analyze_validated_trace, compare_validated_traces


def move(i, x, y, pen=True):
    return {"decision_index": i, "action": "MOVE", "to_xy": [x, y], "pen_down": pen}


def stop(i):
    return {"decision_index": i, "action": "STOP", "to_xy": [], "pen_down": False}


class WormMorphologyTests(unittest.TestCase):
    def test_linear_ink_one_component_two_endpoints(self):
        r = analyze_validated_trace([move(1, 1, 0), move(2, 2, 0), move(3, 3, 0), stop(4)])
        self.assertEqual((r["ink_components_four_connected"], r["ink_endpoints"],
                          r["ink_cycle_rank"], r["distinct_intended_cells"]), (1, 2, 0, 3))
        self.assertEqual(r["ink_bbox_inclusive_xyxy"], [1, 0, 3, 0])
        self.assertEqual((r["pen_down_runs"], r["longest_pen_down_run"]), (1, 3))
        self.assertFalse(r["native_material_verified"])

    def test_lifted_transfer_does_not_connect_distant_ink(self):
        r = analyze_validated_trace([move(1, 0, 0), move(2, 1, 0, False),
                                     move(3, 2, 0, False), move(4, 3, 0), stop(5)])
        self.assertEqual((r["distinct_intended_cells"], r["ink_components_four_connected"],
                          r["pen_down_runs"], r["ink_isolated_cells"]), (2, 2, 2, 2))

    def test_lifted_then_adjacent_ink_is_one_component_but_two_runs(self):
        r = analyze_validated_trace([move(1, 0, 0), move(2, 1, 0, False),
                                     move(3, 1, 1), stop(4)])
        self.assertEqual((r["ink_components_four_connected"], r["pen_down_runs"]), (2, 2))
        # Distinct cells are diagonal so not four-connected.
        t = analyze_validated_trace([move(1, 1, 0), move(2, 2, 0, False),
                                     move(3, 2, 1, False), move(4, 1, 1), stop(5)])
        self.assertEqual((t["ink_components_four_connected"], t["pen_down_runs"]), (1, 2))

    def test_revisiting_ink_not_new_component(self):
        r = analyze_validated_trace([move(1, 1, 0), move(2, 2, 0),
                                     move(3, 1, 0), stop(4)])
        self.assertEqual((r["intended_contact_moves"], r["distinct_intended_cells"],
                          r["ink_components_four_connected"]), (3, 2, 1))

    def test_square_loop_and_cross_junction(self):
        square = [move(1, 0, 0), move(2, 1, 0), move(3, 1, 1), move(4, 0, 1), stop(5)]
        a = analyze_validated_trace(square)
        self.assertEqual((a["ink_cycle_rank"], a["ink_endpoints"], a["ink_junctions_degree_3plus"]),
                          (1, 0, 0))
        cross = [move(1, 1, 0), move(2, 1, 1), move(3, 0, 1),
                 move(4, 1, 1), move(5, 2, 1), move(6, 1, 1),
                 move(7, 1, 2), stop(8)]
        b = analyze_validated_trace(cross)
        self.assertEqual((b["ink_components_four_connected"], b["ink_junctions_degree_3plus"],
                          b["ink_endpoints"]), (1, 1, 4))

    def test_early_stop_produces_no_ink(self):
        a = analyze_validated_trace([stop(1)])
        self.assertEqual((a["intended_contact_moves"], a["ink_cycle_rank"], a["pen_down_runs"]),
                          (0, 0, 0))
        self.assertIsNone(a["ink_bbox_inclusive_xyxy"])

    def test_divergence_choice_cursor_versus_ink_are_distinct(self):
        free = [move(1, 1, 0, False), move(2, 2, 0, False), move(3, 3, 0), stop(4)]
        soft = [move(1, 0, 1, False), move(2, 0, 2, False), move(3, 0, 3), stop(4)]
        p = compare_validated_traces(free, soft)
        self.assertEqual((p["first_different_control_index"], p["first_different_cursor_choice_index"],
                          p["first_different_pen_down_destination_index"]), (1, 1, 3))
        self.assertFalse(p["model_behavior_verified"])

    def test_same_destination_pen_change_does_not_falsify_cursor_divergence(self):
        free = [move(1, 1, 0, False), stop(2)]
        soft = [move(1, 1, 0, True), stop(2)]
        p = compare_validated_traces(free, soft)
        self.assertEqual(p["first_different_control_index"], 1)
        self.assertIsNone(p["first_different_cursor_choice_index"])
        self.assertEqual(p["first_different_pen_down_destination_index"], 1)

    def test_stop_vs_truncated_arm_is_control_not_cursor_divergence(self):
        # One arm records terminal STOP; the other has the same final cursor
        # but a shorter, explicitly truncated synthetic observation.
        free = [move(1, 1, 0), stop(2)]
        soft = [move(1, 1, 0)]
        for a, b in ((free, soft), (soft, free)):
            with self.subTest(order=(len(a), len(b))):
                result = compare_validated_traces(a, b)
                self.assertEqual(result["first_different_control_index"], 2)
                self.assertIsNone(result["first_different_cursor_choice_index"])
                self.assertIsNone(result["first_different_pen_down_destination_index"])

    def test_stop_vs_actual_extra_move_is_cursor_divergence(self):
        free = [move(1, 1, 0), stop(2)]
        soft = [move(1, 1, 0), move(2, 2, 0)]
        for a, b in ((free, soft), (soft, free)):
            with self.subTest(order=(a[-1]["action"], b[-1]["action"])):
                result = compare_validated_traces(a, b)
                self.assertEqual(result["first_different_control_index"], 2)
                self.assertEqual(result["first_different_cursor_choice_index"], 2)
                self.assertEqual(result["first_different_pen_down_destination_index"], 2)

    def test_malformed_and_fake_boolean_decisions_rejected(self):
        for bad in ([{"decision_index": True, "action": "STOP", "to_xy": [], "pen_down": False}],
                    [move(1, -1, 0)], [dict(move(1, 1, 0), pen_down=1)],
                    [stop(1), move(2, 1, 0)], [dict(stop(1), to_xy=[1, 0])]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                analyze_validated_trace(bad)

    def test_idempotent_canonical_digest_and_no_mutation(self):
        a = [move(1, 1, 0), stop(2)]
        before = repr(a)
        self.assertEqual(analyze_validated_trace(a), analyze_validated_trace(a))
        self.assertEqual(repr(a), before)
        self.assertEqual(len(analyze_validated_trace(a)["ordered_ink_intent_sha256"]), 64)
        self.assertEqual(len(analyze_validated_trace(a)["validated_trace_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
