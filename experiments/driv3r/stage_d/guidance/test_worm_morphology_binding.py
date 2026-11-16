"""Offline-only, integrated custody fixtures for line-worm ink intentions."""
import hashlib
import struct
import unittest
from copy import deepcopy

try:
    from .worm_morphology_binding import bind_synthetic_worm_morphology
except ImportError:
    from experiments.driv3r.stage_d.guidance.worm_morphology_binding import bind_synthetic_worm_morphology


def move(i, x, y, pen=True):
    return {"decision_index": i, "action": "MOVE",
            "to_xy": [x, y], "pen_down": pen}


def stop(i):
    return {"decision_index": i, "action": "STOP",
            "to_xy": [], "pen_down": False}


class SourceBoundWormMorphologyTests(unittest.TestCase):
    def setUp(self):
        raw = struct.pack("<12H", *([0, 0, 1000, 65535] * 3))
        self.kw = dict(
            guidance_u16le=raw, width=4, height=3, origin_xy=[0, 0],
            guidance_sha256=hashlib.sha256(raw).hexdigest(),
            initial_xy=[1, 1], initial_world_sha256="a" * 64,
            scenario_sha256="b" * 64, source_revision="c" * 40,
            seed=570, max_decisions=8,
            free_trace=[move(1, 0, 1), move(2, 0, 0), stop(3)],
            soft_trace=[move(1, 2, 1), move(2, 3, 1), stop(3)],
        )

    def audit(self, **updates):
        args = deepcopy(self.kw)
        args.update(updates)
        return bind_synthetic_worm_morphology(**args)

    def test_source_trajectory_and_intent_join(self):
        report = self.audit()
        self.assertEqual(report["protocol"],
                         "cybersand.driv3r.d1b-worm-morphology-bound/1")
        self.assertEqual(report["source_revision"], "c" * 40)
        self.assertEqual(report["initial_world_sha256"], "a" * 64)
        for arm in ("free", "soft"):
            movement = report["movement_audit"]["arms"][arm]
            ink = report["morphology_audit"]["arms"][arm]
            self.assertEqual(movement["trajectory_sha256"], ink["validated_trace_sha256"])
            self.assertEqual(movement["pen_down_intent_moves"], ink["intended_contact_moves"])
            self.assertEqual(movement["pen_down_intent_reference"]["unique_target_cells"],
                             ink["distinct_intended_cells"])
        self.assertFalse(report["native_material_verified"])
        self.assertFalse(report["model_behavior_verified"])
        self.assertEqual(report["native_undo_executions"], 0)

    def test_byte_stable_repeat_and_input_unmodified(self):
        original = deepcopy(self.kw)
        self.assertEqual(self.audit(), self.audit())
        self.assertEqual(self.kw, original)
        self.assertEqual(len(self.audit()["morphology_audit_sha256"]), 64)
        self.assertEqual(len(self.audit()["movement_audit_sha256"]), 64)

    def test_first_move_bound_to_actual_shared_initial_cursor(self):
        with self.assertRaisesRegex(ValueError, "teleport"):
            self.audit(soft_trace=[move(1, 3, 1), stop(2)])

    def test_mid_trace_diagonal_or_skip_rejected_before_morphology(self):
        for trace in ([move(1, 2, 1), move(2, 3, 2), stop(3)],
                      [move(1, 2, 1, False), move(2, 3, 1), move(3, 0, 1), stop(4)]):
            with self.subTest(trace=trace), self.assertRaisesRegex(ValueError, "teleport"):
                self.audit(soft_trace=trace)

    def test_lifted_cursor_changes_topology_only_at_pen_contact(self):
        report = self.audit(
            soft_trace=[move(1, 2, 1, False), move(2, 3, 1, False), stop(3)])
        soft = report["morphology_audit"]["arms"]["soft"]
        self.assertEqual(soft["intended_contact_moves"], 0)
        self.assertEqual(soft["ink_components_four_connected"], 0)
        self.assertEqual(report["movement_audit"]["arms"]["soft"]["movement_cells"], 2)

    def test_stop_first_yields_no_intent_and_not_fake_completion(self):
        report = self.audit(soft_trace=[stop(1)])
        self.assertEqual(report["morphology_audit"]["arms"]["soft"]["ink_bbox_inclusive_xyxy"], None)
        self.assertTrue(report["movement_audit"]["arms"]["soft"]["terminal_stop"])
        self.assertFalse(report["movement_audit"]["matched_movement_counts"])

    def test_refuse_fake_field_hash_and_source_or_snapshot_identity(self):
        for bad in (dict(guidance_sha256="0" * 64),
                    dict(source_revision="unfrozen"),
                    dict(initial_world_sha256="invalid"),
                    dict(scenario_sha256="invalid"),
                    dict(seed=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.audit(**bad)

    def test_pen_control_type_and_nonterminal_stop_rejected(self):
        for trace in ([dict(move(1, 2, 1), pen_down="false"), stop(2)],
                      [stop(1), move(2, 2, 1)]):
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                self.audit(soft_trace=trace)

    def test_repeated_pen_down_cell_remains_distinct_contact(self):
        report = self.audit(
            soft_trace=[move(1, 2, 1), move(2, 1, 1),
                        move(3, 2, 1), stop(4)])
        soft = report["morphology_audit"]["arms"]["soft"]
        self.assertEqual((soft["intended_contact_moves"],
                          soft["distinct_intended_cells"],
                          soft["ink_components_four_connected"]), (3, 2, 1))


if __name__ == "__main__":
    unittest.main()
