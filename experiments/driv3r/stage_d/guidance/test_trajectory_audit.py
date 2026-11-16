"""Strict stdlib qualification of lane-only synthetic guidance trajectory metrics."""
import copy
from hashlib import sha256
import struct
import unittest

from experiments.driv3r.stage_d.guidance.trajectory_audit import audit_synthetic_trajectory


FIELD = struct.pack("<6H", 65535, 65535, 200, 0, 50000, 65535)
SOURCE = "a" * 40
SCENARIO = "b" * 64


def step(index, action, path, before, after):
    return {"decision_index": index, "action": action, "path_xy": path,
            "progress_before": before, "progress_after": after}


def audit(steps, **changes):
    params = dict(guidance_u16le=FIELD, width=3, height=2,
                  origin_xy=[4, 7], initial_xy=[4, 7],
                  guidance_sha256=sha256(FIELD).hexdigest(),
                  scenario_sha256=SCENARIO, source_revision=SOURCE,
                  steps=steps, exit_below=300, reenter_at_or_above=50000)
    params.update(changes)
    return audit_synthetic_trajectory(**params)


class TrajectoryAuditTests(unittest.TestCase):
    def test_progressive_unique_preference_contact(self):
        record = audit([step(1, "MOVE", [[5, 7], [6, 7]], 0, 1),
                        step(2, "MOVE", [[6, 8]], 1, 2),
                        step(3, "STOP", [], 2, 2)])
        self.assertEqual(record["productive_new_cells"], 3)
        self.assertEqual(record["productive_guidance_sum_u16"], 65535 + 200 + 65535)
        self.assertEqual(record["synthetic_declared_net_progress"], 2)
        self.assertEqual(record["path_samples"], 3)
        self.assertEqual(record["positive_guidance_cells_in_field"], 5)
        self.assertTrue(record["terminal_stop"])
        self.assertFalse(record["native_execution_verified"])
        self.assertEqual(record["native_undo_executions"], 0)

    def test_stationary_peak_cannot_be_rewarded(self):
        stopped = audit([step(1, "STOP", [], 100, 100)])
        self.assertEqual(stopped["productive_new_cells"], 0)
        self.assertEqual(stopped["productive_guidance_sum_u16"], 0)
        self.assertEqual(stopped["path_samples"], 0)
        with self.assertRaisesRegex(ValueError, "STOP may not claim"):
            audit([step(1, "STOP", [], 100, 101)])

    def test_retrace_never_recredits_unique_cells(self):
        record = audit([step(1, "MOVE", [[5, 7]], 0, 1),
                        step(2, "MOVE", [[4, 7], [5, 7]], 1, 2)])
        self.assertEqual(record["path_samples"], 3)
        self.assertEqual(record["revisited_samples"], 2)
        self.assertEqual(record["productive_new_cells"], 1)
        self.assertEqual(record["productive_guidance_sum_u16"], 65535)

    def test_hysteresis_transition_is_not_a_permission(self):
        record = audit([step(1, "MOVE", [[5, 7], [6, 7]], 0, 1),
                        step(2, "MOVE", [[6, 8], [5, 8]], 1, 2)])
        self.assertEqual(record["hysteresis_events"],
                         {"propose_recovery": 1, "reentered": 1})
        self.assertTrue(record["engaged_at_end"])
        self.assertFalse(record["native_material_authority"])

    def test_negative_progress_separate_from_coverage(self):
        record = audit([step(1, "MOVE", [[5, 7]], 5, 3),
                        step(2, "MOVE", [[6, 7]], 3, 4)])
        self.assertEqual(record["synthetic_declared_net_progress"], -1)
        self.assertEqual(record["synthetic_declared_negative_progress"], 2)
        self.assertEqual(record["productive_new_cells"], 1)
        self.assertEqual(record["productive_guidance_sum_u16"], 200)
        self.assertEqual(record["no_positive_progress_moves"], 1)

    def test_rejects_teleports_or_diagonals(self):
        for path in ([[6, 7]], [[5, 8]], [[4, 7]]):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "non-four-connected"):
                audit([step(1, "MOVE", path, 0, 0)])

    def test_rejects_outside_field_and_wrong_origin(self):
        with self.assertRaisesRegex(ValueError, "outside numeric"):
            audit([step(1, "MOVE", [[3, 7]], 0, 0)])
        with self.assertRaisesRegex(ValueError, "outside numeric"):
            audit([step(1, "STOP", [], 0, 0)], initial_xy=[0, 0])

    def test_rejects_byte_and_sha_tamper(self):
        with self.assertRaisesRegex(ValueError, "bytes/hash"):
            audit([step(1, "STOP", [], 0, 0)], guidance_u16le=FIELD[:-2] + b"\0\0")
        with self.assertRaisesRegex(ValueError, "bytes/hash"):
            audit([step(1, "STOP", [], 0, 0)], guidance_sha256="0" * 64)

    def test_rejects_decision_or_progress_discontinuity(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            audit([step(2, "MOVE", [[5, 7]], 0, 1)])
        with self.assertRaisesRegex(ValueError, "continuity"):
            audit([step(1, "MOVE", [[5, 7]], 0, 1), step(2, "STOP", [], 0, 0)])

    def test_rejects_actions_after_terminal_stop(self):
        with self.assertRaisesRegex(ValueError, "terminal"):
            audit([step(1, "STOP", [], 0, 0), step(2, "MOVE", [[5, 7]], 0, 1)])

    def test_rejects_macro_and_episode_budget_overrun(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            audit([step(1, "MOVE", [[5, 7], [6, 7]], 0, 1)], max_segment_cells=1)
        with self.assertRaisesRegex(ValueError, "invalid bounded decision/macro policy"):
            audit([step(1, "STOP", [], 0, 0)], max_decisions=0)

    def test_rejects_bool_counts_and_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            audit([step(1, "STOP", [], 0, 0)], width=True)
        with self.assertRaisesRegex(ValueError, "invalid synthetic decision keys"):
            audit([{**step(1, "STOP", [], 0, 0), "native_authorised": True}])
        with self.assertRaisesRegex(ValueError, "invalid supplied objective"):
            audit([step(1, "MOVE", [[5, 7]], 0, True)])

    def test_input_digest_and_determinism(self):
        steps = [step(1, "MOVE", [[5, 7], [6, 7]], 0, 1)]
        first = audit(steps)
        self.assertEqual(first, audit(copy.deepcopy(steps)))
        self.assertEqual(len(first["input_trajectory_sha256"]), 64)
        second = audit([step(1, "MOVE", [[5, 7], [6, 7]], 0, 2)])
        self.assertNotEqual(first["input_trajectory_sha256"], second["input_trajectory_sha256"])


if __name__ == "__main__":
    unittest.main()
