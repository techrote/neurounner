"""Pure-CPU hypothetical D-1B fork-controls regression tests."""
import copy
import unittest
from experiments.driv3r.stage_d.guidance.fork_probe import compare_frozen_fork


def sample():
    before = {"goal_score": 10, "violations": 0, "progress_cells": 10}
    return [
        {"id": "safe", "to_xy": [4, 4], "geometry_progress": 1, "guidance_u16": 0,
         "before": before, "after": {"goal_score": 11, "violations": 0, "progress_cells": 11}},
        {"id": "trap", "to_xy": [4, 5], "geometry_progress": 0, "guidance_u16": 60000,
         "before": before, "after": {"goal_score": 9, "violations": 1, "progress_cells": 10}},
    ]


def run(rows=None, **kwargs):
    return compare_frozen_fork(world_sha256="a"*64, guide_sha256="b"*64,
                               legal_candidates=sample() if rows is None else rows, **kwargs)


class ForkProbeTests(unittest.TestCase):
    def test_neutral_and_misleading_guide(self):
        probe = run(undo_eligible=True, hypothetical_receipt_available=True)
        self.assertEqual(probe["selection"]["geometry"]["candidate_id"], "safe")
        self.assertEqual(probe["selection"]["guidance"]["candidate_id"], "trap")
        self.assertEqual(probe["selection"]["critic"]["critic"]["choice"], "KEEP")
        self.assertEqual(probe["selection"]["combined"]["critic"]["choice"], "UNDO")
        self.assertTrue(probe["guide_changes_choice"])
        self.assertEqual(probe["native_material_writes"], 0)
        self.assertEqual(probe["native_undo_executions"], 0)

    def test_no_receipt_proposes_stop_not_real_undo(self):
        probe = run(undo_eligible=True, hypothetical_receipt_available=False)
        self.assertEqual(probe["selection"]["combined"]["critic"]["choice"], "STOP")
        self.assertFalse(probe["selection"]["combined"]["critic"]["executed"])

    def test_zero_gain_and_order_permutation(self):
        a = run(gain=0)
        self.assertFalse(a["guide_changes_choice"])
        b = run(list(reversed(sample())), gain=0)
        self.assertEqual(a, b)

    def test_guide_requires_enough_weight(self):
        self.assertFalse(run(gain=1)["guide_changes_choice"])
        self.assertTrue(run(gain=2)["guide_changes_choice"])

    def test_rejects_illegal_candidate_values(self):
        for key, value in (("guidance_u16", 65536), ("geometry_progress", True),
                           ("to_xy", [1.2, 4]), ("before", {"foo": 1})):
            rows = sample()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run(rows)

    def test_rejects_duplicate_and_unknown_fields(self):
        rows = sample()
        rows[1]["id"] = "safe"
        with self.assertRaises(ValueError):
            run(rows)
        rows = sample()
        rows[1]["protection_write"] = True
        with self.assertRaises(ValueError):
            run(rows)

    def test_rejects_nonshared_before_state(self):
        rows = sample()
        rows[1]["before"] = {"goal_score": 9, "violations": 0, "progress_cells": 9}
        with self.assertRaisesRegex(ValueError, "frozen before-state"):
            run(rows)

    def test_fails_closed_invalid_provenance_or_flags(self):
        with self.assertRaises(ValueError):
            compare_frozen_fork(world_sha256="bad", guide_sha256="b"*64, legal_candidates=sample())
        for flag in (1, "yes", None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                run(hypothetical_receipt_available=flag)

    def test_unrelated_hypothetical_metric_does_not_change_legality(self):
        altered = copy.deepcopy(sample())
        altered[1]["after"]["goal_score"] = 100
        a, b = run(), run(altered)
        self.assertEqual(a["declared_legal_id_target_sha256"], b["declared_legal_id_target_sha256"])
        self.assertNotEqual(a["candidate_effects_sha256"], b["candidate_effects_sha256"])
        self.assertEqual(a["selection"]["guidance"]["candidate_id"],
                         b["selection"]["guidance"]["candidate_id"])


if __name__ == "__main__":
    unittest.main()
