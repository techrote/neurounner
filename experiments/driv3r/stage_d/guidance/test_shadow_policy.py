"""Metamorphic consistency checks for the two D-1B synthetic shadow consumers.

No native World, inference, material Undo, training or external resources.
"""
from hashlib import sha256
from itertools import product
import unittest

from experiments.driv3r.stage_d.guidance.shadow_policy import POLICY, reference_verdict
from experiments.driv3r.stage_d.guidance.study import CriticSnapshot, shadow_verdict
from experiments.driv3r.stage_d.guidance.feedback_audit import seal_case, audit_shadow_cases

SCENARIO = sha256(b"d1b-shadow-policy-parity-v1").hexdigest()


def compare(before, after, *, eligible=True, receipt=True, shadow_choice="KEEP"):
    b = dict(zip(("goal_score", "violations", "progress_cells"), before))
    a = dict(zip(("goal_score", "violations", "progress_cells"), after))
    direct, reason = reference_verdict(
        b, a, undo_eligible=eligible, verified_native_receipt=receipt)
    verdict = shadow_verdict(
        before=CriticSnapshot("before", *before),
        after=CriticSnapshot("after", *after),
        undo_eligible=eligible, verified_native_receipt=receipt)
    synthetic = seal_case({
        "case_id": "paired", "decision_index": 1, "scenario_sha256": SCENARIO,
        "before": {"snapshot_id": "before", **b},
        "after": {"snapshot_id": "after", **a},
        "hypothetical_undo_eligible": eligible,
        "hypothetical_receipt_available": receipt,
        "shadow_choice": shadow_choice,
    })
    audited = audit_shadow_cases([synthetic])
    assert audited["journal"][0]["reference"] == verdict["decision"] == direct
    assert verdict["reason"] == reason
    assert audited["native_material_writes"] == audited["native_undo_executions"] == 0
    return direct, reason, audited


class ShadowPolicyParityTests(unittest.TestCase):
    def test_violation_reduction_without_score_or_route_progress_is_keep(self):
        choice, reason, audited = compare((10, 3, 5), (10, 2, 5),
                                          shadow_choice="UNDO")
        self.assertEqual((choice, reason), ("KEEP", "nonregressive_progress"))
        self.assertEqual(audited["errors"]["false_undo"], 1)
        self.assertEqual(audited["confusion"]["KEEP"]["UNDO"], 1)

    def test_flat_outcome_and_missing_hypothetical_receipt(self):
        self.assertEqual(compare((10, 3, 5), (10, 3, 5))[0], "UNDO")
        self.assertEqual(compare((10, 3, 5), (10, 3, 5),
                                 receipt=False)[0], "STOP")
        self.assertEqual(compare((10, 3, 5), (10, 3, 5),
                                 eligible=False)[0], "STOP")

    def test_regression_wins_even_if_other_metrics_improve(self):
        self.assertEqual(compare((10, 3, 5), (11, 4, 6))[0], "UNDO")
        self.assertEqual(compare((10, 3, 5), (9, 2, 6))[0], "UNDO")

    def test_all_108_matched_synthetic_metric_and_flag_combinations(self):
        before = (10, 2, 5)
        n = 0
        for goal, violations, progress, eligible, receipt in product(
                (9, 10, 11), (1, 2, 3), (4, 5, 6),
                (False, True), (False, True)):
            with self.subTest(after=(goal, violations, progress),
                              eligible=eligible, receipt=receipt):
                compare(before, (goal, violations, progress),
                        eligible=eligible, receipt=receipt)
            n += 1
        self.assertEqual(n, 108)

    def test_reference_flags_not_coerced(self):
        b = {"goal_score": 10, "violations": 3, "progress_cells": 5}
        for flag in ("false", 0, 1, None):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "boolean"):
                reference_verdict(b, b, undo_eligible=flag,
                                  verified_native_receipt=True)


if __name__ == "__main__":
    unittest.main()
