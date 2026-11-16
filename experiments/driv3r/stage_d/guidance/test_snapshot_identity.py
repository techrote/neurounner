"""Cross-case frozen snapshot identity must survive re-sealing and retry IDs.

Synthetic-only fixtures: neither model inference nor real native Undo.
"""
import unittest
from hashlib import sha256

from experiments.driv3r.stage_d.guidance.feedback_audit import audit_shadow_cases, seal_case

SCENARIO = sha256(b"d1b-frozen-snapshot-identity").hexdigest()


def probe(name, index, *, before_id=None, after_id=None,
          before=(10, 1, 3), after=(11, 1, 4), choice="KEEP"):
    return seal_case({
        "case_id": name, "decision_index": index,
        "scenario_sha256": SCENARIO,
        "before": {"snapshot_id": before_id or name + ":before",
                   "goal_score": before[0], "violations": before[1],
                   "progress_cells": before[2]},
        "after": {"snapshot_id": after_id or name + ":after",
                  "goal_score": after[0], "violations": after[1],
                  "progress_cells": after[2]},
        "hypothetical_undo_eligible": True,
        "hypothetical_receipt_available": True,
        "shadow_choice": choice,
    })


class SnapshotIdentityTests(unittest.TestCase):
    def test_reused_after_as_next_before_with_identical_metrics_is_allowed(self):
        first = probe("one", 1)
        second = probe("two", 2, before_id="one:after", before=(11, 1, 4),
                       after=(12, 0, 5))
        audited = audit_shadow_cases([first, second])
        self.assertEqual(audited["forward_decision_indices_seen"], 2)
        self.assertEqual(audited["native_material_writes"], 0)

    def test_same_id_with_changed_goal_is_rejected_even_when_cases_resealed(self):
        first = probe("one", 1)
        second = probe("two", 2, before_id="one:after", before=(12, 1, 4))
        with self.assertRaisesRegex(ValueError, "conflicting synthetic snapshot identity"):
            audit_shadow_cases([first, second])

    def test_same_id_with_changed_violation_or_progress_is_rejected(self):
        first = probe("one", 1)
        for changed in ((11, 2, 4), (11, 1, 5)):
            with self.subTest(changed=changed):
                second = probe("two", 2, before_id="one:after", before=changed)
                with self.assertRaisesRegex(ValueError, "conflicting synthetic snapshot identity"):
                    audit_shadow_cases([first, second])

    def test_revisited_frozen_id_is_consistent_across_nonconsecutive_probes(self):
        first = probe("one", 1)
        middle = probe("middle", 2)
        last = probe("last", 3, after_id="one:before", after=(10, 1, 3))
        audited = audit_shadow_cases([first, middle, last])
        self.assertEqual(audited["forward_decision_indices_seen"], 3)

    def test_distinct_snapshot_ids_may_have_identical_metrics(self):
        a = probe("one", 1, choice="UNDO", before=(10, 2, 4), after=(9, 2, 4))
        b = probe("two", 2, choice="UNDO", before=(10, 2, 4), after=(9, 2, 4))
        audited = audit_shadow_cases([a, b])
        self.assertEqual(audited["errors"]["oscillation_risk"], 1)


if __name__ == "__main__":
    unittest.main()
