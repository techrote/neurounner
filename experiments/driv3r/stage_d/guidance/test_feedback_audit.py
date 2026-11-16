"""Adversarial, no-model/no-native fixtures for D-1B feedback accounting."""
import copy
import hashlib
import unittest

from experiments.driv3r.stage_d.guidance.feedback_audit import audit_shadow_cases, seal_case

SCENARIO = hashlib.sha256(b"d1b-local-synthetic-scenario").hexdigest()


def pair(name, decision, before=(10, 0, 3), after=(12, 0, 6),
         choice="KEEP", eligible=True, receipt=True):
    return seal_case({"case_id": name, "decision_index": decision,
                      "scenario_sha256": SCENARIO,
                      "before": {"snapshot_id": name + "_before", "goal_score": before[0],
                                 "violations": before[1], "progress_cells": before[2]},
                      "after": {"snapshot_id": name + "_after", "goal_score": after[0],
                                "violations": after[1], "progress_cells": after[2]},
                      "hypothetical_undo_eligible": eligible,
                      "hypothetical_receipt_available": receipt,
                      "shadow_choice": choice})


class FeedbackAuditTests(unittest.TestCase):
    def test_beneficial_keep_and_regression_undo(self):
        result = audit_shadow_cases([pair("good", 1), pair("bad", 2,
            after=(8, 1, 3), choice="UNDO")])
        self.assertEqual(result["reference_counts"], {"KEEP": 1, "STOP": 0, "UNDO": 1})
        self.assertEqual(result["confusion"]["UNDO"]["UNDO"], 1)
        self.assertEqual(result["hypothetical_undo_proposals"], 1)
        self.assertEqual(result["native_undo_executions"], 0)
        self.assertFalse(result["model_inference_performed"])

    def test_unverified_receipt_is_hypothetical_stop(self):
        result = audit_shadow_cases([pair("unverified", 1, after=(8, 1, 3),
                                          choice="STOP", receipt=False)])
        self.assertEqual(result["reference_counts"]["STOP"], 1)
        self.assertFalse(result["journal"][0]["undo_admissible_if_native_were_qualified"])

    def test_false_undo_and_false_keep_are_not_improvements(self):
        result = audit_shadow_cases([pair("falseundo", 1, choice="UNDO"),
                                     pair("falsekeep", 2, after=(8, 1, 3), choice="KEEP")])
        self.assertEqual(result["errors"]["false_undo"], 1)
        self.assertEqual(result["errors"]["false_keep"], 1)

    def test_proposed_undo_without_receipt_is_flagged_inadmissible(self):
        result = audit_shadow_cases([pair("unsafe", 1, after=(8, 1, 3),
                                          choice="UNDO", receipt=False)])
        self.assertEqual(result["errors"]["inadmissible_undo"], 1)
        self.assertEqual(result["native_material_writes"], 0)

    def test_stop_is_terminal(self):
        with self.assertRaisesRegex(ValueError, "terminal"):
            audit_shadow_cases([pair("stop", 1, choice="STOP"), pair("later", 2)])

    def test_hash_tamper_and_cross_scenario_rejected(self):
        first = pair("first", 1)
        forged = copy.deepcopy(first)
        forged["after"]["goal_score"] = -999
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit_shadow_cases([forged])
        later = pair("other", 2)
        later["scenario_sha256"] = "f" * 64
        later = seal_case({k: v for k, v in later.items() if k != "case_sha256"})
        with self.assertRaisesRegex(ValueError, "unmatched"):
            audit_shadow_cases([first, later])

    def test_repeated_case_id_and_order_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_shadow_cases([pair("a", 1), pair("a", 2)])
        with self.assertRaisesRegex(ValueError, "unordered"):
            audit_shadow_cases([pair("a", 2), pair("b", 1)])

    def test_shadow_undo_budget_is_six_not_native_undo_authorization(self):
        cases = [pair("u" + str(i), i, after=(8, 1, 3), choice="UNDO")
                 for i in range(1, 8)]
        self.assertEqual(audit_shadow_cases(cases[:6])["hypothetical_undo_proposals"], 6)
        with self.assertRaisesRegex(ValueError, "undo proposals exceed"):
            audit_shadow_cases(cases)

    def test_strict_bool_finite_metric_and_choice_validation(self):
        invalid = pair("bad", 1)
        invalid["before"]["goal_score"] = True
        invalid = seal_case({k: v for k, v in invalid.items() if k != "case_sha256"})
        with self.assertRaisesRegex(ValueError, "goal score"):
            audit_shadow_cases([invalid])
        invalid = pair("bad", 1, choice="RETRY")
        with self.assertRaisesRegex(ValueError, "invalid shadow choice"):
            audit_shadow_cases([invalid])

    def test_oscillation_risk_not_material_rollback(self):
        first = pair("a", 1, after=(8, 1, 3), choice="UNDO")
        second = pair("b", 2, after=(8, 1, 3), choice="UNDO")
        # Intentionally identical metric contents with distinct case IDs.
        second["before"] = copy.deepcopy(first["before"])
        second["after"] = copy.deepcopy(first["after"])
        second = seal_case({k: v for k, v in second.items() if k != "case_sha256"})
        result = audit_shadow_cases([first, second])
        self.assertEqual(result["errors"]["oscillation_risk"], 1)
        self.assertEqual(result["native_undo_executions"], 0)


if __name__ == "__main__":
    unittest.main()
