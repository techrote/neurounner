"""Adversarial tests for a single auditable, non-mutating custody correction."""
import unittest
from contrastive_audit_existing import repaired_alias

GOOD = {"inputs_unchanged": True, "runtime_unchanged": True,
        "harness_unchanged": True, "exit_code": 0, "timed_out": False}


class CustodyAliasTests(unittest.TestCase):
    def test_exact_alias_copy(self):
        original = dict(GOOD)
        repaired = repaired_alias(original)
        self.assertEqual(original, GOOD)
        self.assertNotIn("inputs_unchanged", repaired)
        self.assertIs(repaired["input_unchanged"], True)
        self.assertEqual(len(repaired), len(original))

    def test_missing_false_or_ambiguous_alias_rejected(self):
        for malformed in (
            {}, {**GOOD, "inputs_unchanged": False},
            {**GOOD, "input_unchanged": True},
            {**GOOD, "runtime_unchanged": False},
            {**GOOD, "harness_unchanged": False},
            {**GOOD, "exit_code": 1},
            {**GOOD, "timed_out": True},
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    repaired_alias(malformed)


if __name__ == "__main__":
    unittest.main()
