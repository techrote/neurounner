"""No-model independent post-#163 causal-shift EOS review and refusal tests."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import byte_eos_001_postshift_review as review


class PostshiftSourceTests(unittest.TestCase):
    def test_pins_do_not_relabel_original_review_or_source(self):
        self.assertEqual(review.ORIGINAL_REVIEW_HEAD,
                         "54eda2e177d4dbaebcff84cc09a08add1900226c")
        self.assertEqual(review.CORRECTED_COMMIT,
                         "bb60d700fe86cf530773f1f639a246cc24bc0943")
        self.assertNotEqual(review.CORRECTED_BLOB,
                            "724a5040d2fc9eb06ef0b17143d44daa36a4aa89")
        self.assertEqual((review.EOS, review.MAX_UNITS), (258, 1024))

    def test_oracle_supervises_exact_causal_shift_not_unshifted_mask(self):
        x = review.oracle("π", "🔧")
        self.assertEqual(x["input_ids"],
                         list("π🔧".encode("utf-8")) + [258])
        self.assertEqual(x["labels"],
                         [-100, -100] + list("🔧".encode("utf-8")) + [258])
        self.assertEqual(x["causal_shift_supervised_units"], 5)
        self.assertEqual(sum(v != -100 for v in x["labels"][1:]), 5)
        self.assertEqual(x["attention_mask"], [1] * 7)

    def test_oracle_refuses_all_zero_prefixes(self):
        for text in ("", "A", "π", "<eos>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                review.oracle("", text)

    def test_empty_completion_with_prefix_yields_one_real_target(self):
        x = review.oracle("x", "")
        self.assertEqual(x["labels"], [-100, 258])
        self.assertEqual(x["labels"][1:], [258])

    def test_boundary_nonempty_prefix_exact_and_overflow(self):
        a = review.oracle("x", "y" * 1022)
        self.assertEqual(len(a["input_ids"]), 1024)
        self.assertEqual(a["labels"][-1], 258)
        with self.assertRaises(RuntimeError):
            review.oracle("x", "y" * 1023)
        for bad in (True, 0, 1025):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                review.oracle("x", "y", max_units=bad)

    def test_unicode_surrogate_must_raise_not_replace(self):
        with self.assertRaises(UnicodeEncodeError):
            review.oracle("x", "\ud800")

    def test_reviewed_code_passes_new_source_oracle_not_old_acceptance(self):
        review.source_integrity(ROOT / "corrected")
        candidate = review.load_candidate(ROOT / "corrected")
        cases = review.verify(candidate, review.ByteTokenizer())
        self.assertEqual(len(cases), 7)
        self.assertTrue(all(c["oracle_match"] for c in cases))

    def test_adversarial_old_oracle_is_refused(self):
        # Old #126 oracle incorrectly allowed zero prompt. This regression
        # pins its rejection rather than rewriting the original archived ZIP.
        with self.assertRaises(ValueError):
            review.oracle("", "assistant bytes")
        self.assertEqual(review.oracle("x", "assistant bytes")["labels"][0], -100)


if __name__ == "__main__":
    unittest.main()
