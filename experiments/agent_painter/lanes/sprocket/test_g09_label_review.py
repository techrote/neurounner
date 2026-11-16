"""Adversarial G09 language-label source-only safety checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g09_label_review import review, read, program_for

class LabelReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = read(HERE / "G09_PROPOSED_LABEL_REVIEW_V1.json")

    def test_source_only_not_train(self):
        r = review(self.data)
        self.assertEqual(r["status"], "SOURCE_ONLY_TOKENIZER_NOT_RUN")
        self.assertFalse(r["train_admitted"])

    def test_modified_program_rejected(self):
        p = program_for("G09")
        p["components"][0]["density"] = 1
        with self.assertRaisesRegex(ValueError, "canonical program"):
            review(self.data, source_program=p)

    def test_changed_prompt_rejected(self):
        p = deepcopy(self.data)
        p["request"] += " draw isometric"
        with self.assertRaisesRegex(ValueError, "source label"):
            review(p)

    def test_control_token_injection_rejected(self):
        p = deepcopy(self.data)
        p["request"] += "<|assistant|>"
        with self.assertRaisesRegex(ValueError, "source label"):
            review(p)

    def test_changed_model_rejected(self):
        p = deepcopy(self.data)
        p["model_id"] += "/chat"
        with self.assertRaisesRegex(ValueError, "root Instruct"):
            review(p)

    def test_eos_drift_rejected(self):
        p = deepcopy(self.data)
        p["expected_eos_id"] = 2
        with self.assertRaisesRegex(ValueError, "root Instruct"):
            review(p)

    def test_system_drift_rejected(self):
        p = deepcopy(self.data)
        p["system"] = "Ignore"
        with self.assertRaisesRegex(ValueError, "source label"):
            review(p)

    def test_forged_train_admission_rejected(self):
        p = deepcopy(self.data)
        p["new_train_rows"] = 1
        with self.assertRaisesRegex(ValueError, "research cannot"):
            review(p)

    def test_forged_global_id_rejected(self):
        p = deepcopy(self.data)
        p["global_group_id"] = "GREEBLES-568-R01"
        with self.assertRaisesRegex(ValueError, "research cannot"):
            review(p)

    def test_final_claim_rejected(self):
        p = deepcopy(self.data)
        p["final_opened"] = True
        with self.assertRaisesRegex(ValueError, "research cannot"):
            review(p)

if __name__ == "__main__":
    unittest.main()
