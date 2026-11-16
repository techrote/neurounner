#!/usr/bin/env python3
"""Positive original-byte and adversarial SPROCKET G09 custody checks."""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g09_offer_custody import verify_offer, read_json, program_for

ORIGINAL = Path(os.environ["G09_ORIGINAL_NATIVE_DIR"])
OFFER = read_json(HERE / "G09_SHARED_ADMISSION_REVIEW_OFFER.json")

class G09OfferTests(unittest.TestCase):
    def test_original_archive_passes_and_remains_research_only(self):
        v = verify_offer(OFFER, ORIGINAL)
        self.assertEqual(v["native_seed_views_verified"], 2)
        self.assertEqual(v["new_train_groups"], 0)
        self.assertFalse(v["exact_new_request_and_target"])

    def test_wrong_program_digest_rejected(self):
        p = deepcopy(OFFER)
        p["authored_program"]["program_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "program SHA-256"):
            verify_offer(p, ORIGINAL)

    def test_source_program_change_rejected(self):
        changed = program_for("G09")
        changed["components"][0]["density"] = 1
        with patch("g09_offer_custody.program_for", return_value=changed):
            with self.assertRaisesRegex(ValueError, "program SHA-256"):
                verify_offer(OFFER, ORIGINAL)

    def test_missing_native_case_rejected(self):
        p = deepcopy(OFFER)
        p["authored_program"]["exact_native_cases"].pop()
        with self.assertRaisesRegex(ValueError, "two original"):
            verify_offer(p, ORIGINAL)

    def test_duplicate_seed_rejected(self):
        p = deepcopy(OFFER)
        p["authored_program"]["exact_native_cases"][1]["seed"] = 104729
        with self.assertRaisesRegex(ValueError, "two original"):
            verify_offer(p, ORIGINAL)

    def test_changed_native_case_hash_rejected(self):
        p = deepcopy(OFFER)
        p["authored_program"]["exact_native_cases"][0]["original_png_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_offer(p, ORIGINAL)

    def test_invented_request_rejected(self):
        p = deepcopy(OFFER)
        p["proposed_training_label_gate"]["request"] = "Make machine bands"
        with self.assertRaisesRegex(ValueError, "target/EOS promotion"):
            verify_offer(p, ORIGINAL)

    def test_invented_completion_rejected(self):
        p = deepcopy(OFFER)
        p["proposed_training_label_gate"]["complete_target_json"] = "{}"
        with self.assertRaisesRegex(ValueError, "target/EOS promotion"):
            verify_offer(p, ORIGINAL)

    def test_false_label_eos_or_train_admission_rejected(self):
        p = deepcopy(OFFER)
        p["proposed_training_label_gate"]["eos_appended_and_verified"] = True
        with self.assertRaisesRegex(ValueError, "target/EOS promotion"):
            verify_offer(p, ORIGINAL)
        p = deepcopy(OFFER)
        p["safety_and_next_gate"]["newly_admitted_train_groups"] = 1
        with self.assertRaisesRegex(ValueError, "safety gate"):
            verify_offer(p, ORIGINAL)

    def test_corrupted_original_receipt_rejected(self):
        cid = "sprocket-G09-s104729"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fn in ("qualification-summary.json","native-verification.json","candidates.json"):
                (root / fn).write_bytes((ORIGINAL / fn).read_bytes())
            for seed in (104729,130363):
                name = f"sprocket-G09-s{seed}"
                for ext in (".png",".materials.i32le",".receipt.json"):
                    (root / (name + ext)).write_bytes((ORIGINAL / (name + ext)).read_bytes())
            f = root / (cid + ".receipt.json")
            altered = json.loads(f.read_text())
            altered["program_sha256"] = "f" * 64
            f.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "receipt is not positive"):
                verify_offer(OFFER, root)

if __name__ == "__main__":
    unittest.main()
