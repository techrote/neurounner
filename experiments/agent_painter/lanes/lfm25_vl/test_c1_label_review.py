#!/usr/bin/env python3
"""Model-free adversarial tests for C1 review-only training-label preparation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from c1_label_review import LABEL, SOURCE, CUSTODY, TRAINER, review


class C1ReviewLabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.paths = {
            "label": base / "label.json",
            "source": base / "proposals.json",
            "custody": base / "custody.json",
            "trainer": base / "trainer.py",
        }
        for key, original in (("label", LABEL), ("source", SOURCE),
                              ("custody", CUSTODY), ("trainer", TRAINER)):
            self.paths[key].write_bytes(original.read_bytes())

    def modify(self, key, callback):
        data = json.loads(self.paths[key].read_text())
        callback(data)
        self.paths[key].write_text(json.dumps(data, indent=2) + "\n")

    def check(self):
        return review(**self.paths)

    def test_source_bound_label_has_no_model_claim(self):
        result = self.check()
        self.assertEqual(result["status"], "SOURCE_CUSTODY_PASS_OFFICIAL_PROCESSOR_UNMEASURED")
        self.assertEqual(result["global_split"], "NOT_ADMITTED")
        self.assertEqual(result["original_native_job"], "FAILED_POST_RUN_INPUT_CUSTODY_KEY")
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(result["model_generated_views"], 0)
        self.assertFalse(result["final_opened"])

    def test_changed_teacher_program_rejected(self):
        self.modify("source", lambda d: d["proposals"][0]["program"]["components"][0].update(border=2))
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_prompt_rejected(self):
        self.modify("label", lambda d: d["framing"].update(prompt_exact="Generic panels"))
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_system_rejected(self):
        original = self.paths["trainer"].read_text()
        self.paths["trainer"].write_text(original.replace(
            "Follow the requested structure.", "Guess a structure.", 1))
        with self.assertRaises(ValueError):
            self.check()

    def test_false_train_admission_rejected(self):
        self.modify("label", lambda d: d["restrictions"].update(split="TRAIN"))
        with self.assertRaises(ValueError):
            self.check()

    def test_fake_optimizer_or_native_success_rejected(self):
        for key in ("optimizer_updates", "new_native_executions", "model_generated_views"):
            path = self.paths["label"]
            original = path.read_bytes()
            self.modify("label", lambda d, k=key: d["restrictions"].update({k: 1}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
            path.write_bytes(original)

    def test_unfrozen_model_eos_or_token_claim_rejected(self):
        for key, value in (("eos_token_id", 2), ("max_total_tokens", 4096),
                           ("tokenizer_counts_claimed_without_hosted_measurement", True)):
            path = self.paths["label"]
            original = path.read_bytes()
            self.modify("label", lambda d, k=key, v=value: d["framing"].update({k: v}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
            path.write_bytes(original)

    def test_seed_variants_must_remain_one_structure(self):
        self.modify("label", lambda d: d["source"].update(independent_structural_groups=2))
        with self.assertRaises(ValueError):
            self.check()

    def test_failed_native_source_cannot_be_laundered(self):
        self.modify("label", lambda d: d["source"].update(original_native_run_conclusion="success"))
        with self.assertRaises(ValueError):
            self.check()

    def test_duplicate_review_label_keys_refused(self):
        raw = self.paths["label"].read_text()
        self.paths["label"].write_text(raw.replace(
            '"schema": "cybersand.lfm25_vl.c1-label-framing-review/1",',
            '"schema": "cybersand.lfm25_vl.c1-label-framing-review/1", "schema": "forged",', 1))
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main()
