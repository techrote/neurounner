"""Adversarial source-only A02 v5 nomination gates. No FINAL or native writes."""
import copy
import json
from pathlib import Path
import unittest

from a02_admission_nomination import ROOT, validate


class A02NominationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((ROOT / "a02_admission_nomination_v1.json").read_text(encoding="utf-8"))

    def rejected(self, change):
        candidate = copy.deepcopy(self.original)
        change(candidate)
        with self.assertRaises(ValueError):
            validate(candidate)

    def test_real_source_and_byte_budget(self):
        result = validate(self.original)
        self.assertEqual(result["status"], "NOMINATION_SOURCE_PASS_NOT_ADMITTED")
        self.assertLessEqual(result["complete_byte_tokens_including_eos"], 2048)
        self.assertEqual(result["true_eos_token"], 258)
        self.assertEqual(result["new_train_groups"], 0)

    def test_reject_fabricated_train_admission(self):
        self.rejected(lambda d: d.update(training_admitted=True))

    def test_reject_train_split_relabel(self):
        self.rejected(lambda d: d.update(split="TRAIN"))

    def test_reject_two_seed_views_as_two_compositions(self):
        self.rejected(lambda d: d.update(distinct_compositions=2))

    def test_reject_seeds_changed_or_duplicated(self):
        self.rejected(lambda d: d.update(native_host_seeds=[104729, 104729]))

    def test_reject_changed_model_pin(self):
        self.rejected(lambda d: d.update(model="dotlabs/void.1@main"))

    def test_reject_changed_native_source(self):
        self.rejected(lambda d: d.update(original_native_source="0" * 40))

    def test_reject_changed_native_artifact_hash(self):
        self.rejected(lambda d: d.update(original_native_zip_sha256="0" * 64))

    def test_reject_changed_parent_corpus(self):
        self.rejected(lambda d: d.update(original_corpus_manifest_sha256="0" * 64))

    def test_reject_unsupported_playability_claim(self):
        self.rejected(lambda d: d["semantics"].update(gravity_jump_climb_reachability_proven=True))

    def test_reject_side_slots_promoted_to_rooms(self):
        self.rejected(lambda d: d["semantics"].update(traversable_rooms=3))

    def test_reject_missing_nontraversable_prompt_qualifier(self):
        self.rejected(lambda d: d.update(prompt=d["prompt"].replace(
            "non-traversable service slots", "wide playable rooms")))

    def test_reject_invalid_utf8_surrogate(self):
        self.rejected(lambda d: d.update(prompt=d["prompt"] + "\ud800"))

    def test_reject_claimed_final_access(self):
        self.rejected(lambda d: d.update(final_targets_opened=True))

    def test_reject_fabricated_optimizer_or_model_painting(self):
        self.rejected(lambda d: d.update(model_optimizer_steps=1))
        self.rejected(lambda d: d.update(model_generated_paintings=1))

    def test_reject_unreviewed_new_property(self):
        self.rejected(lambda d: d.update(native_admission={"state": "PASS"}))


if __name__ == "__main__":
    unittest.main()
