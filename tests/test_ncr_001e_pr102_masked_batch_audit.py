"""Negative original-byte/source and false #72 acceptance sentinels; no LLM."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from ncr_001e_pr102_masked_batch_audit import (  # noqa:E402
    ARCHIVE, load_originals, verify_originals
)


class ReviewerArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.p = load_originals()

    def test_original_receipt_exact_sha_and_numeric_claims(self):
        verify_originals(copy.deepcopy(self.data), copy.deepcopy(self.p))

    def test_archive_binary_replacement_refused(self):
        original = ARCHIVE.read_text(encoding="ascii")
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"corrupt.txt"
            path.write_text(("A" if original[0]!="A" else "B")+original[1:])
            with self.assertRaises((ValueError,RuntimeError)):
                load_originals(path)
            path.write_text("not-actually-encoded")
            with self.assertRaises((ValueError,RuntimeError)):
                load_originals(path)

    def test_changed_artifact_or_source_refused(self):
        for section, field, value in (
            ("original_actions","original_artifact_zip_sha256","0"*64),
            ("original_actions","original_artifact_zip_archived_in_git",True),
            ("original_actions","reviewer_pr_head_during_original_execution","wrong"),
            ("source","reviewed_candidate_commit","wrong"),
            ("source","model_variant","sprocket_chat"),
            ("source","model_weight_file","chat/model.safetensors"),
            ("source","model_weight_sha256","0"*64),
            ("source","original_frozen_instruct_hook_commit","wrong"),
            ("source","publisher_code_is_training_build_attestation",True),
            ("runner","runner_name","other"),
            ("runner","pid",1),
        ):
            p=copy.deepcopy(self.p)
            p[section][field]=value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_originals(self.data,p)

    def test_false_training_and_model_semantic_promotion_refused(self):
        for field in ("training_performed","heldout_evaluation","train_authorized",
                      "model_generated_tool_semantics_qualified",
                      "final_four_model_72_accepted",
                      "cross_process_cold_reload_in_this_new_review"):
            p=copy.deepcopy(self.p)
            p["evidence_limit"][field]=True
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                verify_originals(self.data,p)
        for field in ("train_authorized","heldout_evaluation",
                      "semantic_tool_or_memory_quality_qualified",
                      "final_four_model_72_pass","comparative_training"):
            d=copy.deepcopy(self.data)
            d["pr102-independent-batch-review.json"][field]=True
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                verify_originals(d,self.p)

    def test_nonzero_masked_padding_cache_corruption_refused(self):
        for check, field, value in (
            ("left_frozen_zero_exact","max_abs",0.1),
            ("right_other_request_isolation","pass",False),
            ("left_masked_pad_control_no_effect_1","max_abs",1e-4),
            ("right_masked_pad_control_no_effect_0","atol",.1),
            ("left_masked_batch_full_vs_4_cached","max_abs",.01),
            ("left_masked_batch_full_vs_4_cached","atol",.01),
            ("left_masked_batch_full_vs_4_cached","pass",False),
            ("unmasked_pad_injection_detected","max_abs",0.0),
            ("unmasked_pad_injection_detected","expected_to_change",False),
        ):
            d=copy.deepcopy(self.data)
            d["pr102-independent-batch-review.json"]["checks"][check][field]=value
            with self.subTest(check=check,field=field),self.assertRaises(RuntimeError):
                verify_originals(d,self.p)

    def test_relabelled_checkpoint_bytes_or_runner_refused(self):
        d=copy.deepcopy(self.data)
        d["pr102-independent-prefetch.json"]["files"]["model.safetensors"]["sha256"]="0"*64
        with self.assertRaises(RuntimeError):
            verify_originals(d,self.p)
        d=copy.deepcopy(self.data)
        d["pr102-independent-batch-review.json"]["backbone_after_sha256"]="0"*64
        with self.assertRaises(RuntimeError):
            verify_originals(d,self.p)
        d=copy.deepcopy(self.data)
        d["pr102-independent-batch-review.json"]["wire"]["primary"]["trained_units"]=0
        with self.assertRaises(RuntimeError):
            verify_originals(d,self.p)

    def test_frozen_conformance_does_not_admit_four_models(self):
        self.assertFalse(self.data["pr102-independent-batch-review.json"]["final_four_model_72_pass"])
        self.assertFalse(self.p["evidence_limit"]["final_four_model_72_accepted"])
        self.assertFalse(self.p["evidence_limit"]["train_authorized"])
        self.assertEqual(self.data["pr102-independent-batch-review.json"]["optimizer_steps"],0)


if __name__=="__main__":
    unittest.main()
