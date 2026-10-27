"""No-model readback tamper sentinels for new original rightpad + HF.generate run."""
from __future__ import annotations

import base64
import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from ncr_001e_pr113_rightpad_receipt_audit import (  # noqa:E402
    ARCHIVE, load_originals, verify_originals
)


def changed_report(members, callback):
    result=dict(members)
    j=json.loads(members["pr113-rightpad-generate.json"])
    callback(j)
    result["pr113-rightpad-generate.json"]=(
        json.dumps(j,sort_keys=True).encode("utf-8"))
    return result


class OriginalRightPadArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.p=load_originals()

    def test_true_source_bound_positive_original_readback(self):
        verify_originals(self.m,copy.deepcopy(self.p))

    def test_corrupted_compressed_archive_or_source_member_rejected(self):
        encoded=ARCHIVE.read_text(encoding="ascii")
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"mutated.b64"
            path.write_text(("A" if encoded[0]!="A" else "B")+encoded[1:])
            with self.assertRaises((ValueError,RuntimeError)):
                load_originals(path)
            path.write_text("%%%%not-valid-base64")
            with self.assertRaises((ValueError,RuntimeError)):
                load_originals(path)

    def test_original_zip_or_source_identity_mutation_refused(self):
        tests=(
            ("original_actions","original_zip_sha256","0"*64),
            ("original_actions","original_zip_bytes_committed",True),
            ("original_actions","original_pr_branch_head","wrong"),
            ("original_model","variant","sprocket_chat"),
            ("original_model","weight_sha256","0"*64),
            ("original_model","model_weight_file","chat/model.safetensors"),
            ("original_model","chat_sibling_weight_sha256","0"*64),
            ("original_model","reviewed_candidate_pr102_commit","wrong"),
            ("original_model","hook_source_commit","wrong"),
            ("original_model","publisher_source_is_training_build_attestation",True),
            ("original_runner","pid",1),
            ("original_runner","runner_name","unknown"),
        )
        for section,key,value in tests:
            p=copy.deepcopy(self.p);p[section][key]=value
            with self.subTest(section=section,key=key),self.assertRaises(RuntimeError):
                verify_originals(self.m,p)

    def test_full_original_source_hash_and_weight_inventory_must_match(self):
        for filename in (
            "model.safetensors","config.json","tokenizer.json",
            "tokenizer_config.json","special_tokens_map.json",
        ):
            for report in ("pr113-rightpad-prefetch.json",
                           "pr113-rightpad-generate.json"):
                for key,value in (("sha256","0"*64),("bytes",123)):
                    cloned=dict(self.m)
                    j=json.loads(cloned[report])
                    member = "files" if "prefetch" in report else "original_source_files"
                    j[member][filename][key]=value
                    cloned[report]=json.dumps(j).encode()
                    with self.subTest(file=filename,report=report,key=key), \
                            self.assertRaises(RuntimeError):
                        verify_originals(cloned,self.p)

    def test_frozen_masked_gap_and_full_vs_cache_tolerance_cannot_be_relaxed(self):
        probes=(
            ("rightpad_KV_gap_vs_unpadded_control_context","max_abs",.005),
            ("rightpad_KV_gap_vs_unpadded_unrelated_request","atol",.005),
            ("rightpad_masked_tool_result_no_effect","max_abs",.0001),
            ("generation_followed_by_fresh_A_B_A_exact","exact_required",False),
            ("unmasked_rightpad_negative_detected","max_abs",0),
            ("generate_full_vs_cache_active_untrained_step_3","max_abs_finite",.001),
            ("generate_full_vs_cache_disabled_step_2","rtol",.01),
            ("generate_zero_original_exact_cache_True_step_3","max_abs_finite",.0001),
        )
        for name,key,value in probes:
            altered=changed_report(self.m,lambda j:
                                  j["checks"][name].__setitem__(key,value))
            with self.subTest(name=name,key=key),self.assertRaises(RuntimeError):
                verify_originals(altered,self.p)

    def test_any_training_evaluation_or_general_model_competence_claim_refused(self):
        for key in ("train_authorized","actual_training","heldout_evaluation",
                    "model_tool_memory_semantic_pass",
                    "final_independent_four_model_72_admission",
                    "comparative_NCA_MLP_experiment",
                    "sealed_evaluation_fixture_changed",
                    "protocol_changed","reviewed_PR102_modified"):
            p=copy.deepcopy(self.p)
            p["scope"][key]=True
            with self.subTest(scope=key),self.assertRaises(RuntimeError):
                verify_originals(self.m,p)
        for key in ("train_authorized","any_model_training","heldout_evaluation",
                    "semantic_tool_memory_qualified",
                    "final_four_model_72_conformance"):
            altered=changed_report(self.m,lambda j: j.__setitem__(key,True))
            with self.subTest(original=key),self.assertRaises(RuntimeError):
                verify_originals(altered,self.p)

    def test_all_six_original_generated_sequences_and_scores_not_relabelled(self):
        def edits(j,mode):
            if mode=="remove":
                j["generations"].pop()
            elif mode=="pretend_model_tool":
                j["generations"][0]["not_behavioral_tool_or_memory_qualification"]=False
            elif mode=="forge_ids":
                j["generations"][0]["original_generated_ids"]=[6,7,10,11]
            elif mode=="forge_scores":
                j["generations"][2]["scores_sha256"][0]="f"*64
            elif mode=="remove_nonfinite":
                j["checks"]["generate_full_vs_cache_zero_step_3"]["nonfinite_logits"]=0
            elif mode=="false_cache":
                j["generations"][1]["use_cache"]=False
        for mode in ("remove","pretend_model_tool","forge_ids","forge_scores",
                     "remove_nonfinite","false_cache"):
            altered=changed_report(self.m,lambda j: edits(j,mode))
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                verify_originals(altered,self.p)

    def test_failed_original_source_only_runs_remain_negative(self):
        for mode in ("remove","source_success","model_executed"):
            p=copy.deepcopy(self.p)
            if mode=="remove":p["pre_model_failures"].clear()
            elif mode=="source_success":p["pre_model_failures"][0]["result"]="success"
            else:p["pre_model_failures"][1]["model_executed"]=True
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                verify_originals(self.m,p)

    def test_original_scope_is_not_full_four_lane_or_optimized_policy(self):
        r=json.loads(self.m["pr113-rightpad-generate.json"])
        self.assertIs(r["train_authorized"],False)
        self.assertEqual(r["optimizer_steps"],0)
        self.assertIs(r["final_four_model_72_conformance"],False)
        self.assertEqual(self.p["results"]["generated_response_length"],4)
        self.assertEqual(len(r["generations"]),6)


if __name__=="__main__":
    unittest.main()
