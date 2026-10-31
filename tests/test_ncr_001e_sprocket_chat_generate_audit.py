"""Fail-closed no-model audit tests for original Sprocket Chat reviewer evidence."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from ncr_001e_sprocket_chat_generate_audit import (  # noqa:E402
    ARCHIVE, load_originals, verify_originals
)


def mutate_report(members, fn):
    result=dict(members)
    j=json.loads(result["sprocket-chat-generate-review.json"])
    fn(j)
    result["sprocket-chat-generate-review.json"]=json.dumps(j).encode()
    return result


class ChatGenerateArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.p=load_originals()

    def test_positive_original_archive_and_all_metrics(self):
        verify_originals(copy.deepcopy(self.m),copy.deepcopy(self.p))

    def test_corrupt_archive_transport_refused(self):
        raw=ARCHIVE.read_text(encoding="ascii")
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.b64"
            p.write_text(("A" if raw[0]!="A" else "B")+raw[1:])
            with self.assertRaises((ValueError,RuntimeError)):
                load_originals(p)
            p.write_text("not-base64")
            with self.assertRaises((ValueError,RuntimeError)):
                load_originals(p)

    def test_chat_instruct_variant_and_source_swaps_refused(self):
        for section,key,value in (
            ("source","variant","instruct"),
            ("source","checkpoint_file","model.safetensors"),
            ("source","checkpoint_sha256","0"*64),
            ("source","distinct_instruct_checkpoint_sha256","0"*64),
            ("source","original_hook_commit","wrong"),
            ("source","formatter_git_blob","wrong"),
            ("source","instruct_tool_memory_semantics_inherited",True),
            ("runner","pid",1),
            ("runner","runner_name","other"),
            ("original_actions","original_zip_sha256","0"*64),
            ("original_actions","original_zip_bytes_committed",True),
        ):
            p=copy.deepcopy(self.p);p[section][key]=value
            with self.subTest(section=section,key=key),self.assertRaises(RuntimeError):
                verify_originals(self.m,p)

    def test_training_heldout_tool_memory_and_final_pass_claims_refused(self):
        for key in ("train_authorized","actual_training","heldout_evaluation",
                    "tool_memory_semantics_qualified","final_four_model_72_conformance",
                    "comparative_NCA_MLP_training",
                    "model_generated_tool_memory_competence"):
            p=copy.deepcopy(self.p);p["scope"][key]=True
            with self.subTest(scope=key),self.assertRaises(RuntimeError):
                verify_originals(self.m,p)
        for key in ("train_authorized","actual_training","heldout_evaluation",
                    "tool_memory_semantics_qualified",
                    "final_four_model_72_conformance"):
            m=mutate_report(self.m,lambda j:j.__setitem__(key,True))
            with self.subTest(report=key),self.assertRaises(RuntimeError):
                verify_originals(m,self.p)

    def test_all_source_weight_tokenizer_files_bound_to_original_pins(self):
        files=("README.md","chat/model.safetensors","chat/config.json",
               "chat/tokenizer.json","chat/tokenizer_config.json")
        for name in files:
            for report in ("sprocket-chat-generate-prefetch.json",
                           "sprocket-chat-generate-review.json"):
                for key,value in (("sha256","0"*64),("bytes",0)):
                    cloned=dict(self.m)
                    j=json.loads(cloned[report])
                    group="files" if "prefetch" in report else "actual_source_files"
                    j[group][name][key]=value
                    cloned[report]=json.dumps(j).encode()
                    with self.subTest(name=name,report=report,key=key), \
                            self.assertRaises(RuntimeError):
                        verify_originals(cloned,self.p)

    def test_cache_mask_and_negative_sentinel_cannot_be_relabelled(self):
        changes=(
            ("rightgap_vs_unpadded_long_chat","max_abs",.01),
            ("rightgap_vs_unpadded_short_chat","rtol",.01),
            ("masked_padding_control_id7_exact","max_abs",1e-4),
            ("post_generation_A_B_A_exact","pass",False),
            ("unmasked_padding_negative_detected","max_abs",0),
            ("generate_full_vs_cache_zero_step_2","max_abs_finite",.001),
            ("generate_zero_exact_cache_True_step_3","exact_required",False),
        )
        for name,key,value in changes:
            m=mutate_report(self.m,lambda j:
                            j["checks"][name].__setitem__(key,value))
            with self.subTest(name=name,key=key),self.assertRaises(RuntimeError):
                verify_originals(m,self.p)

    def test_all_six_raw_generation_records_required_and_not_tool_semantic(self):
        def change(j,mode):
            if mode=="drop":j["generations"].pop()
            elif mode=="ids":j["generations"][0]["generated_ids"]=[6,7,10,11]
            elif mode=="tool":j["generations"][0]["tool_memory_semantics_qualified"]=True
            elif mode=="score":j["generations"][2]["score_sha256"][0]="f"*64
            elif mode=="cache":j["generations"][1]["use_cache"]=False
            elif mode=="step":j["generations"][4]["optimizer_steps"]=1
        for mode in ("drop","ids","tool","score","cache","step"):
            m=mutate_report(self.m,lambda j:change(j,mode))
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                verify_originals(m,self.p)

    def test_original_failed_source_test_remains_pre_model_negative(self):
        for key,value in (("result","success"),("model_executed",True),
                          ("failed_stage","model")):
            p=copy.deepcopy(self.p);p["preserved_pre_model_failure"][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                verify_originals(self.m,p)


if __name__=="__main__":
    unittest.main()
