"""Fail-closed no-model tests for original void.1 2048 evidence archive."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from ncr_001e_void_2048_archive_audit import (  # noqa:E402
    ARCHIVE, load_originals, verify_originals
)


def mutate_report(members, fn):
    out=dict(members)
    j=json.loads(out["void1-2048-context.json"])
    fn(j)
    out["void1-2048-context.json"]=json.dumps(j).encode()
    return out


class Void2048ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.p=load_originals()

    def test_positive_original_full_archive(self):
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

    def test_source_model_tokenizer_hook_runner_mutations_refused(self):
        for section,key,value in (
            ("source","checkpoint_sha256","0"*64),
            ("source","checkpoint_file","other.safetensors"),
            ("source","tokenizer_source_sha256","0"*64),
            ("source","original_hook_commit","wrong"),
            ("source","max_context_units",1024),
            ("runner","pid",1),
            ("runner","runner_name","other"),
            ("original_actions","original_zip_sha256","0"*64),
            ("original_actions","original_zip_bytes_committed",True),
        ):
            p=copy.deepcopy(self.p);p[section][key]=value
            with self.subTest(section=section,key=key),self.assertRaises(RuntimeError):
                verify_originals(self.m,p)

    def test_training_history_eos_propensity_and_final_pass_refused(self):
        for key in ("train_authorized","actual_training","heldout_evaluation",
                    "historical_byte_eos_impact_measured",
                    "final_four_model_72_conformance",
                    "future_byte_eos_serializer_integrated",
                    "comparative_NCA_MLP_training"):
            p=copy.deepcopy(self.p);p["scope"][key]=True
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                verify_originals(self.m,p)
        p=copy.deepcopy(self.p)
        p["scope"]["eos_non_emission_in_four_units_is_not_propensity_estimate"]=False
        with self.assertRaises(RuntimeError):
            verify_originals(self.m,p)

    def test_all_three_context_lengths_cache_and_exact_gates_immutable(self):
        changes=(
            ("zero_exact_2048","max_abs",.001),
            ("full_vs_2032_prefill_16_cached_2048","max_abs",.001),
            ("full_vs_1008_prefill_16_cached_1024","atol",.01),
            ("strict_future_suffix_2048","pass",False),
            ("A_B_A_request_reset_exact","exact_required",False),
            ("generate_full_vs_cache_disabled_step_1","max_abs_finite",.001),
            ("generate_zero_exact_cache_True_step_2","max_abs_finite",1e-4),
        )
        for name,key,value in changes:
            m=mutate_report(self.m,lambda j:
                            j["checks"][name].__setitem__(key,value))
            with self.subTest(name=name,key=key),self.assertRaises(RuntimeError):
                verify_originals(m,self.p)

    def test_raw_generation_records_cannot_be_relabelled_as_eos_evidence(self):
        def change(j,mode):
            if mode=="drop":j["generations"].pop()
            elif mode=="ids":j["generations"][0]["generated_ids"]=[258,258,258,258]
            elif mode=="eos":j["generations"][0]["actual_eos_258_emitted"]=True
            elif mode=="impact":j["generations"][0]["historical_byte_eos_impact_measured"]=True
            elif mode=="score":j["generations"][2]["score_sha256"][0]="f"*64
            elif mode=="opt":j["generations"][4]["optimizer_steps"]=1
        for mode in ("drop","ids","eos","impact","score","opt"):
            m=mutate_report(self.m,lambda j:change(j,mode))
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                verify_originals(m,self.p)

    def test_all_model_tokenizer_source_files_must_match_original_pins(self):
        for name in ("model.safetensors","config.json","tokenization_byte.py",
                     "byte_vocab.json","tokenizer_config.json","special_tokens_map.json"):
            for report in ("void1-2048-prefetch.json","void1-2048-context.json"):
                for key,value in (("sha256","0"*64),("bytes",0)):
                    m=dict(self.m)
                    j=json.loads(m[report])
                    group="files" if "prefetch" in report else "actual_source_files"
                    j[group][name][key]=value
                    m[report]=json.dumps(j).encode()
                    with self.subTest(name=name,report=report,key=key), \
                            self.assertRaises(RuntimeError):
                        verify_originals(m,self.p)


if __name__=="__main__":
    unittest.main()
