"""No-model negative tests for exact SmolLM2 2048 actual-host raw receipt custody."""
from __future__ import annotations
import copy
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001e_smol_2048_archive_audit as au  # noqa: E402


class Smol2048OriginalReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads((au.ARCHIVE/"provenance.json").read_text())
        cls.prod=json.loads((au.ARCHIVE/"producer.json").read_text())
        cls.cons=json.loads((au.ARCHIVE/"consumer.json").read_text())
        cls.protocol=json.loads(au.PROTOCOL.read_text())
        cls.pins=json.loads(au.PINS.read_text())

    def test_original_complete_sha_custody_and_fail_closed_gate(self):
        result=au.audit()
        self.assertEqual(result["original_actions_run"],40651696185)
        self.assertEqual(result["original_actions_artifact_id"],12800615190)
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["full_model_72_gate"])
        self.assertEqual(result["original_model_invocations_added_by_this_audit"],0)

    def test_exact_raw_hash_and_git_blob_are_independently_recomputed(self):
        for name in ("producer.json","consumer.json"):
            with self.subTest(name=name):
                raw=(au.ARCHIVE/name).read_bytes()
                m=self.meta["original_members"][name]
                self.assertEqual(len(raw),m["bytes"])
                self.assertEqual(au.digest(raw),m["sha256"])
                self.assertEqual(au.git_blob(raw),m["git_blob_sha"])

    def test_missing_or_corrupted_complete_original_refuses(self):
        for mutation in ("missing","tampered","reformatted"):
            with self.subTest(mutation=mutation),TemporaryDirectory() as td:
                root=Path(td)
                for name in ("provenance.json","producer.json","consumer.json"):
                    shutil.copyfile(au.ARCHIVE/name,root/name)
                target=root/"producer.json"
                if mutation=="missing":
                    target.unlink()
                    with self.assertRaises(FileNotFoundError): au.audit(root)
                elif mutation=="tampered":
                    raw=target.read_bytes()
                    target.write_bytes(b"?" + raw[1:])
                    with self.assertRaises(RuntimeError): au.audit(root)
                else:
                    target.write_text(json.dumps(json.loads(target.read_text())))
                    with self.assertRaises(RuntimeError): au.audit(root)

    def test_false_training_final_pass_or_variant_refuses_even_rehashed(self):
        for key,value in (("training_authorized",True),("optimizer_steps",1),
                          ("per_model_gate",True),("heldout_evaluation",True),
                          ("lane","sprocket_chat"),
                          ("status","PASS_FINAL_72")):
            with self.subTest(key=key):
                x=copy.deepcopy(self.prod)
                x[key]=value
                with self.assertRaises(RuntimeError):
                    au.validate_reports(x,self.cons,self.meta,self.protocol,self.pins)

    def test_tampered_causal_cache_and_hf_generate_scores_refuse_even_rehashed(self):
        for path,value in (
            (("checks","active_tail_2048_prefill_2032_plus_16","max_abs"),0.01),
            (("checks","active_tail_1024_prefill_1008_plus_16","within"),False),
            (("checks","future_suffix_256","max_abs"),0.001),
            (("checks","hf_generate_active_cache_on_off_scores","max_abs"),0.01),
            (("checks","zero_noop_512","exact"),False),
            (("checks","request_A_B_A_exact"),False),
        ):
            with self.subTest(path=path):
                x=copy.deepcopy(self.prod)
                d=x
                for key in path[:-1]: d=d[key]
                d[path[-1]]=value
                with self.assertRaises(RuntimeError):
                    au.validate_reports(x,self.cons,self.meta,self.protocol,self.pins)

    def test_false_cross_process_or_provenance_refuses(self):
        for path,value in (
            (("process","pid"),self.prod["process"]["pid"]),
            (("source","hook_commit"),"wrong_hook"),
            (("source","actual_full_weight_sha256"),"chat-not-smol"),
            (("checks","wrong_revision_cold_load_refused"),False),
        ):
            with self.subTest(path=path):
                p,c=copy.deepcopy(self.prod),copy.deepcopy(self.cons)
                d=p if path[0]=="source" else c
                for key in path[:-1]: d=d[key]
                d[path[-1]]=value
                with self.assertRaises(RuntimeError):
                    au.validate_reports(p,c,self.meta,self.protocol,self.pins)

    def test_protocol_false_admission_or_numerical_waiver_refuses(self):
        for change in ("train","models","atol"):
            with self.subTest(change=change):
                p=copy.deepcopy(self.protocol)
                if change=="train": p["train_authorized"]=True
                if change=="models": del p["models"]["sprocket_instruct"]
                if change=="atol": p["conformance"]["full_vs_cached_prefill_atol"]=0.1
                with self.assertRaises(RuntimeError):
                    au.validate_reports(self.prod,self.cons,self.meta,p,self.pins)

    def test_source_only_cannot_manufacture_original_binary_archival_claim(self):
        self.assertFalse(self.meta["original_actions_artifact"]["original_zip_committed_in_git"])
        self.assertIsNone(self.meta["original_members"]["adapter/adapter.pt"]["committed"])
        self.assertIsNone(self.meta["original_members"]["cold-reference.pt"]["committed"])
        self.assertIsNone(self.meta["original_members"]["adapter/metadata.json"]["committed"])


if __name__=="__main__":
    unittest.main()
