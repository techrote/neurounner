"""Adversarial original PR199 whole-container and decoded Git-log custody; no models."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001f_corrected_samehost_custody as custody

class CorrectedOriginalCustodyTests(unittest.TestCase):
    def test_original_all_three_zips_and_complete_logs(self):
        result = custody.audit()
        self.assertEqual(result["complete_original_zip_containers"], 3)
        self.assertEqual(result["original_decoded_job_logs"], 3)
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["full_72_conformance_pass"])
        self.assertFalse(result["actual_compute_budget_qualified"])

    def mutation(self, filename):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            m = json.loads((custody.EVIDENCE/custody.MANIFEST).read_text())
            for f in [custody.MANIFEST]+[a[k] for a in m["artifacts"].values()
                                          for k in ("zip_path","log_path")]:
                shutil.copy2(custody.EVIDENCE/f,d/f)
            p = d/filename
            raw = bytearray(p.read_bytes())
            raw[len(raw)//2] ^= 1
            p.write_bytes(raw)
            with self.assertRaises((ValueError, zipfile.BadZipFile)):
                custody.audit(d)

    def test_every_original_zip_mutation_refused(self):
        for n in ("source","chat","instruct"):
            with self.subTest(variant=n):
                m=json.loads((custody.EVIDENCE/custody.MANIFEST).read_text())
                self.mutation(m["artifacts"][n]["zip_path"])

    def test_every_original_decoded_job_log_mutation_refused(self):
        m=json.loads((custody.EVIDENCE/custody.MANIFEST).read_text())
        for n in ("source","chat","instruct"):
            with self.subTest(variant=n):
                self.mutation(m["artifacts"][n]["log_path"])

    def test_manifest_git_blob_mutation_refused(self):
        self.mutation(custody.MANIFEST)

    def original_report(self, variant):
        m=json.loads((custody.EVIDENCE/custody.MANIFEST).read_text())
        item=m["artifacts"][variant]
        with zipfile.ZipFile(custody.EVIDENCE/item["zip_path"]) as z:
            return json.loads(z.read("report.json")),item

    def test_cross_variant_model_weight_and_cpu_substitution_refused(self):
        r,item=self.original_report("chat")
        for key,value in (("variant","instruct"),):
            x=copy.deepcopy(r);x[key]=value
            with self.assertRaisesRegex(ValueError,"REFUSED"):
                custody.check_model_report(x,"chat",item)
        for k in ("cpu_model","cpu_threads","cpu_threads_at_report_start"):
            x=copy.deepcopy(r);x["runner"][k]="wrong" if k=="cpu_model" else 3
            with self.assertRaisesRegex(ValueError,"REFUSED"):
                custody.check_model_report(x,"chat",item)
        x=copy.deepcopy(r);x["source_identity"]["weight_path"]="model.safetensors"
        with self.assertRaisesRegex(ValueError,"REFUSED"):
            custody.check_model_report(x,"chat",item)

    def test_false_training_or_scientific_admission_refused(self):
        r,item=self.original_report("instruct")
        for k in ("train_authorized","full_72_conformance_pass",
                  "actual_compute_budget_qualified","scientific_admission"):
            x=copy.deepcopy(r);x[k]=True
            with self.subTest(k=k),self.assertRaisesRegex(ValueError,"REFUSED"):
                custody.check_model_report(x,"instruct",item)
        x=copy.deepcopy(r);x["optimizer_steps"]=1
        with self.assertRaisesRegex(ValueError,"REFUSED"):
            custody.check_model_report(x,"instruct",item)

    def test_original_startup_and_effective_four_threads_are_distinct(self):
        r,item=self.original_report("instruct")
        self.assertEqual(r["runner"]["cpu_threads_at_report_start"],2)
        self.assertEqual(r["runner"]["cpu_threads"],4)
        for key,bad in (("cpu_threads",2),("cpu_threads_at_report_start",4)):
            x=copy.deepcopy(r);x["runner"][key]=bad
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"REFUSED"):
                custody.check_model_report(x,"instruct",item)

    def test_raw_timing_median_and_ratio_forgery_refused(self):
        r,item=self.original_report("chat")
        for key in ("median_complete_host_forward_ms","raw_complete_frozen_host_forward_ns",
                    "nca_over_mlp_total_host_wall_ratio"):
            x=copy.deepcopy(r);row=x["synthetic_cases"][0]
            if key=="median_complete_host_forward_ms":
                row[key]["nca"] += 100
            elif key=="raw_complete_frozen_host_forward_ns":
                row[key]["nca"][1] += 1_000_000_000
            else:row[key] += 0.5
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"REFUSED"):
                custody.check_model_report(x,"chat",item)

    def test_forged_schedule_and_other_variant_weights_refused(self):
        r,item=self.original_report("chat")
        x=copy.deepcopy(r)
        x["synthetic_cases"][0]["forward_call_order"][1]=["frozen","nca","mlp"]
        with self.assertRaisesRegex(ValueError,"REFUSED"):
            custody.check_model_report(x,"chat",item)
        x=copy.deepcopy(r)
        x["source_identity"]["weight_sha256"]="1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
        with self.assertRaisesRegex(ValueError,"REFUSED"):
            custody.check_model_report(x,"chat",item)

if __name__=="__main__":
    unittest.main()
