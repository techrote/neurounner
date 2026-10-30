"""NCR-001F corrected original physical-host sample-envelope source-only regressions."""
from __future__ import annotations
import copy
import math
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001f_corrected_samehost_budget_delta as delta

def fixture():
    return {
      "synthetic_shape":[1,512], "warmup_rounds":1, "measured_rounds":3,
      "forward_call_order":copy.deepcopy(delta.ORDER),
      "raw_complete_frozen_host_forward_ns":{
          "frozen":[100_000_000,101_000_000,99_000_000],
          "nca":[110_000_000,115_000_000,105_000_000],
          "mlp":[100_000_000,112_000_000,108_000_000]},
      "median_complete_host_forward_ms":{
          "frozen":100.0,"nca":110.0,"mlp":108.0},
      "nca_over_mlp_total_host_wall_ratio":110/108,
      "actual_compute_budget_qualified":False,"measured_flops":False}

class CorrectedBudgetDelta(unittest.TestCase):
    def test_all_original_archives_read_only_and_four_derived_rows(self):
        x=delta.audit()
        self.assertEqual(x["status"],delta.STATUS)
        self.assertEqual(x["original_run"],40904839128)
        self.assertEqual(set(x["lanes"]),{"sprocket_chat","sprocket_instruct"})
        self.assertNotEqual(x["lanes"]["sprocket_chat"]["frozen_checkpoint_sha256"],
                            x["lanes"]["sprocket_instruct"]["frozen_checkpoint_sha256"])
        self.assertNotEqual(x["lanes"]["sprocket_chat"]["original_job_id"],
                            x["lanes"]["sprocket_instruct"]["original_job_id"])
        for lane in x["lanes"].values():
            self.assertEqual(lane["effective_cpu_threads"],4)
            self.assertEqual([a["context_units"] for a in lane["cases"]],[512,2048])
            for row in lane["cases"]:
                self.assertEqual(row["samples_per_path"],3)
                self.assertFalse(row["bounds_are_statistical_confidence_interval"])
                self.assertFalse(row["adapter_isolated_latency_measured"])
                self.assertFalse(row["actual_compute_budget_qualified"])
        for key in ("new_physical_execution","comparative_training","heldouts_opened",
                    "train_authorized","full_72_conformance_pass",
                    "actual_compute_budget_qualified"):
            self.assertIs(x[key],False)

    def test_source_proxy_recalculated_not_physical_flops(self):
        x=delta.proxy()
        self.assertEqual((x["nca_dense_conv_mac_per_token"],
                          x["mlp_dense_mac_per_token"]),(2371584,1966080))
        self.assertEqual(x["extra_nca_conv_mac_per_token"],405504)
        self.assertEqual(x["source_only_nca_over_mlp_mac_ratio"],1.20625)
        self.assertEqual(x["nca_parameters"]-x["mlp_parameters"],472)
        self.assertIs(x["reference_script_executed"],False)
        self.assertIs(x["measured_flops"],False)
        self.assertIs(x["matched_training_compute"],False)

    def test_unregistered_proxy_geometries_or_ablation_refused(self):
        for kwargs in ({"hidden":576},{"taps":2},{"steps":1},{"steps":2},
                       {"grid":16},{"channels":8},{"bottleneck":128},
                       {"steps":True}):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(
                    ValueError,"REFUSED"):
                delta.proxy(**kwargs)

    def test_exact_observed_extrema_and_whole_backbone_normalizer(self):
        x=delta.envelope(fixture(),512)
        self.assertAlmostEqual(x["nca_over_mlp_complete_host_median_ratio"],110/108)
        self.assertAlmostEqual(x["observed_unpaired_ratio_min"],105/112)
        self.assertAlmostEqual(x["observed_unpaired_ratio_max"],115/100)
        self.assertTrue(x["observed_extrema_enclose_one"])
        self.assertAlmostEqual(x["nca_over_frozen_whole_host_median_ratio"],1.1)
        self.assertFalse(x["bounds_are_statistical_confidence_interval"])

    def test_nonoverlap_remains_not_confidence_or_compute_acceptance(self):
        r=fixture()
        r["raw_complete_frozen_host_forward_ns"]["nca"]=[
            150_000_000,155_000_000,145_000_000]
        r["median_complete_host_forward_ms"]["nca"]=150.0
        r["nca_over_mlp_total_host_wall_ratio"]=150/108
        x=delta.envelope(r,512)
        self.assertFalse(x["observed_extrema_enclose_one"])
        self.assertFalse(x["bounds_are_statistical_confidence_interval"])
        self.assertFalse(x["actual_compute_budget_qualified"])

    def test_forged_median_ratio_schedule_or_admission_refused(self):
        for kind in ("median","ratio","zero","missing","extra","two_samples",
                     "order","admission","flops","context","warmup","nan","boolean"):
            r=fixture()
            raw=r["raw_complete_frozen_host_forward_ns"]
            if kind=="median":r["median_complete_host_forward_ms"]["nca"]+=1
            elif kind=="ratio":r["nca_over_mlp_total_host_wall_ratio"]+=0.5
            elif kind=="zero":raw["mlp"][0]=0
            elif kind=="missing":raw.pop("frozen")
            elif kind=="extra":raw["void1"]=[1,2,3]
            elif kind=="two_samples":raw["nca"].pop()
            elif kind=="order":r["forward_call_order"][1]=["nca","frozen","mlp"]
            elif kind=="admission":r["actual_compute_budget_qualified"]=True
            elif kind=="flops":r["measured_flops"]=True
            elif kind=="context":r["synthetic_shape"]=[1,2048]
            elif kind=="warmup":r["warmup_rounds"]=2
            elif kind=="nan":r["median_complete_host_forward_ms"]["nca"]=math.nan
            elif kind=="boolean":raw["nca"][0]=True
            with self.subTest(kind=kind),self.assertRaisesRegex(
                    (ValueError,TypeError),"REFUSED"):
                delta.envelope(r,512)

    def test_reject_unsupported_256_units_and_boolean_median(self):
        with self.assertRaisesRegex(ValueError,"REFUSED"):
            delta.envelope(fixture(),256)
        r=fixture()
        r["median_complete_host_forward_ms"]["nca"]=True
        with self.assertRaisesRegex(ValueError,"REFUSED"):
            delta.envelope(r,512)

if __name__=="__main__":
    unittest.main()
