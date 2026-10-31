#!/usr/bin/env python3
"""NCR-001F read-only corrected physical-host three-sample source delta. NO TRAIN."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import zipfile
import ncr_001f_corrected_samehost_custody as custody

ROOT = Path(__file__).resolve().parents[1]
STATUS = "SOURCE_BOUND_CORRECTED_SPROCKET_WALL_ENVELOPES__BUDGET_NOT_ADMITTED"
PARENT = "55026d914efd3ef4548da0644ccd27bb8de3074f"
ORDER = [["frozen","nca","mlp"], ["nca","mlp","frozen"], ["mlp","frozen","nca"]]
KINDS = {"frozen", "nca", "mlp"}

def require(ok, why):
    if not ok:
        raise ValueError("NCR001F_WALL_DELTA_REFUSED: " + why)

def proxy(*, hidden=1280, taps=3, steps=3, grid=8, channels=4, bottleneck=256):
    """Source-only dense/Conv MAC proxy, NOT observed FLOPs or training cost."""
    require(tuple(type(x) is int for x in (hidden,taps,steps,grid,channels,bottleneck)) ==
            (True,)*6 and (hidden,taps,steps,grid,channels,bottleneck) ==
            (1280,3,3,8,4,256), "unregistered geometry or one-step ablation")
    field=grid*grid*channels
    conv=grid*grid*((channels*4)*channels*9+(channels*2)*(channels*4))
    mlp=taps*2*hidden*bottleneck
    nca=taps*(2*hidden*field+steps*conv)
    nca_params=(2*hidden+hidden*field+field+(channels*4)*channels*9+
                channels*4+(channels*2)*(channels*4)+channels*2+field*hidden)
    mlp_params=2*hidden+hidden*bottleneck+2*bottleneck+bottleneck*hidden
    require((nca,mlp,nca_params,mlp_params)==(2371584,1966080,658904,658432),
            "published #98 pinned arithmetic disagrees")
    return {"nca_dense_conv_mac_per_token":nca, "mlp_dense_mac_per_token":mlp,
            "extra_nca_conv_mac_per_token":nca-mlp,
            "source_only_nca_over_mlp_mac_ratio":nca/mlp,
            "nca_parameters":nca_params, "mlp_parameters":mlp_params,
            "reference_pr98_head":"c8139cc36b1105f6bcafd585b766ff470e8c9e1d",
            "reference_pr98_script_blob":"6ab8ffb3e3327d7a6a92eb3cfb0075679f2425f1",
            "reference_script_executed":False, "measured_flops":False,
            "matched_training_compute":False}

def envelope(row, context):
    """All-pair observed extrema; NOT a statistical confidence interval."""
    require(type(row) is dict and context in (512,2048) and
            row.get("synthetic_shape")==[1,context] and
            row.get("warmup_rounds")==1 and row.get("measured_rounds")==3 and
            row.get("forward_call_order")==ORDER, "source schedule or length drift")
    raw=row.get("raw_complete_frozen_host_forward_ns")
    med=row.get("median_complete_host_forward_ms")
    require(type(raw) is dict and type(med) is dict and
            set(raw)==set(med)==KINDS, "three measured paths required")
    calculated={}
    for k in KINDS:
        seq=raw[k]
        require(type(seq) is list and len(seq)==3 and
                all(type(x) is int and x>0 for x in seq),
                "exactly three valid positive nanosecond samples required")
        calculated[k]=statistics.median(seq)/1e6
        require(type(med[k]) in (int,float) and math.isfinite(med[k]) and
                math.isclose(calculated[k],med[k],rel_tol=1e-12,abs_tol=1e-8),
                "recorded median forged or nonfinite")
    ratio=calculated["nca"]/calculated["mlp"]
    reported=row.get("nca_over_mlp_total_host_wall_ratio")
    require(type(reported) in (int,float) and math.isfinite(reported) and
            math.isclose(ratio,reported,rel_tol=1e-12,abs_tol=1e-12) and
            row.get("actual_compute_budget_qualified") is False and
            row.get("measured_flops") is False, "source ratio/compute gate drift")
    low=min(raw["nca"])/max(raw["mlp"])
    high=max(raw["nca"])/min(raw["mlp"])
    require(0<low<=ratio<=high, "raw sample extrema inconsistent")
    return {"context_units":context, "samples_per_path":3,
            "median_complete_host_ms":calculated,
            "nca_over_mlp_complete_host_median_ratio":ratio,
            "observed_unpaired_ratio_min":low,
            "observed_unpaired_ratio_max":high,
            "observed_extrema_enclose_one":low<=1<=high,
            "nca_over_frozen_whole_host_median_ratio":
                calculated["nca"]/calculated["frozen"],
            "mlp_over_frozen_whole_host_median_ratio":
                calculated["mlp"]/calculated["frozen"],
            "bounds_are_statistical_confidence_interval":False,
            "adapter_isolated_latency_measured":False,
            "measured_flops":False,"actual_compute_budget_qualified":False}

def audit():
    original=custody.audit()
    require(original["status"]==
            "PASS_CORRECTED_ORIGINAL_PR199_THREE_ZIPS_AND_LOGS__72_73_BLOCKED" and
            original["original_source_head"]==
            "2e2d7c86b066e96207afd6355ddbfbe535d3718e" and
            original["original_run"]==40904839128 and
            original["complete_original_zip_containers"]==3 and
            original["original_decoded_job_logs"]==3 and
            original["train_authorized"] is False and
            original["full_72_conformance_pass"] is False and
            original["actual_compute_budget_qualified"] is False,
            "original permanent three-ZIP three-log audit did not pass")
    manifest_bytes=(custody.EVIDENCE/custody.MANIFEST).read_bytes()
    require(custody.gitblob(manifest_bytes)==custody.MANIFEST_BLOB,
            "original immutable manifest altered")
    manifest=json.loads(manifest_bytes)
    require(manifest["variant_hosts_are_distinct"] is True and
            manifest["corrected_four_thread_measurement_metadata"]==
            {"startup_cpu_threads":2,"effective_cpu_threads":4},
            "physical runner/thread metadata changed")
    lanes={}
    for variant in ("chat","instruct"):
        meta=manifest["artifacts"][variant]
        with zipfile.ZipFile(custody.EVIDENCE/meta["zip_path"]) as z:
            report=json.loads(z.read("report.json"))
        custody.check_model_report(report,variant,meta)
        require(len(report["synthetic_cases"])==2 and
                report["runner"]["cpu_threads"]==4 and
                report["optimizer_steps"]==0, "original case/host changed")
        lanes["sprocket_"+variant]={
            "variant":variant,"frozen_checkpoint_path":meta["weight_path"],
            "frozen_checkpoint_sha256":meta["weight_sha256"],
            "original_job_id":meta["job_id"],
            "original_zip_sha256":meta["sha256"],
            "effective_cpu_threads":4,
            "separate_runner_cpu_model":report["runner"]["cpu_model"],
            "cases":[envelope(row,n) for row,n in zip(
                report["synthetic_cases"],(512,2048))],
            "trained_quality_measured":False,
            "actual_compute_budget_qualified":False}
    require(lanes["sprocket_chat"]["frozen_checkpoint_sha256"] !=
            lanes["sprocket_instruct"]["frozen_checkpoint_sha256"] and
            lanes["sprocket_chat"]["original_job_id"] !=
            lanes["sprocket_instruct"]["original_job_id"],
            "distinct Chat/root physical sources collapsed")
    return {"schema":"ncr001f-corrected-original-wall-envelope-v1",
            "status":STATUS,"source_only":True,
            "original_pr":199,"original_run":40904839128,
            "custody_pr":201,"custody_head":PARENT,
            "model_proxy":proxy(),"lanes":lanes,
            "no_cross_model_wall_ratio_comparison":True,
            "new_physical_execution":False,"new_optimizer_steps":0,
            "comparative_training":False,"heldouts_opened":False,
            "generated_tool_memory_semantics_admitted":False,
            "void1_historical_eos_258_training_effect":"UNMEASURED",
            "train_authorized":False,"full_72_conformance_pass":False,
            "actual_compute_budget_qualified":False}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    data=json.dumps(audit(),indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(data,encoding="utf-8")
    print(data,end="")
