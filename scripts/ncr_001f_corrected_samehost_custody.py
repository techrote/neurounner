#!/usr/bin/env python3
"""Fail-closed custody audit of ORIGINAL #199 Actions ZIP containers and job logs; no models."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review/pr199-corrected-samehost"
MANIFEST = "original-actions-custody.json"
MANIFEST_BLOB = "87a1cfa919cad3103d1c075e5b01f685d12ffd0c"
ORIGINAL_SOURCE = "2e2d7c86b066e96207afd6355ddbfbe535d3718e"
ORIGINAL_RUN = "40904839128"
STAGE = "MEASURED_SAME_FROZEN_HOST_SYNTHETIC_FORWARD_ONLY__BUDGET_NOT_ADMITTED"
TOKEN_HASH = "d1535b03aa7e925ff35edcc5bc58a4493efdc587cc37bd7b2e2a3b81459f7e4a"

def refuse(cond: bool, why: str) -> None:
    if not cond:
        raise ValueError("NCR001F_ORIGINAL_CUSTODY_REFUSED: " + why)

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def gitblob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\x00" + data).hexdigest()

def check_model_report(r: dict, variant: str, meta: dict) -> None:
    refuse(r["variant"] == variant and r["status"] == STAGE,
           "variant or original physical classification changed")
    source = r["source_identity"]
    refuse(source["weight_path"] == meta["weight_path"] and
           source["weight_sha256"] == meta["weight_sha256"] and
           source["revision"] == "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
           "variant checkpoint or frozen source changed")
    runner = r["runner"]
    refuse(str(runner["github_run_id"]) == ORIGINAL_RUN and
           runner["expected_pr_head"] == ORIGINAL_SOURCE and
           runner["cpu_model"] == meta["cpu_model"] and
           runner["cpu_threads_at_report_start"] == 2 and
           runner["cpu_threads"] == 4 and
           runner["torch"] == "2.14.1+cpu" and
           runner["transformers"] == "4.57.6",
           "original runtime/source fields or historical startup thread discrepancy changed")
    refuse(r["frozen_host_loaded"] is True and
           r["frozen_backbone_parameters"] == 501090560 and
           r["nca_trainable_parameters"] == 658904 and
           r["mlp_trainable_parameters"] == 658432 and
           r["frozen_backbone_before_sha256"] == r["frozen_backbone_after_sha256"],
           "original separate frozen model/adapters changed")
    refuse(r["input_token_ids_sha256"] == TOKEN_HASH and
           r["actual_context_units"] == [512, 2048],
           "original synthetic input source/context drift")
    for key in ("train_authorized", "full_72_conformance_pass",
                "actual_compute_budget_qualified", "training_performed",
                "scientific_admission", "heldouts_opened",
                "comparative_nca_mlp_training",
                "generated_tool_memory_semantics_qualified"):
        refuse(r[key] is False, "forbidden scientific/training claim: " + key)
    refuse(r["optimizer_steps"] == 0 and
           r["nca_manual_untrained_effect"] > 0 and
           r["mlp_manual_untrained_effect"] > 0 and
           r["peak_process_ru_maxrss_kib"] > 1000000,
           "historical intervention/optimizer/whole-process memory drift")
    refuse(len(r["synthetic_cases"]) == 2, "original 512/2048 cases missing")
    for row, length in zip(r["synthetic_cases"], (512, 2048)):
        refuse(row["synthetic_shape"] == [1, length] and
               row["warmup_rounds"] == 1 and
               row["measured_rounds"] == 3 and
               row["forward_call_order"] == [
                   ["frozen","nca","mlp"], ["nca","mlp","frozen"],
                   ["mlp","frozen","nca"]],
               "original measurement schedule drift")
        samples = row["raw_complete_frozen_host_forward_ns"]
        medians = row["median_complete_host_forward_ms"]
        refuse(set(samples) == set(medians) == {"frozen","nca","mlp"},
               "original three complete-host paths missing")
        for kind in ("frozen","nca","mlp"):
            nums = samples[kind]
            refuse(len(nums) == 3 and
                   all(type(n) is int and n > 0 for n in nums),
                   "original raw nanosecond sample missing")
            refuse(math.isclose(statistics.median(nums)/1e6, medians[kind],
                                abs_tol=1e-8, rel_tol=1e-12),
                   "median not reproducible from original raw samples")
        refuse(math.isclose(medians["nca"]/medians["mlp"],
                            row["nca_over_mlp_total_host_wall_ratio"],
                            abs_tol=1e-12, rel_tol=1e-12) and
               row["actual_compute_budget_qualified"] is False and
               row["measured_flops"] is False,
               "historical derived ratio or compute gate forged")

def audit(evidence: Path = EVIDENCE) -> dict:
    raw_manifest = (evidence / MANIFEST).read_bytes()
    refuse(gitblob(raw_manifest) == MANIFEST_BLOB,
           "immutable original manifest Git blob changed")
    m = json.loads(raw_manifest)
    refuse(m["original_pr"] == 199 and
           m["original_run"] == 40904839128 and
           m["original_source_head"] == ORIGINAL_SOURCE and
           m["parent_source_head"] == "2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80" and
           set(m["artifacts"]) == {"source","chat","instruct"},
           "original run/stack/three-artifact inventory drift")
    refuse(m["scientific_flags"] == {
        "train_authorized": False, "full_72_conformance_pass": False,
        "actual_compute_budget_qualified": False, "optimizer_steps": 0} and
           m["audit_is_offline_no_model"] is True and
           m["variant_hosts_are_distinct"] is True and
           m["corrected_four_thread_measurement_metadata"] == {
               "startup_cpu_threads": 2, "effective_cpu_threads": 4} and
           m["pre_correction_run_retained_separately"] == 40894595108 and
           m["generated_tool_memory_semantics_qualified"] is False and
           m["void_original_training_eos_258_impact"] == "UNMEASURED",
           "scientific custody scope changed")
    for variant in ("source","chat","instruct"):
        item = m["artifacts"][variant]
        raw_zip = (evidence / item["zip_path"]).read_bytes()
        refuse(len(raw_zip) == item["bytes"] and
               sha256(raw_zip) == item["sha256"] and
               gitblob(raw_zip) == item["git_blob"],
               variant + " original complete Actions ZIP changed")
        with zipfile.ZipFile(evidence / item["zip_path"]) as z:
            names = z.namelist()
            refuse(len(names) == len(set(names)) == len(item["zip_members"]) and
                   set(names) == set(item["zip_members"]) and
                   all(n == Path(n).name and not z.getinfo(n).is_dir() for n in names),
                   variant + " malformed original archive member inventory")
            refuse(z.read("head.txt").decode().strip() == ORIGINAL_SOURCE,
                   variant + " original checked-out head changed")
            if variant == "source":
                refuse(b"Ran 9 tests" in z.read("tests.log") and
                       b"OK" in z.read("tests.log"),
                       "original source test evidence changed")
            else:
                raw_report = z.read("report.json")
                refuse(sha256(raw_report) == item["report_sha256"],
                       variant + " raw physical report bytes changed")
                check_model_report(json.loads(raw_report), variant, item)
        raw_log = (evidence / item["log_path"]).read_bytes()
        refuse(gitblob(raw_log) == item["log_git_blob"],
               variant + " committed full original decoded job log changed")
        refuse(ORIGINAL_SOURCE.encode() in raw_log and ORIGINAL_RUN.encode() in raw_log,
               variant + " original GitHub log provenance absent")
        if variant == "source":
            refuse(b"Ran 9 tests" in raw_log, "source job log did not run original tests")
        else:
            refuse(b"PASS_MEASURED_SAME_HOST_FORWARD_ONLY__SCIENTIFIC_GATES_BLOCKED" in raw_log,
                   variant + " original physical completion log missing")
    refuse(m["artifacts"]["chat"]["job_id"] !=
           m["artifacts"]["instruct"]["job_id"] and
           m["artifacts"]["chat"]["cpu_model"].startswith("AMD EPYC 7763") and
           m["artifacts"]["instruct"]["cpu_model"].startswith("AMD EPYC 7763") and
           m["artifacts"]["chat"]["weight_sha256"] !=
           m["artifacts"]["instruct"]["weight_sha256"],
           "distinct host/model lanes silently collapsed")
    return {"status":"PASS_CORRECTED_ORIGINAL_PR199_THREE_ZIPS_AND_LOGS__72_73_BLOCKED",
            "original_source_head":ORIGINAL_SOURCE, "original_run":int(ORIGINAL_RUN),
            "complete_original_zip_containers":3, "original_decoded_job_logs":3,
            "new_model_execution":False, "optimizer_steps":0,
            "train_authorized":False, "full_72_conformance_pass":False,
            "actual_compute_budget_qualified":False}


def audit_original_job_ledger(payload: dict, evidence: Path = EVIDENCE) -> dict:
    """Independently bind original GitHub job IDs/names to one completed run; no model."""
    raw_manifest = (evidence / MANIFEST).read_bytes()
    refuse(gitblob(raw_manifest) == MANIFEST_BLOB,
           "immutable manifest changed before original live job-ledger audit")
    manifest = json.loads(raw_manifest)
    refuse(manifest["original_run"] == int(ORIGINAL_RUN),
           "original job-ledger run is not the archived source run")
    expected_names = {
        "source": "offline-source-and-refusals",
        "chat": "real-separate-frozen-variant (chat)",
        "instruct": "real-separate-frozen-variant (instruct)",
    }
    jobs = payload.get("jobs") if type(payload) is dict else None
    refuse(type(jobs) is list and len(jobs) == 3,
           "original Actions job inventory incomplete or has unexpected jobs")
    actual = {j.get("id"): j for j in jobs if type(j) is dict}
    refuse(len(actual) == 3, "original Actions job IDs duplicated or malformed")
    for variant, name in expected_names.items():
        pinned_id = manifest["artifacts"][variant]["job_id"]
        job = actual.get(pinned_id)
        refuse(type(job) is dict and job.get("name") == name and
               job.get("run_id") == int(ORIGINAL_RUN) and
               job.get("status") == "completed" and
               job.get("conclusion") == "success",
               variant + " original job ID/name/run/conclusion not independently established")
    return {"status": "PASS_ORIGINAL_GITHUB_JOBS_LEDGER_BINDING",
            "run": int(ORIGINAL_RUN), "jobs_checked": len(actual)}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
