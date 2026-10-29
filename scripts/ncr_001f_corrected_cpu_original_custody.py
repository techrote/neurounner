#!/usr/bin/env python3
"""Source-only, byte-exact audit of corrected synthetic CPU adapter evidence.

Original GH Actions run 40870764754; no model/tokenizer/optimizer/training here.
The historical uncorrected cohort stays immutable and separate.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/sprocket-adapter-cpu-corrected-20261029"
MANIFEST = "original-corrected-cpu-provenance.json"
MANIFEST_BLOB = "900b4053851d354862998d2dc73f0b92040cd838"
ZIP_NAME = "original-actions-40870764754.zip"
ZIP_SHA256 = "fe1b04341ee5792ab316f7199fa2052163fb3fdc11241c5fe2e046012404f8c8"
ZIP_GIT_BLOB = "c5f449fcf05df8c576ef10d25f1bada44655f2a8"
ZIP_BYTES = 5696
JOB_LOG = "original-job-123436909436.log"
JOB_LOG_BLOB = "669c7c73322673aef51c6929cf3ac961150a7d7c"
SOURCE_HEAD = "c8139cc36b1105f6bcafd585b766ff470e8c9e1d"
MEMBERS = {
    "lscpu.txt": (3189, "4476e601951bb5e5dc85ca4b9d2e3bc795bb83c0d4e62b8d9f2bdec6659b0e7a"),
    "pip-freeze.txt": (1551, "ea4b616e5ccd392060001737203319bf998f7cc3308f775e18289b02a7ebe050"),
    "report.json": (5634, "ca89f5a6ab30ca857e0b97f54b7888229406ea74ec49ed277b49089acdb247a1"),
    "runner.log": (307, "9adf3a63a06a7d7d6ad7c016b7365dc3a14ecb8c6d90832a341216075786807b"),
    "source-head.txt": (41, "7a69d71287b66d1a59a457fcb78e61fa63178ff5672939a8d69e2b49cad8c937"),
    "tests.log": (2090, "0e040b2d63a985db15853730498636955a23f437856d9f85a1dfa4c54e99b7a2"),
    "uname.txt": (128, "a64b6d81308dc728b286592ae78f6594e185e6db8c474be3c8886f7694f38bb7"),
}
SOURCES = {
    "sprocket_chat": ("chat/model.safetensors", "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
    "sprocket_instruct": ("model.safetensors", "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"),
}
MEDIANS = {
    1: (1.323654, 0.459195, 2.882553),
    16: (4.095753, 1.010530, 4.053074),
    64: (6.803301, 2.530500, 2.688520),
}

def refuse(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("NCR001F_CORRECTED_CPU_CUSTODY_REFUSED: " + why)

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def gitblob(b: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(b)).encode("ascii") + b"\0" + b).hexdigest()

def audit_zip(raw: bytes) -> dict:
    refuse(len(raw) == ZIP_BYTES and sha256(raw) == ZIP_SHA256
           and gitblob(raw) == ZIP_GIT_BLOB,
           "original corrected Actions ZIP bytes/Git blob changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        refuse(len(names) == len(MEMBERS) == len(set(names))
               and set(names) == set(MEMBERS), "original ZIP member inventory changed")
        for name, (size, digest) in MEMBERS.items():
            info = z.getinfo(name)
            refuse(not info.is_dir() and name == Path(name).name
                   and info.file_size == size
                   and sha256(z.read(name)) == digest,
                   "original ZIP member bytes changed: " + name)
        report = json.loads(z.read("report.json"))
        source_head = z.read("source-head.txt")
        tests = z.read("tests.log")
        runner = z.read("runner.log")
    refuse(source_head == (SOURCE_HEAD + "\n").encode(),
           "original measured source revision changed")
    refuse(b"Ran 9 tests" in tests and b"\nOK" in tests
           and b"Failed to initialize NumPy: No module named" in runner,
           "original test log or NumPy warning changed")
    refuse(report["status"] == "ADAPTER_ONLY_CPU_FORWARD_MICROBENCH__FROZEN_HOST_NOT_EXECUTED__NO_TRAIN"
           and report["experiment"] == "NCR-001F" and report["issue"] == 73
           and report["seed"] == 20261029
           and report["optimizer_steps"] == 0,
           "original status/seed/optimizer claim altered")
    for flag in ("train_authorized", "full_72_conformance_pass",
                 "actual_matched_frozen_host_budget_pass", "training_performed",
                 "heldouts_opened", "frozen_model_weights_downloaded",
                 "frozen_model_inference_executed", "tokenizers_executed"):
        refuse(report[flag] is False, "forged model/scientific status: " + flag)
    r = report["runner"]
    refuse(r["github_sha"] == SOURCE_HEAD and r["github_run_id"] == "40870764754"
           and r["github_run_attempt"] == "1" and r["github_job"] == "adapter-only-no-model"
           and r["torch"] == "2.14.1+cpu" and r["torch_num_threads"] == 4
           and r["python"] == "3.12.3"
           and r["cpu_model"] == "AMD EPYC 7763 64-Core Processor",
           "original runner/source/package identity changed")
    variants = report["source_variants_verified_not_executed"]
    refuse(set(variants) == set(SOURCES), "variant population changed")
    for lane, (weight, digest) in SOURCES.items():
        refuse(variants[lane] == {
            "source_revision": "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
            "weight_path": weight, "weight_sha256": digest,
        }, "variant source identity changed: " + lane)
    cases = report["synthetic_cases"]
    refuse([r["synthetic_shape"][1] for r in cases] == [1, 16, 64],
           "synthetic lengths changed")
    for row in cases:
        length = row["synthetic_shape"][1]
        n, m, ratio = MEDIANS[length]
        refuse(row["synthetic_shape"] == [1, length, 1280]
               and row["tap_count"] == 3 and row["nca_steps_per_tap"] == 3
               and row["warmup_rounds_per_adapter"] == 2
               and row["measured_rounds_per_adapter"] == 5
               and row["includes_frozen_transformer"] is False
               and row["includes_backward_or_optimizer"] is False
               and row["timed_interval_includes_noop_or_finite_checks"] is False
               and row["warmup_and_postflight_noop_state_checks"] is True,
               "corrected synthetic timing method/budget changed")
        for kind, expected in (("nca", n), ("mlp", m)):
            series = row["raw_forward_three_tap_ms"][kind]
            refuse(len(series) == 5 and all(type(x) in (int, float)
                   and math.isfinite(x) and x > 0 for x in series)
                   and abs(statistics.median(series) - expected) < 1e-6
                   and abs(row["median_forward_three_tap_ms"][kind] - expected) < 1e-6,
                   "original corrected CPU timing series changed: " + kind)
        refuse(abs(row["median_nca_over_mlp_wall_ratio"] - ratio) < 1e-6,
               "original corrected timing ratio changed")
    return report

def audit(evidence: Path = EVIDENCE) -> dict:
    manifest_raw = (evidence / MANIFEST).read_bytes()
    refuse(gitblob(manifest_raw) == MANIFEST_BLOB,
           "original provenance manifest Git blob changed")
    manifest = json.loads(manifest_raw)
    refuse(manifest["source_head"] == SOURCE_HEAD
           and manifest["original_actions_run"] == 40870764754
           and manifest["original_job_id"] == 123436909436
           and manifest["original_artifact_id"] == 12896225909
           and manifest["original_zip"] == {
               "path": ZIP_NAME, "bytes": ZIP_BYTES,
               "sha256": ZIP_SHA256, "git_blob": ZIP_GIT_BLOB,
           }, "original source/run/ZIP manifest identity changed")
    refuse(set(manifest["original_members"]) == set(MEMBERS)
           and all(manifest["original_members"][name] == {"bytes": size, "sha256": digest}
                   for name, (size, digest) in MEMBERS.items()),
           "original member manifest mismatch")
    refuse(manifest["original_decoded_job"]["git_blob"] == JOB_LOG_BLOB
           and manifest["original_decoded_job"]["path"] == JOB_LOG,
           "original decoded job-log provenance changed")
    refuse(all(manifest[flag] is False for flag in (
        "train_authorized", "full_72_conformance_pass",
        "actual_matched_frozen_host_budget_pass", "scientific_admission",
        "new_physical_model_executed", "heldouts_opened"))
        and manifest["optimizer_steps"] == 0
        and manifest["original_training_impact_void1_eos_258"] == "UNMEASURED",
        "forged admission/optimizer/EOS status in provenance")
    original_log = (evidence / JOB_LOG).read_bytes()
    refuse(gitblob(original_log) == JOB_LOG_BLOB,
           "complete decoded original GitHub job-log Git blob changed")
    refuse(b"Ran 9 tests" in original_log
           and b"ADAPTER_ONLY_MEASURED__REAL_MODEL_BUDGET_UNQUALIFIED__NO_TRAIN" in original_log
           and b"Failed to initialize NumPy: No module named" in original_log,
           "original job-log positive/negative observations changed")
    original_report = audit_zip((evidence / ZIP_NAME).read_bytes())
    refuse(manifest["original_measurement"]["synthetic_t"] == [1, 16, 64]
           and manifest["original_measurement"]["timed_interval_validation_included"] is False,
           "corrected timing provenance changed")
    return {
        "status": "PASS_ORIGINAL_CORRECTED_CPU_ZIP_AND_JOB_GIT_CUSTODY__MODEL_HOST_BLOCKED",
        "original_run": 40870764754, "source_head": SOURCE_HEAD,
        "zip_sha256": ZIP_SHA256, "zip_git_blob": ZIP_GIT_BLOB,
        "decoded_job_log_git_blob": JOB_LOG_BLOB, "members": len(MEMBERS),
        "corrected_synthetic_med_ms": {
            str(r["synthetic_shape"][1]): r["median_forward_three_tap_ms"]
            for r in original_report["synthetic_cases"]
        },
        "new_physical_model_executed": False,
        "actual_matched_frozen_host_budget_pass": False,
        "train_authorized": False, "full_72_conformance_pass": False,
        "optimizer_steps": 0,
    }

if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
