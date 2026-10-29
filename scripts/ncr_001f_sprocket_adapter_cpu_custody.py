#!/usr/bin/env python3
"""Read-only byte-exact audit of original #40867677298 CPU adapter-only receipt.

No PyTorch, frozen model, tokenizer, optimizer or held-out execution.
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
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/sprocket-adapter-cpu-microbench-20261029"
ORIGINAL = "original-actions-40867677298.zip"
ORIGINAL_SIZE = 5589
ORIGINAL_SHA256 = "d5bad11d2e28a6b40aa9717046464f9436a5e07ef800559f8044e494c826eea0"
ORIGINAL_GIT_BLOB = "38a428b1a0fae2c7b9163f200a2b83f37f369add"
SOURCE_HEAD = "1f44c775c392cea430457c86253c1f00deea4e70"
RUN_ID = "40867677298"
MEMBERS = {
    "lscpu.txt": (3189, "4476e601951bb5e5dc85ca4b9d2e3bc795bb83c0d4e62b8d9f2bdec6659b0e7a"),
    "pip-freeze.txt": (1551, "ea4b616e5ccd392060001737203319bf998f7cc3308f775e18289b02a7ebe050"),
    "report.json": (5282, "efd79fb2419c7d8ee98e3b4be4d24bce79b74bc323858f644ed4bafb8603def4"),
    "runner.log": (307, "9adf3a63a06a7d7d6ad7c016b7365dc3a14ecb8c6d90832a341216075786807b"),
    "source-head.txt": (41, "f12bec081902a93826dc95c49bb7317814b99cc23589e385356ce815d8ad5b24"),
    "tests.log": (1718, "e45cffb4274d9135cd3cd7c1bb00331eba7831dc448d52a1c550288a74307eb2"),
    "uname.txt": (128, "a64b6d81308dc728b286592ae78f6594e185e6db8c474be3c8886f7694f38bb7"),
}
SOURCES = {
    "sprocket_chat": ("chat/model.safetensors", "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
    "sprocket_instruct": ("model.safetensors", "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"),
}


def refuse(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("NCR001F_ORIGINAL_CPU_CUSTODY_REFUSED: " + message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def audit_zip(raw: bytes) -> dict:
    refuse(len(raw) == ORIGINAL_SIZE and sha(raw) == ORIGINAL_SHA256
           and gitblob(raw) == ORIGINAL_GIT_BLOB,
           "complete original Actions ZIP container changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        refuse(len(names) == len(MEMBERS) == len(set(names))
               and set(names) == set(MEMBERS), "missing or extra original member")
        for name, (size, digest) in MEMBERS.items():
            info = z.getinfo(name)
            refuse(not info.is_dir() and name == Path(name).name,
                   "unsafe archived file member")
            member = z.read(name)
            refuse(len(member) == info.file_size == size
                   and sha(member) == digest,
                   "changed archived original member: " + name)
        report = json.loads(z.read("report.json"))
        source_head = z.read("source-head.txt")
        tests_log = z.read("tests.log").decode("utf-8")
        runner_log = z.read("runner.log").decode("utf-8")
    refuse(source_head == (SOURCE_HEAD + "\n").encode(),
           "source Git head member changed")
    # The original runner.log contains a real PyTorch missing-NumPy warning,
    # not the workflow's later success sentinel. Preserve that negative as-is.
    refuse("Ran 7 tests" in tests_log and "\nOK" in tests_log
           and "Failed to initialize NumPy: No module named 'numpy'" in runner_log,
           "original tests or runner warning changed")
    refuse(report["status"] == "ADAPTER_ONLY_CPU_FORWARD_MICROBENCH__FROZEN_HOST_NOT_EXECUTED__NO_TRAIN"
           and report["experiment"] == "NCR-001F" and report["issue"] == 73
           and report["seed"] == 20261029 and report["optimizer_steps"] == 0
           and report["adapter_only_proxy_not_model_latency"] is True
           and report["model_free_shared_geometry"]["nca_over_mlp_mac_ratio"] == 1.20625,
           "unadmitted CPU proxy classification/geometry changed")
    false_flags = ("train_authorized", "full_72_conformance_pass",
                   "training_performed", "heldouts_opened",
                   "frozen_model_weights_downloaded", "frozen_model_inference_executed",
                   "actual_matched_frozen_host_budget_pass", "tokenizers_executed")
    refuse(all(report[k] is False for k in false_flags),
           "forged training, inference, heldout or scientific admission")
    info = report["runner"]
    refuse(info["github_sha"] == SOURCE_HEAD
           and info["github_run_id"] == RUN_ID
           and info["github_run_attempt"] == "1"
           and info["github_job"] == "adapter-only-no-model"
           and info["torch"] == "2.14.1+cpu"
           and info["torch_num_threads"] == 4
           and report["peak_process_rss_kib"] > 0,
           "runner/process/torch source identity changed")
    lanes = report["source_variants_verified_not_executed"]
    refuse(set(lanes) == set(SOURCES), "Sprocket variants collapsed or extra lane")
    for lane, (weight, digest) in SOURCES.items():
        refuse(lanes[lane] == {
            "source_revision": "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
            "weight_path": weight, "weight_sha256": digest,
        }, "frozen Sprocket identity changed: " + lane)
    cases = report["synthetic_cases"]
    refuse(len(cases) == 3 and [x["synthetic_shape"][1] for x in cases] == [1,16,64],
           "unpreregistered synthetic length or missing data")
    for row in cases:
        refuse(row["synthetic_shape"][0] == 1 and row["synthetic_shape"][2] == 1280
               and row["tap_count"] == 3 and row["nca_steps_per_tap"] == 3
               and row["warmup_rounds_per_adapter"] == 2
               and row["measured_rounds_per_adapter"] == 5
               and row["includes_frozen_transformer"] is False
               and row["includes_backward_or_optimizer"] is False,
               "changed original synthetic adapter budget")
        for lane in ("nca", "mlp"):
            raw_times = row["raw_forward_three_tap_ms"][lane]
            med = row["median_forward_three_tap_ms"][lane]
            refuse(len(raw_times) == 5 and all(
                type(t) in (int, float) and math.isfinite(t) and t > 0
                for t in raw_times)
                and abs(statistics.median(raw_times) - med) <= 1e-6,
                "missing/invalid original timing series or median")
        ratio = (row["median_forward_three_tap_ms"]["nca"]
                 / row["median_forward_three_tap_ms"]["mlp"])
        refuse(abs(ratio - row["median_nca_over_mlp_wall_ratio"]) < 1e-6,
               "forged original timing ratio")
    return {
        "status": "PASS_ORIGINAL_COMPLETE_CPU_ADAPTER_ZIP_IN_GIT__FROZEN_HOST_UNMEASURED",
        "original_run": int(RUN_ID),
        "original_source_head": SOURCE_HEAD,
        "original_zip_sha256": ORIGINAL_SHA256,
        "original_zip_git_blob": ORIGINAL_GIT_BLOB,
        "original_zip_bytes": ORIGINAL_SIZE,
        "original_member_count": len(MEMBERS),
        "source_variants_distinct_not_executed": list(SOURCES),
        "measured_adapter_only_lengths": [1,16,64],
        "median_three_tap_forward_ms": {
            str(row["synthetic_shape"][1]): row["median_forward_three_tap_ms"]
            for row in cases
        },
        "actual_matched_frozen_host_budget_pass": False,
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "new_frozen_model_execution": False,
        "optimizer_steps": 0,
    }


def audit(evidence: Path = EVIDENCE) -> dict:
    return audit_zip((evidence / ORIGINAL).read_bytes())


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
