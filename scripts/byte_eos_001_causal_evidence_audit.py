#!/usr/bin/env python3
"""BYTE-EOS-001 original Actions job-log evidence readback (no model/network).

This is a custody/integrity check for source-only serializer tests; it is NOT
independent scientific admission, Qwen3 runtime proof, or permission to train.
The two original Actions ZIPs remain tracked by digest but are not yet in Git.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/byte-eos-001/causal-shift-evidence"
SOURCE_HEAD = "9fbdee880697903aaf5b91cc29442c9e53479e50"
EXPECTED_RUNS = (40728812489, 40728952524)


def refuse(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def git_blob_sha(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data).hexdigest()


def audit(evidence: Path = EVIDENCE) -> dict:
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    refuse(manifest["status"] == "SOURCE_ONLY_PASS__FINAL_72_BLOCKED__NO_TRAINING",
           "False NCR admission or report status")
    refuse(manifest["source_pr"] == 163 and manifest["original_source_head"] == SOURCE_HEAD,
           "Original source PR or exact source head changed")
    refuse(manifest["parent_pr"] == 94 and
           manifest["parent_exact_head"] ==
           "b1c558f7d46f5d7c76b4e6336a37143886d16c08",
           "Reviewed parent changed")
    flags = manifest["source_claims"]
    for key in ("train_authorized", "model_executed", "heldout_opened",
                "comparative_training_performed", "full_72_conformance_pass",
                "independent_scientific_review"):
        refuse(flags[key] is False, "False scientific qualification: " + key)
    refuse(flags["optimizer_steps"] == 0, "Nonzero optimizer steps in source receipt")
    custody = manifest["evidence_storage"]
    refuse(custody["decoded_original_job_logs_committed"] is True and
           custody["raw_zip_contents_independently_sha256_verified"] is True and
           custody["raw_zip_bytes_committed"] is True,
           "Missing permanent original ZIP custody assertion")
    archive_path = evidence / "original-actions-zipset.tar.xz.base64"
    compressed = base64.b64decode(archive_path.read_text(encoding="ascii").strip(),
                                  validate=True)
    refuse(len(compressed) == custody["zipset_decoded_tar_xz_bytes"] and
           hashlib.sha256(compressed).hexdigest() == custody["zipset_decoded_tar_xz_sha256"],
           "Original raw ZIP-set TAR/XZ archive bytes changed")
    raw_zips = {}
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:xz") as bundle:
        for member in bundle:
            refuse(member.isfile() and not member.name.startswith("/") and
                   ".." not in member.name and member.size < 100000,
                   "Unsafe original ZIP archive member")
            refuse(member.name not in raw_zips, "Duplicate original ZIP container")
            raw_zips[member.name] = bundle.extractfile(member).read()
    expected_names = {
        f"byte-eos-001-causal-shift-run{n}.zip" for n in EXPECTED_RUNS
    }
    refuse(set(raw_zips) == expected_names,
           "Original raw ZIP container identity/count changed")
    runs = manifest["original_runs"]
    refuse(tuple(x["run_id"] for x in runs) == EXPECTED_RUNS,
           "Original runner identities/order changed")
    for n, run in enumerate(runs):
        path = evidence / f'run-{run["run_id"]}-full-job.log'
        raw = path.read_bytes()
        refuse(git_blob_sha(raw) == run["full_log_git_blob"],
               "Original decoded full job log Git blob changed")
        log = raw.decode("utf-8")
        for witness in ("Ran 16 tests in", "\n", "Locked protocol, sources and original receipt",
                        f'HEAD={run["source_head"]}', '"result": "success"',
                        '"train_authorized": false', f'"run_id": "{run["run_id"]}"',
                        f'"runner_name": "{run["runner"]}"'):
            refuse(witness in log, f'Original job {run["run_id"]} missing source witness: {witness}')
        refuse(run["tests_passed"] == 16 and run["result"] == "success",
               "Original tests no longer a bounded positive witness")
        zip_bytes = raw_zips[f'byte-eos-001-causal-shift-run{run["run_id"]}.zip']
        refuse(hashlib.sha256(zip_bytes).hexdigest() == run["artifact_sha256"],
               "Original GitHub Actions ZIP byte digest mismatch")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            refuse(sorted(z.namelist()) == ["receipt.json", "test.log"],
                   "Unexpected original ZIP contents")
            member_log = z.read("test.log")
            member_receipt = z.read("receipt.json")
        refuse(hashlib.sha256(member_log).hexdigest() == run["test_log_sha256"] and
               hashlib.sha256(member_receipt).hexdigest() == run["artifact_receipt_sha256"],
               "Raw original ZIP member changed or tampered")
        original = json.loads(member_receipt)
        refuse(original["run_id"] == str(run["run_id"]) and
               original["runner_name"] == run["runner"] and
               original["expected_pr_head"] == run["source_head"] and
               original["result"] == "success" and
               original["train_authorized"] is False and
               original["optimizer_steps"] == 0 and
               original["full_72_conformance_pass"] is False and
               original["sha256"]["test_log"] == run["test_log_sha256"],
               "Original raw Actions receipt false-admission or identity mismatch")
        refuse(len(run["artifact_sha256"]) == 64 and
               len(run["artifact_receipt_sha256"]) == 64 and
               len(run["test_log_sha256"]) == 64,
               "Original artifact integrity hashes missing")
        if n == 0:
            anomaly = run["provenance_anomaly"]
            refuse(anomaly["kind"] == "GITHUB_SHA_MISLABELED_AS_SOURCE_HEAD" and
                   anomaly["receipt_head_sha"] != run["source_head"] and
                   anomaly["expected_checkout_head"] == run["source_head"],
                   "Historical wrong-head field must remain a documented negative")
            refuse(f'"head_sha": "{anomaly["receipt_head_sha"]}"' in log,
                   "Historical wrong-head receipt not retained")
        else:
            refuse(run["head_matches_checkout"] is True and
                   run["source_head"] == SOURCE_HEAD,
                   "Corrected actual checkout-head receipt missing")
            refuse(f'"head_sha": "{SOURCE_HEAD}"' in log and
                   f'"expected_pr_head": "{SOURCE_HEAD}"' in log,
                   "Corrected exact-source head not present")
    return {
        "status": "PASS_ORIGINAL_LOG_AND_RAW_ZIP_BYTE_READBACK__NO_MODEL",
        "source_pr": 163,
        "original_source_head": SOURCE_HEAD,
        "original_runs": list(EXPECTED_RUNS),
        "all_16_test_checks_each_run": True,
        "historical_provenance_anomaly_preserved": True,
        "raw_actions_zip_bytes_in_git": True,
        "final_72_conformance_pass": False,
        "train_authorized": False,
    }


if __name__ == "__main__":
    try:
        result = audit()
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"FAIL_ORIGINAL_LOG_READBACK: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
