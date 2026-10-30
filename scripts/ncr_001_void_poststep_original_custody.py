#!/usr/bin/env python3
"""Original #210 frozen Qwen3 physical ZIP/job-log custody, read-only, no model.

Fail closed on literal Git tree object identities, raw bytes, original ZIP
members and the exact public-synthetic post-step cold/KV scope. Does NOT
qualify four-model #72 science, EOS generation, matched quality or training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import zipfile

import ncr_001_void_poststep_bootstrap_custody as acquisition

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/void-poststep-cold-20261030"
MANIFEST_BLOB = "3b56c4b3e05da323e65cc3a187e5d2d320a7a007"
FILES = {
    "source-original.zip": (
        369, "3313913ce5dd5e28f4f89f777f3e9d99271c894ee301e495b311a9b3adca992b",
        "79380c50def07d6720d0928860e440e12008bea7"),
    "host-original.zip": (
        6770, "75df0cc9733d4873516880ebe2ce449cc5be4d79da03166ee408798dad9ce64b",
        "30262254a4f1ce2ef8361ae89a578f49333f9788"),
    "source-original-job.log": (
        22795, "121779b202b1b505deb81ca9db4082a117b2424401e4162e224529e582d66df5",
        "548ca750c0950b4683c128a0c7df709d019e69df"),
    "host-original-job.log": (
        53480, "8b3c6a826ebc16d22a264e8b9ebd039331d76da6995da4f4a61e4b60e2582e0a",
        "4c2639e300567088efab3a5eef8c2e62a0a9ba83"),
}
EXPECTED_RUN = 41058648266
EXPECTED_HEAD = "f78d302ab6062c3739315c3e04317a9a97b97191"
BOUNDARY = {
    "actual_compute_budget_qualified": False,
    "comparative_training": False,
    "full_72_conformance_pass": False,
    "heldouts_opened": False,
    "historical_train002b_eos_effect_measured": False,
    "per_model_gate": False,
    "train_authorized": False,
}


def refuse(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError("void1 original poststep custody REFUSED: " + why)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def git_entry(path: Path, expected: str) -> None:
    rel = path.relative_to(ROOT).as_posix()
    res = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "HEAD", "--", rel],
                         capture_output=True, text=True, check=False)
    refuse(res.returncode == 0 and
           res.stdout == f"100644 blob {expected}\t{rel}\n",
           "committed Git tree identity/mode drift: " + rel)


def read_and_check(evidence: Path = EVIDENCE, *, verify_git: bool = False) -> tuple[dict, dict]:
    path = evidence / "manifest.json"
    refuse(path.is_file() and not path.is_symlink(), "missing original manifest")
    raw_manifest = path.read_bytes()
    refuse(blob(raw_manifest) == MANIFEST_BLOB, "original manifest Git blob drift")
    if verify_git:
        git_entry(path, MANIFEST_BLOB)
    manifest = json.loads(raw_manifest)
    refuse(manifest["schema"] == "ncr001-void-poststep-cold-original-custody-v1"
           and manifest["run"] == EXPECTED_RUN
           and manifest["source_head"] == EXPECTED_HEAD
           and manifest["attempt"] == 1
           and manifest["repository"] == "techrote/neurounner"
           and manifest["source_only_run_job_success"] is True
           and manifest["physical_host_run_job_success"] is True
           and manifest["retention_of_original_zips_and_logs"] == "permanent_git_bytes"
           and manifest["scientific_boundary"] == BOUNDARY
           and set(manifest["files"]) == set(FILES),
           "original manifest scope or training/holdout gate changed")
    refuse(manifest["original"]["source"]["original_job"] == 124072474714
           and manifest["original"]["host"]["original_job"] == 124072552187
           and manifest["original"]["source"]["original_artifact"] == 12976819289
           and manifest["original"]["host"]["original_artifact"] == 12977852817,
           "original Actions job/artifact provenance substituted")
    originals: dict[str, bytes] = {}
    for name, (size, digest, git_sha) in FILES.items():
        f = evidence / name
        refuse(f.is_file() and not f.is_symlink(), "missing or linked original stream: " + name)
        raw = f.read_bytes()
        refuse(len(raw) == size and sha(raw) == digest and blob(raw) == git_sha
               and manifest["files"][name] ==
               {"size": size, "sha256": digest, "git_blob": git_sha},
               "original whole-file ZIP/job-log bytes changed: " + name)
        if verify_git:
            git_entry(f, git_sha)
        originals[name] = raw
    for kind in ("source", "host"):
        original = manifest["original"][kind]
        refuse(original["zip_sha256"] == FILES[kind + "-original.zip"][1]
               and original["zip_bytes"] == FILES[kind + "-original.zip"][0]
               and original["decoded_log_sha256"] == FILES[kind + "-original-job.log"][1]
               and original["decoded_log_bytes"] == FILES[kind + "-original-job.log"][0],
               "manifest original stream identity disagreement")
        with zipfile.ZipFile(io.BytesIO(originals[kind + "-original.zip"])) as archive:
            refuse(archive.testzip() is None, "original ZIP CRC failure")
            members = {
                name: {"bytes": len(archive.read(name)),
                       "sha256": sha(archive.read(name))}
                for name in archive.namelist()
            }
            refuse(members == manifest["members"][kind],
                   "original ZIP member contents replaced: " + kind)
            if kind == "source":
                refuse(set(members) == {"tests.log"}
                       and b"Ran 4 tests" in archive.read("tests.log")
                       and b"OK" in archive.read("tests.log"),
                       "four exact-head source sentinels absent")
            else:
                refuse(len(members) == 7 and not any(name.endswith(
                           ("adapter.pt", "metadata.json")) for name in members),
                       "unlisted/missing physical source member or persisted trained adapter")
                producer = json.loads(archive.read(
                    "ncr001-void-poststep-cold/producer.json"))
                consumer = json.loads(archive.read(
                    "ncr001-void-poststep-cold/consumer.json"))
                diag = json.loads(archive.read(
                    "ncr001-void-nca-learning/receipt.json"))
    status = "POSTSTEP_COLD_DIAGNOSTIC_ONLY__SCIENTIFIC_72_BLOCKED"
    same_sample = producer["sample"] == consumer["sample"]
    # The original real model, original EOS serializer and original cold process
    # generated these exact tensor IDs, labels and mask digests. Compare the
    # three committed receipts to each other instead of guessing literal hashes.
    # Both complete ZIP and individual receipt members are already SHA-pinned.
    expected_sample = diag["synthetic_sample"]
    refuse(isinstance(expected_sample, dict) and
           producer["sample"] == consumer["sample"] == expected_sample and
           all(isinstance(expected_sample.get(k), str) and
               len(expected_sample[k]) == 64 and
               all(c in "0123456789abcdef" for c in expected_sample[k])
               for k in ("ids_sha256", "labels_sha256", "mask_sha256")),
           "original diagnostic/producer/consumer token-label-mask witness drift")
    refuse(producer["status"] == consumer["status"] == status and
           producer["phase"] == "producer" and consumer["phase"] == "consumer" and
           type(producer["diagnostic_optimizer_steps"]) is int and
           producer["diagnostic_optimizer_steps"] == 1 and
           type(consumer["diagnostic_optimizer_steps"]) is int and
           consumer["diagnostic_optimizer_steps"] == 0 and
           type(diag["diagnostic_optimizer_steps"]) is int and
           diag["diagnostic_optimizer_steps"] == 1 and
           # The reused original #172 function was invoked from #210 with
           # this explicit NEW_DIAGNOSTIC marker; this is NOT original #172
           # physical evidence and must never be relabelled as such.
           diag["status"] == "NEW_DIAGNOSTIC_NOT_ORIGINAL_172" and
           # The original one-step routine writes its last on-disk receipt
           # immediately after the second backward. Its final in-memory
           # checks are verified by the producer but not written again.
           # Validate the *actual archived partial* receipt, not fictitious
           # persisted checks that never existed in the original artifact.
           isinstance(diag.get("post_warmup"), dict) and
           isinstance(diag["post_warmup"].get("gradient_l1"), dict) and
           all(isinstance(diag["post_warmup"]["gradient_l1"].get(name), (int, float)) and
               math.isfinite(diag["post_warmup"]["gradient_l1"][name]) and
               diag["post_warmup"]["gradient_l1"][name] > 0
               for name in ("readout.weight", "observe.weight",
                            "perception.weight", "rule.weight",
                            "input_norm.weight")) and
           isinstance(diag["post_warmup"].get("ordered_field_trace"), list) and
           len(diag["post_warmup"]["ordered_field_trace"]) == 3 and
           isinstance(diag["post_warmup"].get("native_causal_eos_loss"), (int, float)) and
           math.isfinite(diag["post_warmup"]["native_causal_eos_loss"]) and
           diag["post_warmup"]["native_causal_eos_loss"] > 0 and
           same_sample and producer["sample"]["real_eos_id"] == 258 and
           producer["sample"]["total_units"] == 31 and
           producer["sample"]["supervised_after_shift"] == 21 and
           producer["sample"] == expected_sample,
           "original diagnostic step or exact public shifted-EOS witness changed")
    refuse(producer["frozen_backbone_sha256"] ==
           consumer["frozen_backbone_sha256"] ==
           manifest["original"]["physical_result"]["frozen_backbone_sha256"] and
           producer["active_logits_sha256"] == consumer["active_logits_sha256"] and
           producer["producer_pid"] == consumer["producer_pid"] and
           producer["producer_pid"] != consumer["cold_process_pid"] and
           producer["threads_effective_after_load"] ==
           consumer["threads_effective_after_load"] == 4 and
           consumer["post_step_cold_reload_exact"] is True and
           consumer["per_request_aba_exact"] is True and
           type(consumer["incremental_tokens"]) is int and
           consumer["incremental_tokens"] == 4 and
           type(consumer["context_units"]) is int and
           consumer["context_units"] == 31 and
           math.isfinite(consumer["post_step_last_four_true_kv_max_abs"]) and
           0 <= consumer["post_step_last_four_true_kv_max_abs"] <= 5e-4 and
           consumer["post_step_last_four_true_kv_max_abs"] ==
           manifest["original"]["physical_result"]["post_step_last_four_true_kv_max_abs"],
           "poststep cold Qwen3 identity/KV boundary substituted")
    refuse(all(producer[k] is False and consumer[k] is False
               for k in ("train_authorized", "comparative_training_performed",
                         "heldouts_opened", "full_72_conformance_pass",
                         "independent_scientific_review")),
           "comparative training/science gate falsely promoted")
    report = {
        "status": "PASS_ORIGINAL_POSTSTEP_COLD_BYTES__SCIENTIFIC_72_BLOCKED",
        "original_run": EXPECTED_RUN, "original_head": EXPECTED_HEAD,
        "source_and_host_jobs": [124072474714, 124072552187],
        "original_zips_and_logs": 4, "original_zip_members": 8,
        "source_tests": 4, "effective_threads": 4, "synthetic_nca_steps": 1,
        "cache_checked_tail_tokens": 4, "context_units": 31,
        "poststep_cache_max_abs":
            consumer["post_step_last_four_true_kv_max_abs"],
        "separate_process_cold_adapter": True,
        "train_authorized": False, "full_72_conformance_pass": False,
        "per_model_gate": False, "actual_compute_budget_qualified": False,
        "original_training_impact_measured": False,
        "scientific_review_complete": False, "fresh_physical_execution": False,
    }
    return report, originals


def audit(*, live: bool = False) -> dict:
    report, originals = read_and_check(verify_git=True)
    if live:
        fetched, raw = acquisition.audit_originals()
        manifest = json.loads((EVIDENCE / "manifest.json").read_text())
        refuse(fetched == manifest and
               all(raw[name] == originals[name] for name in FILES),
               "fresh re-download is not identical to original committed ZIP/log")
        report["original_ledger_freshly_rechecked"] = True
    else:
        report["original_ledger_freshly_rechecked"] = False
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(live=args.live), sort_keys=True))
