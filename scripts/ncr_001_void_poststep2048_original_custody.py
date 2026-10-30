#!/usr/bin/env python3
"""Read-only permanent custody audit for original poststep2048 Actions bytes.

No model, tokenizer, optimizer, heldout or adapter state is loaded. Literal Git
tree blob/mode identities, complete original ZIPs/logs, original ZIP members,
and exact physical semantics are required. --live additionally re-downloads
the same original Actions ZIP/log streams and requires byte equality.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import ncr_001_void_poststep2048_bootstrap_custody as acquisition

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/void-poststep2048-20261030"
MANIFEST = EVIDENCE / "manifest.json"
MANIFEST_BLOB = "462a43cf3e56b5edefe9dcf14ffc6dae199a3158"
EXPECTED_HEAD = "520c4ae5b3152aff8d94760dab2e3103746f7334"
EXPECTED_RUN = 41133450850
FILES = {
    "source-original.zip": (
        558,
        "024ed9a0ac0d89338ef249ba79b3ebb4a20a7659c714fe2e557dd055379cc806",
        "1bbace701bea59969854cbdf50fcb0444d2aba4c",
    ),
    "host-original.zip": (
        10002,
        "db66de004d26a6c24539b08898f85df88b3be7631ba300655a82bd021de71145",
        "e164d792be74857d6f7a2384cc316bd7631162a0",
    ),
    "source-original-job.log": (
        52537,
        "c9ebe268baca70c7f3a79eff29f74a07943dbd096bf50e97d07b4cb3814bd8ea",
        "55faf7ca98b0529f4f713087bc0dcaa91bbd2d1f",
    ),
    "host-original-job.log": (
        82834,
        "5cff4684863fc505a752d779b60a884e91327627b99dcea0e65da50d4f263d75",
        "a062dbfed90d643cca0162ebe446daf64e6ee5f2",
    ),
}
BOUNDARY = {
    "actual_compute_budget_qualified": False,
    "comparative_training": False,
    "full_72_conformance_pass": False,
    "generated_eos_skill_qualified": False,
    "heldouts_opened": False,
    "historical_train002b_eos_effect_measured": False,
    "independent_scientific_review": False,
    "per_model_gate": False,
    "train_authorized": False,
}


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("poststep2048 custody REFUSED: " + reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def git_entry(path: Path, expected: str) -> None:
    rel = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "HEAD", "--", rel],
        capture_output=True, text=True, check=False,
    )
    refuse(
        result.returncode == 0
        and result.stdout == f"100644 blob {expected}\t{rel}\n",
        "committed Git tree mode/type/blob changed: " + rel,
    )


def read_and_check(*, verify_git: bool = True) -> tuple[dict, dict[str, bytes]]:
    refuse(MANIFEST.is_file() and not MANIFEST.is_symlink(),
           "manifest missing or symlink")
    manifest_raw = MANIFEST.read_bytes()
    refuse(blob(manifest_raw) == MANIFEST_BLOB,
           "literal manifest Git blob changed")
    if verify_git:
        git_entry(MANIFEST, MANIFEST_BLOB)
    manifest = json.loads(manifest_raw)
    refuse(
        manifest["schema"] == "ncr001-void-poststep2048-original-custody-v1"
        and manifest["repository"] == "techrote/neurounner"
        and manifest["run"] == EXPECTED_RUN
        and manifest["attempt"] == 1
        and manifest["source_head"] == EXPECTED_HEAD
        and manifest["source_branch"] ==
            "review/ncr001e-void-poststep2048-20261030"
        and manifest["source_only_job_success"] is True
        and manifest["physical_host_job_success"] is True
        and manifest["retention_of_original_zips_and_logs"] ==
            "permanent_git_bytes"
        and manifest["scientific_boundary"] == BOUNDARY
        and set(manifest["files"]) == set(FILES),
        "manifest source/retention/scientific boundary changed",
    )
    originals: dict[str, bytes] = {}
    for name, (size, digest, git_sha) in FILES.items():
        path = EVIDENCE / name
        refuse(path.is_file() and not path.is_symlink(),
               "missing or linked original: " + name)
        raw = path.read_bytes()
        refuse(
            len(raw) == size and sha(raw) == digest and blob(raw) == git_sha
            and manifest["files"][name] ==
                {"size": size, "sha256": digest, "git_blob": git_sha},
            "literal original whole stream changed: " + name,
        )
        if verify_git:
            git_entry(path, git_sha)
        originals[name] = raw

    for kind in ("source", "host"):
        meta = manifest["original"][kind]
        refuse(
            meta["zip_bytes"] == FILES[kind + "-original.zip"][0]
            and meta["zip_sha256"] == FILES[kind + "-original.zip"][1]
            and meta["decoded_log_bytes"] ==
                FILES[kind + "-original-job.log"][0]
            and meta["decoded_log_sha256"] ==
                FILES[kind + "-original-job.log"][1],
            "manifest stream metadata changed: " + kind,
        )
        with zipfile.ZipFile(io.BytesIO(
                originals[kind + "-original.zip"])) as archive:
            refuse(archive.testzip() is None, "original ZIP CRC: " + kind)
            members = {
                member: {
                    "bytes": len(archive.read(member)),
                    "sha256": sha(archive.read(member)),
                }
                for member in archive.namelist()
            }
            refuse(members == manifest["members"][kind],
                   "original ZIP member inventory/content drift: " + kind)
            if kind == "source":
                refuse(
                    set(members) == {"tests.log"}
                    and b"Ran 10 tests" in archive.read("tests.log")
                    and b"OK" in archive.read("tests.log"),
                    "source 10-test original proof missing",
                )
            else:
                refuse(
                    len(members) == 8
                    and "ncr001-void-poststep2048/producer.json" in members
                    and "ncr001-void-poststep2048/consumer.json" in members
                    and "ncr001-void-poststep2048/runner.txt" in members
                    and not any(name.endswith(
                        (".pt", ".safetensors", "metadata.json"))
                        for name in members),
                    "physical member set incomplete or adapter persisted",
                )
                producer = json.loads(archive.read(
                    "ncr001-void-poststep2048/producer.json"))
                consumer = json.loads(archive.read(
                    "ncr001-void-poststep2048/consumer.json"))
                derived = acquisition.require_semantics(producer, consumer)
                refuse(derived == manifest["original"]["physical_result"],
                       "primary physical semantic derivation changed")

    source_log = originals["source-original-job.log"]
    host_log = originals["host-original-job.log"]
    refuse(
        EXPECTED_HEAD.encode() in source_log
        and b"Ran 10 tests" in source_log and b"OK" in source_log
        and b"PASS_BOUNDED_POSTSTEP2048_SOURCE_DIFF" in source_log,
        "complete original source decoded-log evidence absent",
    )
    refuse(
        EXPECTED_HEAD.encode() in host_log
        and b"SOURCE_SHA256 model.safetensors " +
            acquisition.WEIGHTS.encode() in host_log
        and b"PASS_POSTSTEP2048_CACHE_MASK_SUFFIX_GENERATE" in host_log
        and b'"incremental_positions": 2048' in host_log,
        "complete original host decoded-log evidence absent",
    )
    report = {
        "status":
            "PASS_ORIGINAL_POSTSTEP2048_COMPLETE_BYTES__SCIENTIFIC_72_UNADMITTED",
        "original_run": EXPECTED_RUN,
        "original_head": EXPECTED_HEAD,
        "source_job": manifest["original"]["source"]["original_job"],
        "host_job": manifest["original"]["host"]["original_job"],
        "original_zips_and_logs": 4,
        "original_zip_members": 9,
        "physical_result": manifest["original"]["physical_result"],
        "permanent_git_custody": True,
        "fresh_physical_execution": False,
        "train_authorized": False,
        "per_model_gate": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "comparative_training": False,
        "heldouts_opened": False,
        "actual_compute_budget_qualified": False,
        "generated_eos_skill_qualified": False,
        "historical_train002b_eos_effect_measured": False,
    }
    return report, originals


def audit(*, live: bool = False) -> dict:
    report, originals = read_and_check(verify_git=True)
    if live:
        manifest, raw = acquisition.audit_originals()
        committed_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        refuse(
            manifest == committed_manifest
            and all(raw[name] == originals[name] for name in FILES),
            "fresh GitHub original streams differ from permanent Git bytes",
        )
        report["original_ledger_freshly_rechecked"] = True
    else:
        report["original_ledger_freshly_rechecked"] = False
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(live=args.live), sort_keys=True))
