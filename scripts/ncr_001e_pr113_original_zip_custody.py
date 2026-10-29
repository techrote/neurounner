#!/usr/bin/env python3
"""Exact original PR #113 ZIP and GitHub-decoded job-log custody; no model."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import ncr_001e_pr113_rightpad_receipt_audit as prior

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review/pr113-rightpad-generation"
MANIFEST = "original-actions-custody.json"
MANIFEST_GIT_BLOB = "ecff0787c849428e788d3d8a2401792c2c8b12c2"
ORIGINAL_ZIP = "original-actions-40598034739.zip"
ZIP_SHA256 = "30bc7fa3231047561b5ab0142b59b7012e7e1ee4e72329198f7d778b65be9a1e"
ZIP_GIT_BLOB = "6e231bf8cfc909bb943ab3807cc4d2b4f7ad5413"
ZIP_BYTES = 5754
LOG_BLOBS = {
    "original-source-job-122518832102.log": "072ad20c4be7622f31d252bf836e5e1ab26fc269",
    "original-physical-job-122518920068.log": "c8e2846a1539a4aaede774861c9a41429c1d6394",
}
MEMBERS = {
    "pr113-rightpad-generate.json": (13952, "c50bc096161a2b0cfe40c19278adada101cb7314bff822064117ae347f232224"),
    "pr113-rightpad-packages.txt": (1714, "38376d0be1f9ec1d5b922c7c5fa493dd969f01ddc2453cff6cc94ffbe4f58f9f"),
    "pr113-rightpad-prefetch.json": (1056, "e121751fb3299154830b452e9d01e1ced59ccb1928d7b8230441663d0ff41918"),
    "pr113-rightpad-source-commit.txt": (41, "30196dc5793cd6f3e727168110e4fd5a2c82950dc1e74bcef972416eee8ada3a"),
    "pr113-rightpad-source-hashes.sha256": (445, "f9902886d0487f1bb9c79b932b5b2b198092f2f0f9e743a440f59dbfefd481bc"),
}

def refuse(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("NCR001_PR113_ORIGINAL_CUSTODY_REFUSED: " + why)

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def gitblob(b: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(b)).encode("ascii") + b"\x00" + b).hexdigest()

def check_zip(raw: bytes) -> dict[str, bytes]:
    refuse(len(raw) == ZIP_BYTES and sha256(raw) == ZIP_SHA256
           and gitblob(raw) == ZIP_GIT_BLOB,
           "complete original Actions ZIP container bytes changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        refuse(len(names) == len(MEMBERS) == len(set(names))
               and set(names) == set(MEMBERS),
               "missing, duplicated or unexpected original ZIP member")
        found = {}
        for name in names:
            info = z.getinfo(name)
            refuse(not info.is_dir() and name == Path(name).name,
                   "original ZIP unsafe member")
            contents = z.read(name)
            size, digest = MEMBERS[name]
            refuse(info.file_size == len(contents) == size
                   and sha256(contents) == digest,
                   "original member byte/hash mismatch: " + name)
            found[name] = contents
    return found

def audit(evidence: Path = EVIDENCE) -> dict:
    raw_manifest = (evidence / MANIFEST).read_bytes()
    refuse(gitblob(raw_manifest) == MANIFEST_GIT_BLOB,
           "committed original manifest blob altered")
    manifest = json.loads(raw_manifest)
    refuse(manifest["train_authorized"] is False
           and manifest["full_72_conformance_pass"] is False
           and manifest["optimizer_steps"] == 0
           and manifest["independent_scientific_admission"] is False
           and manifest["generated_tool_memory_competence"] is False
           and manifest["original_root_six_token_incomplete_prefix_retained"] is True
           and manifest["original_failed_runs_preserved"] == [40597707675,40597891619],
           "scientific status or historical negative results changed")
    refuse(manifest["original_run"] == 40598034739
           and manifest["original_artifact_id"] == 12777685787
           and manifest["source_head"] == prior.ORIGINAL_HEAD
           and manifest["reviewed_pr_102_head"] == prior.REVIEWED_102,
           "original run and source identity changed")
    refuse(manifest["original_zip"] == {
        "path": ORIGINAL_ZIP, "bytes": ZIP_BYTES,
        "sha256": ZIP_SHA256, "git_blob": ZIP_GIT_BLOB,
    }, "original ZIP manifest mismatch")
    for name, blob in LOG_BLOBS.items():
        raw = (evidence / name).read_bytes()
        refuse(gitblob(raw) == blob, "complete decoded original job log modified: " + name)
    raw_zip = (evidence / ORIGINAL_ZIP).read_bytes()
    found = check_zip(raw_zip)
    original_members, original_manifest = prior.load_originals()
    prior.verify_originals(original_members, original_manifest)
    refuse(found == original_members,
           "complete original ZIP members disagree with separately archived text originals")
    refuse(set(manifest["members"]) == set(MEMBERS)
           and all(manifest["members"][n] == {
             "bytes": info[0], "sha256": info[1]
           } for n,info in MEMBERS.items()),
           "manifest five-member source inventory altered")
    refuse(manifest["original_jobs"]["source"]["git_blob"] ==
           LOG_BLOBS["original-source-job-122518832102.log"]
           and manifest["original_jobs"]["physical"]["git_blob"] ==
           LOG_BLOBS["original-physical-job-122518920068.log"],
           "original job ledger changed")
    return {
        "status": "PASS_ORIGINAL_PR113_COMPLETE_ZIP_AND_BOTH_JOB_LOGS__72_BLOCKED",
        "source_pr": 113, "source_head": prior.ORIGINAL_HEAD,
        "original_run": 40598034739, "original_zip_sha256": ZIP_SHA256,
        "original_zip_git_blob": ZIP_GIT_BLOB, "original_members": len(found),
        "original_decoded_job_logs": len(LOG_BLOBS),
        "new_model_execution": False, "optimizer_steps": 0,
        "train_authorized": False, "full_72_conformance_pass": False,
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-external-zip", type=Path)
    args = p.parse_args()
    if args.verify_external_zip:
        check_zip(args.verify_external_zip.read_bytes())
        print("PASS_ORIGINAL_PR113_EXTERNAL_ACTIONS_ZIP_VERIFIED")
    else:
        print(json.dumps(audit(), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
