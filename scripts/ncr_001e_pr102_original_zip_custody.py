#!/usr/bin/env python3
"""Read-only byte-exact custody of original PR102 Actions ZIP, source TAR and job log. NO MODEL."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import ncr_001e_sprocket_wire_archive as prior

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "experiments/ncr-001/evidence-review/pr102-publisher-original"
MANIFEST = FOLDER / "original-custody.json"
ORIGINAL_ZIP = FOLDER / "original-actions-40573188117.zip"
ORIGINAL_TAR = FOLDER / "original-source-snapshot-40573188117.tar.gz"
ORIGINAL_LOG = FOLDER / "original-job-122432551094.log"
MANIFEST_BLOB = "d5d5803e3bef4c4363a67de97f05d48dea3e02aa"
ZIP_SHA = "7e38a9c8438701c460597fc87016f3a26ece2d12b97482cfbcb44b0e6974a2ab"
ZIP_BLOB = "4a54a61b56ad0dc326082a157974c0737f28a294"
TAR_SHA = "bb6109524362c8be1b19b420e7c646de4cc9a685ca7157e6afa55e0c203ae961"
TAR_BLOB = "f56ae18f10afbe1ae87f579f422ba8d302a8d25a"
# Independently pinned complete GitHub-decoded original job log Git blob; original log text, not HTTP log transport ZIP.
ORIGINAL_LOG_BLOB = "7a377e6cd90ae2de16a3eacc8cc20da120ec3272"

def refuse(ok: bool, why: str):
    if not ok:
        raise ValueError("NCR001_PR102_CUSTODY_REFUSED: " + why)

def sha(raw: bytes):
    return hashlib.sha256(raw).hexdigest()

def blob(raw: bytes):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

def check_zip(raw: bytes) -> dict[str, bytes]:
    refuse(len(raw) == 79486 and sha(raw) == ZIP_SHA and blob(raw) == ZIP_BLOB,
           "original complete ZIP container altered")
    original_text, original_provenance = prior.load_originals()
    prior.verify_original(original_text, original_provenance)
    inventory = json.loads(MANIFEST.read_bytes())["members"]
    found = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        refuse(len(names) == len(set(names)) == 8 and set(names) == set(inventory),
               "ZIP member inventory, order-independent count, or duplicate mismatch")
        for name in names:
            info = z.getinfo(name)
            refuse(not info.is_dir() and info.filename == Path(name).name,
                   "unsafe or nested ZIP member")
            contents = z.read(name)
            size, digest = inventory[name]
            refuse(info.file_size == len(contents) == size and sha(contents) == digest,
                   "member mismatch: " + name)
            if name in original_text:
                refuse(contents == original_text[name], "original archived text mismatch: " + name)
            found[name] = contents
    refuse(found["checkout-sha.txt"].decode().strip() ==
           "905e237136e4ea75208a8ed92392489a9f62849d", "source checkout mismatch")
    return found

def check_job(raw: bytes):
    refuse(len(raw) > 10000 and b"40573188117" in raw
           and b"905e237136e4ea75208a8ed92392489a9f62849d" in raw,
           "original decoded complete job log identity missing")
    if ORIGINAL_LOG_BLOB is not None:
        refuse(blob(raw) == ORIGINAL_LOG_BLOB, "original decoded job Git blob changed")

def audit(folder: Path = FOLDER):
    raw_manifest = (folder / "original-custody.json").read_bytes()
    refuse(blob(raw_manifest) == MANIFEST_BLOB, "committed manifest raw Git blob altered")
    m = json.loads(raw_manifest)
    refuse(m["train_authorized"] is False
           and m["full_72_conformance_pass"] is False
           and m["generated_tool_memory_competence"] is False
           and m["new_model_execution"] is False
           and m["optimizer_steps"] == 0
           and m["historical_negative_five_of_seven_budget_exhausted"] is True,
           "false scientific admission or lost negative")
    refuse(m["original_run"] == 40573188117 and m["original_job"] == 122432551094
           and m["original_artifact_id"] == 12766071456
           and m["original_run_head"] == "905e237136e4ea75208a8ed92392489a9f62849d"
           and m["original_source_pr_head"] == "3592841802d85c442ca2521479b5d8795bee7233",
           "source/run identity altered")
    refuse(m["original_zip"]["sha256"] == ZIP_SHA
           and m["original_zip"]["git_blob"] == ZIP_BLOB
           and m["original_source_tar"]["sha256"] == TAR_SHA
           and m["original_source_tar"]["git_blob"] == TAR_BLOB,
           "manifest original ZIP/tar identity altered")
    zipped = check_zip((folder / "original-actions-40573188117.zip").read_bytes())
    tar = (folder / "original-source-snapshot-40573188117.tar.gz").read_bytes()
    refuse(len(tar) == 68602 and sha(tar) == TAR_SHA and blob(tar) == TAR_BLOB
           and tar == zipped["source-snapshot.tar.gz"],
           "standalone original binary source TAR changed or does not match ZIP")
    check_job((folder / "original-job-122432551094.log").read_bytes())
    refuse(ORIGINAL_LOG_BLOB is not None, "original complete job log Git blob not yet pinned")
    return {"status":"PASS_ORIGINAL_PR102_WHOLE_ZIP_SOURCE_TAR_AND_LOG_CUSTODY__SCIENTIFIC_72_BLOCKED",
            "original_run":40573188117,"original_members":len(zipped),
            "original_zip_sha256":ZIP_SHA,"source_tar_sha256":TAR_SHA,
            "original_job_git_blob":ORIGINAL_LOG_BLOB,"train_authorized":False,
            "full_72_conformance_pass":False,"new_model_execution":False}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify-external-zip",type=Path)
    p.add_argument("--extract-source-tar",type=Path)
    p.add_argument("--verify-job",type=Path)
    args=p.parse_args()
    if args.verify_external_zip:
        z=check_zip(args.verify_external_zip.read_bytes())
        if args.extract_source_tar:
            args.extract_source_tar.write_bytes(z["source-snapshot.tar.gz"])
        if args.verify_job:
            check_job(args.verify_job.read_bytes())
        print("PASS_EXACT_ORIGINAL_PR102_ZIP_SOURCE_TAR_AND_HOST_JOB")
    else:
        print(json.dumps(audit(),sort_keys=True,indent=2))
if __name__=="__main__":
    main()
