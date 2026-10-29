#!/usr/bin/env python3
"""Source-only exact ZIP and two complete decoded original Actions-job custody.

Read-only offline corroboration of original #99 void.1 frozen MLP prepass,
NOT fresh Qwen3 inference, separate-process cold reload or #72 admission.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
import zipfile

import ncr_001f_void_mlp_receipt_audit as original

EVIDENCE = original.EVIDENCE
ZIP_FILE = "original-actions-zip.base64"
MANIFEST_FILE = "original-actions-custody.json"
MANIFEST_GIT_BLOB = "cff5a8b28239d3bad8b0d0fc92a28d44efac1461"
ZIP_GIT_BLOB = "ffba95801464a046e3a61ecf07e06052b99b4d57"
ORIGINAL_ZIP_SIZE = 3508
ORIGINAL_ZIP_SHA256 = "8e780dc30262d139fccb7c7d00962ab5b702280ef96fd76575f57c41d1435de0"
ORIGINAL_HEAD = "35c4eb966f12965c7640d1fd8a0c6d896b075791"
JOBS = {
    "original-source-job-122256460886.log":
        (122256460886, "5dd15def669be3e16870a94ba1e1ed5fa6808bce"),
    "original-model-job-122256966798.log":
        (122256966798, "3c51293865c36c188137b8e4a4ee47f082e0bd27"),
}
MEMBERS = {
    "void1-source-prefetch.json":
        (1593, "761b1a350eb3270ea6b6b44e4a3acb640753a1d9149f6580656340081f12266c"),
    "void1-real-mlp.json":
        (8103, "f962abde997b0215be51df2210df4d61d5a2a453773d7c1047df0497bfba7c4a"),
}


def refuse(predicate: bool, msg: str) -> None:
    if not predicate:
        raise ValueError("NCR001_VOID_ORIGINAL_REFUSED: " + msg)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_zip(raw: bytes, original_members: dict[str, bytes] | None = None) -> dict:
    refuse(len(raw) == ORIGINAL_ZIP_SIZE and sha(raw) == ORIGINAL_ZIP_SHA256,
           "full original Actions ZIP bytes/hash mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        refuse(len(names) == len(MEMBERS) == len(set(names))
               and set(names) == set(MEMBERS), "original ZIP member names/duplicates")
        for name in names:
            info = archive.getinfo(name)
            refuse(not info.is_dir() and not name.startswith("/")
                   and ".." not in Path(name).parts and info.file_size < 100000,
                   "ZIP member path/type/size unsafe")
            member = archive.read(name)
            size, digest = MEMBERS[name]
            refuse(len(member) == size and sha(member) == digest,
                   "original member mutated: " + name)
            if original_members is not None:
                refuse(member == original_members[name],
                       "member differs from independently committed raw JSON: " + name)
    return {"zip_bytes": len(raw), "zip_sha256": sha(raw),
            "members": len(MEMBERS)}


def audit(evidence: Path = EVIDENCE) -> dict:
    # The existing primary auditor protects its separate original raw JSON custody.
    prior = original.audit(evidence)
    refuse(prior["training_admitted"] is False and prior["optimizer_steps"] == 0,
           "earlier source-only prepass gate elevated")
    manifest_raw = (evidence / MANIFEST_FILE).read_bytes()
    refuse(original.gitblob(manifest_raw) == MANIFEST_GIT_BLOB,
           "committed original Actions manifest bytes altered")
    manifest = json.loads(manifest_raw)
    refuse(manifest["original_source_head"] == ORIGINAL_HEAD
           and manifest["original_run_id"] == 40523446884
           and manifest["original_artifact_id"] == 12741959402
           and manifest["original_zip_bytes"] == ORIGINAL_ZIP_SIZE
           and manifest["original_zip_sha256"] == ORIGINAL_ZIP_SHA256
           and manifest["original_zip_base64_git_blob"] == ZIP_GIT_BLOB,
           "original source, artifact or archive identity changed")
    refuse(manifest["model"] == "dotlabs/void.1"
           and manifest["eos_token_id"] == 258
           and manifest["train_authorized"] is False
           and manifest["full_72_conformance_pass"] is False
           and manifest["original_training_impact_measured"] is False
           and manifest["cold_separate_process_qualified"] is False
           and manifest["optimizer_steps"] == 0
           and manifest["heldouts_opened"] is False
           and manifest["comparative_training"] is False,
           "scientific, EOS, cold reload or training admission fabricated")
    archive_raw = (evidence / ZIP_FILE).read_bytes()
    refuse(original.gitblob(archive_raw) == ZIP_GIT_BLOB,
           "committed full original ZIP base64 blob altered")
    try:
        raw_zip = base64.b64decode(archive_raw.strip(), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("NCR001_VOID_ORIGINAL_REFUSED: invalid archive encoding") from exc
    original_json = original.read_original_bytes(
        (evidence / original.ARCHIVE.name).read_bytes().strip(),
        json.loads((evidence / original.PROVENANCE.name).read_bytes()))
    zip_result = validate_zip(raw_zip, original_json)
    for path, (job_id, blob) in JOBS.items():
        info = manifest["complete_original_decoded_job_logs"][path]
        refuse(info["job_id"] == job_id and info["git_blob"] == blob
               and info["status"] == "success", "original job manifest modified")
        log = (evidence / path).read_bytes()
        refuse(original.gitblob(log) == blob
               and ORIGINAL_HEAD.encode("ascii") in log
               and b"2026-10-27" in log,
               "original complete decoded job log changed: " + path)
    model_log = (evidence / "original-model-job-122256966798.log").read_bytes()
    refuse(b"PASS_SOURCE_BOUND_VOID1_UTF8_MLP_REAL_MODEL_PREPASS__TRAINING_BLOCKED"
           in model_log and b"40523446884" in model_log,
           "original narrow physical host PASS or run identity missing")
    return {"status": "PASS_ORIGINAL_VOID1_MLP_ACTIONS_ZIP_AND_BOTH_JOB_LOGS__72_BLOCKED",
            "original": zip_result,
            "original_run": 40523446884,
            "original_job_ids": [122256460886, 122256966798],
            "source_head": ORIGINAL_HEAD,
            "new_physical_model_execution": False,
            "full_72_conformance_pass": False,
            "train_authorized": False,
            "optimizer_steps": 0}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
