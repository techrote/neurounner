#!/usr/bin/env python3
"""Fail-closed complete original #41021703985 source-only review ZIP/log custody.

This audits SOURCE and a real pinned TOKENIZER only. It does not load Qwen3,
train an adapter, adjudicate #72 scientific admission, or use heldouts.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/ncr-001/byte-eos-001/postshift-original-20261030"
ARCHIVE = DIR / "original-actions-a1.zip.b64"
JOB_LOG = DIR / "original-decoded-job.log"
MANIFEST = DIR / "manifest.json"
ZIP_SHA = "fcc87e0204f9edfed67436e02d554f34b76ebc3a5673d151010b6cb59ec6d032"
ZIP_GIT_BLOB = "16fe5a4f3133e1607157c9b0c915d562857beaa9"
LOG_SHA = "0fae31f94e1ad7de417e3e0f7b2cd03841d3c483d35a580488eea450cf0020df"
LOG_GIT_BLOB = "0cd2b6b5c5b978b51f09903f6e24aa8fa5eee618"
RUN_SHA = "e429cca8b0feeff2f1e3672b71efaf1f6c2e3502"
CORRECTED_MAIN = "bb60d700fe86cf530773f1f639a246cc24bc0943"
TOKENIZER_SHA = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
MEMBERS = {
    "byte-eos-postshift-pr-checkout.txt":
        (41, "9defb797e1f37ed98fd55d1294ffbe5e7204e0f1c159caab41bbdec6a4f05c45"),
    "byte-eos-postshift-real.json":
        (1999, "91ba32147b30a6e77488e6a4dc2d50be2cf1f65d5fafc8e35f76024a45e4a614"),
    "byte-eos-postshift-corrected-commit.txt":
        (41, "e6e1590cd1de35ce7eb3e3f981a13a06c93d00fabe681422d15fb5c832b24e9b"),
    "byte-eos-postshift-source-hashes.sha256":
        (552, "4cab431b48166bbfc36be42c886b82408ea0d9335dd590a356f92eb86d4bc820"),
    "byte-eos-postshift-synthetic.json":
        (1981, "1c3f5762c812902cc126bf1f0423afffd28f58c67f97e7cb21f6d0f0be1cc220"),
}


def refuse(condition, why):
    if not condition:
        raise RuntimeError("BYTE_EOS_POSTSHIFT_CUSTODY_REFUSED: " + why)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def blob_sha(data):
    raw = b"blob " + str(len(data)).encode() + b"\0" + data
    return hashlib.sha1(raw).hexdigest()


def verify(archive_path=ARCHIVE, job_log_path=JOB_LOG, manifest_path=MANIFEST,
           check_branch=True):
    manifest = json.loads(Path(manifest_path).read_bytes())
    artifact = manifest["original_artifact"]
    run = manifest["source_run"]
    logref = manifest["original_decoded_job_log"]
    pins = manifest["corrected_source"]
    gate = manifest["boundaries"]
    refuse(manifest["schema"] == 1
           and manifest["experiment"] == "BYTE-EOS-001"
           and manifest["issue"] == 92 and manifest["pr"] == 126,
           "wrong evidence identity")
    refuse(run["run_id"] == 41021703985
           and run["job_id"] == 123950278740
           and run["head_sha"] == RUN_SHA
           and run["event"] == "push" and run["status"] == "success"
           and run["runner_name"] == "GitHub Actions 1000048717"
           and run["runner_image"] == "ubuntu-24.04"
           and run["python"] == "3.12.3"
           and run["transformers"] == "4.57.6"
           and run["original_126_review_head"] ==
           "54eda2e177d4dbaebcff84cc09a08add1900226c",
           "original exact-head CI identity/runner changed")
    refuse(artifact["id"] == 12962350242
           and artifact["name"] == "byte-eos-postshift-source-review-a1"
           and artifact["compressed_zip_bytes"] == 2845
           and artifact["sha256"] == ZIP_SHA
           and artifact["git_blob_base64"] == ZIP_GIT_BLOB,
           "original artifact identity/ZIP changed")
    refuse(logref == {
        "bytes": 81917, "sha256": LOG_SHA, "git_blob_sha1": LOG_GIT_BLOB
    }, "original decoded job log manifest changed")
    refuse(pins["main_commit"] == CORRECTED_MAIN
           and pins["serializer_blob"] == "4c00c78b6776ade6901a99eee9a42e7ca61c698b"
           and pins["reviewer_blob"] == "f65698929d00a7b452f72a6b7a0824f68efb9c59"
           and pins["reviewer_tests_blob"] == "1e3dd77426c33ab9ad31fb0a69a1a143167549eb"
           and pins["real_byte_tokenizer_sha256"] == TOKENIZER_SHA,
           "reviewed corrected source/real tokenizer changed")
    refuse(manifest["observed"] == {
        "original_archive_tests_passed": 6,
        "corrected_source_tests_passed": 8,
        "real_byte_tokenizer_oracle_cases_passed": 7,
        "empty_prompt_cases_refused": 4,
    }, "original source-only observations changed")
    refuse(gate["model_weights_loaded"] is False
           and type(gate["optimizer_steps"]) is int and gate["optimizer_steps"] == 0
           and gate["heldout_evaluation"] is False
           and gate["comparative_training"] is False
           and gate["train_authorized"] is False
           and gate["full_72_conformance_pass"] is False
           and gate["future_training_integration_accepted"] is False
           and gate["original_train002b_effect"] == "UNMEASURED",
           "false training/scientific admission")
    refuse(manifest["original_historical_126_zip_sha256"] ==
           "9cde28bc5a927c262a0eda907f445491f4937c0ed7283a1ea535d2b5b323b1bf",
           "original old review ZIP relabelled")

    encoded = Path(archive_path).read_bytes()
    refuse(blob_sha(encoded) == ZIP_GIT_BLOB,
           "ZIP text Git-blob pinned identity changed")
    try:
        original = base64.b64decode(b"".join(encoded.split()), validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise RuntimeError("BYTE_EOS_POSTSHIFT_CUSTODY_REFUSED: invalid base64") from e
    refuse(len(original) == 2845 and sha256(original) == ZIP_SHA,
           "full original compressed ZIP bytes changed")
    contents = {}
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        refuse(z.testzip() is None, "original ZIP CRC failed")
        infos = z.infolist()
        refuse([i.filename for i in infos] == list(MEMBERS)
               and list(artifact["members"]) == list(MEMBERS),
               "original ZIP member set/order altered")
        for info in infos:
            raw = z.read(info.filename)
            n,digest = MEMBERS[info.filename]
            refuse(info.file_size == len(raw) == n
                   and sha256(raw) == digest
                   and artifact["members"][info.filename] ==
                   {"bytes": n, "sha256": digest},
                   "original ZIP member corrupted: " + info.filename)
            contents[info.filename] = raw
    refuse(contents["byte-eos-postshift-pr-checkout.txt"] ==
           (RUN_SHA + "\n").encode(),
           "original source checkout not literal reviewed head")
    refuse(contents["byte-eos-postshift-corrected-commit.txt"] ==
           (CORRECTED_MAIN + "\n").encode(),
           "corrected #163 checkout changed")
    for name in ("real", "synthetic"):
        report = json.loads(contents["byte-eos-postshift-" + name + ".json"])
        refuse(report["runner"]["github_sha"] == RUN_SHA
               and report["runner"]["github_run_id"] == "41021703985"
               and report["runner"]["runner_name"] ==
               "GitHub Actions 1000048717"
               and report["reviewed_corrected_source_commit"] == CORRECTED_MAIN
               and report["void_tokenizer_source_sha256"] == TOKENIZER_SHA
               and report["reviewed_corrected_serializer_git_blob"] ==
               "4c00c78b6776ade6901a99eee9a42e7ca61c698b",
               "recorded real/synthetic source/run drift: " + name)
        refuse(report["runtime"] == (
                   "pinned-real-ByteTokenizer-transformers-4.57.6"
                   if name == "real" else "synthetic-byte-codec-source-only")
               and report["status"] == (
                   "PASS_CORRECTED_POSTSHIFT_REAL_TOKENIZER_SOURCE_ONLY"
                   if name == "real" else "PASS_CORRECTED_POSTSHIFT_SYNTHETIC_SOURCE_ONLY")
               and report["empty_prompt_refused_count"] == 4
               and [c["name"] for c in report["cases"]] == [
                   "ascii", "unicode", "literal_eos_content",
                   "empty_completion", "one_byte_prefix",
                   "emoji_boundary", "max_1024"]
               and all(c["oracle_match"] is True for c in report["cases"])
               and report["training_authorized"] is False
               and type(report["optimizer_steps"]) is int
               and report["optimizer_steps"] == 0
               and report["new_model_execution"] is False
               and report["full_72_conformance_pass"] is False
               and report["future_training_integration_accepted"] is False
               and report["historical_generation_effect"] == "UNMEASURED",
               "false model/scientific admission or unexpected oracle result")
    log = Path(job_log_path).read_bytes()
    refuse(len(log) == 81917 and sha256(log) == LOG_SHA
           and blob_sha(log) == LOG_GIT_BLOB,
           "complete original API decoded job log changed")
    for needle in (
        b"Ran 6 tests", b"Ran 8 tests",
        b"PASS_CORRECTED_POSTSHIFT_REAL_TOKENIZER_SOURCE_ONLY",
        b"PASS_BYTE_EOS_INDEPENDENT_ARCHIVE__INTEGRATION_BLOCKED",
        b"MODEL_WEIGHTS_DOWNLOADED false",
        b"Artifact ID 12962350242",
    ):
        refuse(needle in log, "missing positive/negative original log evidence")
    if check_branch:
        def git(*args):
            return subprocess.run(["git", "-C", str(ROOT), *args],
                                  text=True, capture_output=True)
        refuse(git("merge-base", "--is-ancestor", RUN_SHA, "HEAD").returncode == 0,
               "original reviewer source is not an ancestor")
        for path, digest in (
            ("scripts/byte_eos_001_postshift_review.py",
             "f65698929d00a7b452f72a6b7a0824f68efb9c59"),
            ("tests/test_byte_eos_001_postshift_review.py",
             "1e3dd77426c33ab9ad31fb0a69a1a143167549eb"),
        ):
            out = git("hash-object", path)
            refuse(out.returncode == 0 and out.stdout.strip() == digest,
                   "reviewer source edited after original execution")
    return {
        "status": "PASS_ORIGINAL_POSTSHIFT_FULL_ZIP_LOG_GIT_CUSTODY__SOURCE_ONLY",
        "original_run_id": 41021703985,
        "original_job_id": 123950278740,
        "original_source_sha": RUN_SHA,
        "original_zip_sha256": ZIP_SHA,
        "original_decoded_job_log_sha256": LOG_SHA,
        "original_original_126_zip_sha256":
            "9cde28bc5a927c262a0eda907f445491f4937c0ed7283a1ea535d2b5b323b1bf",
        "model_weights_loaded": False, "optimizer_steps": 0,
        "train_authorized": False, "scientific_72_admitted": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
