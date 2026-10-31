#!/usr/bin/env python3
"""Fail-closed permanent custody audit of original PR #202 Actions ZIPs/job logs.

No model, tokenizer, optimizer, network or held-out data. The original Git
blob IDs below are independent constants, not editable claims in the manifest.
Original GitHub API-decoded job logs are NOT compressed HTTP wire streams.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review/pr202-masked2048-original"
ORIGINAL_SOURCE = "b52af229dbfd0220d8d89f0d18e58733a2c3b56b"
ORIGINAL_PARENT = "2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80"
ORIGINAL_RUN = 40939386209
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
MANIFEST_BLOB = "bf17d3e3cd8ae9567b1e08c95f06c001bbbbd2eb"
ORIGINAL_BLOBS = {
    "original-source-12925688929.zip": "113f7be899c979fdd0ce9088e2497f4927f18873",
    "original-chat-12927370503.zip": "b7415f877dd90d2696b3e87daa7c5a9c8e67ae28",
    "original-instruct-12926034445.zip": "acd7d0046e0ef5344a374f3a34068725efc26ab6",
    "original-source-job-123678411187.log": "a2dd28eceffaf488b063887f6e9b1ce6e8a0239a",
    "original-chat-job-123678836606.log": "656cbbad05959d734ee676f37a3ccf5b8801a466",
    "original-instruct-job-123678836615.log": "1c20f203f616ca7d1db1bda744e5443c22577e83",
}
ORIGINAL_ARCHIVE_SHA256 = {
    "source": "a61769d08582dfb8f8dfc07c02a385c34380627cae3822cb9820889fece6e223",
    "chat": "d0fac3a396c201d0f5377160381d47e101a62194e577dd0f3e7b51a4254a3753",
    "instruct": "59b65a0ef8be4191fe1ad63ec016694f5bc88f03dd647bece9635cfa5c2499fd",
}
SOURCE_FILES = {
    "chat": ("chat/model.safetensors", "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
    "instruct": ("model.safetensors", "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"),
}
ORIGINAL_CACHE_MAX = {
    "chat": (2.2232532501220703e-05, 2.2232532501220703e-05),
    "instruct": (1.5735626220703125e-05, 2.2649765014648438e-05),
}
JOB_IDS = {"source": 123678411187, "chat": 123678836606, "instruct": 123678836615}
ARTIFACT_IDS = {"source": 12925688929, "chat": 12927370503, "instruct": 12926034445}


def refuse(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PR202_ORIGINAL_CUSTODY_REFUSED: " + message)


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit(evidence: Path = EVIDENCE) -> dict:
    manifest_path = evidence / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    refuse(git_blob(raw_manifest) == MANIFEST_BLOB, "original manifest Git blob changed")
    m = json.loads(raw_manifest)
    refuse(m["schema"] == "ncr001-pr202-original-github-actions-custody-v1", "schema drift")
    refuse(m["source_pr"] == 202 and m["source_sha"] == ORIGINAL_SOURCE and
           m["original_parent_sha"] == ORIGINAL_PARENT and
           m["source_run"] == ORIGINAL_RUN, "original source/run lineage mismatch")
    refuse(m["protocol_blob"] == PROTOCOL_BLOB and m["pins_blob"] == PINS_BLOB,
           "original protocol/source pin drift")
    refuse(m["train_authorized"] is False and
           m["full_72_conformance_pass"] is False and
           m["optimizer_steps"] == 0, "unqualified scientific/training assertion")
    records = m["original_files"]
    refuse(set(records) == set(ORIGINAL_BLOBS), "exact original 3 ZIP + 3 log inventory drift")
    verified = {}
    for filename, fixed_blob in ORIGINAL_BLOBS.items():
        path = evidence / filename
        raw = path.read_bytes()
        record = records[filename]
        refuse(git_blob(raw) == fixed_blob == record["git_blob_sha1"],
               "Git blob modified: " + filename)
        refuse(sha256(raw) == record["sha256"] and len(raw) == record["bytes"],
               "SHA/length changed: " + filename)
        verified[filename] = {"bytes": len(raw), "sha256": sha256(raw), "git_blob": fixed_blob}

    for variant in ("source", "chat", "instruct"):
        zp = evidence / f"original-{variant}-{ARTIFACT_IDS[variant]}.zip"
        lp = evidence / f"original-{variant}-job-{JOB_IDS[variant]}.log"
        refuse(sha256(zp.read_bytes()) == ORIGINAL_ARCHIVE_SHA256[variant],
               "original full ZIP digest mismatch " + variant)
        joblog = lp.read_bytes().decode("utf-8-sig")
        refuse("40939386209" in joblog and
               (ORIGINAL_SOURCE in joblog or "ncr001-masked2048" in joblog),
               "original job log/source pairing lost " + variant)
        with zipfile.ZipFile(zp) as z:
            refuse(z.testzip() is None, "original ZIP CRC failed " + variant)
            members = records[zp.name]["members"]
            refuse(set(z.namelist()) == set(members), "original ZIP member list drift")
            for name in z.namelist():
                refuse(".." not in Path(name).parts and not name.startswith("/"),
                       "unexpected ZIP extraction path")
                data = z.read(name)
                refuse(len(data) == members[name]["bytes"] and
                       sha256(data) == members[name]["sha256"],
                       "original ZIP member SHA changed " + name)
            if variant == "source":
                tests = z.read("source-tests.log").decode("utf-8")
                refuse("Ran 9 tests" in tests and "\nOK" in tests,
                       "original nine offline tests missing")
                continue
            producer = json.loads(z.read("producer.json"))
            consumer = json.loads(z.read("consumer.json"))
            expected_variant = "sprocket_" + variant
            refuse(producer["status"] == "PASS_MASKED2048_PRODUCER" and
                   consumer["status"] == "PASS_MASKED2048_SEPARATE_PYTHON_COLD_CONSUMER",
                   "original producer/consumer status changed")
            refuse(producer["variant"] == consumer["variant"] == expected_variant,
                   "Chat/root-Instruct identity conflation")
            p, c = producer["runner"], consumer["runner"]
            refuse(p["head_sha"] == c["head_sha"] == ORIGINAL_SOURCE and
                   str(p["run_id"]) == str(c["run_id"]) == str(ORIGINAL_RUN),
                   "original exact-source/runner ID drift")
            refuse(p["pid"] != c["pid"] and p["host"] == c["host"] and
                   p["cpu_threads"] == c["cpu_threads"] == 4 and
                   p["torch"] == c["torch"] == "2.14.1+cpu",
                   "distinct cold interpreter or original runner mismatch")
            for rep in (producer, consumer):
                refuse(rep["train_authorized"] is False and
                       rep["full_72_conformance_pass"] is False and
                       rep["optimizer_steps"] == 0 and
                       rep["heldouts_opened"] is False and
                       rep["original_frozen_model_executed"] is True,
                       "false original model/optimizer/evaluation status")
                refuse(rep["frozen_before_sha256"] == rep["frozen_after_sha256"],
                       "frozen model drift")
                weig, weight_sha = SOURCE_FILES[variant]
                refuse(rep["verified_frozen_source_files"][weig]["sha256"] == weight_sha and
                       rep["verified_frozen_source_files"][weig]["bytes"] == 1002207984,
                       "wrong original full frozen Sprocket checkpoint")
            refuse(producer["source_sha256"] == consumer["source_sha256"] and
                   producer["synthetic_input_sha256"] == consumer["synthetic_input_sha256"] and
                   producer["verified_frozen_source_files"] ==
                   consumer["verified_frozen_source_files"],
                   "producer/consumer source or input changed")
            for side, expected_max in zip(("left", "right"), ORIGINAL_CACHE_MAX[variant]):
                key = side + "_2048_full_vs_four_real_cached"
                cache = producer["checks"][key]
                refuse(cache["pass"] is True and cache["finite"] is True and
                       cache["shape"] == [2, 4, 32000] and
                       cache["atol"] == cache["rtol"] == .0005 and
                       cache["max_abs"] == expected_max,
                       "original cache/full mismatch or relaxed gate " + variant + side)
                refuse(producer["checks"][side + "_mask_corruption_sentinel"] is True,
                       "mask-failure sentinel missing")
                cold = consumer["checks"][side + "_cold_new_python_same_model_2048"]
                refuse(cold["pass"] is True and cold["exact"] is True and
                       cold["max_abs"] == 0 and cold["shape"] == [2, 4, 32000],
                       "original cold reload is not byte-exact")
            for name, digest in producer["adapter_sha256"].items():
                refuse(sha256(z.read("adapter/" + name)) == digest,
                       "original serialized adapter bytes changed")
            refuse(sha256(z.read("reference.pt")) ==
                   producer["reference_file_sha256"] ==
                   consumer["reference_file_sha256"],
                   "original cold reference file changed")
            refuse("PASS_MASKED2048_PRODUCER" in joblog and
                   "PASS_MASKED2048_SEPARATE_PYTHON_COLD_CONSUMER" in joblog,
                   "original physical job-log phase witness absent")
    return {"status": "PASS_ORIGINAL_PR202_3ZIP_3LOG_GIT_BLOB_SOURCE_AND_NUMERIC_RECEIPT",
            "source_sha": ORIGINAL_SOURCE, "source_run": ORIGINAL_RUN,
            "original_zip_count": 3, "original_decoded_log_count": 3,
            "files": verified, "train_authorized": False,
            "full_72_conformance_pass": False}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
