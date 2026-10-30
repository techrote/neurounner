#!/usr/bin/env python3
"""Fail-closed audit: ORIGINAL PR205 3 whole Actions ZIPs and 3 full decoded job logs.

No model loading, network, optimizer, semantic tool/memory admission or training.
Git blob constants are independently fixed in this source, not taken on trust from
the editable manifest. GitHub API-decoded job logs are not HTTP compressed streams.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review/pr205-alltoken2048-original"
ORIGINAL_SOURCE = "0c866f5b65380ef6629a5bf084d017e886b91e52"
ORIGINAL_PARENT = "b085d0cec41473b33d844acbf644b57304526779"
ORIGINAL_RUN = 40987371706
CANCELLED_FIRST_RUN = 40987308916
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
POSTLEASE_BLOB = "66e6b4d97eea5e917339b8e7bfb5d1c191db4a2d"
MANIFEST_BLOB = "7d5bb5343198b4d303039a98ecb02cf52fec8264"
ARTIFACT_IDS = {"source": 12947388112, "chat": 12947358773, "instruct": 12947986971}
JOB_IDS = {"source": 123841492749, "chat": 123841738132, "instruct": 123841738103}
ORIGINAL_BLOBS = {
    "original-source-12947388112.zip": "0d02d61a0b61e21888e14988b33215e22ae8e618",
    "original-chat-12947358773.zip": "981d43a4f4d2ff1ecd7bf9792f9bc37cec25482e",
    "original-instruct-12947986971.zip": "ae512e5716382fe250cf3975e850a9e0040c6a11",
    "original-source-job-123841492749.log": "2e584743350cea7b50ae9671bf5ee8a357b27021",
    "original-chat-job-123841738132.log": "82e03205186e0241d8d1d7325f283d1e3694841a",
    "original-instruct-job-123841738103.log": "63bfabff6f993ff5046a2c0ffa1ccd8f35a282a1",
}
ORIGINAL_ARCHIVE_SHA256 = {
    "source": "0d5b5a8c99d701a1b6c1df6234a36912c979fd52939b29332542509ed4170c69",
    "chat": "05d4b24909d612f08abf6dd65f498f71f31daab53b4c3cc3cfc8d0d881a40f36",
    "instruct": "6d519ff309f81dfd805cfc108c926d22eafaebd12a2854fc8fba4377ca199b32",
}
ORIGINAL_CACHE_MAX_ABS = {
    "chat": 0.00026702880859375,
    "instruct": 0.000110626220703125,
}
SOURCE_FILES = {
    "chat": ("chat/model.safetensors",
             "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
    "instruct": ("model.safetensors",
                 "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"),
}
PREFIXES = [1, 2, 16, 64, 128, 256, 512, 1024, 1536, 2048]
PASS_STATUS = "PASS_2048_EVERY_TOKEN_TRUE_KV_STREAM__FINAL_72_BLOCKED"


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("PR205_ORIGINAL_CUSTODY_REFUSED: " + reason)


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(evidence: Path = EVIDENCE) -> dict:
    raw_manifest = (evidence / "manifest.json").read_bytes()
    refuse(git_blob(raw_manifest) == MANIFEST_BLOB,
           "original manifest Git blob changed")
    m = json.loads(raw_manifest)
    refuse(m["schema"] == "ncr001-pr205-original-alltoken2048-github-actions-custody-v1",
           "original archival schema drift")
    refuse(m["source_pr"] == 205 and m["source_sha"] == ORIGINAL_SOURCE and
           m["source_run"] == ORIGINAL_RUN and
           m["original_parent_sha"] == ORIGINAL_PARENT, "source lineage mismatch")
    refuse(m["protocol_blob"] == PROTOCOL_BLOB and
           m["pins_blob"] == PINS_BLOB and
           m["postlease_blob"] == POSTLEASE_BLOB, "protocol/source identity drift")
    refuse(m["train_authorized"] is False and
           m["full_72_conformance_pass"] is False and
           m["actual_compute_budget_qualified"] is False and
           m["optimizer_steps"] == 0, "false scientific/training admission")
    records = m["original_files"]
    refuse(set(records) == set(ORIGINAL_BLOBS), "exact original 3-ZIP/3-log inventory drift")
    checked = {}
    for filename, fixed_blob in ORIGINAL_BLOBS.items():
        raw = (evidence / filename).read_bytes()
        record = records[filename]
        refuse(git_blob(raw) == fixed_blob == record["git_blob_sha1"],
               "original Git blob modified: " + filename)
        refuse(sha256(raw) == record["sha256"] and len(raw) == record["bytes"],
               "original SHA/length drift: " + filename)
        checked[filename] = {"bytes": len(raw), "git_blob": fixed_blob,
                             "sha256": sha256(raw)}

    lanes = {}
    for lane in ("source", "chat", "instruct"):
        zp = evidence / f"original-{lane}-{ARTIFACT_IDS[lane]}.zip"
        lp = evidence / f"original-{lane}-job-{JOB_IDS[lane]}.log"
        refuse(sha256(zp.read_bytes()) == ORIGINAL_ARCHIVE_SHA256[lane],
               "original complete compressed ZIP changed " + lane)
        log = lp.read_bytes().decode("utf-8-sig")
        refuse(str(ORIGINAL_RUN) in log and
               ("ncr001" in log.lower() or ORIGINAL_SOURCE in log),
               "original complete decoded job log/source witness absent " + lane)
        with zipfile.ZipFile(zp) as z:
            refuse(z.testzip() is None, "original ZIP CRC corrupt " + lane)
            members = records[zp.name]["members"]
            refuse(set(z.namelist()) == set(members), "original ZIP members drift " + lane)
            for name in z.namelist():
                refuse(".." not in Path(name).parts and not name.startswith("/"),
                       "unsafe ZIP path " + lane)
                raw = z.read(name)
                refuse(len(raw) == members[name]["bytes"] and
                       sha256(raw) == members[name]["sha256"],
                       "original ZIP member SHA/length drift: " + name)
            if lane == "source":
                src_log = z.read("tests.log").decode("utf-8")
                refuse("Ran 12 tests" in src_log and "\nOK" in src_log,
                       "original 12/12 source tests not verified")
                lanes[lane] = {"source_tests": 12, "original_physical_model": False}
                continue
            r = json.loads(z.read("report.json"))
            refuse(r["schema"] == "ncr001-sprocket-alltoken-stream-2048-v1" and
                   r["lane"] == "sprocket_" + lane and r["status"] == PASS_STATUS,
                   "original frozen model status/lane drift " + lane)
            refuse(r["runner"]["head_sha"] == ORIGINAL_SOURCE and
                   str(r["runner"]["run_id"]) == str(ORIGINAL_RUN) and
                   r["runner"]["cpu_threads"] == 4 and
                   r["runner"]["torch"] == "2.14.1+cpu" and
                   r["runner"]["transformers"] == "4.57.6",
                   "original host, package or exact source identity drift " + lane)
            refuse(r["original_source_postlease_git_blob"] == POSTLEASE_BLOB and
                   r["source_protocol_git_blob"] == PROTOCOL_BLOB and
                   r["source_pins_git_blob"] == PINS_BLOB and
                   r["frozen_source_base_commit"] == ORIGINAL_PARENT,
                   "original model/protocol/source identity drift " + lane)
            refuse(r["original_frozen_model_executed"] is True and
                   r["train_authorized"] is False and
                   r["full_72_conformance_pass"] is False and
                   r["actual_compute_budget_qualified"] is False and
                   r["optimizer_steps"] == 0 and
                   r["comparative_training"] is False and
                   r["heldouts_opened"] is False and
                   r["generated_tool_memory_semantics_qualified"] is False,
                   "false original training/quality/behavioral admission " + lane)
            refuse(r["batch_size"] == 1 and r["context_units"] == 2048 and
                   r["all_prefixes_run_without_cache"] is False and
                   r["all_token_streaming_cache_observation_only"] is True and
                   r["last_verified_position"] == 2047,
                   "context/prefix scope improperly enlarged " + lane)
            weig, expected_sha = SOURCE_FILES[lane]
            f = r["frozen_source_files"][weig]
            refuse(f["sha256"] == expected_sha and f["bytes"] == 1002207984,
                   "original full checkpoint variant/weight drift " + lane)
            refuse(r["frozen_backbone_before"] == r["frozen_backbone_after"] and
                   r["checks"]["native_zero_logit_parity"] == 0 and
                   r["checks"]["native_disabled_logit_parity"] == 0 and
                   r["checks"]["altered_future_suffix_max_abs"] == 0 and
                   r["checks"]["manual_intervention_learned"] is False and
                   r["checks"]["manual_untrained_effect_max_abs"] > 0,
                   "frozen/no-op/future intervention witness drift " + lane)
            c = r["all_token_check"]
            refuse(c["tokens_compared"] == c["all_real_token_positions"] == 2048 and
                   c["cached_step_size"] == 1 and
                   c["numeric_atol"] == c["numeric_rtol"] == 0.0005 and
                   c["all_steps_past_key_values_returned"] is True and
                   c["all_steps_depth_field_per_current_token"] is True and
                   c["max_abs"] == ORIGINAL_CACHE_MAX_ABS[lane] and
                   c["max_abs_position"] == 0,
                   "original 2048 actual cache scores/constraint changed " + lane)
            refuse(c["uncached_prefix_checkpoint_positions"] == PREFIXES and
                   set(c["uncached_prefix_max_abs"]) ==
                   {str(x) for x in PREFIXES} and
                   set(c["bucket_max_abs"]) == {"0", "1", "2", "3"},
                   "falsely reported 2048 uncached prefixes " + lane)
            refuse(PASS_STATUS in log and
                   f"NCR001_ALL_TOKEN_PROGRESS {lane} 2048" in log,
                   "original full physical job log final/partial phase missing " + lane)
            refuse(PASS_STATUS in z.read("model.log").decode("utf-8"),
                   "original raw in-artifact model witness missing " + lane)
            lanes[lane] = {"model": weig, "max_abs": c["max_abs"],
                           "tokens": c["tokens_compared"],
                           "uncached_prefixes": len(PREFIXES),
                           "optimizer_steps": 0}
    return {"status": "PASS_ORIGINAL_PR205_3ZIP_3LOG_GIT_BLOBS_2048_STREAM",
            "source_sha": ORIGINAL_SOURCE, "original_run": ORIGINAL_RUN,
            "original_zip_count": 3, "original_decoded_log_count": 3,
            "checked_files": checked, "lanes": lanes,
            "train_authorized": False, "full_72_conformance_pass": False}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
