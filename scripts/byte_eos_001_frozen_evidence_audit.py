#!/usr/bin/env python3
"""Audit the *original* BYTE-EOS-001 frozen Qwen3 Action ZIP and job logs offline.

Exact original artifact and full logs are immutable; this is custody and
causal-label witness verification, NOT independent scientific/model admission.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import lzma
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/byte-eos-001/frozen-forward-evidence"
SOURCE_HEAD = "06b858a70d7019398cdc7a3f205ae24423f57c67"
SOURCE_LOG_BLOB = "ff47f266250a2383ef282e405e3ec26ef03b9e7a"
MODEL_LOG_BLOB = "592d2fcf59f6dd71c18c1dd5aa9bc5590d9849ab"
ORIGINAL_ARCHIVE_SHA = "23f0c5bbca8cfacf9050656ed3a972ca4d716ce0c9db04c40a7bcb524c03f770"
COMPRESSED_ARCHIVE_SHA = "ffdac15a7660f369f1af0a9ebbb9ca585f90a08d2112acc62acefa406f696de7"
REPORT_SHA = "ce8de8ce0e0f80bf59d5b24810101f17a4cce9fb73ab5190eda26457efce1f21"
WEIGHT_SHA = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER_SHA = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
FROZEN_FINGERPRINT = "d711be05faf9064801eba19e91e2f58ff9d40ad1c066fea640542762d98423bb"
CASES = (
    ("empty_completion_after_one_byte_context", "x", "", 8.765803337097168, 1,
     "bad93ae2ae4d7d19f8a05abdac93a86453afac7e8095e3a78a7e9193d10655ab"),
    ("mixed_utf8", "Q: π?\nA: ", "🔧 café 日本語", 3.0656538009643555, 21,
     "7e06b05406eba794c10e9176c6f101d973774e03536b6f0626180b83dcb85acc"),
    ("literal_eos_text_is_plain_bytes", "Q:", "<eos>", 4.957742691040039, 6,
     "16641b92bb1c70d35f7b8f5167d9c5b74e1dba9f5705331931601770729b1ede"),
)
EXPECTED_MEMBERS = {
    "byte-eos-frozen-qwen-console.log": "8acc9c9a6aaff984bc7e480cf656813e8ca97515cd7455a7c36c075644ff58ad",
    "byte-eos-frozen-qwen-forward.json": REPORT_SHA,
    "byte-eos-packages.txt": "45eacdddf315bf94dc46da90a3b25cf7f1a3c90874a2d6a805c6ba3f26f135fe",
    "byte-eos-prefetch.log": "9ae71eee0eb14814504f64c7ba5ad0cbf0fc5831cf6501a2dc0636f23ba7ab6c",
    "byte-eos-source-head.txt": "b5889b4eacf28d5562247ba7625a62102437715a6daf23e6a90685b4434bf0e4",
    "byte-eos-source-sha256.txt": "d444a9fe60183f65b55d3db783219594db6b7b3c797c3303dcbea462dff70e91",
    "void-byte-source-audit.json": "14ac28204837d7615220981c5b3ec3c86b9effe548306047b012fed69e0efa1c",
}


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("Original frozen Qwen3 evidence REFUSED: " + reason)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blobsha(b: bytes) -> str:
    return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()


def validate_report(report: dict) -> None:
    refuse(report["status"] == "PASS_FROZEN_QWEN3_CAUSAL_EOS_FORWARD_ONLY__72_BLOCKED"
           and report["run_id"] == "40738222796" and
           report["source_head"] == SOURCE_HEAD and
           report["runner_name"] == "GitHub Actions 1000046597" and
           report["model_executed"] is True and
           report["byte_tokenizer_executed"] is True,
           "original frozen host/source identity or actual model execution")
    for field in ("train_authorized", "backward_executed",
                  "comparative_training_performed", "heldout_opened",
                  "historical_train002b_effect_measured",
                  "full_72_conformance_pass", "independent_scientific_review"):
        refuse(report[field] is False, "false training/qualification claim: "+field)
    refuse(report["optimizer_steps"] == 0, "nonzero optimizer steps")
    refuse(report["revision"] == "9c2be5087772867d45961abd11f6b35110617563"
           and report["repo"] == "dotlabs/void.1" and
           report["source_files"]["model.safetensors"]["sha256"] == WEIGHT_SHA
           and report["source_files"]["model.safetensors"]["bytes"] == 360622744
           and report["source_files"]["tokenization_byte.py"]["sha256"] == TOKENIZER_SHA,
           "original model files not hash-bound")
    refuse(report["backbone_sha_before"] ==
           report["backbone_sha_after"] == FROZEN_FINGERPRINT,
           "frozen tensor state was changed or relabeled")
    refuse(len(report["cases"]) == len(CASES), "missing/extra original source cases")
    for actual, (name, prompt, completion, loss, supervised, logit_sha) in zip(report["cases"], CASES):
        p = list(prompt.encode("utf-8", "strict"))
        c = list(completion.encode("utf-8", "strict"))
        expected = c + [258]
        labels = [-100] * len(p) + expected
        input_ids = p + expected
        refuse(actual["name"] == name and actual["synthetic_prompt"] == prompt and
               actual["synthetic_completion"] == completion, "source case mutation")
        refuse(actual["input_ids"] == input_ids and actual["labels"] == labels and
               actual["attention_mask"] == [1] * len(input_ids) and
               actual["sequence_units"] == len(input_ids) and
               actual["causal_supervised_units"] == supervised == len(expected) and
               actual["terminal_eos_id"] == 258 and
               actual["all_real_tokens_attended"] is True,
               "source byte, label or post-shift target altered")
        refuse(actual["logits_sha256"] == logit_sha and
               math.isfinite(actual["original_qwen3_loss"]) and
               math.isfinite(actual["manual_shifted_ce"]) and
               actual["difference"] == 0.0 and
               actual["original_qwen3_loss"] ==
                    actual["manual_shifted_ce"] == loss,
               "loss/CE/logits witness changed")
    refuse(report["cases"][0]["labels"] == [-100, 258],
           "EOS-only completion without prompt must be supervised")


def audit(evidence: Path = EVIDENCE) -> dict:
    m = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    refuse(m["schema"] == "byte-eos-001-frozen-qwen-original-host-evidence-v1"
           and m["status"] == "PASS_ORIGINAL_FROZEN_QWEN3_FORWARD_ONLY__FINAL_72_BLOCKED"
           and m["original_source_head"] == SOURCE_HEAD and
           m["original_pr"] == 165 and m["parent_pr"] == 164 and
           m["parent_head"] == "796408aff6b56c1d9b732329d957749634a0830b"
           and m["original_workflow_run"] == 40738222796,
           "original source/parent/run swapped")
    for field in ("train_authorized", "full_72_conformance_pass",
                  "comparative_training_performed", "heldout_opened",
                  "backward_executed", "independent_scientific_review",
                  "historical_train002b_effect_measured", "nca_mlp_integration_qualified"):
        refuse(m["protocol_claims"][field] is False, "manifest false admission " + field)
    refuse(m["protocol_claims"]["optimizer_steps"] == 0, "manifest optimizer step")
    jobs = m["jobs"]
    refuse(jobs["source"]["job_id"] == 122992443448 and
           jobs["source"]["conclusion"] == "success" and
           jobs["source"]["tests_passed"] == 30 and
           jobs["source"]["log_git_blob"] == SOURCE_LOG_BLOB and
           jobs["original_model"]["job_id"] == 122992543245 and
           jobs["original_model"]["conclusion"] == "success" and
           jobs["original_model"]["runner"] == "GitHub Actions 1000046597" and
           jobs["original_model"]["log_git_blob"] == MODEL_LOG_BLOB,
           "original job/results/runners changed")
    for key, digest, phrase in (
        ("source", SOURCE_LOG_BLOB, "Ran 30 tests in"),
        ("original_model", MODEL_LOG_BLOB,
         "PASS_FROZEN_QWEN3_CAUSAL_EOS_FORWARD_ONLY__72_BLOCKED"),
    ):
        p = evidence / Path(jobs[key]["log_path"]).name
        raw = p.read_bytes()
        refuse(blobsha(raw) == digest, "raw decoded Git job log bytes changed")
        log = raw.decode("utf-8-sig")
        refuse(phrase in log and SOURCE_HEAD in log, "original source job lacks full witness")
    a = m["artifact"]
    refuse(a["id"] == 12838158155 and a["workflow_run"] == 40738222796 and
           a["original_archive_zip_sha256"] == ORIGINAL_ARCHIVE_SHA and
           a["original_archive_zip_bytes"] == 6625 and
           a["compressed_zip_xz_sha256"] == COMPRESSED_ARCHIVE_SHA and
           a["compressed_zip_xz_bytes"] == 6092 and
           a["members"] == EXPECTED_MEMBERS,
           "original artifact identity/digests changed")
    compressed_b64 = (evidence / Path(a["compressed_zip_base64_path"]).name).read_text("ascii").strip()
    compressed = base64.b64decode(compressed_b64, validate=True)
    refuse(len(compressed) == 6092 and sha(compressed) == COMPRESSED_ARCHIVE_SHA,
           "base64/LZMA raw original archive digest changed")
    original_zip = lzma.decompress(compressed, memlimit=32*1024*1024)
    refuse(len(original_zip) == 6625 and sha(original_zip) == ORIGINAL_ARCHIVE_SHA,
           "original Actions ZIP bytes differ")
    with zipfile.ZipFile(io.BytesIO(original_zip)) as z:
        refuse(len(z.infolist()) == 7 and
               sorted(z.namelist()) == sorted(EXPECTED_MEMBERS),
               "wrong original Actions ZIP members")
        raw_files = {}
        for member in z.infolist():
            refuse(not member.is_dir() and member.file_size < 100000 and
                   not member.filename.startswith("/") and
                   ".." not in member.filename, "unsafe original ZIP entry")
            original = z.read(member)
            refuse(sha(original) == EXPECTED_MEMBERS[member.filename],
                   "original member byte mutation: "+member.filename)
            raw_files[member.filename] = original
    report = json.loads(raw_files["byte-eos-frozen-qwen-forward.json"])
    refuse(sha(raw_files["byte-eos-frozen-qwen-forward.json"]) == REPORT_SHA,
           "original full model receipt changed")
    validate_report(report)
    refuse(raw_files["byte-eos-source-head.txt"].decode().strip() == SOURCE_HEAD,
           "original artifact source checkout mismatch")
    refuse("torch==2.14.1+cpu" in raw_files["byte-eos-packages.txt"].decode() and
           "transformers==4.57.6" in raw_files["byte-eos-packages.txt"].decode(),
           "original CPU runtime environment mismatch")
    refuse(m["original_model"]["original_weight_sha256"] == WEIGHT_SHA and
           m["original_model"]["tokenizer_source_sha256"] == TOKENIZER_SHA and
           m["original_model"]["original_backbone_state_sha256"] == FROZEN_FINGERPRINT
           and m["original_model"]["model_execution"] is True and
           m["original_model"]["supervised_forward_only"] is True,
           "original host source claims drifted")
    for actual, expected in zip(m["original_model"]["cases"], CASES):
        refuse(actual["name"] == expected[0] and
               actual["causal_supervised_units"] == expected[4] and
               actual["loss"] == expected[3] and
               actual["logits_sha256"] == expected[5],
               "model receipt and manifest disagree")
    return {
        "status": "PASS_ORIGINAL_BYTES_AND_SOURCE_ONLY_FORWARD_LOSS__72_BLOCKED",
        "source_head": SOURCE_HEAD, "run": 40738222796,
        "original_artifact_zip_sha256": ORIGINAL_ARCHIVE_SHA,
        "original_report_sha256": REPORT_SHA,
        "source_tests_passed": 30, "real_frozen_model_cases": len(CASES),
        "backbone_unchanged": True, "train_authorized": False,
        "full_72_conformance_pass": False, "independent_scientific_review": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(audit(), indent=2, sort_keys=True))
    except Exception as exc:
        print("ORIGINAL_QWEN_EVIDENCE_REFUSED:", type(exc).__name__, str(exc),
              file=sys.stderr)
        raise
