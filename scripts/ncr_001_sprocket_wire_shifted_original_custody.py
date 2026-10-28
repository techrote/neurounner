#!/usr/bin/env python3
"""Independent offline custody of original NCR-001 Sprocket causal wire source run.

Reconstitutes the ORIGINAL GitHub Actions artifact (including 5 unaltered
members) and both full decoded GitHub Actions job logs. No weights, tokenizer,
optimizer, network, heldouts, quality analysis, or scientific admission.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/sprocket-wire-shifted-loss-20261028"
HEAD = "adfc3cb3ba2b2b6b54b0b10e3aa1ae19d4ff10ba"
FAILED_HEAD = "e6673ff5b53ec702799d4134181dc55b49692d34"
ZIP_SHA = "f077715068453c3a043b4269dd020ecca5f369770539116d88b39c806db65520"
XZ_SHA = "8305071494df3450ae2d6fa2e62700ce0bc355af3609357cc55be904a9616b94"
ZIP_BLOB = "b62955d7eca53ae6933b10fddd0bdd7689374e0d"
FAILED_LOG_BLOB = "5a8324c5e33f38f0098fe9cd95699a3d98bd772b"
PASSED_LOG_BLOB = "7e824e04f542de36ad4c372c58fe921c55afdaa1"
MEMBERS = {
    "install.log": (5251, "7f35101d483358774f432847026c3a2650b9bfd043aafb9e34b5bbdb7f27924e"),
    "original-custody-readback.json": (964, "b3a36f30f356981db8cde41d24a72b326d96a2cf16afec857bcf64a945d884ea"),
    "real-tokenizer.log": (2350, "71fcbe8652bf5cc23c35be3d7d9b94c081dcdb7dff4d718fa111212b86ece3ca"),
    "receipt.json": (2405, "0f2a051cc81a3eb77f8af0ac7a959b0e5743e11cb4def12afaebc65db1e14909"),
    "source-tests.log": (2333, "10a129310946ffc55998314ad0b830a21c9d3ed64f6eae6bae0d8f40b06fb4f7"),
}
TARGET = "f8453abe8ee1780ecd9f536a90f6eb1cc114af7e704f52110889165ffecc85a4"
ATTENTION = "fafee010cc27daeb6996b93f5d110b7943b85d8ee704032f80d1ef7be5e7b4cd"
UNSHIFTED = "24b4289e273e579d166411baae776ec83d2d534036e595eb47322681c792d97c"


def demand(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("NCR001_SHIFTED_ORIGINAL_CUSTODY_REFUSED: " + reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def original_zip(evidence: Path = EVIDENCE) -> bytes:
    raw = (evidence / "original-actions-zip.xz.base64").read_bytes()
    demand(blob(raw) == ZIP_BLOB, "Git original compressed artifact blob changed")
    encoded = raw.strip()
    demand(len(encoded) == 6560, "archived original base64 length changed")
    compressed = base64.b64decode(encoded, validate=True)
    demand(len(compressed) == 4920 and sha(compressed) == XZ_SHA,
           "compressed original Actions archive digest changed")
    uncompressed = lzma.decompress(compressed, memlimit=128 * 1024 * 1024)
    demand(len(uncompressed) == 5179 and sha(uncompressed) == ZIP_SHA,
           "original Actions ZIP not byte-identical")
    return uncompressed


def check_receipt(r: dict, m: dict) -> None:
    expected = m["receipt"]
    demand(r["status"] == expected["status"] ==
           "PASS_CAUSAL_SHIFTED_MASK_SOURCE_ONLY__FINAL_72_BLOCKED",
           "original result state changed")
    demand(r["source_head"] == expected["source_head"] == HEAD and
           r["lane"] == "sprocket_instruct" and
           r["experiment"] == "NCR-001E", "original root Instruct model identity changed")
    for field in ("long_units", "short_units", "assistant_supervised_units",
                  "real_end_id", "real_end_units_long"):
        demand(r[field] == expected[field], "source lengths, end ID, or loss units changed: " + field)
    demand(r["long_units"] == 123 and r["short_units"] == 26
           and r["assistant_supervised_units"] == 45
           and r["real_end_id"] == 5 and r["real_end_units_long"] == 12
           and r["synthetic_left_pad_units"] == expected["left_pad_units"] == 97,
           "original all-token mask numbers changed")
    for key, target in (("causal_shifted_labels_sha256", TARGET),
                        ("padded_attention_sha256", ATTENTION),
                        ("unshifted_labels_sha256", UNSHIFTED)):
        demand(r[key] == expected[key] == target,
               "source causal/attention/label SHA changed: " + key)
    demand(r["naive_pad_equals_end_refused"] is True
           and expected["naive_pad_equals_end_refused"] is True,
           "false acceptance of ID-5 end/padding equivalence")
    for field in ("train_authorized", "full_72_conformance_pass",
                  "independent_scientific_review", "new_model_execution",
                  "model_weights_downloaded", "comparative_training",
                  "heldouts_opened", "generated_tool_memory_behaviour_qualified"):
        demand(r[field] is False and m["scientific_gates"][
            ("new_physical_model_execution" if field == "new_model_execution"
             else "model_generated_tool_memory_semantics_qualified" if field ==
             "generated_tool_memory_behaviour_qualified" else field)] is False,
             "false scientific/model/training admission: " + field)
    demand(r["optimizer_steps"] == m["scientific_gates"]["optimizer_steps"] == 0
           and r["historical_incomplete_tool_call_preserved"] is True
           and m["scientific_gates"]["historical_incomplete_six_token_tool_call_retained"] is True
           and m["scientific_gates"]["historical_void_byte_eos_effect_measured"] is False,
           "historical failure or optimizer activity hidden")
    demand(r["runner"]["run_id"] == str(m["attempts"]["passed"]["run_id"]) ==
           "40788864348"
           and r["runner"]["name"] == m["attempts"]["passed"]["runner_name"] ==
           "GitHub Actions 1000047035"
           and r["runner"]["python"] == "3.12.3",
           "original CI runner identity changed")
    demand(r["original_physical_run"] == 40718706396
           and r["original_source_fullmask_run"] == 40782101613
           and r["original_custody_run"] == 40783012701,
           "original physical/source/custody chain changed")
    demand(r["source"]["experiments/ncr-001/protocol.json"] ==
           m["source_gates"]["protocol_git_blob"] ==
           "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
           and r["source"]["experiments/ncr-001/source-pins.json"] ==
           m["source_gates"]["source_pins_git_blob"] ==
           "59fff1aed872b5563aa6898171983973e0cdb02c",
           "original four-model protocol/source pins changed")
    demand(m["source_gates"]["model_variants"] ==
           ["smollm2", "sprocket_chat", "sprocket_instruct", "void1"]
           and m["source_gates"]["sprocket_chat_weight_file"] == "chat/model.safetensors"
           and m["source_gates"]["sprocket_instruct_weight_file"] == "model.safetensors",
           "frozen Chat and Instruct identities collapsed")


def audit(evidence: Path = EVIDENCE, live_zip: bytes | None = None) -> dict:
    m = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    demand(m["schema"] == "ncr001-sprocket-wire-shifted-loss-original-custody-v1"
           and m["lane"] == "sprocket_instruct"
           and m["status"] == "ORIGINAL_SOURCE_TOKENIZER_ONLY__FINAL_72_SCIENCE_BLOCKED"
           and m["parent_pr"] == 184 and m["parent_head"] == HEAD,
           "original PR or custody identity changed")
    attempts = m["attempts"]
    failed, passed = attempts["failed_checkout"], attempts["passed"]
    demand(failed["run_id"] == 40788806988 and failed["job_id"] == 123168595600
           and failed["source_head"] == FAILED_HEAD
           and failed["conclusion"] == "failure"
           and failed["phase"] == "actions_checkout_before_any_tests"
           and passed["run_id"] == 40788864348 and passed["job_id"] == 123168786515
           and passed["source_head"] == HEAD and passed["conclusion"] == "success"
           and passed["source_tests_passed"] == 14,
           "original failed or corrected CI identities changed")
    for record, git_blob, must_include in (
        (failed, FAILED_LOG_BLOB, b"could not be found"),
        (passed, PASSED_LOG_BLOB, b"Ran 14 tests in"),
    ):
        raw = (evidence / record["decoded_job_log"]).read_bytes()
        demand(blob(raw) == record["git_blob_sha"] == git_blob
               and must_include in raw,
               "original complete failed/success CI log missing or altered")
    source = (evidence / failed["decoded_job_log"]).read_bytes()
    demand(b"Ran 14 tests in" not in source, "failed setup misreported as tests run")
    log = (evidence / passed["decoded_job_log"]).read_bytes()
    demand(b"PASS_CAUSAL_SHIFTED_MASK_SOURCE_ONLY__FINAL_72_BLOCKED" in log
           and HEAD.encode() in log and b"GitHub Actions 1000047035" in log,
           "corrected CI job lacks source/tokenizer witness")
    a = m["artifact"]
    demand(a["id"] == 12859838740 and a["run_id"] == 40788864348
           and a["zip_bytes"] == 5179 and a["zip_sha256"] == ZIP_SHA
           and a["archive_git_blob"] == ZIP_BLOB
           and a["xz_bytes"] == 4920 and a["xz_sha256"] == XZ_SHA,
           "original Actions artifact identity changed")
    original = original_zip(evidence)
    if live_zip is not None:
        demand(original == live_zip, "Git Actions ZIP bytes not equal to original GitHub artifact")
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        all_info = z.infolist()
        demand(len(all_info) == len(MEMBERS)
               and set(z.namelist()) == set(MEMBERS)
               and len(z.namelist()) == len(set(z.namelist())),
               "original Actions ZIP member list changed")
        for name, (size, digest) in MEMBERS.items():
            info = z.getinfo(name)
            demand(not info.is_dir() and info.file_size == size
                   and z.read(name) and sha(z.read(name)) == digest
                   and a["members"][name] == {"bytes": size, "sha256": digest},
                   "original ZIP member corruption or manifest mutation: " + name)
        receipt = json.loads(z.read("receipt.json"))
        check_receipt(receipt, m)
        tests = z.read("source-tests.log")
        demand(b"Ran 14 tests in" in tests and b"OK" in tests,
               "source adversarial tests no longer evidenced")
        custody = json.loads(z.read("original-custody-readback.json"))
        demand(custody["train_authorized"] is False
               and custody["new_model_execution"] is False
               and custody["full_72_conformance_pass"] is False,
               "upstream original custody incorrectly promoted")
    return {
        "status": "PASS_COMPLETE_ORIGINAL_ZIP_LOGS_AND_NEGATIVE_RESULT__72_BLOCKED",
        "original_pr": 184, "original_head": HEAD,
        "original_artifact_id": 12859838740,
        "original_actions_run": 40788864348,
        "original_failed_checkout_run": 40788806988,
        "zip_sha256": ZIP_SHA, "members": len(MEMBERS),
        "source_tests_passed": 14, "loss_sha256": TARGET,
        "real_end_attention_sha256": ATTENTION,
        "new_model_execution": False, "train_authorized": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
