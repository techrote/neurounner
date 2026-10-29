#!/usr/bin/env python3
"""NCR-001E: exact original Sprocket CHAT synthetic native-CE Actions ZIP custody.

Never load a model, train, modify original receipts, inspect heldouts or promote
preliminary source/readback to #72 scientific acceptance. The only accepted ZIP
is the original 6,735-byte Actions artifact identified by independent SHA-256,
six original ZIP member SHA-256s, original physical source/runner hashes,
frozen Chat (not root Instruct) and a closed scientific training gate.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/chat-native-ce-20261029"
BUNDLE = "original-actions-zip.base64"
MANIFEST = "manifest.json"
JOB_LOG = "original-physical-job-123202104527.log"
JOB_BLOB = "05c88bb48111b597d53fce011adfb2a0fa1de38c"
ORIGINAL_BUNDLE_GIT_BLOB = "daac05623ff4c8360ef89a040ec7a536e58b2286"
ORIGINAL_MANIFEST_GIT_BLOB = "1640210cf26e4c5cc3f6d3bc3b952f6830e565ec"
ORIGINAL_CUSTODY_BOOTSTRAP_LOG = "original-custody-bootstrap-job-123204676106.log"
ORIGINAL_CUSTODY_BOOTSTRAP_GIT_BLOB = "5a7273716360ca3ab988206e316b9d0d8a417546"
ORIGINAL_READONLY_LOG = "original-readonly-job-123205777828.log"
ORIGINAL_READONLY_LOG_GIT_BLOB = "fc71281fa1fbc5ed4ecaeb52f69ab9616b4e4e1b"
ORIGINAL_PR = 191
ORIGINAL_HEAD = "6ae912c02c8f9a44bde606b5daa2c73ff8f70567"
ORIGINAL_RUN = 40798938605
ORIGINAL_JOB = 123202104527
ORIGINAL_ARTIFACT = 12865680062
ORIGINAL_ZIP_SIZE = 6735
ORIGINAL_ZIP_SHA256 = "69a8c342fb98e266e1cdf1867e4b256968f403c0702255fe6a18c320ab6ca945"
CHAT_SHA = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
INSTRUCT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
FORMAT_HEAD = "2271218d625e1462b2cf218445da975efe17fe32"
PARENT_CUSTODY_HEAD = "5794964f2ef8f9640669b77d7da74cfcec4d9ddb"
STATUS = "PASS_ORIGINAL_FROZEN_CHAT_CE_ZIP_AND_JOB_CUSTODY__FINAL_72_BLOCKED"
MEMBERS = {
    "install-torch.log": (2828, "e22706889e57350a2e24c1de3385fc5b4afa5c0bea905c5cc653666c5c898cf7"),
    "original-postlease-custody.json": (455, "f8aed9b63ab1b5c8531982d8fd234040d88158f72848c82e40e34517208ccca0"),
    "install-source.log": (5030, "6eb602751c6dd1040a3ada2cee66e5a3f1f0e141e78d19511e59fd9e32e86a67"),
    "physical-host.log": (3726, "bd608b6a81dee3ba45cf95c5f04545ffd55f3d5d20b0ec15eef9131a34868146"),
    "receipt.json": (4200, "a0c80adcad1a53f44f72d361039c35b9e83008aa83bb1b3458fd7e9aecbe9004"),
    "source-tests.log": (1827, "a0078d7bba7382b0d67146ec54be529e358adfdfcba73aa82f9c3c6c7b0ad6e8"),
}


def refuse(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("NCR001_CHAT_ORIGINAL_CUSTODY_REFUSED: " + message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") +
                        b"\x00" + raw).hexdigest()


def validate_receipt(r: dict) -> dict:
    refuse(r["schema_version"] == 1
           and r["status"] ==
               "PASS_FROZEN_CHAT_ORDINARY_SYNTHETIC_ASSISTANT_CE_ONLY__72_BLOCKED"
           and r["experiment"] == "NCR-001E"
           and r["lane"] == "sprocket_chat"
           and r["frozen_original_chat_executed"] is True,
           "not original physical Sprocket Chat synthetic CE")
    for flag in ("train_authorized", "full_72_conformance_pass",
                 "independent_scientific_review", "tool_memory_behaviour_qualified",
                 "heldouts_opened", "comparative_nca_mlp_training",
                 "trained_checkpoint_created"):
        refuse(r[flag] is False, "false scientific or training claim: " + flag)
    refuse(r["optimizer_steps"] == 0
           and r["historical_root_instruct_incomplete_tool_call_retained"] is True,
           "original negative result or optimizer count lost")
    refuse(r["source_head"] == ORIGINAL_HEAD
           and r["original_postlease_custody_parent"] == PARENT_CUSTODY_HEAD
           and r["original_formatter_source_head"] == FORMAT_HEAD,
           "original source/formatting lineage changed")
    p = r["sources"]
    refuse(p["protocol_git_blob"] == PROTOCOL_BLOB
           and p["source_pins_git_blob"] == PINS_BLOB
           and p["chat_checkpoint_sha256"] == CHAT_SHA
           and p["excluded_root_instruct_checkpoint_sha256"] == INSTRUCT_SHA,
           "original Chat source, separate sibling or protocol pins changed")
    wt = r["full_verified_source_files"]["chat/model.safetensors"]
    refuse(wt == {"bytes": 1002207984, "sha256": CHAT_SHA}
           and "model.safetensors" not in r["full_verified_source_files"],
           "wrong model or missing full physical Chat weight verification")
    for fname, expected in (
        ("README.md", "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"),
        ("chat/config.json", "557aacbcb7e23a322221ab91b62c4343dd22130388efc3eb4dcdfe1edfd8d1b7"),
        ("chat/tokenizer.json", "178a63d94a7de4aa273a6a9c218180c3b7d1912f5a18a91be34bab252f7277eb"),
        ("chat/tokenizer_config.json", "1d1c0733848e0c78a2ec48a7f20cd55ec5ee0b4cbcda9785249af9e489823408"),
    ):
        refuse(r["full_verified_source_files"][fname]["sha256"] == expected,
               "source model card/tokenizer/config changed: " + fname)
    runner = r["runner"]
    refuse(runner["run"] == str(ORIGINAL_RUN)
           and runner["name"] == "GitHub Actions 1000047257"
           and runner["host"] == "runnervm48cm3" and runner["pid"] == 2477
           and runner["torch"] == "2.14.1+cpu"
           and runner["transformers"] == "4.57.6"
           and runner["python"] == "3.12.3"
           and runner["os"] == "Linux",
           "original physical runner or package identity changed")
    ce = r["ce"]
    refuse(ce["native"] == 4.101235866546631
           and ce["independent"] == 4.101239204406738
           and ce["abs_difference"] == 3.337860107421875e-06
           and ce["targets"] == 16
           and ce["finite"] is True
           and ce["tolerance_ce_only"] == 1e-5,
           "original physical native/independent CE numbers changed")
    for control, n in (("nca", 658904), ("mlp", 658432)):
        x = r["controls"][control]
        refuse(x == {"adapter_untrained": True,
                     "disabled_logits_exact": True, "optimizer_steps": 0,
                     "trainable_adapter_parameters": n,
                     "zero_and_disabled_native_loss_exact": True,
                     "zero_logits_exact": True},
               "frozen Chat matched-control no-op status changed: " + control)
    refuse(r["backbone_before_sha256"] ==
           r["backbone_after_sha256"] ==
               "828166c92e6ebc43aa5e9c0643da22389e77de19bf6ec6170d5e94b150e985ff",
           "frozen original full-tensor fingerprint changed")
    trans = r["synthetic_transport"]
    refuse(trans["shape"] == [2, 37] and trans["left_padding"] == [19, 0]
           and trans["supervised_targets"] == 16
           and trans["attended_true_end_id5"] is True
           and trans["input_ids_sha256"] ==
               "951364b925047f0b742d2acf5ba61e5179f2069b8e78019e969ea166bd8892f8"
           and trans["attention_mask_sha256"] ==
               "2d3508b3ff52203eccd28feb29c2a36ba9124329de01105403ccdfa386f9da0b"
           and trans["labels_sha256"] ==
               "deeddb47562fc8c6791758d5f7b9cadab4ffa1f626dea1adc2ceea28e16ff7a3",
           "original synthetic Chat token/attention/causal label arrays changed")
    refuse(r["synthetic_only_fixture_sha256"] ==
           "787d7cd150f1a1d4a62197aa49f9474c04e5f7b5b4491afa4a3e57c872c62904",
           "source synthetic diagnostic identity changed")
    items = trans["case_sources"]
    refuse(len(items) == 2 and
           [(p["prompt_units"], p["supervised_units"]) for p in items] ==
               [(10, 8), (29, 8)],
           "changed Chat ordinary dialogue or assistant target count")
    return {
        "original_physical_chat_forward": True,
        "source_head": ORIGINAL_HEAD,
        "native_ce": ce["native"], "independent_ce": ce["independent"],
        "assistant_supervised_targets": 16,
        "train_authorized": False, "full_72_conformance_pass": False,
        "optimizer_steps": 0,
    }


def validate_zip(original: bytes) -> dict:
    refuse(len(original) == ORIGINAL_ZIP_SIZE and sha(original) == ORIGINAL_ZIP_SHA256,
           "original GitHub Actions ZIP complete container bytes changed")
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        names = z.namelist()
        refuse(len(names) == len(MEMBERS) == len(set(names))
               and set(names) == set(MEMBERS),
               "original Actions ZIP members missing/duplicated/unexpected")
        for name in names:
            info = z.getinfo(name)
            refuse(not info.is_dir() and not name.startswith("/")
                   and ".." not in Path(name).parts and info.file_size < 100000,
                   "unsafe original ZIP member")
            raw = z.read(name)
            size, checksum = MEMBERS[name]
            refuse(len(raw) == size and sha(raw) == checksum,
                   "original ZIP member bytes/hash changed: " + name)
        result = validate_receipt(json.loads(z.read("receipt.json")))
        logs = z.read("physical-host.log")
        tests = z.read("source-tests.log")
        refuse(b"PASS_FROZEN_CHAT_ORDINARY_SYNTHETIC_ASSISTANT_CE_ONLY__72_BLOCKED" in logs
               and b"40798938605" in logs and b"Ran 11 tests" in tests
               and b"\nOK" in tests,
               "original frozen Chat host result or 11 source sentinels lost")
        return {
            "status": "PASS_ORIGINAL_CHAT_ZIP_SOURCE_RECEIPT__72_BLOCKED",
            "original_zip_sha256": ORIGINAL_ZIP_SHA256,
            "original_zip_bytes": ORIGINAL_ZIP_SIZE,
            "original_raw_members": len(MEMBERS),
            "original_receipt_sha256": MEMBERS["receipt.json"][1], **result,
        }


def fixed_manifest() -> dict:
    return {
        "schema": "ncr001-chat-native-ce-original-zip-custody-v1",
        "experiment": "NCR-001E",
        "scope": "ORIGINAL_PHYSICAL_FROZEN_CHAT_ASSISTANT_CE_ONLY__FINAL_72_BLOCKED",
        "original_pr": ORIGINAL_PR, "original_head": ORIGINAL_HEAD,
        "original_run": ORIGINAL_RUN, "original_job": ORIGINAL_JOB,
        "original_artifact_id": ORIGINAL_ARTIFACT,
        "original_artifact_name": "ncr001-chat-native-ce-a1",
        "original_artifact_expiry_utc": "2026-11-28T01:13:04Z",
        "original_zip_sha256": ORIGINAL_ZIP_SHA256,
        "original_zip_bytes": ORIGINAL_ZIP_SIZE,
        "raw_members": {n: {"bytes": x[0], "sha256": x[1]} for n, x in MEMBERS.items()},
        "original_complete_job_log_path": JOB_LOG,
        "original_complete_job_log_git_blob": JOB_BLOB,
        "root_instruct_checkpoint_explicitly_excluded": INSTRUCT_SHA,
        "original_chat_checkpoint": CHAT_SHA,
        "protocol_git_blob": PROTOCOL_BLOB,
        "source_pins_git_blob": PINS_BLOB,
        "original_formatter_source": FORMAT_HEAD,
        "negative_original_root_tool_call_preserved": True,
        "train_authorized": False, "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "heldouts_opened": False, "comparative_training": False,
        "optimizer_steps": 0, "trained_checkpoint_created": False,
    }


def audit(evidence: Path = EVIDENCE) -> dict:
    manifest_raw = (evidence / MANIFEST).read_bytes()
    refuse(gitblob(manifest_raw) == ORIGINAL_MANIFEST_GIT_BLOB,
           "original source-bound committed manifest raw Git blob changed")
    manifest = json.loads(manifest_raw)
    refuse(manifest == fixed_manifest(),
           "committed original source/run/negative-evidence manifest changed")
    raw_job = (evidence / JOB_LOG).read_bytes()
    refuse(gitblob(raw_job) == JOB_BLOB
           and b"PASS_FROZEN_CHAT_ORDINARY_SYNTHETIC_ASSISTANT_CE_ONLY__72_BLOCKED" in raw_job
           and b"Ran 11 tests" in raw_job,
           "complete decoded original physical job log bytes changed")
    bootstrap = (evidence / ORIGINAL_CUSTODY_BOOTSTRAP_LOG).read_bytes()
    refuse(gitblob(bootstrap) == ORIGINAL_CUSTODY_BOOTSTRAP_GIT_BLOB
           and b"Ran 14 tests" in bootstrap
           and b"PASS_ORIGINAL_CHAT_FULL_ZIP_AND_JOB_LOG_CUSTODY__72_BLOCKED" in bootstrap,
           "original exact-head archive promotion job-log bytes changed")
    previous_readonly = (evidence / ORIGINAL_READONLY_LOG).read_bytes()
    refuse(gitblob(previous_readonly) == ORIGINAL_READONLY_LOG_GIT_BLOB
           and b"Ran 16 tests" in previous_readonly
           and b"IMMUTABLE_ORIGINAL_CHAT_ARCHIVE__READ_ONLY" in previous_readonly
           and b"PASS_ORIGINAL_CHAT_FULL_ZIP_AND_JOB_LOG_CUSTODY__72_BLOCKED" in previous_readonly,
           "complete earlier exact-head read-only CI log bytes or checks changed")
    try:
        raw_bundle = (evidence / BUNDLE).read_bytes()
        refuse(gitblob(raw_bundle) == ORIGINAL_BUNDLE_GIT_BLOB,
               "original complete ZIP ASCII/base64 Git blob changed")
        original_zip = base64.b64decode(raw_bundle.strip(), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("NCR001_CHAT_ORIGINAL_CUSTODY_REFUSED: unparseable original archive") from exc
    result = validate_zip(original_zip)
    return {"status": "PASS_ORIGINAL_CHAT_FULL_ZIP_AND_JOB_LOG_CUSTODY__72_BLOCKED",
            "original": result, "decoded_job_git_blob": JOB_BLOB,
            "original_bootstrap_job_git_blob": ORIGINAL_CUSTODY_BOOTSTRAP_GIT_BLOB,
            "original_readonly_job_git_blob": ORIGINAL_READONLY_LOG_GIT_BLOB,
            "original_raw_zip_archive_git_blob": ORIGINAL_BUNDLE_GIT_BLOB,
            "original_manifest_git_blob": ORIGINAL_MANIFEST_GIT_BLOB,
            "train_authorized": False, "full_72_conformance_pass": False,
            "new_physical_model_executed_this_audit": False, "optimizer_steps": 0}


def stage(input_zip: Path, evidence: Path = EVIDENCE) -> dict:
    original_zip = input_zip.read_bytes()
    result = validate_zip(original_zip)
    refuse(gitblob((evidence / JOB_LOG).read_bytes()) == JOB_BLOB,
           "retained original physical job log mismatch")
    evidence.mkdir(parents=True, exist_ok=True)
    target = evidence / BUNDLE
    manifest = evidence / MANIFEST
    encoded = base64.b64encode(original_zip) + b"\n"
    expected_manifest = json.dumps(fixed_manifest(), indent=2, sort_keys=True) + "\n"
    refuse(not target.exists() or target.read_bytes() == encoded,
           "existing original ZIP custody changed: refusing replacement")
    refuse(not manifest.exists() or
           manifest.read_text(encoding="utf-8") == expected_manifest,
           "existing original manifest changed: refusing replacement")
    target.write_bytes(encoded)
    manifest.write_text(expected_manifest, encoding="utf-8")
    refuse(audit(evidence)["original"] == result,
           "independent staged archive readback failed")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=Path, help="original downloaded GitHub ZIP")
    args = p.parse_args()
    result = stage(args.stage) if args.stage else audit()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
