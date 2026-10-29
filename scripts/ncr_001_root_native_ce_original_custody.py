#!/usr/bin/env python3
"""NCR-001E read-only validation and immutable original ZIP promotion (#188).

--stage accepts ONLY the original Actions ZIP, pins all six member hashes,
and writes a losslessly encoded archive and fixed provenance on a descendant
draft branch. --check verifies committed immutable original evidence offline.
Neither mode loads a model, opens heldouts, runs backward, or trains.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/root-native-ce-20261029"
ARCHIVE = "original-actions-zip.xz.base64"
MANIFEST = "manifest.json"
JOB = "decoded-job-123178045642.log"
ORIGINAL_SHA = "85e6499b53efea03be5a331d97519e6fda1f84d4a0a944b27bb22d3a7bddb4eb"
SOURCE = "1178b896848df0984a0670180c62d6f8e9b9a512"
ROOT_WEIGHT = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
CHAT_WEIGHT = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
JOB_BLOB = "43ffa04c411f4672ed04c693cab5ece36a462f69"
MEMBERS = {
    "install-tokenizer.log": (5030, "8c1d73d82a07d6c6ac7051121dcc9b3a7ffaa1183e11b36fb06442d779d0c006"),
    "install-torch.log": (2828, "c34a1795bcea72216f615f3f22bd4d8ac44eb017bfdff18a965e2dbe860b061c"),
    "native-host.log": (3181, "8691f89584b80324101d79b2d9cc273dd1c26c398137c6a034aa9ee04e6ea2ff"),
    "negative-tests.log": (992, "b447e481794de37b67f792ef65579ae54a3acfe2085039f18b5f921b530b375a"),
    "original-shifted-custody.json": (745, "73103ff135f24cefc62d313129974520be26469fd0d10aa30dd8bb1bad5a7e4d"),
    "receipt.json": (3543, "968c44d19d982a417b562b0a34af51b75281f9e0bb598e85d45b00fb03c8ffe7"),
}


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("NCR001_ROOT_CE_ORIGINAL_REFUSED: " + reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()


def validate_receipt(r: dict) -> dict:
    refuse(r["status"] == "PASS_FROZEN_ROOT_INSTRUCT_SYNTHETIC_MASKED_CE_ONLY__72_BLOCKED"
           and r["experiment"] == "NCR-001E" and r["lane"] == "sprocket_instruct",
           "original root-only status/lane changed")
    for key in ("train_authorized", "full_72_conformance_pass",
                "heldouts_opened", "independent_scientific_review",
                "comparative_training", "new_model_training",
                "model_generated_tool_memory_semantics_qualified"):
        refuse(r[key] is False, "false admission: " + key)
    refuse(r["optimizer_steps"] == 0 and r["original_frozen_model_executed_this_run"] is True
           and r["original_incomplete_tool_call_preserved"] is True,
           "physical execution/negative/optimizer count changed")
    src = r["source"]
    refuse(src["root_weight_sha256"] == ROOT_WEIGHT
           and src["excluded_chat_weight_sha256"] == CHAT_WEIGHT
           and src["protocol_git_blob"] == "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
           and src["source_pins_git_blob"] == "59fff1aed872b5563aa6898171983973e0cdb02c",
           "frozen source variants or protocol identity changed")
    v = r["verified_source_files"]["model.safetensors"]
    refuse(v == {"bytes": 1002207984, "sha256": ROOT_WEIGHT},
           "physical 1002207984-byte root checkpoint hash changed")
    rt = r["runner"]
    refuse(rt["run_id"] == "40791640014"
           and rt["name"] == "GitHub Actions 1000047091"
           and rt["host"] == "runnervm48cm3"
           and rt["pid"] == 2714
           and rt["torch"] == "2.14.1+cpu"
           and rt["transformers"] == "4.57.6",
           "original host/runner/version identity changed")
    nc = r["native_ce"]
    refuse(nc["supervised_target_count"] == 45 and nc["finite"] is True
           and nc["native"] == 3.735790967941284
           and nc["manual_logsumexp"] == 3.7357990741729736
           and nc["abs_difference"] == 8.106231689453125e-06,
           "original real-host numeric CE receipt changed")
    refuse(nc["abs_difference"] <= 1e-5, "unqualified loss mismatch")
    for lane, params in (("nca", 658904), ("mlp", 658432)):
        c = r["controls"][lane]
        refuse(c == {"adapter_untrained": True, "disabled_logits_exact": True,
                      "loss": nc["native"], "native_ce_exact": True,
                      "trainable_parameters": params, "zero_logits_exact": True},
               "untrained no-op hook observation changed: " + lane)
    tr = r["transport"]
    refuse(tr["batch_shape"] == [2, 123] and tr["synthetic_left_pad_units"] == 97
           and tr["supervised_after_causal_shift"] == 45
           and tr["original_attention_sha256"] ==
               "fafee010cc27daeb6996b93f5d110b7943b85d8ee704032f80d1ef7be5e7b4cd"
           and tr["original_shifted_targets_sha256"] ==
               "f8453abe8ee1780ecd9f536a90f6eb1cc114af7e704f52110889165ffecc85a4",
           "pinned original attention/causal target hash changed")
    refuse(r["frozen_backbone_before_sha256"] ==
           r["frozen_backbone_after_sha256"] ==
               "2b74a745899820d912879734cfb50269c6ec8ea563263e7818c5a734b511d399",
           "frozen original backbone fingerprint changed")
    return {"ce": nc["native"], "targets": 45, "train_authorized": False,
            "full_72_conformance_pass": False, "original_physical_model_executed": True}


def validate_zip(original: bytes) -> dict:
    refuse(len(original) == 6277 and sha(original) == ORIGINAL_SHA,
           "original GitHub Actions ZIP container bytes/hash changed")
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        names = z.namelist()
        refuse(len(names) == len(MEMBERS) and len(set(names)) == len(names)
               and set(names) == set(MEMBERS), "missing/extra/duplicated original members")
        for name in names:
            item = z.getinfo(name)
            refuse(not item.is_dir() and not name.startswith("/")
                   and ".." not in Path(name).parts,
                   "unsafe original ZIP path")
            raw = z.read(name)
            n, digest = MEMBERS[name]
            refuse(len(raw) == n and sha(raw) == digest,
                   "original member changed: " + name)
        receipt = json.loads(z.read("receipt.json"))
        validated = validate_receipt(receipt)
        native = z.read("native-host.log")
        refusal = z.read("negative-tests.log")
        refuse(b"PASS_FROZEN_ROOT_INSTRUCT_SYNTHETIC_MASKED_CE_ONLY__72_BLOCKED" in native
               and b"40791640014" in native and b"Ran 6 tests" in refusal
               and b"OK" in refusal, "original native/negative witness missing")
        return {"source_head": SOURCE, "original_zip_sha256": ORIGINAL_SHA,
                "original_zip_bytes": len(original), "raw_members": len(names),
                "original_receipt_sha256": MEMBERS["receipt.json"][1], **validated}


def immutable_manifest() -> dict:
    return {
        "schema": "ncr001-root-instruct-native-ce-original-zip-custody-v1",
        "scope": "ORIGINAL_ACTUAL_FROZEN_ROOT_SYNTHETIC_CE__FINAL_72_BLOCKED",
        "original_pr": 188,
        "original_head": SOURCE,
        "original_run": 40791640014,
        "original_job": 123178045642,
        "original_artifact_id": 12861059281,
        "original_artifact_name": "ncr001-root-native-ce-a1",
        "original_artifact_expiry_utc": "2026-11-28T00:06:01Z",
        "original_artifact_zip_bytes": 6277,
        "original_artifact_zip_sha256": ORIGINAL_SHA,
        "original_members": {k: {"bytes": n, "sha256": h} for k, (n, h) in MEMBERS.items()},
        "original_complete_decoded_job_log": JOB,
        "original_complete_decoded_job_git_blob": JOB_BLOB,
        "weight_sha256_root_instruct": ROOT_WEIGHT,
        "excluded_chat_weight_sha256": CHAT_WEIGHT,
        "source_protocol_blob": "14ce5fbaa38f8806c8c8738f19c01b91385eebaf",
        "source_pin_blob": "59fff1aed872b5563aa6898171983973e0cdb02c",
        "inherited_original_tool_call_prefix_incomplete": True,
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "optimizer_steps": 0,
        "heldouts_opened": False,
        "comparative_training": False,
        "new_model_training": False,
    }


def audit(evidence: Path = EVIDENCE) -> dict:
    found = json.loads((evidence / MANIFEST).read_text(encoding="utf-8"))
    refuse(found == immutable_manifest(), "original manifest changed, including negative results")
    logged = (evidence / JOB).read_bytes()
    refuse(gitblob(logged) == JOB_BLOB and
           b"PASS_FROZEN_ROOT_INSTRUCT_SYNTHETIC_MASKED_CE_ONLY__72_BLOCKED" in logged
           and b"Ran 6 tests" in logged,
           "full original physical GitHub job log changed")
    b64 = (evidence / ARCHIVE).read_bytes().strip()
    compressed = base64.b64decode(b64, validate=True)
    original = lzma.decompress(compressed, memlimit=64*1024*1024)
    result = validate_zip(original)
    return {"status": "PASS_ORIGINAL_FULL_ZIP_AND_LOG_GIT_CUSTODY__72_BLOCKED",
            "decoded_full_job_blob": JOB_BLOB,
            "source_and_original_zip": result, "train_authorized": False,
            "full_72_conformance_pass": False, "new_model_executed": False,
            "optimizer_steps": 0}


def stage(input_path: Path, evidence: Path = EVIDENCE) -> dict:
    original = input_path.read_bytes()
    validated = validate_zip(original)
    refuse(gitblob((evidence / JOB).read_bytes()) == JOB_BLOB,
           "original full GitHub job log changed")
    evidence.mkdir(parents=True, exist_ok=True)
    original_b64 = base64.b64encode(lzma.compress(original, preset=9 | lzma.PRESET_EXTREME)) + b"\n"
    target = evidence / ARCHIVE
    manifest_file = evidence / MANIFEST
    refuse(not target.exists() or target.read_bytes() == original_b64,
           "cannot replace an existing altered original ZIP archive")
    expected = json.dumps(immutable_manifest(), indent=2, sort_keys=True) + "\n"
    refuse(not manifest_file.exists() or manifest_file.read_text(encoding="utf-8") == expected,
           "cannot replace an altered historical manifest")
    target.write_bytes(original_b64)
    manifest_file.write_text(expected, encoding="utf-8")
    refuse(audit(evidence)["source_and_original_zip"] == validated,
           "post-stage independent custody self-readback mismatch")
    return validated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, help="one ORIGINAL downloaded Actions ZIP")
    opts = ap.parse_args()
    r = stage(opts.stage) if opts.stage else audit()
    print(json.dumps(r, sort_keys=True))


if __name__ == "__main__":
    main()
