#!/usr/bin/env python3
"""Offline whole-container custody of PR #159 original Sprocket diagnostic ZIPs.

Source, Chat and root Instruct are distinct; does NOT execute any model,
optimizer, tokenizer or held-out fixture or confer #72 scientific acceptance.
"""
from __future__ import annotations
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/sprocket-learning-20261028"
ORIGINAL_HEAD = "df8479aae34550cba014d910a73e30e59cd493e2"
ORIGINAL_RUN = 40710171622
MANIFEST_GIT_BLOB = "afd9fec5a60702ed6a99274470125e52f2c9c38c"
ORIGINALS = {
    "source": (12826525345, 1024,
       "c66706d3e541925638e901d37232f204c480f0016aba393023fa925c67a61242",
       {"lease-tests.log": (1883,"f6fadaaac9cf97dcf63c39f2dcbb14b42cb08ad6a285903bacf2fd50234eb70d"),
        "source-tests.log": (1729,"6e93487ad1984eec8bb80c5a5e55534cc303acc878ec0a14acd5577ba3465a35")}),
    "chat": (12826246889, 2955,
       "e040e1ce982a1c0f0c5e99e014cada04cc41b334f1755d60cf37fd0bb909a05b",
       {"host.log": (237,"24bc76df7bbf527ce16ac679c9be346827d02945979c9bfffb009a76333b5ed1"),
        "receipt.json": (5978,"27a0a2890d427f39b8214cf0d16a2a4530f9965d597708133be74df87157e3b5")}),
    "instruct": (12825942600, 3008,
       "9656ff3f8019f497cff62d816276f44bdc33f149c07072357bd5cfd9c90d0f93",
       {"host.log": (241,"2960eb4eb8a173fd4756be6a4d9563408c573ba2fc84b38f9b582228eeac4f66"),
        "receipt.json": (6097,"3dd6c391137dc1e6561360a0587b84c9cf55b4d4a7872d73c089eacdbda31b9a")}),
}
# Git hashes of the original ASCII base64 text, including its final newline.
ENCODED_GIT_BLOBS = {
    "source": "518344afea1956d356bd9a661ad1ded5dde8184b",
    "chat": "df27a222657606813d8d20ba95bcc293bd8231cb",
    "instruct": "cd5781917412f755e79d7938f96c4a104643055d",
}


def refuse(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("NCR001_SPROCKET_ORIGINAL_ZIP_REFUSED: " + message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()


def b64_path(lane: str) -> str:
    return f"original-actions-{lane}.zip.b64"


def check_zip(lane: str, raw: bytes) -> dict[str, bytes]:
    _, size, sha, members = ORIGINALS[lane]
    refuse(len(raw) == size and sha256(raw) == sha,
           "whole original Actions ZIP bytes mismatch: " + lane)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        refuse(len(names) == len(members) == len(set(names))
               and set(names) == set(members),
               "ZIP member set altered: " + lane)
        found = {}
        for name in names:
            info = z.getinfo(name)
            refuse(not info.is_dir() and name == Path(name).name,
                   "unsafe original ZIP name")
            contents = z.read(name)
            target_size, target_sha = members[name]
            refuse(info.file_size == len(contents) == target_size
                   and sha256(contents) == target_sha,
                   "original ZIP member corruption: " + lane + "/" + name)
            found[name] = contents
        return found


def audit(evidence: Path = EVIDENCE) -> dict:
    raw_manifest = (evidence / "manifest.json").read_bytes()
    refuse(gitblob(raw_manifest) == MANIFEST_GIT_BLOB,
           "committed original source/member manifest blob altered")
    manifest = json.loads(raw_manifest)
    refuse(manifest["original_source_head"] == ORIGINAL_HEAD
           and manifest["original_run_id"] == ORIGINAL_RUN
           and manifest["train_authorized"] is False
           and manifest["full_72_conformance_pass"] is False
           and manifest["independent_scientific_review"] is False
           and manifest["comparative_training_performed"] is False,
           "original provenance or scientific training gate changed")
    out = {}
    for lane in ORIGINALS:
        original_id, size, checksum, members = ORIGINALS[lane]
        old = manifest["original_actions_artifacts"][lane]
        refuse(old["id"] == original_id and old["zip_sha256"] == checksum,
               "original artifact provenance changed: " + lane)
        refuse(set(old["members"]) == set(members)
               and all(old["members"][n] == {"bytes": b, "sha256": h}
                       for n, (b, h) in members.items()),
               "original raw member provenance changed: " + lane)
        encoded = (evidence / b64_path(lane)).read_bytes()
        refuse(gitblob(encoded) == ENCODED_GIT_BLOBS[lane],
               "committed ASCII ZIP blob altered: " + lane)
        try:
            raw = base64.b64decode(encoded.strip(), validate=True)
        except (ValueError, binascii.Error) as ex:
            raise ValueError("NCR001_SPROCKET_ORIGINAL_ZIP_REFUSED: invalid ASCII ZIP") from ex
        check_zip(lane, raw)
        out[lane] = {"original_artifact": original_id, "zip_sha256": checksum,
                     "original_zip_bytes": size, "raw_members": len(members),
                     "encoded_git_blob": ENCODED_GIT_BLOBS[lane]}
    return {"status": "PASS_THREE_EXACT_ORIGINAL_ACTIONS_ZIPS__FINAL_72_BLOCKED",
            "original_source_head": ORIGINAL_HEAD, "original_run": ORIGINAL_RUN,
            "lane_archives": out, "raw_members": 6,
            "new_model_execution": False, "optimizer_steps": 0,
            "train_authorized": False, "full_72_conformance_pass": False,
            "independent_scientific_review": False,
            "comparative_training_performed": False}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
