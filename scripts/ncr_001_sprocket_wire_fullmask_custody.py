#!/usr/bin/env python3
"""Offline byte-exact custody of original NCR-001E Sprocket full-mask Actions.

No network, HF model, tokenizer import, optimizer, held-out access or review
admission. Independently check original Git-custodied Actions ZIP containers,
their members, complete job logs and source-bound original receipt.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/sprocket-wire-fullmask-20261028"
HEAD = "40efcb5748894c6c7847c18b1ad442a455bd0aa3"
MODEL_HEAD = "aadbfe823c9778066acac0a0869e0c559a533f05"
BUNDLE_BLOB = "87d6c24f77724f68966b18f8f109f2e13ebf49f2"
XZ_SHA = "cc1010df896950d5debe0eec45052657cbdc14451c0f53cce5ee0b79aa48e856"
TAR_SHA = "43f3c8cb77955099ba9436e02e4fe0181b00201eeeb6df0939555cf56fd12917"
SOURCE_JOB_BLOB = "0b93b10de7cae1648e2b2b8f5930fc5f0ec4bc31"
TOKENIZER_JOB_BLOB = "27ce77df703effe7a3b9bde0fa0e72741de02fbd"
RESULT_BLOB = "8cd6f06a3c80af0cbb8e06279ed69aed97d90806"
RESULT_SHA = "f3fd86f2071666411441c45053f38f4fa00731685814267849abb37c14fbca90"
ORIGINAL_FILES = {
    "artifact-12857787221.zip": (
        1307, "6502f7766e385a7825585dd21fed8129b3f1a095afb8569bd4a1dfabdc739279",
        {"source-tests.log": (2234, "dba6ced64176c460ac655400b58621f818264299a28f309af84a06e1dd514d2f"),
         "original-archive-audit.json": (866, "f2e6ff362bd23c0b2a7c9075ebc661919966df9b623ff78aa492cbc0ea96d9ba")}),
    "artifact-12856874121.zip": (
        3774, "b941b681d61b0d1182eb3d71578f368aa1026e0628d05725668fe1e6295a50dd",
        {"host-tokenizer-only.log": (3333, "78c3da6ecfc62260aef76a9a4356bc23e060f257444f095236628d844fb97eb8"),
         "receipt.json": (3528, RESULT_SHA)}),
}


def refuse(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("NCR001_ORIGINAL_FULLMASK_CUSTODY_REFUSED: " + why)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") +
                        b"\x00" + raw).hexdigest()


def check_report(report: dict, manifest: dict) -> dict:
    """Never elevate source-only replay into model-host science acceptance."""
    refuse(report["status"] ==
           "PASS_FULL_POSITION_SOURCE_MASK_REPLAY__72_REVIEW_BLOCKED",
           "original result status changed")
    refuse(report["lane"] == "sprocket_instruct"
           and report["experiment"] == "NCR-001E"
           and report["schema_version"] == 1,
           "model variant or report identity changed")
    for key in ("train_authorized", "full_72_conformance_pass",
                "independent_scientific_review", "comparative_training_performed",
                "heldouts_opened", "model_generated_tool_memory_semantics_qualified",
                "physical_model_executed_this_run"):
        refuse(report.get(key) is False, "false scientific admission: " + key)
    refuse(type(report["optimizer_steps"]) is int
           and report["optimizer_steps"] == 0, "original optimizer steps changed")
    refuse(report["original_physical_run"] == 40718706396
           and report["retained_original_failed_source_run"] == 40718490313
           and report["historical_model_output"] ==
           "INCOMPLETE_SIX_TOKEN_TOOL_CALL_PREFIX",
           "historical model failure erased")
    full = report["full_position_mask"]
    refuse(full == {"all_positions_checked": True, "units": 123,
                     "assistant_units": 45, "host_units": 78},
           "full position vector count/meaning changed")
    digests = {
        "assistant_mask_sha256": "ee59ba7074f1b60569ba3eca00793924561ac28782572a3a097768c342aff3a2",
        "attention_mask_sha256": "fafee010cc27daeb6996b93f5d110b7943b85d8ee704032f80d1ef7be5e7b4cd",
        "ids_sha256": "27974f2d94b6a9400c7808dd3f953981a8da3f2309d221a3755156541455adfc",
        "prompt_sha256": "b6afa2dfe73d29a609b3e949242eb07f875cd9865a7764a26aa6334c367e955a",
    }
    refuse(report["replayed_hashes"] == digests,
           "original source label/prompt/attention hash altered")
    referee = report["runner"]
    refuse(referee["head_sha"] == HEAD and referee["run_id"] == "40782101613"
           and referee["attempt"] == "1" and
           referee["name"] == "GitHub Actions 1000046991"
           and referee["python"] == "3.12.3"
           and referee["physical_model_executed"] is False,
           "original runner/source identity changed")
    original = report["original_archive"]
    refuse(original["source_head"] == MODEL_HEAD
           and original["original_action_run"] == 40718706396
           and original["original_failed_source_run"] == 40718490313
           and original["retained_original_failures"] is True
           and original["source_and_lease_test_counts"] == [11, 10]
           and original["physical_checks"] == 24
           and original["full_72_admitted"] is False
           and original["training_authorized"] is False
           and original["tool_memory_semantics_admitted"] is False,
           "archived historical physical source/gates altered")
    refuse(report["original_host_receipt_sha256"] ==
           "44f59fdf8eaa65ff6c4721b7f9c0c40986297442cf93382dce9201c83f33dc6e",
           "original source-model receipt identity changed")
    refuse(report["source"]["publisher_renderer_git_blob"] ==
           "4e558ae530a1b79050e22bee587b3caf056e4563"
           and report["source"]["protocol_git_blob"] ==
           "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
           and report["source"]["source_pins_git_blob"] ==
           "59fff1aed872b5563aa6898171983973e0cdb02c",
           "pinned publisher/source/protocol changed")
    refuse(manifest["result"]["prompt_sha256"] == digests["prompt_sha256"]
           and manifest["result"]["ids_sha256"] == digests["ids_sha256"]
           and manifest["result"]["assistant_mask_sha256"] ==
           digests["assistant_mask_sha256"]
           and manifest["result"]["attention_mask_sha256"] ==
           digests["attention_mask_sha256"], "manifest source hashes altered")
    return {"original_head": HEAD, "physical_parent_head": MODEL_HEAD,
            "full_positions": 123, "assistant_positions": 45,
            "original_host_run": 40718706396, "source_test_count": 13}


def audit(evidence: Path = EVIDENCE, original_zip_files: dict[str, bytes] | None = None) -> dict:
    m = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    refuse(m["schema"] == "ncr-001-sprocket-fullmask-original-actions-custody-v1"
           and m["original_pr_head"] == HEAD and
           m["upstream_original_model_head"] == MODEL_HEAD
           and m["parent_pr"] == 181 and m["original_workflow_run"] == 40782101613
           and m["original_workflow_conclusion"] == "success"
           and m["upstream_physical_run"] == 40718706396
           and m["upstream_failed_source_run"] == 40718490313,
           "original source/run identity changed")
    for key in ("train_authorized", "full_72_conformance_pass",
                "independent_scientific_review", "comparative_training_performed",
                "heldouts_opened", "model_generated_tool_memory_semantics_qualified",
                "legacy_byte_eos_effect_measured"):
        refuse(m["qualification"][key] is False, "manifest false admission " + key)
    refuse(m["qualification"]["optimizer_steps"] == 0
           and m["qualification"]["historical_tool_call_prefix_incomplete"] is True
           and m["result"]["units"] == 123
           and m["result"]["assistant_units"] == 45
           and m["result"]["host_units"] == 78,
           "manifest source/negative result changed")
    b = (evidence / m["archived_bundle"]["path"]).read_bytes()
    refuse(blob(b) == BUNDLE_BLOB and
           m["archived_bundle"]["git_blob_sha"] == BUNDLE_BLOB
           and len(b.rstrip(b"\n")) == 6648,
           "Git original bundle or metadata changed")
    xz = base64.b64decode(b.strip(), validate=True)
    refuse(len(xz) == 4984 and sha(xz) == XZ_SHA
           and m["archived_bundle"]["compressed_sha256"] == XZ_SHA,
           "Git original compressed archive corrupt")
    raw_tar = lzma.decompress(xz, memlimit=128 * 1024 * 1024)
    refuse(len(raw_tar) == 10240 and sha(raw_tar) == TAR_SHA
           and m["archived_bundle"]["tar_sha256"] == TAR_SHA,
           "original uncompressed TAR digest wrong")
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive:
        records = archive.getmembers()
        refuse(len(records) == len(ORIGINAL_FILES)
               and {item.name for item in records} == set(ORIGINAL_FILES)
               and all(item.isfile() and item.name in ORIGINAL_FILES
                       for item in records), "original ZIP container bundle changed")
        zips = {entry.name: archive.extractfile(entry).read() for entry in records}
    for name, (length, checksum, members) in ORIGINAL_FILES.items():
        raw = zips[name]
        meta = m["artifacts"][name]
        refuse(len(raw) == length and sha(raw) == checksum
               and meta["zip_sha256"] == checksum
               and meta["zip_size"] == length,
               "original ZIP bytes/manifest mismatch: " + name)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            refuse(len(names) == len(set(names)) and set(names) == set(members),
                   "missing, duplicate or unexpected ZIP member: " + name)
            for member, (size, digest) in members.items():
                info = archive.getinfo(member)
                data = archive.read(member)
                refuse(not info.is_dir() and ".." not in Path(member).parts
                       and len(data) == size and sha(data) == digest
                       and meta["members"][member] == {"size": size, "sha256": digest},
                       "original ZIP member bytes/manifest changed: " + member)
    if original_zip_files is not None:
        refuse(set(original_zip_files) == set(zips), "live original ZIP IDs do not match")
        for name, original in original_zip_files.items():
            refuse(original == zips[name], "live GitHub Actions original differs: " + name)
    source_log = (evidence / m["logs"]["source"]["path"]).read_bytes()
    tokenizer_log = (evidence / m["logs"]["tokenizer"]["path"]).read_bytes()
    raw_result = (evidence / m["result"]["path"]).read_bytes()
    refuse(blob(source_log) == SOURCE_JOB_BLOB
           and m["logs"]["source"]["git_blob_sha"] == SOURCE_JOB_BLOB
           and b"Ran 13 tests" in source_log and b"OK" in source_log,
           "original full source/negative job log corrupted")
    refuse(blob(tokenizer_log) == TOKENIZER_JOB_BLOB
           and m["logs"]["tokenizer"]["git_blob_sha"] == TOKENIZER_JOB_BLOB
           and b"PASS_FULL_POSITION_SOURCE_MASK_REPLAY__72_REVIEW_BLOCKED"
           in tokenizer_log and b"GitHub Actions 1000046991" in tokenizer_log,
           "original full real-tokenizer job log corrupted")
    refuse(blob(raw_result) == RESULT_BLOB and sha(raw_result) == RESULT_SHA
           and m["result"]["git_blob_sha"] == RESULT_BLOB
           and m["result"]["original_member_sha256"] == RESULT_SHA,
           "source-derived original fullmask JSON bytes corrupted")
    with zipfile.ZipFile(io.BytesIO(zips["artifact-12856874121.zip"])) as archive:
        refuse(archive.read("receipt.json") == raw_result,
               "original physical-receipt crosscheck JSON differs from Git")
        original = json.loads(archive.read("receipt.json"))
    summary = check_report(original, m)
    return {
        "status": "PASS_ORIGINAL_ACTIONS_CONTAINER_AND_FULLMASK_SOURCE_CUSTODY__72_BLOCKED",
        "source_head": HEAD, "original_actions_run": 40782101613,
        "original_artifact_zip_sha256": {name: x[1] for name, x in ORIGINAL_FILES.items()},
        "original_source_job_id": 123146082042,
        "original_tokenizer_job_id": 123146150961,
        "archive_members": 4, **summary,
        "train_authorized": False, "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "new_model_execution": False, "comparative_training": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
