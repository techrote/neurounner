#!/usr/bin/env python3
"""Byte-exact offline custody and fail-closed audit of PR #161 original evidence.

Run under stdlib only: NO model loads, network, optimizer or scientific
admission. The original PR head and original Actions ZIPs are immutable.
Never accept a digest reconstructed from a modified or repaired receipt.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REL = "experiments/ncr-001/controls-evidence/postlease-wire-20261028"
BUNDLE = ROOT / REL / "original-three-actions-zip-containers.tar.xz.b64"
HEAD = "aadbfe823c9778066acac0a0869e0c559a533f05"
RUN = "40718706396"
FIRST_FAILED_RUN = "40718490313"
EXPECTED_XZ_SHA256 = "d85eea819ff67e2d2441370d7b1086079eed8446c9136f3e776de48ba328601c"
EXPECTED_TAR_SHA256 = "be9c9bce04dca42e84ba3afa7c0b965845ccecb6f678230d4de6c9d9723e04ac"
ORIGINAL_ZIPS = {
    "artifact-12829497410.zip": {
        "sha256": "6e3a15f086be29bfbccd2d99f9fab82ef73b24cc94de1d46e7c4e9c426e44435",
        "size": 1083,
        "members": {
            "lease-tests.log": (1883, "db387029ad79cd225977eef8154288099ea854f5336b8f91b40806a9c1b55047"),
            "source-tests.log": (1995, "6cb4e0b9a301ac5415be2791e92f99cb2120be2d8daa0ea55dcc60142835112f"),
        },
    },
    "artifact-12828893309.zip": {
        "sha256": "ae3c4623818e41667cb4df8be37dc9b0f2ecf456184b030dc3476b98a565b5eb",
        "size": 3265,
        "members": {
            "host.log": (285, "cfd42c28c21271819602b84222c26514f6a771b131bc326b5fc344a7891d1cb6"),
            "receipt.json": (9458, "44f59fdf8eaa65ff6c4721b7f9c0c40986297442cf93382dce9201c83f33dc6e"),
        },
    },
    "artifact-12829691760.zip": {
        "sha256": "2c6b78409962a077e4b77d9dcc39c094416e4eee7eb985dc75ccdac2b2c27716",
        "size": 1063,
        "members": {
            "source-tests.log": (12611, "027ba446f3105da49a4517564834ee9af9acec9751ae1944cb0fb77b41ab086d"),
        },
    },
}
INSTRUCT = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
CHAT = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
RENDERER_BLOB = "4e558ae530a1b79050e22bee587b3caf056e4563"
WIRE_HEAD = "3592841802d85c442ca2521479b5d8795bee7233"
MAX_TOL = 0.0005


def demand(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("NCR-001E evidence refusal: " + message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_zip_members(binary: bytes, meta: dict) -> dict[str, bytes]:
    demand(len(binary) == meta["size"] and sha(binary) == meta["sha256"],
           "original Actions ZIP container bytes/hash changed")
    with zipfile.ZipFile(io.BytesIO(binary)) as archive:
        names = archive.namelist()
        demand(len(names) == len(set(names)) and set(names) == set(meta["members"]),
               "missing, duplicate or unexpected original ZIP member")
        result = {}
        for name in names:
            info = archive.getinfo(name)
            length, digest = meta["members"][name]
            demand(not info.is_dir() and ".." not in Path(name).parts
                   and not name.startswith("/"), "unsafe ZIP member path")
            data = archive.read(name)
            demand(len(data) == length and sha(data) == digest,
                   "original raw ZIP member byte/hash mismatch: " + name)
            result[name] = data
        return result


def original_zip_bytes(path: Path = BUNDLE) -> dict[str, bytes]:
    b64 = path.read_bytes().strip()
    demand(len(b64) == 6984, "original bundle base64 length changed")
    xz = base64.b64decode(b64, validate=True)
    demand(sha(xz) == EXPECTED_XZ_SHA256,
           "original three-ZIP compressed archive changed")
    tar = lzma.decompress(xz)
    demand(sha(tar) == EXPECTED_TAR_SHA256,
           "original three-ZIP uncompressed archive changed")
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as archive:
        entries = archive.getmembers()
        demand(len(entries) == len(ORIGINAL_ZIPS) and
               {x.name for x in entries} == set(ORIGINAL_ZIPS)
               and all(x.isfile() for x in entries), "ZIP bundle membership changed")
        return {entry.name: archive.extractfile(entry).read() for entry in entries}


def verify_original_host(receipt: dict) -> dict:
    """Validate the immutable source result; do NOT infer tool competence."""
    for key in ("train_authorized", "full_72_conformance_pass",
                "model_generated_tool_memory_semantics_qualified",
                "publisher_source_checkpoint_build_attested",
                "comparative_training_performed", "heldout_opened"):
        demand(receipt.get(key) is False, "false admission or unsafe claim: " + key)
    demand(receipt.get("optimizer_steps") == 0 and
           type(receipt["optimizer_steps"]) is int,
           "unexpected optimizer steps")
    demand(receipt.get("status") == "PASS_SOURCE_STRUCTURED_POSTLEASE_WIRE_CACHE_ONLY",
           "original frozen-host status changed")
    demand(receipt.get("experiment") == "NCR-001" and
           receipt.get("lane") == "sprocket_instruct" and
           receipt.get("original_frozen_model_executed") is True and
           receipt.get("source_structural_wire_qualified") is True,
           "wrong lane or source/physical witness identity")
    runner = receipt.get("runner", {})
    demand(runner.get("head_sha") == HEAD and runner.get("run_id") == RUN and
           runner.get("torch") == "2.14.1+cpu" and
           runner.get("transformers") == "4.57.6" and
           runner.get("runner_os") == "Linux" and
           type(runner.get("pid")) is int and runner["pid"] > 0 and
           bool(runner.get("runner_name")) and bool(runner.get("host")),
           "original runner identity/package/source changed")
    source = receipt.get("source_identity", {})
    demand(source.get("model_id") == "kandivault/sprocket-500m"
           and source.get("revision") == "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
           and source.get("weight_path") == "model.safetensors"
           and source.get("weight_sha256") == INSTRUCT,
           "Chat substituted for exact original frozen Instruct")
    files = receipt.get("verified_source_files", {})
    demand(files.get("model.safetensors", {}).get("sha256") == INSTRUCT
           and files["model.safetensors"].get("bytes") == 1002207984,
           "actual complete original model weight digest/length changed")
    demand(receipt.get("in_memory_frozen_before_sha256")
           == receipt.get("in_memory_frozen_after_sha256")
           and isinstance(receipt["in_memory_frozen_before_sha256"], str)
           and len(receipt["in_memory_frozen_before_sha256"]) == 64,
           "actual original frozen parameters mutated")
    authority = receipt.get("wire_authority", {})
    demand(authority.get("reviewed_tree") == WIRE_HEAD
           and authority.get("reviewed_renderer_blob") == RENDERER_BLOB
           and authority.get("format_source_tree")
           == "2271218d625e1462b2cf218445da975efe17fe32",
           "publisher wire provenance changed")
    wire = receipt.get("wire", {})
    demand(wire.get("sequence_units") == 123 and wire.get("assistant_loss_units") == 45
           and wire.get("padded_shape") == [2, 123]
           and wire.get("synthetic_pad_positions") == 97
           and wire.get("source_end_id5_count") == 12
           and wire.get("source_end_id5_assistant_label_count") == 4
           and wire.get("memory_read_host_model_host") == [False, True, False]
           and wire.get("tool_call_labelled") is True
           and wire.get("assistant_memory_write_labelled") is True
           and wire.get("host_tool_result_unlabelled") is True
           and wire.get("real_end_marker_id_5_visible_with_attention") is True
           and wire.get("naive_attention_mask_detected_invalid") is True
           and wire.get("source_is_diagnostic_not_heldout") is True,
           "publisher host-vs-model SFT loss/attention mask semantics changed")
    check = receipt.get("checks", {})
    required = []
    for adapter in ("nca", "mlp"):
        for name in ("zero_vs_original", "disabled_vs_original",
                     "zero_disabled_parity", "explicit_left_pad",
                     "left_pad_short", "left_pad_long"):
            key = adapter + "_" + name
            demand(check.get(key, {}).get("pass") is True,
                   "frozen model parity or source padding gate failed: " + key)
            required.append(key)
        for boundary in ("assistant_tool_call", "host_tool_result",
                         "assistant_memory_write", "host_memory_read"):
            key = adapter + "_cache_" + boundary
            item = check.get(key, {})
            demand(item.get("pass") is True and item.get("atol") == MAX_TOL
                   and item.get("rtol") == MAX_TOL
                   and type(item.get("max_abs")) in (int, float)
                   and 0 <= item["max_abs"] <= MAX_TOL
                   and item.get("steps") == 4,
                   "source-structured true incremental KV cache gate failed: " + key)
            required.append(key)
        impact = check.get(adapter + "_untrained_manual_effect", {})
        demand(impact.get("pass") is True and
               impact.get("trained") is False and
               0 < impact.get("max_abs", 0) < 1,
               "manual untrained intervention falsely qualified")
        generated = check.get(adapter + "_raw_untrained_generation", {})
        demand(generated.get("semantic_score", "unknown") is None
               and generated.get("source_only_diagnostic") is True
               and isinstance(generated.get("disabled_token_ids"), list)
               and isinstance(generated.get("active_token_ids"), list)
               and len(generated["disabled_token_ids"]) == 6
               and len(generated["active_token_ids"]) == 6,
               "raw generation promoted to semantic competence")
        # The actual observed incomplete tool-call prefix is a negative result.
        demand(generated.get("active_text") == '<|tool_call|>{"name":"'
               and generated.get("disabled_text") == '<|tool_call|>{"name":"',
               "original incomplete tool output was rewritten")
    demand(set(check) == set(required) |
           {a + "_" + suffix for a in ("nca", "mlp") for suffix in (
               "raw_untrained_generation", "untrained_manual_effect")},
           "unexpected/missing original model check")
    return {
        "source_head": HEAD, "original_action_run": int(RUN),
        "model": "sprocket_instruct", "original_host_status": receipt["status"],
        "physical_checks": len(check), "causal_bound": MAX_TOL,
        "tool_memory_semantics_admitted": False, "full_72_admitted": False,
        "training_authorized": False,
    }


def audit() -> dict:
    zips = original_zip_bytes()
    members = {name: raw_zip_members(zips[name], meta)
               for name, meta in ORIGINAL_ZIPS.items()}
    source = members["artifact-12829497410.zip"]
    failed = members["artifact-12829691760.zip"]["source-tests.log"]
    demand(b"Ran 11 tests" in source["source-tests.log"]
           and b"OK" in source["source-tests.log"]
           and b"Ran 10 tests" in source["lease-tests.log"]
           and b"OK" in source["lease-tests.log"],
           "successful original source/lease results missing")
    demand(b"Ran 11 tests" in failed and b"FAILED (errors=11)" in failed
           and b"unexpected keyword" not in failed,
           "failed test-binding run lost or misreported")
    host = members["artifact-12828893309.zip"]
    demand(b"PASS_SOURCE_STRUCTURED_POSTLEASE_WIRE_CACHE_ONLY"
           in host["host.log"] and b'"optimizer_steps": 0' in host["host.log"],
           "original source-matched physical log not preserved")
    receipt = json.loads(host["receipt.json"])
    result = verify_original_host(receipt)
    result["original_zip_sha256"] = {
        name: meta["sha256"] for name, meta in ORIGINAL_ZIPS.items()}
    result["original_failed_source_run"] = int(FIRST_FAILED_RUN)
    result["retained_original_failures"] = True
    result["source_and_lease_test_counts"] = [11, 10]
    result["raw_artifact_members"] = sum(len(meta["members"]) for meta in
                                          ORIGINAL_ZIPS.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    opts = parser.parse_args()
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2 if opts.json else None))


if __name__ == "__main__":
    main()
