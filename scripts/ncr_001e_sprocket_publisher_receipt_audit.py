#!/usr/bin/env python3
"""Independently re-read original #102 frozen Instruct Actions receipt bytes.

Standard library only. Does NOT execute frozen model, check model quality, train,
inspect held-out fixtures or confer the independent #72 admission.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review/sprocket-publisher-wire"
ARCHIVE = EVIDENCE / "original-instruct.json.gz.b64"
PROVENANCE = EVIDENCE / "provenance.json"

# Original independently downloaded Actions ZIP and *unmodified* JSON hashes.
ORIGINAL_JSON_SHA = "58f88ec403f8471e3ccbd76a9e665ef7f3dc7080969c3330117134ac381a29a2"
ORIGINAL_GZIP_SHA = "a744a82ae5dcd93d5dba67cd9aef0df9c4726ad0fcfab4ca6af5882efa0d08ac"
ORIGINAL_ZIP_SHA = "83c3a83c88c20da99bb4b9131c2a67528689f6a6adc7cd27bb2e681f3f6fc82a"
WEIGHT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
PUBLISHER_COMMIT = "a003767e70d766bc14b40c89792543ee8641c8bc"
PUBLISHER_SFT_BLOB = "90bddf8976b9d6410fe7c5c1f084708e1cebab73"
ORIGINAL_HOOK = "b95c010f9b2ba3cbd9f81317981316bb9c8acd19"
RUN_ID = 40567832272


def require(ok, what: str) -> None:
    if not ok:
        raise RuntimeError("NCR-001E publisher-wire archive refusal: " + what)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_archive(archive_path=ARCHIVE, provenance_path=PROVENANCE):
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    artifact = provenance["original_actions_artifact"]
    encoded = Path(archive_path).read_text(encoding="ascii").strip()
    require(len(encoded) <= 15000 and encoded.isascii(), "archive envelope size/encoding")
    compressed = base64.b64decode(encoded, validate=True)
    require(len(compressed) == artifact["archive_gzip_bytes"] == 2093,
            "archive gzip byte size")
    require(sha(compressed) == artifact["archive_gzip_sha256"] ==
            ORIGINAL_GZIP_SHA, "archive gzip SHA256 mismatch")
    original = gzip.decompress(compressed)
    require(len(original) == artifact["original_json_bytes"] == 6816,
            "original receipt length")
    require(sha(original) == artifact["original_json_sha256"] ==
            ORIGINAL_JSON_SHA, "original receipt SHA256 mismatch")
    return json.loads(original), provenance


def verify_receipt(j, p) -> None:
    """Fail closed on semantic overclaim even if somebody recomputes JSON hash."""
    require(p["draft_pr"] == 102 and p["parent_issue"] == 67 and
            p["issue"] == 72, "original research identity")
    origin = p["original_run"]
    require(origin["id"] == RUN_ID and origin["actual_model_result"] == "success"
            and origin["source_only_result"] == "success"
            and origin["actual_model_job_id"] == 122413848927,
            "actual original Actions evidence")
    art = p["original_actions_artifact"]
    require(art["zip_sha256"] == ORIGINAL_ZIP_SHA and art["zip_bytes"] == 2273
            and art["artifact_id"] == 12762719757 and
            art["original_json_sha256"] == ORIGINAL_JSON_SHA and
            art["archive_gzip_sha256"] == ORIGINAL_GZIP_SHA and
            art["original_json_bytes"] == 6816 and
            art["archive_gzip_bytes"] == 2093 and
            art["zip_archived_in_git"] is False and
            art["lossless_original_json"] is True, "original artifact custody")
    pub = p["publisher"]
    require(pub["commit"] == j["publisher_source_commit"] == PUBLISHER_COMMIT
            and pub["sft_data_git_blob"] == j["publisher_sft_git_blob"] ==
            PUBLISHER_SFT_BLOB and pub["is_crypto_training_build_attestation"] is False,
            "publisher source authority is limited to pinned public wire")
    require(j["publisher_harvest_git_blob"] ==
            p["publisher"]["harvest_git_blob"] ==
            "97811aa4245ad3f05f66ce0c2d8aa8c353619267",
            "publisher harvest placement pin")
    model = p["model"]
    require(j["variant"] == "instruct" and
            model["variant"] == "sprocket_instruct"
            and model["file"] == "model.safetensors"
            and model["tap_indices"] == [7, 16, 25], "distinct root Instruct only")
    require(j["original_hook_commit"] == model["original_hook_commit"] ==
            ORIGINAL_HOOK, "original independent NCA hook source pin")
    require(j["checkpoint_sha256"] == model["weight_sha256"] ==
            j["source_files"]["model.safetensors"]["sha256"] == WEIGHT_SHA,
            "verified original full checkpoint bytes")
    require(j["source_files"]["model.safetensors"]["bytes"] ==
            model["weight_bytes"] == 1002207984, "original checkpoint length")
    require(j["source_files"]["README.md"]["sha256"] ==
            model["source_card_sha256"], "HF pinned README source")
    require(j["status"] == p["conformance"]["status"] ==
            "PASS_SOURCE_PUBLISHER_WIRE_CACHE_ONLY", "source-bound result status")
    r = j["runner"]
    pr = p["runner"]
    require(r["github_run_id"] == str(RUN_ID) and
            r["github_sha"] == origin["github_checkout_sha"] and
            r["runner_name"] == pr["name"] and r["pid"] == pr["pid"]
            and r["cpu_count"] == pr["cpu_count"] == 4, "original runner/PID/source")
    require(r["torch"] == pr["torch"] == "2.14.1+cpu" and
            r["transformers"] == pr["transformers"] == "4.57.6",
            "exact original numerical oracle libraries")
    for key in ("actual_model_training", "heldout_evaluation", "train_authorized",
                "tool_and_memory_semantics_qualified",
                "model_generated_tool_result_memory_semantics_test",
                "final_four_model_conformance_pass"):
        require(j[key] is False, "false qualification/training claim: " + key)
    require(j["optimizer_steps"] == p["limits"]["optimizer_steps"] == 0,
            "optimizer training forbidden")
    require(j["source_structural_tool_result_memory_wire_test"] is True
            and p["limits"]["host_injected_wire_only"] is True
            and p["limits"]["final_independent_72_accepted"] is False
            and p["limits"]["train_authorized"] is False,
            "no promotion from source-wire probe to model behaviour/acceptance")
    wire = j["wire"]
    require(wire["not_heldout"] is True
            and wire["tokenizer_units"] ==
            p["conformance"]["synthetic_dialogue_tokenizer_units"] == 120
            and wire["trained_mask_true_units"] ==
            p["conformance"]["assistant_only_mask_true_units"] == 46,
            "non-heldout synthetic wire and expected masked units")
    boundaries = wire["role_boundaries"]
    require([(b["id"], b["trained"]) for b in boundaries
             if b["id"] == 10] == [(10, False), (10, True), (10, False)],
            "memory_read host vs assistant positional ambiguity")
    for token, required in ((6, True), (7, False), (11, True)):
        require([(b["id"], b["trained"]) for b in boundaries
                 if b["id"] == token] == [(token, required)],
                "special SFT marker host/assistant ownership: " + str(token))
    require(wire["tokens_sha256_int64"] ==
            "018e4b37b91a74b2a0ba95edbfc4fe64039bdc5361a75cc9e89a63a566afefdc",
            "source-serialized synthetic token IDs")
    checks = j["checks"]
    expected = {
        "source_wire_tool_call_prefill_incremental": "tool_call_cached_max_abs",
        "source_wire_tool_result_prefill_incremental": "tool_result_cached_max_abs",
        "source_wire_memory_write_prefill_incremental": "memory_write_cached_max_abs",
    }
    for name, metric in expected.items():
        test = checks[name]
        require(test["pass"] is True and test["exact_required"] is False,
                "original one-token cache mismatch: " + name)
        require(test["atol"] == p["conformance"]["cache_atol"] == 0.0005
                and test["rtol"] == p["conformance"]["cache_rtol"] == 0.0005,
                "cache gate illegally relaxed: " + name)
        require(0 <= test["max_abs"] == p["conformance"][metric] <= 0.0005,
                "source-wire cache max deviation exceeds immutable tolerance: " + name)
    for name, metric in (
        ("exact_frozen_source_vs_zero_NCA", "no_op_exact_max_abs"),
        ("source_wire_strict_future_suffix", "strict_future_suffix_max_abs"),
        ("source_wire_independent_request_reset", "request_reset_max_abs"),
    ):
        test = checks[name]
        require(test["pass"] is True and test["exact_required"] is True
                and test["atol"] == test["rtol"] == test["max_abs"] ==
                p["conformance"][metric] == 0, "strict frozen/causal gate: " + name)
    effect = checks["untrained_manual_effect"]
    require(effect["pass"] is True and effect["trained"] is False
            and effect["max_abs"] == p["conformance"]["manual_untrained_effect"] > 0,
            "active but *untrained* original adapter effect")
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    require(protocol["train_authorized"] is False and
            set(protocol["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "four independent model admissions still required")


def main() -> None:
    receipt, provenance = load_archive()
    verify_receipt(receipt, provenance)
    print(json.dumps({
        "stage": "independent_original_actions_receipt_byte_readback_only",
        "status": "PASS_ARCHIVE_SOURCE_WIRE__FINAL_72_BLOCKED",
        "original_json_sha256": ORIGINAL_JSON_SHA,
        "original_zip_sha256": ORIGINAL_ZIP_SHA,
        "actual_model_run": RUN_ID,
        "original_source_model_reexecuted_here": False,
        "independent_four_model_admission": False,
        "train_authorized": False,
        "optimizer_steps": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
