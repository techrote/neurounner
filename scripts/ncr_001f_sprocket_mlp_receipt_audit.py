#!/usr/bin/env python3
"""Read-only independent byte-exact audit of original NCR-001F Sprocket MLP CI JSON.

No torch, transformers, checkpoint weights, optimizer, evaluation or network.
Audit original raw member bytes BEFORE parsing derived model/metrics claims.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/real-sprocket-mlp"
PROVENANCE = EVIDENCE / "provenance.json"
ARCHIVE = EVIDENCE / "original-jsons.tar.gz.b64"

EXPECTED_SOURCE = {
    "scripts/ncr_001f_real_sprocket.py",
    "scripts/ncr_001f_controls.py",
    "scripts/ncr_001f_frozen_hooks.py",
    "experiments/ncr-001/protocol.json",
    "experiments/ncr-001/source-pins.json",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, context: str) -> None:
    if not condition:
        raise ValueError(context)


def audit_receipt(variant: str, actual: dict, prefetch: dict, manifest: dict) -> dict:
    """Assert independent, source-pinned meaning, not only internally hashed bytes."""
    require(variant in ("chat", "instruct"), "unknown source variant")
    source = manifest["variants"][variant]
    other = manifest["variants"]["instruct" if variant == "chat" else "chat"]
    expected_weight = "chat/model.safetensors" if variant == "chat" else "model.safetensors"
    require(source["weight_file"] == expected_weight, "model variant path relabeled")
    require(source["weight_sha256"] != other["weight_sha256"], "variant weights conflated")
    weight = actual["verified_source_files"][expected_weight]
    require(weight["sha256"] == source["weight_sha256"]
            == weight["pinned_sha256"], "complete pinned model weight mismatch")
    require(weight["bytes"] == manifest["actual_weight_bytes_per_variant"],
            "original safetensors size mismatch")
    require(prefetch["source_files"][expected_weight]["sha256"]
            == source["weight_sha256"], "prefetch and real-host model weight disagreement")
    require(prefetch["source_files"][expected_weight]["expected"]
            == source["weight_sha256"], "prefetch expected weight pin mismatch")

    expected_status = f"PASS_SOURCE_BOUND_SPROCKET_{variant.upper()}_MLP_PREPASS__NO_TRAINING_ADMITTED"
    require(actual["status"] == expected_status, "actual model success falsely declared")
    require(prefetch["status"] == "SOURCE_BYTES_PASS", "source prefetch did not pass")
    require(prefetch["variant"] == actual["variant"] == variant, "variant record mismatch")
    require(actual["experiment"] == prefetch.get("experiment", "NCR-001F")
            == "NCR-001F", "wrong research run")
    for key in ("train_authorized", "training_performed", "heldout_data_opened",
                "four_model_actual_host_mlp_qualified", "tool_and_memory_behavior_qualified",
                "separate_process_cold_reload_proved", "comparative_training_performed",
                "independent_ncr_001e_conformance_pass"):
        require(actual[key] is False, "invalid claimed authorization/semantics: " + key)
    require(prefetch["train_authorized"] is False and
            prefetch["training_performed"] is False, "source-stage false training claim")
    require(actual["optimizer_steps"] == 0, "actual optimizer steps observed")
    require(actual["frozen_base_parameters"] == 501090560, "frozen original parameter count changed")
    require(actual["gated_mlp_parameters"] == 658432, "MLP control no longer matched")
    require(actual["reference_nca_parameters"] == 658904, "NCA matching baseline changed")
    require(actual["source_identity"]["weight_path"] == expected_weight
            and actual["source_identity"]["weight_sha256"] == source["weight_sha256"]
            and actual["source_identity"]["revision"] == manifest["model_revision"]
            and actual["source_identity"]["model_id"] == manifest["model_repository"]
            and actual["source_identity"]["taps"] == [7, 16, 25],
            "original source identity or postblock tap drift")
    require(actual["formatter_source"]["reviewed_pr_head"]
            == manifest["formatter_commit"], "ordinary role formatter source drift")
    require(actual["formatter_source"]["script_sha256"]
            == manifest["formatter_script_sha256"], "original formatter code hash drift")
    require(actual["formatter_source"]["tool_memory_grammar_supported"] is False
            and actual["role_sources"]["tool_and_memory_behavior_qualified"] is False,
            "invented Instruct tool or Chat memory grammar")
    require(actual["role_sources"]["user"] == [3]
            and actual["role_sources"]["turn_end"] == [5]
            and actual["role_sources"]["assistant"] == [4],
            "original source role boundaries differ")
    require(actual["long_source_role_tokens"] >= 128
            and actual["long_source_role_tokens"] <= 2048, "synthetic role probe not bounded")

    runner = actual["runner"]
    require(runner["github_sha"] == prefetch["github_sha"]
            == manifest["original_actual_actions_checkout_merge_sha"], "original CI checkout changed")
    require(str(runner["run_id"]) == str(prefetch["run_id"])
            == str(manifest["original_run_id"]), "original Actions run identity changed")
    require(runner["torch"] == manifest["pinned_float32_stack"]["torch"]
            and runner["transformers"] == manifest["pinned_float32_stack"]["transformers"]
            and runner["python"] == manifest["pinned_float32_stack"]["python"]
            and runner["cpu_threads"] == manifest["pinned_float32_stack"]["cpu_threads"],
            "host/runtime stack not as reported")
    require(runner["cpu_model"] == source["runner_cpu"], "original CPU runner confound hidden")
    require(actual["peak_rss_kib"] == source["peak_rss_kib"]
            and 0 < actual["peak_rss_kib"] <= 12 * 1024 * 1024,
            "original RSS source/resource mismatch")
    require(actual["in_memory_base_before_sha256"] ==
            actual["in_memory_base_after_sha256"] ==
            source["base_memory_sha256"], "original frozen model tensor byte mutation")
    require(actual["checks"]["frozen_original_tensor_bytes_unchanged"] is True,
            "original base mutation explicitly reported")

    for name in ("zero_mlp_16_source_role", "zero_mlp_64_source_role",
                 "zero_mlp_128_source_role", "zero_mlp_left_masked_batch2",
                 "zero_mlp_right_masked_batch2", "new_adapter_same_frozen_process_exact",
                 "request_A_B_A_exact", "causal_future_suffix_exact"):
        d = actual["checks"][name]
        require(d["close"] is True and d["exact_required"] is True
                and d["max_abs"] == 0 and d["mean_abs"] == 0,
                "strict no-op/causal/request/cold-adapter parity drift: " + name)
    require(actual["checks"]["zero_mlp_greedy_generation"]
            == {"exact": True, "generated_units": 4}, "zero-init greedy parity absent")
    cache = actual["checks"]["active_48_full_vs_incremental_KV"]
    require(cache["close"] is True
            and cache["atol"] == cache["rtol"] == 0.0005
            and cache["shape"] == [1, 48, 32000]
            and 0 <= cache["max_abs"] == source["active_cache_max_abs"] <= 0.0005,
            "actual incremental cache exceeded frozen protocol tolerance")
    grad = actual["checks"]["readout_gradient_no_optimizer"]
    require(grad["optimizer_steps"] == 0
            and grad["finite_loss"] is True
            and grad["base_gradients_none"] is True
            and grad["readout_grad_l1"] == source["readout_grad_l1"] > 0,
            "failed real-host readout-only backward or misreported optimizer step")
    effect = actual["checks"]["manual_untrained_readout_effect"]
    require(effect["learned"] is False
            and effect["max_abs"] == source["untrained_effect"] > 0,
            "missing genuinely active untrained readout")
    for name in ("lane", "model_revision", "verified_weight_sha256", "weight_path"):
        require(actual["checks"]["refuse_corrupt_metadata_" + name] is True,
                "source-variant adapter refusal missing: " + name)
    require(actual["source_sha256"] == manifest["source_files_sha256"],
            "complete exact experiment implementation hashes diverged")
    for name in EXPECTED_SOURCE:
        require(sha((ROOT / name).read_bytes()) == actual["source_sha256"][name],
                "committed real-model source code differs from original CI: " + name)
    require(set(prefetch["source_files"]) == set(actual["verified_source_files"]),
            "original model source file inventory incomplete")
    for name, item in prefetch["source_files"].items():
        rhs = actual["verified_source_files"][name]
        require(item["sha256"] == item["expected"] == rhs["sha256"]
                == rhs["pinned_sha256"] and item["bytes"] == rhs["bytes"],
                "prefetch and frozen-model actual original bytes disagree: " + name)
    require(actual["verified_source_files"]["README.md"]["sha256"]
            == manifest["original_model_card_sha256"], "unreviewed ordinary-role model card")

    return {
        "variant": variant,
        "original_run_id": runner["run_id"],
        "actual_model_weight_sha256": weight["sha256"],
        "original_runner_cpu": runner["cpu_model"],
        "gated_mlp_parameters": actual["gated_mlp_parameters"],
        "cache_max_abs": cache["max_abs"],
        "causal_suffix_max_abs": actual["checks"]["causal_future_suffix_exact"]["max_abs"],
        "same_process_cold_reload_max_abs":
            actual["checks"]["new_adapter_same_frozen_process_exact"]["max_abs"],
        "peak_rss_kib": actual["peak_rss_kib"],
        "optimizer_steps": 0, "train_authorized": False,
        "aggregate_ncr_001e_final_acceptance": False,
    }


def unpack_archive(data: bytes, manifest: dict) -> dict[str, bytes]:
    """Verify exact compressed and uncompressed original CI-member byte custody."""
    try:
        b64 = data.strip()
        require(len(b64) % 4 == 0, "base64 original archive truncated")
        gz = base64.b64decode(b64, validate=True)
        require(len(gz) == manifest["original_gzip_bytes"]
                and sha(gz) == manifest["original_gzip_sha256"], "archive compressed SHA mismatch")
        tar_bytes = gzip.decompress(gz)
        found: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            for member in archive.getmembers():
                require(member.isfile() and member.name not in found,
                        "unexpected, nonfile or repeated original CI member")
                require(member.name.split("/")[0] in {"chat", "instruct"}
                        and member.name.split("/")[1] in {"prefetch.json", "actual-mlp.json"},
                        "unsafe or unapproved original archive member")
                stream = archive.extractfile(member)
                require(stream is not None, "missing original archive member")
                found[member.name] = stream.read()
        reference = {k: item for x in manifest["variants"].values()
                     for k, item in x["original_jsons"].items()}
        require(set(found) == set(reference), "archived original JSON set incomplete/extra")
        for name, raw in found.items():
            expected = reference[name]
            require(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"],
                    "original uploaded JSON byte hash changed: " + name)
        return found
    except (OSError, EOFError, tarfile.TarError, ValueError) as err:
        raise ValueError("Original CI receipt archive invalid: " + str(err)) from err


def audit() -> dict:
    manifest = json.loads(PROVENANCE.read_bytes())
    require(manifest["experiment"] == "NCR-001F"
            and manifest["original_run_id"] == 40519177555
            and manifest["original_pr"] == 98, "wrong original evidence event")
    require(set(manifest["variants"]) == {"chat", "instruct"}
            and manifest["qualification"]["train_authorized"] is False
            and manifest["qualification"]["optimizer_steps"] == 0
            and manifest["qualification"]["independent_issue_72_final_acceptance"] is False,
            "manifest falsely admits training or conformance")
    original = unpack_archive(ARCHIVE.read_bytes(), manifest)
    results = {}
    for variant in ("chat", "instruct"):
        actual = json.loads(original[f"{variant}/actual-mlp.json"])
        prefetch = json.loads(original[f"{variant}/prefetch.json"])
        results[variant] = audit_receipt(variant, actual, prefetch, manifest)
    return {
        "status": "PASS_ORIGINAL_DUAL_SPROCKET_FROZEN_MLP_RAW_JSON_CUSTODY__FINAL_CONFORMANCE_BLOCKED",
        "original_run": manifest["original_run_url"],
        "models": results,
        "raw_original_json_bytes": sum(len(value) for value in original.values()),
        "source_integrity_readback_pass": True,
        "independent_new_model_invocation": False,
        "independent_ncr_001e_final_conformance_pass": False,
        "train_authorized": False,
        "optimizer_steps": 0,
        "tool_memory_behavior_qualified": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(audit(), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"FAIL_CUSTODY__TRAINING_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
