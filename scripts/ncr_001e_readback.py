#!/usr/bin/env python3
"""NCR-001E: independent *readback* of four original model-hook prepass receipts.

This is not an independent fresh model execution or #72 qualification.
It neither downloads checkpoints nor runs a model, tokenizer or optimizer.
All four original receipts were copied byte-for-byte from disjoint draft PRs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"

SOURCES = {
    "smollm2": {
        "status": "PASS_PRELIMINARY_SOURCE_BOUND_SMOLLM2_HOOK_CONFORMANCE",
        "run_id": 40422167066, "pr": 83,
        "source_head": "ec0d4a80e15f203a06c26ee9c23858f78976370b",
        "expected_frozen_parameters": 134515008,
    },
    "sprocket_chat": {
        "status": "PASS_PRELIMINARY_SOURCE_BOUND_SPROCKET_CHAT_HOOK_CONFORMANCE",
        "run_id": 40426220940, "pr": 86,
        "source_head": "83a56835a7476f17a17980d28dd1a7c14bb6f2f6",
        "expected_frozen_parameters": 501090560,
    },
    "sprocket_instruct": {
        "status": "PASS_PRELIMINARY_SOURCE_BOUND_SPROCKET_INSTRUCT_HOOK_CONFORMANCE",
        "run_id": 40423957135, "pr": 84,
        "source_head": "b95c010f9b2ba3cbd9f81317981316bb9c8acd19",
        "expected_frozen_parameters": 501090560,
    },
    "void1": {
        "status": "PASS_PRELIMINARY_VOID1_QWEN3_HOOK_CONFORMANCE",
        "run_id": 40424675239, "pr": 85,
        "source_head": "5fa3dc8c0de153098d1800f7b0e9b819620aef4",
        "expected_frozen_parameters": 90148352,
    },
}

REMAINING = {
    "smollm2": [
        "Independent fresh model execution with longer sequences/masks, padding and cache modes",
        "Source-bound saved adapter cold-reload and multi-request/reset tests",
        "Second-party reviewer qualification of exact model source/runtime/checkpoint",
    ],
    "sprocket_chat": [
        "Independently verified chat role/template semantics and long prefill/cache replay",
        "Cold reload, request isolation and separate reviewer runtime readback",
        "No cross-variant substitution of root Sprocket Instruct weights",
    ],
    "sprocket_instruct": [
        "Explicit tool-call/tool-result/memory token formatting acceptance",
        "Long cached context, source-bound cold reload and request isolation",
        "Separate reviewer runtime qualification and checkpoint-mismatch refusal",
    ],
    "void1": [
        "Independent audit of custom ByteTokenizer Python before any execution",
        "UTF-8/Unicode/byte-unit/BOS/EOS roundtrip and text-based cached generation",
        "Long context, cold reload and independent runtime review",
    ],
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(check: bool, why: str) -> None:
    if not check:
        raise RuntimeError(why)


def audit_lane(name: str, raw_bytes: bytes, provenance: dict,
               protocol: dict, pins: dict) -> dict:
    require(name in SOURCES and name in protocol["models"] and name in pins["models"],
            f"Unexpected source lane {name}")
    rec = json.loads(raw_bytes)
    src = SOURCES[name]
    run = provenance["source_run"]
    model = protocol["models"][name]
    pin = pins["models"][name]
    expected_digest = run["original_report_sha256"]
    require(sha(raw_bytes) == expected_digest,
            f"{name}: original receipt SHA-256 no longer matches evidence custody")
    expected_size = run.get("original_report_bytes", run.get("report_bytes"))
    require(len(raw_bytes) == expected_size, f"{name}: original receipt byte size mismatch")
    require(run["run_id"] == src["run_id"],
            f"{name}: unexpected Actions run (possible wrong variant)")
    require(run.get("original_run_result", run.get("test_status")) == "success",
            f"{name}: original workflow was not successful")
    require(str(run.get("artifact_zip_sha256", "")).strip() != "",
            f"{name}: original artifact ZIP hash missing")
    require(rec["status"] == src["status"],
            f"{name}: source-specific PASS label mismatch")
    require(rec["experiment"] == provenance["experiment"],
            f"{name}: experiment label mismatch")
    require(rec.get("model_training_performed") is False
            and rec.get("optimizer_steps") == 0
            and rec.get("comparative_evaluation_performed") is False,
            f"{name}: source prepass improperly includes training/comparison")
    require(protocol["train_authorized"] is False,
            "Training admission flag must remain false")
    require(rec["revision"] == model["revision"] == pin["revision"],
            f"{name}: pinned checkpoint revision mismatch")
    require(pin["repo_id"] == model["repo_id"],
            f"{name}: source-model identity mismatch")
    weight_path = model["weight_file"]
    require(weight_path == pin["hf_weight_file"],
            f"{name}: model variant weight path changed")
    actual_src = rec["source_files"].get(weight_path)
    require(actual_src is not None,
            f"{name}: actual downloaded checkpoint file missing from receipt")
    downloaded_sha = actual_src["sha256"]
    expected_sha = pin["weight_declared_lfs_sha256_not_downloaded"]
    require(downloaded_sha == expected_sha
            and actual_src["expected_sha256"] == expected_sha,
            f"{name}: full checkpoint byte digest differs from pinned source")
    require(actual_src["bytes"] == pin["weight_bytes"],
            f"{name}: downloaded model weight size drift")
    require(rec["source_pins_sha256"] == run.get(
        "source_pin_snapshot_sha256",
        run.get("source_pins_sha256_at_run", run.get("source_pins_sha256"))),
        f"{name}: source-pins baseline snapshot differs from original provenance")

    if name == "smollm2":
        require(rec["model"] == model["repo_id"], f"{name}: model name mismatch")
        frozen = rec["base_parameters"]
        params = rec["nca_parameters"]
        listed = provenance["original_model"]
        require(listed["model_safetensors_actual_downloaded_sha256"] == downloaded_sha,
                f"{name}: promoted provenance weight SHA mismatch")
        details = provenance["readback"]
        require(details["optimizer_steps"] == 0 and details["trained_weights"] is False,
                f"{name}: provenance claims model training")
    elif name == "void1":
        require(rec["source_id"] == model["repo_id"], f"{name}: Qwen3 model name mismatch")
        require(rec["custom_tokenizer_code_executed"] is False
                and rec["tokenizer_semantic_qualification"] is False,
                "void1: remote tokenizer execution/semantic approval was not allowed")
        require(rec["byte_tokenizer_source_matches_prior_train002b"] is True,
                "void1: pinned custom tokenizer no longer matches TRAIN-002b")
        require(rec["source_files"]["tokenization_byte.py"]["sha256"]
                == pin["tokenization_byte_py_sha256"]
                == model["tokenizer_reference"]["train002b_source_sha256"],
                "void1: custom ByteTokenizer source SHA diverged")
        require(rec["source_files"]["byte_vocab.json"]["sha256"]
                == pin["byte_vocab_sha256"],
                "void1: byte vocabulary diverged")
        frozen = rec["base_frozen_parameters"]
        params = rec["nca_trainable_parameters"]
        listed = provenance["model"]
        require(listed["actual_model_safetensors_sha256"] == downloaded_sha,
                "void1: archived model digest mismatch")
    else:
        require(rec["model"] == model["repo_id"] and rec["model_variant"] == model["variant"],
                f"{name}: Sprocket Chat/Instruct variant relabelled")
        require(rec["weight_path"] == weight_path,
                f"{name}: Sprocket weight path mismatch")
        require(rec["source_files"][weight_path]["sha256"] == downloaded_sha,
                f"{name}: checkpoint path source mismatch")
        require(pins["models"]["sprocket_chat"]["weight_declared_lfs_sha256_not_downloaded"] !=
                pins["models"]["sprocket_instruct"]["weight_declared_lfs_sha256_not_downloaded"],
                "Sprocket chat/instruct weights must have distinct identities")
        frozen = rec["base_parameters"]
        params = rec["nca_parameters"]
        listed = provenance["model"]
        require(listed["actual_full_safetensors_sha256"] == downloaded_sha,
                f"{name}: promoted provenance weight SHA mismatch")

    require(frozen == src["expected_frozen_parameters"],
            f"{name}: frozen base parameter count mismatch")
    h = model["expected_config"]["hidden_size"]
    require(params == 514 * h + 984, f"{name}: NCA parameter formula changed")

    checks = rec["checks"]
    for k, detail in checks.items():
        if k.startswith("zero_init_logits") or k.startswith("zero_init_len") or k=="zero_init_batch2":
            require(detail["max_abs"] == 0 and detail["close"],
                    f"{name}: zero-init original LM parity broken ({k})")
    require(checks["zero_init_greedy_generation"]["equal"] is True,
            f"{name}: zero-init greedy generation changed")
    grad = checks["frozen_base_readout_gradient"]
    require(grad["base_grads_none"] is True and grad["optimizer_steps"] == 0
            and grad["readout_grad_l1"] > 0,
            f"{name}: gradient or frozen-backbone gate failed")
    cache = checks["nonzero_nca_full_vs_incremental_cache"]
    maxabs = cache["max_abs"]
    max_allowed = protocol["conformance"]["full_vs_cached_prefill_atol"]
    require(cache["close"] is True and 0 <= maxabs <= max_allowed,
            f"{name}: full-vs-cached logits outside preregistered bound")
    causal = checks["nonzero_nca_future_suffix_invariance"]
    require(causal["close"] is True and causal["max_abs"] == 0,
            f"{name}: causal future suffix affected previous tokens")
    perturb = checks["nonzero_untrained_readout_changes_logits"]
    require(perturb["max_abs"] > 0,
            f"{name}: manual nonzero readout did not affect logits")
    require(perturb.get("training", perturb.get("trained")) is False,
            f"{name}: manual nontrained perturbation relabelled as training")

    return {
        "model": name,
        "original_run_id": run["run_id"],
        "original_pr": src["pr"],
        "source_variant": model["variant"],
        "original_full_checkpoint_sha256": downloaded_sha,
        "checkpoint_size_bytes": actual_src["bytes"],
        "frozen_parameters": frozen,
        "nca_parameters": params,
        "zero_init_max_abs": 0,
        "short_cached_max_abs": maxabs,
        "short_causal_suffix_max_abs": causal["max_abs"],
        "manual_not_learned_logit_effect": perturb["max_abs"],
        "readout_grad_l1": grad["readout_grad_l1"],
        "original_report_sha256": expected_digest,
        "readback": "PASS_ORIGINAL_RECEIPT_INTEGRITY_AND_LIMITED_TESTS",
        "independent_runtime_conformance_pass": False,
        "training_admitted": False,
        "remaining": REMAINING[name],
    }


def audit(root: Path = EVIDENCE, protocol_path: Path = PROTOCOL,
          pins_path: Path = PINS) -> dict:
    protocol = load(protocol_path)
    pins = load(pins_path)
    require(protocol["train_authorized"] is False, "Independent audit cannot admit training")
    require(set(protocol["models"]) == set(SOURCES),
            "NCR-001E source matrix must contain exact four models")
    outcome = {}
    for name in SOURCES:
        parent = root / name
        outcome[name] = audit_lane(
            name, (parent/"receipt.json").read_bytes(),
            load(parent/"provenance.json"), protocol, pins)
    return {
        "schema_version": 1,
        "experiment": "NCR-001E",
        "stage": "four-source-preliminary-prepass-readback-only",
        "source_protocol_sha256": sha(protocol_path.read_bytes()),
        "source_pins_sha256": sha(pins_path.read_bytes()),
        "model_lanes": outcome,
        "source_integrity_readback_pass": True,
        "independent_runtime_conformance_pass": False,
        "any_model_training_admitted": False,
        "study_quality_or_speed_result": False,
        "aggregate_status": "PRELIMINARY_SOURCE_READBACK_PASS__FOUR_MODEL_CONFORMANCE_STILL_BLOCKED",
        "limitations": [
            "Rechecks four original receipts/provenance, does not rerun physical model hosts",
            "Not an independent reviewer or a sealed learner evaluation",
            "Sprocket Chat/Instruct formatter and void ByteTokenizer semantics unqualified",
            "No cold checkpoint reload and no long-context multi-request cache qualification",
            "No learned NCA adapter or matched-control language quality/speed evidence",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT/"runtime/ncr-001/four-model-prepass-readback.json")
    args = parser.parse_args()
    result = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "aggregate_status": result["aggregate_status"],
        "source_integrity_readback_pass": True,
        "independent_runtime_conformance_pass": False,
        "models": {name: {
            "max_cache_abs": row["short_cached_max_abs"],
            "weight_sha256": row["original_full_checkpoint_sha256"],
            "training_admitted": False}
            for name, row in result["model_lanes"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
