#!/usr/bin/env python3
"""Fail-closed, model-free promotion gate for the historical fresh48/fresh120 CPU dose.

A partial training checkpoint is retained as evidence, never a completed
candidate. This gate validates original optimizer, source and checkpoint
receipts before inference. It does not inspect FINAL or judge painting quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

TRAINING = Path(__file__).resolve().parents[2] / "training"
sys.path.insert(0, str(TRAINING))
from common import read_json, sha256_file, verify_checkpoint

SHA = re.compile(r"[a-f0-9]{40}\Z")
CORPUS_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
MODEL_REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"
ARMS = {"fresh48": 48, "fresh120": 120}
# Immutable TRAIN-only identities from original native-admitted source633 corpus
# and original actual 48/120 hosted optimizer log. Seed variants are NOT
# independent structural groups. Changing this curriculum requires a new gate.
TRAIN_SHA = "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1"
COMPILER_SHA = "4d8559805df1ff5f3f23277a7bca98e47fa8feb53337aa79e8546d0bb89132f6"
ORIGINAL_SAMPLE_ORDER = (
    "architecture-stair_and_landing_hall-v1-s130363",
    "greebles-switchboard_stack-v1-s104729",
    "pipework-tee_distribution-v1-s130363",
    "architecture-paired_doorway_rooms-v1-s104729",
    "greebles-switchboard_stack-v1-s130363",
    "pipework-tee_distribution-v1-s104729",
    "architecture-stair_and_landing_hall-v1-s104729",
    "greebles-service_cluster-v1-s104729",
    "pipework-two_elbow_route-v1-s130363",
    "architecture-paired_doorway_rooms-v1-s130363",
    "greebles-service_cluster-v1-s130363",
    "pipework-two_elbow_route-v1-s104729",
)
# Exact approved upstream snapshot root-file SHA-256 identities. The normal
# downloader independently hashes bytes; this post-run gate does not download.
MODEL_HASHES = {
    "LICENSE": "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c",
    "README.md": "666cc6b49fcdec9ddd378081c47df5aa11679c12a4a05cbbb436a3107f04ee3b",
    "chat_template.jinja": "309e586e2eda3d7f2db1e2a045bfb07f4c83798b23f7ac587954426302d508e9",
    "config.json": "ab0de32d57b83b8b0cbb4526e096cf1e8adc1d8b6a09cb55df38597866eae53f",
    "generation_config.json": "40d17f9ec64c97e8fd5400540960b6a9761ed6d6acc0b1ab6a6656055e4755b3",
    "processor_config.json": "622b75b531b3f49b1cdf4f90626c34e5ffb4f8bba2b8637807af0462398ae718",
    "tokenizer.json": "f3910942aa907c48b0cc20ec426ee38bfa8dcda8feecf035ced981918cb30f14",
    "tokenizer_config.json": "aed83606e95db808fc4d5312bf117605360e770bfef5a6028c348c3981ce143a",
    "model.safetensors": "2f6deb5dd43707de5cfe3c59470d3bccf4c3112a810a74570499f4728d412eea",
}


# Qualified anchored expression: no vision/projector/embedding LoRA targets.
TARGET_EXPRESSION = (
    r"^model\.language_model\.layers\.(?:(?:0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)"
    r"\.feed_forward\.(?:w1|w2|w3)|(?:2|5|8|10|12|14)"
    r"\.self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|"
    r"(?:0|1|3|4|6|7|9|11|13|15)\.conv\.(?:in_proj|out_proj))$"
)
ATTN_LAYERS = frozenset({2, 5, 8, 10, 12, 14})
TARGET_PATHS = frozenset(
    f"model.language_model.layers.{layer}.{module}"
    for layer in range(16)
    for module in (
        ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3")
        + (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.out_proj") if layer in ATTN_LAYERS
           else ("conv.in_proj", "conv.out_proj"))
    )
)
assert len(TARGET_PATHS) == 92


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact_int(value, expected):
    return type(value) is int and value == expected


def validate(summary, preflight, receipt, progress, *, arm, source_sha, run_id=None):
    """Pure receipt checks, separated for adversarial model-free tests."""
    require(arm in ARMS, "unknown fresh-dose arm")
    require(isinstance(source_sha, str) and SHA.fullmatch(source_sha), "bad source SHA")
    steps = ARMS[arm]
    require(summary.get("status") == "completed" and summary.get("actual_training") is True,
            "partial/non-training summary cannot be promoted")
    for key in ("global_step", "optimizer_steps_this_job", "sample_cursor"):
        require(exact_int(summary.get(key), steps), "incomplete summary " + key)
    require(summary.get("resume") is None, "fresh arms cannot use resumed checkpoints")
    require(summary.get("adapter_changed") is True
            and summary.get("frozen_base_unchanged") is True
            and summary.get("checkpoint_readback_exact") is True,
            "adapter/base/checkpoint integrity not demonstrated")
    require(summary.get("native_execution") is False
            and summary.get("model_quality_claim") is False,
            "training cannot claim native painting or model quality")
    require(summary.get("checkpoint") == f"checkpoint-{steps:06d}", "wrong final checkpoint")
    require(isinstance(summary.get("adapter_before_sha256"), str)
            and summary.get("adapter_before_sha256") != summary.get("adapter_after_sha256"),
            "adapter unchanged")
    config = summary.get("config")
    require(isinstance(config, dict), "missing training config")
    require(config.get("profile") == "lfm25_vl" and config.get("source_sha") == source_sha,
            "source/profile mismatch")
    require(config.get("native_source_sha") == CORPUS_SOURCE
            and config.get("corpus_manifest_sha256") == CORPUS_SHA,
            "native corpus identity mismatch")
    require(config.get("train_sha256") == TRAIN_SHA
            and config.get("compiler_sha256") == COMPILER_SHA,
            "original native-admitted TRAIN/compiler bytes mismatch")
    model = config.get("model")
    require(isinstance(model, dict) and model.get("model_id") == MODEL_ID
            and model.get("revision") == MODEL_REVISION, "official model revision mismatch")
    paths = model.get("lora_target_paths")
    require(model.get("lora_targets") == TARGET_EXPRESSION
            and type(paths) is list and len(paths) == 92
            and all(type(path) is str for path in paths)
            and set(paths) == TARGET_PATHS,
            "unqualified language-only projection inventory")
    require(model.get("framing", {}).get("eos_token_id") == 7
            and model.get("framing", {}).get("images") is False
            and model.get("framing", {}).get("tools") is False,
            "unqualified text-only EOS/framing boundary")
    require(model.get("required_hashes") == MODEL_HASHES
            and model.get("loader") == "Lfm2VlForConditionalGeneration"
            and model.get("trust_remote_code") is False
            and model.get("license") == "lfm1.0"
            and model.get("card_license") == "other"
            and model.get("expected_inventory", {}).get("language_linear_targets") == 92
            and model.get("expected_inventory", {}).get("lora_parameters") == 2998272,
            "original complete multimodal model/LoRA/license profile mismatch")
    model_files = config.get("model_files")
    require(isinstance(model_files, dict) and set(model_files) == set(MODEL_HASHES)
            and all(isinstance(model_files[name], dict)
                    and model_files[name].get("sha256") == expected
                    and type(model_files[name].get("bytes")) is int
                    and model_files[name]["bytes"] > 0
                    for name, expected in MODEL_HASHES.items()),
            "official pinned downloaded-model file receipt mismatch")
    numerics = config.get("numerics")
    require(isinstance(numerics, dict)
            and numerics.get("profile") == "portable-cpu-fp32-v1",
            "unqualified numerical recipe")
    require(config.get("device") == "cpu" and config.get("dtype") == "float32"
            and exact_int(config.get("threads"), 1), "CPU numerical boundary changed")
    require(exact_int(config.get("seed"), 568)
            and exact_int(config.get("max_length"), 2048)
            and exact_int(config.get("rank"), 8)
            and exact_int(config.get("alpha"), 16)
            and config.get("dropout") == 0.05
            and config.get("learning_rate") == 0.0005,
            "unqualified historical LFM dose/trainer configuration")
    require(config.get("final_targets_opened") is False, "FINAL access claimed")
    require(preflight.get("config") == config and preflight.get("resume") is None,
            "preflight configuration/resume mismatch")
    require(exact_int(preflight.get("train_rows"), 12)
            and exact_int(preflight.get("trainable_parameters"), 2998272),
            "TRAIN cohort or qualified LoRA boundary changed")
    require(receipt.get("config") == config, "checkpoint receipt config mismatch")
    for key in ("global_step", "sample_cursor"):
        require(exact_int(receipt.get(key), steps), "partial checkpoint " + key)
    require(receipt.get("frozen_base_sha256") == summary.get("frozen_base_sha256"),
            "frozen base identity mismatch")
    history = receipt.get("history")
    require(type(history) is list and len(history) == steps, "incomplete history")
    require(type(progress) is list and len(progress) == steps, "incomplete progress")
    require(history == progress, "checkpoint history differs from optimizer log")
    # Verify the original balanced cycle over 12 native-admitted TRAIN seed rows
    # (six independent structures). Repeating a single row is not a full dose.
    order = config.get("sample_order")
    require(type(order) is list and len(order) == 12
            and all(type(sample) is str and sample for sample in order)
            and len(set(order)) == 12, "invalid original TRAIN sampler inventory")
    require(tuple(order) == ORIGINAL_SAMPLE_ORDER,
            "sampler inventory is not the original admitted TRAIN identity/order")
    observed = [event.get("sample_id") for event in progress]
    require(observed == [order[index % 12] for index in range(steps)],
            "optimizer sample order differs from original TRAIN sampler")
    require(summary.get("sample_ids_this_job") == observed, "sample order mismatch")
    for index, event in enumerate(progress, 1):
        require(isinstance(event, dict), "malformed optimizer event")
        require(exact_int(event.get("global_step"), index)
                and exact_int(event.get("sample_cursor"), index),
                "nonsequential optimizer event")
        for key in ("loss", "gradient_norm"):
            value = event.get(key)
            require(type(value) in (int, float) and math.isfinite(value)
                    and (value > 0 if key == "gradient_norm" else True),
                    "invalid optimizer " + key)
        require(type(event.get("loss_tokens")) is int and event["loss_tokens"] > 0,
                "missing supervised tokens")
        require(type(event.get("sequence_tokens")) is int
                and event["sequence_tokens"] >= event["loss_tokens"],
                "invalid supervised sequence length")
        for key in ("prompt_bytes", "completion_bytes"):
            require(type(event.get(key)) is int and event[key] > 0,
                    "invalid training payload " + key)
        elapsed = event.get("seconds")
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                "invalid optimizer step duration")
    totals = {
        "loss_tokens_this_job": sum(e["loss_tokens"] for e in progress),
        "sequence_tokens_this_job": sum(e["sequence_tokens"] for e in progress),
        "bytes_this_job": sum(e["prompt_bytes"] + e["completion_bytes"] for e in progress),
    }
    for field, total in totals.items():
        require(exact_int(summary.get(field), total),
                "aggregate TRAIN receipt mismatch: " + field)
    require(summary.get("first_loss") == progress[0]["loss"]
            and summary.get("last_loss") == progress[-1]["loss"],
            "TRAIN first/last loss mismatch")
    if run_id is not None:
        require(str(run_id).isdigit(), "invalid hosted run ID")
        runner = summary.get("runner")
        require(isinstance(runner, dict)
                and runner.get("github_run_id") == str(run_id),
                "hosted run lineage mismatch")


def check(output, *, arm, source_sha, run_id=None):
    require(output.is_dir() and not output.is_symlink(), "missing/symlinked output")
    require(arm in ARMS, "unrecognized dose arm")
    summary = read_json(output / "summary.json")
    preflight = read_json(output / "preflight.json")
    checkpoint = output / f"checkpoint-{ARMS[arm]:06d}"
    require(checkpoint.is_dir() and not checkpoint.is_symlink(),
            "missing/symlinked final checkpoint")
    receipt = verify_checkpoint(checkpoint)  # Full inventory and file bytes.
    require(sha256_file(checkpoint / "inventory.json")
            == summary.get("checkpoint_inventory_sha256"),
            "checkpoint inventory digest mismatch")
    with (output / "progress.jsonl").open(encoding="utf-8") as stream:
        progress = [json.loads(line) for line in stream if line.strip()]
    validate(summary, preflight, receipt, progress, arm=arm,
             source_sha=source_sha, run_id=run_id)
    print(json.dumps({"arm": arm, "source_sha": source_sha,
                      "global_step": ARMS[arm], "model_quality_claim": False,
                      "status": "training-completion-and-custody-pass"}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    check(args.output, arm=args.arm, source_sha=args.source_sha, run_id=args.run_id)


if __name__ == "__main__":
    main()
