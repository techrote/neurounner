#!/usr/bin/env python3
"""WORM W4 evaluation-only salvage of the immutable 180-update failed-job checkpoint.

This does NOT resume or repeat optimization; the failed producer's source identity
and original negative evidence remain unchanged. Host W0 owns all application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import worm_train_w4_v1 as w4
from common import profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint, load_base, verify_checkpoint

PRODUCER_SHA = "45af748feec0da7c25200aad959d0c2214fcb68e"
PRODUCER_RUN = 43701525721
ARTIFACT_ID = 14156798033
ARTIFACT_SHA256 = "0076fdded794ddcbe1153aaeee0b36e9e9f652e2b35b68eb665cde1326ebde53"
TRAIN_SHA = "b41b13c96c112429d76ba4f2a2dfbb50403bcecc24512c838980de536b4af9f2"
DEV_SHA = "fe959db7ad794d5e1db0ab53b35b7f6528bd1de3d7963210e621af70031845ec"


def validate_producer(receipt, original, chosen_profile):
    """Reject wrong step, WORM objective, official model or curriculum source."""
    assert receipt["global_step"] == receipt["sample_cursor"] == 180
    assert len(receipt["history"]) == 180
    assert [e["global_step"] for e in receipt["history"]] == list(range(1, 181))
    assert all(e["gradient_norm"] > 0 and e["loss"] >= 0 for e in receipt["history"])
    config = receipt["config"]
    assert config["source_sha"] == PRODUCER_SHA
    assert config["objective"] == "lfm-worm-action-selection-v4-world-majority-fresh"
    assert config["profile"] == "lfm25_vl"
    assert config["model"] == chosen_profile
    assert config["model"]["revision"] == "fc6221ca597f3315e4f82fc2df606783267b34ba"
    assert config["corpus_manifest"]["WORM_TRAIN_sha256"] == TRAIN_SHA
    assert config["corpus_manifest"]["WORM_DEV_sha256"] == DEV_SHA
    assert config["corpus_manifest"]["W0_source"] == w4.WORM_SHA
    assert config["corpus_manifest"]["support_source"] == w4.SUPPORT_SHA
    assert config["recipe"]["sampler"] == "family-balanced-world-majority-v1-180-distinct-rows"
    assert config["recipe"]["parent"] == "pinned-official-base"
    assert config["recipe"]["rank"] == 8
    assert config["final_targets_opened"] is False
    assert config["native_execution"] is False
    assert sha256_file(original / "worm_train.jsonl") == TRAIN_SHA
    assert sha256_file(original / "worm_dev.jsonl") == DEV_SHA
    assert json.loads((original / "failure.json").read_text())["error_type"] == "AttributeError"
    assert len(json.loads((original / "untrained-first-passes.json").read_text())) == 18
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worm-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_cpu(1)
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    import torch

    chosen_profile = profile("lfm25_vl", require_supported=True)
    checkpoint = args.original / "checkpoint-000180"
    receipt = verify_checkpoint(checkpoint)
    config = validate_producer(receipt, args.original, chosen_profile)
    worm = w4.worm_source(args.worm_root)
    support = w4.support_source(args.support_root, worm)
    rows, custody = w4.dataset(support)
    for split, expected in (("WORM_TRAIN", TRAIN_SHA), ("WORM_DEV", DEV_SHA)):
        path = args.output / (split.lower() + ".jsonl")
        path.write_text("".join(w4.canonical(x) + "\n" for x in rows[split]), encoding="utf-8")
        if sha256_file(path) != expected:
            raise ValueError("source-fenced replay changed original W4 training/DEV bytes")
    if config["model_files"] != json.loads((args.model / "download_receipt.json").read_text())["files"]:
        raise ValueError("pinned full model file inventory changed")

    base, tokenizer = load_base(args.model, chosen_profile)
    model = PeftModel.from_pretrained(
        base, checkpoint / "adapter", is_trainable=True, local_files_only=True)
    adapter = adapter_for(chosen_profile)
    adapter.set_training_mode(model)
    contract = adapter.validate_trainable(model, chosen_profile)
    if fingerprint(model, False) != receipt["frozen_base_sha256"]:
        raise ValueError("frozen official base model fingerprint differs from W4 producer")
    expected_adapter = load_file(str(checkpoint / "adapter" / "adapter_model.safetensors"))
    loaded_adapter = get_peft_model_state_dict(model)
    if expected_adapter.keys() != loaded_adapter.keys() or any(
            not torch.equal(expected_adapter[k], loaded_adapter[k].detach().cpu())
            for k in expected_adapter):
        raise ValueError("LoRA tensor checkpoint reload differs")

    trained = w4.evaluation(model, tokenizer, chosen_profile, rows["WORM_DEV"])
    write_json(args.output / "trained-first-passes.json", trained)
    gallery = args.output / "gallery"
    gallery.mkdir()
    gallery_metrics = []
    for family in w4.FAMILIES:
        for seed in (2101, 2102):
            record, raw = w4.rollout(model, tokenizer, chosen_profile, support,
                                     family, seed, max_steps=48)
            label = f"{family}-model-{seed}"
            (gallery / f"{label}.png").write_bytes(worm.png_bytes(record))
            write_json(gallery / f"{label}.json", record)
            write_json(gallery / f"{label}-raw-first-passes.json", raw)
            gallery_metrics.append({"family": family, "seed": seed,
                                    "policy": "trained-lfm-first-pass",
                                    "metrics": support.statistics(record)})
            for mode in ("random", "wall"):
                control = worm.run(seed, mode=mode, family=family, steps=48)
                worm.write_result(control, gallery)
                gallery_metrics.append({"family": family, "seed": seed,
                                        "policy": mode,
                                        "metrics": support.statistics(control)})
            control = support.rollout(seed, family, 48, support.oracle)
            label = f"{family}-oracle-{seed}"
            (gallery / f"{label}.png").write_bytes(worm.png_bytes(control))
            write_json(gallery / f"{label}.json", control)
            gallery_metrics.append({"family": family, "seed": seed,
                                    "policy": "untrained-oracle",
                                    "metrics": support.statistics(control)})
    write_json(args.output / "gallery-metrics.json", gallery_metrics)
    original_baseline = json.loads((args.original / "untrained-first-passes.json").read_text())
    write_json(args.output / "original-untrained-first-passes.json", original_baseline)
    write_json(args.output / "summary.json", {
        "status": "evaluation-complete-not-producer-success",
        "training": "prior failed W4 run, genuine checkpoint-180 only; no new optimization",
        "optimizer_updates_in_this_run": 0, "producer_steps_verified": 180,
        "producer_run": PRODUCER_RUN, "producer_source": PRODUCER_SHA,
        "producer_artifact_id": ARTIFACT_ID, "producer_artifact_zip_sha256": ARTIFACT_SHA256,
        "checkpoint": "checkpoint-000180", "official_model_revision": chosen_profile["revision"],
        "model_architecture": chosen_profile["architecture"], "model_contract": contract,
        "train_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA,
        "untrained_valid": sum(bool(r["valid_first_pass"]) for r in original_baseline),
        "trained_valid": sum(bool(r["valid_first_pass"]) for r in trained),
        "trained_exact_teacher": sum(r["raw_text"] == r["expected"] for r in trained),
        "trained_distinct_valid_choices": sorted({r["raw_text"] for r in trained if r["valid_first_pass"]}),
        "static_probes_per_arm": len(trained),
        "gallery_metrics": gallery_metrics,
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "native_godot_executions": 0, "sealed_FINAL_opened": False,
        "role": "WORM-001 LFM primary, source-fenced evaluation-only recovery",
    })
    print(w4.canonical({"event": "w4-recovered-model-evaluation-complete",
                        "trained_valid": sum(bool(r["valid_first_pass"]) for r in trained),
                        "model_rollouts": 6}), flush=True)


if __name__ == "__main__":
    main()
