#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRAIN001_REFERENCE = {
    "source": "docs/TRAIN_001_CPU_LORA.md stage 1",
    "cpu_model": "AMD EPYC 7763 64-Core Processor",
    "logical_cpu_count": 4,
    "optimizer_steps": 24,
    "sequence_tokens": 3683,
    "train_seconds": 31.43,
    "sequence_tokens_per_second": 117.18,
    "seconds_per_step": 31.43 / 24,
    "peak_rss_kib": 1748660,
    "trainable_parameters": 2442240,
    "trainable_fraction": 0.01783,
    "checkpoint_bytes": 29721169,
    "first_loss": 4.7135,
    "final_loss": 3.5182,
}


def ratio(a: float | int | None, b: float | int | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return float(a) / float(b)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--resume", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    full = load(args.full)
    lora = load(args.lora)
    resume = load(args.resume)
    for name, obj in [("full", full), ("lora", lora), ("resume", resume)]:
        if obj.get("status") != "success":
            raise SystemExit(f"{name} evidence is not successful")

    ft = full["training"]
    lt = lora["training"]
    fm = full["memory"]
    lm = lora["memory"]
    fc = full["checkpoint"]
    lc = lora["checkpoint"]

    full_cpu = full["runner"]["cpu_model"]
    lora_cpu = lora["runner"]["cpu_model"]
    same_cpu_model = full_cpu == lora_cpu

    summary = {
        "schema_version": 1,
        "experiment": "TRAIN-004",
        "status": "success",
        "source_commit": full.get("source_commit"),
        "base_model": full["base_model"],
        "base_revision": full["base_revision"],
        "dataset": full["dataset"],
        "dataset_sha256": full["dataset_sha256"],
        "same_run_control": {
            "full": {
                "cpu_model": full_cpu,
                "logical_cpu_count": full["runner"]["logical_cpu_count"],
                "trainable_parameters": full["trainable_parameters"],
                "trainable_fraction": full["trainable_fraction"],
                "optimizer_steps": ft["steps_this_run"],
                "sequence_tokens": ft["sequence_tokens"],
                "supervised_tokens": ft["supervised_tokens"],
                "train_seconds": ft["train_seconds"],
                "sequence_tokens_per_second": ft["sequence_tokens_per_second"],
                "supervised_tokens_per_second": ft["supervised_tokens_per_second"],
                "seconds_per_step": ft["seconds_per_step"],
                "first_loss": ft["first_loss"],
                "final_loss": ft["final_loss"],
                "mean_loss": ft["mean_loss"],
                "loss_delta": ft["final_loss"] - ft["first_loss"],
                "peak_rss_kib": fm["peak_rss_kib"],
                "peak_growth_after_first_step_kib": fm[
                    "peak_growth_after_first_step_kib"
                ],
                "checkpoint_bytes": fc["bytes"],
                "checkpoint_manifest_sha256": fc["manifest_sha256"],
                "checkpoint_save_seconds": fc["save_seconds"],
            },
            "lora": {
                "cpu_model": lora_cpu,
                "logical_cpu_count": lora["runner"]["logical_cpu_count"],
                "trainable_parameters": lora["trainable_parameters"],
                "trainable_fraction": lora["trainable_fraction"],
                "optimizer_steps": lt["steps_this_run"],
                "sequence_tokens": lt["sequence_tokens"],
                "supervised_tokens": lt["supervised_tokens"],
                "train_seconds": lt["train_seconds"],
                "sequence_tokens_per_second": lt["sequence_tokens_per_second"],
                "supervised_tokens_per_second": lt["supervised_tokens_per_second"],
                "seconds_per_step": lt["seconds_per_step"],
                "first_loss": lt["first_loss"],
                "final_loss": lt["final_loss"],
                "mean_loss": lt["mean_loss"],
                "loss_delta": lt["final_loss"] - lt["first_loss"],
                "peak_rss_kib": lm["peak_rss_kib"],
                "checkpoint_bytes": lc["bytes"],
                "checkpoint_manifest_sha256": lc["manifest_sha256"],
                "checkpoint_save_seconds": lc["save_seconds"],
            },
            "ratios_full_over_lora": {
                "trainable_parameters": ratio(
                    full["trainable_parameters"], lora["trainable_parameters"]
                ),
                "sequence_tokens_per_second": ratio(
                    ft["sequence_tokens_per_second"],
                    lt["sequence_tokens_per_second"],
                ),
                "supervised_tokens_per_second": ratio(
                    ft["supervised_tokens_per_second"],
                    lt["supervised_tokens_per_second"],
                ),
                "seconds_per_step": ratio(
                    ft["seconds_per_step"], lt["seconds_per_step"]
                ),
                "peak_rss": ratio(fm["peak_rss_kib"], lm["peak_rss_kib"]),
                "checkpoint_size": ratio(fc["bytes"], lc["bytes"]),
            },
            "slowdown_lora_tps_over_full_tps": ratio(
                lt["sequence_tokens_per_second"],
                ft["sequence_tokens_per_second"],
            ),
            "cpu_models_match": same_cpu_model,
            "throughput_caveat": (
                "Full and LoRA control landed on the same CPU model."
                if same_cpu_model
                else (
                    "Full and LoRA control landed on different GitHub-hosted CPU "
                    "models; the direct throughput ratio is therefore host-confounded. "
                    "Use it together with the pinned TRAIN-001 EPYC 7763 reference."
                )
            ),
        },
        "train001_reference": TRAIN001_REFERENCE,
        "full_vs_train001_reference": {
            "throughput_full_over_reference": ratio(
                ft["sequence_tokens_per_second"],
                TRAIN001_REFERENCE["sequence_tokens_per_second"],
            ),
            "slowdown_reference_tps_over_full_tps": ratio(
                TRAIN001_REFERENCE["sequence_tokens_per_second"],
                ft["sequence_tokens_per_second"],
            ),
            "seconds_per_step_full_over_reference": ratio(
                ft["seconds_per_step"],
                TRAIN001_REFERENCE["seconds_per_step"],
            ),
            "peak_rss_full_over_reference": ratio(
                fm["peak_rss_kib"], TRAIN001_REFERENCE["peak_rss_kib"]
            ),
            "checkpoint_full_over_reference": ratio(
                fc["bytes"], TRAIN001_REFERENCE["checkpoint_bytes"]
            ),
            "trainable_full_over_reference": ratio(
                full["trainable_parameters"],
                TRAIN001_REFERENCE["trainable_parameters"],
            ),
            "cpu_match": full_cpu == TRAIN001_REFERENCE["cpu_model"],
        },
        "fresh_runner_resume": {
            "cpu_model": resume["runner"]["cpu_model"],
            "logical_cpu_count": resume["runner"]["logical_cpu_count"],
            "manifest_sha256": resume["verification"]["manifest_sha256"],
            "verified_file_count": resume["verification"]["file_count"],
            "additional_steps": resume["training"]["steps_this_run"],
            "global_step_before": resume["global_step_before"],
            "global_step_after": resume["global_step_after"],
            "first_loss": resume["training"]["first_loss"],
            "final_loss": resume["training"]["final_loss"],
            "sequence_tokens_per_second": resume["training"][
                "sequence_tokens_per_second"
            ],
            "peak_rss_kib": resume["training"]["memory_after_training"][
                "peak_rss_kib"
            ],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("TRAIN004_SUMMARY_BEGIN")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("TRAIN004_SUMMARY_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
