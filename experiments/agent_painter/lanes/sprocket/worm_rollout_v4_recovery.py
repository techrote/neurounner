#!/usr/bin/env python3
"""WORM SPROCKET W4 source-fenced evaluation recovery (NO optimizer updates).

The original 43715212047 evaluation timed out decoding untrained root
completions before reaching its trained arm. Preserve that partial artifact
unaltered. This recovery evaluates only the unchanged trained checkpoint,
together with deterministic same-seed W0 controls, without changing its
greedy policy, action semantics, W0, model source or checkpoint.

The original untrained raw decisions are retained in the failed artifact and
W4 training run; they must not be passed off as measurements from this job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1] / "training"
sys.path[:0] = [str(TRAINING), str(HERE)]
import train as shared
from common import profile, sha256_file, verify_checkpoint, write_json
from lane_adapters import adapter_for
from numerics import configure_cpu
import worm_v1 as contract
import worm_v4 as objective
from worm_rollout_v4 import roll

ORIGINAL_FAILED_RUN = 43715212047
ORIGINAL_PARTIAL_ARTIFACT = 14162494880
ORIGINAL_W4_PRODUCER = 43710874563
ORIGINAL_W4_SOURCE = "c14795efe978dadeaa9e0da60f6273e18ab17b4d"


def spatial(record, prefix=24):
    points = record["trajectory"]
    def extent(seq):
        xs, ys = zip(*seq)
        origin = points[0]
        return {"unique_cells": len({tuple(p) for p in seq}),
                "bbox_width": max(xs) - min(xs) + 1,
                "bbox_height": max(ys) - min(ys) + 1,
                "max_manhattan_from_start": max(
                    abs(x - origin[0]) + abs(y - origin[1]) for x, y in seq)}
    return {"full": extent(points), "prefix24": extent(points[:prefix + 1])}


def main(args):
    if args.steps != 24 or args.training_source != ORIGINAL_W4_SOURCE:
        raise ValueError("recovery must retain exact original 24-step W4 source lease")
    started = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    numerical, runtime = configure_cpu(1)
    selected = profile("sprocket", require_supported=True)
    w0 = contract.load_w0(args.w0)
    checkpoint = verify_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    if (cfg.get("schema") != objective.SCHEMA or cfg.get("model") != selected or
        cfg.get("worm_common_git_commit") != objective.W0_COMMIT or
        cfg.get("worm_support_git_commit") != objective.SUPPORT_COMMIT or
        cfg.get("worm_common_sha256") != sha256_file(args.w0) or
        cfg.get("source_sha") != args.training_source or
        cfg.get("synthetic_not_native") is not True or
        cfg.get("final_opened") is not False or
        cfg.get("numerics") != numerical):
        raise ValueError("immutable training/model/native-safety contract mismatched")
    import torch
    from peft import PeftModel
    base, tokenizer = shared.load_base(args.model, selected)
    adapter = adapter_for(selected)
    tuned = PeftModel.from_pretrained(
        base, args.checkpoint / "adapter", is_trainable=False, local_files_only=True)
    rows = []
    for family in contract.FAMILIES:
        for seed in contract.EVAL_SEEDS:
            for mode in ("random", "wall", "fidget"):
                record = w0.run(seed, mode, family, steps=args.steps)
                w0.write_result(record, out / "png-and-json")
                rows.append({"family": family, "seed": seed, "mode": mode,
                             "model": False, "metrics": record["metrics"],
                             "spatial": spatial(record)})
            # Run the actual CURRENT-observation trained model: do not provide
            # a policy lookup, target, future state, a retry, or a fallback teacher.
            record, receipts = roll(w0, tuned, tokenizer, adapter, seed, family, args.steps)
            record["mode"] = "sprocket-trained"
            record["engine"] = "W0-source-faithful-model-first-pass-not-native"
            record["model_first_passes"] = receipts
            w0.write_result(record, out / "png-and-json")
            rows.append({"family": family, "seed": seed, "mode": "trained",
                         "model": True, "metrics": record["metrics"],
                         "spatial": spatial(record), "decisions": len(receipts),
                         "eos5": sum(r["eos5"] for r in receipts),
                         "legal_raw": sum(r["legal_raw"] for r in receipts),
                         "actions": {a: sum(r["parsed_action_exact"] == a for r in receipts)
                                     for a in contract.ACTIONS}})
            write_json(out / "partial-progress.json",
                       {"completed_rows": len(rows), "trained_rollouts":
                        sum(r["mode"] == "trained" for r in rows),
                        "all_rows": rows, "status": "in-progress-not-complete"})
    write_json(out / "comparison.json",
               {"schema": "cybersand.worm.sprocket-steering-code-v4/recovery1",
                "model": selected["model_id"], "revision": selected["revision"],
                "training_source": args.training_source,
                "training_run": ORIGINAL_W4_PRODUCER,
                "original_failed_eval_run": ORIGINAL_FAILED_RUN,
                "original_partial_artifact": ORIGINAL_PARTIAL_ARTIFACT,
                "untuned_in_this_job": False,
                "untuned_first_pass_in_original_partial_artifact": True,
                "trained_first_pass_in_this_job": True,
                "pinned_w0": objective.W0_COMMIT,
                "adapter_inventory_sha256": sha256_file(
                    args.checkpoint / "inventory.json"),
                "family_count": len(contract.FAMILIES),
                "seed_groups": list(contract.EVAL_SEEDS),
                "decisions_budget": args.steps,
                "complete": True, "native": False, "sealed_final_opened": False,
                "full_art_v2": False, "real_optimizer_steps_in_this_job": 0,
                "trained_raw_decisions": sum(r.get("decisions", 0) for r in rows),
                "trained_raw_valid_eos5": sum(r.get("eos5", 0) for r in rows),
                "trained_raw_legal": sum(r.get("legal_raw", 0) for r in rows),
                "unique_model_png_sha256": len({hashlib.sha256(
                    (out / "png-and-json" /
                     f"{r['family']}-sprocket-trained-{r['seed']}.png").read_bytes()).hexdigest()
                    for r in rows if r["mode"] == "trained"}),
                "rollouts": rows,
                "wall_seconds": time.monotonic() - started,
                "cpu_seconds": time.process_time(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "numerics": numerical, "numerical_runtime": runtime})
    print(json.dumps({"status": "complete-original-W4-train-only-visual-evaluation",
                      "trained_rollouts": sum(r["mode"] == "trained" for r in rows),
                      "controls": sum(r["mode"] != "trained" for r in rows),
                      "trained_decisions": sum(r.get("decisions", 0) for r in rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w0", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-source", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        if args.output.exists():
            write_json(args.output / "failure.json",
                       {"error": str(exc), "exception": type(exc).__name__,
                        "status": "failed-or-partial", "do_not_claim_success": True})
        raise
