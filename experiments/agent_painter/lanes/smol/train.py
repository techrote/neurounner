#!/usr/bin/env python3
"""Bounded Smol lane wrapper; the unchanged shared trainer owns real optimization."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

import contracts
from contracts import common
import train as shared_train


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=120, help="total global target, including resumed updates")
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= contracts.MAX_STEPS or not 60 <= args.max_seconds <= contracts.MAX_SECONDS:
        parser.error("this lane is bounded to 1–120 total updates and 60–1200 seconds")
    if not 1 <= args.checkpoint_every <= min(40, args.steps):
        parser.error("checkpoint interval must be 1–min(40, total steps)")
    args.profile, args.seed, args.threads, args.max_length = "smol", contracts.TRAIN_SEED, 1, 2048
    return args


def main(argv=None):
    args = arguments(argv)
    source = contracts.hosted_source(args.source_sha)
    contracts.frozen_manifest(args.corpus)
    selected = common.profile("smol", require_supported=True)
    if (selected["model_id"], selected["revision"]) != (contracts.MODEL_ID, contracts.MODEL_REVISION):
        raise ValueError("Smol model pin changed")
    if args.output.exists():
        raise ValueError("refusing to overwrite training evidence")
    signal.signal(signal.SIGINT, shared_train.request_stop)
    signal.signal(signal.SIGTERM, shared_train.request_stop)
    started = time.monotonic()
    result = None
    failure = None
    try:
        result = shared_train.train(args)
        return result
    except Exception as error:
        failure = {"status": "FAILED", "error": str(error), "type": type(error).__name__}
        if getattr(args, "_output_owned", False):
            common.write_json(args.output / "failure.json", failure)
        raise
    finally:
        if getattr(args, "_output_owned", False):
            summary_path = args.output / "summary.json"
            summary = common.read_json(summary_path) if summary_path.exists() else {}
            common.write_json(args.output / "lane-receipt.json", {
                "schema_version": 1, "lane": "smol", "source_sha": source,
                "wrapper_sha256": common.sha256_file(__file__),
                "contracts_sha256": common.sha256_file(contracts.__file__),
                "target_global_steps": args.steps, "checkpoint_every": args.checkpoint_every,
                "max_seconds": args.max_seconds, "wall_seconds": time.monotonic() - started,
                "exit_code": result, "status": summary.get("status", "FAILED"), "failure": failure,
                "actual_training": summary.get("actual_training", False),
                "actual_global_step": summary.get("global_step", 0),
                "optimizer_steps_this_job": summary.get("optimizer_steps_this_job", 0),
                "checkpoint": summary.get("checkpoint"),
                "checkpoint_readback_exact": summary.get("checkpoint_readback_exact", False),
                "summary_sha256": common.sha256_file(summary_path) if summary_path.exists() else None,
                "native_execution": False, "model_quality_claim": False, "final_targets_opened": False})


if __name__ == "__main__":
    sys.exit(main())
