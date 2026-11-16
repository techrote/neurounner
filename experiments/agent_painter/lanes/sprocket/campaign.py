#!/usr/bin/env python3
"""Dedicated Sprocket CPU campaign; shared trainer retains all optimization/custody authority."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
import common
import train as shared_train

MODEL = "kandivault/sprocket-500m"
REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
BRANCH = "codex/agent-painter-sprocket-568"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=24)
    parser.add_argument("--max-seconds", type=int, default=2400)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 96 or not 60 <= args.max_seconds <= 2400:
        parser.error("campaign total step target must be 1..96 and deadline 60..2400s")
    if not 1 <= args.checkpoint_every <= min(24, args.steps):
        parser.error("checkpoint interval must be 1..min(24, target)")
    args.profile, args.seed, args.threads, args.max_length = "sprocket", 568, 1, 2048
    return args


def qualify(args):
    if (not re.fullmatch(r"[0-9a-f]{40}", args.source_sha) or
            os.environ.get("GITHUB_ACTIONS") != "true" or
            os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted" or
            os.environ.get("RUNNER_OS") != "Linux" or
            os.environ.get("GITHUB_REPOSITORY") != "techrote/cybersand" or
            os.environ.get("GITHUB_REF") != "refs/heads/" + BRANCH or
            os.environ.get("GITHUB_SHA") != args.source_sha or
            shared_train.source_sha() != args.source_sha):
        raise ValueError("campaign requires the exact committed GitHub-hosted lane source")
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=common.ROOT, text=True).strip():
        raise ValueError("tracked source was modified after checkout")
    selected = common.profile("sprocket", require_supported=True)
    if (selected.get("model_id"), selected.get("revision"), selected.get("subfolder")) != (MODEL, REVISION, ""):
        raise ValueError("Sprocket pinned root Instruct profile changed")
    if common.sha256_file(args.corpus / "admitted-corpus.json") != CORPUS_SHA:
        raise ValueError("native-admitted corpus manifest changed")
    admission = common.read_json(args.corpus / "admitted-corpus.json")
    if admission.get("state") != "native-admitted" or admission.get("source_sha") != NATIVE_SOURCE:
        raise ValueError("native source/admission changed")
    if args.output.exists():
        raise ValueError("cannot overwrite previous campaign evidence")
    return selected


def main(argv=None):
    args = arguments(argv)
    qualify(args)
    signal.signal(signal.SIGTERM, shared_train.request_stop)
    signal.signal(signal.SIGINT, shared_train.request_stop)
    started = time.monotonic()
    result, failure = None, None
    try:
        result = shared_train.train(args)
        return result
    except Exception as error:
        failure = {"status": "FAILED", "type": type(error).__name__, "error": str(error)}
        if getattr(args, "_output_owned", False):
            common.write_json(args.output / "failure.json", failure)
        raise
    finally:
        if getattr(args, "_output_owned", False):
            summary_file = args.output / "summary.json"
            summary = common.read_json(summary_file) if summary_file.exists() else {}
            common.write_json(args.output / "campaign-receipt.json", {
                "schema": "cybersand.agent-painter.sprocket-campaign/1",
                "experiment": "sprocket-fresh-96-001",
                "source_sha": args.source_sha,
                "wrapper_sha256": common.sha256_file(__file__),
                "model": MODEL, "revision": REVISION, "corpus_manifest_sha256": CORPUS_SHA,
                "target_global_steps": args.steps, "checkpoint_every": args.checkpoint_every,
                "max_seconds": args.max_seconds, "elapsed_seconds": time.monotonic() - started,
                "result_code": result, "failure": failure,
                "actual_global_steps": summary.get("global_step", 0),
                "optimizer_steps_this_job": summary.get("optimizer_steps_this_job", 0),
                "actual_training": summary.get("actual_training", False),
                "status": summary.get("status", "FAILED"),
                "checkpoint": summary.get("checkpoint"),
                "checkpoint_readback_exact": summary.get("checkpoint_readback_exact", False),
                "summary_sha256": common.sha256_file(summary_file) if summary_file.exists() else None,
                "native_execution": False, "model_quality_claim": False,
                "final_targets_opened": False,
            })


if __name__ == "__main__":
    sys.exit(main())
