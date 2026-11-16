#!/usr/bin/env python3
"""Versioned bounded VOID campaign on a real GitHub CPU runner; shared trainer unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import signal
import subprocess
import sys

TRAINING = Path(__file__).resolve().parents[2] / "training"
sys.path.insert(0, str(TRAINING))
import train as shared_train
from common import ROOT, profile, read_json, sha256_file, write_json

PARENT = "2a565df45efc568eb96a2166e0b77f299e289625"
NATIVE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
MODEL = "dotlabs/void.1"
REVISION = "9c2be5087772867d45961abd11f6b35110617563"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-sha", "model", "corpus", "output"):
        parser.add_argument("--" + name, required=True, type=Path if name != "source-sha" else str)
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (os.environ.get("GITHUB_ACTIONS") != "true" or
            os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted" or
            os.environ.get("RUNNER_OS") != "Linux" or
            source != args.source_sha or os.environ.get("GITHUB_SHA") != source):
        raise ValueError("VOID campaign requires an exact-source Linux GitHub-hosted runner")
    selected = profile("void", require_supported=True)
    if (selected["model_id"], selected["revision"]) != (MODEL, REVISION):
        raise ValueError("VOID campaign model revision differs from reviewed source")
    admission_file = args.corpus / "admitted-corpus.json"
    admission = read_json(admission_file)
    if admission.get("source_sha") != NATIVE or sha256_file(admission_file) != CORPUS_SHA:
        raise ValueError("VOID campaign native corpus/source differs from frozen admission")
    if args.output.exists():
        raise ValueError("VOID campaign refuses to overwrite existing evidence")
    # Shared trainer always reads resume; this stage is explicitly fresh, not an exact resume.
    args.resume = None
    args.profile = "void"
    args.steps = 120
    args.max_seconds = 1800
    args.checkpoint_every = 40
    args.seed = 568
    args.threads = 1
    args.max_length = 2048
    signal.signal(signal.SIGTERM, shared_train.request_stop)
    signal.signal(signal.SIGINT, shared_train.request_stop)
    try:
        return shared_train.train(args)
    except Exception as error:
        if getattr(args, "_output_owned", False):
            write_json(args.output / "failure.json", {
                "status": "FAILED", "type": type(error).__name__, "error": str(error),
                "source_sha": source, "native_execution": False})
        raise


if __name__ == "__main__":
    sys.exit(main())
