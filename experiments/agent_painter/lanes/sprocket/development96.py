#!/usr/bin/env python3
"""Source-custodied Sprocket step96 DEVELOPMENT comparison; never native acceptance.

Only the six original DEVELOPMENT requests and their two host-seed views are read.
Raw outputs are immutable evidence; no JSON repair, retry, fallback or FINAL access.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import resource
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from common import (compiler_module, profile, read_json, sha256_file, verify_checkpoint,
                    write_json)
from lane_adapters import adapter_identity
from numerics import configure_cpu
from train import SYSTEM, load_base
from development import development_groups, evaluate

TRAINING_SOURCE = "25b24bf80c593b115321387a2a0374dd5ff02548"
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
MODEL_REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
CAMPAIGN_RUN = 43529482591
EVALUATOR_PATH = "lanes/sprocket/development96.py"


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def validate_campaign(directory, checkpoint_directory, selected):
    receipt_file = directory / "campaign-receipt.json"
    summary_file = directory / "summary.json"
    receipt, summary = read_json(receipt_file), read_json(summary_file)
    ensure(receipt.get("schema") == "cybersand.agent-painter.sprocket-campaign/1",
           "unknown campaign receipt schema")
    ensure(receipt.get("source_sha") == TRAINING_SOURCE and
           receipt.get("model") == selected["model_id"] and
           receipt.get("revision") == selected["revision"] and
           receipt.get("corpus_manifest_sha256") == CORPUS_SHA,
           "campaign/model/data provenance mismatch")
    ensure(receipt.get("target_global_steps") == 96 and
           receipt.get("checkpoint_every") == 24 and
           receipt.get("max_seconds") == 2400 and
           receipt.get("result_code") == 0 and
           receipt.get("status") == "completed" and
           receipt.get("actual_global_steps") == 96 and
           receipt.get("optimizer_steps_this_job") == 96 and
           receipt.get("actual_training") is True and
           receipt.get("checkpoint") == "checkpoint-000096" and
           receipt.get("checkpoint_readback_exact") is True,
           "96-step campaign did not finish with real verified updates")
    ensure(receipt.get("summary_sha256") == sha256_file(summary_file),
           "campaign summary bytes changed after training")
    for key, required in (("status", "completed"), ("actual_training", True),
                           ("global_step", 96), ("optimizer_steps_this_job", 96),
                           ("sample_cursor", 96), ("checkpoint", "checkpoint-000096"),
                           ("checkpoint_readback_exact", True),
                           ("frozen_base_unchanged", True),
                           ("adapter_changed", True), ("resume", None),
                           ("native_execution", False)):
        ensure(summary.get(key) == required, "campaign summary mismatch: " + key)
    checkpoint = verify_checkpoint(checkpoint_directory)
    config = checkpoint["config"]
    identity = adapter_identity(selected)
    files = dict(identity["source_files"])
    ensure(files.pop(EVALUATOR_PATH, None) == sha256_file(__file__),
           "new evaluation-only script is not in current adapter source inventory")
    original_identity = {**identity, "source_files": files}
    checks = {
        "global_step": checkpoint.get("global_step") == 96,
        "cursor": checkpoint.get("sample_cursor") == 96,
        "history_length": len(checkpoint.get("history", [])) == 96,
        "history_steps": all(x.get("global_step") == i for i, x in enumerate(checkpoint.get("history", []), 1)),
        "campaign_checkpoint": summary.get("checkpoint_inventory_sha256") == sha256_file(checkpoint_directory / "inventory.json"),
        "source": config.get("source_sha") == TRAINING_SOURCE,
        "profile": config.get("profile") == "sprocket",
        "model": config.get("model") == selected,
        "native_source": config.get("native_source_sha") == NATIVE_SOURCE,
        "corpus": config.get("corpus_manifest_sha256") == CORPUS_SHA,
        "system": config.get("system") == SYSTEM,
        "compiler": config.get("compiler_sha256") == sha256_file(ROOT / "compiler.py"),
        "trainer": config.get("trainer_sha256") == sha256_file(ROOT / "training" / "train.py"),
        "original_adapter": config.get("adapter_contract") == original_identity,
    }
    failures = [key for key, ok in checks.items() if not ok]
    ensure(not failures, "step96 source/checkpoint contract mismatch: " + ", ".join(failures))
    return checkpoint, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    ensure(args.max_new_tokens == 512 and not args.output.exists(),
           "DEVELOPMENT token cap must be 512; output must be new")
    ensure(os.getenv("GITHUB_ACTIONS") == "true" and
           os.getenv("RUNNER_ENVIRONMENT") == "github-hosted" and
           os.getenv("RUNNER_OS") == "Linux" and
           os.getenv("GITHUB_REPOSITORY") == "techrote/cybersand" and
           os.getenv("GITHUB_REF") == "refs/heads/codex/agent-painter-sprocket-568" and
           bool(re.fullmatch(r"[0-9a-f]{40}", os.getenv("GITHUB_SHA", ""))),
           "hosted exact-lane source is required")
    selected = profile("sprocket", require_supported=True)
    ensure((selected.get("model_id"), selected.get("revision"), selected.get("subfolder"))
           == ("kandivault/sprocket-500m", MODEL_REVISION, ""),
           "pinned root Instruct profile mismatch")
    checkpoint_dir = args.campaign / "checkpoint-000096"
    checkpoint, summary = validate_campaign(args.campaign, checkpoint_dir, selected)
    groups = development_groups(args.corpus)
    numerics, runtime = configure_cpu(1)
    ensure(numerics == checkpoint["config"]["numerics"],
           "CPU numerical model differs from original campaign")
    import torch
    from peft import PeftModel
    torch.manual_seed(568)
    args.output.mkdir(parents=True)
    model, tokenizer = load_base(args.model, selected)
    baseline = evaluate(model, tokenizer, groups, compiler_module(), args.output,
                        "untuned", args.max_new_tokens, torch)
    tuned = PeftModel.from_pretrained(
        model, checkpoint_dir / "adapter", is_trainable=False, local_files_only=True)
    candidate = evaluate(tuned, tokenizer, groups, compiler_module(), args.output,
                         "step96", args.max_new_tokens, torch)
    write_json(args.output / "comparison.json", {
        "schema": "cybersand.sprocket-development-first-pass/2",
        "evaluation_source": os.environ["GITHUB_SHA"],
        "trainer_source": TRAINING_SOURCE,
        "trainer_actions_run": CAMPAIGN_RUN,
        "model": selected["model_id"],
        "revision": selected["revision"],
        "native_source": NATIVE_SOURCE,
        "corpus_manifest_sha256": CORPUS_SHA,
        "checkpoint_step": 96,
        "checkpoint_inventory_sha256": sha256_file(checkpoint_dir / "inventory.json"),
        "checkpoint_readback_verified": True,
        "actual_optimizer_steps": summary["optimizer_steps_this_job"],
        "max_new_tokens": args.max_new_tokens,
        "decoding": "first-pass greedy; no repair, retries or fallback",
        "arms": [baseline, candidate],
        "numerics": numerics,
        "runtime": runtime,
        "comparison_scope": "six original DEVELOPMENT language prompts x two host seeds",
        "native_execution": False,
        "native_confirmed_paintings": 0,
        "final_targets_opened": False,
    })
    for arm in ("untuned", "step96"):
        for row in read_json(args.output / (arm + "-raw-development.json")):
            print(json.dumps({
                "arm": arm, "group": row["structural_group"], "family": row["family"],
                "eos": row["raw"]["eos_terminated"],
                "json": row["raw"]["json_valid"], "schema": row["raw"]["program_valid"],
                "compiled_seed_views": row["admitted_seed_programs"],
                "raw_prefix": row["raw"]["raw_text"][:1200],
            }, ensure_ascii=True, sort_keys=True), flush=True)
    print("STEP96 DEVELOPMENT summary", baseline, candidate, flush=True)


if __name__ == "__main__":
    main()
