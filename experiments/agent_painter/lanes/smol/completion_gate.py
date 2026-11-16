#!/usr/bin/env python3
"""Fail-closed SMOL fresh120 training promotion gate; partial checkpoints remain evidence.

This is model-free: it revalidates the original TRAIN rows and native
admission/readback to independently reconstruct the frozen sampler. It does
not open DEVELOPMENT or sealed FINAL targets. The existing native workflow
must not start DEVELOPMENT/native replay unless one fresh 120-step training
job completed its exact requested global target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

TARGET = 120
CHECKPOINT = "checkpoint-000120"
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contracts


def _require(obj, key, expected):
    value = obj.get(key)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"non-promotable SMOL fresh120: {key}: {value!r}, expected {expected!r}")



def frozen_train_order(corpus: Path) -> list[str]:
    """Reconstruct the actual reviewed TRAIN sampler from positive native evidence.

    This reads TRAIN rows and their native receipts only. The sealed FINAL
    contents and DEVELOPMENT targets are never opened.
    """
    manifest = contracts.frozen_manifest(corpus)
    rows, admission = contracts.common.load_training_corpus(corpus)
    if admission != manifest or len(rows) != 12:
        raise ValueError("non-promotable SMOL fresh120: frozen TRAIN corpus changed")
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault(row["family"], []).append(index)
    if set(buckets) != {"greebles", "pipework", "architecture"}:
        raise ValueError("non-promotable SMOL fresh120: TRAIN family partition changed")
    for indexes in buckets.values():
        indexes.sort(key=lambda i: hashlib.sha256(
            f"568:{rows[i]['id']}".encode("utf-8")).digest())
    indexes = [buckets[family][i]
               for i in range(max(map(len, buckets.values())))
               for family in sorted(buckets) if i < len(buckets[family])]
    return [rows[index]["id"] for index in indexes]


def verify_trace(training, summary, receipt, config, expected_order):
    """Model-free corroboration of the real trainer's checkpoint and progress."""
    selected = contracts.common.profile("smol")
    for key, expected in {
        "train_sha256": contracts.TRAIN_SHA256,
        "compiler_sha256": contracts.COMPILER_SHA256,
        "seed": 568, "device": "cpu", "dtype": "float32", "threads": 1,
        "max_length": 2048, "rank": 8, "alpha": 16, "dropout": 0.05,
        "learning_rate": 0.0005, "native_execution_by_trainer": False,
        "final_targets_opened": False,
    }.items():
        _require(config, key, expected)
    model = config.get("model")
    if type(model) is not dict:
        raise ValueError("non-promotable SMOL fresh120: missing model config")
    for key in ("model_id", "revision", "trust_remote_code"):
        _require(model, key, selected[key])
    if not isinstance(config.get("numerics"), dict) or config["numerics"].get("profile") != "portable-cpu-fp32-v1":
        raise ValueError("non-promotable SMOL fresh120: numerical recipe mismatch")
    order = config.get("sample_order")
    if type(order) is not list or len(order) != 12 or len(set(order)) != 12 or not all(type(x) is str and x for x in order):
        raise ValueError("non-promotable SMOL fresh120: invalid frozen TRAIN order")
    if order != expected_order:
        raise ValueError("non-promotable SMOL fresh120: frozen TRAIN sample order differs from native-admitted rows")
    raw = (training / "progress.jsonl").read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if len(lines) != TARGET or any(not line.strip() for line in lines):
        raise ValueError("non-promotable SMOL fresh120: incomplete optimizer progress trace")
    events = [contracts.common.strict_json(line) for line in lines]
    if type(receipt.get("history")) is not list or receipt["history"] != events:
        raise ValueError("non-promotable SMOL fresh120: checkpoint history differs from progress")
    ids = []
    losses = sequences = sizes = 0
    for index, event in enumerate(events):
        if type(event) is not dict:
            raise ValueError("non-promotable SMOL fresh120: invalid optimizer event")
        for key in ("global_step", "sample_cursor"):
            _require(event, key, index + 1)
        _require(event, "sample_id", order[index % 12])
        ids.append(event["sample_id"])
        for key in ("loss", "gradient_norm", "loss_tokens", "sequence_tokens", "prompt_bytes",
                    "completion_bytes", "seconds"):
            value = event.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or (
                value < 0 if key == "seconds" else value <= 0
            ):
                raise ValueError(f"non-promotable SMOL fresh120: invalid {key} at optimizer step {index+1}")
        for key in ("loss_tokens", "sequence_tokens", "prompt_bytes", "completion_bytes"):
            if type(event[key]) is not int:
                raise ValueError("non-promotable SMOL fresh120: noninteger token/byte accounting")
        if event["loss_tokens"] > event["sequence_tokens"]:
            raise ValueError("non-promotable SMOL fresh120: invalid token accounting")
        losses += event["loss_tokens"]
        sequences += event["sequence_tokens"]
        sizes += event["prompt_bytes"] + event["completion_bytes"]
    for key, expected in {
        "sample_ids_this_job": ids, "loss_tokens_this_job": losses,
        "sequence_tokens_this_job": sequences, "bytes_this_job": sizes,
        "first_loss": events[0]["loss"], "last_loss": events[-1]["loss"],
        "frozen_base_sha256": receipt.get("frozen_base_sha256"),
    }.items():
        _require(summary, key, expected)
    if summary.get("numerical_runtime") != receipt.get("numerical_runtime"):
        raise ValueError("non-promotable SMOL fresh120: numerical runtime mismatch")
    fingerprints = [summary.get(key) for key in ("frozen_base_sha256", "adapter_before_sha256", "adapter_after_sha256")]
    if not all(type(x) is str and len(x) == 64 and all(c in "0123456789abcdef" for c in x) for x in fingerprints):
        raise ValueError("non-promotable SMOL fresh120: invalid training fingerprints")
    if fingerprints[1] == fingerprints[2]:
        raise ValueError("non-promotable SMOL fresh120: adapter unchanged")
    return {"optimizer_events": TARGET, "progress_sha256": hashlib.sha256(raw).hexdigest()}


def verify_completion(training: Path, exit_code: int, source_sha: str, corpus: Path) -> dict:
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError("non-promotable SMOL fresh120: trainer exit must be 0; partial checkpoint is evidence only")
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("non-promotable SMOL fresh120: invalid exact source SHA")
    summary = json.loads((training / "summary.json").read_text(encoding="utf-8"))
    lane = json.loads((training / "lane-receipt.json").read_text(encoding="utf-8"))
    if type(summary) is not dict or type(lane) is not dict:
        raise ValueError("non-promotable SMOL fresh120: receipts must be objects")
    requirements = {
        "status": "completed",
        "actual_training": True,
        "global_step": TARGET,
        "sample_cursor": TARGET,
        "optimizer_steps_this_job": TARGET,
        "checkpoint": CHECKPOINT,
        "checkpoint_readback_exact": True,
        "frozen_base_unchanged": True,
        "adapter_changed": True,
        "resume": None,
        "native_execution": False,
        "model_quality_claim": False,
    }
    for k, v in requirements.items():
        _require(summary, k, v)
    for k, v in {
        "lane": "smol",
        "source_sha": source_sha,
        "target_global_steps": TARGET,
        "actual_training": True,
        "actual_global_step": TARGET,
        "optimizer_steps_this_job": TARGET,
        "checkpoint": CHECKPOINT,
        "checkpoint_readback_exact": True,
        "status": "completed",
        "exit_code": 0,
        "native_execution": False,
        "final_targets_opened": False,
    }.items():
        _require(lane, k, v)
    config = summary.get("config")
    if type(config) is not dict:
        raise ValueError("non-promotable SMOL fresh120: missing configuration")
    for k, v in {
        "source_sha": source_sha, "native_source_sha": NATIVE_SOURCE,
        "corpus_manifest_sha256": CORPUS, "profile": "smol",
    }.items():
        _require(config, k, v)
    checkpoint = training / CHECKPOINT
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("non-promotable SMOL fresh120: missing real checkpoint directory")
    for file in ("receipt.json", "inventory.json", "adapter/adapter_model.safetensors"):
        path = checkpoint / file
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"non-promotable SMOL fresh120: missing/empty checkpoint member {file}")
    inventory_hash = hashlib.sha256((checkpoint / "inventory.json").read_bytes()).hexdigest()
    _require(summary, "checkpoint_inventory_sha256", inventory_hash)
    receipt = contracts.common.verify_checkpoint(checkpoint)
    for k, v in {"global_step": TARGET, "sample_cursor": TARGET, "config": config}.items():
        _require(receipt, k, v)
    checkpoint_bytes = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    _require(summary, "checkpoint_bytes", checkpoint_bytes)
    trace = verify_trace(training, summary, receipt, config, frozen_train_order(corpus))
    return {
        "trace": trace,
        "schema": "cybersand.agent-painter.smol-fresh120-promotion/1",
        "source_sha": source_sha, "completed_step": TARGET,
        "checkpoint": CHECKPOINT, "checkpoint_inventory_sha256": inventory_hash,
        "promotable": True, "training": False,
        "native_execution": False, "final_targets_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="unchanged native-admitted original corpus; reads TRAIN only")
    args = parser.parse_args()
    print(json.dumps(verify_completion(args.training, args.exit_code, args.source_sha, args.corpus), sort_keys=True))


if __name__ == "__main__":
    main()
