#!/usr/bin/env python3
"""Read-only, model-free custody audit of SPROCKET's immutable fresh96 artifact.

This does not infer model quality, inspect FINAL, invoke inference, or train.
The original training source/run and positive native TRAIN source are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from common import load_training_corpus, profile, read_json, sha256_file, verify_checkpoint

SOURCE = "25b24bf80c593b115321387a2a0374dd5ff02548"
RUN = "43529482591"
MODEL = "kandivault/sprocket-500m"
REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
STEPS = 96
# Frozen, original native-admitted TRAIN/compiler identities (not arbitrary hex receipts).
TRAIN_SHA = "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1"
COMPILER_SHA = "4d8559805df1ff5f3f23277a7bca98e47fa8feb53337aa79e8546d0bb89132f6"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value, expected):
    return type(value) is type(expected) and value == expected


def digest(value):
    return type(value) is str and SHA256.fullmatch(value) is not None


def validate(summary, preflight, checkpoint, progress, campaign):
    """Pure receipt checks for synthetic adversarial tests; no model imports."""
    config = summary.get("config")
    require(type(config) is dict, "missing training configuration")
    require(summary.get("schema_version") == 1
            and summary.get("status") == "completed"
            and summary.get("actual_training") is True, "not a completed real trainer receipt")
    for key in ("global_step", "optimizer_steps_this_job", "sample_cursor"):
        require(exact(summary.get(key), STEPS), "incorrect optimizer completion: " + key)
    require(summary.get("resume") is None, "historical proof must be fresh")
    require(summary.get("checkpoint") == "checkpoint-000096"
            and summary.get("checkpoint_readback_exact") is True, "unverified terminal checkpoint")
    require(summary.get("adapter_changed") is True
            and summary.get("frozen_base_unchanged") is True
            and digest(summary.get("adapter_before_sha256"))
            and digest(summary.get("adapter_after_sha256"))
            and summary["adapter_before_sha256"] != summary["adapter_after_sha256"]
            and digest(summary.get("frozen_base_sha256")), "adapter or frozen-base identity invalid")
    require(summary.get("native_execution") is False
            and summary.get("model_quality_claim") is False, "trainer cannot claim painting or quality")
    require(config.get("source_sha") == SOURCE
            and config.get("profile") == "sprocket"
            and config.get("native_source_sha") == NATIVE_SOURCE
            and config.get("corpus_manifest_sha256") == CORPUS_SHA
            and config.get("train_sha256") == TRAIN_SHA
            and config.get("compiler_sha256") == COMPILER_SHA,
            "trainer/native/TRAIN/compiler source custody differs")
    model = config.get("model")
    require(type(model) is dict and model.get("model_id") == MODEL
            and model.get("revision") == REVISION and model.get("subfolder") == ""
            and model.get("trust_remote_code") is False, "wrong pinned root model")
    require(config.get("device") == "cpu" and config.get("dtype") == "float32"
            and exact(config.get("threads"), 1) and exact(config.get("seed"), 568)
            and exact(config.get("max_length"), 2048)
            and exact(config.get("rank"), 8) and exact(config.get("alpha"), 16)
            and config.get("dropout") == 0.05, "unqualified CPU/LoRA recipe")
    numerics = config.get("numerics")
    require(type(numerics) is dict
            and numerics.get("profile") == "portable-cpu-fp32-v1", "unqualified numerics")
    require(config.get("final_targets_opened") is False
            and config.get("native_execution_by_trainer") is False, "trainer touched FINAL/native authority")
    require(preflight.get("config") == config and preflight.get("resume") is None
            and exact(preflight.get("train_rows"), 12)
            and exact(preflight.get("trainable_parameters"), 4739072),
            "preflight TRAIN/adaptation identity differs")
    require(preflight.get("model_contract") == summary.get("model_contract")
            and preflight.get("numerical_runtime") == summary.get("numerical_runtime"),
            "preflight model/numerical runtime differs")
    require(checkpoint.get("config") == config
            and exact(checkpoint.get("global_step"), STEPS)
            and exact(checkpoint.get("sample_cursor"), STEPS)
            and checkpoint.get("frozen_base_sha256") == summary.get("frozen_base_sha256"),
            "checkpoint optimizer/base custody differs")
    history = checkpoint.get("history")
    require(type(history) is list and type(progress) is list
            and len(history) == STEPS and len(progress) == STEPS
            and history == progress, "checkpoint history/optimizer log incomplete or divergent")
    order = config.get("sample_order")
    require(type(order) is list and len(order) == 12 and len(set(order)) == 12
            and all(type(s) is str and s for s in order), "non-canonical TRAIN sampler")
    require(summary.get("sample_ids_this_job") ==
            [event.get("sample_id") for event in progress], "summary sampled examples differ")
    for index, event in enumerate(progress):
        require(type(event) is dict
                and exact(event.get("global_step"), index + 1)
                and exact(event.get("sample_cursor"), index + 1)
                and event.get("sample_id") == order[index % 12], "invalid per-step optimizer/sampler order")
        for key in ("loss", "gradient_norm"):
            value = event.get(key)
            require(type(value) in (float, int) and math.isfinite(value)
                    and (value > 0 if key == "gradient_norm" else value >= 0),
                    "non-finite/missing training metric: " + key)
        require(type(event.get("loss_tokens")) is int and event["loss_tokens"] > 0
                and type(event.get("sequence_tokens")) is int
                and event["sequence_tokens"] >= event["loss_tokens"],
                "invalid supervised-token accounting")
    require(summary.get("loss_tokens_this_job") ==
            sum(e["loss_tokens"] for e in progress)
            and summary.get("sequence_tokens_this_job") ==
            sum(e["sequence_tokens"] for e in progress), "aggregate token accounting differs")
    require(summary.get("first_loss") == progress[0]["loss"]
            and summary.get("last_loss") == progress[-1]["loss"], "reported losses differ")
    require(type(summary.get("runner")) is dict
            and summary["runner"].get("github_run_id") == RUN, "not original hosted training run")
    require(campaign.get("schema") == "cybersand.agent-painter.sprocket-campaign/1"
            and campaign.get("experiment") == "sprocket-fresh-96-001"
            and campaign.get("source_sha") == SOURCE
            and campaign.get("model") == MODEL
            and campaign.get("revision") == REVISION
            and campaign.get("corpus_manifest_sha256") == CORPUS_SHA
            and exact(campaign.get("target_global_steps"), STEPS)
            and exact(campaign.get("actual_global_steps"), STEPS)
            and exact(campaign.get("optimizer_steps_this_job"), STEPS)
            and exact(campaign.get("result_code"), 0)
            and campaign.get("failure") is None
            and campaign.get("status") == "completed"
            and campaign.get("actual_training") is True
            and campaign.get("checkpoint") == "checkpoint-000096"
            and campaign.get("checkpoint_readback_exact") is True
            and campaign.get("native_execution") is False
            and campaign.get("model_quality_claim") is False
            and campaign.get("final_targets_opened") is False,
            "wrapper receipt cannot certify full fresh96 completion")


def expected_native_sampler(rows, seed=568):
    """Independent reconstruction of the foundation's family-balanced sampler.

    Consume only positively native-admitted TRAIN rows; NEVER open FINAL targets.
    A self-declared list of twelve distinct sample IDs is not sufficient proof.
    """
    buckets = {}
    for row in rows:
        require(row.get("split") == "TRAIN" and type(row.get("id")) is str
                and row.get("family") in ("greebles", "pipework", "architecture"),
                "non-TRAIN row or invalid family in sampler")
        buckets.setdefault(row["family"], []).append(row["id"])
    require(set(buckets) == {"greebles", "pipework", "architecture"}
            and all(len(ids) == 4 for ids in buckets.values())
            and len({item for ids in buckets.values() for item in ids}) == 12,
            "not the original four-seed-row-per-family TRAIN partition")
    for ids in buckets.values():
        ids.sort(key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest())
    return [buckets[family][i] for i in range(4) for family in sorted(buckets)]


def verify_native_train_custody(config, corpus_dir):
    """Actual gate: hash original native manifest, reread admitted TRAIN, bind IDs.

    This is distinct from the model-free synthetic receipt fixture. The common
    native loader independently revalidates positive material/PNG receipts.
    """
    root = Path(corpus_dir)
    require(root.is_dir() and not root.is_symlink(), "missing actual native corpus")
    manifest_path = root / "admitted-corpus.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink()
            and sha256_file(manifest_path) == CORPUS_SHA,
            "wrong or missing immutable native-admitted corpus manifest")
    rows, manifest = load_training_corpus(root)
    require(manifest.get("source_sha") == NATIVE_SOURCE
            and manifest.get("compiler_sha256") == COMPILER_SHA
            and manifest.get("split_files", {}).get("TRAIN", {}).get("sha256") == TRAIN_SHA,
            "original native TRAIN/compiler identity is not independently corroborated")
    groups = {row.get("group_id", row.get("structural_group")) for row in rows}
    require(len(groups) == 6 and all(
        sum(row.get("group_id", row.get("structural_group")) == group for row in rows) == 2
        for group in groups), "original TRAIN group/seed multiplicity differs")
    require(config.get("sample_order") == expected_native_sampler(rows, 568),
            "training order differs from positively native-admitted TRAIN IDs")
    print("Original native TRAIN: 12/12 positive readbacks, six structures; exact sampler order PASS")


def check(output, corpus_dir):
    require(output.is_dir() and not output.is_symlink(), "missing/symlinked artifact output")
    summary_file = output / "summary.json"
    summary = read_json(summary_file)
    preflight = read_json(output / "preflight.json")
    campaign = read_json(output / "campaign-receipt.json")
    checkpoint_dir = output / "checkpoint-000096"
    require(checkpoint_dir.is_dir() and not checkpoint_dir.is_symlink(), "missing true final checkpoint")
    require(not any(p.is_symlink() for p in checkpoint_dir.rglob("*")),
            "checkpoint contains symlinks")
    checkpoint = verify_checkpoint(checkpoint_dir)  # All actual adapter/optimizer/receipt files hashed.
    require(summary.get("checkpoint_inventory_sha256") ==
            sha256_file(checkpoint_dir / "inventory.json"),
            "terminal inventory digest differs")
    require(campaign.get("summary_sha256") == sha256_file(summary_file),
            "wrapper summary hash differs")
    require(campaign.get("wrapper_sha256") == sha256_file(Path(__file__).with_name("campaign.py")),
            "historical campaign wrapper implementation differs")
    require(summary.get("config", {}).get("model") == profile("sprocket", require_supported=True),
            "full immutable model-profile identity mismatch")
    progress = []
    with (output / "progress.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                progress.append(json.loads(line))
    validate(summary, preflight, checkpoint, progress, campaign)
    verify_native_train_custody(summary["config"], corpus_dir)
    print(json.dumps({"audit": "historical-sprocket-fresh96",
                      "source": SOURCE, "run_id": RUN,
                      "optimizer_steps": 96, "checkpoint_inventory": "byte-verified",
                      "original_train_native_admission": "12/12 revalidated",
                      "original_train_order": "independently reproduced",
                      "quality_evidence": False, "native_execution_by_audit": False,
                      "result": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Original native artifact corpus; mandatory for real receipt audit")
    args = parser.parse_args()
    check(args.output, args.corpus)
