#!/usr/bin/env python3
"""Fail-closed, model-free promotion gate for the exact VOID fresh120 run.

A partial checkpoint is evidence, not a 120-step model. Verify original trainer
exit, summary, complete checkpoint inventory and optimizer-step receipts before
allowing DEVELOPMENT. This does not claim model quality or rerun optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

TARGET = 120
CHECKPOINT = "checkpoint-000120"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


COMMIT = re.compile(r"[0-9a-f]{40}\Z")
TRAINING = Path(__file__).resolve().parents[2] / "training"
MODEL = "dotlabs/void.1"
REVISION = "9c2be5087772867d45961abd11f6b35110617563"
NATIVE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
# Frozen, native-admitted corpus #633 / split TRAIN identities, not claimant inputs.
# A different admitted curriculum requires a separately versioned campaign gate.
TRAIN_SHA256 = "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1"
COMPILER_SHA256 = "4d8559805df1ff5f3f23277a7bca98e47fa8feb53337aa79e8546d0bb89132f6"
TRAIN_IDS = frozenset((
    "greebles-service_cluster-v1-s104729",
    "greebles-service_cluster-v1-s130363",
    "greebles-switchboard_stack-v1-s104729",
    "greebles-switchboard_stack-v1-s130363",
    "pipework-two_elbow_route-v1-s104729",
    "pipework-two_elbow_route-v1-s130363",
    "pipework-tee_distribution-v1-s104729",
    "pipework-tee_distribution-v1-s130363",
    "architecture-paired_doorway_rooms-v1-s104729",
    "architecture-paired_doorway_rooms-v1-s130363",
    "architecture-stair_and_landing_hall-v1-s104729",
    "architecture-stair_and_landing_hall-v1-s130363",
))
TRAIN_SEED_ORDER = (
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
TRAIN_SEED_ORDER_SHA256 = "9209cc4ea49ed24c9bf3d2e2ba7adf421072534f695317664a0b1f8f245c68ee"

# Independently reconstructed from the original native-admitted split and the
# shared train.py seed-568 family-interleaving algorithm. The exact order/digest
# is frozen for this original 120-step contract, never read from the claimant.
CPU_ENV = {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE"}
CPU_EFFECTIVE = {
    "environment": CPU_ENV, "aten_cpu_capability": "DEFAULT",
    "intraop_threads": 1, "interop_threads": 1, "mkl_available": True,
    "mkldnn_enabled": False, "float32_matmul_precision": "highest",
    "cpu_autocast_enabled": False, "default_dtype": "torch.float32",
    "default_device": "cpu", "deterministic_algorithms": True,
    "deterministic_warn_only": False,
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))



def _source_contract(summary: dict, expected_source: str) -> list[str]:
    """Require independent pinned source, model, corpus and CPU identities.

    The synthetic test fixture intentionally has fake tensor bytes. This gate
    checks declared identity and optimizer receipts; it cannot authenticate
    training or replace the original independently qualified native evidence.
    """
    if not isinstance(expected_source, str) or not COMMIT.fullmatch(expected_source):
        raise ValueError("exact producer source SHA required")
    c = summary.get("config")
    if not isinstance(c, dict):
        raise ValueError("missing source/model/native/numerical training config")
    fixed = {
        "schema_version": 1, "profile": "void", "source_sha": expected_source,
        "native_source_sha": NATIVE, "corpus_manifest_sha256": CORPUS,
        "device": "cpu", "dtype": "float32", "threads": 1, "seed": 568,
        "max_length": 2048, "rank": 8, "alpha": 16, "dropout": 0.05,
        "learning_rate": 0.0005, "native_execution_by_trainer": False,
        "final_targets_opened": False,
    }
    for k, v in fixed.items():
        if type(c.get(k)) is not type(v) or c[k] != v:
            raise ValueError("source/model/native/numerical identity mismatch: " + k)
    profile = _read_json(TRAINING / "model_profiles.json")["profiles"]["void"]
    if c.get("model") != profile or profile.get("model_id") != MODEL or profile.get("revision") != REVISION:
        raise ValueError("pinned VOID model/profile revision mismatch")
    files = c.get("model_files")
    if not isinstance(files, dict) or not set(profile["required_hashes"]).issubset(files):
        raise ValueError("pinned model file inventory mismatch")
    # The audited downloader may also fetch optional metadata. Verify each
    # extra hash/size without misrepresenting it as a new immutable model pin.
    for name, item in files.items():
        if (not isinstance(name, str) or not isinstance(item, dict) or
                not isinstance(item.get("sha256"), str) or not HEX64.fullmatch(item["sha256"]) or
                type(item.get("bytes")) is not int or item["bytes"] <= 0):
            raise ValueError("invalid model snapshot file receipt: " + str(name))
    for name, digest in profile["required_hashes"].items():
        item = files[name]
        if (not isinstance(item, dict) or item.get("sha256") != digest or
                type(item.get("bytes")) is not int or item["bytes"] <= 0):
            raise ValueError("pinned model file bytes mismatch: " + name)
    for field, name in (("trainer_sha256", "train.py"), ("contract_sha256", "common.py")):
        if c.get(field) != _sha(TRAINING / name):
            raise ValueError("source-matched trainer/contract mismatch: " + field)
    for field, pinned in (("train_sha256", TRAIN_SHA256),
                          ("compiler_sha256", COMPILER_SHA256)):
        if c.get(field) != pinned:
            raise ValueError("frozen native TRAIN/compiler digest mismatch: " + field)
    cpu = c.get("numerics")
    if (not isinstance(cpu, dict) or cpu.get("profile") != "portable-cpu-fp32-v1" or
            cpu.get("effective") != CPU_EFFECTIVE or
            cpu.get("numerics_sha256") != _sha(TRAINING / "numerics.py") or
            cpu.get("torch_version") != "2.15.0+cpu" or
            not isinstance(cpu.get("torch_build_sha256"), str) or
            not HEX64.fullmatch(cpu["torch_build_sha256"])):
        raise ValueError("pinned portable-cpu-fp32-v1 numerical identity mismatch")
    order = c.get("sample_order")
    if (not isinstance(order, list) or len(order) != len(TRAIN_IDS) or
            any(not isinstance(x, str) for x in order) or
            len(set(order)) != len(TRAIN_IDS) or set(order) != TRAIN_IDS):
        raise ValueError("sample_order differs from native-admitted frozen TRAIN rows")
    if order != list(TRAIN_SEED_ORDER):
        raise ValueError("sample_order differs from frozen native seed568 TRAIN sampler order")
    canonical_order = json.dumps(order, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")
    if hashlib.sha256(canonical_order).hexdigest() != TRAIN_SEED_ORDER_SHA256:
        raise ValueError("frozen native TRAIN sampler order digest mismatch")
    return order


def _progress_guard(summary: dict, checkpoint: Path, history: list[dict], order: list[str]) -> None:
    progress = checkpoint.parent / "progress.jsonl"
    if not progress.is_file() or progress.is_symlink():
        raise ValueError("missing original optimizer progress.jsonl")
    lines = progress.read_text(encoding="utf-8").splitlines()
    if len(lines) != TARGET:
        raise ValueError("progress.jsonl must contain exactly 120 original optimizer events")
    loss_tokens = sequence_tokens = written_bytes = 0
    for i, (line, event) in enumerate(zip(lines, history), 1):
        try:
            observed = json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        except (ValueError, TypeError) as exc:
            raise ValueError("malformed original progress.jsonl event") from exc
        if observed != event:
            raise ValueError("original progress.jsonl differs from checkpoint history")
        canonical = json.dumps(event, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
        if line != canonical:
            raise ValueError("original optimizer event bytes are not canonical training output")
        for k in ("loss", "seconds", "gradient_norm", "sequence_tokens",
                  "loss_tokens", "prompt_bytes", "completion_bytes"):
            if type(event.get(k)) not in (int, float) or not math.isfinite(event[k]):
                raise ValueError("missing finite optimizer loss/tokens/gradient: " + k)
        for k in ("sequence_tokens", "loss_tokens", "prompt_bytes", "completion_bytes"):
            if type(event[k]) is not int or event[k] <= 0:
                raise ValueError("missing positive supervised-token/byte count: " + k)
        if (event["loss"] < 0 or event["seconds"] <= 0 or event["gradient_norm"] <= 0 or
                event["sequence_tokens"] < event["loss_tokens"] or
                event["sample_id"] != order[(i - 1) % len(order)]):
            raise ValueError("invalid optimizer loss/sample order or supervised-token evidence")
        loss_tokens += event["loss_tokens"]
        sequence_tokens += event["sequence_tokens"]
        written_bytes += event["prompt_bytes"] + event["completion_bytes"]
    aggregates = {
        "loss_tokens_this_job": loss_tokens, "sequence_tokens_this_job": sequence_tokens,
        "bytes_this_job": written_bytes, "first_loss": history[0]["loss"],
        "last_loss": history[-1]["loss"],
    }
    for k, v in aggregates.items():
        if type(summary.get(k)) is not type(v) or summary[k] != v:
            raise ValueError("summary disagrees with complete original optimizer log: " + k)


def _checkpoint_guard(summary: dict, checkpoint: Path, order: list[str]) -> dict:
    if checkpoint.name != CHECKPOINT or checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("required complete checkpoint directory missing or symlinked")
    required_paths = {"inventory.json", "receipt.json", "training_state.pt",
                      "adapter/adapter_model.safetensors"}
    if not all((checkpoint / f).is_file() for f in required_paths):
        raise ValueError("complete checkpoint files missing")
    if not isinstance(summary.get("checkpoint_inventory_sha256"), str) or not HEX64.fullmatch(summary["checkpoint_inventory_sha256"]):
        raise ValueError("summary checkpoint inventory hash missing or invalid")
    if _sha(checkpoint / "inventory.json") != summary["checkpoint_inventory_sha256"]:
        raise ValueError("checkpoint inventory differs from completed training summary")
    inventory = _read_json(checkpoint / "inventory.json")
    if not isinstance(inventory, dict) or not required_paths.difference({"inventory.json"}).issubset(inventory):
        raise ValueError("checkpoint inventory is incomplete")
    actual = set()
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in checkpoint inventory")
        if path.is_file():
            actual.add(path.relative_to(checkpoint).as_posix())
    if actual != set(inventory) | {"inventory.json"}:
        raise ValueError("checkpoint files and inventory disagree")
    for name, entry in inventory.items():
        parsed = PurePosixPath(name)
        if (not isinstance(name, str) or parsed.is_absolute() or
                not parsed.parts or any(p in (".", "..") for p in parsed.parts) or
                "\\" in name or parsed.as_posix() != name):
            raise ValueError("unsafe checkpoint inventory entry")
        path = checkpoint.joinpath(*parsed.parts)
        if not path.is_file() or path.is_symlink() or not isinstance(entry, dict):
            raise ValueError("checkpoint inventory entry is not a regular file")
        if (type(entry.get("bytes")) is not int or entry["bytes"] != path.stat().st_size or
                not isinstance(entry.get("sha256"), str) or not HEX64.fullmatch(entry["sha256"]) or
                _sha(path) != entry["sha256"]):
            raise ValueError("checkpoint content hash or length mismatch: " + name)
    receipt = _read_json(checkpoint / "receipt.json")
    if (not isinstance(receipt, dict) or type(receipt.get("global_step")) is not int or
            receipt["global_step"] != TARGET or type(receipt.get("sample_cursor")) is not int or
            receipt["sample_cursor"] != TARGET or receipt.get("config") != summary.get("config") or
            receipt.get("frozen_base_sha256") != summary.get("frozen_base_sha256")):
        raise ValueError("checkpoint step/cursor/config/base receipt differs from summary")
    history = receipt.get("history")
    if not isinstance(history, list) or len(history) != TARGET:
        raise ValueError("checkpoint history does not contain 120 optimizer receipts")
    if not isinstance(summary.get("sample_ids_this_job"), list) or len(summary["sample_ids_this_job"]) != TARGET:
        raise ValueError("summary sample IDs do not contain 120 training updates")
    for index, event in enumerate(history, 1):
        if (not isinstance(event, dict) or type(event.get("global_step")) is not int or
                event["global_step"] != index or type(event.get("sample_cursor")) is not int or
                event["sample_cursor"] != index or event.get("sample_id") != summary["sample_ids_this_job"][index - 1] or
                not isinstance(event.get("gradient_norm"), (float, int)) or
                isinstance(event["gradient_norm"], bool) or not math.isfinite(event["gradient_norm"]) or
                event["gradient_norm"] <= 0):
            raise ValueError("optimizer history step/cursor/sample/gradient mismatch")
    if receipt.get("numerical_runtime") != summary.get("numerical_runtime"):
        raise ValueError("numerical runtime receipt differs from summary")
    _progress_guard(summary, checkpoint, history, order)
    return {"checkpoint_inventory_sha256": summary["checkpoint_inventory_sha256"],
            "verified_optimizer_receipts": len(history), "verified_checkpoint_files": len(inventory)}


def verify_completion(summary: dict, exit_code: int, checkpoint: Path, *, source_sha: str) -> dict:
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError("trainer exit must be zero; partial/failure is non-promotable")
    if not isinstance(summary, dict):
        raise ValueError("training summary must be a JSON object")
    required = {
        "schema_version": 1, "status": "completed", "global_step": TARGET,
        "optimizer_steps_this_job": TARGET, "sample_cursor": TARGET,
        "checkpoint": CHECKPOINT, "actual_training": True,
        "checkpoint_readback_exact": True, "frozen_base_unchanged": True,
        "adapter_changed": True, "resume": None, "native_execution": False,
        "model_quality_claim": False,
    }
    for key, expected in required.items():
        value = summary.get(key)
        if key not in summary or type(value) is not type(expected) or value != expected:
            raise ValueError(f"120-step qualification mismatch: {key}={value!r}")
    if (not isinstance(summary.get("adapter_before_sha256"), str) or
            not isinstance(summary.get("adapter_after_sha256"), str) or
            not HEX64.fullmatch(summary["adapter_before_sha256"]) or
            not HEX64.fullmatch(summary["adapter_after_sha256"]) or
            summary["adapter_before_sha256"] == summary["adapter_after_sha256"]):
        raise ValueError("adapter change receipt is absent or inconsistent")
    order = _source_contract(summary, source_sha)
    verified = _checkpoint_guard(summary, checkpoint, order)
    return {"status": summary["status"], "actual_steps": summary["global_step"],
            "checkpoint": summary["checkpoint"], "wall_seconds": summary.get("wall_seconds"),
            "peak_rss_kib": summary.get("peak_rss_kib"), "promotable": True, **verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    summary = _read_json(args.summary)
    print(json.dumps(verify_completion(summary, args.exit_code, args.checkpoint, source_sha=args.source_sha), sort_keys=True))


if __name__ == "__main__":
    main()
