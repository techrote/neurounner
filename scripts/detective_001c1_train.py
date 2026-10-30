#!/usr/bin/env python3
"""DETECTIVE-001C1: continue one immutable Village LoRA for 64 detective rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import time
from pathlib import Path

import peft
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from detective_001c1_dataset import validate_existing

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-001/training-c1.json"


class TrainError(ValueError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = p.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha_file(p)))
    return h.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               sort_keys=True) + "\n", encoding="utf-8")


def load_cfg(member_id: str, dataset_root: Path, parent: Path) -> tuple[dict, dict, list[dict]]:
    cfg = json.loads(CONFIG.read_text())
    if member_id not in cfg["members"]:
        raise TrainError("member is not admitted for DETECTIVE-001C1")
    meta = cfg["members"][member_id]
    train = cfg["training"]
    if train != {
        "epochs": 1, "learning_rate": 0.00015, "max_length": 512,
        "gradient_clip": 1.0, "max_threads": 4,
        "max_minutes_per_member": 30, "optimizer_steps_per_member": 64,
        "paid_api_calls": 0, "gpu_required": False,
    }:
        raise TrainError("training budget drift")
    manifest = validate_existing(dataset_root)
    entry = manifest["member_files"][member_id]
    data_path = dataset_root / entry["path"]
    if sha_file(data_path) != entry["sha256"]:
        raise TrainError("training shard hash mismatch")
    rows = [json.loads(x) for x in data_path.read_text().splitlines() if x.strip()]
    if len(rows) != 64 or len({x["example_id"] for x in rows}) != 64:
        raise TrainError("training shard cardinality mismatch")
    if directory_hash(parent) != meta["parent_directory_sha256"]:
        raise TrainError("parent adapter directory hash mismatch")
    return cfg, manifest, rows


def encode(tok, system: str, row: dict, max_length: int) -> dict:
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["input"]},
    ]
    full = prompt + [{"role": "assistant", "content": row["target"]}]
    prompt_ids = tok.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False)
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    if len(full_ids) > max_length:
        raise TrainError(
            f"{row['example_id']}: {len(full_ids)} > max length {max_length}")
    if common >= len(full_ids):
        raise TrainError(f"{row['example_id']}: no assistant supervision")
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor(
            [[-100] * common + full_ids[common:]], dtype=torch.long),
        "sequence_tokens": len(full_ids),
        "supervised_tokens": len(full_ids) - common,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg, manifest, rows = load_cfg(
        args.member, args.dataset_root, args.parent_adapter)
    meta = cfg["members"][args.member]
    train = cfg["training"]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(train["max_threads"], os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(meta["seed"])
    torch.manual_seed(meta["seed"])

    weight_path = Path(hf_hub_download(
        cfg["base_model"]["id"], "model.safetensors",
        revision=cfg["base_model"]["revision"]))
    if sha_file(weight_path) != cfg["base_model"]["weights_sha256"]:
        raise TrainError("base model weight SHA mismatch")

    load_start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(
        cfg["base_model"]["id"], revision=cfg["base_model"]["revision"],
        trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["id"], revision=cfg["base_model"]["revision"],
        torch_dtype=torch.float32, trust_remote_code=False)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, args.parent_adapter, is_trainable=True)
    model.config.use_cache = False
    load_seconds = time.monotonic() - load_start

    # Existing Village adapter, not a newly added nested LoRA.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable <= 0 or trainable > 3_000_000:
        raise TrainError(f"unexpected trainable parameter count {trainable}")
    if any(p.requires_grad for p in base.model.parameters()
           if "lora_" not in getattr(p, "_name", "")):
        # PEFT parameter naming is not retained on Parameter itself; the
        # authoritative check below uses named parameters.
        pass
    offenders = [
        name for name, p in model.named_parameters()
        if p.requires_grad and "lora_" not in name
    ]
    if offenders:
        raise TrainError("non-LoRA parameters became trainable: " +
                         ",".join(offenders[:8]))

    encoded = [encode(tok, cfg["system"], row, train["max_length"])
               for row in rows]
    order = list(range(len(rows)))
    random.Random(meta["seed"]).shuffle(order)
    if len(order) != train["optimizer_steps_per_member"]:
        raise TrainError("optimizer schedule/cardinality mismatch")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train["learning_rate"])
    model.train()
    losses = []
    seq_tokens = 0
    supervised_tokens = 0
    started = time.monotonic()
    for step, idx in enumerate(order, 1):
        batch = encoded[idx]
        optimizer.zero_grad(set_to_none=True)
        result = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"])
        loss = result.loss
        if not torch.isfinite(loss):
            raise TrainError(f"nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            train["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach()))
        seq_tokens += batch["sequence_tokens"]
        supervised_tokens += batch["supervised_tokens"]
    train_seconds = time.monotonic() - started

    if len(losses) != 64:
        raise TrainError("incomplete training")
    args.out.mkdir(parents=True, exist_ok=True)
    adapter = args.out / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    trained_hash = directory_hash(adapter)
    parent_hash = directory_hash(args.parent_adapter)
    result = {
        "schema": "detective-witness-training-result-v1",
        "experiment": "DETECTIVE-001C1",
        "issue": 111,
        "member_id": args.member,
        "policy_id": meta["policy_id"],
        "base_model": cfg["base_model"]["id"],
        "base_revision": cfg["base_model"]["revision"],
        "base_weight_sha256": cfg["base_model"]["weights_sha256"],
        "parent_release_tag": cfg["parent_release"]["tag"],
        "parent_archive_sha256": meta["parent_archive_sha256"],
        "parent_directory_sha256": parent_hash,
        "dataset_manifest_sha256": sha_file(
            args.dataset_root / "manifest.json"),
        "training_shard_sha256": manifest["member_files"][args.member]["sha256"],
        "training_example_ids": [x["example_id"] for x in rows],
        "training": {
            "optimizer_steps": len(losses),
            "epochs": train["epochs"],
            "learning_rate": train["learning_rate"],
            "max_length": train["max_length"],
            "gradient_clip": train["gradient_clip"],
            "sequence_tokens": seq_tokens,
            "supervised_tokens": supervised_tokens,
            "seconds": train_seconds,
            "sequence_tokens_per_second":
                seq_tokens / train_seconds if train_seconds else None,
            "supervised_tokens_per_second":
                supervised_tokens / train_seconds if train_seconds else None,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses),
            "losses": losses,
        },
        "model": {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_fraction": trainable / total,
            "only_lora_parameters_trainable": True,
        },
        "adapter": {
            "path": "adapter",
            "directory_sha256": trained_hash,
            "bytes": sum(p.stat().st_size for p in adapter.rglob("*")
                         if p.is_file()),
        },
        "runtime": {
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "load_seconds": load_seconds,
        },
    }
    write_json(args.out / "result.json", result)
    print(json.dumps({
        "member": args.member,
        "steps": len(losses),
        "loss": [losses[0], losses[-1]],
        "parent_sha256": parent_hash,
        "trained_sha256": trained_hash,
        "seconds": train_seconds,
        "peak_rss_kib": result["runtime"]["peak_rss_kib"],
    }, indent=2))


if __name__ == "__main__":
    main()
