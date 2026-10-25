#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from village_001_manifest import build_manifest
from village_001_validate import validate_file

ROOT = Path(__file__).resolve().parents[1]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(item)))
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def member_contract(cfg: dict, member_id: str) -> tuple[dict, dict]:
    manifest = build_manifest(cfg)
    for member in manifest["members"]:
        if member["member_id"] == member_id:
            return manifest, member
    raise ValueError(f"unknown member {member_id}")


def choose_lora_targets(model) -> list[str]:
    desired = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    present = set()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in desired and isinstance(module, torch.nn.Linear):
            present.add(suffix)
    targets = [name for name in desired if name in present]
    if not targets:
        raise RuntimeError("no LoRA target modules found")
    return targets


def load_rows(cfg: dict, member: dict, contract_sha: str) -> tuple[Path, list[dict]]:
    dataset_root = ROOT / cfg["archive"]["dataset_root"]
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("VILLAGE-001B durable dataset manifest is missing")
    dataset_manifest = json.loads(manifest_path.read_text())
    if dataset_manifest.get("stage") != "VILLAGE-001B-member-corpora":
        raise RuntimeError(
            f"unexpected dataset stage {dataset_manifest.get('stage')!r}"
        )
    if dataset_manifest.get("member_contract_sha256") != contract_sha:
        raise RuntimeError("dataset/member population contract mismatch")

    path = dataset_root / "member-corpora" / f"{member['member_id']}.jsonl"
    validate_file(
        path,
        "corpus",
        member["member_id"],
        int(cfg["corpus"]["target_sft_examples_per_member"]),
    )
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    expected_ids = [
        f"{member['member_id']}-E{i:03d}"
        for i in range(1, int(cfg["corpus"]["target_sft_examples_per_member"]) + 1)
    ]
    if [row["example_id"] for row in rows] != expected_ids:
        raise RuntimeError(f"{member['member_id']}: noncanonical corpus ID order")
    if any(row["lineage"] != member["lineage"] for row in rows):
        raise RuntimeError(f"{member['member_id']}: corpus lineage mismatch")
    if any(row["sibling"] != member["sibling"] for row in rows):
        raise RuntimeError(f"{member['member_id']}: corpus sibling mismatch")
    return path, rows


def encode_example(tok, system: str, row: dict, max_length: int) -> dict:
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["input"]},
    ]
    full = prompt + [{"role": "assistant", "content": row["target"]}]
    prompt_ids = tok.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tok.apply_chat_template(
        full,
        tokenize=True,
        add_generation_prompt=False,
    )
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    if len(full_ids) > max_length:
        raise RuntimeError(
            f"{row['example_id']}: tokenized length {len(full_ids)} > {max_length}"
        )
    if common >= len(full_ids):
        raise RuntimeError(f"{row['example_id']}: no supervised assistant tokens")
    labels = [-100] * common + full_ids[common:]
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "sequence_tokens": len(full_ids),
        "supervised_tokens": len(full_ids) - common,
    }


def schedule(n: int, epochs: int, seed: int) -> list[int]:
    order = []
    for epoch in range(epochs):
        indices = list(range(n))
        random.Random(seed + epoch * 100003).shuffle(indices)
        order.extend(indices)
    return order


@torch.inference_mode()
def evaluate(model, tok, system: str, scenarios: list[dict], max_new_tokens: int) -> list[dict]:
    old_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    model.eval()
    rows = []
    for item in scenarios:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": item["prompt"]},
        ]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        started = time.monotonic()
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        elapsed = time.monotonic() - started
        generated = out[0, ids.shape[-1] :]
        response = tok.decode(generated, skip_special_tokens=True).strip()
        rows.append(
            {
                "scenario_id": item["id"],
                "family": item["family"],
                "prompt": item["prompt"],
                "response": response,
                "generated_tokens": int(generated.numel()),
                "elapsed_seconds": elapsed,
                "tokens_per_second": (
                    int(generated.numel()) / elapsed if elapsed else None
                ),
            }
        )
    model.config.use_cache = old_cache
    return rows


def save_checkpoint(
    *,
    root: Path,
    model,
    optimizer,
    next_step: int,
    losses: list[float],
    sequence_tokens: int,
    supervised_tokens: int,
    training_seconds: float,
    source_run_id: str | None,
) -> None:
    tmp = root.with_name(root.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    adapter = tmp / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save(torch.get_rng_state(), tmp / "torch-rng.pt")
    write_json(
        tmp / "progress.json",
        {
            "schema_version": 1,
            "next_step": next_step,
            "losses": losses,
            "sequence_tokens": sequence_tokens,
            "supervised_tokens": supervised_tokens,
            "training_seconds": training_seconds,
            "source_run_id": source_run_id,
        },
    )
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)


def load_checkpoint(root: Path) -> dict | None:
    progress = root / "progress.json"
    if not progress.is_file():
        return None
    required = [
        root / "adapter" / "adapter_config.json",
        root / "optimizer.pt",
        root / "torch-rng.pt",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError("partial checkpoint is internally incomplete")
    return json.loads(progress.read_text())


def copy_resume(resume: Path, out: Path) -> None:
    if not resume.is_dir():
        raise FileNotFoundError(resume)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(resume, out)


def validate_complete_artifact(out: Path, member_id: str, corpus_sha: str) -> bool:
    result_path = out / "result.json"
    adapter = out / "adapter"
    if not result_path.is_file() or not adapter.is_dir():
        return False
    result = json.loads(result_path.read_text())
    if result.get("experiment") != "VILLAGE-001C":
        raise RuntimeError("resume result is not VILLAGE-001C")
    if result.get("member_id") != member_id:
        raise RuntimeError("resume result member mismatch")
    if result.get("training_dataset_sha256") != corpus_sha:
        raise RuntimeError("resume result corpus hash mismatch")
    recorded = result.get("adapter") or {}
    if recorded.get("sha256") != directory_hash(adapter):
        raise RuntimeError("resume adapter hash mismatch")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/config.json",
    )
    parser.add_argument("--member", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=40)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    population, member = member_contract(cfg, args.member)
    contract_sha = population["member_contract_sha256"]
    train_cfg = cfg["sft_training"]
    base_cfg = cfg["base_model"]

    eval_cfg_path = ROOT / cfg["corpus"]["common_heldout_config"]
    eval_doc = json.loads(eval_cfg_path.read_text())
    eval_cfg = eval_doc["eval"]
    system = eval_cfg["system"]

    train_path, train_rows = load_rows(cfg, member, contract_sha)
    corpus_sha = sha256_file(train_path)

    out = args.out
    if args.resume:
        copy_resume(args.resume, out)
    else:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

    if validate_complete_artifact(out, member["member_id"], corpus_sha):
        print(
            json.dumps(
                {
                    "member": member["member_id"],
                    "resume": "complete_artifact_reused",
                    "training_dataset_sha256": corpus_sha,
                },
                indent=2,
            )
        )
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    seed = int(member["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    load_started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(
        base_cfg["id"],
        revision=base_cfg["revision"],
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_cfg["id"],
        revision=base_cfg["revision"],
        torch_dtype=torch.float32,
    )
    base.config.use_cache = False
    total_parameters = sum(p.numel() for p in base.parameters())

    checkpoint_dir = out / "checkpoint"
    checkpoint = load_checkpoint(checkpoint_dir)
    if checkpoint:
        model = PeftModel.from_pretrained(
            base,
            checkpoint_dir / "adapter",
            is_trainable=True,
        )
        targets = list(getattr(model.peft_config["default"], "target_modules", []) or [])
    else:
        targets = choose_lora_targets(base)
        model = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=int(train_cfg["lora_r"]),
                lora_alpha=int(train_cfg["lora_alpha"]),
                lora_dropout=float(train_cfg["lora_dropout"]),
                bias="none",
                target_modules=targets,
            ),
        )

    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    encoded = [
        encode_example(
            tok,
            system,
            row,
            int(train_cfg["max_length"]),
        )
        for row in train_rows
    ]
    order = schedule(
        len(encoded),
        int(train_cfg["epochs"]),
        seed,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    start_step = 0
    losses: list[float] = []
    sequence_tokens = 0
    supervised_tokens = 0
    training_seconds = 0.0
    resumed_from_run_id = None

    if checkpoint:
        start_step = int(checkpoint["next_step"])
        losses = [float(x) for x in checkpoint["losses"]]
        sequence_tokens = int(checkpoint["sequence_tokens"])
        supervised_tokens = int(checkpoint["supervised_tokens"])
        training_seconds = float(checkpoint["training_seconds"])
        resumed_from_run_id = checkpoint.get("source_run_id")
        if start_step < 0 or start_step > len(order):
            raise RuntimeError(f"invalid checkpoint next_step {start_step}")
        if len(losses) != start_step:
            raise RuntimeError("checkpoint loss count does not match next_step")
        optimizer.load_state_dict(
            torch.load(
                checkpoint_dir / "optimizer.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
        rng = torch.load(
            checkpoint_dir / "torch-rng.pt",
            map_location="cpu",
            weights_only=False,
        )
        torch.set_rng_state(rng)

    load_seconds = time.monotonic() - load_started
    model.train()
    segment_started = time.monotonic()
    checkpoint_interval = max(1, int(args.checkpoint_interval))

    for step in range(start_step, len(order)):
        ex = encoded[order[step]]
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=ex["input_ids"],
            attention_mask=ex["attention_mask"],
            labels=ex["labels"],
        )
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(train_cfg["gradient_clip"]),
        )
        optimizer.step()
        losses.append(float(loss.detach()))
        sequence_tokens += ex["sequence_tokens"]
        supervised_tokens += ex["supervised_tokens"]

        next_step = step + 1
        if next_step % checkpoint_interval == 0 or next_step == len(order):
            training_seconds += time.monotonic() - segment_started
            save_checkpoint(
                root=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                next_step=next_step,
                losses=losses,
                sequence_tokens=sequence_tokens,
                supervised_tokens=supervised_tokens,
                training_seconds=training_seconds,
                source_run_id=os.environ.get("GITHUB_RUN_ID"),
            )
            segment_started = time.monotonic()

    final_adapter = out / "adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    model.save_pretrained(final_adapter, safe_serialization=True)
    adapter_info = {
        "path": "adapter",
        "bytes": sum(
            path.stat().st_size
            for path in final_adapter.rglob("*")
            if path.is_file()
        ),
        "sha256": directory_hash(final_adapter),
    }

    responses = evaluate(
        model,
        tok,
        system,
        eval_cfg["scenarios"],
        int(eval_cfg["max_new_tokens"]),
    )
    if len(responses) != len(eval_cfg["scenarios"]):
        raise RuntimeError("held-out response count mismatch")

    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001C",
        "issue": 12,
        "member_id": member["member_id"],
        "lineage": member["lineage"],
        "sibling": member["sibling"],
        "specialization": member["specialization"],
        "member_seed": seed,
        "member_contract_sha256": contract_sha,
        "base_model": base_cfg["id"],
        "base_revision": base_cfg["revision"],
        "training_config": train_cfg,
        "training_dataset": str(train_path.relative_to(ROOT)),
        "training_dataset_sha256": corpus_sha,
        "training_example_ids": [row["example_id"] for row in train_rows],
        "training_source_kind_counts": dict(
            Counter(row["source_kind"] for row in train_rows)
        ),
        "runner": {
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
        "packages": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "load_seconds": load_seconds,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": (
            trainable_parameters / total_parameters if total_parameters else 0
        ),
        "lora_targets": sorted(str(x) for x in targets),
        "training": {
            "epochs": int(train_cfg["epochs"]),
            "optimizer_steps": len(losses),
            "schedule_steps": len(order),
            "sequence_tokens": sequence_tokens,
            "supervised_tokens": supervised_tokens,
            "seconds": training_seconds,
            "sequence_tokens_per_second": (
                sequence_tokens / training_seconds if training_seconds else None
            ),
            "supervised_tokens_per_second": (
                supervised_tokens / training_seconds if training_seconds else None
            ),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "losses": losses,
        },
        "recovery": {
            "checkpoint_interval_steps": checkpoint_interval,
            "resumed": bool(checkpoint),
            "resumed_from_step": start_step,
            "resumed_from_run_id": resumed_from_run_id,
            "final_checkpoint_step": len(order),
        },
        "peak_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "adapter": adapter_info,
        "evaluation": {
            "system": system,
            "hidden_member_prompt_used": False,
            "max_new_tokens": int(eval_cfg["max_new_tokens"]),
            "responses": responses,
        },
    }
    write_json(out / "result.json", result)

    # Completed artifacts are self-sufficient; partial checkpoints are retained only
    # when a lane exits before result.json is written.
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    print(
        json.dumps(
            {
                "member": member["member_id"],
                "cpu": result["runner"]["cpu_model"],
                "steps": result["training"]["optimizer_steps"],
                "training_seconds": training_seconds,
                "sequence_tokens_per_second": result["training"][
                    "sequence_tokens_per_second"
                ],
                "loss": [
                    result["training"]["first_loss"],
                    result["training"]["last_loss"],
                ],
                "adapter": adapter_info,
                "held_out_responses": len(responses),
                "resumed": result["recovery"]["resumed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
