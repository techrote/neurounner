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
import traceback
from pathlib import Path
from typing import Any

import peft
import psutil
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
BASE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
DATASET = ROOT / "experiments" / "train-001" / "kernel_compiler_dataset.json"
SEED = 20261024
MAX_LENGTH = 256
LR_FULL = 5e-5
LR_LORA = 8e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MEMORY_GUARD_BEFORE_FIRST_STEP_GIB = 3.0
MEMORY_GUARD_AFTER_FIRST_STEP_GIB = 2.0
MEMORY_GUARD_DURING_TRAIN_GIB = 1.5


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def rss_kib() -> int:
    return psutil.Process().memory_info().rss // 1024


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def memory_snapshot() -> dict[str, int]:
    vm = psutil.virtual_memory()
    return {
        "rss_kib": rss_kib(),
        "peak_rss_kib": peak_rss_kib(),
        "system_total_kib": int(vm.total // 1024),
        "system_available_kib": int(vm.available // 1024),
    }


def guard_memory(stage: str, minimum_available_gib: float) -> None:
    available = psutil.virtual_memory().available
    floor = int(minimum_available_gib * 1024**3)
    if available < floor:
        raise MemoryError(
            f"TRAIN-004 memory guard at {stage}: available={available / 1024**3:.2f} GiB "
            f"is below safety floor {minimum_available_gib:.2f} GiB"
        )


def package_versions() -> dict[str, str]:
    import huggingface_hub
    import safetensors

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "psutil": psutil.__version__,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_sha256() -> str:
    return sha256_file(DATASET)


def source_commit() -> str | None:
    return os.environ.get("TRAIN004_SOURCE_SHA") or os.environ.get("GITHUB_SHA")


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(SEED)
    torch.manual_seed(SEED)


def compact_target(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def tokenize_example(tokenizer, system: str, item: dict[str, Any]) -> dict[str, Any]:
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": item["input"]},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": compact_target(item["output"])}
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False
    )
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    if len(full_ids) > MAX_LENGTH:
        raise RuntimeError(
            f"example length {len(full_ids)} exceeds max_length={MAX_LENGTH}"
        )
    labels = [-100] * common + full_ids[common:]
    supervised = sum(x != -100 for x in labels)
    if supervised == 0:
        raise RuntimeError("assistant target was entirely masked")
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "sequence_tokens": len(full_ids),
        "supervised_tokens": supervised,
    }


def choose_lora_targets(model) -> list[str]:
    desired = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    present = set()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in desired and isinstance(module, torch.nn.Linear):
            present.add(suffix)
    targets = [name for name in desired if name in present]
    if not targets:
        raise RuntimeError("could not find expected LoRA target modules")
    return targets


def build_train_model(mode: str):
    before_tokenizer = memory_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    before_model = memory_snapshot()
    t0 = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        torch_dtype=torch.float32,
    )
    load_seconds = time.monotonic() - t0
    base.config.use_cache = False
    after_model = memory_snapshot()

    lora_targets: list[str] = []
    if mode == "lora":
        lora_targets = choose_lora_targets(base)
        cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=lora_targets,
        )
        model = get_peft_model(base, cfg)
    elif mode == "full":
        model = base
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(mode)

    return (
        model,
        tokenizer,
        lora_targets,
        {
            "before_tokenizer_load": before_tokenizer,
            "before_model_load": before_model,
            "after_model_load": after_model,
        },
        load_seconds,
    )


def optimizer_for(model, mode: str) -> torch.optim.Optimizer:
    lr = LR_FULL if mode == "full" else LR_LORA
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def optimizer_config(mode: str) -> dict[str, Any]:
    return {
        "name": "AdamW",
        "learning_rate": LR_FULL if mode == "full" else LR_LORA,
        "betas": list(BETAS),
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP,
        "full_parameter_lr_rationale": (
            "5e-5 is a conservative single-point full-parameter fine-tuning rate for "
            "the pinned 135M causal LM; TRAIN-004 intentionally does not sweep it."
            if mode == "full"
            else None
        ),
    }


def checkpoint_metadata(
    mode: str,
    global_step: int,
    sample_cursor: int,
    lora_targets: list[str],
) -> dict[str, Any]:
    return {
        "experiment": "TRAIN-004",
        "mode": mode,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_commit": source_commit(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_sha256(),
        "global_step": global_step,
        "sample_cursor": sample_cursor,
        "seed": SEED,
        "dtype": "float32",
        "optimizer": optimizer_config(mode),
        "lora": (
            {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "targets": lora_targets,
            }
            if mode == "lora"
            else None
        ),
        "packages": package_versions(),
    }


def save_checkpoint(
    outdir: Path,
    model,
    optimizer: torch.optim.Optimizer,
    mode: str,
    global_step: int,
    sample_cursor: int,
    lora_targets: list[str],
) -> dict[str, Any]:
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_started = time.monotonic()
    model_dir = outdir / ("adapter" if mode == "lora" else "model")
    model.save_pretrained(model_dir, safe_serialization=True)

    state = {
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "sample_cursor": sample_cursor,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    torch.save(state, outdir / "train_state.pt")
    (outdir / "metadata.json").write_text(
        json.dumps(
            checkpoint_metadata(mode, global_step, sample_cursor, lora_targets),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    files = []
    for path in sorted(p for p in outdir.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(outdir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "experiment": "TRAIN-004",
        "mode": mode,
        "source_commit": source_commit(),
        "files": files,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (outdir / "manifest.json").write_bytes(manifest_bytes)
    manifest_sha = sha256_bytes(manifest_bytes)
    total_bytes = sum(p.stat().st_size for p in outdir.rglob("*") if p.is_file())
    save_seconds = time.monotonic() - save_started
    by_path = {row["path"]: row for row in files}

    return {
        "bytes": total_bytes,
        "save_seconds": save_seconds,
        "manifest_sha256": manifest_sha,
        "files": files,
        "model_payload_bytes": sum(
            row["bytes"]
            for path, row in by_path.items()
            if path.startswith("model/") or path.startswith("adapter/")
        ),
        "train_state_bytes": by_path["train_state.pt"]["bytes"],
    }


def verify_checkpoint(
    checkpoint: Path, expected_manifest_sha256: str | None = None
) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise RuntimeError(
            "checkpoint manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha}"
        )
    manifest = json.loads(manifest_bytes)
    verified = []
    for row in manifest["files"]:
        path = checkpoint / row["path"]
        if not path.is_file():
            raise RuntimeError(f"checkpoint file missing: {row['path']}")
        actual_size = path.stat().st_size
        if actual_size != row["bytes"]:
            raise RuntimeError(
                f"checkpoint size mismatch for {row['path']}: "
                f"expected {row['bytes']}, got {actual_size}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != row["sha256"]:
            raise RuntimeError(
                f"checkpoint hash mismatch for {row['path']}: "
                f"expected {row['sha256']}, got {actual_sha}"
            )
        verified.append(row["path"])
    return {
        "manifest_sha256": manifest_sha,
        "file_count": len(verified),
        "verified_files": verified,
    }


def train_steps(
    model,
    optimizer: torch.optim.Optimizer,
    encoded: list[dict[str, Any]],
    steps: int,
    global_step: int,
    sample_cursor: int,
    *,
    full_memory_guard: bool,
) -> dict[str, Any]:
    losses: list[float] = []
    step_history: list[dict[str, Any]] = []
    sequence_tokens = 0
    supervised_tokens = 0
    first_step_memory = None

    if full_memory_guard:
        guard_memory("before_first_optimizer_step", MEMORY_GUARD_BEFORE_FIRST_STEP_GIB)

    model.train()
    started = time.monotonic()
    for local_index in range(steps):
        ex = encoded[sample_cursor % len(encoded)]
        sample_cursor += 1
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=ex["input_ids"],
            attention_mask=ex["attention_mask"],
            labels=ex["labels"],
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
        )
        optimizer.step()
        global_step += 1

        value = float(loss.detach())
        losses.append(value)
        sequence_tokens += ex["sequence_tokens"]
        supervised_tokens += ex["supervised_tokens"]
        snap = memory_snapshot()
        step_history.append(
            {
                "local_step": local_index + 1,
                "global_step": global_step,
                "sample_index": (sample_cursor - 1) % len(encoded),
                "sequence_tokens": ex["sequence_tokens"],
                "supervised_tokens": ex["supervised_tokens"],
                "loss": value,
                "step_seconds": time.monotonic() - step_started,
                "rss_kib": snap["rss_kib"],
                "peak_rss_kib": snap["peak_rss_kib"],
                "system_available_kib": snap["system_available_kib"],
            }
        )

        if local_index == 0:
            first_step_memory = snap
            if full_memory_guard:
                guard_memory(
                    "after_first_optimizer_step",
                    MEMORY_GUARD_AFTER_FIRST_STEP_GIB,
                )
        elif full_memory_guard:
            guard_memory("during_training", MEMORY_GUARD_DURING_TRAIN_GIB)

    train_seconds = time.monotonic() - started
    final_memory = memory_snapshot()
    first_peak = (
        first_step_memory["peak_rss_kib"] if first_step_memory is not None else None
    )
    final_peak = final_memory["peak_rss_kib"]

    return {
        "steps_this_run": steps,
        "global_step_end": global_step,
        "sample_cursor_end": sample_cursor,
        "sequence_tokens": sequence_tokens,
        "supervised_tokens": supervised_tokens,
        "train_seconds": train_seconds,
        "sequence_tokens_per_second": (
            sequence_tokens / train_seconds if train_seconds else None
        ),
        "supervised_tokens_per_second": (
            supervised_tokens / train_seconds if train_seconds else None
        ),
        "seconds_per_step": train_seconds / steps if steps else None,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "loss_history": losses,
        "step_history": step_history,
        "memory_after_first_optimizer_step": first_step_memory,
        "memory_after_training": final_memory,
        "peak_growth_after_first_step_kib": (
            final_peak - first_peak if first_peak is not None else None
        ),
    }


def base_evidence(mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "TRAIN-004",
        "status": "running",
        "mode": mode,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_commit": source_commit(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_sha256(),
        "seed": SEED,
        "dtype": "float32",
        "runner": {
            "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
        },
        "packages": package_versions(),
        "optimizer": optimizer_config(mode),
        "memory_guard": {
            "before_first_step_min_available_gib": MEMORY_GUARD_BEFORE_FIRST_STEP_GIB,
            "after_first_step_min_available_gib": MEMORY_GUARD_AFTER_FIRST_STEP_GIB,
            "during_train_min_available_gib": MEMORY_GUARD_DURING_TRAIN_GIB,
        },
        "memory_at_process_entry": memory_snapshot(),
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def run_train(args: argparse.Namespace) -> int:
    configure_runtime()
    evidence = base_evidence(args.mode)
    write_json(args.evidence_out, evidence)
    try:
        data = json.loads(DATASET.read_text())
        system = data["system"]
        train_items = data["train"]

        model, tokenizer, lora_targets, load_memory, load_seconds = build_train_model(
            args.mode
        )
        evidence["load_seconds"] = load_seconds
        evidence["memory"] = load_memory
        evidence["lora"] = (
            {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "targets": lora_targets,
            }
            if args.mode == "lora"
            else None
        )

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        evidence["total_parameters"] = total
        evidence["trainable_parameters"] = trainable
        evidence["trainable_fraction"] = trainable / total

        optimizer = optimizer_for(model, args.mode)
        evidence["memory"]["after_optimizer_construction"] = memory_snapshot()
        encoded = [tokenize_example(tokenizer, system, item) for item in train_items]

        training = train_steps(
            model,
            optimizer,
            encoded,
            args.steps,
            0,
            0,
            full_memory_guard=args.mode == "full",
        )
        evidence["training"] = training
        evidence["memory"]["after_first_optimizer_step"] = training[
            "memory_after_first_optimizer_step"
        ]
        evidence["memory"]["after_training"] = training["memory_after_training"]
        evidence["memory"]["peak_rss_kib"] = training["memory_after_training"][
            "peak_rss_kib"
        ]
        evidence["memory"]["peak_growth_after_first_step_kib"] = training[
            "peak_growth_after_first_step_kib"
        ]

        checkpoint = save_checkpoint(
            args.checkpoint_out,
            model,
            optimizer,
            args.mode,
            training["global_step_end"],
            training["sample_cursor_end"],
            lora_targets,
        )
        evidence["checkpoint"] = checkpoint
        evidence["memory"]["after_checkpoint_save"] = memory_snapshot()
        evidence["status"] = "success"
        write_json(args.evidence_out, evidence)
        print("TRAIN004_EVIDENCE_BEGIN")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        print("TRAIN004_EVIDENCE_END")
        return 0
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "memory": memory_snapshot(),
        }
        write_json(args.evidence_out, evidence)
        print("TRAIN004_EVIDENCE_BEGIN")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        print("TRAIN004_EVIDENCE_END")
        raise


def run_resume(args: argparse.Namespace) -> int:
    configure_runtime()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "TRAIN-004",
        "phase": "fresh-runner-resume",
        "status": "running",
        "source_commit": source_commit(),
        "checkpoint": str(args.checkpoint_in),
        "runner": {
            "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
        },
        "packages": package_versions(),
        "memory_at_process_entry": memory_snapshot(),
        "additional_steps_requested": args.additional_steps,
    }
    write_json(args.evidence_out, evidence)

    try:
        verification = verify_checkpoint(
            args.checkpoint_in, args.expected_manifest_sha256
        )
        evidence["verification"] = verification

        metadata = json.loads((args.checkpoint_in / "metadata.json").read_text())
        evidence["checkpoint_metadata"] = metadata
        if metadata.get("experiment") != "TRAIN-004":
            raise RuntimeError("checkpoint is not TRAIN-004")
        if metadata.get("mode") != "full":
            raise RuntimeError("resume proof requires the full-parameter checkpoint")
        if metadata.get("base_model") != BASE_MODEL:
            raise RuntimeError("base model mismatch")
        if metadata.get("base_revision") != BASE_REVISION:
            raise RuntimeError("base revision mismatch")
        if metadata.get("dataset_sha256") != dataset_sha256():
            raise RuntimeError("dataset SHA-256 mismatch")

        data = json.loads(DATASET.read_text())
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        evidence["memory_before_model_load"] = memory_snapshot()
        load_started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_in / "model",
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.config.use_cache = False
        evidence["model_load_seconds"] = time.monotonic() - load_started
        evidence["memory_after_model_load"] = memory_snapshot()

        optimizer = optimizer_for(model, "full")
        evidence["memory_after_optimizer_construction"] = memory_snapshot()
        state = torch.load(
            args.checkpoint_in / "train_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        evidence["memory_after_optimizer_restore"] = memory_snapshot()

        global_step = int(state["global_step"])
        sample_cursor = int(state["sample_cursor"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        evidence["global_step_before"] = global_step
        evidence["sample_cursor_before"] = sample_cursor

        encoded = [
            tokenize_example(tokenizer, data["system"], item) for item in data["train"]
        ]
        training = train_steps(
            model,
            optimizer,
            encoded,
            args.additional_steps,
            global_step,
            sample_cursor,
            full_memory_guard=True,
        )
        evidence["training"] = training
        evidence["global_step_after"] = training["global_step_end"]
        evidence["sample_cursor_after"] = training["sample_cursor_end"]
        evidence["status"] = "success"
        write_json(args.evidence_out, evidence)
        print("TRAIN004_RESUME_EVIDENCE_BEGIN")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        print("TRAIN004_RESUME_EVIDENCE_END")
        return 0
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "memory": memory_snapshot(),
        }
        write_json(args.evidence_out, evidence)
        print("TRAIN004_RESUME_EVIDENCE_BEGIN")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        print("TRAIN004_RESUME_EVIDENCE_END")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TRAIN-004 bounded SmolLM2-135M CPU full-parameter feasibility experiment"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--mode", choices=["full", "lora"], required=True)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--checkpoint-out", type=Path, required=True)
    train.add_argument("--evidence-out", type=Path, required=True)

    resume = sub.add_parser("resume")
    resume.add_argument("--checkpoint-in", type=Path, required=True)
    resume.add_argument("--additional-steps", type=int, required=True)
    resume.add_argument("--evidence-out", type=Path, required=True)
    resume.add_argument("--expected-manifest-sha256")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "train":
        if args.steps < 1 or args.steps > 32:
            raise SystemExit("--steps must be in 1..32 for TRAIN-004")
        return run_train(args)
    if args.command == "resume":
        if args.additional_steps < 1 or args.additional_steps > 16:
            raise SystemExit("--additional-steps must be in 1..16 for TRAIN-004")
        return run_resume(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
