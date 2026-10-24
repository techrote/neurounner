#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import shutil
import sys
import time
from pathlib import Path

import psutil
import torch
import transformers
import peft
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET = ROOT / "experiments" / "train-001" / "kernel_compiler_dataset.json"
RESULTS = ROOT / "results"
EXPECTED_KEYS = ["trigger", "state", "update", "action", "failure"]
SEED = 20261024

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
    # Linux ru_maxrss is KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def package_versions() -> dict:
    import huggingface_hub
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "huggingface_hub": huggingface_hub.__version__,
    }

def extract_json(text: str):
    a = text.find("{")
    b = text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a:b+1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def schema_score(text: str) -> dict:
    obj = extract_json(text)
    if obj is None:
        return {"json": False, "exact_keys": False, "key_count": 0}
    keys = list(obj.keys())
    return {
        "json": True,
        "exact_keys": set(keys) == set(EXPECTED_KEYS) and len(keys) == len(EXPECTED_KEYS),
        "key_count": sum(k in obj for k in EXPECTED_KEYS),
    }

def compact_target(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def tokenize_example(tokenizer, system: str, item: dict, max_length: int = 256):
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
    # Find exact common prefix rather than assuming the template length.
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    if len(full_ids) > max_length:
        # These examples are deliberately short; fail closed if that changes.
        raise RuntimeError(f"example length {len(full_ids)} exceeds max_length={max_length}")
    labels = [-100] * common + full_ids[common:]
    if all(x == -100 for x in labels):
        raise RuntimeError("assistant target was entirely masked")
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "tokens": len(full_ids),
    }

@torch.inference_mode()
def evaluate(model, tokenizer, system: str, examples: list[dict]) -> dict:
    model.eval()
    rows = []
    for item in examples:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": item["input"]},
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        t0 = time.monotonic()
        out = model.generate(
            ids,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.monotonic() - t0
        generated = out[0, ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        rows.append({
            "input": item["input"],
            "expected": item["output"],
            "output": text,
            "score": schema_score(text),
            "generated_tokens": int(generated.numel()),
            "elapsed_seconds": elapsed,
        })
    return {
        "rows": rows,
        "json_rate": sum(r["score"]["json"] for r in rows) / len(rows),
        "exact_schema_rate": sum(r["score"]["exact_keys"] for r in rows) / len(rows),
        "mean_expected_key_count": sum(r["score"]["key_count"] for r in rows) / len(rows),
    }

def choose_lora_targets(model) -> list[str]:
    desired = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = set()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in desired and isinstance(module, torch.nn.Linear):
            present.add(suffix)
    targets = [x for x in desired if x in present]
    if not targets:
        raise RuntimeError("could not find expected LoRA target modules")
    return targets

def build_model(base_revision: str, checkpoint_in: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=base_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=base_revision,
        torch_dtype=torch.float32,
    )
    base.config.use_cache = False

    if checkpoint_in is None:
        targets = choose_lora_targets(base)
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=targets,
        )
        model = get_peft_model(base, config)
        return model, tokenizer, targets

    model = PeftModel.from_pretrained(
        base,
        checkpoint_in / "adapter",
        is_trainable=True,
    )
    targets = sorted({
        name.rsplit(".", 2)[-2] if ".lora_" in name else name.rsplit(".", 1)[-1]
        for name, p in model.named_parameters() if p.requires_grad
    })
    return model, tokenizer, targets

def save_checkpoint(
    outdir: Path,
    model,
    optimizer,
    global_step: int,
    sample_cursor: int,
    base_revision: str,
    lora_targets: list[str],
    stage: int,
):
    outdir.mkdir(parents=True, exist_ok=True)
    adapter_dir = outdir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    state = {
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "sample_cursor": sample_cursor,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    torch.save(state, outdir / "train_state.pt")
    metadata = {
        "base_model": BASE_MODEL,
        "base_revision": base_revision,
        "global_step": global_step,
        "sample_cursor": sample_cursor,
        "stage": stage,
        "lora_targets": lora_targets,
        "seed": SEED,
        "packages": package_versions(),
        "git_sha": os.environ.get("GITHUB_SHA"),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    total_bytes = sum(p.stat().st_size for p in outdir.rglob("*") if p.is_file())
    return total_bytes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1,2], required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--checkpoint-in", type=Path)
    ap.add_argument("--checkpoint-out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(SEED)
    torch.manual_seed(SEED)

    data = json.loads(DATASET.read_text())
    system = data["system"]
    train_items = data["train"]
    eval_items = data["eval"]

    if args.stage == 1:
        base_revision = HfApi().model_info(BASE_MODEL).sha
        checkpoint_in = None
        previous_state = None
    else:
        if args.checkpoint_in is None:
            raise SystemExit("--checkpoint-in required for stage 2")
        checkpoint_in = args.checkpoint_in
        meta = json.loads((checkpoint_in / "metadata.json").read_text())
        base_revision = meta["base_revision"]
        previous_state = torch.load(checkpoint_in / "train_state.pt", map_location="cpu", weights_only=False)

    t_load = time.monotonic()
    model, tokenizer, lora_targets = build_model(base_revision, checkpoint_in)
    load_seconds = time.monotonic() - t_load

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=8e-4,
        betas=(0.9,0.95),
        weight_decay=0.01,
    )

    if previous_state is not None:
        optimizer.load_state_dict(previous_state["optimizer"])
        global_step = int(previous_state["global_step"])
        sample_cursor = int(previous_state["sample_cursor"])
        random.setstate(previous_state["python_rng"])
        torch.set_rng_state(previous_state["torch_rng"])
    else:
        global_step = 0
        sample_cursor = 0

    baseline = evaluate(model, tokenizer, system, eval_items)

    encoded = [tokenize_example(tokenizer, system, x) for x in train_items]
    losses = []
    train_tokens = 0
    t0 = time.monotonic()
    rss_before = rss_kib()

    model.train()
    for _ in range(args.steps):
        ex = encoded[sample_cursor % len(encoded)]
        sample_cursor += 1
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=ex["input_ids"],
            attention_mask=ex["attention_mask"],
            labels=ex["labels"],
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        global_step += 1
        losses.append(float(loss.detach()))
        train_tokens += ex["tokens"]

    train_seconds = time.monotonic() - t0
    after = evaluate(model, tokenizer, system, eval_items)
    checkpoint_bytes = save_checkpoint(
        args.checkpoint_out,
        model,
        optimizer,
        global_step,
        sample_cursor,
        base_revision,
        lora_targets,
        args.stage,
    )

    result = {
        "experiment": "TRAIN-001",
        "stage": args.stage,
        "base_model": BASE_MODEL,
        "base_revision": base_revision,
        "runner": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model(),
        },
        "packages": package_versions(),
        "load_seconds": load_seconds,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "lora_targets": lora_targets,
        "steps_this_stage": args.steps,
        "global_step": global_step,
        "train_tokens": train_tokens,
        "train_seconds": train_seconds,
        "train_tokens_per_second": train_tokens / train_seconds if train_seconds else None,
        "mean_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "rss_before_kib": rss_before,
        "rss_after_kib": rss_kib(),
        "peak_rss_kib": peak_rss_kib(),
        "checkpoint_bytes": checkpoint_bytes,
        "evaluation_before_stage": baseline,
        "evaluation_after_stage": after,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"train001-stage{args.stage}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
