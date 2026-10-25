#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments" / "train-001" / "kernel_compiler_dataset.json"
DEFAULT_LOCAL_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_LOCAL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SEED = 20261025


def cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(
                    ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
                ).strip()
                if value:
                    return value
            except Exception:
                pass
    if os.name == "nt":
        value = os.environ.get("PROCESSOR_IDENTIFIER")
        if value:
            return value
    return platform.processor() or "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def machine_metadata() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(ROOT)
    return {
        "lane": os.environ.get("RUNNER001_LANE"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "image_os": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "memory_bytes": vm.total,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "python": sys.version.split()[0],
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_sha": os.environ.get("GITHUB_SHA"),
    }


def timed_json_work(dataset_bytes: bytes) -> dict[str, Any]:
    obj = json.loads(dataset_bytes)
    canonical = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    loops = 4000
    t0 = time.perf_counter()
    total = 0
    for _ in range(loops):
        parsed = json.loads(canonical)
        encoded = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        total += len(canonical) + len(encoded)
    elapsed = time.perf_counter() - t0
    return {
        "loops": loops,
        "bytes_processed": total,
        "seconds": elapsed,
        "mib_per_second": total / elapsed / (1024 * 1024),
        "canonical_bytes": len(canonical),
    }


def timed_hash_work(dataset_bytes: bytes) -> dict[str, Any]:
    seed = hashlib.sha256(dataset_bytes).digest()
    block = (seed * ((4 * 1024 * 1024 // len(seed)) + 1))[: 4 * 1024 * 1024]
    loops = 64
    t0 = time.perf_counter()
    digest = None
    for _ in range(loops):
        digest = hashlib.sha256(block).hexdigest()
    elapsed = time.perf_counter() - t0
    total = len(block) * loops
    return {
        "loops": loops,
        "bytes_processed": total,
        "seconds": elapsed,
        "mib_per_second": total / elapsed / (1024 * 1024),
        "last_digest": digest,
    }


def timed_compress_work(dataset_bytes: bytes) -> dict[str, Any]:
    block = (dataset_bytes * ((8 * 1024 * 1024 // len(dataset_bytes)) + 1))[: 8 * 1024 * 1024]
    loops = 8
    sizes: list[int] = []
    t0 = time.perf_counter()
    for _ in range(loops):
        sizes.append(len(zlib.compress(block, level=6)))
    elapsed = time.perf_counter() - t0
    total = len(block) * loops
    return {
        "loops": loops,
        "bytes_processed": total,
        "seconds": elapsed,
        "mib_per_second": total / elapsed / (1024 * 1024),
        "mean_compressed_bytes": statistics.fmean(sizes),
        "compression_ratio": statistics.fmean(sizes) / len(block),
        "zlib_version": zlib.ZLIB_VERSION,
    }


def torch_matmul_benchmark(torch, threads: int) -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    n = 1024
    reps = 6
    a = torch.randn((n, n), dtype=torch.float32)
    b = torch.randn((n, n), dtype=torch.float32)
    _ = a @ b
    t0 = time.perf_counter()
    checksum = 0.0
    for _ in range(reps):
        c = a @ b
        checksum += float(c[0, 0])
    elapsed = time.perf_counter() - t0
    flops = 2 * (n ** 3) * reps
    return {
        "matrix_size": n,
        "repetitions": reps,
        "threads": threads,
        "seconds": elapsed,
        "estimated_gflops": flops / elapsed / 1e9,
        "checksum": checksum,
    }


def choose_lora_targets(model, torch) -> list[str]:
    desired = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = set()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in desired and isinstance(module, torch.nn.Linear):
            present.add(suffix)
    targets = [x for x in desired if x in present]
    if not targets:
        raise RuntimeError("no expected LoRA target modules found")
    return targets


def tokenize_training_example(tokenizer, torch, system: str, item: dict[str, Any]):
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": item["input"]},
    ]
    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": json.dumps(item["output"], ensure_ascii=False, separators=(",", ":")),
        }
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
    labels = [-100] * common + full_ids[common:]
    if all(x == -100 for x in labels):
        raise RuntimeError("training target was entirely masked")
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "sequence_tokens": len(full_ids),
        "supervised_tokens": sum(x != -100 for x in labels),
    }


def local_model_bench(
    dataset: dict[str, Any],
    model_id: str,
    revision: str,
    threads: int,
) -> dict[str, Any]:
    import huggingface_hub
    import peft
    import psutil
    import safetensors
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "psutil": psutil.__version__,
    }

    before = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
    )
    model.eval()
    load_seconds = time.perf_counter() - t0
    after_load = psutil.Process().memory_info().rss

    system = dataset["system"]
    eval_item = dataset["eval"][0]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": eval_item["input"]},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    attention_mask = torch.ones_like(ids)

    with torch.inference_mode():
        _ = model.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_rows = []
    generated_total = 0
    generation_seconds = 0.0
    with torch.inference_mode():
        for _ in range(3):
            t1 = time.perf_counter()
            out = model.generate(
                ids,
                attention_mask=attention_mask,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - t1
            generated = int(out.shape[-1] - ids.shape[-1])
            generated_total += generated
            generation_seconds += elapsed
            generation_rows.append(
                {"generated_tokens": generated, "seconds": elapsed, "tokens_per_second": generated / elapsed}
            )

    model.train()
    targets = choose_lora_targets(model, torch)
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=8e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    batches = [
        tokenize_training_example(tokenizer, torch, system, item)
        for item in dataset["train"][:4]
    ]
    losses = []
    train_tokens = 0
    train_supervised_tokens = 0
    t2 = time.perf_counter()
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        train_tokens += batch["sequence_tokens"]
        train_supervised_tokens += batch["supervised_tokens"]
    train_seconds = time.perf_counter() - t2
    after_train = psutil.Process().memory_info().rss

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "model": model_id,
        "revision": revision,
        "versions": versions,
        "threads": threads,
        "load": {
            "seconds": load_seconds,
            "rss_before_bytes": before,
            "rss_after_bytes": after_load,
            "rss_growth_bytes": after_load - before,
        },
        "generation": {
            "prompt_tokens": int(ids.shape[-1]),
            "runs": generation_rows,
            "generated_tokens": generated_total,
            "seconds": generation_seconds,
            "tokens_per_second": generated_total / generation_seconds,
        },
        "lora_training": {
            "steps": len(batches),
            "sequence_tokens": train_tokens,
            "supervised_tokens": train_supervised_tokens,
            "seconds": train_seconds,
            "sequence_tokens_per_second": train_tokens / train_seconds,
            "supervised_tokens_per_second": train_supervised_tokens / train_seconds,
            "losses": losses,
            "targets": targets,
            "trainable_parameters": trainable,
            "total_parameters": total_params,
            "trainable_fraction": trainable / total_params,
            "rss_after_bytes": after_train,
        },
    }


def github_models_status() -> dict[str, Any]:
    return {
        "status": "retired",
        "retired_date": "2026-07-30",
        "reason": (
            "GitHub Models playground, catalog, inference API, and BYOK were "
            "fully retired by GitHub on 2026-07-30. No remote throughput number "
            "is fabricated from the retired endpoint."
        ),
        "docs": "https://docs.github.com/en/github-models",
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    ap.add_argument("--local-revision", default=DEFAULT_LOCAL_REVISION)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    os.environ["RUNNER001_LANE"] = args.lane
    dataset_raw_bytes = DATASET.read_bytes()
    dataset = json.loads(dataset_raw_bytes)
    dataset_bytes = canonical_json_bytes(dataset)

    import psutil
    import torch

    threads = min(4, psutil.cpu_count(logical=True) or 1)
    started = time.time()
    result: dict[str, Any] = {
        "schema": "neurounner.runner001.v1",
        "started_unix": started,
        "source": {
            "issue": 51,
            "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
            "dataset_raw_sha256": sha256_file(DATASET),
            "dataset_canonical_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "dataset_canonical_bytes": len(dataset_bytes),
            "local_model": args.local_model,
            "local_revision": args.local_revision,
        },
        "machine": machine_metadata(),
        "benchmarks": {},
    }

    result["benchmarks"]["json_roundtrip"] = timed_json_work(dataset_bytes)
    result["benchmarks"]["sha256"] = timed_hash_work(dataset_bytes)
    result["benchmarks"]["zlib"] = timed_compress_work(dataset_bytes)
    result["benchmarks"]["torch_matmul"] = torch_matmul_benchmark(torch, threads)
    result["benchmarks"]["local_model"] = local_model_bench(
        dataset, args.local_model, args.local_revision, threads
    )
    result["benchmarks"]["github_models"] = github_models_status()
    result["finished_unix"] = time.time()
    result["wall_seconds"] = result["finished_unix"] - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    local = result["benchmarks"]["local_model"]
    remote = result["benchmarks"]["github_models"]
    print(json.dumps({
        "lane": args.lane,
        "cpu": result["machine"]["cpu_model"],
        "runner_arch": result["machine"]["runner_arch"],
        "local_generation_tok_s": local["generation"]["tokens_per_second"],
        "lora_sequence_tok_s": local["lora_training"]["sequence_tokens_per_second"],
        "torch_gflops": result["benchmarks"]["torch_matmul"]["estimated_gflops"],
        "github_models_status": remote["status"],
        "github_models_median_s": remote.get("median_seconds"),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
