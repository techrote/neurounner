#!/usr/bin/env python3
"""WORM-001 LFM W1: genuine bounded CPU LoRA on local legal steering actions.

New synthetic action objective; NEVER native Art v2 or FINAL. Uses the qualified
full LFM2.5-VL adapter, official processor, CPU numerics, and checkpoint writer.
The shepherd's pinned W0 walker alone owns legality and PNG rasterisation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint, load_base, save_checkpoint, state_for

WORM_SHA = "2be9fd02ee525f2eac999555d4e7392a1dbc739b"
WORM_BLOB = "04b9cf0dae74bcc9898e3641418973787141814b"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
SYSTEM = ("You are a tiny procedural CyberSand worm steering brain. "
          "Return exactly ONE uppercase token from the listed legal actions. "
          "Do not describe the drawing. The host controls materials, obstacles, limits and Stop.")
TRAIN_SEEDS = (11, 29, 43)
DEV_SEEDS = (73, 89)
STOP_REQUESTED = False


def handle_stop(_sig, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def worm_source(path):
    root = Path(path).resolve()
    source = root / "worm.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("missing exact W0 source")
    if git("rev-parse", "HEAD", cwd=root.parents[3]) != WORM_SHA:
        raise ValueError("unqualified W0 checkout")
    if git("hash-object", str(source)) != WORM_BLOB:
        raise ValueError("W0 walker bytes mismatch")
    sys.path.insert(0, str(root))
    worm = importlib.import_module("worm")
    if (worm.ACTIONS != ACTIONS or worm.FAMILIES != FAMILIES
            or worm.run.__module__ != "worm"):
        raise ValueError("W0 observation/action contract mismatch")
    return worm


def make_prompt(observation):
    if set(observation["legal"]) - set(ACTIONS):
        raise ValueError("unknown legal action")
    compact = {key: observation[key] for key in (
        "family", "heading", "local_5x5", "legal", "last_actions", "tick")}
    return canonical(compact)


def dataset(worm):
    all_rows = {}
    for split, seeds in (("WORM_TRAIN", TRAIN_SEEDS), ("WORM_DEV", DEV_SEEDS)):
        rows = []
        for family in FAMILIES:
            for seed in seeds:
                record = worm.run(seed, "fidget", family, steps=96, size=96)
                for index, event in enumerate(record["events"]):
                    if index % 5:
                        continue
                    action = event["applied"]
                    observation = event["observation"]
                    if action not in observation["legal"] or action not in ACTIONS:
                        raise ValueError("teacher action not legal in original W0 observation")
                    rows.append({"id": f"{split}-{family}-{seed}-{index}",
                                 "family": family, "seed": seed,
                                 "prompt": make_prompt(observation),
                                 "completion": action,
                                 "legal": observation["legal"]})
        if not rows:
            raise ValueError("empty worm split")
        all_rows[split] = rows
    if {r["seed"] for r in all_rows["WORM_TRAIN"]} & {r["seed"] for r in all_rows["WORM_DEV"]}:
        raise ValueError("worm TRAIN/DEV seed leakage")
    return all_rows


def generate(model, tokenizer, selected, prompt, legal, *, max_tokens=10):
    from lane_adapters import encode_prompt
    import torch
    ids = encode_prompt(tokenizer, prompt, SYSTEM, selected)
    if len(ids) + max_tokens > selected["max_sequence_tokens"]:
        raise ValueError("inference context overflow; no truncation")
    started = time.monotonic()
    model.eval()
    inputs = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                             do_sample=False, max_new_tokens=max_tokens,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id,
                             use_cache=True)[0, len(ids):].tolist()
    eos = bool(out) and out[-1] == tokenizer.eos_token_id
    decoded = tokenizer.decode(out, skip_special_tokens=True)
    # No stripping, JSON repair, retries, fallback teacher or other model-output edits.
    valid = eos and decoded in legal and decoded in ACTIONS
    return {"raw_text": decoded, "token_ids": out, "eos_terminated": eos,
            "raw_legal": decoded in legal, "valid_first_pass": valid,
            "input_tokens": len(ids), "output_tokens": len(out),
            "seconds": time.monotonic() - started}


def evaluation(model, tokenizer, selected, rows, *, limit=9):
    chosen = [r for family in FAMILIES
              for r in [x for x in rows if x["family"] == family][:limit // 3]]
    return [{"id": row["id"], "expected": row["completion"],
             "legal": row["legal"],
             **generate(model, tokenizer, selected, row["prompt"], row["legal"])}
            for row in chosen]


def rollout(model, tokenizer, selected, worm, family, seed, max_steps=16):
    raw_actions = []
    receipts = []
    for _ in range(max_steps):
        # Host W0 supplies the next REAL observation and legality. A temporary
        # STOP only exposes the next observation; never feeds the model a teacher.
        observation_record = worm.run(
            seed, mode="replay", family=family, steps=len(raw_actions) + 1,
            size=96, raw_actions=raw_actions + ["STOP"])
        if len(observation_record["events"]) != len(raw_actions) + 1:
            break
        observation = observation_record["events"][-1]["observation"]
        first = generate(model, tokenizer, selected, make_prompt(observation),
                         observation["legal"], max_tokens=8)
        raw_actions.append(first["raw_text"])
        receipts.append({"observation": observation, **first})
        if first["raw_text"] == "STOP":
            break
        # Bad model output remains unmodified and is handled only by the W0 host.
        applied = worm.run(seed, mode="replay", family=family, steps=len(raw_actions),
                           size=96, raw_actions=raw_actions)
        if applied["events"][-1]["applied"] == "STOP":
            break
    record = worm.run(seed, mode="replay", family=family,
                      steps=len(raw_actions), size=96, raw_actions=raw_actions)
    return record, receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--deadline", type=int, default=1600)
    args = parser.parse_args()
    if not 1 <= args.steps <= 8 or not 180 <= args.deadline <= 2400:
        raise ValueError("WORM pilot bounded to 1–8 steps and 180–2400 seconds")
    if args.output.exists():
        raise ValueError("immutable output lease already exists")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    source = git("rev-parse", "HEAD", cwd=ROOT.parents[1])
    write_json(args.output / "lease.json", {"experiment": "lfm-worm-w1-fresh-v1",
               "source_sha": source, "worm_sha": WORM_SHA,
               "github_run_id": os.environ.get("GITHUB_RUN_ID"),
               "requested_steps": args.steps, "created_unix": time.time()})
    worm = worm_source(args.worm_root)
    rows = dataset(worm)
    for split, records in rows.items():
        (args.output / (split.lower() + ".jsonl")).write_text(
            "".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    manifest = {"schema": "cybersand.worm.lfm-synthetic-train/1",
                "objective": "local-action-fidget-imitation",
                "W0_source": WORM_SHA, "W0_blob": WORM_BLOB,
                "TRAIN_seeds": TRAIN_SEEDS, "DEV_seeds": DEV_SEEDS,
                "TRAIN_count": len(rows["WORM_TRAIN"]),
                "DEV_count": len(rows["WORM_DEV"]),
                "WORM_TRAIN_sha256": sha256_file(args.output / "worm_train.jsonl"),
                "WORM_DEV_sha256": sha256_file(args.output / "worm_dev.jsonl"),
                "sealed_FINAL_opened": False, "native_execution": False}
    write_json(args.output / "corpus-manifest.json", manifest)
    selected = profile("lfm25_vl", require_supported=True)
    numerics, runtime = configure_cpu(1)
    import torch
    from peft import LoraConfig, get_peft_model
    random.seed(612568)
    torch.manual_seed(612568)
    base, tokenizer = load_base(args.model, selected)
    adapter = adapter_for(selected)
    if adapter is None:
        raise ValueError("missing source-fenced LFM lane adapter")
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", target_modules=selected["lora_targets"]))
    adapter.set_training_mode(model)
    contract = adapter.validate_trainable(model, selected)
    encoded = [adapter.encode_row(tokenizer, r, SYSTEM, 2048)
               for r in rows["WORM_TRAIN"]]
    if any(not e["target_tokens"] or e["target_tokens"] > 32 for e in encoded):
        raise ValueError("unbounded action teacher response")
    config = {"objective": "lfm-worm-action-selection-v1-fresh",
              "source_sha": source, "profile": "lfm25_vl",
              "model": selected, "numerics": numerics,
              "model_files": json.loads((args.model / "download_receipt.json").read_text())["files"],
              "corpus_manifest": manifest, "system": SYSTEM,
              "recipe": {"rank": 8, "alpha": 16, "dropout": 0.05, "lr": 0.0005,
                         "threads": 1, "seed": 612568, "parent": "pinned-official-base",
                         "optimizer": "AdamW", "betas": [0.9, 0.95],
                         "weight_decay": 0.01, "dtype": "fp32"},
              "final_targets_opened": False, "native_execution": False,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "peft", "torchvision", "Pillow")}}
    write_json(args.output / "preflight.json", {
        "config": config, "model_contract": contract,
        "train_rows": len(encoded), "dev_rows": len(rows["WORM_DEV"]),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "sequence_max": max(len(e["input_ids"]) for e in encoded),
        "numerical_runtime": runtime})
    # Original first-pass negative evidence before any optimizer update.
    baseline = evaluation(model, tokenizer, selected, rows["WORM_DEV"])
    write_json(args.output / "untrained-first-passes.json", baseline)
    adapter.set_training_mode(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.0005,
                                   betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    frozen_before = fingerprint(model, False)
    adapter_before = fingerprint(model, True)
    history = []
    last_checkpoint = None
    for step in range(1, args.steps + 1):
        if STOP_REQUESTED or time.monotonic() - started > args.deadline - 120:
            break
        item = encoded[(step - 1) * 7 % len(encoded)]
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        ids = torch.tensor([item["input_ids"]], dtype=torch.long)
        labels = torch.tensor([item["labels"]], dtype=torch.long)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                     labels=labels, use_cache=False).loss
        if not torch.isfinite(loss).item():
            raise ValueError("nonfinite LFM WORM loss")
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients) or not any(torch.count_nonzero(g).item() for g in gradients):
            raise ValueError("zero, missing or nonfinite actual LoRA gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        optimizer.step()
        scheduler.step()
        event = {"global_step": step, "sample_id": item["id"],
                 "loss": float(loss.item()), "gradient_norm": norm,
                 "target_tokens": item["target_tokens"],
                 "elapsed_seconds": time.monotonic() - t0}
        history.append(event)
        print(canonical({"event": "real-worm-optimizer-step", **event}), flush=True)
        (args.output / "progress.jsonl").open("a").write(canonical(event) + "\n")
        if step % 2 == 0 or step == args.steps:
            last_checkpoint = save_checkpoint(args.output, model, tokenizer,
                state_for(optimizer, scheduler, step, step, config, frozen_before, history),
                runtime)
    if not history:
        raise ValueError("no actual optimizer step: not a training pass")
    if last_checkpoint is None or int(last_checkpoint.name.split("-")[-1]) != len(history):
        last_checkpoint = save_checkpoint(args.output, model, tokenizer,
            state_for(optimizer, scheduler, len(history), len(history),
                      config, frozen_before, history), runtime)
    frozen_after = fingerprint(model, False)
    adapter_after = fingerprint(model, True)
    if frozen_before != frozen_after or adapter_before == adapter_after:
        raise ValueError("LFM frozen base changed or adapter failed to update")
    trained = evaluation(model, tokenizer, selected, rows["WORM_DEV"])
    write_json(args.output / "trained-first-passes.json", trained)
    gallery = args.output / "gallery"
    gallery.mkdir()
    for family in FAMILIES:
        record, raw = rollout(model, tokenizer, selected, worm, family, seed=89, max_steps=16)
        worm.write_result(record, gallery)
        write_json(gallery / f"{family}-model-89-raw-decisions.json", raw)
        for mode in ("random", "wall", "fidget"):
            worm.write_result(worm.run(89, mode=mode, family=family,
                                      steps=16, size=96), gallery)
    elapsed = time.monotonic() - started
    write_json(args.output / "summary.json", {
        "status": "completed" if len(history) == args.steps else "partial",
        "actual_training": True, "optimizer_updates": len(history),
        "requested_updates": args.steps, "model_class": "Lfm2VlForConditionalGeneration",
        "model_revision": selected["revision"], "trainable_parameters": 2998272,
        "checkpoint": last_checkpoint.name, "checkpoint_readback_exact": True,
        "frozen_base_unchanged": True, "adapter_changed": True,
        "adapter_before_sha256": adapter_before,
        "adapter_after_sha256": adapter_after,
        "source_sha": source, "W0_source": WORM_SHA,
        "dataset_sha256": manifest["WORM_TRAIN_sha256"],
        "untrained_first_pass_valid": sum(x["valid_first_pass"] for x in baseline),
        "trained_first_pass_valid": sum(x["valid_first_pass"] for x in trained),
        "evaluated_first_passes_per_arm": len(baseline),
        "wall_seconds": elapsed, "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "native_execution": False, "sealed_FINAL_opened": False,
        "experimental_worm_only": True})
    if len(history) != args.steps:
        raise ValueError("bounded WORM run partial; saved real checkpoints, not completed")
    print(canonical({"event": "worm-run-finished", "steps": len(history),
                     "wall_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    try:
        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGINT, handle_stop)
        main()
    except Exception as exc:
        # Distinct failure evidence; never claim a partly completed run as success.
        arguments = sys.argv
        if "--output" in arguments:
            out = Path(arguments[arguments.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "failure.json", {"status": "FAILED",
                       "error_type": type(exc).__name__, "error": str(exc)})
        raise
