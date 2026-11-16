#!/usr/bin/env python3
"""W8: fresh full-LFM WORM paired legal-mask contrastive training.

Only TRAIN W0 observations feed the optimizer. Every changed legal mask is
marked synthetic (NOT a W0-host or Godot-native observation). The reviewed
complete multimodal parent and official processor/EOS7 remain unchanged;
the 92 reviewed rank-8 language targets train while vision/projector stay
frozen. The long model-in-loop gallery is a separate checkpoint-only gate.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time

import worm_train_w7_v1 as w7
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint, load_base, save_checkpoint, state_for

VERSION = "lfm-worm-w8-paired-legal-contrast-fresh-v1"
SAMPLE_SEED = 612801
TRAIN_SEED = 612802
PAIR_COUNT = 45
STEPS = 180


def paired_training_rows(splits):
    """45 W0 TRAIN observations each paired with a synthetic legal exclusion.

    Exactly three originals per movement action and family. The target for
    the modified case is another action already legal in that original W0
    state; only its legal list changes. No DEVELOPMENT or FINAL target read.
    """
    train = splits["WORM_TRAIN"]
    dev_prompts = {r["prompt"] for r in splits["WORM_DEV"]}
    rng = random.Random(SAMPLE_SEED)
    pairs = []
    seen = set()
    for family in w7.FAMILIES:
        for action in w7.ACTIONS[:-1]:
            candidates = sorted((r for r in train if r["family"] == family
                                 and r["completion"] == action
                                 and r["origin"] == "W0-observed"),
                                key=lambda x: x["id"])
            rng.shuffle(candidates)
            if len(candidates) < 3:
                raise ValueError(f"too few original W0 TRAIN rows: {family}/{action}")
            for index, original in enumerate(candidates[:3]):
                obs = json.loads(original["prompt"])
                legal = list(obs["legal"])
                if action not in legal or "STOP" not in legal:
                    raise ValueError("original W0 action or STOP not legal")
                alternatives = [a for a in w7.ACTIONS[:-1] if a != action and a in legal]
                alternate = (alternatives[(index + w7.ACTIONS.index(action))
                                          % len(alternatives)] if alternatives else "STOP")
                modified = dict(obs)
                modified["legal"] = [a for a in legal if a != action]
                original_prompt, changed_prompt = canonical(obs), canonical(modified)
                if original_prompt != original["prompt"] or original_prompt == changed_prompt:
                    raise ValueError("original observation mutated unexpectedly")
                if alternate not in modified["legal"] or action in modified["legal"]:
                    raise ValueError("bad synthetic legal exclusion")
                stem = f"{family}:{action}:{index}:{original['id']}"
                pair = (
                    {"id": stem + ":original", "pair_id": stem,
                     "family": family, "prompt": original_prompt,
                     "completion": action, "legal": legal,
                     "origin": "W0-observed-TRAIN"},
                    {"id": stem + ":legal-exclusion", "pair_id": stem,
                     "family": family, "prompt": changed_prompt,
                     "completion": alternate, "legal": modified["legal"],
                     "origin": "synthetic-TRAIN-not-W0",
                     "excluded_action": action},
                )
                for row in pair:
                    if row["prompt"] in seen or row["prompt"] in dev_prompts:
                        raise ValueError("duplicate TRAIN observation or DEVELOPMENT overlap")
                    if row["completion"] not in row["legal"]:
                        raise ValueError("teacher target outside legal set")
                    seen.add(row["prompt"])
                pairs.append(pair)
    if len(pairs) != PAIR_COUNT or len(seen) != PAIR_COUNT * 2:
        raise ValueError("unexpected distinct paired TRAIN count")
    return pairs


def main():
    p = argparse.ArgumentParser()
    for key in ("model", "worm_root", "support_root", "output"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--deadline", type=int, required=True)
    args = p.parse_args()
    if args.steps != STEPS or not 2200 <= args.deadline <= 2900:
        raise ValueError("W8 exactly 180 optimizer steps and finite wall limit")
    if args.output.exists():
        raise ValueError("immutable W8 output lease exists")
    args.output.mkdir(parents=True)
    t0 = time.monotonic()
    source = w7.git("rev-parse", "HEAD", cwd=w7.ROOT.parents[1])
    write_json(args.output / "lease.json", {
        "experiment": VERSION, "source_sha": source,
        "github_run": os.environ.get("GITHUB_RUN_ID"),
        "github_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "W0_source": w7.WORM_SHA, "support_source": w7.SUPPORT_SHA,
        "requested_steps": STEPS, "new_stage_from": "pinned-base-not-W7",
        "sealed_FINAL_opened": False, "native_execution": False})
    worm = w7.worm_source(args.worm_root)
    support = w7.support_source(args.support_root, worm)
    splits, custody = w7.dataset(support)
    pairs = paired_training_rows(splits)
    rows = [x for pair in pairs for x in pair]
    (args.output / "optimizer-pairs.jsonl").write_text(
        "".join(canonical(x) + "\n" for x in rows), encoding="utf-8")
    manifest = {
        "schema": "cybersand.worm.lfm-w8-paired-legal-contrast/1",
        "W0_source": w7.WORM_SHA, "support_source": w7.SUPPORT_SHA,
        "original_W0_observations": PAIR_COUNT,
        "synthetic_not_host_executed": PAIR_COUNT,
        "distinct_training_samples": len(rows), "pair_count": len(pairs),
        "TRAIN_original_rows": len(splits["WORM_TRAIN"]),
        "DEV_original_rows": len(splits["WORM_DEV"]),
        "pair_bytes_sha256": sha256_file(args.output / "optimizer-pairs.jsonl"),
        "upstream_custody": custody,
        "sealed_FINAL_opened": False, "native_execution": False,
        "TRAIN_DEV_prompt_overlap": 0}
    write_json(args.output / "corpus-manifest.json", manifest)
    selected = profile("lfm25_vl", require_supported=True)
    numerics, runtime = configure_cpu(1)
    import torch
    from peft import LoraConfig, get_peft_model
    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    base, processor = load_base(args.model, selected)
    adapter = adapter_for(selected)
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", target_modules=selected["lora_targets"]))
    adapter.set_training_mode(model)
    contract = adapter.validate_trainable(model, selected)
    encoded = [adapter.encode_row(
        processor, {**row, "completion": w7.ACTION_CODES[row["completion"]]},
        w7.SYSTEM, 2048) for row in rows]
    if any(not x["target_tokens"] or x["target_tokens"] > 32 or
           len(x["input_ids"]) > selected["max_sequence_tokens"] for x in encoded):
        raise ValueError("official processor target/EOS framing mismatch")
    recipe = {
        "rank": 8, "alpha": 16, "dropout": 0.05,
        "lr": 0.0005, "threads": 1, "seed": TRAIN_SEED,
        "parent": "complete pinned official LFM2.5-VL-450M",
        "optimizer": "AdamW", "betas": [0.9, 0.95],
        "weight_decay": 0.01, "dtype": "fp32",
        "sampler": "paired-45-W0-plus-45-synthetic-legal-exclusions-two-epochs",
        "sampler_seed": SAMPLE_SEED, "epochs": 2,
        "steps": STEPS, "action_codes": w7.ACTION_CODES}
    config = {
        "objective": VERSION, "source_sha": source, "profile": "lfm25_vl",
        "model": selected, "numerics": numerics,
        "model_files": json.loads((args.model / "download_receipt.json").read_text())["files"],
        "corpus_manifest": manifest, "system": w7.SYSTEM, "recipe": recipe,
        "final_targets_opened": False, "native_execution": False,
        "packages": {key: importlib.metadata.version(key)
                     for key in ("torch", "transformers", "peft", "torchvision", "Pillow")}}
    write_json(args.output / "preflight.json", {
        "config": config, "contract": contract,
        "trainable_parameters": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        "max_sequence_tokens": max(len(x["input_ids"]) for x in encoded),
        "numerical_runtime": runtime})
    baseline = w7.evaluation(model, processor, selected, splits["WORM_DEV"])
    write_json(args.output / "untrained-first-passes.json", baseline)
    adapter.set_training_mode(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.0005,
                                   betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    frozen_before = fingerprint(model, False)
    adapter_before = fingerprint(model, True)
    rng = random.Random(SAMPLE_SEED + 1)
    schedule = []
    for epoch in range(2):
        indices = list(range(PAIR_COUNT))
        rng.shuffle(indices)
        for i in indices:
            schedule += [2 * i, 2 * i + 1]
    if len(schedule) != STEPS or len(set(schedule)) != 90:
        raise ValueError("W8 sampler schedule mismatch")
    write_json(args.output / "sampler-receipt.json", {
        "version": VERSION, "indices": schedule,
        "IDs": [encoded[i]["id"] for i in schedule],
        "seed": SAMPLE_SEED + 1, "epochs": 2,
        "originals": 45, "synthetic_exclusions": 45})
    history = []
    checkpoint = None
    for step, index in enumerate(schedule, 1):
        if w7.STOP_REQUESTED or time.monotonic() - t0 > args.deadline - 180:
            break
        x = encoded[index]
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        ids = torch.tensor([x["input_ids"]], dtype=torch.long)
        labels = torch.tensor([x["labels"]], dtype=torch.long)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                     labels=labels, use_cache=False).loss
        if not torch.isfinite(loss).item():
            raise ValueError("W8 non-finite optimizer loss")
        loss.backward()
        grads = [p.grad for p in trainable if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads) or not any(torch.count_nonzero(g).item() for g in grads):
            raise ValueError("W8 zero/missing/nonfinite LoRA gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
        optimizer.step()
        scheduler.step()
        event = {
            "global_step": step, "sample_id": x["id"],
            "kind": rows[index]["origin"],
            "epoch": (step - 1) // 90 + 1,
            "loss": float(loss.item()), "gradient_norm": norm,
            "target_tokens": x["target_tokens"],
            "elapsed_seconds": time.monotonic() - tick}
        history.append(event)
        with (args.output / "progress.jsonl").open("a") as out:
            out.write(canonical(event) + "\n")
        print(canonical({"event": "real-W8-optimizer", **event}), flush=True)
        if step % 45 == 0 or step == STEPS:
            checkpoint = save_checkpoint(args.output, model, processor,
                state_for(optimizer, scheduler, step, step,
                          config, frozen_before, history), runtime)
    if not history:
        raise ValueError("no W8 optimizer update")
    if checkpoint is None or int(checkpoint.name.split("-")[-1]) != len(history):
        checkpoint = save_checkpoint(args.output, model, processor,
            state_for(optimizer, scheduler, len(history), len(history),
                      config, frozen_before, history), runtime)
    frozen_after, adapter_after = fingerprint(model, False), fingerprint(model, True)
    if frozen_before != frozen_after or adapter_before == adapter_after:
        raise ValueError("frozen base mutated or LoRA failed to change")
    trained = None
    if time.monotonic() - t0 < args.deadline - 180:
        trained = w7.evaluation(model, processor, selected, splits["WORM_DEV"])
        write_json(args.output / "trained-first-passes.json", trained)
    write_json(args.output / "summary.json", {
        "status": "complete" if len(history) == STEPS else "partial",
        "actual_optimizer_updates": len(history), "requested_updates": STEPS,
        "source_sha": source, "model_revision": selected["revision"],
        "model_class": "Lfm2VlForConditionalGeneration",
        "checkpoint": checkpoint.name,
        "frozen_base_unchanged": True, "LoRA_changed": True,
        "base_fingerprint": frozen_after,
        "adapter_fingerprint": adapter_after,
        "optimizer_pair_sha256": manifest["pair_bytes_sha256"],
        "untrained_dev_valid": sum(x["valid_first_pass"] for x in baseline),
        "trained_dev_valid": sum(x["valid_first_pass"] for x in trained) if trained is not None else None,
        "trained_dev_teacher_matches": sum(x["raw_action"] == x["expected"] for x in trained) if trained is not None else None,
        "dev_probe_denominator": len(trained) if trained is not None else 0,
        "online_model_rollouts": 0,
        "online_gallery_deferred_to_separate_checkpoint_only_run": True,
        "wall_seconds": time.monotonic() - t0,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sealed_FINAL_opened": False, "native_execution": False})
    if len(history) != STEPS:
        raise ValueError("W8 partial optimizer; checkpoint retained without success claim")
    print(canonical({"event": "W8-complete", "optimizer_steps": len(history)}), flush=True)


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGTERM, w7.handle_stop)
        signal.signal(signal.SIGINT, w7.handle_stop)
        main()
    except Exception as e:
        args = sys.argv
        if "--output" in args:
            out = Path(args[args.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "failure.json", {
                "status": "FAILED", "error_type": type(e).__name__,
                "error": str(e), "experiment": VERSION})
        raise
