#!/usr/bin/env python3
"""WORM-001 SPROCKET W4 CPU LoRA pilot: six-action single-digit output.

WORM synthetic steering does NOT impersonate Art v2 native-admitted TRAIN. The
pinned loader, model bytes, CPU numerical profile, LoRA boundary, atomic checkpoint
mechanics, and optimizer checks are reused without changing shared source.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(HERE))
import train as shared
from common import digest_json, profile, sha256_file, write_json
from lane_adapters import adapter_for, adapter_identity
from numerics import configure_cpu
import worm_v1 as contract
import worm_v4 as objective

STOP = False


def stop_handler(_signum, _frame):
    global STOP
    STOP = True


def first_pass(model, tokenizer, adapter, rows, label, out):
    """Unmodified greedy token decisions, true EOS5 required, no retry or repair."""
    import torch
    model.eval()
    results = []
    for row in rows:
        prefix = adapter.encode_prompt(tokenizer, row["prompt"], objective.SYSTEM)
        with torch.no_grad():
            generated = model.generate(input_ids=torch.tensor([prefix], dtype=torch.long),
                                       do_sample=False, max_new_tokens=12, pad_token_id=1,
                                       eos_token_id=5, use_cache=True)
        ids = generated[0].tolist()[len(prefix):]
        has_eos = 5 in ids and ids[-1] == 5
        content_ids = ids[:-1] if has_eos else ids
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        choice = tokenizer.decode(content_ids, skip_special_tokens=False)
        decoded = objective.CODE_TO_ACTION.get(choice)
        legal = has_eos and decoded in row["legal"] and all(i >= 16 for i in content_ids)
        results.append({"id": row["id"], "family": row["family"], "seed": row["seed"],
                        "prefix_ids": prefix, "completion_ids": ids, "raw": raw,
                        "content_code": choice, "decoded_action": decoded, "eos5": has_eos, "raw_legal": legal,
                        "applied_if_hosted": decoded if legal else ("RETRACE" if "RETRACE" in row["legal"] else "STOP"),
                        "legal_actions": row["legal"], "teacher_reference_not_quality_gate": row["action"],
                        "teacher_code": row["completion"]})
    write_json(out / (label + "-raw-first-pass.json"), results)
    return {"decisions": len(results), "eos": sum(x["eos5"] for x in results),
            "legal": sum(x["raw_legal"] for x in results)}


def main(args):
    if args.steps < 1 or args.steps > 96 or args.deadline < 90 or args.deadline > 1800:
        raise ValueError("WORM v4 must remain a finite 1-96 step campaign")
    started = time.monotonic()
    selected = profile("sprocket", require_supported=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "lease.json", {"source_sha": shared.source_sha(),
              "model": selected["model_id"], "revision": selected["revision"],
              "experiment": "worm-sprocket-steering-code-v4-fresh", "steps": args.steps,
              "deadline_seconds": args.deadline, "run_id": os.environ.get("GITHUB_RUN_ID"),
              "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "native": False})
    numeric, numerical_runtime = configure_cpu(1)
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(612)
    random.seed(612)

    support = objective.load_support(args.support, args.w0)
    w0 = support.worm
    datasets, generated = objective.dataset(support, args.output / "dataset")
    train_rows = datasets["train"]
    heldout_rows = datasets["eval"]
    splits = {name: {"sha256": generated[name]["sha256"], "rows": generated[name]["rows"],
                     "actions": generated[name]["by_action"]} for name in ("train", "eval")}
    adapter = adapter_for(selected)
    base, tokenizer = shared.load_base(args.model, selected)
    # W4 digit coding must be one root-Instruct noncontrol lexical token each.
    action_token_ids = {code: tokenizer.encode(code, add_special_tokens=False)
                        for code in objective.CODE_TO_ACTION}
    if any(len(ids) != 1 or ids[0] < 16 for ids in action_token_ids.values()):
        raise ValueError("W4 code action must encode as exactly one noncontrol token")
    if len({ids[0] for ids in action_token_ids.values()}) != 6:
        raise ValueError("W4 code token collision")
    encoded = [adapter.encode_row(tokenizer, row, objective.SYSTEM, 2048) for row in train_rows]
    if max(len(x["input_ids"]) for x in encoded) > 2048 or any(x["target_tokens"] < 2 for x in encoded):
        raise ValueError("unsafe synthetic framing or truncated action target")
    # Balanced per family/action, with explicit repeated rare motifs and fixed target steps.
    import hashlib
    buckets = {}
    for index, item in enumerate(train_rows):
        buckets.setdefault((item["action"], item["family"]), []).append(index)
    for action in contract.ACTIONS:
        for family in contract.FAMILIES:
            bucket = buckets.get((action, family))
            if not bucket:
                raise ValueError("missing legal TRAIN bucket " + action + "/" + family)
            bucket.sort(key=lambda i: hashlib.sha256(
                ("612:v4:" + train_rows[i]["id"]).encode()).digest())
    order = []
    for offset in range((args.steps + 17) // 18):
        for action in contract.ACTIONS:
            for family in contract.FAMILIES:
                bucket = buckets[(action, family)]
                order.append(bucket[offset % len(bucket)])
    order = order[:args.steps]
    if len(order) != args.steps:
        raise ValueError("incomplete bounded balanced training order")

    packages = {name: importlib.metadata.version(name) for name in shared.PACKAGES}
    w0_hash = sha256_file(args.w0)
    cfg = {"schema": objective.SCHEMA, "model": selected, "source_sha": shared.source_sha(),
           "objective": "v4 one-digit six-action steering, full observation, support-v0 teacher, EOS5",
           "action_to_code": objective.ACTION_TO_CODE, "action_code_token_ids": action_token_ids,
           "sampler": "96-step fresh action/family cycling, sha256(612:v4:row-id) within buckets",
           "worm_common_git_commit": objective.W0_COMMIT, "worm_common_sha256": w0_hash,
           "worm_support_git_commit": objective.SUPPORT_COMMIT,
           "worm_support_sha256": sha256_file(args.support),
           "model_files": json.loads((args.model / "download_receipt.json").read_text())["files"],
           "synthetic_train_sha256": splits["train"]["sha256"],
           "synthetic_eval_sha256": splits["eval"]["sha256"],
           "synthetic_not_native": True, "final_opened": False,
           "training_code_sha256": sha256_file(__file__), "data_code_sha256": sha256_file(objective.__file__),
           "shared_trainer_sha256": sha256_file(TRAINING / "train.py"),
           "shared_checkpoint_contract_sha256": sha256_file(TRAINING / "common.py"),
           "adapter_contract": adapter_identity(selected),
           "chat_template_hash": digest_json(tokenizer.chat_template),
           "system_hash": digest_json(objective.SYSTEM), "system": objective.SYSTEM,
           "sample_order": [train_rows[i]["id"] for i in order],
           "literal_train_eval_overlap": generated["literal_prompt_intersection"],
           "numerics": numeric, "packages": packages, "python": platform.python_version(),
           "seed": 612, "learning_rate": 0.0005, "rank": 8, "alpha": 16, "dropout": 0.05,
           "optimizer": "AdamW betas=(0.9,0.95) weight_decay=0.01",
           "max_length": 2048, "threads": 1, "dtype": "float32", "device": "cpu"}
    write_json(args.output / "preflight.json",
               {"config": cfg, "train_rows": len(train_rows), "heldout_rows": len(heldout_rows),
                "train_action_counts": splits["train"]["actions"], "eval_action_counts": splits["eval"]["actions"],
                "max_train_tokens": max(len(x["input_ids"]) for x in encoded),
                "action_code_token_ids": action_token_ids,
                "train_eos_masks_pass": all(x["input_ids"][-1] == 5 and x["labels"][-1] == 5 for x in encoded),
                "numerical_runtime": numerical_runtime,
                "baseline_w0": "source-faithful, not native"})
    # Select deterministic disjoint heldout probes before training and preserve originals.
    samples = objective.probes(heldout_rows, 18)
    if len(samples) < 9:
        raise ValueError("too few precommitted heldout action probes")
    write_json(args.output / "precommitted_probes.json",
               [{"id": r["id"], "family": r["family"], "target": r["action"], "target_code": r["completion"]} for r in samples])
    untuned = first_pass(base, tokenizer, adapter, samples, "untuned", args.output)
    if time.monotonic() - started > args.deadline - 120:
        raise TimeoutError("insufficient remaining time for a real optimizer update")
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                 lora_dropout=0.05, bias="none", target_modules=selected["lora_targets"]))
    adapter.set_training_mode(model)
    model_contract = adapter.validate_trainable(model, selected)
    if model_contract["trainable_parameters"] != 4739072 or model_contract["lora_modules"] != 182:
        raise ValueError("reviewed LoRA whitelist diverged")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.0005, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    frozen_before = shared.fingerprint(model, False)
    adapter_before = shared.fingerprint(model, True)
    step, cursor, history = 0, 0, []
    last_checkpoint = None
    write_json(args.output / "model_contract.json", model_contract)
    while step < args.steps and not STOP and time.monotonic() - started < args.deadline - 90:
        row = encoded[order[cursor % len(order)]]
        started_step = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = torch.tensor([row["input_ids"]], dtype=torch.long)
        labels = torch.tensor([row["labels"]], dtype=torch.long)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
        if not torch.isfinite(loss).item():
            raise ValueError("nonfinite optimizer loss")
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients) or not any(torch.count_nonzero(g).item() for g in gradients):
            raise ValueError("zero, absent, or nonfinite LoRA gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        optimizer.step()
        scheduler.step()
        step += 1
        cursor += 1
        event = {"step": step, "id": row["id"], "loss": loss.item(), "grad_norm": norm,
                 "loss_tokens": row["target_tokens"], "sequence_tokens": len(row["input_ids"]),
                 "seconds": time.monotonic() - started_step}
        history.append(event)
        with (args.output / "progress.jsonl").open("a") as fp:
            fp.write(contract.canonical(event) + "\n")
        print(contract.canonical({"optimizer": event}), flush=True)
        if step % 12 == 0 or step == args.steps:
            last_checkpoint = shared.save_checkpoint(args.output, model, tokenizer,
                     shared.state_for(optimizer, scheduler, step, cursor, cfg, frozen_before, history), numerical_runtime)
    if step < 1:
        raise RuntimeError("no real optimizer update")
    if last_checkpoint is None or step % 12:
        last_checkpoint = shared.save_checkpoint(args.output, model, tokenizer,
                 shared.state_for(optimizer, scheduler, step, cursor, cfg, frozen_before, history), numerical_runtime)
    frozen_after, adapter_after = shared.fingerprint(model, False), shared.fingerprint(model, True)
    if frozen_after != frozen_before or adapter_after == adapter_before:
        raise ValueError("frozen base altered or LoRA unchanged")
    trained = None
    if time.monotonic() - started < args.deadline - 60:
        trained = first_pass(model, tokenizer, adapter, samples, "trained", args.output)
    write_json(args.output / "summary.json",
               {"actual_training": True, "status": "complete" if step == args.steps else "partial",
                "steps": step, "target_steps": args.steps, "checkpoint": last_checkpoint.name,
                "checkpoint_inventory_sha256": sha256_file(last_checkpoint / "inventory.json"),
                "frozen_base_equal": True, "adapter_changed": True,
                "frozen_base_sha256": frozen_after, "adapter_before_sha256": adapter_before,
                "adapter_after_sha256": adapter_after, "train": splits["train"], "eval": splits["eval"],
                "untuned_static_first_pass": untuned, "trained_static_first_pass": trained,
                "native_execution": False, "full_art_program_evaluation": False,
                "novel_only_probes": True, "raw_support_manifest": generated["support_manifest"],
                "sealed_final_opened": False, "wall_seconds": time.monotonic()-started,
                "cpu_seconds": time.process_time(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    return 0 if step == args.steps else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--w0", type=Path, required=True)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--deadline", type=int, default=1750)
    a = p.parse_args()
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    # sys.exit(0) raises SystemExit; do not label a completed campaign failed.
    # A nonzero main() result still exits nonzero, preserving partial-run CI failure.
    try:
        exit_code = main(a)
    except BaseException as exc:
        if a.output.exists():
            write_json(a.output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                      "status": "FAILED_OR_INTERRUPTED",
                      "not_training_success": True})
        raise
    raise SystemExit(exit_code)
