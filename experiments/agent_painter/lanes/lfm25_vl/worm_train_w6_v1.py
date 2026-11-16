#!/usr/bin/env python3
"""WORM-001 LFM W6: explicit TRAIN-only anti-right4-loop CPU LoRA.

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

WORM_SHA = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
WORM_BLOB = "8979b6f0296e81051194bacc907992d38e006605"
SUPPORT_SHA = "cc27b3206cb61f98432fa95b799c9b9ff508c2bc"
SUPPORT_BLOB = "f205a3a9f33632aeaafa2daa8ce25b2484bc6032"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
SAMPLER_VERSION = "lfm-worm-w6-anti-right4-family-v1"
SYSTEM = ("You are a tiny procedural CyberSand worm steering brain. "
          "Return exactly ONE uppercase token from the listed legal actions. "
          "After four consecutive RIGHT actions avoid RIGHT if another move is legal. "
          "Use local visited marks, obstacles and the family. The host owns material and Stop.")
TRAIN_SEEDS = tuple(range(1101, 1113))
DEV_SEEDS = tuple(range(2101, 2107))
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


def support_source(path, worm):
    root = Path(path).resolve()
    source = root / "curriculum.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("missing frozen LFM support curriculum")
    if git("rev-parse", "HEAD", cwd=root.parents[3]) != SUPPORT_SHA:
        raise ValueError("unqualified LFM support checkout")
    if git("hash-object", str(source)) != SUPPORT_BLOB:
        raise ValueError("LFM support bytes mismatch")
    spec = importlib.util.spec_from_file_location("lfm_worm_support_v2", source)
    if spec is None or spec.loader is None:
        raise ValueError("unloadable support source")
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    if support.VERSION != "worm-lfm-local-decisions-v2" or support.worm is not worm:
        raise ValueError("support curriculum/host contract mismatch")
    if support.COMMON_SOURCE_SHA != WORM_SHA:
        raise ValueError("support version tied to another W0")
    return support


def dataset(support):
    """Versioned support W0 observations + separately identified local adverse cases.

    DEVELOPMENT is never encoded into an optimizer batch. Full source observations
    are copied into W6's official-processor framing; direct prompt overlap is
    rechecked after model-specific formatting, not trusted merely by world seed.
    """
    raw_train, raw_dev, split_audit = support.split_distinct_actions()
    adverse_train = support.adverse_rows("train")
    adverse_dev = support.adverse_rows("development")
    support.checked_splits(raw_train + adverse_train, raw_dev + adverse_dev)
    def convert(items, split):
        rows = []
        for item in items:
            observation = item["observation"]
            answer = item["target_action"]
            if answer not in observation["legal"] or answer not in ACTIONS:
                raise ValueError("unqualified W6 teacher action")
            rows.append({"id": item["id"], "family": observation["family"],
                         "seed_group": item["world_group"],
                         "origin": item["observation_source"] if "observation_source" in item else "W0-observed",
                         "prompt": make_prompt(observation),
                         "completion": answer, "legal": observation["legal"]})
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate W6 source row IDs")
        return rows
    train = convert(raw_train + adverse_train, "WORM_TRAIN")
    dev = convert(raw_dev + adverse_dev, "WORM_DEV")
    # W6 counterfactual TRAIN additions: legal mask and local pixels originate
    # from real W0 TRAIN only. The last-four-RIGHT history is deliberately
    # synthetic, NOT a physically replayed world. No DEV/FINAL teacher is read.
    excluded = {r["prompt"] for r in train + dev}
    preferences = {
        "greebles": ("BRANCH", "BRANCH", "BRANCH", "LEFT", "LEFT",
                     "STRAIGHT", "RETRACE"),
        "pipework": ("STRAIGHT", "STRAIGHT", "STRAIGHT", "STRAIGHT",
                     "BRANCH", "LEFT", "RETRACE"),
        "architecture": ("STRAIGHT", "STRAIGHT", "LEFT", "LEFT",
                         "LEFT", "BRANCH", "RETRACE"),
    }
    anti = []
    for row in list(train):
        if row["origin"] != "W0-observed":
            continue
        state = json.loads(row["prompt"])
        candidates = [a for a in preferences[row["family"]]
                      if a in state["legal"] and a != "RIGHT"]
        if not candidates:
            continue
        state["last_actions"] = ["RIGHT"] * 4
        state["tick"] = max(4, int(state["tick"]))
        prompt = canonical(state)
        if prompt in excluded:
            continue
        excluded.add(prompt)
        selection = int(hashlib.sha256(
            (row["id"] + "|W6-anti-right4").encode()
        ).hexdigest(), 16) % len(candidates)
        anti.append({
            "id": "W6-anti-right4:" + row["id"],
            "family": row["family"], "seed_group": row["seed_group"],
            "origin": "counterfactual-four-RIGHT-history-not-W0-replay",
            "prompt": prompt, "completion": candidates[selection],
            "legal": state["legal"],
        })
    train.extend(anti)
    anti_counts = {family: sum(r["family"] == family for r in anti)
                   for family in FAMILIES}
    if min(anti_counts.values()) < 20:
        raise ValueError("insufficient distinct W6 TRAIN loop counterfactuals")
    pt, pd = {r["prompt"] for r in train}, {r["prompt"] for r in dev}
    if pt & pd:
        raise ValueError("W6 LFM-framed literal prompt overlap")
    if len(pt) != len(train) or len(pd) != len(dev):
        raise ValueError("W6 duplicate source prompts")
    return {"WORM_TRAIN": train, "WORM_DEV": dev}, {
        "support_source": SUPPORT_SHA, "support_blob": SUPPORT_BLOB,
        "support_version": support.VERSION,
        "W0_source": WORM_SHA, "W0_blob": WORM_BLOB,
        "support_split_audit": split_audit,
        "anti_right4_train_rows_by_family": anti_counts,
        "anti_right4_is_synthetic_not_W0_or_native": True,
        "world_train_rows": len(raw_train), "world_dev_rows": len(raw_dev),
        "adverse_train_rows": len(adverse_train), "adverse_dev_rows": len(adverse_dev),
        "literal_prompt_overlap": 0, "prompt_unique_per_split": True,
        "TRAIN_world_seed_groups": list(TRAIN_SEEDS),
        "DEV_world_seed_groups": list(DEV_SEEDS),
    }


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


def evaluation(model, tokenizer, selected, rows, *, limit=18):
    # Frozen first original row per family and each of SIX actions, including STOP.
    # Because dataset() checks literal overlap, every scored probe is novel
    # relative to TRAIN; report motif provenance separately in receipts.
    chosen = [next(r for r in rows
                   if r["family"] == family and r["completion"] == action)
              for family in FAMILIES for action in ACTIONS]
    if len(chosen) != limit:
        raise ValueError("unexpected W6 development action coverage")
    return [{"id": row["id"], "family": row["family"],
             "expected": row["completion"], "legal": row["legal"],
             "seed_group": row["seed_group"], "origin": row["origin"],
             **generate(model, tokenizer, selected, row["prompt"], row["legal"])}
            for row in chosen]


def rollout(model, tokenizer, selected, support, family, seed, max_steps=32):
    """One real first pass on each current host observation, no teacher/hiding.

    W0 callback sees host-deepcopied observations; return the literal decoded
    answer, including mistakes. Host alone decides legality, STOP and material.
    """
    original = []
    def choose(observation):
        first = generate(model, tokenizer, selected,
                         make_prompt(observation), observation["legal"],
                         max_tokens=10)
        original.append({"observation": observation, **first})
        return first["raw_text"]
    record = support.rollout(seed, family, max_steps, choose)
    if len(record["events"]) != len(original):
        raise ValueError("host callback/raw-receipt count mismatch")
    return record, original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worm-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--deadline", type=int, default=1600)
    args = parser.parse_args()
    if not 1 <= args.steps <= 120 or not 180 <= args.deadline <= 2700:
        raise ValueError("WORM W6 pilot bounded to 1–180 steps and 180–2700 seconds")
    if args.output.exists():
        raise ValueError("immutable output lease already exists")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    source = git("rev-parse", "HEAD", cwd=ROOT.parents[1])
    write_json(args.output / "lease.json", {"experiment": "lfm-worm-w6-anti-right4-support-v2-fresh-v1",
               "source_sha": source, "worm_sha": WORM_SHA,
               "support_sha": SUPPORT_SHA,
               "github_run_id": os.environ.get("GITHUB_RUN_ID"),
               "requested_steps": args.steps, "created_unix": time.time()})
    worm = worm_source(args.worm_root)
    support = support_source(args.support_root, worm)
    rows, custody = dataset(support)
    for split, records in rows.items():
        (args.output / (split.lower() + ".jsonl")).write_text(
            "".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    manifest = {"schema": "cybersand.worm.lfm-support-v2-train/1",
                "objective": "local-steering-with-TRAIN-derived-anti-right4-counterfactuals",
                **custody,
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
    random.seed(612571)
    torch.manual_seed(612571)
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
    config = {"objective": "lfm-worm-action-selection-v6-anti-right4-fresh",
              "source_sha": source, "profile": "lfm25_vl",
              "model": selected, "numerics": numerics,
              "model_files": json.loads((args.model / "download_receipt.json").read_text())["files"],
              "corpus_manifest": manifest, "system": SYSTEM,
              "recipe": {"rank": 8, "alpha": 16, "dropout": 0.05, "lr": 0.0005,
                         "threads": 1, "seed": 612571, "parent": "pinned-official-base",
                         "support": "worm-lfm-local-decisions-v2",
                         "optimizer": "AdamW", "sampler": "family-half-original-half-anti-right4-120-distinct-v1", "betas": [0.9, 0.95],
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
    # 20 original support/action-adverse rows plus 20 explicit anti-loop
    # counterfactuals per family. This is NEW pinned-parent training, not resume.
    chooser = random.Random(1612571)
    quotas = {"STRAIGHT": 3, "LEFT": 4, "RIGHT": 3,
              "BRANCH": 4, "RETRACE": 4, "STOP": 2}
    if sum(quotas.values()) != 20 or set(quotas) != set(ACTIONS):
        raise ValueError("W6 original TRAIN quota contract changed")
    by_family = {}
    sample_distribution = {}
    for family in FAMILIES:
        original = []
        for action in ACTIONS:
            matching = [i for i, row in enumerate(rows["WORM_TRAIN"])
                        if row["family"] == family
                        and row["completion"] == action
                        and row["origin"] != "counterfactual-four-RIGHT-history-not-W0-replay"]
            observed = [i for i in matching
                        if rows["WORM_TRAIN"][i]["origin"] == "W0-observed"]
            synthetic = [i for i in matching
                         if rows["WORM_TRAIN"][i]["origin"] != "W0-observed"]
            chooser.shuffle(observed)
            chooser.shuffle(synthetic)
            selected_rows = (observed + synthetic)[:quotas[action]]
            if len(selected_rows) != quotas[action]:
                raise ValueError(f"insufficient W6 original {family}/{action} TRAIN")
            original.extend(selected_rows)
        anti = [i for i, row in enumerate(rows["WORM_TRAIN"])
                if row["family"] == family
                and row["origin"] == "counterfactual-four-RIGHT-history-not-W0-replay"]
        chooser.shuffle(anti)
        anti = anti[:20]
        if len(anti) != 20 or len(original) != 20:
            raise ValueError("W6 per-family 20 original / 20 synthetic failed")
        anti_actions = {rows["WORM_TRAIN"][i]["completion"] for i in anti}
        if len(anti_actions) < 2 or "RIGHT" in anti_actions:
            raise ValueError("W6 anti-right selection has insufficient alternative actions")
        chosen = original + anti
        chooser.shuffle(chosen)
        if len(chosen) != len(set(chosen)):
            raise ValueError("W6 repeated source row within family")
        by_family[family] = chosen
        sample_distribution[family] = {
            "action_counts": {
                action: sum(rows["WORM_TRAIN"][i]["completion"] == action
                            for i in chosen) for action in ACTIONS},
            "origin_counts": {
                origin: sum(rows["WORM_TRAIN"][i]["origin"] == origin
                            for i in chosen)
                for origin in ("W0-observed",
                    "counterfactual-no-world-or-native-receipt",
                    "counterfactual-four-RIGHT-history-not-W0-replay")},
            "original_count": 20,
            "anti_right4_count": 20,
            "anti_right4_distinct_targets": sorted(anti_actions)}
    schedule = [by_family[family][cycle]
                for cycle in range(40) for family in FAMILIES]
    if len(schedule) != 120 or args.steps != 120 or len(set(schedule)) != 120:
        raise ValueError("W6 exact 120-step source-distinct schedule mismatch")
    write_json(args.output / "sampler-receipt.json", {
        "version": SAMPLER_VERSION, "deterministic_seed": 1612571,
        "global_updates": 120, "distinct_source_sample_ids": len(set(schedule)),
        "per_family": sample_distribution,
        "counterfactual_not_host_executed": True})
    history = []
    last_checkpoint = None
    for step in range(1, args.steps + 1):
        if STOP_REQUESTED or time.monotonic() - started > args.deadline - 120:
            break
        item = encoded[schedule[step - 1]]
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
        if step % 30 == 0 or step == args.steps:
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
    gallery_metrics = []
    for family in FAMILIES:
        for seed in (2101, 2102):
            record, raw = rollout(model, tokenizer, selected, support, family,
                                  seed=seed, max_steps=48)
            label = f"{family}-model-{seed}"
            (gallery / f"{label}.png").write_bytes(worm.png_bytes(record))
            write_json(gallery / f"{label}.json", record)
            write_json(gallery / f"{label}-raw-first-passes.json", raw)
            gallery_metrics.append({"family": family, "seed": seed,
                                    "policy": "trained-lfm-first-pass",
                                    "metrics": support.statistics(record)})
            for mode in ("random", "wall"):
                control = worm.run(seed, mode=mode, family=family, steps=48)
                worm.write_result(control, gallery)
                gallery_metrics.append({"family": family, "seed": seed,
                                        "policy": mode,
                                        "metrics": support.statistics(control)})
            control = support.rollout(seed, family, 48, support.oracle)
            name = f"{family}-oracle-{seed}"
            (gallery / f"{name}.png").write_bytes(worm.png_bytes(control))
            write_json(gallery / f"{name}.json", control)
            gallery_metrics.append({"family": family, "seed": seed,
                                    "policy": "untrained-oracle",
                                    "metrics": support.statistics(control)})
    write_json(args.output / "gallery-metrics.json", gallery_metrics)
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
        "trained_exact_teacher_matches": sum(x["raw_text"] == x["expected"] for x in trained),
        "trained_distinct_valid_choices": sorted({x["raw_text"] for x in trained if x["valid_first_pass"]}),
        "balanced_teacher_schedule": False,
        "sampler_version": SAMPLER_VERSION,
        "sampler_receipt": sample_distribution,
        "synthetic_novel_input_test_only": True,
        "counterfactual_four_RIGHT_history_tested": True,
        "support_source": SUPPORT_SHA,
        "gallery_metrics": gallery_metrics,
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
