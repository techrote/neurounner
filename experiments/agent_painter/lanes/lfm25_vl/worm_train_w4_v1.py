#!/usr/bin/env python3
"""WORM-001 LFM W4: distinct-prompt six-action curriculum CPU LoRA.

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
SYSTEM = ("You are a tiny procedural CyberSand worm steering brain. "
          "Return exactly ONE uppercase token from the listed legal actions. "
          "Do not describe the drawing. The host controls materials, obstacles, limits and Stop.")
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
    are copied into W4's official-processor framing; direct prompt overlap is
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
                raise ValueError("unqualified W4 teacher action")
            rows.append({"id": item["id"], "family": observation["family"],
                         "seed_group": item["world_group"],
                         "origin": item["observation_source"] if "observation_source" in item else "W0-observed",
                         "prompt": make_prompt(observation),
                         "completion": answer, "legal": observation["legal"]})
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate W4 source row IDs")
        return rows
    train = convert(raw_train + adverse_train, "WORM_TRAIN")
    dev = convert(raw_dev + adverse_dev, "WORM_DEV")
    pt, pd = {r["prompt"] for r in train}, {r["prompt"] for r in dev}
    if pt & pd:
        raise ValueError("W4 LFM-framed literal prompt overlap")
    if len(pt) != len(train) or len(pd) != len(dev):
        raise ValueError("W4 duplicate source prompts")
    return {"WORM_TRAIN": train, "WORM_DEV": dev}, {
        "support_source": SUPPORT_SHA, "support_blob": SUPPORT_BLOB,
        "support_version": support.VERSION,
        "W0_source": WORM_SHA, "W0_blob": WORM_BLOB,
        "support_split_audit": split_audit,
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
        raise ValueError("unexpected W4 development action coverage")
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
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--deadline", type=int, default=1600)
    args = parser.parse_args()
    if not 1 <= args.steps <= 180 or not 180 <= args.deadline <= 2700:
        raise ValueError("WORM W4 pilot bounded to 1–180 steps and 180–2700 seconds")
    if args.output.exists():
        raise ValueError("immutable output lease already exists")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    source = git("rev-parse", "HEAD", cwd=ROOT.parents[1])
    write_json(args.output / "lease.json", {"experiment": "lfm-worm-w4-support-v2-fresh-v1",
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
                "objective": "local-steering-oracle-plus-adverse-balanced",
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
    config = {"objective": "lfm-worm-action-selection-v4-world-majority-fresh",
              "source_sha": source, "profile": "lfm25_vl",
              "model": selected, "numerics": numerics,
              "model_files": json.loads((args.model / "download_receipt.json").read_text())["files"],
              "corpus_manifest": manifest, "system": SYSTEM,
              "recipe": {"rank": 8, "alpha": 16, "dropout": 0.05, "lr": 0.0005,
                         "threads": 1, "seed": 612568, "parent": "pinned-official-base",
                         "support": "worm-lfm-local-decisions-v2",
                         "optimizer": "AdamW", "sampler": "family-balanced-world-majority-v1-180-distinct-rows", "betas": [0.9, 0.95],
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
    # W4 intentionally replaces W3's artificial equal-class schedule.
    # Exact W0-observed world labels dominate; counterfactual local cases are
    # retained as 12 distinct adverse rows per family for dead-end recovery.
    # One distinct anchor per action/family preserves all six legal actions.
    # This is a NEW fresh recipe, not optimizer resumption from W3.
    chooser = random.Random(1612569)
    by_family = {}
    for family in FAMILIES:
        group = [(i, row) for i, row in enumerate(rows["WORM_TRAIN"])
                 if row["family"] == family]
        anchors = []
        for action in ACTIONS:
            candidates = [i for i, r in group if r["completion"] == action]
            if not candidates:
                raise ValueError("missing legal W4 family/action supervision")
            # Prefer genuine W0 world observations where available.
            candidates.sort(key=lambda i: (rows["WORM_TRAIN"][i]["origin"] != "W0-observed", i))
            anchors.append(candidates[0])
        # Do not shadow the pinned official LFM profile used by evaluation.
        selected_indices = set(anchors)
        world = [i for i, r in group
                 if r["origin"] == "W0-observed" and i not in selected_indices]
        adverse = [i for i, r in group
                   if r["origin"] != "W0-observed" and i not in selected_indices]
        chooser.shuffle(world)
        chooser.shuffle(adverse)
        # 6 mandatory anchors + 42 natural local-action examples + 12
        # adverse recovery examples = 60 per family, 180 distinct updates.
        sequence = anchors + world[:42] + adverse[:12]
        if len(sequence) != 60 or len(set(sequence)) != 60:
            raise ValueError("W4 sampler could not build 60 distinct rows/family")
        chooser.shuffle(sequence)
        by_family[family] = sequence
    schedule = [by_family[family][cycle]
                for cycle in range(60) for family in FAMILIES]
    if len(schedule) != 180 or args.steps != 180:
        raise ValueError("W4 pinned 180-step sampler budget mismatch")
    sample_distribution = {
        family: {
            "action_counts": {action: sum(
                rows["WORM_TRAIN"][i]["completion"] == action
                for i in by_family[family]) for action in ACTIONS},
            "origin_counts": {
                origin: sum(rows["WORM_TRAIN"][i]["origin"] == origin
                            for i in by_family[family])
                for origin in ("W0-observed", "counterfactual-no-world-or-native-receipt")
            }
        } for family in FAMILIES
    }
    write_json(args.output / "sampler-receipt.json", {
        "version": "lfm-worm-w4-family-world-majority-v1",
        "deterministic_seed": 1612569,
        "global_updates": 180,
        "distinct_source_sample_ids": len(set(schedule)),
        "per_family": sample_distribution})
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
        if step % 45 == 0 or step == args.steps:
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
        "sampler_version": "lfm-worm-w4-family-world-majority-v1",
        "sampler_receipt": sample_distribution,
        "synthetic_novel_input_test_only": True,
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
