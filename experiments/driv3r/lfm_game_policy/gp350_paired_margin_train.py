#!/usr/bin/env python3
"""GP350-006: bounded real full-LFM350 LoRA with paired geometry margins.

Synthetic three-way forward-2-cell counterfactuals on W0-observed states;
explicit ray feature is a *derived observation*, not native visual grounding.
This is a FRESH LoRA from pinned full parent, NOT GP350-005 resumed. The
objective requires each matched marker variant to prefer its distinct legal
steering action, rather than allowing a motif-wide constant action.

Research-only. Does not edit, instantiate or write to CyberSand game runtime.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import time

import lfm350 as c
import gp350_triad as old
import gp350_geometry_contrast_train as geom

SCHEMA = "driv3r.lfm350.gp350006-paired-margin/1"
STEPS = 96
CHECKPOINTS = (32, 64, 96)
LR = 3.0e-4
MARGIN = 0.75
PAIR_MARGIN = 1.5
CE_WEIGHT = 0.35
PAIR_WEIGHT = 0.65
LIMIT = 1950
STOP_REQUESTED = False


def stopped(_signal, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def ray2(obs):
    direction = ((1, 0), (0, 1), (-1, 0), (0, -1))[obs["heading"] % 4]
    x, y = 2 + 2 * direction[0], 2 + 2 * direction[1]
    return obs["local_5x5"][y][x]


def rich_prompt(obs):
    original = c.prompt_for(obs)
    c.check(original.endswith("\nchoice="), "action boundary changed")
    # Action choice remains last. This explicit local derived feature is a
    # separately controlled representation ablation, NOT screenshot vision.
    return original[:-len("choice=")] + "forward_two_cell=" + ray2(obs) + "\nchoice="


def rich_score(model, tokenizer, ids, obs):
    import torch
    input_ids = tokenizer(rich_prompt(obs), return_tensors="pt",
                          add_special_tokens=False)["input_ids"]
    c.check(4 < input_ids.shape[1] <= 256, "rich prompt token budget changed")
    logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    scores = logits.index_select(0, torch.tensor(ids, dtype=torch.long))
    c.check(torch.isfinite(scores).all().item(), "nonfinite steering logits")
    return scores


def contrasting(triples):
    rows = [t for t in triples if len({r["target"] for r in t}) == 3]
    c.check(rows and all(len(t) == 3 for t in rows), "bad counterfactual triples")
    for triple in rows:
        parent = triple[0]["observation"]
        c.check([ray2(r["observation"]) for r in triple] == [".", "#", "+"],
                "forward marker encoding shifted")
        c.check(all(r["target"] in r["observation"]["legal"] for r in triple),
                "synthetic target not host legal")
        c.check(all(r["observation"]["legal"] == parent["legal"] for r in triple),
                "matched legal menu drift")
    return rows


def choose_probe(triples, budget=36):
    """Unique held-out seed/family/tick observations, balanced across motifs."""
    selected = []
    groups = collections.defaultdict(list)
    for triple in contrasting(triples):
        r = triple[0]
        groups[r["observation"]["family"]].append(triple)
    per = budget // len(c.FAMILIES)
    for family in c.FAMILIES:
        group = groups[family]
        c.check(len(group) >= per, "heldout family coverage lost")
        # Spread evaluation among independent seeds, headings and time bands
        # instead of taking 36 copies of one starting 5x5 context.
        seen = set()
        unique = []
        for triple in sorted(group, key=lambda t: (
                t[0]["observation"]["tick"] % 5, t[0]["seed"],
                t[0]["observation"]["tick"])):
            row = triple[0]
            obs = row["observation"]
            key = (row["seed"], obs["heading"], obs["tick"],
                   tuple(obs["local_5x5"]))
            if key not in seen:
                unique.append(triple)
                seen.add(key)
        c.check(len(unique) >= per, "insufficient distinct heldout contexts")
        for i in range(per):
            selected.append(unique[(i * len(unique)) // per])
    c.check(len(selected) == budget, "probe budget drift")
    return selected


def probe(model, tokenizer, ids, triples):
    import torch
    model.eval()
    cases = []
    with torch.inference_mode():
        for triple in triples:
            choices = []
            for row in triple:
                obs = row["observation"]
                z = rich_score(model, tokenizer, ids, obs)
                raw = c.ACTIONS[int(z.argmax().item())]
                legal = max(obs["legal"], key=lambda a: float(z[c.ACTIONS.index(a)]))
                choices.append({
                    "ray2": ray2(obs), "family": obs["family"],
                    "target": row["target"], "raw": raw, "legal": legal,
                    "match": legal == row["target"], "raw_legal": raw in obs["legal"],
                    "six_logits": [float(v) for v in z],
                })
            cases.append({
                "world": triple[0]["seed"], "tick": triple[0]["observation"]["tick"],
                "heading": triple[0]["observation"]["heading"],
                "local_context": triple[0]["observation"]["local_5x5"],
                "synthetic_two_cell_counterfactual": True,
                "choices": choices,
                "three_distinct_legal": len({r["legal"] for r in choices}) == 3,
                "fully_matched": all(r["match"] for r in choices),
            })
    return {
        "n": len(cases), "decisions": 3 * len(cases),
        "three_distinct": sum(x["three_distinct_legal"] for x in cases),
        "fully_matched": sum(x["fully_matched"] for x in cases),
        "teacher_matches": sum(r["match"] for x in cases for r in x["choices"]),
        "raw_illegal": sum(not r["raw_legal"] for x in cases for r in x["choices"]),
        "cases": cases,
    }


def paired_loss(scores, triple):
    import torch
    import torch.nn.functional as F
    c.check(len(scores) == len(triple) == 3, "paired objective needs three scores")
    ce = []
    for z, row in zip(scores, triple):
        legal = list(row["observation"]["legal"])
        index = torch.tensor([c.ACTIONS.index(a) for a in legal], dtype=torch.long)
        masked = z.index_select(0, index)
        ce.append(F.cross_entropy(masked.unsqueeze(0),
                                  torch.tensor([legal.index(row["target"])])))
    pairs = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a = c.ACTIONS.index(triple[i]["target"])
        b = c.ACTIONS.index(triple[j]["target"])
        d_i = scores[i][a] - scores[i][b]
        d_j = scores[j][a] - scores[j][b]
        # The paired relative logit gap *must switch sign* when one and only
        # one observed marker changes; penalize both sides and the gap change.
        pairs.append((F.softplus(MARGIN - d_i) +
                      F.softplus(MARGIN + d_j) +
                      F.softplus(PAIR_MARGIN - (d_i - d_j))) / 3.0)
    ce_loss = torch.stack(ce).mean()
    margin_loss = torch.stack(pairs).mean()
    total = CE_WEIGHT * ce_loss + PAIR_WEIGHT * margin_loss
    c.check(torch.isfinite(total).item(), "nonfinite paired objective")
    return total, float(ce_loss.detach()), float(margin_loss.detach())


def self_test(w0_path):
    w0 = c.load_w0(w0_path)
    tr = contrasting(geom.geometry_triples(w0, geom.TRAIN_SEEDS))
    dev = contrasting(geom.geometry_triples(w0, geom.DEV_SEEDS))
    c.check(len(tr) >= 80 and len(dev) >= 20, "insufficient matched training data")
    c.check({r[0]["seed"] for r in tr}.isdisjoint({r[0]["seed"] for r in dev}),
            "train and heldout worlds overlap")
    selected = choose_probe(dev)
    c.check(len(selected) == 36, "heldout probe short")
    for triple in tr + selected:
        texts = [rich_prompt(x["observation"]) for x in triple]
        c.check(len(set(texts)) == 3 and
                all(text.endswith("\nchoice=") for text in texts),
                "matched counterfactual prompts collapsed")
    print(c.canonical({
        "GP350006_SELFTEST": "PASS", "train_contrasts": len(tr),
        "heldout_contrasts": len(dev), "probe": len(selected),
        "disjoint_world_seeds": True, "observations_explicitly_synthetic": True,
        "derived_raycast_feature": "forward_two_cell",
    }), flush=True)


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import HfApi, snapshot_download

    torch.set_num_threads(2)
    torch.manual_seed(3506006)
    random.seed(3506006)
    started = time.monotonic()
    out = args.out
    c.check(not out.exists(), "immutable research lease reused")
    out.mkdir(parents=True)
    c.save_json(out / "lease.json", {
        "schema": SCHEMA, "source_sha": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"), "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "model": c.MODEL, "revision": c.REVISION, "fresh_parent": True,
        "steps": STEPS, "backward_passes": STEPS,
        "native": False, "draft_hold": True, "optimizer_lease": "gp350-006",
        "cross_entropy_only": False, "learned_vision": False,
    })
    w0 = c.load_w0(args.w0)
    train = contrasting(geom.geometry_triples(w0, geom.TRAIN_SEEDS))
    dev = contrasting(geom.geometry_triples(w0, geom.DEV_SEEDS))
    probes = choose_probe(dev)
    c.check({r[0]["seed"] for r in train}.isdisjoint({r[0]["seed"] for r in dev}),
            "world data contamination")
    raw = ("\n".join(c.canonical(row) for row in train) + "\n").encode()
    (out / "train-triples.jsonl").write_bytes(raw)
    c.save_json(out / "curriculum.json", {
        "train_triples": len(train), "dev_triples": len(dev),
        "heldout_probe_triples": len(probes),
        "train_seed_worlds": list(geom.TRAIN_SEEDS),
        "dev_seed_worlds": list(geom.DEV_SEEDS),
        "training_sha256": c.sha_bytes(raw),
        "all_three_way_target_contrasts": True,
        "two_cell_feature_is_derived_not_native_vision": True,
        "teacher_labels_are_synthetic": True,
    })
    info = HfApi().model_info(c.MODEL, revision=c.REVISION)
    c.check(info.sha == c.REVISION, "Hugging Face revision drift")
    snapshot_download(c.MODEL, revision=c.REVISION, local_dir=args.cache,
        max_workers=2, allow_patterns=[
            "*.safetensors", "*.safetensors.index.json", "config.json",
            "generation_config.json", "tokenizer*", "special_tokens_map.json",
            "added_tokens.json", "vocab.*", "merges.txt", "README.md", "LICENSE*"])
    inventory = {str(p.relative_to(args.cache)): c.sha(p)
                 for p in sorted(args.cache.rglob("*"))
                 if p.is_file() and ".cache" not in p.parts}
    c.check(any(k.endswith(".safetensors") for k in inventory), "pinned weights absent")
    c.save_json(out / "model-download.json", {"model": c.MODEL,
                 "revision": info.sha, "sha256": inventory})
    tokenizer = AutoTokenizer.from_pretrained(args.cache, local_files_only=True,
                                             trust_remote_code=False)
    ids = c.id_map(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        args.cache, local_files_only=True, trust_remote_code=False, dtype=torch.float32)
    c.check(base.config.model_type == "lfm2" and base.config.num_hidden_layers == 16,
            "unexpected full LFM350 architecture")
    targets = [n for n, m in base.named_modules()
               if isinstance(m, torch.nn.Linear) and ".self_attn." in n
               and n.endswith((".q_proj", ".v_proj"))]
    c.check(len(targets) == 12, "full attention q/v inventory drift")
    base.requires_grad_(False)
    base.eval()
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
        bias="none", target_modules=targets))
    trainable = [p for p in model.parameters() if p.requires_grad]
    c.check(len(trainable) == 24 and all("lora_" in n for n, p in
            model.named_parameters() if p.requires_grad), "base accidentally trainable")
    frozen = c.base_fingerprint(model)
    adapter_before = {n: c.sha_bytes(p.detach().cpu().numpy().tobytes())
                      for n, p in model.named_parameters() if "lora_" in n}
    c.save_json(out / "preflight.json", {
        "token_ids": ids, "trainable_params": sum(p.numel() for p in trainable),
        "targets": targets, "rank": 8, "alpha": 16,
        "loss": "0.35 masked CE + 0.65 pairwise relative logit softplus margin",
        "margin": MARGIN, "pair_margin": PAIR_MARGIN, "lr": LR,
        "optimizer": "AdamW", "torch": torch.__version__,
        "base_model": "full pinned LiquidAI LFM2.5-350M", "fp32": True,
    })
    stock_probe = probe(model, tokenizer, ids, probes)
    c.save_json(out / "probe-stock.json", stock_probe)
    # Override scoring only for THIS isolated research process. All existing
    # W0 host and replay functions continue to own legal action application.
    previous_score = c.score
    c.score = rich_score
    stock_rollouts = old.rollout(w0, model, tokenizer, ids, out, "stock")
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    by_family = collections.defaultdict(list)
    for triple in train:
        by_family[triple[0]["observation"]["family"]].append(triple)
    rng = random.Random(3506007)
    for family in c.FAMILIES:
        c.check(len(by_family[family]) >= 24, "too few family contrasts")
        rng.shuffle(by_family[family])
    counts = collections.Counter()
    progress = []
    for step in range(STEPS):
        if STOP_REQUESTED or time.monotonic() - started > args.deadline - 480:
            break
        family = c.FAMILIES[step % len(c.FAMILIES)]
        group = by_family[family]
        triple = group[counts[family] % len(group)]
        counts[family] += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = [rich_score(model, tokenizer, ids, row["observation"])
                  for row in triple]
        loss, ce_val, paired_val = paired_loss(scores, triple)
        loss.backward()
        grads = [p.grad for p in trainable if p.grad is not None]
        c.check(grads and any(torch.count_nonzero(g).item() for g in grads),
                "zero LoRA gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        c.check(math.isfinite(norm), "bad optimizer gradient")
        optimizer.step()
        receipt = {"step": step + 1, "family": family, "seed": triple[0]["seed"],
                   "target_triplet": [r["target"] for r in triple],
                   "loss": float(loss.detach()), "ce": ce_val,
                   "pair_margin": paired_val, "grad_norm": norm,
                   "model_gradient_forwards": 3, "real_backward_passes": 1,
                   "actual_adamw_steps": 1}
        progress.append(receipt)
        with (out / "optimizer.jsonl").open("a") as file:
            file.write(c.canonical(receipt) + "\n")
        print(c.canonical({"GP350006_ADAMW": receipt}), flush=True)
        if (step + 1) in CHECKPOINTS:
            model.save_pretrained(out / f"adapter-step-{step+1}",
                                  safe_serialization=True)
    c.check(progress, "no actual optimizer steps")
    c.check(c.base_fingerprint(model) == frozen, "frozen base changed")
    adapter_after = {n: c.sha_bytes(p.detach().cpu().numpy().tobytes())
                     for n, p in model.named_parameters() if "lora_" in n}
    c.check(any(adapter_before[n] != adapter_after[n] for n in adapter_before),
            "no changed LoRA parameters")
    if len(progress) not in CHECKPOINTS:
        model.save_pretrained(out / f"adapter-step-{len(progress)}",
                              safe_serialization=True)
    trained_probe = probe(model, tokenizer, ids, probes)
    c.save_json(out / "probe-trained.json", trained_probe)
    trained_rollouts = old.rollout(w0, model, tokenizer, ids, out, "trained")
    c.score = previous_score
    controls = []
    for seed in old.ROLLOUT:
        for family in c.FAMILIES:
            for mode in ("random", "wall", "fidget"):
                record = w0.run(seed, mode=mode, family=family, steps=24)
                img = w0.png_bytes(record)
                fname = f"control-{mode}-{seed}-{family}.png"
                (out / fname).write_bytes(img)
                controls.append({"controller": mode, "family": family, "seed": seed,
                    "png": fname, "png_sha256": c.sha_bytes(img),
                    "unique_cells": record["metrics"]["unique_trail_cells"],
                    "illegal_applications": record["metrics"]["invalid_raw_choices"]})
    checkpoints = {str(p.relative_to(out)): c.sha(p)
                   for p in out.rglob("adapter-step-*/*") if p.is_file()}
    c.check(checkpoints, "missing immutable LoRA checkpoints")
    all_images = stock_rollouts + trained_rollouts + controls
    c.check(len(all_images) == 45 and
            all(c.sha(out / row["png"]) == row["png_sha256"] for row in all_images),
            "original PNG output checksum changed")
    c.save_json(out / "summary.json", {
        "schema": SCHEMA, "status": "complete" if len(progress) == STEPS else "partial",
        "source_sha": os.getenv("GITHUB_SHA"), "parent_model": c.MODEL,
        "model_revision": c.REVISION, "fresh_parent": True,
        "updates": len(progress), "backward_passes": len(progress),
        "gradient_model_forwards": 3 * len(progress),
        "frozen_reference_unchanged": True, "adapter_numerically_changed": True,
        "adapter_sha256": checkpoints, "stock_probe": {
            k: v for k, v in stock_probe.items() if k != "cases"},
        "trained_probe": {k: v for k, v in trained_probe.items() if k != "cases"},
        "stock": stock_rollouts, "trained": trained_rollouts,
        "controls": controls, "original_pngs": len(all_images),
        "native_paint": 0, "host": "synthetic_W0_only",
        "geometry_feature": "derived forward_two_cell (not native vision)",
        "comparability": "matched stock parent on new prompt, not GP350005 old prompt",
        "d1_matched": "NOT RUN", "safety": "W0 host legal actions and replay",
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    c.check(len(progress) == STEPS, "bounded deadline before completion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w0", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--deadline", type=int, default=LIMIT)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    c.check(args.deadline == LIMIT, "immutable deadline changed")
    if args.self_test:
        self_test(args.w0)
        return
    try:
        run(args)
    except BaseException as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        failure = args.out / "failure.json"
        if not failure.exists():
            failure.write_text(json.dumps({
                "kind": type(exc).__name__, "message": str(exc),
                "source": os.getenv("GITHUB_SHA"), "success": False}) + "\n")
        raise


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    main()
