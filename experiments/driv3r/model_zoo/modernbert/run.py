"""MZ-006: bidirectional ModernBERT with real last-block game-policy tuning.

Synthetic CyberSand-style game worlds only. Reads prior model-zoo research
helpers; never imports or edits Godot/native/Workbench or production content.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import statistics
import time

from experiments.driv3r.model_zoo.mamba130.counterfactual import (
    samples, raw_features, paired_switch_metrics, class_histogram,
)
from experiments.driv3r.model_zoo.mamba130.run_counterfactual import measure
from experiments.driv3r.model_zoo.mamba130.world import ACTIONS, World
from experiments.driv3r.model_zoo.rwkv7.run import fit, safe_scores, render_gallery

MODEL = "answerdotai/ModernBERT-base"
PIN = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
SCHEMA = "MZ006-MODERNBERT-GAME-v1"


def verify_splits(train, dev, test):
    groups = [{r[2] for r in p} for p in (train, dev, test)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("TRAIN/DEV/TEST world-seed leakage")
    for p in (dev, test):
        if paired_switch_metrics(p, [r[1] for r in p])["changed_target_pairs"] < 20:
            raise ValueError("insufficient goal-swap counterfactuals")


def select_trainable(parameter_names):
    """Unfreeze the final actual pretrained encoder block and new classifier.

    Fail closed if the installed Transformers version changes its parameter
    structure. This prevents silently training a classifier-only head.
    """
    names = list(parameter_names)
    layers = [int(m.group(1)) for name in names
              if (m := re.search(r"(?:^|\.)layers\.(\d+)\.", name))]
    if len(set(layers)) < 20:
        raise RuntimeError("ModernBERT encoder-layer manifest not recognized")
    last = max(layers)
    selected = []
    for name in names:
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        top = name.split(".")[0]
        if (m and int(m.group(1)) == last) or top in ("head", "classifier"):
            selected.append(name)
    if not any(re.search(r"(?:^|\.)layers\.%d\." % last, n) for n in selected):
        raise RuntimeError("no pretrained encoder block selected")
    if not any(n.split(".")[0] in ("head", "classifier") for n in selected):
        raise RuntimeError("no classifier selected")
    return selected, last


def p95(values):
    return round(sorted(values)[int(0.95 * (len(values) - 1))], 3)


def experiment(args):
    if min(args.train_seeds, args.dev_seeds, args.steps, args.updates, args.rollout_steps) <= 0:
        raise ValueError("positive research budgets required")
    import torch
    from PIL import Image, ImageDraw
    from huggingface_hub import model_info
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(42019)
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_sha = os.getenv("GITHUB_SHA", "LOCAL-UNVERIFIED")
    meta = model_info(MODEL, revision=PIN)
    if meta.sha != PIN:
        raise RuntimeError("immutable upstream model mismatch")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=PIN, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=PIN, num_labels=len(ACTIONS),
        trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation="sdpa")
    if model.config.model_type != "modernbert" or model.config.num_labels != len(ACTIONS):
        raise RuntimeError("wrong pretrained model type or output classes")
    total_parameters = sum(p.numel() for p in model.parameters())
    if not 140_000_000 <= total_parameters <= 170_000_000:
        raise RuntimeError("wrong pretrained parameter count")
    load_seconds = round(time.monotonic() - started, 3)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train = samples(range(400, 400 + args.train_seeds), args.steps)
    dev = samples(range(900, 900 + args.dev_seeds), args.steps)
    test = samples(range(1200, 1200 + args.dev_seeds), args.steps)
    verify_splits(train, dev, test)

    # Shared exact-seed curriculum with the previous model-zoo probes;
    # pretrained-model features and structured game features are separate.
    train_prompts = [r[0] for r in train]
    dev_prompts = [r[0] for r in dev]
    test_prompts = [r[0] for r in test]
    y = torch.tensor([r[1] for r in train], dtype=torch.long)
    enc_train = tokenizer(train_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)
    enc_dev = tokenizer(dev_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=128)
    enc_test = tokenizer(test_prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=128)
    token_lengths = [int(v) for v in enc_train["attention_mask"].sum(-1).tolist()]
    if not all(5 <= n <= 128 for n in token_lengths):
        raise RuntimeError("invalid or truncated game observation")
    if max(token_lengths) >= 128:
        raise RuntimeError("input hit truncation limit")

    base = model.base_model
    if base is None:
        raise RuntimeError("pretrained encoder inaccessible")
    def cached_features(encoded):
        blocks = []
        with torch.inference_mode():
            for start in range(0, len(encoded["input_ids"]), args.eval_batch):
                batch = {k: v[start:start + args.eval_batch] for k, v in encoded.items()}
                hidden = base(**batch, return_dict=True).last_hidden_state
                if hidden.ndim != 3 or hidden.shape[-1] < 128 or not torch.isfinite(hidden).all():
                    raise RuntimeError("invalid pretrained bidirectional encoder features")
                blocks.append(hidden[:, 0, :].float().cpu().clone())
        return torch.cat(blocks)

    frozen_x = cached_features(enc_train)
    frozen_d = cached_features(enc_dev)
    frozen_t = cached_features(enc_test)
    frozen_head, frozen_losses = fit(torch, frozen_x, y, args.updates + args.warmup_updates, 42019)
    h_x = torch.tensor([raw_features(p) for p in train_prompts], dtype=torch.float32)
    h_d = torch.tensor([raw_features(p) for p in dev_prompts], dtype=torch.float32)
    h_t = torch.tensor([raw_features(p) for p in test_prompts], dtype=torch.float32)
    cheap, cheap_losses = fit(torch, h_x, y, args.updates + args.warmup_updates, 42019)

    chosen, last_layer = select_trainable([n for n, _ in model.named_parameters()])
    for name, parameter in model.named_parameters():
        if name in chosen:
            parameter.requires_grad_(True)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    count = sum(p.numel() for _, p in trainable)
    if not 500_000 <= count <= 15_000_000:
        raise RuntimeError("unexpected trainable-layer parameter count: " + str(count))
    layer_names = {n for n, _ in trainable if f".layers.{last_layer}." in n}
    layer_params = [p for n, p in trainable if n in layer_names]
    head_params = [p for n, p in trainable if n not in layer_names]
    if len(layer_names) < 4 or len(head_params) < 2:
        raise RuntimeError(f"failed to partition pretrained block and classifier: block={len(layer_names)} head={len(head_params)}")
    hist = torch.bincount(y, minlength=len(ACTIONS)).float()
    weights = torch.zeros(len(ACTIONS))
    present = hist > 0
    weights[present] = (hist[present].sum() / hist[present]).sqrt()
    weights[present] /= weights[present].mean()
    perm = torch.randperm(len(y))
    cursor = 0
    def take_batch():
        nonlocal perm, cursor
        if cursor + args.batch_size > len(y):
            perm = torch.randperm(len(y))
            cursor = 0
        ids = perm[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        return ids, {k: v[ids] for k, v in enc_train.items()}
    def update(optimizer):
        ids, batch = take_batch()
        tic = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = torch.nn.functional.cross_entropy(outputs.logits, y[ids], weight=weights)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite optimizer loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        return round(float(loss.detach()), 5), (time.perf_counter() - tic) * 1000

    # Stage 1: warm up the new, randomly initialized classifier to avoid
    # immediately perturbing a good pretrained representation with a bad head.
    # Stage 2: unfreeze a real pretrained encoder block at a conservative LR.
    model.eval()
    for parameter in layer_params:
        parameter.requires_grad_(False)
    warmup_opt = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=.01)
    warmup_losses, train_ms = [], []
    for _ in range(args.warmup_updates):
        loss, elapsed = update(warmup_opt)
        warmup_losses.append(loss)
        train_ms.append(elapsed)
    for parameter in layer_params:
        parameter.requires_grad_(True)
    tune_opt = torch.optim.AdamW([
        {"params": layer_params, "lr": 5e-5},
        {"params": head_params, "lr": 2e-4},
    ], weight_decay=.01)
    losses = []
    for _ in range(args.updates):
        loss, elapsed = update(tune_opt)
        losses.append(loss)
        train_ms.append(elapsed)
    model.eval()

    prediction_ms = []
    def live_scores(encoded, track=True):
        result = []
        with torch.inference_mode():
            for start in range(0, len(encoded["input_ids"]), args.eval_batch):
                batch = {k: v[start:start + args.eval_batch] for k, v in encoded.items()}
                tic = time.perf_counter()
                logits = model(**batch).logits
                if track:
                    prediction_ms.append((time.perf_counter() - tic) * 1000 / len(logits))
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite classifier logits")
                result.extend(logits.argmax(-1).tolist())
        return result
    tuned_d = live_scores(enc_dev)
    tuned_t = live_scores(enc_test)
    with torch.inference_mode():
        frozen_dev = frozen_head(frozen_d).argmax(-1).tolist()
        frozen_test = frozen_head(frozen_t).argmax(-1).tolist()
        cheap_dev = cheap(h_d).argmax(-1).tolist()
        cheap_test = cheap(h_t).argmax(-1).tolist()
    maj = int(torch.bincount(y, minlength=len(ACTIONS)).argmax())
    scores = {
        "modernbert_tuned": {"dev": measure(dev, tuned_d), "test": measure(test, tuned_t)},
        "modernbert_frozen": {"dev": measure(dev, frozen_dev),
                              "test": measure(test, frozen_test)},
        "structured": {"dev": measure(dev, cheap_dev), "test": measure(test, cheap_test)},
        "majority": {"action": ACTIONS[maj],
                     "test": measure(test, [maj] * len(test))},
    }

    # Four-way same-world actual evolving legal-action trajectory comparison.
    worlds, traces = [], []
    for seed in (1825, 1826, 1827, 1828):
        for kind in ("modernbert_tuned", "structured", "random-nonstop", "teacher"):
            w = World.make(seed)
            rng = random.Random(seed * 101 + 17)
            events = []
            raw_invalid = 0
            for _ in range(args.rollout_steps):
                legal = w.legal()
                if kind == "teacher":
                    pick = raw = w.teacher(rng)
                elif kind == "random-nonstop":
                    pick = raw = rng.choice([a for a in legal if a != "STOP"] or ["STOP"])
                else:
                    observation = w.prompt()
                    if kind == "structured":
                        with torch.inference_mode():
                            logits = cheap(torch.tensor(
                                raw_features(observation), dtype=torch.float32).unsqueeze(0))[0]
                    else:
                        encoded = tokenizer(observation, return_tensors="pt",
                                            truncation=True, max_length=128)
                        tic = time.perf_counter()
                        with torch.inference_mode():
                            logits = model(**encoded).logits[0]
                        prediction_ms.append((time.perf_counter() - tic) * 1000)
                    raw = ACTIONS[int(logits.argmax())]
                    raw_invalid += int(raw not in legal)
                    pick = safe_scores(logits, legal)
                events.append({"turn": w.turns, "raw": raw,
                               "selected": pick, "legal": legal})
                w.apply(pick)
                if w.stopped:
                    break
            traces.append({"seed": seed, "policy": kind, "painted": len(w.painted),
                           "turns": len(events), "raw_illegal": raw_invalid, "events": events})
            worlds.append(w)
    render_gallery(Image, ImageDraw, worlds, traces, out / "rollouts.png")
    checkpoint = out / "modernbert_game_lastblock.pt"
    torch.save({
        "schema": SCHEMA, "source_sha": source_sha, "model": MODEL, "revision": PIN,
        "actions": ACTIONS, "last_pretrained_layer": last_layer,
        "trainable_state": {n: p.detach().cpu() for n, p in trainable},
        "frozen_head": frozen_head.state_dict(),
        "structured_head": cheap.state_dict(),
        "train_seeds": args.train_seeds, "updates": args.updates, "warmup_updates": args.warmup_updates,
    }, checkpoint)
    corpus = json.dumps({"train": train, "dev": dev, "test": test}, sort_keys=True).encode()
    result = {
        "schema": SCHEMA, "source_sha": source_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "revision": PIN, "encoder_parameters": total_parameters,
        "pretrained_layer_tuned": last_layer, "trainable_parameters": count,
        "trainable_manifest": chosen, "load_seconds": load_seconds,
        "train_examples": len(train), "dev_examples": len(dev), "test_examples": len(test),
        "train_histogram": class_histogram(train), "updates": args.updates,
        "batch_size": args.batch_size, "warmup_updates": args.warmup_updates, "warmup_losses": warmup_losses,
        "training_losses": losses, "frozen_head_losses": frozen_losses,
        "structured_losses": cheap_losses,
        "scores": scores,
        "token_median": statistics.median(token_lengths),
        "token_max": max(token_lengths),
        "tuned_forward_amortized_p50_ms": round(statistics.median(prediction_ms), 3),
        "tuned_forward_amortized_p95_ms": p95(prediction_ms),
        "optimizer_step_p50_ms": round(statistics.median(train_ms), 3),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(), "traces": traces,
        "limitations": [
            "Synthetic scripted-label micro-worlds only, not native CyberSand PAINT.",
            "Actual pretrained last encoder block and classifier are optimized, not full model.",
            "Frozen-head and structured controls receive same optimizer update count, but full-batch updates vs tuned minibatches differ in sample exposure.",
            "Raw model legality is measured separately from mandatory host action masking.",
            "Token input format is unstructured raw observation, no equivalent Qwen chat template.",
            "No model promotion, game edits, or paid GPU inference.",
        ],
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "REPORT.md").write_text(
        "# MZ-006 ModernBERT-base pretrained-layer game training\n\n"
        f"Git source {source_sha}, model {MODEL}@{PIN}.\n\n"
        f"Real pretrained last block {last_layer} + classifier, {count} trained parameters.\n\n"
        f"TRAIN/DEV/TEST {len(train)}/{len(dev)}/{len(test)}; {args.warmup_updates} head warmup + {args.updates} encoder optimizer steps.\n\n"
        f"Tuned TEST {json.dumps(scores['modernbert_tuned']['test'])}\n\n"
        f"Frozen TEST {json.dumps(scores['modernbert_frozen']['test'])}\n\n"
        f"Structured TEST {json.dumps(scores['structured']['test'])}\n\n"
        f"Majority TEST {json.dumps(scores['majority']['test'])}\n\n"
        f"Checkpoint SHA256 {result['checkpoint_sha256']}\n\n"
        "Research synthetic games only. No CyberSand game code/content changed. Draft/HOLD.\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("traces", "training_losses", "frozen_head_losses",
                                   "structured_losses", "warmup_losses", "trainable_manifest")},
                     indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mz006-modernbert-evidence")
    ap.add_argument("--train-seeds", type=int, default=14)
    ap.add_argument("--dev-seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--updates", type=int, default=320)
    ap.add_argument("--warmup-updates", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--rollout-steps", type=int, default=16)
    experiment(ap.parse_args())


if __name__ == "__main__":
    main()
