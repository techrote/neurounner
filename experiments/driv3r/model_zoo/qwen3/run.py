"""MZ-005: pinned Qwen3-0.6B instruction-tuned transformer -> real game-policy head.

Synthetic CyberSand-style worlds only, independent of game/Workbench implementation.
The pretrained backbone is strictly frozen; the attached action head IS trained.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import time

from experiments.driv3r.model_zoo.mamba130.counterfactual import (
    samples, raw_features, paired_switch_metrics, class_histogram
)
from experiments.driv3r.model_zoo.mamba130.run_counterfactual import measure
from experiments.driv3r.model_zoo.mamba130.world import ACTIONS, World
from experiments.driv3r.model_zoo.rwkv7.run import fit, safe_scores, render_gallery

MODEL = "Qwen/Qwen3-0.6B"
PIN = "c1899de289a04d12100db370d81485cdf75e47ca"
SCHEMA = "MZ005-QWEN3-GAME-v1"
SYSTEM = (
    "You steer a small 2D painting/game worm. Observe the local material grid, "
    "blocked/protected cells, facing, goal and host-provided legal actions. "
    "Select a useful action, prioritizing progress and recovery while respecting "
    "legality. You cannot edit the world directly."
)


def chat_messages(world_prompt):
    if not world_prompt.startswith("WORM d=") or " legal=" not in world_prompt:
        raise ValueError("unrecognized game observation")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": world_prompt}]


def verify_splits(train, dev, test):
    groups = [{r[2] for r in part} for part in (train, dev, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("TRAIN/DEV/TEST world-seed leakage")
    for part in (dev, test):
        if paired_switch_metrics(part, [r[1] for r in part])["changed_target_pairs"] < 20:
            raise ValueError("insufficient counterfactual goal switches")


def run(args):
    if args.train_seeds < 4 or args.dev_seeds < 2 or args.epochs < 1:
        raise ValueError("invalid experiment budget")
    import torch
    from PIL import Image, ImageDraw
    from huggingface_hub import model_info
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(42017)
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = os.getenv("GITHUB_SHA", "LOCAL-UNVERIFIED")
    info = model_info(MODEL, revision=PIN)
    if info.sha != PIN:
        raise RuntimeError("immutable upstream revision mismatch")
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=PIN, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        MODEL, revision=PIN, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32)
    if model.config.model_type != "qwen3":
        raise RuntimeError("model is not a supported Qwen3 backbone")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters())
    if not 400_000_000 <= params <= 800_000_000:
        raise RuntimeError(f"unexpected model parameter count: {params}")
    load_s = round(time.monotonic() - t0, 3)
    train = samples(range(400, 400 + args.train_seeds), args.steps)
    dev = samples(range(900, 900 + args.dev_seeds), args.steps)
    test = samples(range(1200, 1200 + args.dev_seeds), args.steps)
    verify_splits(train, dev, test)

    cache = {}
    forward_ms = []
    token_lengths = []
    def feature(prompt):
        if prompt not in cache:
            text = tok.apply_chat_template(
                chat_messages(prompt), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            encoded = tok(text, return_tensors="pt", truncation=True,
                          max_length=256)
            ntokens = int(encoded["input_ids"].shape[1])
            if not 5 <= ntokens <= 256:
                raise RuntimeError("invalid tokenized game observation length")
            token_lengths.append(ntokens)
            tic = time.perf_counter()
            with torch.inference_mode():
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False, return_dict=True)
            forward_ms.append((time.perf_counter() - tic) * 1000)
            hidden = outputs.last_hidden_state
            if hidden.ndim != 3 or not torch.isfinite(hidden).all():
                raise RuntimeError("invalid genuine pretrained hidden state")
            cache[prompt] = hidden[0, -1, :].detach().float().cpu().clone()
        return cache[prompt]

    X = torch.stack([feature(r[0]) for r in train])
    V = torch.stack([feature(r[0]) for r in dev])
    T = torch.stack([feature(r[0]) for r in test])
    Y = torch.tensor([r[1] for r in train], dtype=torch.long)
    H = torch.tensor([raw_features(r[0]) for r in train], dtype=torch.float32)
    HD = torch.tensor([raw_features(r[0]) for r in dev], dtype=torch.float32)
    HT = torch.tensor([raw_features(r[0]) for r in test], dtype=torch.float32)

    trained, losses = fit(torch, X, Y, args.epochs, 42017)
    cheap, cheap_losses = fit(torch, H, Y, args.epochs, 42017)
    with torch.inference_mode():
        qd = trained(V).argmax(-1).tolist()
        qt = trained(T).argmax(-1).tolist()
        cd = cheap(HD).argmax(-1).tolist()
        ct = cheap(HT).argmax(-1).tolist()
    majority = int(torch.bincount(Y, minlength=len(ACTIONS)).argmax())
    scores = {
        "qwen3": {"dev": measure(dev, qd), "test": measure(test, qt)},
        "structured": {"dev": measure(dev, cd), "test": measure(test, ct)},
        "majority": {"action": ACTIONS[majority],
                     "test": measure(test, [majority] * len(test))}
    }
    worlds, traces = [], []
    for seed in (1815, 1816, 1817, 1818):
        for kind in ("qwen3", "structured", "random-nonstop", "teacher"):
            w = World.make(seed)
            rng = random.Random(seed * 101 + 17)
            events = []
            invalid = 0
            for _ in range(args.rollout_steps):
                legal = w.legal()
                if kind == "teacher":
                    pick = raw = w.teacher(rng)
                elif kind == "random-nonstop":
                    pick = raw = rng.choice([a for a in legal if a != "STOP"] or ["STOP"])
                else:
                    with torch.inference_mode():
                        ft = (feature(w.prompt()) if kind == "qwen3"
                              else torch.tensor(raw_features(w.prompt()), dtype=torch.float32))
                        head = trained if kind == "qwen3" else cheap
                        logits = head(ft.unsqueeze(0))[0]
                    raw = ACTIONS[int(logits.argmax())]
                    invalid += int(raw not in legal)
                    pick = safe_scores(logits, legal)
                events.append({"turn": w.turns, "raw": raw,
                               "legal": legal, "selected": pick})
                w.apply(pick)
                if w.stopped:
                    break
            worlds.append(w)
            traces.append({"seed": seed, "policy": kind, "painted": len(w.painted),
                           "turns": len(events), "raw_illegal": invalid, "events": events})
    render_gallery(Image, ImageDraw, worlds, traces, out / "rollouts.png")

    checkpoint = out / "qwen3_game_head.pt"
    torch.save({
        "schema": SCHEMA, "source_sha": source, "model": MODEL, "revision": PIN,
        "actions": ACTIONS, "system_prompt": SYSTEM, "training_world_seeds": args.train_seeds,
        "head": trained.state_dict(), "structured": cheap.state_dict()
    }, checkpoint)
    corpus = json.dumps({"train": train, "dev": dev, "test": test}, sort_keys=True).encode()
    result = {
        "schema": SCHEMA, "source_sha": source, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "revision": PIN,
        "pretrained_parameters": params,
        "trained_head_parameters": sum(p.numel() for p in trained.parameters()),
        "backbone_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "load_seconds": load_s,
        "train_examples": len(train), "dev_examples": len(dev), "test_examples": len(test),
        "train_histogram": class_histogram(train), "epochs": args.epochs,
        "qwen_losses": losses, "structured_losses": cheap_losses,
        "scores": scores, "pretrained_forward_calls": len(forward_ms),
        "pretrained_forward_p50_ms": round(statistics.median(forward_ms), 3),
        "pretrained_forward_p95_ms": round(sorted(forward_ms)[int(.95 * (len(forward_ms) - 1))], 3),
        "input_token_median": statistics.median(token_lengths),
        "input_token_max": max(token_lengths),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "traces": traces,
        "limitations": [
            "The pretrained Qwen3 backbone is FROZEN; only the head has optimizer updates.",
            "Chat-formatted Qwen3 inputs differ from raw-prompt Pythia/RWKV baselines.",
            "Synthetic 2D game worlds, not actual native CyberSand or real PAINT receipts.",
            "Host legal masking is mandatory; raw illegal action proposals are separately counted.",
            "Teacher labels are scripted; goal-swap accuracy is not independent semantic proof.",
            "No paid inference, no model production promotion, no game code/content edits."
        ]
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "REPORT.md").write_text(
        "# MZ-005 Qwen3-0.6B game-policy onboarding\n\n"
        + f"Source {source}; upstream {MODEL}@{PIN}\n\n"
        + f"TRAIN/DEV/TEST {len(train)}/{len(dev)}/{len(test)}; optimizer {args.epochs} updates.\n\n"
        + f"Qwen3 TEST {json.dumps(scores['qwen3']['test'])}\n\n"
        + f"Structured TEST {json.dumps(scores['structured']['test'])}\n\n"
        + f"Majority TEST {json.dumps(scores['majority']['test'])}\n\n"
        + f"CPU p50/p95 {result['pretrained_forward_p50_ms']}/{result['pretrained_forward_p95_ms']} ms; "
        + f"peak RSS {result['peak_rss_mib']} MiB.\n\n"
        + f"Checkpoint SHA256 {result['checkpoint_sha256']}\n\n"
        + "Research-only, held-out synthetic game actions. Draft/HOLD; no CyberSand game edits.\n"
    )
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("traces", "qwen_losses", "structured_losses")}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mz005-qwen3-evidence")
    ap.add_argument("--train-seeds", type=int, default=14)
    ap.add_argument("--dev-seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--rollout-steps", type=int, default=16)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
