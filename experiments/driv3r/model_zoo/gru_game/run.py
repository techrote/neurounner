"""MZ-007b: 19M-parameter GRU with input normalization and current-state skip.

External source-fenced CyberSand-like research worlds only. The actual game,
Godot, native simulation, Workbench and production assets are never modified.
This is a distinct recurrent architecture, NOT a pretrained language model.
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
    samples, raw_features, paired_switch_metrics, class_histogram,
)
from experiments.driv3r.model_zoo.mamba130.run_counterfactual import measure
from experiments.driv3r.model_zoo.mamba130.world import ACTIONS, World
from experiments.driv3r.model_zoo.rwkv7.run import render_gallery, safe_scores

SCHEMA = "MZ007-GRU19M-GAME-v2"
ARCHITECTURE = "scratch-gru8-h640-context4-layernorm-current-state-skip"
WIDTH = 114
CONTEXT = 4


def goal_neutral(features):
    """Prior observations must not leak any candidate's counterfactual goal."""
    v = list(features)
    if len(v) != WIDTH:
        raise ValueError("wrong game observation dimensions")
    v[4:6] = (0., 0.)
    return v


def encode_context(current, history):
    """Three preceding goal-neutral states plus the current observed goal."""
    state = [goal_neutral(x) for x in history[-(CONTEXT - 1):]]
    padding = [[0.] * WIDTH for _ in range(CONTEXT - 1 - len(state))]
    return padding + state + [list(current)]


def observations(rows):
    """Use only earlier time steps from each same-seed history (no future leakage)."""
    keys = {}
    for row in rows:
        key = (row[2], row[3])
        if key not in keys:
            keys[key] = goal_neutral(raw_features(row[0]))
    seqs = []
    for prompt, _label, seed, turn, _pair in rows:
        history = [keys[(seed, prev)] for prev in range(max(0, turn - 3), turn)
                   if (seed, prev) in keys]
        seqs.append(encode_context(raw_features(prompt), history))
    return seqs


def verify_splits(train, dev, test):
    sets = [set(row[2] for row in partition) for partition in (train, dev, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("TRAIN/DEV/TEST seed leakage")
    for rows in (dev, test):
        if paired_switch_metrics(rows, [row[1] for row in rows])["changed_target_pairs"] < 25:
            raise RuntimeError("counterfactual evaluation too small")


def make_network(torch):
    nn = torch.nn

    class RecurrentGamePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            # MZ-007a failed at learning goal swaps despite 600 optimizer updates.
            # Keep the same 19M recurrent backbone but avoid forcing the current
            # goal and action mask through eight recurrent layers and four ticks.
            self.input_norm = nn.LayerNorm(WIDTH)
            self.rnn = nn.GRU(WIDTH, 640, num_layers=8, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(640 + WIDTH),
                nn.Linear(640 + WIDTH, 128),
                nn.GELU(),
                nn.Linear(128, len(ACTIONS)),
            )

        def forward(self, x):
            normalized = self.input_norm(x)
            state = self.rnn(normalized)[0][:, -1, :]
            # The direct current-percept path tests whether the recurrent stack
            # was bottlenecking useful goal/legality information.
            return self.head(torch.cat((state, normalized[:, -1, :]), dim=-1))

    return RecurrentGamePolicy()


def make_control(torch):
    nn = torch.nn
    return nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 64),
                         nn.GELU(), nn.Linear(64, len(ACTIONS)))


def pct95(data):
    return round(sorted(data)[int(0.95 * (len(data) - 1))], 3)


def experiment(args):
    if min(args.train_seeds, args.dev_seeds, args.steps, args.updates,
           args.batch_size, args.rollout_steps) <= 0:
        raise ValueError("positive budgets required")
    import torch
    from PIL import Image, ImageDraw

    torch.set_num_threads(2)
    torch.manual_seed(42021)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = os.getenv("GITHUB_SHA", "LOCAL-UNVERIFIED")

    train = samples(range(400, 400 + args.train_seeds), args.steps)
    dev = samples(range(900, 900 + args.dev_seeds), args.steps)
    test = samples(range(1200, 1200 + args.dev_seeds), args.steps)
    verify_splits(train, dev, test)

    train_x = torch.tensor(observations(train), dtype=torch.float32)
    dev_x = torch.tensor(observations(dev), dtype=torch.float32)
    test_x = torch.tensor(observations(test), dtype=torch.float32)
    y = torch.tensor([row[1] for row in train], dtype=torch.long)
    ctrl_x = train_x[:, -1, :]
    ctrl_dev = dev_x[:, -1, :]
    ctrl_test = test_x[:, -1, :]

    network = make_network(torch)
    control = make_control(torch)
    parameters = sum(p.numel() for p in network.parameters())
    if not 18_000_000 <= parameters <= 22_000_000:
        raise RuntimeError("wrong recurrent-model parameter count")
    net_opt = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=.01)
    ctrl_opt = torch.optim.AdamW(control.parameters(), lr=.002, weight_decay=.01)
    hist = torch.bincount(y, minlength=len(ACTIONS)).float()
    weights = torch.zeros(len(ACTIONS))
    present = hist > 0
    weights[present] = (hist[present].sum() / hist[present]).sqrt()
    weights[present] /= weights[present].mean()

    generator = torch.Generator().manual_seed(42111)
    order = torch.randperm(len(train), generator=generator)
    cursor = 0
    loss_history = {"gru": [], "structured": []}
    update_ms = []
    for step in range(args.updates):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(train), generator=generator)
            cursor = 0
        ids = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        for name, net, opt, data in (
            ("gru", network, net_opt, train_x),
            ("structured", control, ctrl_opt, ctrl_x),
        ):
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits = net(data[ids])
            loss = torch.nn.functional.cross_entropy(logits, y[ids], weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError(name + " nonfinite optimizer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            loss_history[name].append(round(float(loss.detach()), 5))
            if name == "gru":
                update_ms.append((time.perf_counter() - start) * 1000)

    network.eval()
    control.eval()
    forward_ms = []

    def predict(net, data, measured=False):
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(data), args.eval_batch):
                batch = data[start:start + args.eval_batch]
                tic = time.perf_counter()
                logits = net(batch)
                if measured:
                    forward_ms.append((time.perf_counter() - tic) * 1000 / len(batch))
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite game policy inference")
                predictions.extend(logits.argmax(-1).tolist())
        return predictions

    d_gru = predict(network, dev_x, True)
    t_gru = predict(network, test_x, True)
    d_ctrl = predict(control, ctrl_dev)
    t_ctrl = predict(control, ctrl_test)
    ablated = test_x.clone()
    ablated[:, :-1, :] = 0.
    t_no_history = predict(network, ablated, True)
    majority = int(hist.argmax())

    scores = {
        "gru": {"dev": measure(dev, d_gru), "test": measure(test, t_gru)},
        "gru_no_history": {"test": measure(test, t_no_history)},
        "structured": {"dev": measure(dev, d_ctrl), "test": measure(test, t_ctrl)},
        "majority": {"action": ACTIONS[majority],
                     "test": measure(test, [majority] * len(test))},
    }

    worlds, traces = [], []
    for seed in (1840, 1841, 1842, 1843):
        for kind in ("gru", "structured", "random-nonstop", "teacher"):
            world = World.make(seed)
            rng = random.Random(seed * 101 + 17)
            previous = []
            events = []
            raw_invalid = 0
            for _ in range(args.rollout_steps):
                prompt = world.prompt()
                current = raw_features(prompt)
                legal = world.legal()
                if kind == "teacher":
                    choice = raw = world.teacher(rng)
                elif kind == "random-nonstop":
                    choice = raw = rng.choice([a for a in legal if a != "STOP"] or ["STOP"])
                else:
                    with torch.inference_mode():
                        if kind == "gru":
                            x = torch.tensor([encode_context(current, previous)], dtype=torch.float32)
                            logits = network(x)[0]
                        else:
                            x = torch.tensor([current], dtype=torch.float32)
                            logits = control(x)[0]
                    raw = ACTIONS[int(logits.argmax())]
                    raw_invalid += int(raw not in legal)
                    choice = safe_scores(logits, legal)
                events.append({"turn": world.turns, "raw": raw,
                               "legal": legal, "selected": choice})
                previous.append(goal_neutral(current))
                world.apply(choice)
                if world.stopped:
                    break
            traces.append({"seed": seed, "policy": kind, "painted": len(world.painted),
                           "turns": len(events), "raw_illegal": raw_invalid, "events": events})
            worlds.append(world)
    render_gallery(Image, ImageDraw, worlds, traces, out / "rollouts.png")

    checkpoint = out / "gru19m_game_policy.pt"
    torch.save({
        "schema": SCHEMA, "source_sha": source, "architecture": ARCHITECTURE,
        "action_contract": ACTIONS, "context": CONTEXT, "recurrent_weights": network.state_dict(),
        "structured_control": control.state_dict(), "updates": args.updates,
        "train_seeds": args.train_seeds, "seed": 42021,
    }, checkpoint)
    corpus = json.dumps({"train": train, "dev": dev, "test": test}, sort_keys=True).encode()
    result = {
        "schema": SCHEMA, "source_sha": source,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": ARCHITECTURE, "model_provenance": "fully trained from random seed 42021; no external pretrained weights",
        "experiment_reason": "MZ007a 19M GRU hit 21.47% TEST accuracy, 0/416 goal-swap pairs, versus 86.22% structured. MZ007b adds input LayerNorm and a current-state skip while retaining the same 8x640 recurrent core.",
        "reference_a_source_sha": "8d83f4c6f62006607a4e4edacf86b77568b379f2",
        "parameters": parameters, "trainable_parameters": parameters,
        "train_examples": len(train), "dev_examples": len(dev), "test_examples": len(test),
        "train_histogram": class_histogram(train),
        "updates": args.updates, "batch_size": args.batch_size, "learning_rate": args.lr,
        "sequence_length": CONTEXT, "feature_dimensions": WIDTH,
        "losses": loss_history, "scores": scores,
        "forward_amortized_p50_ms": round(statistics.median(forward_ms), 3),
        "forward_amortized_p95_ms": pct95(forward_ms),
        "optimizer_step_p50_ms": round(statistics.median(update_ms), 3),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "traces": traces,
        "limitations": [
            "From-scratch recurrent model, not a pretrained SLM; architecture fully source-controlled.",
            "External synthetic CyberSand-style game worlds, not native PAINT or Workbench.",
            "Scripted teacher supervision, not an independent task-solving metric.",
            "Training baseline has matched minibatch indices/update counts but different model capacity.",
            "Sequence history is goal-neutral and restricted to past world observations.",
            "Goal-swap paired TEST probes isolate target changes, but do not prove intelligence.",
            "Host legal masking mandatory; raw illegal choices separately reported.",
            "No CyberSand game code/content changes or production promotion.",
        ]
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "REPORT.md").write_text(
        "# MZ-007 ~19M GRU game controller\n\n"
        f"Git source {source}; from-scratch random seed 42021, {parameters} parameters.\n\n"
        f"TRAIN/DEV/TEST {len(train)}/{len(dev)}/{len(test)}, {args.updates} real optimizer updates.\n\n"
        f"GRU TEST {json.dumps(scores['gru']['test'])}\n\n"
        f"GRU history ablated TEST {json.dumps(scores['gru_no_history']['test'])}\n\n"
        f"Structured TEST {json.dumps(scores['structured']['test'])}\n\n"
        f"Majority TEST {json.dumps(scores['majority']['test'])}\n\n"
        f"Checkpoint SHA256 {result['checkpoint_sha256']}\n\n"
        "Research only, synthetic source-fenced game worlds, no native game edits. Draft/HOLD.\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("traces", "losses")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mz007-evidence")
    parser.add_argument("--train-seeds", type=int, default=120)
    parser.add_argument("--dev-seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=.0004)
    experiment(parser.parse_args())


if __name__ == "__main__":
    main()
