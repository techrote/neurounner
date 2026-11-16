"""MZ-010: width-vs-depth recurrent scaling on MZ-009's closed-loop material forks.

Research-only. All environment transitions, splits and legality contracts are
imported read-only from the immutable MZ-009 stacked branch. No game changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time

from experiments.driv3r.model_zoo.material_forks_mz009 import run as env

SCHEMA = "MZ010-WIDE-GRU-CLOSED-LOOP-v1"
PARENT_HEAD = "3a4eded62900aad3df91fd9b2a1ae28bbb4a0c0e"


def build_models(torch, hidden=1504, layers=2):
    # The ~20M model is deliberately WIDE and SHALLOW, not a repeat of the
    # failed 8-layer 19M GRU in MZ-008's alternate credit-assignment study.
    wide, _ = env.networks(torch, hidden=hidden, layers=layers)
    small, current = env.networks(torch, hidden=128, layers=2)
    return {"wide": wide, "small": small, "current": current}


def model_counts(models):
    return {name: sum(p.numel() for p in net.parameters())
            for name, net in models.items()}


def evaluate(torch, model, x, y):
    model.eval()
    with torch.inference_mode():
        return env.class_scores(model(x), y, x)


def causal_tests(torch, model, x, y):
    model.eval()
    fork = (x[:, :, 1] > 0) & (y != -100)
    first = fork & (torch.cumsum(fork.to(torch.int), dim=1) == 1)
    flipped = x.clone()
    flipped[:, :2, 17:19] = x[:, :2, 17:19].flip(-1)
    deleted = x.clone()
    deleted[:, :4, 17:21] = 0
    with torch.inference_mode():
        normal = model(x).argmax(-1)
        inverted = model(flipped).argmax(-1)
        removed = model(deleted).argmax(-1)
        score = env.class_scores(model(deleted), y, deleted)
    n = max(1, int(first.sum()))
    return {
        "first_fork_beacon_flip_action_switch": round(
            float(((inverted != normal) & first).sum()) / n, 5),
        "first_fork_cue_deletion_action_switch": round(
            float(((removed != normal) & first).sum()) / n, 5),
        "cue_deleted_fork_accuracy": score["fork_accuracy"],
        "paired_first_fork_decisions": int(first.sum()),
    }


def bench_forward(torch, model, seq_len, trials=60):
    x = torch.zeros((1, seq_len, env.D))
    model.eval()
    with torch.inference_mode():
        for _ in range(8):
            model(x)
        results = []
        for _ in range(trials):
            start = time.perf_counter()
            model(x)
            results.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": round(statistics.median(results), 4),
            "p95_ms": round(sorted(results)[int(.95 * (trials - 1))], 4),
            "sequence_length": seq_len,
            "single_example_unbatched": True}


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(61010)
    sampler = torch.Generator().manual_seed(61011)

    train = env.dataset(range(1000, 1000 + args.train_seeds))
    dev = env.dataset(range(3000, 3000 + args.eval_seeds))
    test = env.dataset(range(5000, 5000 + args.eval_seeds))
    train_seeds = {row[2][0] for row in train}
    dev_seeds = {row[2][0] for row in dev}
    test_seeds = {row[2][0] for row in test}
    assert not (train_seeds & dev_seeds or train_seeds & test_seeds
                or dev_seeds & test_seeds), "world split leakage"
    x, y = env.to_tensors(torch, train)
    xd, yd = env.to_tensors(torch, dev)
    xt, yt = env.to_tensors(torch, test)
    models = build_models(torch, args.hidden, args.layers)
    counts = model_counts(models)
    if not 20_000_000 <= counts["wide"] <= 24_000_000:
        raise ValueError("MZ-010 wide model must have 20M-24M parameters")
    optim = {
        name: torch.optim.AdamW(net.parameters(),
                                lr=args.wide_lr if name == "wide" else .003,
                                weight_decay=.001)
        for name, net in models.items()
    }
    weight = torch.tensor([.5, 3., 3., 1.5, .5])
    losses = {name: [] for name in models}
    wide_step_ms = []
    for step in range(args.updates):
        ids = torch.randint(len(x), (args.batch,), generator=sampler)
        for name, net in models.items():
            net.train()
            tick = time.perf_counter()
            optim[name].zero_grad(set_to_none=True)
            logits = net(x[ids])
            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), y[ids], weight=weight,
                ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {name} optimizer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim[name].step()
            if step % 25 == 0 or step == args.updates - 1:
                losses[name].append({"step": step + 1,
                                     "cross_entropy": round(float(loss.detach()), 6)})
            if name == "wide":
                wide_step_ms.append((time.perf_counter() - tick) * 1000)
        if (step + 1) % 100 == 0:
            print(f"MZ-010 completed {step + 1}/{args.updates} actual optimizer updates",
                  flush=True)

    scores = {
        name: {"dev": evaluate(torch, model, xd, yd),
               "test": evaluate(torch, model, xt, yt),
               "causal": causal_tests(torch, model, xt, yt)}
        for name, model in models.items()
    }

    seeds = range(7000, 7000 + args.rollout_seeds)
    rollouts = {
        **{name: env.evaluate_rollouts(torch, model, seeds)
           for name, model in models.items()},
        "wide_history_1": env.evaluate_rollouts(torch, models["wide"], seeds,
                                                context_limit=1),
        "wide_history_4": env.evaluate_rollouts(torch, models["wide"], seeds,
                                                context_limit=4),
        "wide_history_8": env.evaluate_rollouts(torch, models["wide"], seeds,
                                                context_limit=8),
        "wide_forced_wrong_first_fork": env.evaluate_rollouts(
            torch, models["wide"], seeds, force_error=True),
    }
    gallery = env.build_gallery(
        torch,
        [models["wide"], models["current"], models["wide"], models["wide"]],
        [None, None, 1, 4],
        list(range(7000, 7004)),
    ).replace("MZ-009", "MZ-010 wide GRU (20M vs small GRU)").replace(
        "GRU full history", "20M wide GRU full history")

    bench = {
        "wide_single": bench_forward(torch, models["wide"], 1),
        "wide_8": bench_forward(torch, models["wide"], 8),
        "small_single": bench_forward(torch, models["small"], 1),
        "small_8": bench_forward(torch, models["small"], 8),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = os.getenv("GITHUB_SHA", "LOCAL-UNVERIFIED")
    checkpoint = out / "models.pt"
    torch.save({
        "schema": SCHEMA, "source_sha": source,
        "parent_source_sha": PARENT_HEAD,
        "actions": env.ACTIONS, "hidden": args.hidden,
        "layers": args.layers,
        "models": {name: net.state_dict() for name, net in models.items()},
    }, checkpoint)
    report = {
        "schema": SCHEMA,
        "source_sha": source,
        "parent_source_sha": PARENT_HEAD,
        "parameters": counts,
        "training": {
            "train_seeds": args.train_seeds,
            "eval_seeds": args.eval_seeds,
            "rollout_seeds": args.rollout_seeds,
            "updates_per_model": args.updates,
            "batch_size": args.batch,
            "train_episodes": len(train),
            "test_episodes": len(test),
            "wide_lr": args.wide_lr,
            "seed_families_split_disjoint": True,
        },
        "scores": scores,
        "rollouts": rollouts,
        "latency_cpu": bench,
        "wide_optimizer_step_p50_ms": round(statistics.median(wide_step_ms), 3),
        "peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "losses": losses,
        "limitations": [
            "Procedural synthetic material-fork host only, no native CyberSand",
            "20M model is wider, not sequence/context-compute matched",
            "Teacher labels supplied for supervised actions; rollout separately executed",
            "No promotion on eventual completion without collision-free success",
            "Runs only on the research-only MZ-009 stacked source; Draft/HOLD",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "trajectories.svg").write_text(gallery)
    (out / "REPORT.md").write_text(
        "# MZ-010: wide 20M GRU versus shallow small GRU on closed-loop material forks\n\n"
        + "Exact source: " + source + "\n\n"
        + "20M fork accuracy: " + str(scores["wide"]["test"]["fork_accuracy"])
        + "\nSmall GRU fork accuracy: " + str(scores["small"]["test"]["fork_accuracy"])
        + "\n20M collision-free completion: "
        + str(rollouts["wide"]["clean_success_rate"])
        + "\nSmall GRU collision-free completion: "
        + str(rollouts["small"]["clean_success_rate"])
        + "\n\nResearch only, synthetic, Draft/HOLD; no CyberSand game changes.\n"
    )
    print(json.dumps({
        "source_sha": source, "parameters": counts, "scores": scores,
        "rollouts_summary": {
            k: {s: v[s] for s in ("success_rate", "clean_success_rate", "collisions",
                                  "raw_illegal")}
            for k, v in rollouts.items()},
        "latency_cpu": bench, "peak_rss_mib": report["peak_rss_mib"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    }, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="mz010-evidence")
    p.add_argument("--train-seeds", type=int, default=160)
    p.add_argument("--eval-seeds", type=int, default=24)
    p.add_argument("--rollout-seeds", type=int, default=8)
    p.add_argument("--updates", type=int, default=800)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--hidden", type=int, default=1504)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--wide-lr", type=float, default=.002)
    run(p.parse_args())


if __name__ == "__main__":
    main()
