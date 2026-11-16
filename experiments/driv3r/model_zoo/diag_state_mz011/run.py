"""MZ-011: input-gated diagonal recurrent state versus GRU for game memory.

Research-only external procedural host. Reuses MZ-009 world and action
contracts read-only; no CyberSand production/gameplay modifications.
This is a diagonal gated-state model, not a claim to reproduce S4 or Mamba.
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

SCHEMA = "MZ011-DIAGONAL-GATED-STATE-v1"
PARENT_HEAD = "3a4eded62900aad3df91fd9b2a1ae28bbb4a0c0e"


def diagonal_model(torch, hidden=224):
    nn = torch.nn

    class DiagonalCell(nn.Module):
        """Input-only gates and proposals; O(H) state transition per step.

        h_t = a(x_t)*h_{t-1} + (1-a(x_t))*tanh(v(x_t)).
        Each latent channel has a diagonal retention coefficient. The second
        layer receives a history-bearing first-layer state, allowing nonlinear
        combinations of temporally separated cues.
        """

        def __init__(self, input_size, size):
            super().__init__()
            self.size = size
            self.proj = nn.Linear(input_size, 2 * size)
            with torch.no_grad():
                self.proj.bias[:size].fill_(3.0)  # initially slow forgetting

        def sequence(self, x):
            retain, value = self.proj(x).chunk(2, dim=-1)
            retain = torch.sigmoid(retain)
            value = torch.tanh(value)
            state = x.new_zeros(x.shape[0], self.size)
            outputs = []
            for t in range(x.shape[1]):
                state = retain[:, t] * state + (1.0 - retain[:, t]) * value[:, t]
                outputs.append(state)
            return torch.stack(outputs, dim=1)

        def step(self, x, state=None):
            retain, value = self.proj(x).chunk(2, dim=-1)
            retain = torch.sigmoid(retain)
            value = torch.tanh(value)
            if state is None:
                state = x.new_zeros(x.shape[0], self.size)
            return retain * state + (1.0 - retain) * value

    class DiagonalPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(env.D)
            self.cells = nn.ModuleList([
                DiagonalCell(env.D, hidden),
                DiagonalCell(hidden, hidden),
            ])
            self.head = nn.Sequential(
                nn.LayerNorm(env.D + hidden),
                nn.Linear(env.D + hidden, 96),
                nn.GELU(),
                nn.Linear(96, len(env.ACTIONS)),
            )

        def forward(self, x):
            z = self.norm(x)
            h = z
            for cell in self.cells:
                h = cell.sequence(h)
            return self.head(torch.cat((z, h), dim=-1))

        def step(self, observation, state=None):
            """One observation, persistent bounded state: O(1) in history."""
            z = self.norm(observation)
            next_state = []
            h = z
            for i, cell in enumerate(self.cells):
                h = cell.step(h, None if state is None else state[i])
                next_state.append(h)
            return self.head(torch.cat((z, h), dim=-1)), tuple(next_state)

    return DiagonalPolicy()


def build_models(torch, hidden=224):
    small, reactive = env.networks(torch, hidden=128, layers=2)
    return {"diagonal": diagonal_model(torch, hidden),
            "small_gru": small, "current_only": reactive}


def parameter_counts(models):
    return {name: sum(p.numel() for p in m.parameters()) for name, m in models.items()}


def score(torch, model, x, y):
    model.eval()
    with torch.inference_mode():
        return env.class_scores(model(x), y, x)


def causal_probe(torch, model, x, y):
    fork = (x[:, :, 1] > 0) & (y != -100)
    first = fork & (torch.cumsum(fork.int(), dim=1) == 1)
    flipped = x.clone()
    flipped[:, :2, 17:19] = flipped[:, :2, 17:19].flip(-1)
    erased = x.clone()
    erased[:, :4, 17:21] = 0
    model.eval()
    with torch.inference_mode():
        original = model(x).argmax(-1)
        altered = model(flipped).argmax(-1)
        deleted = model(erased).argmax(-1)
        erased_score = env.class_scores(model(erased), y, erased)
    count = max(1, int(first.sum()))
    return {
        "first_fork_pairs": int(first.sum()),
        "beacon_flip_action_switch": round(
            float(((original != altered) & first).sum()) / count, 5),
        "all_early_cues_erased_action_switch": round(
            float(((original != deleted) & first).sum()) / count, 5),
        "cue_erased_fork_accuracy": erased_score["fork_accuracy"],
    }


def stateful_play(torch, model, seed, beacon, inversion):
    """Execute real host transitions with O(1)-memory cached recurrent state."""
    world = env.World(seed, beacon, inversion)
    state = None
    raw_illegal = 0
    forks = []
    with torch.inference_mode():
        for _ in range(env.MAX_STEPS):
            observation = torch.tensor([world.observe()], dtype=torch.float32)
            logits, state = model.step(observation, state)
            logits = logits[0]
            mask = world.legal()
            raw = int(logits.argmax())
            raw_illegal += int(not mask[raw])
            legal_mask = torch.tensor([not permitted for permitted in mask])
            action = int(logits.masked_fill(legal_mask, -1e9).argmax())
            if world.stage == "fork":
                forks.append(action)
            world.step(action)
            if world.done:
                break
    success = world.done and not world.failed and world.painted == 3
    return {"success": success, "clean_success": success and world.collisions == 0,
            "collisions": world.collisions, "painted": world.painted,
            "illegal_raw": raw_illegal, "decisions": world.tick,
            "fork_choices": forks}


def stateful_rollouts(torch, model, seeds):
    episodes = [stateful_play(torch, model, s, cue, inv)
                for s in seeds for cue in range(2) for inv in range(2)]
    n = len(episodes)
    return {
        "episodes": n,
        "success_rate": round(sum(r["success"] for r in episodes) / n, 5),
        "clean_success_rate": round(sum(r["clean_success"] for r in episodes) / n, 5),
        "collisions": sum(r["collisions"] for r in episodes),
        "raw_illegal": sum(r["illegal_raw"] for r in episodes),
        "sample": episodes[:8],
    }


def latency(torch, model, length, trials=60, cached=False):
    x = torch.zeros((1, length, env.D))
    model.eval()
    with torch.inference_mode():
        state = None
        if cached:
            for _ in range(20):
                _, state = model.step(x[:, -1], state)
        for _ in range(8):
            if cached:
                model.step(x[:, -1], state)
            else:
                model(x)
        elapsed = []
        for _ in range(trials):
            start = time.perf_counter()
            if cached:
                model.step(x[:, -1], state)
            else:
                model(x)
            elapsed.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": round(statistics.median(elapsed), 4),
            "p95_ms": round(sorted(elapsed)[int(.95 * (trials-1))], 4),
            "sequence_length": length,
            "cached_state": cached,
            "batch": 1}


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(61111)
    sampler = torch.Generator().manual_seed(61112)
    train = env.dataset(range(1000, 1000 + args.train_seeds))
    dev = env.dataset(range(3000, 3000 + args.eval_seeds))
    test = env.dataset(range(5000, 5000 + args.eval_seeds))
    assert {r[2][0] for r in train}.isdisjoint({r[2][0] for r in dev + test})
    assert {r[2][0] for r in dev}.isdisjoint({r[2][0] for r in test})
    x, y = env.to_tensors(torch, train)
    xd, yd = env.to_tensors(torch, dev)
    xt, yt = env.to_tensors(torch, test)
    models = build_models(torch, args.hidden)
    counts = parameter_counts(models)
    optimizer = {name: torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=.001)
                 for name, m in models.items()}
    weight = torch.tensor([.5, 3., 3., 1.5, .5])
    losses = {name: [] for name in models}
    training_ms = {name: [] for name in models}
    for update in range(args.updates):
        ids = torch.randint(len(x), (args.batch,), generator=sampler)
        for name, net in models.items():
            net.train()
            start = time.perf_counter()
            optimizer[name].zero_grad(set_to_none=True)
            pred = net(x[ids])
            loss = torch.nn.functional.cross_entropy(
                pred.transpose(1, 2), y[ids], weight=weight, ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss for {name} at update {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer[name].step()
            training_ms[name].append((time.perf_counter() - start) * 1000)
            if update % 25 == 0 or update == args.updates - 1:
                losses[name].append({"step": update + 1,
                                     "loss": round(float(loss.detach()), 6)})
        if (update + 1) % 100 == 0:
            print(f"MZ-011 completed {update+1}/{args.updates} updates per model",
                  flush=True)
    for m in models.values():
        m.eval()

    scores = {
        name: {"dev": score(torch, model, xd, yd),
               "test": score(torch, model, xt, yt),
               "causal": causal_probe(torch, model, xt, yt)}
        for name, model in models.items()
    }
    rollout_seeds = range(7000, 7000 + args.rollout_seeds)
    rollouts = {
        **{name: env.evaluate_rollouts(torch, net, rollout_seeds)
           for name, net in models.items()},
        "diagonal_last_1": env.evaluate_rollouts(
            torch, models["diagonal"], rollout_seeds, context_limit=1),
        "diagonal_last_4": env.evaluate_rollouts(
            torch, models["diagonal"], rollout_seeds, context_limit=4),
        "diagonal_last_8": env.evaluate_rollouts(
            torch, models["diagonal"], rollout_seeds, context_limit=8),
        "diagonal_forced_wrong_first_fork": env.evaluate_rollouts(
            torch, models["diagonal"], rollout_seeds, force_error=True),
        "diagonal_cached_state": stateful_rollouts(
            torch, models["diagonal"], rollout_seeds),
    }
    batch = [stateful_play(torch, models["diagonal"], s, cue, inv)
             for s in rollout_seeds for cue in range(2) for inv in range(2)]
    prefix = [env.play(torch, models["diagonal"], s, cue, inv)
              for s in rollout_seeds for cue in range(2) for inv in range(2)]
    parity = sum(a["fork_choices"] == b["fork_choices"] and
                 a["collisions"] == b["collisions"] and
                 a["success"] == b["success"]
                 for a, b in zip(batch, prefix))
    times = {
        "diagonal_1": latency(torch, models["diagonal"], 1),
        "diagonal_8": latency(torch, models["diagonal"], 8),
        "diagonal_32": latency(torch, models["diagonal"], 32),
        "diagonal_cached_step": latency(torch, models["diagonal"], 1, cached=True),
        "small_gru_1": latency(torch, models["small_gru"], 1),
        "small_gru_8": latency(torch, models["small_gru"], 8),
    }
    gallery = env.build_gallery(
        torch,
        [models["diagonal"], models["current_only"],
         models["diagonal"], models["diagonal"]],
        [None, None, 1, 4], list(range(7000, 7004)),
    ).replace("MZ-009", "MZ-011 diagonal gated-state").replace(
        "GRU full history", "Diagonal full history").replace(
        "GRU last observation", "Diagonal last observation").replace(
        "GRU four observations", "Diagonal last four observations")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = os.environ.get("GITHUB_SHA", "LOCAL-UNVERIFIED")
    checkpoint = out / "models.pt"
    torch.save({
        "schema": SCHEMA, "source_sha": source, "parent_source_sha": PARENT_HEAD,
        "actions": env.ACTIONS, "hidden": args.hidden,
        "models": {name: net.state_dict() for name, net in models.items()},
    }, checkpoint)
    report = {
        "schema": SCHEMA, "source_sha": source, "parent_source_sha": PARENT_HEAD,
        "parameters": counts,
        "training": {
            "train_seed_families": args.train_seeds,
            "eval_seed_families": args.eval_seeds,
            "rollout_seed_families": args.rollout_seeds,
            "updates_per_model": args.updates, "batch": args.batch, "lr": args.lr,
            "train_episodes": len(train), "test_episodes": len(test),
            "disjoint_split_families": True,
        },
        "scores": scores, "rollouts": rollouts,
        "cached_vs_prefix_episode_parity": {
            "matched": parity, "episodes": len(batch)},
        "cpu_latency": times,
        "train_step_p50_ms": {
            k: round(statistics.median(v), 4) for k, v in training_ms.items()},
        "peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "losses": losses,
        "limitations": [
            "synthetic MZ-009 material-fork world, not native CyberSand",
            "simple diagonal input-gated recurrence, not a pretrained S4/Mamba",
            "equal sampled minibatches and updates, not equal FLOPs/parameters",
            "independently executed host-legal closed-loop trajectories",
            "no owner promotion or game changes; Draft/HOLD",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "trajectories.svg").write_text(gallery)
    (out / "REPORT.md").write_text(
        "# MZ-011 — input-gated diagonal state memory versus GRU\n\n"
        + f"Exact research source: `{source}`\n\n"
        + "Fork accuracy: "
        + ", ".join(f"{k}={v['test']['fork_accuracy']}"
                    for k, v in scores.items())
        + "\n\nCollision-free success: "
        + ", ".join(f"{k}={rollouts[k]['clean_success_rate']}"
                    for k in models)
        + "\n\nSynthetic only, research-only, Draft/HOLD.\n")
    print(json.dumps({
        "source_sha": source, "parameters": counts, "scores": scores,
        "rollouts_summary": {k: {
            s: v[s] for s in ("success_rate", "clean_success_rate",
                              "collisions", "raw_illegal")}
            for k, v in rollouts.items()},
        "cached_parity": report["cached_vs_prefix_episode_parity"],
        "cpu_latency": times, "peak_rss_mib": report["peak_rss_mib"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    }, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="mz011-evidence")
    p.add_argument("--train-seeds", type=int, default=160)
    p.add_argument("--eval-seeds", type=int, default=24)
    p.add_argument("--rollout-seeds", type=int, default=8)
    p.add_argument("--updates", type=int, default=800)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--hidden", type=int, default=224)
    p.add_argument("--lr", type=float, default=.003)
    run(p.parse_args())


if __name__ == "__main__":
    main()
