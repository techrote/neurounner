"""MZ-012: real training of a rotary/phase-gated recurrent game controller.

All data is synthetic, action-driven CyberSand-style material-fork terrain;
no CyberSand gameplay, Workbench, native or Godot sources are modified.
This is a custom gated real-valued phase recurrence, NOT a pretrained LRU/S4.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, statistics, time
from pathlib import Path

from experiments.driv3r.model_zoo.material_forks_mz009 import run as host
from experiments.driv3r.model_zoo.diag_state_mz011 import run as prior

SCHEMA = "MZ012-PHASE-GATED-MEMORY-v1"
PARENT_SHA = "bcd759b8f8a351998d61829b3f527c965ca041d4"
MAX_T = 64


class VariableCorridorWorld(host.World):
    """Independent action-driven host variant with longer hidden-cue delays.

    Unlike MZ-009's fixed 8/6/6 corridors, all three corridor lengths are
    seed-variable. OOD corridors have delays outside the training interval.
    Host legality, protected material collisions, painting and RETRACE stay
    exactly those of the source-pinned MZ-009 world.
    """
    def __init__(self, seed, cue, inversion, ood=False):
        super().__init__(seed, cue, inversion)
        rng = random.Random(seed * 65711 + 12331)
        bounds = ((14, 20), (10, 14), (10, 14)) if ood else (
            (8, 13), (6, 10), (6, 10))
        self.lengths = tuple(rng.randint(lo, hi) for lo, hi in bounds)

    def observe(self):
        x = super().observe()
        x[5] = self.corridor / self.lengths[min(self.fork, 2)]
        x[6] = self.tick / MAX_T
        return x

    def step(self, action):
        if self.stage != "corridor":
            return super().step(action)
        if self.done:
            raise RuntimeError("step after terminal")
        if not self.legal()[action]:
            raise ValueError("unmasked illegal action")
        self.previous_action = action
        self.tick += 1
        if action == 4:
            self.done, self.failed = True, True
        else:
            self.corridor += 1
            if self.corridor >= self.lengths[self.fork]:
                self.stage = "fork"
        return self.done


def demonstration(seed, cue, inv, bad_fork=-1, ood=False):
    w = VariableCorridorWorld(seed, cue, inv, ood=ood)
    xs, ys, injected = [], [], False
    for _ in range(MAX_T):
        xs.append(w.observe())
        target = w.preferred_action()
        ys.append(target)
        force_bad = w.stage == "fork" and w.fork == bad_fork and not injected
        action = (3 - target) if force_bad else target
        injected |= force_bad
        w.step(action)
        if w.done:
            break
    assert w.done and not w.failed and w.painted == 3, (seed, cue, inv, w.stage)
    assert len(xs) <= MAX_T
    return xs, ys, (seed, cue, inv, bad_fork)


def dataset(seeds, ood=False):
    return [demonstration(s, cue, inv, bad, ood=ood)
            for s in seeds for cue in range(2) for inv in range(2)
            for bad in (-1, 0, 1)]


def tensors(torch, rows):
    x = torch.zeros((len(rows), MAX_T, host.D), dtype=torch.float32)
    y = torch.full((len(rows), MAX_T), -100, dtype=torch.long)
    for i, (obs, target, _) in enumerate(rows):
        x[i, :len(obs)] = torch.tensor(obs, dtype=torch.float32)
        y[i, :len(target)] = torch.tensor(target, dtype=torch.long)
    return x, y


def phase_model(torch, hidden=224):
    """Two real-valued rotary diagonal cells with learned phase and gating."""
    nn = torch.nn
    assert hidden % 2 == 0

    class RotaryCell(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.h = h
            self.proj = nn.Linear(d, 2*h)
            self.angle = nn.Parameter(torch.zeros(h // 2))
            self.decay = nn.Parameter(torch.full((h,), 5.0))
            with torch.no_grad():
                self.proj.bias[:h].fill_(2.7)

        def transition(self, gate, value, state, c, s):
            pair = state.reshape(state.shape[0], -1, 2)
            a, b = pair.unbind(-1)
            rotated = torch.stack((c*a - s*b, s*a + c*b), -1)
            rotated = rotated.reshape(state.shape)
            return gate * self.decay.sigmoid() * rotated + (1-gate)*value

        def sequence(self, x):
            gate, value = self.proj(x).chunk(2, -1)
            gate, value = gate.sigmoid(), value.tanh()
            state = x.new_zeros(x.shape[0], self.h)
            c, s = self.angle.cos(), self.angle.sin()
            outputs = []
            for t in range(x.shape[1]):
                state = self.transition(gate[:, t], value[:, t], state, c, s)
                outputs.append(state)
            return torch.stack(outputs, 1)

        def step(self, x, state=None):
            gate, value = self.proj(x).chunk(2, -1)
            if state is None:
                state = x.new_zeros(x.shape[0], self.h)
            return self.transition(gate.sigmoid(), value.tanh(), state,
                                   self.angle.cos(), self.angle.sin())

    class PhasePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(host.D)
            self.cells = nn.ModuleList([
                RotaryCell(host.D, hidden), RotaryCell(hidden, hidden)])
            self.head = nn.Sequential(
                nn.LayerNorm(host.D + hidden),
                nn.Linear(host.D + hidden, 96), nn.GELU(),
                nn.Linear(96, len(host.ACTIONS)))

        def forward(self, x):
            z = self.norm(x)
            h = z
            for cell in self.cells:
                h = cell.sequence(h)
            return self.head(torch.cat((z, h), -1))

        def step(self, obs, state=None):
            z = self.norm(obs)
            h, states = z, []
            for j, cell in enumerate(self.cells):
                h = cell.step(h, None if state is None else state[j])
                states.append(h)
            return self.head(torch.cat((z, h), -1)), tuple(states)

    return PhasePolicy()


def build_models(torch, hidden=224):
    gru, reactive = host.networks(torch, hidden=128, layers=2)
    return {
        "phase": phase_model(torch, hidden),
        "diagonal": prior.diagonal_model(torch, hidden),
        "small_gru": gru,
        "current_only": reactive,
    }


def counts(models):
    return {k: sum(p.numel() for p in v.parameters()) for k, v in models.items()}


def incremental(torch, net, obs, state):
    if hasattr(net, "step"):
        return net.step(obs, state)
    if hasattr(net, "core"):
        z = net.norm(obs).unsqueeze(1)
        out, next_state = net.core(z, state)
        return net.head(torch.cat((z, out), -1)[:, 0]), next_state
    return net(obs), None


def play(torch, net, seed, cue, inv, *, ood=False,
         cached=False, context=None):
    w = VariableCorridorWorld(seed, cue, inv, ood)
    history, state, illegal, decisions = [], None, 0, []
    with torch.inference_mode():
        for _ in range(MAX_T):
            x = w.observe()
            if cached:
                logits, state = incremental(
                    torch, net, torch.tensor([x], dtype=torch.float32), state)
                logits = logits[0]
            else:
                history.append(x)
                recent = history if context is None else history[-context:]
                logits = net(torch.tensor([recent], dtype=torch.float32))[0, -1]
            legal = w.legal()
            raw = int(logits.argmax())
            illegal += int(not legal[raw])
            action = int(logits.masked_fill(
                torch.tensor([not allowed for allowed in legal]), -1e9).argmax())
            if w.stage == "fork":
                decisions.append({
                    "fork": w.fork,
                    "chosen": host.ACTIONS[action],
                    "correct": action == w.preferred_action(),
                    "material_left": w.left_materials[w.fork],
                })
            w.step(action)
            if w.done:
                break
    success = w.done and not w.failed and w.painted == 3
    return {"success": success, "clean": success and w.collisions == 0,
            "collisions": w.collisions, "painted": w.painted,
            "illegal_raw": illegal, "steps": w.tick,
            "fork_decisions": decisions, "lengths": list(w.lengths)}


def rollouts(torch, net, seeds, *, ood=False, cached=False, context=None):
    rows = [play(torch, net, seed, cue, inv, ood=ood,
                 cached=cached, context=context)
            for seed in seeds for cue in range(2) for inv in range(2)]
    n = len(rows)
    return {"episodes": n, "eventual_rate": round(sum(x["success"] for x in rows)/n, 5),
            "clean_rate": round(sum(x["clean"] for x in rows)/n, 5),
            "protected_collisions": sum(x["collisions"] for x in rows),
            "raw_illegal": sum(x["illegal_raw"] for x in rows),
            "mean_steps": round(sum(x["steps"] for x in rows)/n, 3),
            "sample": rows[:4]}


def causal(torch, net, x, y):
    fork = (x[:, :, 1] > 0) & (y != -100)
    first = fork & (torch.cumsum(fork.int(), 1) == 1)
    changed = x.clone()
    changed[:, :2, 17:19] = changed[:, :2, 17:19].flip(-1)
    with torch.inference_mode():
        before = net(x).argmax(-1)
        after = net(changed).argmax(-1)
    return {
        "first_fork_pairs": int(first.sum()),
        "beacon_flip_action_switch_rate": round(
            float(((before != after) & first).sum())/max(1, int(first.sum())), 5),
    }


def latency(torch, net, steps, cached, samples=50):
    x = torch.zeros((1, steps, host.D), dtype=torch.float32)
    state = None
    with torch.inference_mode():
        for _ in range(8):
            if cached:
                incremental(torch, net, x[:, -1], state)
            else:
                net(x)
        measurements = []
        for _ in range(samples):
            start = time.perf_counter()
            if cached:
                incremental(torch, net, x[:, -1], state)
            else:
                net(x)
            measurements.append((time.perf_counter()-start)*1000)
    return {"p50_ms": round(statistics.median(measurements), 4),
            "p95_ms": round(sorted(measurements)[int(.95*(samples-1))], 4),
            "batch": 1, "sequence": steps, "cached": cached}


def gallery(torch, models, seeds):
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="990" height="460" viewBox="0 0 990 460">',
        '<rect width="990" height="460" fill="#121b28"/>',
        '<text x="18" y="25" fill="#fff" font-size="17">MZ-012 | long-corridor painting / protected terrain forks</text>',
    ]
    for row, (name, net) in enumerate(models.items()):
        yy = 45 + row * 99
        parts.append(f'<text x="18" y="{yy+15}" fill="#fff" font-size="12">{name}</text>')
        for col, seed in enumerate(seeds):
            xx = 170 + col * 204
            result = play(torch, net, seed, col%2, (col//2)%2, ood=True, cached=True)
            parts.append(f'<rect x="{xx}" y="{yy}" width="194" height="86" rx="6" fill="#26374a"/>')
            parts.append(f'<text x="{xx+6}" y="{yy+14}" fill="#e2e8f0" font-size="10">seed {seed} / span {result["lengths"]}</text>')
            for event in result["fork_decisions"][:3]:
                fx, fy = xx + 31 + event["fork"]*49, yy + 42
                parts.append(f'<circle cx="{fx}" cy="{fy}" r="10" fill="none" stroke="{("#2dd4bf" if event["correct"] else "#fb7185")}" stroke-width="2.5"/>')
                parts.append(f'<text x="{fx-4}" y="{fy+4}" fill="#fff" font-size="11">{("L" if event["chosen"]=="LEFT" else "R")}</text>')
            parts.append(f'<text x="{xx+6}" y="{yy+75}" fill="#fff" font-size="10">clean {int(result["clean"])} / protected hits {result["collisions"]} / paint {result["painted"]}</text>')
    parts.append('</svg>')
    return "\n".join(parts)+"\n"


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(61212)
    gen = torch.Generator().manual_seed(61216)
    train = dataset(range(1000, 1000+args.train_seeds))
    dev = dataset(range(3000, 3000+args.eval_seeds))
    test = dataset(range(5000, 5000+args.eval_seeds))
    shifted = dataset(range(6000, 6000+args.eval_seeds), ood=True)
    x, y = tensors(torch, train)
    xd, yd = tensors(torch, dev)
    xt, yt = tensors(torch, test)
    xs, ys = tensors(torch, shifted)
    models = build_models(torch, args.hidden)
    optim = {k: torch.optim.AdamW(m.parameters(), lr=args.lr)
             for k, m in models.items()}
    weights = torch.tensor([.5, 3., 3., 1.5, .5])
    sampled_losses, training_ms = {}, {}
    for name in models:
        sampled_losses[name], training_ms[name] = [], []
    for update in range(args.updates):
        ids = torch.randint(len(x), (args.batch,), generator=gen)
        for name, model in models.items():
            t = time.perf_counter()
            optim[name].zero_grad(set_to_none=True)
            logits = model(x[ids])
            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), y[ids],
                weight=weights, ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {name}, step {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim[name].step()
            training_ms[name].append((time.perf_counter()-t)*1000)
            if (update+1)%25==0 or update==0:
                sampled_losses[name].append([update+1, round(float(loss.detach()), 6)])
        if (update+1)%100==0:
            print(f"MZ-012 trained {update+1}/{args.updates} equal minibatches",flush=True)
    for model in models.values():
        model.eval()
    scores = {}
    for name, model in models.items():
        with torch.inference_mode():
            scores[name] = {
                "dev": host.class_scores(model(xd), yd, xd),
                "test": host.class_scores(model(xt), yt, xt),
                "shifted_long_test": host.class_scores(model(xs), ys, xs)}
        scores[name]["history_causality"] = causal(torch, model, xt, yt)
    rollout_seeds = range(7000, 7000+args.rollout_seeds)
    closed_loop = {
        name: {"in_distribution": rollouts(torch, model, rollout_seeds),
               "shifted_long": rollouts(torch, model, rollout_seeds, ood=True, cached=True)}
        for name, model in models.items()}
    for key in ("phase", "diagonal"):
        closed_loop[key]["recent_1"] = rollouts(
            torch, models[key], rollout_seeds, ood=True, context=1)
        closed_loop[key]["recent_8"] = rollouts(
            torch, models[key], rollout_seeds, ood=True, context=8)
    parity = {}
    for name, model in models.items():
        pairs = [(play(torch, model, s, cue, inv, ood=True),
                  play(torch, model, s, cue, inv, ood=True, cached=True))
                 for s in range(7000, 7002) for cue in range(2) for inv in range(2)]
        parity[name] = {
            "matched": sum(a == b for a,b in pairs), "episodes": len(pairs)}
    timings = {name: {"cached_single": latency(torch, model, 1, True),
                      "prefix_48": latency(torch, model, 48, False)}
               for name, model in models.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = os.environ.get("GITHUB_SHA", "LOCAL-UNVERIFIED")
    checkpoint = out/"models.pt"
    torch.save({
        "schema": SCHEMA, "source_sha": sha, "parent_sha": PARENT_SHA,
        "hidden": args.hidden, "actions": host.ACTIONS,
        "models": {k:m.state_dict() for k,m in models.items()},
    }, checkpoint)
    report = {
        "schema": SCHEMA, "source_sha": sha, "parent_sha": PARENT_SHA,
        "parameters": counts(models),
        "training": {"seed_families": args.train_seeds,
                     "evaluation_families": args.eval_seeds,
                     "updates_per_model": args.updates, "batch": args.batch,
                     "learning_rate": args.lr, "train_episodes": len(train),
                     "test_episodes": len(test),
                     "shifted_test_episodes": len(shifted),
                     "disjoint_seed_families": True},
        "scores": scores,
        "rollouts": closed_loop,
        "stateful_prefix_parity": parity,
        "cpu_latency": timings,
        "training_step_p50_ms": {name: round(statistics.median(t), 4)
                                 for name,t in training_ms.items()},
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
        "losses": sampled_losses,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "limitations": [
            "synthetic material-fork and painting outcome, no native CyberSand",
            "training variability and shifted corridors, not general OOD competence",
            "cached-state GRU comparison included; CPU runner is not physical game hardware",
            "same sampled minibatches/updates, not equal FLOPs or parameter count",
            "research-only Draft/HOLD; no game integration or production settings changed",
        ],
    }
    (out/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    (out/"trajectories.svg").write_text(gallery(
        torch, models, range(7000, 7004)))
    (out/"REPORT.md").write_text(
        "# MZ-012 — phase-gated recurrence on variable material corridors\n\n"
        + f"Source: \`{sha}\`\n\n"
        + "In-distribution and shifted-long held-out forks:\n\n"
        + "\n".join(
            f"- {name}: ID {s['test']['fork_accuracy']}, shifted "
            f"{s['shifted_long_test']['fork_accuracy']}, "
            f"clean long {closed_loop[name]['shifted_long']['clean_rate']}"
            for name,s in scores.items())
        + "\n\nSynthetic only; no production game changes; Draft/HOLD.\n")
    print(json.dumps({
        "source_sha": sha, "parameters": report["parameters"],
        "scores": scores,
        "long_clean_rates": {k:v["shifted_long"]["clean_rate"]
                             for k,v in closed_loop.items()},
        "stateful_prefix_parity": parity,
        "cpu_latency": timings,
        "peak_rss_mib": report["peak_rss_mib"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    }, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="mz012-evidence")
    p.add_argument("--train-seeds", type=int, default=128)
    p.add_argument("--eval-seeds", type=int, default=24)
    p.add_argument("--rollout-seeds", type=int, default=8)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=224)
    p.add_argument("--lr", type=float, default=.003)
    run(p.parse_args())


if __name__ == "__main__":
    main()
