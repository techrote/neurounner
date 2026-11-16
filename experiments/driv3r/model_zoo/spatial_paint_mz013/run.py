"""MZ-013: real training on explicit 2-D movement, protected materials and PAINT.

All environments are original, deterministic synthetic research hosts. No CyberSand
game/runtime/authoring code is imported or modified. The 20M causal transformer
is a randomly initialized compact architecture, not a pretrained SLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import statistics
import time
from pathlib import Path

ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN", "PAINT", "RETRACE", "WAIT", "STOP")
LEFT, RIGHT, UP, DOWN, PAINT, RETRACE, WAIT, STOP = range(len(ACTIONS))
D = 155
MAX_T = 64
SCHEMA = "MZ013-SPATIAL-PAINT-v1"
PARENT_SHA = "7e6c3e416588549a2bfe6d83c5d75e948d64f7f3"


class World:
    """Three stations on a genuine 2-D cell grid, with legal move/paint/retrace.

    At each station one mineral is safe and the other protected. The safe
    identity = early beacon XOR delayed inversion. At the branch the earlier
    instructions are absent. Entering the protected cell creates a physical
    detour and a collision, requiring RETRACE to restore the path.
    """
    def __init__(self, seed: int, cue: int, invert: int, ood: bool = False):
        rng = random.Random(seed * 23117 + 1413)
        gaps = (5, 8) if ood else (3, 5)
        self.stations = []
        x = 1
        for _ in range(3):
            x += rng.randint(*gaps)
            self.stations.append(x)
        self.last = x
        self.width = x + 2
        self.top = [rng.randrange(2) for _ in self.stations]
        self.decoration = [rng.random() for _ in range(4)]
        self.cue, self.invert = cue, invert
        self.safe = cue ^ invert
        self.grid = [[1] * self.width for _ in range(7)]
        for xx in range(1, self.last + 1):
            self.grid[3][xx] = 0
        for i, sx in enumerate(self.stations):
            self.grid[2][sx] = 2 + self.top[i]
            self.grid[4][sx] = 2 + (1 - self.top[i])
        self.x, self.y = 1, 3
        self.path = []
        self.tick = 0
        self.painted = set()
        self.collisions = 0
        self.hazard = False
        self.last_action = WAIT
        self.done = False
        self.failed = False
        self.trace = [(self.x, self.y)]

    def tile(self, x, y):
        return self.grid[y][x] if 0 <= y < 7 and 0 <= x < self.width else 1

    def station_index(self):
        try:
            return self.stations.index(self.x)
        except ValueError:
            return -1

    def legal(self):
        mask = [0] * len(ACTIONS)
        mask[STOP] = 1
        if self.done:
            return mask
        if self.tick < 4:
            mask[WAIT] = 1
        elif self.hazard:
            mask[RETRACE] = int(bool(self.path))
        elif self.y != 3:
            mask[RETRACE] = int(bool(self.path))
            mask[PAINT] = int(self.tile(self.x, self.y) in (2, 3))
        else:
            mask[RIGHT] = int(self.x < self.last)
            mask[LEFT] = int(self.x > 1)
            mask[RETRACE] = int(bool(self.path))
            i = self.station_index()
            if i >= 0 and i not in self.painted:
                mask[UP] = mask[DOWN] = 1
        return mask

    def observe(self):
        vector = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                tile = self.tile(self.x + dx, self.y + dy)
                vector.extend(float(j == tile) for j in range(5))
        vector.extend((self.x / 26, self.y / 6, len(self.painted) / 3,
                       self.tick / MAX_T, float(self.hazard),
                       float(self.tick < 4)))
        vector.extend(self.legal())
        vector.extend(float(j == self.last_action) for j in range(len(ACTIONS)))
        vector.extend((float(self.tick < 2 and self.cue == 0) * 2,
                       float(self.tick < 2 and self.cue == 1) * 2,
                       float(2 <= self.tick < 4 and self.invert == 0) * 2,
                       float(2 <= self.tick < 4 and self.invert == 1) * 2))
        vector.extend(self.decoration)
        assert len(vector) == D
        return vector

    def teacher(self):
        if self.tick < 4:
            return WAIT
        if self.hazard:
            return RETRACE
        if self.y != 3:
            return PAINT if self.tile(self.x, self.y) in (2 + self.safe,) else RETRACE
        if len(self.painted) == 3:
            return STOP
        i = self.station_index()
        if i >= 0 and i not in self.painted:
            return UP if self.top[i] == self.safe else DOWN
        return RIGHT

    def step(self, action):
        if self.done:
            raise RuntimeError("step after terminal")
        if not self.legal()[action]:
            raise ValueError(f"illegal host action {ACTIONS[action]} at {self.x},{self.y}")
        self.last_action = action
        self.tick += 1
        if action == STOP:
            self.done = True
            self.failed = not (len(self.painted) == 3 and self.y == 3)
        elif action == RETRACE:
            self.x, self.y = self.path.pop()
            self.hazard = False
        elif action in (LEFT, RIGHT, UP, DOWN):
            dx, dy = {LEFT: (-1, 0), RIGHT: (1, 0),
                      UP: (0, -1), DOWN: (0, 1)}[action]
            self.path.append((self.x, self.y))
            self.x += dx
            self.y += dy
            if self.tile(self.x, self.y) in (2, 3) and (
                    self.tile(self.x, self.y) != 2 + self.safe):
                self.collisions += 1
                self.hazard = True
        elif action == PAINT:
            idx = self.station_index()
            if idx < 0 or self.tile(self.x, self.y) != 2 + self.safe:
                self.collisions += 1
                self.hazard = True
            else:
                self.grid[self.y][self.x] = 4
                self.painted.add(idx)
        self.trace.append((self.x, self.y))
        return self.done


def episode(seed, cue, inv, bad=-1, ood=False):
    w = World(seed, cue, inv, ood)
    observations, targets = [], []
    injected = False
    for _ in range(MAX_T):
        observations.append(w.observe())
        action = w.teacher()
        targets.append(action)
        if (bad >= 0 and not injected and w.y == 3 and
                w.station_index() == bad and
                action in (UP, DOWN)):
            executed = DOWN if action == UP else UP
            injected = True
        else:
            executed = action
        w.step(executed)
        if w.done:
            break
    assert w.done and not w.failed and w.painted == {0, 1, 2}, (
        seed, cue, inv, bad, w.tick, w.painted)
    return observations, targets, (seed, cue, inv, bad)


def corpus(seeds, ood=False):
    return [episode(s, cue, inv, bad, ood) for s in seeds
            for cue in range(2) for inv in range(2) for bad in (-1, 1)]


def tensors(torch, rows):
    x = torch.zeros((len(rows), MAX_T, D), dtype=torch.float32)
    y = torch.full((len(rows), MAX_T), -100, dtype=torch.long)
    for i, (observations, actions, _) in enumerate(rows):
        assert len(observations) <= MAX_T
        x[i, :len(observations)] = torch.tensor(observations)
        y[i, :len(actions)] = torch.tensor(actions)
    return x, y


def models(torch, width=656):
    nn = torch.nn

    class DiagCell(nn.Module):
        def __init__(self, d, hidden=224):
            super().__init__()
            self.proj = nn.Linear(d, hidden * 2)
            self.hidden = hidden
            with torch.no_grad():
                self.proj.bias[:hidden].fill_(3.)

        def sequence(self, x):
            retain, value = self.proj(x).chunk(2, -1)
            retain, value = retain.sigmoid(), value.tanh()
            state = x.new_zeros((x.shape[0], self.hidden))
            steps = []
            for t in range(x.shape[1]):
                state = retain[:, t] * state + (1 - retain[:, t]) * value[:, t]
                steps.append(state)
            return torch.stack(steps, 1)

        def step(self, x, state=None):
            retain, value = self.proj(x).chunk(2, -1)
            if state is None:
                state = x.new_zeros((x.shape[0], self.hidden))
            return retain.sigmoid() * state + (1 - retain.sigmoid()) * value.tanh()

    class Diagonal(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(D)
            self.cells = nn.ModuleList((DiagCell(D), DiagCell(224)))
            self.head = nn.Sequential(nn.LayerNorm(D + 224),
                                      nn.Linear(D + 224, 96), nn.GELU(),
                                      nn.Linear(96, len(ACTIONS)))
        def forward(self, x):
            z = self.norm(x)
            h = z
            for cell in self.cells:
                h = cell.sequence(h)
            return self.head(torch.cat((z, h), -1))
        def step(self, x, state=None):
            z = self.norm(x)
            h = z
            states = []
            for i, cell in enumerate(self.cells):
                h = cell.step(h, None if state is None else state[i])
                states.append(h)
            return self.head(torch.cat((z, h), -1)), tuple(states)

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(D)
            self.core = nn.GRU(D, 128, 2, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(D + 128),
                                      nn.Linear(D + 128, 96), nn.GELU(),
                                      nn.Linear(96, len(ACTIONS)))
        def forward(self, x):
            z = self.norm(x)
            h, _ = self.core(z)
            return self.head(torch.cat((z, h), -1))
        def step(self, x, state=None):
            z = self.norm(x)
            h, new = self.core(z[:, None, :], state)
            return self.head(torch.cat((z, h[:, 0]), -1)), new

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(D, width)
            self.pos = nn.Embedding(MAX_T, width)
            layer = nn.TransformerEncoderLayer(
                width, 8, dim_feedforward=width * 4, dropout=0.,
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, len(ACTIONS))
        def forward(self, x):
            t = x.shape[1]
            h = self.project(x) + self.pos(
                torch.arange(t, device=x.device))[None, :, :]
            causal = torch.ones((t, t), device=x.device, dtype=torch.bool).triu(1)
            return self.head(self.norm(self.encoder(h, mask=causal)))

    class Reactive(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 128),
                                      nn.GELU(), nn.Linear(128, len(ACTIONS)))
        def forward(self, x):
            return self.core(x)

    return {"transformer20m": Transformer(), "diagonal": Diagonal(),
            "gru": GRU(), "reactive": Reactive()}


def single(torch, net, obs, history, state=None, cache=True, context=None):
    x = torch.tensor([obs], dtype=torch.float32)
    if cache and hasattr(net, "step"):
        logits, state = net.step(x, state)
        return logits[0], state
    history.append(obs)
    recent = history if context is None else history[-context:]
    logits = net(torch.tensor([recent], dtype=torch.float32))[0, -1]
    return logits, None


def play(torch, net, seed, cue, inv, *, ood=False, context=None, cached=True):
    world = World(seed, cue, inv, ood)
    history, state, actions, illegal = [], None, [], 0
    with torch.inference_mode():
        for _ in range(MAX_T):
            logits, state = single(
                torch, net, world.observe(), history, state,
                cache=cached and context is None, context=context)
            legal = world.legal()
            raw = int(logits.argmax())
            illegal += int(not legal[raw])
            masked = logits.masked_fill(
                torch.tensor([not ok for ok in legal]), -1e9)
            chosen = int(masked.argmax())
            actions.append((world.x, world.y, ACTIONS[chosen]))
            world.step(chosen)
            if world.done:
                break
    success = world.done and not world.failed and len(world.painted) == 3
    return {"success": success, "clean": success and world.collisions == 0,
            "paints": len(world.painted), "collisions": world.collisions,
            "illegal_raw": illegal, "steps": world.tick,
            "actions": actions, "path": world.trace,
            "stations": world.stations}


def rollouts(torch, net, seeds, *, ood=False, context=None, cached=True):
    results = [play(torch, net, s, c, i, ood=ood, context=context, cached=cached)
               for s in seeds for c in (0, 1) for i in (0, 1)]
    n = len(results)
    return {"episodes": n,
            "success_rate": round(sum(r["success"] for r in results) / n, 5),
            "clean_rate": round(sum(r["clean"] for r in results) / n, 5),
            "protected_collisions": sum(r["collisions"] for r in results),
            "total_paints": sum(r["paints"] for r in results),
            "raw_illegal": sum(r["illegal_raw"] for r in results),
            "mean_steps": round(sum(r["steps"] for r in results) / n, 3),
            "sample": results[:4]}


def evaluate(torch, net, x, y, batch=12):
    correct = total = branch_correct = branch_total = 0
    paint_correct = paint_total = illegal = 0
    with torch.inference_mode():
        for j in range(0, len(y), batch):
            yy = y[j:j+batch]
            xx = x[j:j+batch]
            t = int((yy != -100).sum(1).max())
            logits = net(xx[:, :t])
            pred = logits.argmax(-1)
            labels = yy[:, :t]
            valid = labels != -100
            branch = (labels == UP) | (labels == DOWN)
            painted = labels == PAINT
            total += int(valid.sum())
            correct += int(((pred == labels) & valid).sum())
            branch_total += int(branch.sum())
            branch_correct += int(((pred == labels) & branch).sum())
            paint_total += int(painted.sum())
            paint_correct += int(((pred == labels) & painted).sum())
            masks = xx[:, :t, 131:139]
            illegal += int((((masks.gather(-1, pred[..., None]).squeeze(-1) < .5)
                             & valid)).sum())
    return {"accuracy": round(correct / max(1, total), 5),
            "branch_accuracy": round(branch_correct / max(1, branch_total), 5),
            "branch_count": branch_total,
            "paint_accuracy": round(paint_correct / max(1, paint_total), 5),
            "paint_count": paint_total, "raw_illegal": illegal,
            "valid_actions": total}


def timing(torch, net, *, cached):
    obs = torch.zeros((1, D))
    x = torch.zeros((1, 32, D))
    elapsed = []
    with torch.inference_mode():
        for i in range(28):
            t = time.perf_counter()
            if cached and hasattr(net, "step"):
                net.step(obs, None)
            else:
                net(x)
            if i >= 8:
                elapsed.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(statistics.median(elapsed), 4),
            "p95_ms": round(sorted(elapsed)[18], 4),
            "mode": "cached_single" if cached and hasattr(net, "step") else "prefix_32",
            "batch": 1}


def svg_gallery(rollouts_data):
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="550" viewBox="0 0 1000 550">',
              '<rect width="1000" height="550" fill="#111827"/>',
              '<text x="20" y="28" fill="#fff" font-size="17">MZ-013 | original closed-loop 2-D paint / protected-material trajectories</text>']
    for row, (name, data) in enumerate(rollouts_data.items()):
        yy = 48 + 120 * row
        pieces.append(f'<text x="16" y="{yy+15}" fill="#fff" font-size="12">{name}</text>')
        for col, trace in enumerate(data["sample"][:4]):
            xx = 178 + col*205
            pieces.append(f'<rect x="{xx}" y="{yy}" width="190" height="106" rx="5" fill="#243349"/>')
            pts = trace["path"]
            poly = " ".join(f"{xx+9+x*7},{yy+14+y*10}" for x,y in pts)
            pieces.append(f'<polyline points="{poly}" fill="none" stroke="#93c5fd" stroke-width="1.5"/>')
            for sx in trace["stations"]:
                pieces.append(f'<circle cx="{xx+9+sx*7}" cy="{yy+44}" r="3.8" fill="#fbbf24"/>')
            pieces.append(f'<text x="{xx+5}" y="{yy+98}" fill="#e5e7eb" font-size="10">paint {trace["paints"]}/3, hits {trace["collisions"]}, clean {int(trace["clean"])}</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)+"\n"


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(61313)
    sampler = torch.Generator().manual_seed(61317)
    train = corpus(range(1000, 1000 + args.train_seeds))
    dev = corpus(range(3000, 3000 + args.eval_seeds))
    test = corpus(range(5000, 5000 + args.eval_seeds))
    ood = corpus(range(6000, 6000 + args.eval_seeds), ood=True)
    x, y = tensors(torch, train)
    xd, yd = tensors(torch, dev)
    xt, yt = tensors(torch, test)
    xo, yo = tensors(torch, ood)
    nets = models(torch, args.width)
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in nets.items()}
    assert 20000000 <= counts["transformer20m"] <= 30000000, counts
    optim = {k: torch.optim.AdamW(n.parameters(), lr=args.lr)
             for k, n in nets.items()}
    weights = torch.tensor([1., 1., 2., 2., 3., 2., .5, 2.])
    losses = {k: [] for k in nets}
    for update in range(args.updates):
        idx = torch.randint(len(x), (args.batch,), generator=sampler)
        labels = y[idx]
        limit = int((labels != -100).sum(1).max())
        for name, net in nets.items():
            net.train()
            optim[name].zero_grad(set_to_none=True)
            pred = net(x[idx, :limit])
            loss = torch.nn.functional.cross_entropy(
                pred.transpose(1, 2), labels[:, :limit],
                weight=weights, ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {name} at {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optim[name].step()
            if update == 0 or (update+1) % 25 == 0:
                losses[name].append([update+1, round(float(loss.detach()), 6)])
        if (update+1) % 50 == 0:
            print(f"MZ-013 {update+1}/{args.updates} same minibatches/model", flush=True)
    for net in nets.values():
        net.eval()
    scores = {
        name: {"dev": evaluate(torch, net, xd, yd),
               "test": evaluate(torch, net, xt, yt),
               "shifted_spacing": evaluate(torch, net, xo, yo)}
        for name, net in nets.items()}
    seed_family = range(7000, 7000+args.rollout_seeds)
    trajectories = {
        name: rollouts(torch, net, seed_family, ood=True)
        for name, net in nets.items()}
    # A short-context ablation tests the memory claim rather than just accuracy.
    trajectories["diagonal_recent_4"] = rollouts(
        torch, nets["diagonal"], seed_family, ood=True, context=4, cached=False)
    # Compare cached recurrent state with prefix-replay on EXACT same worlds.
    parity = {}
    for name in ("diagonal", "gru"):
        pairs = [(play(torch, nets[name], s, c, i, ood=True),
                  play(torch, nets[name], s, c, i, ood=True, cached=False))
                 for s in range(7000, 7002) for c in range(2) for i in range(2)]
        parity[name] = {"matched": sum(a == b for a, b in pairs),
                        "episodes": len(pairs)}
    latency = {name: timing(torch, net, cached=name in ("diagonal", "gru"))
               for name, net in nets.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = os.getenv("GITHUB_SHA", "LOCAL-UNVERIFIED")
    checkpoint = out/"models.pt"
    torch.save({"schema": SCHEMA, "source_sha": source, "parent_sha": PARENT_SHA,
                "width": args.width, "actions": ACTIONS,
                "models": {k: net.state_dict() for k, net in nets.items()}}, checkpoint)
    report = {
        "schema": SCHEMA, "source_sha": source, "parent_sha": PARENT_SHA,
        "parameters": counts,
        "training": {"updates_per_model": args.updates, "batch": args.batch,
                     "train_seed_families": args.train_seeds,
                     "eval_seed_families": args.eval_seeds,
                     "train_episodes": len(train), "test_episodes": len(test),
                     "shifted_episodes": len(ood), "learning_rate": args.lr,
                     "disjoint_seed_families": True},
        "scores": scores, "closed_loop": trajectories, "cached_parity": parity,
        "cpu_latency": latency,
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
        "losses": losses,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "limitations": [
            "original 2-D synthetic corridor/station host, not native CyberSand",
            "one station pair per stop; no general 2-D pathfinding or fluid dynamics",
            "randomly initialized 20M causal transformer, not pretrained language model",
            "equal sample exposure/updates, not matched FLOPs or optimized learning rates",
            "host legal-action masks apply at execution; raw illegal predictions recorded",
            "Draft/HOLD; no production changes or game integration",
        ],
    }
    (out/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    (out/"trajectories.svg").write_text(svg_gallery(
        {k: trajectories[k] for k in nets}))
    (out/"REPORT.md").write_text(
        "# MZ-013: real spatial painting and protected-cell recovery\n\n"
        + f"Exact research source: {source}\n\n"
        + "\n".join(f"- {k}: TEST branch {v['test']['branch_accuracy']}, "
                    f"PAINT {v['test']['paint_accuracy']}, "
                    f"OOD clean {trajectories[k]['clean_rate']}"
                    for k, v in scores.items())
        + "\n\nSynthetic research only. Draft/HOLD. No game modification.\n")
    print(json.dumps({
        "source_sha": source, "parameters": counts, "scores": scores,
        "ood_clean_rate": {k:v["clean_rate"] for k,v in trajectories.items()},
        "cached_parity": parity, "cpu_latency": latency,
        "peak_rss_mib": report["peak_rss_mib"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    }, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mz013-evidence")
    parser.add_argument("--train-seeds", type=int, default=96)
    parser.add_argument("--eval-seeds", type=int, default=16)
    parser.add_argument("--rollout-seeds", type=int, default=4)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--width", type=int, default=656)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
