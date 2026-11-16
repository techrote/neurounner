"""MZ-008: counterfactual transient-material game-memory policy experiment.

External synthetic CyberSand-like worlds. No native/Godot/Workbench imports or
production game changes. The authoritative material/stencil cue is visible
three observations before the action and absent from the current observation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import time

from experiments.driv3r.model_zoo.mamba130.world import ACTIONS, DIRS, World
from experiments.driv3r.model_zoo.mamba130.counterfactual import raw_features

SCHEMA = "MZ008-TRANSIENT-LAW-v1"
WIDTH = 126
HISTORY = 4
ACTION_SET = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH")


def neutral(features):
    v = list(features)
    if len(v) != 114:
        raise ValueError("unexpected CyberSand research percept width")
    v[4:6] = (0., 0.)
    return v


def goals(world):
    result = []
    for dx, dy in DIRS:
        choices = [(world.pos[0] + dx*k, world.pos[1] + dy*k)
                   for k in range(1, 6)]
        goal = next((p for p in choices if world.safe(p)), None)
        if goal is None:
            return ()
        result.append(goal)
    return tuple(result)


def encode(history, current, rule, distractor, direction):
    """One authoritative earlier cue; two irrelevant subsequent color cues.

    Four present-time goal-direction bits are visible. The hidden rule must
    be remembered and composed with the currently visible direction.
    """
    if len(history) != 3 or rule not in range(4) or direction not in range(4):
        raise ValueError("invalid temporal input")
    result = []
    for tick, old in enumerate(history):
        hue = rule if tick == 0 else (distractor + tick) % 4
        cue = [float(hue == k) for k in range(4)]
        cue += [float(tick == 0)]
        cue += [float(tick == j) for j in range(3)]
        result.append(neutral(old) + cue + [0.] * 4)
    result.append(list(current) + [0.] * 8 +
                  [float(direction == i) for i in range(4)])
    if any(len(v) != WIDTH for v in result):
        raise AssertionError("bad observation dimensions")
    return result


def make_rows(seeds, steps=18):
    """Balanced rule x goal quartet at each eligible evolving game snapshot.

    Current observation is *identical* for every rule in the same quartet.
    Sequence histories come only from past world observations. All labelled
    actions are host-legal. World evolution is independent of hidden rule.
    """
    rows = []
    for seed in seeds:
        world = World.make(seed)
        rng = random.Random(seed * 101 + 17)
        previous = []
        for turn in range(steps):
            original_goal = world.goal
            observed = raw_features(world.prompt())
            if len(previous) == 3 and all(a in world.legal() for a in ACTION_SET):
                destinations = goals(world)
                if destinations:
                    for direction, destination in enumerate(destinations):
                        world.goal = destination
                        current = raw_features(world.prompt())
                        for law in range(4):
                            label = ACTIONS.index(ACTION_SET[(direction + law) % 4])
                            rows.append((encode(previous, current, law, seed+turn,
                                                direction), label, seed, turn,
                                         direction, law, tuple(world.legal())))
            world.goal = original_goal
            previous.append(neutral(observed))
            previous = previous[-3:]
            action = world.teacher(rng)
            if action == "STOP":
                break
            world.apply(action)
    return rows


def verify_rows(train, dev, test):
    seeds = [set(r[2] for r in partition) for partition in (train, dev, test)]
    if any(seeds[i] & seeds[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("train/dev/test world seed leakage")
    if min(len(train), len(dev), len(test)) < 100:
        raise AssertionError("insufficient scenario coverage")
    for partition in (train, dev, test):
        by_current = defaultdict(list)
        for row in partition:
            context, label, seed, turn, direction, law, legal = row
            assert ACTIONS[label] in legal and all(a in legal for a in ACTION_SET)
            by_current[seed, turn, direction].append((context[-1], label))
        for quartet in by_current.values():
            assert len(quartet) == 4
            assert len(set(y for _, y in quartet)) == 4
            assert all(x == quartet[0][0] for x, _ in quartet)
        freq = Counter(r[1] for r in partition)
        assert len(freq) == 4 and len(set(freq.values())) == 1


def make_networks(torch, arch):
    nn = torch.nn
    hidden, layers = (128, 2) if arch == "small" else (640, 8)

    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(WIDTH)
            self.gru = nn.GRU(WIDTH, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden+WIDTH),
                                      nn.Linear(hidden+WIDTH, 128), nn.GELU(),
                                      nn.Linear(128, len(ACTIONS)))

        def forward(self, x):
            normalized = self.norm(x)
            memory = self.gru(normalized)[0][:, -1, :]
            return self.head(torch.cat((memory, normalized[:, -1, :]), -1))

    # A strong full-context MLP as well as a current-observation-only control.
    flat = nn.Sequential(nn.LayerNorm(WIDTH*HISTORY),
                         nn.Linear(WIDTH*HISTORY, 256), nn.GELU(),
                         nn.Linear(256, len(ACTIONS)))
    current = nn.Sequential(nn.LayerNorm(WIDTH),
                            nn.Linear(WIDTH, 128), nn.GELU(),
                            nn.Linear(128, len(ACTIONS)))
    return Recurrent(), flat, current


def metrics(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("bad score length")
    actual = [r[1] for r in rows]
    accuracy = sum(a == b for a, b in zip(actual, predictions)) / len(rows)
    invalid = sum(ACTIONS[p] not in r[6] for r, p in zip(rows, predictions))
    f1 = []
    for action in ACTION_SET:
        a = ACTIONS.index(action)
        support = sum(y == a for y in actual)
        tp = sum(y == a and p == a for y, p in zip(actual, predictions))
        predicted = sum(p == a for p in predictions)
        recall = tp / support if support else 0.
        precision = tp / predicted if predicted else 0.
        f1.append(2 * recall * precision / (recall + precision)
                  if recall + precision else 0.)
    paired = {}
    for name, key in (
        ("law_swap", lambda r: (r[2], r[3], r[4])),
        ("goal_swap", lambda r: (r[2], r[3], r[5])),
    ):
        groups = defaultdict(list)
        for row, prediction in zip(rows, predictions):
            groups[key(row)].append((row[1], int(prediction)))
        total = both = switches = 0
        for variants in groups.values():
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    a, pa = variants[i]; b, pb = variants[j]
                    if a == b:
                        continue
                    total += 1
                    both += int(a == pa and b == pb)
                    switches += int(pa != pb)
        paired[name] = {
            "changed_pairs": total, "both_correct": both,
            "both_correct_rate": round(both / total, 4) if total else 0.,
            "switch_rate": round(switches / total, 4) if total else 0.,
        }
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1)/len(f1), 4),
        "raw_invalid": invalid,
        "pairs": paired,
    }


def experiment(args):
    if min(args.train_seeds, args.dev_seeds, args.steps, args.updates,
           args.batch_size, args.eval_batch) <= 0:
        raise ValueError("positive budget required")
    import torch
    from PIL import Image, ImageDraw
    torch.set_num_threads(2)
    torch.manual_seed(8138)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = os.environ.get("GITHUB_SHA", "LOCAL-UNVERIFIED")
    train = make_rows(range(3000, 3000+args.train_seeds), args.steps)
    dev = make_rows(range(5000, 5000+args.dev_seeds), args.steps)
    test = make_rows(range(7000, 7000+args.dev_seeds), args.steps)
    verify_rows(train, dev, test)
    X = torch.tensor([r[0] for r in train], dtype=torch.float32)
    Y = torch.tensor([r[1] for r in train], dtype=torch.long)
    warm = torch.tensor([ACTIONS.index(ACTION_SET[r[5]]) for r in train])
    D = torch.tensor([r[0] for r in dev], dtype=torch.float32)
    T = torch.tensor([r[0] for r in test], dtype=torch.float32)
    recurrent, flat, current = make_networks(torch, args.arch)
    nets = dict(recurrent=recurrent, flat=flat, current=current)
    counts = {name: sum(p.numel() for p in net.parameters())
              for name, net in nets.items()}
    if args.arch == "small":
        assert 100_000 < counts["recurrent"] < 1_000_000
    else:
        assert 18_000_000 < counts["recurrent"] < 22_000_000
    opts = {
        "recurrent": torch.optim.AdamW(recurrent.parameters(), lr=args.lr),
        "flat": torch.optim.AdamW(flat.parameters(), lr=.0012),
        "current": torch.optim.AdamW(current.parameters(), lr=.0012),
    }
    def forward(name, frames):
        if name == "flat":
            return flat(frames.flatten(1))
        if name == "current":
            return current(frames[:, -1, :])
        return recurrent(frames)
    generator = torch.Generator().manual_seed(9404)
    losses = {name: [] for name in nets}
    update_ms = []
    for step in range(args.updates):
        ids = torch.randint(len(train), (args.batch_size,), generator=generator)
        label = warm[ids] if step < args.updates // 3 else Y[ids]
        for name in nets:
            start = time.perf_counter()
            opts[name].zero_grad(set_to_none=True)
            logits = forward(name, X[ids])
            loss = torch.nn.functional.cross_entropy(logits, label)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite "+name+" optimizer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nets[name].parameters(), 1.)
            opts[name].step()
            losses[name].append(round(float(loss.detach()), 5))
            if name == "recurrent":
                update_ms.append((time.perf_counter()-start)*1000)
        if step % 100 == 0:
            print("train", args.arch, step, "loss", losses["recurrent"][-1],
                  flush=True)
    for net in nets.values():
        net.eval()
    forward_ms = []
    def predict(name, frames):
        output = []
        with torch.inference_mode():
            for index in range(0, len(frames), args.eval_batch):
                batch = frames[index:index+args.eval_batch]
                start = time.perf_counter()
                logits = forward(name, batch)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite inference "+name)
                if name == "recurrent":
                    forward_ms.append((time.perf_counter()-start)*1000/len(batch))
                output.extend(logits.argmax(-1).tolist())
        return output
    scores = {
        name: {"dev": metrics(dev, predict(name, D)),
               "test": metrics(test, predict(name, T))}
        for name in nets
    }
    without_history = T.clone()
    without_history[:, :3, :] = 0.
    scores["recurrent_no_history"] = {
        "test": metrics(test, predict("recurrent", without_history))
    }
    wrong_law = T.clone()
    for i in range(0, len(test), 4):
        wrong_law[i:i+4, :3, :] = T[[i+1, i+2, i+3, i], :3, :]
    scores["recurrent_wrong_law"] = {
        "test": metrics(test, predict("recurrent", wrong_law))
    }
    scores["oracle"] = {
        "test": metrics(test, [r[1] for r in test])
    }
    scores["majority"] = {
        "test": metrics(test, [ACTIONS.index(ACTION_SET[0])]*len(test))
    }
    predictions = predict("recurrent", T)
    # Four original held-out physical-like percepts x four expired cues;
    # blue cells are painted; orange protected; charcoal obstacles.
    img = Image.new("RGB", (4*220, 4*220), "#f9f9f9")
    draw = ImageDraw.Draw(img)
    quartet_starts, visited = [], set()
    for i, row in enumerate(test):
        key = (row[2], row[3], row[4])
        if key not in visited:
            quartet_starts.append(i)
            visited.add(key)
        if len(quartet_starts) == 4:
            break
    gallery = []
    for column, base in enumerate(quartet_starts):
        for law in range(4):
            row = test[base+law]
            obs = row[0][-1]
            ox, oy = column*220, law*220
            for j in range(5):
                for k in range(5):
                    cell = obs[7+(j*5+k)*4:7+(j*5+k+1)*4]
                    symbol = max(range(4), key=lambda z: cell[z])
                    color = ("#e7e7e7", "#30333a", "#cb753e", "#3a89c9")[symbol]
                    draw.rectangle((ox+k*40, oy+j*38, ox+k*40+37,
                                    oy+j*38+35), fill=color)
            wanted, got = ACTIONS[row[1]], ACTIONS[predictions[base+law]]
            draw.text((ox+4, oy+193),
                      f"cue={law} goal={row[4]} {got}/{wanted}", fill="#222222")
            gallery.append({
                "seed": row[2], "turn": row[3], "goal_dir": row[4],
                "expired_rule": law, "teacher": wanted, "prediction": got,
                "offered_actions": list(row[6]),
            })
    img.save(out/"counterfactual_gallery.png")
    checkpoint = out/f"{args.arch}_memory_policy.pt"
    torch.save({
        "schema": SCHEMA, "arch": args.arch, "source_sha": source,
        "actions": ACTIONS, "context_length": HISTORY, "width": WIDTH,
        "trained_updates": args.updates,
        "recurrent": recurrent.state_dict(), "flat": flat.state_dict(),
        "current": current.state_dict(),
    }, checkpoint)
    result = {
        "schema": SCHEMA, "source_sha": source,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": f"GRU-{args.arch}-source-fenced-memory-law",
        "parameters": counts, "train_examples": len(train),
        "dev_examples": len(dev), "test_examples": len(test),
        "train_seeds": args.train_seeds, "dev_seeds": args.dev_seeds,
        "updates": args.updates, "batch_size": args.batch_size,
        "losses": losses, "scores": scores, "gallery": gallery,
        "train_histogram": dict(Counter(ACTIONS[r[1]] for r in train)),
        "cpu_forward_amortized_p50_ms": round(statistics.median(forward_ms), 3),
        "optimizer_update_p50_ms": round(statistics.median(update_ms), 3),
        "peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "limitations": [
            "A synthetic expired material/stencil rule, NOT an actual CyberSand game feature.",
            "Only host-legal quartet actions on eligible synthetic situations.",
            "Four observations and scripted modulo-4 action labels, not native gameplay.",
            "Warmup first third teaches rule recall, then rule+goal composition.",
            "Same training indices, unequal FLOPs and capacity.",
            "CPU inference times amortized over batches; no single-action latency claim.",
            "No CyberSand game source/content changes; Draft/HOLD only.",
        ],
    }
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"REPORT.md").write_text(
        "# MZ-008 transient-material law memory experiment\n\n"
        f"Source: {source}; architecture: {args.arch}; "
        f"recurrent params: {counts['recurrent']}.\n\n"
        f"TRAIN/DEV/TEST {len(train)}/{len(dev)}/{len(test)}; "
        f"{args.updates} real minibatch optimizer updates per policy.\n\n"
        f"Recurrent TEST {json.dumps(scores['recurrent']['test'])}\n\n"
        f"Flat-memory TEST {json.dumps(scores['flat']['test'])}\n\n"
        f"Current-only TEST {json.dumps(scores['current']['test'])}\n\n"
        f"Ablated TEST {json.dumps(scores['recurrent_no_history']['test'])}\n\n"
        f"Wrong-law TEST {json.dumps(scores['recurrent_wrong_law']['test'])}\n\n"
        f"Checkpoint SHA256 {result['checkpoint_sha256']}\n\n"
        "No production game modifications, no native Workbench integration.\n")
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("losses","gallery")}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("small","large"), required=True)
    p.add_argument("--out", default="mz008-evidence")
    p.add_argument("--train-seeds", type=int, default=100)
    p.add_argument("--dev-seeds", type=int, default=18)
    p.add_argument("--steps", type=int, default=18)
    p.add_argument("--updates", type=int, default=550)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=.001)
    experiment(p.parse_args())

if __name__ == "__main__":
    main()
