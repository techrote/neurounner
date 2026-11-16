"""Self-contained synthetic game-policy population pilot (not native or an SLM).

No game imports, writes, production controls or fabricated native receipts.
All policy choices come from a host-computed finite legal menu. Real small
neural models are optimised and compared with three nonterminal controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NICHES = ("navigator", "branch_artist", "recoverer", "texture_worm",
          "material_cautious", "chaotic_brush")
DIRECTIONS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
FEATURES = ("bias", "paint", "retrace", "stop", "goal_progress", "onward",
            "blocked", "junction", "turn", "frontier", "material_edge",
            "paint_density", "early", "protected_near", "goal_switched",
            "direction_x", "direction_y", "guidance", "motif", "recovery_need")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % 2**32


def softmax(x):
    z = np.exp(np.clip(x - np.max(x), -45, 0))
    return z / z.sum()


@dataclass
class World:
    seed: int
    width: int = 22
    height: int = 22
    max_steps: int = 30
    obstacles: set = field(default_factory=set)
    protected: set = field(default_factory=set)
    painted: set = field(default_factory=set)
    parent: dict = field(default_factory=dict)
    cursor: tuple = (3, 11)
    goal: tuple = (18, 11)
    second_goal: tuple = (18, 18)
    steps: int = 0
    switched: bool = False
    heading: str = "E"
    history: list = field(default_factory=list)
    stopped: bool = False

    @classmethod
    def generate(cls, seed, max_steps=30):
        rng = np.random.default_rng(seed)
        w = cls(seed=seed, max_steps=max_steps)
        w.protected = {(x, y) for x in range(9, 12) for y in range(8, 14)}
        reserved = {(3, 11), (18, 11), (18, 18)}
        for y in range(1, w.height - 1):
            for x in range(1, w.width - 1):
                if (x, y) in w.protected or (x, y) in reserved:
                    continue
                if rng.random() < .11:
                    w.obstacles.add((x, y))
        wall_x = 7 + (seed % 3)
        gap = 4 + (seed % 12)
        for y in range(3, 19):
            if abs(y - gap) > 1 and (wall_x, y) not in w.protected and (wall_x, y) != (3, 11):
                w.obstacles.add((wall_x, y))
        w.painted = {w.cursor}
        return w

    def open(self, p):
        x, y = p
        return (1 <= x < self.width - 1 and 1 <= y < self.height - 1
                and p not in self.obstacles and p not in self.protected and p not in self.painted)

    def legal(self):
        if self.stopped or self.steps >= self.max_steps:
            return ["STOP"]
        actions = [a for a, (dx, dy) in DIRECTIONS.items()
                   if self.open((self.cursor[0] + dx, self.cursor[1] + dy))]
        if self.cursor in self.parent:
            actions.append("RETRACE")
        return actions + ["STOP"]

    def neighbours(self, p, occupied):
        return sum((p[0] + dx, p[1] + dy) in occupied for dx, dy in DIRECTIONS.values())

    def feature(self, action):
        if action not in self.legal():
            raise ValueError("action not in host-legal catalog")
        before = self.cursor
        move = action in DIRECTIONS
        retrace = action == "RETRACE"
        stop = action == "STOP"
        if move:
            dx, dy = DIRECTIONS[action]
            after = (before[0] + dx, before[1] + dy)
        elif retrace:
            after = self.parent[before]
        else:
            after = before
        progress = (abs(before[0] - self.goal[0]) + abs(before[1] - self.goal[1])
                    - abs(after[0] - self.goal[0]) - abs(after[1] - self.goal[1]))
        onward = sum(self.open((after[0] + dx, after[1] + dy)) for dx, dy in DIRECTIONS.values())
        hazards = self.neighbours(after, self.obstacles | self.protected)
        neighbours = self.neighbours(after, self.painted)
        junction = sum(p == before for p in self.parent.values()) >= 1
        old_heading = DIRECTIONS.get(self.heading, (1, 0))
        vec = DIRECTIONS[action] if move else (-old_heading[0], -old_heading[1]) if retrace else (0, 0)
        turn = float(move and vec != old_heading)
        goal_vec = np.array([self.goal[0] - before[0], self.goal[1] - before[1]], dtype=float)
        guidance = float(np.dot(goal_vec, np.array(vec)) / (np.linalg.norm(goal_vec) + 1)) / 10
        motif = float(move and (after[0] + 2 * after[1] + self.seed) % 4 == 0)
        no_exits = float(sum(self.open((before[0] + dx, before[1] + dy))
                             for dx, dy in DIRECTIONS.values()) <= 1)
        return np.asarray([
            1., float(move), float(retrace), float(stop), progress, onward / 4,
            hazards / 4, float(junction), turn, float(move and onward >= 2),
            hazards / 4, neighbours / 4, 1 - self.steps / self.max_steps,
            self.neighbours(after, self.protected) / 4, float(self.switched),
            vec[0], vec[1], guidance, motif, no_exits
        ], dtype=np.float64)

    def step(self, action):
        if action not in self.legal():
            raise ValueError(f"illegal model action: {action}")
        before = self.cursor
        if action in DIRECTIONS:
            dx, dy = DIRECTIONS[action]
            after = (before[0] + dx, before[1] + dy)
            if after in self.painted:
                raise AssertionError("paint overwrite")
            self.painted.add(after)
            self.parent[after] = before
            self.cursor = after
            self.heading = action
        elif action == "RETRACE":
            self.cursor = self.parent[before]
        else:
            self.stopped = True
        self.steps += 1
        self.history.append(dict(turn=self.steps, action=action, start=before,
                                 end=self.cursor, painted=len(self.painted)))
        if not self.switched and self.steps >= self.max_steps // 2:
            self.goal = self.second_goal
            self.switched = True

    def descriptor(self):
        a = [h["action"] for h in self.history]
        moves = [x for x in a if x in DIRECTIONS]
        turns = sum(x != y for x, y in zip(moves, moves[1:]))
        child_counts = {p: 0 for p in self.painted}
        for v in self.parent.values():
            child_counts[v] += 1
        junctions = sum(v > 1 for v in child_counts.values())
        return [
            round((len(self.painted) - 1) / max(1, self.max_steps), 4),
            round(a.count("RETRACE") / max(1, len(a)), 4),
            round(turns / max(1, len(moves) - 1), 4),
            round(junctions / max(1, len(self.painted)), 4),
            round((abs(self.cursor[0] - 3) + abs(self.cursor[1] - 11)) / 36, 4)
        ]


def utility(niche, x, world):
    f = dict(zip(FEATURES, x))
    if f["stop"]:
        return 0.0 if len(world.legal()) == 1 else -2.5
    base = .14 * f["paint"] - .08 * f["retrace"]
    if niche == "navigator":
        return base + 1.6 * f["goal_progress"] + .5 * f["onward"] + .2 * f["guidance"]
    if niche == "branch_artist":
        return (base + .85 * f["junction"] * f["paint"] + .55 * f["frontier"]
                + .55 * f["turn"] + .25 * f["retrace"])
    if niche == "recoverer":
        return (base + f["onward"] - .75 * f["blocked"]
                + .5 * f["retrace"] * f["recovery_need"] + .35 * f["goal_progress"])
    if niche == "texture_worm":
        return base + .75 * f["motif"] + .7 * f["turn"] + .35 * f["paint_density"] - .1 * f["blocked"]
    if niche == "material_cautious":
        return (base + .5 * f["goal_progress"] - 1.6 * f["protected_near"]
                - .8 * f["blocked"] + .5 * f["onward"])
    if niche == "chaotic_brush":
        return (base + .85 * f["turn"] + .5 * f["motif"] + .35 * f["frontier"]
                + .2 * f["goal_switched"])
    raise ValueError(niche)


class Policy:
    """24-hidden-unit tanh policy, trained by actual Adam gradient updates."""
    def __init__(self, seed, hidden=24, lr=.007, temperature=1.0):
        rng = np.random.default_rng(seed)
        self.p = dict(W1=rng.normal(0, .15, (len(FEATURES), hidden)),
                      b1=np.zeros(hidden), W2=rng.normal(0, .15, hidden), b2=np.zeros(1))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.updates = 0
        self.lr = lr
        self.temperature = temperature

    def logits(self, X):
        return np.tanh(X @ self.p["W1"] + self.p["b1"]) @ self.p["W2"] + self.p["b2"][0]

    def predict(self, X):
        return softmax(self.logits(X) / self.temperature)

    def choice(self, X, rng=None):
        if rng is None:
            return int(np.argmax(self.logits(X)))
        return int(rng.choice(len(X), p=self.predict(X)))

    def teach(self, X, target):
        predicted = self.predict(X)
        H = np.tanh(X @ self.p["W1"] + self.p["b1"])
        d = (predicted - target) / self.temperature
        dH = d[:, None] * self.p["W2"][None, :] * (1 - H * H)
        g = dict(W1=X.T @ dH, b1=dH.sum(axis=0),
                 W2=H.T @ d, b2=np.array([d.sum()]))
        norm = math.sqrt(sum(float(np.square(v).sum()) for v in g.values()))
        if norm > 8:
            g = {k: v * (8 / norm) for k, v in g.items()}
        self.updates += 1
        for k, grad in g.items():
            self.m[k] = .9 * self.m[k] + .1 * grad
            self.v[k] = .999 * self.v[k] + .001 * grad * grad
            m = self.m[k] / (1 - .9 ** self.updates)
            v = self.v[k] / (1 - .999 ** self.updates)
            self.p[k] -= self.lr * m / (np.sqrt(v) + 1e-8)
        return -float(np.sum(target * np.log(np.maximum(predicted, 1e-12))))

    def mutate(self, seed):
        rng = np.random.default_rng(seed)
        c = Policy(seed, hidden=len(self.p["W2"]),
                   lr=float(np.clip(self.lr * rng.choice([.6, .85, 1.15, 1.4]), .0005, .025)),
                   temperature=float(np.clip(self.temperature * rng.choice([.8, 1., 1.2]), .35, 1.8)))
        for k, v in self.p.items():
            c.p[k] = v + rng.normal(0, .025, v.shape)
        return c

    def serialized(self):
        return dict(format="population-mlp-v1", features=list(FEATURES),
                    hidden=len(self.p["W2"]), lr=self.lr, temperature=self.temperature,
                    optimizer_steps=self.updates,
                    parameters={k: v.tolist() for k, v in self.p.items()})


def episode(policy, niche, seed, max_steps=30):
    world = World.generate(seed, max_steps)
    rng = np.random.default_rng(stable_seed(seed, niche, "eval"))
    reward = 0.
    while not world.stopped and world.steps < world.max_steps:
        actions = world.legal()
        X = np.stack([world.feature(a) for a in actions])
        if policy is None:
            choice = int(rng.integers(len(actions)))
        elif policy == "nonstop_random":
            eligible = [i for i, a in enumerate(actions) if a != "STOP"]
            choice = int(rng.choice(eligible)) if eligible else actions.index("STOP")
        elif policy == "persistent":
            choice = next((i for i, a in enumerate(actions) if a == world.heading), None)
            if choice is None:
                choice = next((i for i, a in enumerate(actions) if a in DIRECTIONS), None)
            if choice is None:
                choice = actions.index("RETRACE") if "RETRACE" in actions else actions.index("STOP")
        elif policy == "synthetic_oracle":
            choice = int(np.argmax([utility(niche, x, world) for x in X]))
        else:
            choice = policy.choice(X)
        reward += utility(niche, X[choice], world)
        world.step(actions[choice])
        if actions[choice] == "STOP":
            break
    return reward, world


def train(policy, niche, seeds, target_temperature=.42):
    losses = []
    for seed in seeds:
        rng = np.random.default_rng(stable_seed(seed, niche, "train"))
        world = World.generate(seed)
        while not world.stopped and world.steps < world.max_steps:
            actions = world.legal()
            X = np.stack([world.feature(a) for a in actions])
            target = softmax(np.array([utility(niche, x, world) for x in X]) / target_temperature)
            losses.append(policy.teach(X, target))
            # Real supervised gradient updates; policy-selected sequential states.
            # 25% teacher sampling diversifies and prevents early STOP collapse.
            choice = int(rng.choice(len(actions), p=target)) if rng.random() < .25 else policy.choice(X, rng)
            world.step(actions[choice])
            if actions[choice] == "STOP":
                break
    return dict(optimizer_steps=policy.updates,
                mean_cross_entropy=round(float(np.mean(losses)), 5))


def evaluate(policy, niche, seeds):
    scores, descriptors, painted, retraces, early = [], [], [], [], []
    for s in seeds:
        reward, w = episode(policy, niche, s)
        scores.append(reward)
        descriptors.append(w.descriptor())
        painted.append(len(w.painted) - 1)
        retraces.append(sum(x["action"] == "RETRACE" for x in w.history))
        early.append(w.stopped and w.steps < w.max_steps)
    return dict(mean_reward=round(float(np.mean(scores)), 5),
                mean_painted=round(float(np.mean(painted)), 3),
                mean_retrace=round(float(np.mean(retraces)), 3),
                early_stop_rate=round(float(np.mean(early)), 3),
                mean_descriptor=np.round(np.mean(descriptors, axis=0), 4).tolist(),
                per_seed_reward=[round(float(v), 4) for v in scores])


def draw_svg(w):
    size = w.width * 17
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">',
             '<rect width="100%" height="100%" fill="#141b24"/>']
    for y in range(w.height):
        for x in range(w.width):
            p = (x, y)
            fill = "#344451" if p in w.obstacles else "#76363b" if p in w.protected else "#263440"
            lines.append(f'<rect x="{x*17}" y="{y*17}" width="16" height="16" fill="{fill}"/>')
    for x, y in sorted(w.painted):
        lines.append(f'<rect x="{x*17+2}" y="{y*17+2}" width="12" height="12" fill="#64cdb2"/>')
    for h in w.history:
        if h["action"] == "RETRACE":
            x, y = h["end"]
            lines.append(f'<circle cx="{x*17+8.5}" cy="{y*17+8.5}" r="4" fill="#ffc271"/>')
    lines.append('<circle cx="59.5" cy="195.5" r="6" fill="#f4e9a3"/>')
    lines.append(f'<circle cx="{w.cursor[0]*17+8.5}" cy="{w.cursor[1]*17+8.5}" r="5" fill="#e8b45a"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def experiment(output, seed=619, population=3, train_episodes=36,
               selection_episodes=8, test_episodes=12):
    if not (1 <= population <= 12 and 1 <= train_episodes <= 1000 and
            1 <= selection_episodes <= 100 and 1 <= test_episodes <= 100):
        raise ValueError("bounded campaign parameters required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "gallery").mkdir()
    train_seeds = [stable_seed(seed, "train", i) for i in range(train_episodes)]
    select_seeds = [stable_seed(seed, "selection", i) for i in range(selection_episodes)]
    test_seeds = [stable_seed(seed, "test", i) for i in range(test_episodes)]
    if len(set(train_seeds + select_seeds + test_seeds)) != sum(map(len, (train_seeds, select_seeds, test_seeds))):
        raise AssertionError("train/selection/holdout seed collision")
    records, champions, winners = [], {}, {}
    for niche in NICHES:
        ranked = []
        for i in range(population):
            name = f"{niche}-g0-{i}"
            p = Policy(stable_seed(seed, name), lr=.004 * (1 + i / 2), temperature=.85 + i * .13)
            untrained = evaluate(p, niche, select_seeds)
            training = train(p, niche, train_seeds)
            result = evaluate(p, niche, select_seeds)
            records.append(dict(id=name, niche=niche, generation=0, parent=None,
                                training=training, initial_selection_reward=untrained["mean_reward"],
                                selection=result))
            ranked.append((result["mean_reward"], name, p))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_reward, best_name, best = ranked[0]
        for j in range(max(1, population // 2)):
            name = f"{niche}-g1-{j}"
            child = best.mutate(stable_seed(seed, name))
            training = train(child, niche,
                             [stable_seed(seed, niche, "mutate", j, i)
                              for i in range(max(4, train_episodes // 2))])
            result = evaluate(child, niche, select_seeds)
            records.append(dict(id=name, niche=niche, generation=1,
                                parent=best_name, training=training, selection=result))
            if result["mean_reward"] > best_reward:
                best_reward, best_name, best = result["mean_reward"], name, child
        champions[niche], winners[niche] = best_name, best
        (output / "models" / f"{niche}.json").write_text(json.dumps(best.serialized()) + "\n")
        _, w = episode(best, niche, test_seeds[0])
        (output / "gallery" / f"{niche}.svg").write_text(draw_svg(w))
        (output / "gallery" / f"{niche}-trajectory.json").write_text(
            json.dumps(dict(synthetic=True, native=False, seed=test_seeds[0],
                            policy_id=best_name, niche=niche, descriptor=w.descriptor(),
                            trace=w.history, protected=sorted(w.protected),
                            obstacles=sorted(w.obstacles), painted=sorted(w.painted)),
                       indent=2) + "\n")
    controls = {n: {k: evaluate(p, n, test_seeds) for k, p in
                    (("uniform_legal", None), ("nonstop_random", "nonstop_random"),
                     ("persistent", "persistent"), ("synthetic_oracle", "synthetic_oracle"))}
                for n in NICHES}
    final = {n: evaluate(winners[n], n, test_seeds) for n in NICHES}
    descriptors = np.asarray([final[n]["mean_descriptor"] for n in NICHES])
    diversity = np.mean([np.linalg.norm(a-b) for i, a in enumerate(descriptors)
                         for b in descriptors[i+1:]])
    manifest = dict(format="population-research-v1", kind="SYNTHETIC_NEURAL_POLICY_TRAINING",
                    native=False, language_model=False,
                    training_method="on-policy teacher-augmented differentiable MLP supervised Adam",
                    seed=seed, niches=list(NICHES), candidate_count=len(records),
                    train_seed_count=len(train_seeds),
                    selection_seed_count=len(select_seeds),
                    heldout_seed_count=len(test_seeds),
                    seed_hashes={k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                                 for k, v in (("train", train_seeds), ("selection", select_seeds),
                                              ("heldout", test_seeds))},
                    champion_ids=champions, selection_lineage=records,
                    heldout_controls=controls, heldout_champions=final,
                    mean_pairwise_descriptor_diversity=round(float(diversity), 5),
                    limitations=["synthetic 22x22 grid, NOT Godot/native PAINT",
                                 "small trained neural scoring MLPs, NOT pretrained SLMs",
                                 "host-legal action menus guarantee legal actions, not learned legality",
                                 "hand-designed synthetic reward, NOT human judgements",
                                 "PBT selection seeds distinct from final untouched test seeds"])
    (output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=619)
    parser.add_argument("--population", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=36)
    parser.add_argument("--selection-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=12)
    a = parser.parse_args()
    r = experiment(a.output, a.seed, a.population, a.train_episodes,
                   a.selection_episodes, a.test_episodes)
    for niche in NICHES:
        baseline = r["heldout_controls"][niche]["nonstop_random"]["mean_reward"]
        trained = r["heldout_champions"][niche]["mean_reward"]
        print(f"{niche:19} control={baseline:8.3f} trained={trained:8.3f} "
              f"champion={r['champion_ids'][niche]}")
    print("descriptor_diversity:", r["mean_pairwise_descriptor_diversity"])


if __name__ == "__main__":
    main()
