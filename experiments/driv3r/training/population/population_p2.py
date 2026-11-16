"""P2: paired during-optimizer diversity versus reward-only policy training.

All worlds are synthetic and host-masked. Real Adam gradient updates.
No native/game changes, SLM claim, human ratings or paid inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .population_lab import (
        DIRECTIONS, FEATURES, NICHES, Policy, World, draw_svg, episode,
        evaluate, softmax, stable_seed, utility,
    )
    from .population_p1 import load_policy, model_sha
except ImportError:
    from population_lab import (
        DIRECTIONS, FEATURES, NICHES, Policy, World, draw_svg, episode,
        evaluate, softmax, stable_seed, utility,
    )
    from population_p1 import load_policy, model_sha


STYLES = ("zigzag", "persistent", "branching")
ARMS = ("reward_only", "during_training_novelty")


def split_seeds(seed, n_train, n_selection, n_holdout):
    values = {
        name: [stable_seed("p2", seed, name, i) for i in range(n)]
        for name, n in (("train", n_train), ("selection", n_selection),
                        ("holdout", n_holdout))
    }
    flat = sum(values.values(), [])
    if len(set(flat)) != len(flat):
        raise AssertionError("P2 split collision")
    return values


def novelty_key(world: World, action: str):
    """Coarse state-action key; training-only counters, never held-out data."""
    dx, dy = DIRECTIONS.get(action, (0, 0))
    after = (world.cursor[0] + dx, world.cursor[1] + dy)
    return (action, world.heading, int(world.switched),
            int(after[0] >= world.width // 2),
            int(after[1] >= world.height // 2),
            min(2, len(world.painted) // 10))


def morphology_bonus(world: World, action: str, features, style: str):
    """Small, explicitly artificial phenotype shaping at optimizer time."""
    f = dict(zip(FEATURES, features))
    if action == "STOP":
        return -0.55 if len(world.legal()) > 1 else 0.
    if style == "zigzag":
        return .8 * f["turn"] + .35 * f["motif"] + .2 * f["paint"]
    if style == "persistent":
        return .65 * (1 - f["turn"]) * f["paint"] + .35 * f["onward"]
    if style == "branching":
        return (.85 * f["retrace"] * (len(world.painted) > 5)
                + .55 * f["junction"] * f["paint"]
                + .3 * f["frontier"])
    raise ValueError(style)


def train_arm(policy, niche, seeds, style, arm, *,
              novelty_archive=None, strength=.8):
    """Paired optimizer experiment, no selection/holdout observations.

    Novelty arm adjusts soft training targets with state-action rarity across
    previous population members and a per-lineage morphology preference.
    Both arms use the same seed stream, starting weights and nominal LR.
    The arm divergence is an actual gradient-target intervention.
    """
    if arm not in ARMS or style not in STYLES or not 0 <= strength <= 2:
        raise ValueError("unrecognised or unbounded P2 arm")
    archive = novelty_archive if novelty_archive is not None else Counter()
    local = Counter()
    losses = []
    for seed in seeds:
        world = World.generate(seed)
        rng = np.random.default_rng(stable_seed("p2", seed, niche, style, "sampling"))
        while not world.stopped and world.steps < world.max_steps:
            menu = world.legal()
            X = np.stack([world.feature(a) for a in menu])
            scores = np.array([utility(niche, f, world) for f in X])
            if arm == "during_training_novelty":
                bonuses = []
                for action, features in zip(menu, X):
                    key = novelty_key(world, action)
                    # Counts from earlier individual policies AND this rollout.
                    rarity = 1. / np.sqrt(1 + archive[key] + local[key])
                    bonuses.append(.6 * rarity
                                   + morphology_bonus(world, action, features, style))
                scores += strength * np.asarray(bonuses)
            target = softmax(scores / .42)
            losses.append(policy.teach(X, target))
            choice = (int(rng.choice(len(menu), p=target))
                      if rng.random() < .25 else policy.choice(X, rng))
            action = menu[choice]
            # Read-only training-population archive; no test-world information.
            local[novelty_key(world, action)] += 1
            world.step(action)
            if action == "STOP":
                break
    if arm == "during_training_novelty":
        archive.update(local)
    return dict(adam_steps=policy.updates,
                mean_cross_entropy=round(float(np.mean(losses)), 6),
                training_action_bins=len(local),
                training_action_events=sum(local.values()))


def trajectory_distance(left: World, right: World):
    """Same-world paint Jaccard and padded action disagreement, [0,1]."""
    a = left.painted - {(3, 11)}
    b = right.painted - {(3, 11)}
    occupancy = 1 - (len(a & b) / len(a | b) if a | b else 1)
    aa = [h["action"] for h in left.history]
    bb = [h["action"] for h in right.history]
    n = max(len(aa), len(bb), 1)
    disagreement = sum((aa[i] if i < len(aa) else "<END>")
                       != (bb[i] if i < len(bb) else "<END>")
                       for i in range(n)) / n
    return round(float(.75 * occupancy + .25 * disagreement), 6)


def heldout_dispersion(policies, niche, seeds):
    """Inspect ALL trained offspring AFTER selection, not for choosing them."""
    distances = []
    signatures = set()
    per_seed = []
    for seed in seeds:
        worlds = [episode(policy, niche, seed)[1] for policy in policies]
        pairwise = [trajectory_distance(a, b)
                    for i, a in enumerate(worlds) for b in worlds[i + 1:]]
        distances.extend(pairwise)
        per_seed.append(round(float(np.mean(pairwise)), 6))
        for world in worlds:
            direction = tuple(h["action"] for h in world.history[:8])
            signatures.add((tuple(round(v, 1) for v in world.descriptor()),
                            direction))
    return dict(mean_pairwise_trajectory_distance=round(float(np.mean(distances)), 6),
                per_seed_pairwise_distance=per_seed,
                phenotype_signatures=len(signatures))


def run_p2(p0_output: Path, output: Path, seed=625, members=3,
           train_episodes=18, selection_episodes=8, holdout_episodes=12,
           strength=.8):
    if not (2 <= members <= 6 and 4 <= train_episodes <= 100
            and 2 <= selection_episodes <= 30 and 2 <= holdout_episodes <= 30
            and 0 < strength <= 2):
        raise ValueError("bounded training configuration required")
    p0_output, output = Path(p0_output), Path(output)
    source = json.loads((p0_output / "results.json").read_text())
    if source.get("format") != "population-research-v1" or source.get("language_model") is not False:
        raise ValueError("incompatible P0 input")
    splits = split_seeds(seed, train_episodes, selection_episodes, holdout_episodes)
    # Also reject explicit P0 split reuse, not just the P2 internal collision.
    old = set()
    for mode in ("train", "selection", "test"):
        old.update(stable_seed(source["seed"], mode, i)
                   for i in range({"train": source["train_seed_count"],
                                   "selection": source["selection_seed_count"],
                                   "test": source["heldout_seed_count"]}[mode]))
    if any(s in old for values in splits.values() for s in values):
        raise AssertionError("P0/P2 individual seed overlap")
    bases = {n: load_policy(p0_output / "models" / (n + ".json")) for n in NICHES}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "gallery").mkdir()
    candidates, comparison, all_policies = [], {}, {}
    for niche in NICHES:
        archive = Counter()
        comparison[niche] = {}
        for arm in ARMS:
            trained = []
            for member in range(members):
                style = STYLES[member % len(STYLES)]
                ident = f"{niche}-{arm}-{member}"
                # Matched initial weights/mutation and hyperparameters by member.
                parent = bases[niche]
                child = parent.mutate(stable_seed("p2", seed, niche, member))
                training = train_arm(child, niche, splits["train"], style, arm,
                                     novelty_archive=archive, strength=strength)
                selection = evaluate(child, niche, splits["selection"])
                candidates.append(dict(id=ident, niche=niche, arm=arm,
                                       parent="P0 champion", member=member,
                                       style=style, training=training,
                                       selection=selection,
                                       model_sha256=model_sha(child)))
                (output / "models" / (ident + ".json")).write_text(
                    json.dumps(child.serialized(), sort_keys=True) + "\n")
                trained.append((ident, child, selection["mean_reward"]))
                all_policies[ident] = child
            # Identical reward-only champion-selection rule in both arms.
            winner = max(trained, key=lambda item: (item[2], item[0]))
            comparison[niche][arm] = dict(
                champion_id=winner[0],
                selection_reward=winner[2],
                train_member_ids=[t[0] for t in trained])
        comparison[niche]["controls"] = {
            "p0": evaluate(bases[niche], niche, splits["holdout"]),
            "nonstop_random": evaluate("nonstop_random", niche, splits["holdout"]),
        }

    # Selection is COMPLETE before holdout trajectories are inspected.
    for niche in NICHES:
        for arm in ARMS:
            ids = comparison[niche][arm]["train_member_ids"]
            policies = [all_policies[i] for i in ids]
            comparison[niche][arm]["heldout_dispersion"] = heldout_dispersion(
                policies, niche, splits["holdout"])
            chosen_id = comparison[niche][arm]["champion_id"]
            chosen = all_policies[chosen_id]
            comparison[niche][arm]["heldout_champion"] = evaluate(
                chosen, niche, splits["holdout"])
            _, world = episode(chosen, niche, splits["holdout"][0])
            stem = f"{niche}-{arm}"
            (output / "gallery" / (stem + ".svg")).write_text(draw_svg(world))
            (output / "gallery" / (stem + "-trace.json")).write_text(
                json.dumps(dict(synthetic=True, native=False,
                                seed=splits["holdout"][0], policy_id=chosen_id,
                                protected=sorted(world.protected),
                                obstacles=sorted(world.obstacles),
                                painted=sorted(world.painted),
                                trajectory=world.history,
                                descriptor=world.descriptor()), indent=2) + "\n")
    diffs = [comparison[n]["during_training_novelty"]["heldout_dispersion"][
                 "mean_pairwise_trajectory_distance"]
             - comparison[n]["reward_only"]["heldout_dispersion"][
                 "mean_pairwise_trajectory_distance"] for n in NICHES]
    manifest = dict(
        format="population-research-p2",
        kind="SYNTHETIC_ADAM_DURING_TRAINING_NOVELTY_ABLATION",
        native=False, language_model=False, seed=seed,
        source_p0_seed=source["seed"],
        candidates=candidates, candidate_count=len(candidates),
        optimizer_trained_count=len(candidates),
        arms=list(ARMS), members_per_niche_per_arm=members,
        train_episodes=train_episodes, novelty_strength=strength,
        split_hashes={k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                      for k, v in splits.items()},
        comparisons=comparison,
        dispersion_differences_per_niche=dict(zip(NICHES, diffs)),
        mean_dispersion_difference=round(float(np.mean(diffs)), 6),
        limitations=[
            "Synthetic hand-designed rewards/worlds, not native CyberSand gameplay",
            "Trained tiny MLP models, NOT pretrained SLMs",
            "Novelty combines action-key rarity with hand-designed morphology rewards",
            "Host-masked legal actions, not learned safety",
            "Heldout dispersion measured after selection for all offspring",
            "No human preferences or artistic quality ground truth",
            "Three initially matched members per arm; sequential novelty archive within one arm",
        ],
    )
    (output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=625)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--train-episodes", type=int, default=18)
    ap.add_argument("--selection-episodes", type=int, default=8)
    ap.add_argument("--holdout-episodes", type=int, default=12)
    ap.add_argument("--novelty-strength", type=float, default=.8)
    a = ap.parse_args()
    result = run_p2(a.p0_output, a.output, a.seed, a.members,
                    a.train_episodes, a.selection_episodes, a.holdout_episodes,
                    a.novelty_strength)
    print("P2 seed", result["seed"], "trained", result["optimizer_trained_count"],
          "mean heldout trajectory dispersion delta", result["mean_dispersion_difference"])
    for niche in NICHES:
        x = result["comparisons"][niche]
        print(niche, "reward-only",
              x["reward_only"]["heldout_dispersion"]["mean_pairwise_trajectory_distance"],
              "novelty-training",
              x["during_training_novelty"]["heldout_dispersion"]["mean_pairwise_trajectory_distance"],
              "reward-only champion", x["reward_only"]["heldout_champion"]["mean_reward"],
              "novelty champion", x["during_training_novelty"]["heldout_champion"]["mean_reward"])


if __name__ == "__main__":
    main()
