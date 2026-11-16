"""P3: factorial ablation of rarity and morphology during real game-policy training.

The four arms share source champions, mutated initial weights, worlds, RNG
streams, optimizer hyperparameters and selection rule. Only the teacher-target
bonuses differ. This is an external synthetic MLP experiment, NOT native
CyberSand painting, human aesthetic preference, or pretrained SLM training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .population_lab import NICHES, Policy, World, draw_svg, episode, evaluate, softmax, stable_seed, utility
    from .population_p1 import load_policy, model_sha
    from .population_p2 import STYLES, heldout_dispersion, morphology_bonus, novelty_key, split_seeds as p2_splits, trajectory_distance
except ImportError:
    from population_lab import NICHES, Policy, World, draw_svg, episode, evaluate, softmax, stable_seed, utility
    from population_p1 import load_policy, model_sha
    from population_p2 import STYLES, heldout_dispersion, morphology_bonus, novelty_key, split_seeds as p2_splits, trajectory_distance


ARMS = ("reward_only", "rarity_only", "morphology_only", "combined")


def split_seeds(seed, n_train, n_selection, n_holdout):
    splits = {name: [stable_seed("p3", seed, name, i) for i in range(count)]
              for name, count in (("train", n_train), ("selection", n_selection),
                                  ("holdout", n_holdout))}
    flat = sum(splits.values(), [])
    if len(flat) != len(set(flat)):
        raise AssertionError("P3 train/selection/holdout collision")
    return splits


def teacher_scores(world, niche, actions, X, arm, style, archive, local, strength):
    """Exactly isolate rarity and morphology factors before gradient updates."""
    if arm not in ARMS or style not in STYLES or not 0 <= strength <= 2:
        raise ValueError("invalid P3 arm/style/strength")
    base = np.asarray([utility(niche, x, world) for x in X], dtype=float)
    if arm in ("rarity_only", "combined"):
        rarity = np.asarray([.6 / np.sqrt(1 + archive[novelty_key(world, a)]
                                           + local[novelty_key(world, a)])
                             for a in actions])
        base += strength * rarity
    if arm in ("morphology_only", "combined"):
        base += strength * np.asarray([
            morphology_bonus(world, action, features, style)
            for action, features in zip(actions, X)])
    return base


def train_arm(policy, niche, seeds, style, arm, archive, strength):
    """True Adam updates, identical random streams per matched member/arm."""
    local = Counter()
    losses = []
    initial_updates = policy.updates
    for seed in seeds:
        world = World.generate(seed)
        rng = np.random.default_rng(stable_seed("p3", seed, niche, style, "sampling"))
        while not world.stopped and world.steps < world.max_steps:
            actions = world.legal()   # host-owned, finite legal actions
            X = np.stack([world.feature(a) for a in actions])
            scores = teacher_scores(world, niche, actions, X, arm, style,
                                    archive, local, strength)
            target = softmax(scores / .42)
            losses.append(policy.teach(X, target))
            chosen = (int(rng.choice(len(actions), p=target))
                      if rng.random() < .25 else policy.choice(X, rng))
            action = actions[chosen]
            local[novelty_key(world, action)] += 1
            world.step(action)
            if action == "STOP":
                break
    # Every arm has its OWN archived previous members; no leakage across arms.
    archive.update(local)
    return dict(adam_steps=policy.updates - initial_updates,
                mean_cross_entropy=round(float(np.mean(losses)), 6),
                visited_training_bins=len(local),
                training_actions=sum(local.values()))


def long_horizon(policies, niche, seeds, horizon):
    """Stress-test preselected members beyond the default 30-decision horizon."""
    distances, rewards, painted, retraces, stops = [], [], [], [], []
    signatures = set()
    for seed in seeds:
        worlds = []
        for policy in policies:
            reward, world = episode(policy, niche, seed, max_steps=horizon)
            worlds.append(world)
            rewards.append(reward)
            painted.append(len(world.painted) - 1)
            retraces.append(sum(t["action"] == "RETRACE" for t in world.history))
            stops.append(int(world.stopped and world.steps < horizon))
            signatures.add((tuple(np.round(world.descriptor(), 1)),
                            tuple(t["action"] for t in world.history[:8])))
            # Host-mask and game-boundary checks for every original trajectory.
            assert not (world.painted & (world.protected | world.obstacles))
            assert all(t["action"] in ("N", "E", "S", "W", "RETRACE", "STOP")
                       for t in world.history)
        distances += [trajectory_distance(a, b)
                      for i, a in enumerate(worlds) for b in worlds[i + 1:]]
    return dict(horizon=horizon,
                mean_distance=round(float(np.mean(distances)), 6),
                mean_reward=round(float(np.mean(rewards)), 5),
                mean_painted=round(float(np.mean(painted)), 3),
                mean_retraces=round(float(np.mean(retraces)), 3),
                early_stop_rate=round(float(np.mean(stops)), 3),
                phenotype_signatures=len(signatures))


def run_p3(p0_output: Path, output: Path, seed=627, members=3,
           train_episodes=18, selection_episodes=8, holdout_episodes=12,
           strength=.8, horizon=60):
    if not (2 <= members <= 6 and 4 <= train_episodes <= 100 and
            2 <= selection_episodes <= 30 and 2 <= holdout_episodes <= 30 and
            0 < strength <= 2 and 31 <= horizon <= 128):
        raise ValueError("P3 requires bounded genuine training, selection and holdout")
    p0_output, output = Path(p0_output), Path(output)
    source = json.loads((p0_output / "results.json").read_text())
    if source.get("format") != "population-research-v1" or source.get("language_model") is not False:
        raise ValueError("P3 refuses incompatible P0 source")
    splits = split_seeds(seed, train_episodes, selection_episodes, holdout_episodes)
    # Cross-campaign leakage guard: reject P0 and previously declared P2 cohorts.
    p0_seeds = [stable_seed(source["seed"], part, i)
                for part, count in (("train", source["train_seed_count"]),
                                    ("selection", source["selection_seed_count"]),
                                    ("test", source["heldout_seed_count"]))
                for i in range(count)]
    p2_seeds = [s for cohort in (625, 626)
                for group in p2_splits(cohort, 18, 8, 12).values() for s in group]
    old = set(p0_seeds + p2_seeds)
    if any(s in old for group in splits.values() for s in group):
        raise AssertionError("P3 overlaps earlier frozen campaign seeds")

    parents = {n: load_policy(p0_output / "models" / (n + ".json")) for n in NICHES}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "gallery").mkdir()
    candidates, comparisons, all_policies = [], {}, {}
    for niche in NICHES:
        comparisons[niche] = {}
        for arm in ARMS:
            archive = Counter()
            trained = []
            for member in range(members):
                style = STYLES[member % len(STYLES)]
                ident = f"{niche}-{arm}-{member}"
                # Each arm gets an EXACT matching initial model by member.
                child = parents[niche].mutate(stable_seed("p3", seed, niche, member))
                initial_sha = model_sha(child)
                training = train_arm(child, niche, splits["train"], style, arm,
                                     archive, strength)
                selection = evaluate(child, niche, splits["selection"])
                assert training["adam_steps"] > 0
                assert model_sha(child) != initial_sha
                candidates.append(dict(id=ident, niche=niche, arm=arm, member=member,
                                       style=style, source_parent="P0 champion",
                                       initial_model_sha256=initial_sha,
                                       trained_model_sha256=model_sha(child),
                                       training=training, selection=selection))
                (output / "models" / (ident + ".json")).write_text(
                    json.dumps(child.serialized(), sort_keys=True) + "\n")
                all_policies[ident] = child
                trained.append((ident, child, selection["mean_reward"]))
            # Selection ONLY from separate selection worlds, never held-out.
            winner = max(trained, key=lambda item: (item[2], item[0]))
            comparisons[niche][arm] = dict(champion_id=winner[0],
                                         selection_reward=winner[2],
                                         members=[entry[0] for entry in trained],
                                         training_archive_bins=len(archive))
    # All policy training and selection decisions are frozen before *any*
    # held-out evaluation (including the P0 and random controls).
    for niche in NICHES:
        comparisons[niche]["controls"] = dict(
            p0=evaluate(parents[niche], niche, splits["holdout"]),
            nonstop_random=evaluate("nonstop_random", niche, splits["holdout"]))
        for arm in ARMS:
            group = comparisons[niche][arm]
            policies = [all_policies[ident] for ident in group["members"]]
            champion = all_policies[group["champion_id"]]
            group["heldout_dispersion"] = heldout_dispersion(
                policies, niche, splits["holdout"])
            group["heldout_champion"] = evaluate(champion, niche, splits["holdout"])
            group["long_horizon"] = long_horizon(
                policies, niche, splits["holdout"], horizon)
            _, world = episode(champion, niche, splits["holdout"][0], max_steps=horizon)
            stem = f"{niche}-{arm}"
            (output / "gallery" / (stem + ".svg")).write_text(draw_svg(world))
            (output / "gallery" / (stem + "-trace.json")).write_text(
                json.dumps(dict(synthetic=True, native=False, seed=splits["holdout"][0],
                                horizon=horizon, policy_id=group["champion_id"],
                                protected=sorted(world.protected),
                                obstacles=sorted(world.obstacles),
                                painted=sorted(world.painted),
                                trajectory=world.history,
                                descriptor=world.descriptor()), indent=2) + "\n")

    summary = {}
    for arm in ARMS:
        distances = [comparisons[n][arm]["heldout_dispersion"][
                     "mean_pairwise_trajectory_distance"] for n in NICHES]
        long_distances = [comparisons[n][arm]["long_horizon"]["mean_distance"]
                          for n in NICHES]
        rewards = [comparisons[n][arm]["heldout_champion"]["mean_reward"]
                   for n in NICHES]
        summary[arm] = dict(mean_30_step_dispersion=round(float(np.mean(distances)), 6),
                            mean_long_horizon_dispersion=round(float(np.mean(long_distances)), 6),
                            mean_champion_reward=round(float(np.mean(rewards)), 5))
    result = dict(format="population-research-p3",
                  kind="SYNTHETIC_ADAM_FACTORIAL_RARITY_MORPHOLOGY_ABLATION",
                  seed=seed, native=False, language_model=False,
                  source_p0_seed=source["seed"], arms=list(ARMS),
                  candidates=candidates, candidate_count=len(candidates),
                  optimizer_trained_count=len(candidates),
                  total_new_adam_steps=sum(c["training"]["adam_steps"] for c in candidates),
                  members_per_niche_per_arm=members, train_episodes=train_episodes,
                  selection_episodes=selection_episodes,
                  holdout_episodes=holdout_episodes, long_horizon=horizon,
                  intervention_strength=strength,
                  split_hashes={k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                                for k, v in splits.items()},
                  comparison=comparisons, summary=summary,
                  limitations=[
                      "Synthetic handmade reward/worlds; no native game readbacks",
                      "Actual Adam-trained small MLPs, not pretrained SLMs",
                      "Rarity uses a coarse state-action key, morphology uses handcrafted styles",
                      "Different on-policy state distributions emerge after the intervention",
                      "Host-masked actions do not establish a model-internal safety mechanism",
                      "Within-arm archive is sequential across members; novelty treatment not iid",
                      "No human aesthetic rating; longer horizon is synthetic only",
                  ])
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=627)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--train-episodes", type=int, default=18)
    ap.add_argument("--selection-episodes", type=int, default=8)
    ap.add_argument("--holdout-episodes", type=int, default=12)
    ap.add_argument("--intervention-strength", type=float, default=.8)
    ap.add_argument("--long-horizon", type=int, default=60)
    a = ap.parse_args()
    result = run_p3(a.p0_output, a.output, seed=a.seed, members=a.members,
                    train_episodes=a.train_episodes,
                    selection_episodes=a.selection_episodes,
                    holdout_episodes=a.holdout_episodes,
                    strength=a.intervention_strength, horizon=a.long_horizon)
    print("P3 seed", a.seed, "newly trained", result["candidate_count"],
          "optimizer steps", result["total_new_adam_steps"])
    for arm in ARMS:
        print(arm, result["summary"][arm])


if __name__ == "__main__":
    main()
