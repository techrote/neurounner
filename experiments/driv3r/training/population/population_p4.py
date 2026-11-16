"""P4: online trajectory-level novelty learning with sham-feedback controls.

Synthetic host-legal MLP training, NOT pretrained SLM or native PAINT.
Feedback sees completed TRAINING rollouts only. Selection and held-out worlds
are unseen until all model weights and champions are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .population_lab import DIRECTIONS, NICHES, World, draw_svg, episode, evaluate, softmax, stable_seed, utility
    from .population_p1 import load_policy, model_sha
    from .population_p2 import heldout_dispersion, morphology_bonus, STYLES
    from .population_p3 import long_horizon
except ImportError:
    from population_lab import DIRECTIONS, NICHES, World, draw_svg, episode, evaluate, softmax, stable_seed, utility
    from population_p1 import load_policy, model_sha
    from population_p2 import heldout_dispersion, morphology_bonus, STYLES
    from population_p3 import long_horizon

ARMS = ("reward_only", "sham_feedback", "trajectory_novelty", "morphology_only")


def splits_for(seed, train, selection, holdout):
    splits = {kind: [stable_seed("p4", seed, kind, i) for i in range(count)]
              for kind, count in (("train", train), ("selection", selection), ("holdout", holdout))}
    seq = sum(splits.values(), [])
    if len(seq) != len(set(seq)):
        raise AssertionError("P4 seed overlap")
    return splits


def phenotype(world):
    """Measured rollout behaviour; no style-target or reward labels."""
    actions = [item["action"] for item in world.history]
    moves = [a for a in actions if a in DIRECTIONS]
    painted = world.painted
    quadrant = [sum(int((x >= 11) == east and (y >= 11) == south) for x, y in painted)
                / max(1, len(painted)) for east, south in ((False, False), (True, False),
                                                           (False, True), (True, True))]
    direction = [moves.count(d) / max(1, len(moves)) for d in DIRECTIONS]
    return np.asarray(world.descriptor() + direction + quadrant +
                      [float(world.stopped and world.steps < world.max_steps)], dtype=float)


def trajectory_novelty(world, archive, k=4):
    """Nearest archived-phenotype distance, with training-only memory."""
    if not archive:
        return .5
    vec = phenotype(world)
    distances = sorted(float(np.linalg.norm(vec - x)) for x in archive)
    mean = float(np.mean(distances[:min(k, len(distances))]))
    return mean / (1 + mean)


def feedback_target(base_target, chosen, advantage, intensity):
    """Post-episode supervised surrogate, NOT claimed as REINFORCE."""
    if not 0 <= intensity <= 1 or len(base_target) < 1:
        raise ValueError("invalid P4 feedback strength/menu")
    target = np.asarray(base_target, dtype=float)
    if len(target) < 2 or abs(advantage) < 1e-12:
        return target.copy()
    signal = np.zeros(len(target))
    if advantage > 0:
        signal[chosen] = 1.
    else:
        signal[:] = 1 / (len(target) - 1)
        signal[chosen] = 0.
    amount = min(.65, intensity * min(1., abs(advantage) * 3))
    return (1 - amount) * target + amount * signal


def train_member(policy, niche, train_seeds, arm, style, archive, strength, rng_seed):
    """Real sequential Adam updates, identical initial weights and train seeds."""
    if arm not in ARMS or style not in STYLES or not 0 < strength <= 1:
        raise ValueError("invalid P4 treatment")
    rng = np.random.default_rng(rng_seed)
    updates_before = policy.updates
    losses, novelties, reversals = [], [], []
    for seed in train_seeds:
        world = World.generate(seed)
        decisions = []
        while not world.stopped and world.steps < world.max_steps:
            menu = world.legal()
            X = np.stack([world.feature(action) for action in menu])
            base = np.asarray([utility(niche, x, world) for x in X], dtype=float)
            if arm == "morphology_only":
                base += strength * np.asarray([morphology_bonus(world, action, x, style)
                                               for action, x in zip(menu, X)])
            target = softmax(base / .42)
            chosen = (int(rng.choice(len(menu), p=target))
                      if rng.random() < .25 else policy.choice(X, rng))
            decisions.append((X, target, chosen))
            world.step(menu[chosen])
        novelty = trajectory_novelty(world, archive)
        baseline = .5 if len(novelties) < 3 else float(np.mean(novelties[-8:]))
        feedback = (novelty if arm == "trajectory_novelty" else
                    float(np.random.default_rng(stable_seed("p4-sham", rng_seed, seed)).random())
                    if arm == "sham_feedback" else baseline)
        advantage = feedback - baseline
        reversals.append(int(advantage < 0))
        for X, target, chosen in decisions:
            if arm in ("trajectory_novelty", "sham_feedback"):
                target = feedback_target(target, chosen, advantage, strength)
            losses.append(policy.teach(X, target))
        archive.append(phenotype(world))
        novelties.append(novelty)
    return dict(adam_steps=policy.updates - updates_before,
                mean_loss=round(float(np.mean(losses)), 6),
                mean_training_novelty=round(float(np.mean(novelties)), 6),
                negative_feedback_episodes=sum(reversals),
                episodes=len(train_seeds))


def run_p4(p0_output, output, seed=629, members=3, train_episodes=18,
           selection_episodes=8, holdout_episodes=12, strength=.8, horizon=60):
    if not (2 <= members <= 6 and 4 <= train_episodes <= 100
            and 2 <= selection_episodes <= 30 and 2 <= holdout_episodes <= 30
            and 0 < strength <= 1 and 31 <= horizon <= 128):
        raise ValueError("bounded P4 training/holdout configuration required")
    p0_output, output = Path(p0_output), Path(output)
    source = json.loads((p0_output / "results.json").read_text())
    if source.get("format") != "population-research-v1" or source.get("language_model") is not False:
        raise ValueError("P4 requires original synthetic P0 provenance")
    splits = splits_for(seed, train_episodes, selection_episodes, holdout_episodes)
    old = {stable_seed(source["seed"], part, i)
           for part, count in (("train", source["train_seed_count"]),
                               ("selection", source["selection_seed_count"]),
                               ("test", source["heldout_seed_count"]))
           for i in range(count)}
    if any(s in old for seeds in splits.values() for s in seeds):
        raise AssertionError("P4 overlaps P0 source seeds")
    parents = {n: load_policy(p0_output / "models" / (n + ".json")) for n in NICHES}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "gallery").mkdir()
    records, comparison, policies = [], {}, {}
    for niche in NICHES:
        comparison[niche] = {}
        for arm in ARMS:
            archive = []  # Fresh arm-scoped training archive; no selection/test input.
            group = []
            for m in range(members):
                style = STYLES[m % len(STYLES)]
                child = parents[niche].mutate(stable_seed("p4", seed, niche, m))
                before = model_sha(child)
                training = train_member(child, niche, splits["train"], arm, style,
                                        archive, strength,
                                        stable_seed("p4", seed, niche, m, "action_sampling"))
                ident = f"{niche}-{arm}-{m}"
                choice = evaluate(child, niche, splits["selection"])
                if training["adam_steps"] <= 0 or model_sha(child) == before:
                    raise AssertionError("P4 lacks genuine optimizer updates")
                records.append(dict(id=ident, niche=niche, arm=arm, member=m,
                                    source="P0 champion", initial_sha256=before,
                                    model_sha256=model_sha(child),
                                    training=training, selection=choice))
                (output / "models" / f"{ident}.json").write_text(
                    json.dumps(child.serialized(), sort_keys=True) + "\n")
                policies[ident] = child
                group.append(ident)
            champion = max(group, key=lambda name: (next(c["selection"]["mean_reward"]
                                  for c in records if c["id"] == name), name))
            comparison[niche][arm] = dict(members=group, champion_id=champion,
                                          training_archive_size=len(archive))
    # All model weights and selection decisions are frozen BEFORE held-out evaluation.
    for niche in NICHES:
        comparison[niche]["controls"] = dict(
            p0=evaluate(parents[niche], niche, splits["holdout"]),
            nonstop_random=evaluate("nonstop_random", niche, splits["holdout"]))
        for arm in ARMS:
            row = comparison[niche][arm]
            candidates = [policies[c] for c in row["members"]]
            row["heldout_dispersion"] = heldout_dispersion(candidates, niche, splits["holdout"])
            row["heldout_long"] = long_horizon(candidates, niche, splits["holdout"], horizon)
            row["heldout_champion"] = evaluate(policies[row["champion_id"]], niche,
                                               splits["holdout"])
            _, world = episode(policies[row["champion_id"]], niche, splits["holdout"][0],
                               max_steps=horizon)
            if world.painted & (world.protected | world.obstacles):
                raise AssertionError("P4 host legality violated")
            stem = f"{niche}-{arm}"
            (output / "gallery" / f"{stem}.svg").write_text(draw_svg(world))
            (output / "gallery" / f"{stem}-trace.json").write_text(json.dumps(
                dict(synthetic=True, native=False, policy_id=row["champion_id"],
                     seed=splits["holdout"][0], painted=sorted(world.painted),
                     protected=sorted(world.protected), obstacles=sorted(world.obstacles),
                     trajectory=world.history, descriptor=world.descriptor()), indent=2) + "\n")
    summary = {arm: dict(
        short_dispersion=round(float(np.mean([comparison[n][arm]["heldout_dispersion"]
                                              ["mean_pairwise_trajectory_distance"]
                                              for n in NICHES])), 6),
        long_dispersion=round(float(np.mean([comparison[n][arm]["heldout_long"]
                                             ["mean_distance"] for n in NICHES])), 6),
        mean_reward=round(float(np.mean([comparison[n][arm]["heldout_champion"]
                                         ["mean_reward"] for n in NICHES])), 6))
        for arm in ARMS}
    result = dict(format="population-research-p4", kind="SYNTHETIC_POST_EPISODE_TRAJECTORY_FEEDBACK",
                  seed=seed, native=False, language_model=False, source_p0_seed=source["seed"],
                  arms=list(ARMS), candidate_count=len(records), candidates=records,
                  adam_updates=sum(c["training"]["adam_steps"] for c in records),
                  train_episodes=train_episodes, selection_episodes=selection_episodes,
                  holdout_episodes=holdout_episodes, members=members,
                  intervention_strength=strength, horizon=horizon,
                  split_hashes={k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                                for k, v in splits.items()},
                  comparison=comparison, summary=summary,
                  limitations=["Synthetic game/world/reward, NOT native CyberSand PAINT",
                               "Real Adam-trained MLPs, NOT pretrained SLM adapters",
                               "Post-episode feedback is a supervised surrogate, not REINFORCE",
                               "Selection based only on held-out-from-training selection reward",
                               "Training archive shared sequentially between members in each arm",
                               "Trajectory-phenotype distance remains hand-engineered measurement",
                               "No human preference ratings or game-integrated receipts"])
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=629)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--train-episodes", type=int, default=18)
    ap.add_argument("--selection-episodes", type=int, default=8)
    ap.add_argument("--holdout-episodes", type=int, default=12)
    ap.add_argument("--strength", type=float, default=.8)
    ap.add_argument("--horizon", type=int, default=60)
    args = ap.parse_args()
    result = run_p4(args.p0_output, args.output, args.seed, args.members,
                    args.train_episodes, args.selection_episodes,
                    args.holdout_episodes, args.strength, args.horizon)
    print(json.dumps(dict(seed=result["seed"], candidates=result["candidate_count"],
                          adam_updates=result["adam_updates"], summary=result["summary"]), indent=2))


if __name__ == "__main__":
    main()
