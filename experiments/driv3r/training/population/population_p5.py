"""P5: matched-feedback causal ablation, selection MAP-Elites and recovery stress.

Actual Adam-trained tiny MLPs on identical FROZEN training rollouts per
matched member. This is OFF-POLICY supervised training, not online RL or SLM
finetuning. No CyberSand game code/content imports or modifications.
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
    from .population_p2 import STYLES, morphology_bonus, heldout_dispersion
    from .population_p3 import long_horizon
    from .population_p4 import phenotype, trajectory_novelty
except ImportError:
    from population_lab import DIRECTIONS, NICHES, World, draw_svg, episode, evaluate, softmax, stable_seed, utility
    from population_p1 import load_policy, model_sha
    from population_p2 import STYLES, morphology_bonus, heldout_dispersion
    from population_p3 import long_horizon
    from population_p4 import phenotype, trajectory_novelty

ARMS = ("reward_only", "matched_sham", "measured_novelty", "morphology_only")


def splits_for(seed, train, select, holdout):
    groups = {k: [stable_seed("p5", seed, k, i) for i in range(n)]
              for k, n in (("train", train), ("selection", select), ("holdout", holdout))}
    values = sum(groups.values(), [])
    if len(values) != len(set(values)):
        raise AssertionError("P5 world seed leakage")
    return groups


def episode_advantages(novelties):
    """Centered fractional ranks: equal novelty => equal signal, mean zero."""
    x = np.asarray(novelties, dtype=float)
    if not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("invalid novelty values")
    result = np.empty(len(x), dtype=float)
    for i, v in enumerate(x):
        result[i] = (np.sum(x < v) + .5 * np.sum(x == v)) / len(x) - .5
    return result


def matched_sham(advantages, seed):
    """An exact permutation of the same signed signals, no new amplitude RNG."""
    rng = np.random.default_rng(seed)
    out = np.asarray(advantages)[rng.permutation(len(advantages))]
    if len(out) > 1 and np.array_equal(out, advantages) and len(set(advantages)) > 1:
        out = np.roll(out, 1)
    return out


def feedback_target(base, chosen, signal, strength):
    if not 0 < strength <= 2 or len(base) < 1:
        raise ValueError("invalid feedback")
    target = np.asarray(base, dtype=float)
    if abs(signal) < 1e-12 or len(target) == 1:
        return target.copy()
    alt = np.zeros(len(base), dtype=float)
    if signal > 0:
        alt[chosen] = 1
    else:
        alt[:] = 1 / (len(base) - 1)
        alt[chosen] = 0
    weight = min(.65, strength * abs(signal))
    return (1 - weight) * target + weight * alt


def frozen_rollouts(source, niche, train_seeds, style, sample_seed):
    """Collect ONE training-only dataset; every intervention sees identical X."""
    rng = np.random.default_rng(sample_seed)
    archive, episodes, novelty = [], [], []
    for seed in train_seeds:
        w = World.generate(seed)
        observations = []
        while not w.stopped and w.steps < w.max_steps:
            menu = w.legal()
            X = np.stack([w.feature(action) for action in menu])
            reward = np.asarray([utility(niche, x, w) for x in X])
            target = softmax(reward / .42)
            morph = np.asarray([morphology_bonus(w, action, x, style)
                                for action, x in zip(menu, X)])
            chosen = (int(rng.choice(len(menu), p=target)) if rng.random() < .25
                      else source.choice(X, rng))
            observations.append((X, target, chosen, softmax((reward + .8 * morph) / .42)))
            w.step(menu[chosen])
        novelty.append(trajectory_novelty(w, archive))
        archive.append(phenotype(w))
        episodes.append(observations)
    return episodes, novelty


def train_arms(source, niche, seeds, style, sample_seed, strength):
    episodes, novelty = frozen_rollouts(source, niche, seeds, style, sample_seed)
    signal = episode_advantages(novelty)
    sham = matched_sham(signal, stable_seed("p5", sample_seed, "shuffle"))
    policies, training = {}, {}
    for arm in ARMS:
        # Copy source parameters *and optimizer state* to guarantee matched initialisation.
        from copy import deepcopy
        policy = deepcopy(source)
        before = model_sha(policy)
        losses = []
        for i, observations in enumerate(episodes):
            for X, base_target, chosen, morphology_target in observations:
                if arm == "morphology_only":
                    target = morphology_target
                elif arm in ("matched_sham", "measured_novelty"):
                    target = feedback_target(base_target, chosen,
                                             sham[i] if arm == "matched_sham" else signal[i],
                                             strength)
                else:
                    target = base_target
                losses.append(policy.teach(X, target))
        if policy.updates <= source.updates or model_sha(policy) == before:
            raise AssertionError("missing real Adam training")
        policies[arm] = policy
        training[arm] = dict(adam_steps=policy.updates - source.updates,
                             mean_loss=round(float(np.mean(losses)), 6),
                             source_sha256=before, model_sha256=model_sha(policy))
    return policies, training, dict(
        episodes=len(episodes), decisions=sum(map(len, episodes)),
        novelty=[round(float(v), 8) for v in novelty],
        actual_signal=[round(float(v), 8) for v in signal],
        sham_signal=[round(float(v), 8) for v in sham],
        exact_signal_multiset_match=sorted(signal.tolist()) == sorted(sham.tolist()),
        signal_alignment_changed=not np.array_equal(signal, sham))


def behaviour_bin(metrics):
    """Predeclared coarse topology cells derived from SELECTION-only worlds."""
    d = metrics["mean_descriptor"]
    return (min(2, int(d[0] * 3)), min(2, int(d[1] * 8)),
            min(2, int(d[2] * 3)), min(2, int(d[3] * 16)))


def elite_ids(ids, records, capacity=3):
    """Best reward in each selection-phenotype bin; then limit to capacity."""
    best = {}
    for ident in ids:
        record = records[ident]
        key = tuple(record["selection_bin"])
        if key not in best or (record["selection"]["mean_reward"], ident) > (
                records[best[key]]["selection"]["mean_reward"], best[key]):
            best[key] = ident
    return sorted(best.values(), key=lambda i: (-records[i]["selection"]["mean_reward"], i))[:capacity]


def challenge_world(seed, horizon):
    """Deterministic synthetic cul-de-sac, material patch and switched goal."""
    w = World.generate(seed, horizon)
    # East of the start is a one-cell dead-end; RETRACE is required if entered.
    w.obstacles.update(((5, 11), (4, 10), (4, 12)))
    w.obstacles.update((x, 15) for x in range(11, 19) if x != 16)
    w.protected.update((x, y) for x in (13, 14) for y in (8, 9, 10))
    w.obstacles.difference_update(w.protected | w.painted | {(4, 11)})
    w.protected.discard((4, 11))
    w.goal, w.second_goal = (18, 5), (4, 18)
    return w


def challenge_episode(policy, niche, seed, horizon):
    w = challenge_world(seed, horizon)
    reward = 0.
    while not w.stopped and w.steps < w.max_steps:
        menu = w.legal()
        X = np.stack([w.feature(a) for a in menu])
        chosen = policy.choice(X)
        reward += utility(niche, X[chosen], w)
        w.step(menu[chosen])
    assert not (w.painted & (w.protected | w.obstacles))
    return reward, w


def challenge_eval(policies, niche, seeds, horizon):
    rows = []
    for policy in policies:
        for seed in seeds:
            score, w = challenge_episode(policy, niche, seed, horizon)
            rows.append(dict(reward=score, painted=len(w.painted) - 1,
                             retraces=sum(e["action"] == "RETRACE" for e in w.history),
                             early_stop=int(w.stopped and w.steps < horizon),
                             goal_switched=int(w.switched)))
    return dict(mean_reward=round(float(np.mean([r["reward"] for r in rows])), 5),
                mean_painted=round(float(np.mean([r["painted"] for r in rows])), 3),
                mean_retraces=round(float(np.mean([r["retraces"] for r in rows])), 3),
                early_stop_rate=round(float(np.mean([r["early_stop"] for r in rows])), 3),
                goal_switch_rate=round(float(np.mean([r["goal_switched"] for r in rows])), 3))


def run_p5(p0_output, output, seed=631, members=4, train_episodes=18,
           selection_episodes=8, holdout_episodes=12, strength=.8, horizon=60):
    if not (2 <= members <= 8 and 4 <= train_episodes <= 100 and
            2 <= selection_episodes <= 30 and 2 <= holdout_episodes <= 30 and
            0 < strength <= 2 and 31 <= horizon <= 128):
        raise ValueError("invalid bounded P5 campaign")
    p0_output, output = Path(p0_output), Path(output)
    source = json.loads((p0_output / "results.json").read_text())
    if source.get("format") != "population-research-v1" or source.get("language_model") is not False:
        raise ValueError("requires original P0 synthetic provenance")
    splits = splits_for(seed, train_episodes, selection_episodes, holdout_episodes)
    p0_seeds = {stable_seed(source["seed"], part, i)
                for part, count in (("train", source["train_seed_count"]),
                                    ("selection", source["selection_seed_count"]),
                                    ("test", source["heldout_seed_count"])) for i in range(count)}
    if p0_seeds.intersection(sum(splits.values(), [])):
        raise AssertionError("P5/P0 seed overlap")
    parents = {n: load_policy(p0_output / "models" / (n + ".json")) for n in NICHES}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "gallery").mkdir()
    candidates, policies, byid, comparison = [], {}, {}, {}
    for niche in NICHES:
        comparison[niche] = {}
        group = {a: [] for a in ARMS}
        for member in range(members):
            base = parents[niche].mutate(stable_seed("p5", seed, niche, member))
            styles = STYLES[member % len(STYLES)]
            trained, stats, feedback = train_arms(
                base, niche, splits["train"], styles, stable_seed("p5", seed, niche, member, "sample"), strength)
            if not feedback["exact_signal_multiset_match"]:
                raise AssertionError("sham amplitude not matched")
            for arm, policy in trained.items():
                ident = f"{niche}-{arm}-{member}"
                selection = evaluate(policy, niche, splits["selection"])
                record = dict(id=ident, niche=niche, arm=arm, member=member,
                              style=styles, parent="P0 champion", training=stats[arm],
                              feedback=feedback, selection=selection,
                              selection_bin=list(behaviour_bin(selection)))
                candidates.append(record)
                policies[ident], byid[ident] = policy, record
                group[arm].append(ident)
                (output / "models" / (ident + ".json")).write_text(
                    json.dumps(policy.serialized(), sort_keys=True) + "\n")
        for arm in ARMS:
            ids = group[arm]
            reward_rank = sorted(ids, key=lambda i: (-byid[i]["selection"]["mean_reward"], i))
            elites = elite_ids(ids, byid, capacity=min(3, len(ids)))
            comparison[niche][arm] = dict(member_ids=ids,
                champion_id=reward_rank[0], elite_ids=elites,
                selection_archive_cells=len(set(tuple(byid[i]["selection_bin"]) for i in ids)),
                selection_elite_cells=len(elites),
                reward_matched_ids=reward_rank[:len(elites)])
    # Training and selection have terminated. Held-out worlds are opened ONLY here.
    for niche in NICHES:
        comparison[niche]["controls"] = dict(
            p0=evaluate(parents[niche], niche, splits["holdout"]),
            nonstop_random=evaluate("nonstop_random", niche, splits["holdout"]))
        for arm in ARMS:
            row = comparison[niche][arm]
            ids = row["member_ids"]
            members_p = [policies[i] for i in ids]
            row["heldout_dispersion"] = heldout_dispersion(members_p, niche, splits["holdout"])
            row["heldout_long"] = long_horizon(members_p, niche, splits["holdout"], horizon)
            row["heldout_champion"] = evaluate(policies[row["champion_id"]], niche, splits["holdout"])
            elite_p = [policies[i] for i in row["elite_ids"]]
            row["challenge_all"] = challenge_eval(members_p, niche, splits["holdout"], horizon)
            row["challenge_elites"] = challenge_eval(elite_p, niche, splits["holdout"], horizon)
            reward_matched = [policies[i] for i in row["reward_matched_ids"]]
            row["challenge_reward_matched"] = challenge_eval(reward_matched, niche, splits["holdout"], horizon)
            row["elite_heldout_diversity"] = (heldout_dispersion(elite_p, niche, splits["holdout"])
                                               if len(elite_p) >= 2 else None)
            row["reward_matched_heldout_diversity"] = (heldout_dispersion(reward_matched, niche, splits["holdout"])
                                                        if len(reward_matched) >= 2 else None)
            _, w = challenge_episode(policies[row["champion_id"]], niche, splits["holdout"][0], horizon)
            stem = f"{niche}-{arm}"
            (output / "gallery" / (stem + ".svg")).write_text(draw_svg(w))
            (output / "gallery" / (stem + "-trace.json")).write_text(json.dumps(
                dict(synthetic=True, native=False, challenge=True,
                     policy_id=row["champion_id"], seed=splits["holdout"][0],
                     trajectory=w.history, descriptor=w.descriptor(),
                     painted=sorted(w.painted), protected=sorted(w.protected),
                     obstacles=sorted(w.obstacles)), indent=2) + "\n")
    summary = {arm: dict(
        dispersion=round(float(np.mean([comparison[n][arm]["heldout_long"]["mean_distance"]
                                         for n in NICHES])), 6),
        mean_reward=round(float(np.mean([comparison[n][arm]["heldout_champion"]["mean_reward"]
                                          for n in NICHES])), 6),
        challenge_retraces=round(float(np.mean([comparison[n][arm]["challenge_all"]["mean_retraces"]
                                                 for n in NICHES])), 6),
        archive_cells=sum(comparison[n][arm]["selection_archive_cells"] for n in NICHES))
        for arm in ARMS}
    result = dict(format="population-research-p5", kind="MATCHED_OFF_POLICY_EPISODE_FEEDBACK",
                  native=False, language_model=False, seed=seed, arms=list(ARMS),
                  candidate_count=len(candidates), candidates=candidates,
                  adam_updates=sum(c["training"]["adam_steps"] for c in candidates),
                  train_episodes=train_episodes, selection_episodes=selection_episodes,
                  holdout_episodes=holdout_episodes, members=members,
                  intervention_strength=strength, horizon=horizon,
                  split_hashes={k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                                for k, v in splits.items()},
                  comparison=comparison, summary=summary,
                  limitations=["MLP optimizer training, not pretrained language-model adaptation",
                               "Synthetic worlds/rewards, not native CyberSand game use",
                               "Offline, identical frozen source-policy rollouts; not online RL",
                               "Measured and sham feedback use EXACTLY the same signal multiset per matched member",
                               "MAP-Elites binning is hand-designed; selection archive is not training novelty",
                               "Sham label permutation tests episode alignment, not all sources of feedback noise",
                               "No human ratings, no real PAINT material receipts"])
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=631)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--train-episodes", type=int, default=18)
    ap.add_argument("--selection-episodes", type=int, default=8)
    ap.add_argument("--holdout-episodes", type=int, default=12)
    ap.add_argument("--strength", type=float, default=.8)
    ap.add_argument("--horizon", type=int, default=60)
    args = ap.parse_args()
    result = run_p5(args.p0_output, args.output, args.seed, args.members,
                    args.train_episodes, args.selection_episodes,
                    args.holdout_episodes, args.strength, args.horizon)
    print(json.dumps(dict(seed=args.seed, candidates=result["candidate_count"],
                          updates=result["adam_updates"], summary=result["summary"]), indent=2))


if __name__ == "__main__":
    main()
