"""P1 diversity/observation experiments for game-facing synthetic neural policies.

This is a tiny real-optimizer policy ecology, not pretrained SLM inference,
native Godot PAINT, or a generalisation claim about the actual game.
Only finite host-legal actions are ever passed to the synthetic World.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    from .population_lab import (
        FEATURES, DIRECTIONS, NICHES, Policy, World, draw_svg, episode,
        evaluate, softmax, stable_seed, train, utility,
    )
except ImportError:
    from population_lab import (
        FEATURES, DIRECTIONS, NICHES, Policy, World, draw_svg, episode,
        evaluate, softmax, stable_seed, train, utility,
    )


def load_policy(path: Path) -> Policy:
    """Reload only explicitly schema-compatible original trained weights."""
    raw = json.loads(Path(path).read_text())
    if raw.get("format") != "population-mlp-v1" or raw.get("features") != list(FEATURES):
        raise ValueError("incompatible population-model schema")
    hidden = raw.get("hidden")
    if not isinstance(hidden, int) or not (1 <= hidden <= 128):
        raise ValueError("invalid hidden dimension")
    p = Policy(0, hidden=hidden, lr=float(raw["lr"]),
               temperature=float(raw["temperature"]))
    if set(raw["parameters"]) != set(p.p):
        raise ValueError("missing or unknown parameter")
    for k in p.p:
        v = np.asarray(raw["parameters"][k], dtype=np.float64)
        if v.shape != p.p[k].shape or not np.isfinite(v).all():
            raise ValueError("invalid policy weights")
        p.p[k] = v.copy()
    p.updates = int(raw.get("optimizer_steps", 0))
    # Reinitialize optimizer for further training; no old Adam moments were
    # serialized. Resetting the timestep avoids invalid bias correction.
    p.updates = 0
    return p


def model_sha(policy: Policy) -> str:
    data = json.dumps(policy.serialized(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def select_seed_lists(seed: int, train_count: int, selection_count: int,
                      holdout_count: int):
    splits = {
        name: [stable_seed(seed, "p1", name, i) for i in range(count)]
        for name, count in (
            ("train", train_count),
            ("selection", selection_count),
            ("holdout", holdout_count),
        )
    }
    flat = sum(splits.values(), [])
    if len(flat) != len(set(flat)):
        raise AssertionError("P1 split collision")
    return splits


def candidate_behaviour(policy, niche, seeds):
    """Selection-only behaviour vector: topology + action + spatial statistics."""
    rewards, vectors = [], []
    for s in seeds:
        r, w = episode(policy, niche, s)
        rewards.append(r)
        desc = w.descriptor()
        moves = [h["action"] for h in w.history if h["action"] in DIRECTIONS]
        direction_hist = [moves.count(d) / max(1, len(moves)) for d in DIRECTIONS]
        halves = [
            sum((x >= 11) == east and (y >= 11) == south for x, y in w.painted)
            / max(1, len(w.painted))
            for east, south in ((False, False), (True, False),
                                (False, True), (True, True))
        ]
        stopped_early = float(w.stopped and w.steps < w.max_steps)
        vectors.append(desc + direction_hist + halves + [stopped_early])
    return dict(mean_reward=round(float(np.mean(rewards)), 6),
                mean_vector=np.round(np.mean(vectors, axis=0), 6).tolist(),
                per_seed_reward=np.round(rewards, 5).tolist())


def novelty_scores(records, k=4):
    """kNN over selection-only trajectories, never held-out trajectories."""
    vec = np.asarray([r["selection"]["mean_vector"] for r in records], dtype=float)
    if len(vec) < 2:
        return [0.0] * len(vec)
    distances = np.linalg.norm(vec[:, None, :] - vec[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbours = min(k, len(vec) - 1)
    return np.round(np.mean(np.sort(distances, axis=1)[:, :neighbours], axis=1), 6).tolist()


def nearest_pairwise_distance(records):
    if len(records) < 2:
        return 0.0
    x = np.array([r["selection"]["mean_vector"] for r in records], dtype=float)
    return round(float(np.mean([np.linalg.norm(a-b)
                                for i, a in enumerate(x) for b in x[i+1:]])), 6)


def choose_survivors(records):
    """Choose on selection data ONLY; all candidates, including failures, retained."""
    scores = novelty_scores(records)
    for rec, novelty in zip(records, scores):
        rec["selection_novelty_knn4"] = novelty
        v = rec["selection"]["mean_vector"]
        rec["behaviour_bin"] = [
            min(2, int(v[i] * 3)) for i in (1, 2, 3)
        ]  # retrace, turn, branch bins
    archive = {}
    for r in records:
        key = (r["niche"], *r["behaviour_bin"])
        if key not in archive or r["selection"]["mean_reward"] > archive[key]["selection"]["mean_reward"]:
            archive[key] = r
    chosen = {}
    for niche in sorted({r["niche"] for r in records}):
        group = [r for r in records if r["niche"] == niche]
        reward_best = max(group, key=lambda r: (r["selection"]["mean_reward"], r["id"]))
        rvals = [r["selection"]["mean_reward"] for r in group]
        nvals = [r["selection_novelty_knn4"] for r in group]
        def scale(value, values):
            lo, hi = min(values), max(values)
            return (value - lo) / (hi - lo) if hi > lo else 0.
        # Reward remains co-primary; novelty is not a surrogate for safety.
        diversity_best = max(group, key=lambda r: (
            .55 * scale(r["selection"]["mean_reward"], rvals)
            + .45 * scale(r["selection_novelty_knn4"], nvals), r["id"]))
        chosen[niche] = dict(reward=reward_best["id"], diversity=diversity_best["id"])
    return chosen, sorted(r["id"] for r in archive.values())


def frozen_action_scores(policy, seed):
    """Static prior from the initial observation; unknown actions get mean logit."""
    w = World.generate(seed)
    acts = w.legal()
    x = np.stack([w.feature(a) for a in acts])
    logits = policy.logits(x)
    return dict(zip(acts, (float(z) for z in logits))), float(np.mean(logits))


def observation_probe(policy, niche, seeds):
    """Same-state action/probability divergence and separate frozen-prior rollout.

    Frozen scores remain indexed by action identity. Current host-legal menus
    are always honoured; absent initial actions use the initial mean score.
    This is a static-action-prior *ablation*, not a causal feature-attribution
    experiment or a direct test of learning to respond to goals.
    """
    changed, compared, after_changed, after_compared = 0, 0, 0, 0
    total_l1, frozen_scores = 0., []
    for s in seeds:
        initial, default = frozen_action_scores(policy, s)
        w = World.generate(s)
        while not w.stopped and w.steps < w.max_steps:
            menu = w.legal()
            X = np.stack([w.feature(a) for a in menu])
            live_p = policy.predict(X)
            static_logits = np.array([initial.get(a, default) for a in menu])
            static_p = softmax(static_logits / policy.temperature)
            dynamic = int(np.argmax(live_p))
            frozen = int(np.argmax(static_p))
            compared += 1
            changed += int(dynamic != frozen)
            if w.switched:
                after_compared += 1
                after_changed += int(dynamic != frozen)
            total_l1 += float(np.abs(live_p - static_p).sum()) / 2
            w.step(menu[dynamic])
        # Paired separate rollouts, with exact initial score map and legal masks.
        frozen_world = World.generate(s)
        cumulative = 0.
        while not frozen_world.stopped and frozen_world.steps < frozen_world.max_steps:
            menu = frozen_world.legal()
            logits = [initial.get(a, default) for a in menu]
            choice = int(np.argmax(logits))
            action = menu[choice]
            cumulative += utility(niche, frozen_world.feature(action), frozen_world)
            frozen_world.step(action)
        frozen_scores.append(cumulative)
    return dict(
        same_state_action_divergence=round(changed / max(1, compared), 6),
        same_state_total_variation=round(total_l1 / max(1, compared), 6),
        post_goal_switch_divergence=round(after_changed / max(1, after_compared), 6),
        decision_comparisons=compared,
        post_switch_comparisons=after_compared,
        frozen_prior_mean_reward=round(float(np.mean(frozen_scores)), 5),
        live_mean_reward=evaluate(policy, niche, seeds)["mean_reward"],
        caveat="static initial-action-logit prior with current host-legal mask; unknown initial actions receive mean logit",
    )


def experiment_p1(p0_output: Path, output: Path, seed=620, offspring=4,
                  train_episodes=18, selection_episodes=8, holdout_episodes=12):
    if not (1 <= offspring <= 8 and 4 <= train_episodes <= 150
            and 2 <= selection_episodes <= 64 and 2 <= holdout_episodes <= 64):
        raise ValueError("bounded P1 campaign parameters required")
    p0_output = Path(p0_output)
    output = Path(output)
    source = json.loads((p0_output / "results.json").read_text())
    if source.get("format") != "population-research-v1" or source.get("language_model") is not False:
        raise ValueError("P0 artifact has incompatible provenance")
    seeds = select_seed_lists(seed, train_episodes, selection_episodes, holdout_episodes)
    # Ensure this campaign cannot silently use P0's selection/holdout worlds.
    hashes = {k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
              for k, v in seeds.items()}
    if any(h in source.get("seed_hashes", {}).values() for h in hashes.values()):
        raise AssertionError("P1/P0 split reuse")
    bases = {n: load_policy(p0_output / "models" / (n + ".json")) for n in NICHES}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "gallery").mkdir()

    candidates, policies = [], {}
    for ni, niche in enumerate(NICHES):
        base = bases[niche]
        for variant in range(offspring + 1):
            name = f"{niche}-p1-{variant}"
            if variant == 0:
                policy = base
                lineage = dict(parent="P0 champion", technique="unmodified P0 reload")
                training = dict(optimizer_steps=0, mean_cross_entropy=None)
            else:
                policy = base.mutate(stable_seed(seed, niche, variant, "offspring"))
                rng = np.random.default_rng(stable_seed(seed, niche, variant, "noise"))
                if variant % 3 == 2:
                    for k in policy.p:
                        policy.p[k] += rng.normal(0, .13, policy.p[k].shape)
                    technique = "mutation+large-noise"
                elif variant % 3 == 0:
                    donor_name = NICHES[(ni + variant) % len(NICHES)]
                    donor = bases[donor_name]
                    for k in policy.p:
                        policy.p[k] = .72 * policy.p[k] + .28 * donor.p[k]
                    technique = "cross-niche-weight-mix+repair"
                else:
                    technique = "mutation+retraining"
                lineage = dict(parent=f"{niche} P0 champion",
                               donor=donor_name if technique.startswith("cross") else None,
                               technique=technique)
                # Fully separate new training seeds. No P1 held-out observations
                # are inspected during optimiser updates or survivor selection.
                training = train(policy, niche, seeds["train"],
                                 target_temperature=.32 if variant % 2 else .6)
            sel = candidate_behaviour(policy, niche, seeds["selection"])
            policies[name] = policy
            records = dict(id=name, niche=niche, variant=variant, lineage=lineage,
                           training=training, selection=sel, parameter_sha256=model_sha(policy))
            candidates.append(records)
            (output / "models" / f"{name}.json").write_text(
                json.dumps(policy.serialized(), sort_keys=True) + "\n")
    survivors, archive = choose_survivors(candidates)
    finalists = {
        method: [next(r for r in candidates if r["id"] == survivors[n][method]) for n in NICHES]
        for method in ("reward", "diversity")
    }

    # Held-out data is touched only AFTER candidate generation and selection.
    comparison = {}
    for niche in NICHES:
        comparison[niche] = {}
        for mode in ("reward", "diversity"):
            ident = survivors[niche][mode]
            p = policies[ident]
            comparison[niche][mode] = dict(
                id=ident,
                evaluation=evaluate(p, niche, seeds["holdout"]),
                observation_ablation=observation_probe(p, niche, seeds["holdout"]),
            )
            _, world = episode(p, niche, seeds["holdout"][0])
            (output / "gallery" / f"{niche}-{mode}.svg").write_text(draw_svg(world))
            (output / "gallery" / f"{niche}-{mode}-trace.json").write_text(json.dumps(
                dict(synthetic=True, native=False, model_kind="trained-MLP",
                     seed=seeds["holdout"][0], policy_id=ident, niche=niche,
                     selection_method=mode, descriptor=world.descriptor(),
                     trajectory=world.history, obstacles=sorted(world.obstacles),
                     protected=sorted(world.protected), painted=sorted(world.painted)),
                indent=2) + "\n")
        comparison[niche]["controls"] = {
            "nonstop_random": evaluate("nonstop_random", niche, seeds["holdout"]),
            "persistent": evaluate("persistent", niche, seeds["holdout"]),
            "synthetic_oracle": evaluate("synthetic_oracle", niche, seeds["holdout"]),
            "p0_champion": evaluate(bases[niche], niche, seeds["holdout"]),
        }
    heldout_vectors = {}
    for mode in ("reward", "diversity"):
        chosen = [comparison[n][mode]["evaluation"]["mean_descriptor"] for n in NICHES]
        heldout_vectors[mode] = round(float(np.mean([
            np.linalg.norm(np.array(a) - np.array(b))
            for i, a in enumerate(chosen) for b in chosen[i+1:]
        ])), 6)

    summary = dict(
        format="population-research-p1",
        kind="SYNTHETIC_TRAINED_MLP_NOVELTY_ABLATION",
        native=False, language_model=False,
        source_p0_seed=source["seed"], seed=seed,
        generated_candidates=len(candidates), trained_offspring=len(candidates)-len(NICHES),
        candidate_lineage=candidates,
        splits=dict(counts={k: len(v) for k, v in seeds.items()},
                    sha256=hashes, p0_sha256=source.get("seed_hashes")),
        archive_ids=archive,
        archive_bins=len(archive),
        survivor_ids=survivors,
        selection_descriptor_distance={
            mode: nearest_pairwise_distance(finalists[mode]) for mode in finalists
        },
        holdout_descriptor_distance=heldout_vectors,
        heldout_comparison=comparison,
        limitations=[
            "Artificial reward/World, NOT real CyberSand game/native PAINT",
            "Small actual trained neural MLPs, NOT pretrained SLMs",
            "Host-computed finite legal action menu, NOT learned safety",
            "No human ratings, no claims of artistic preference",
            "Static-prior ablation maps actions by identity and uses current legal mask",
            "Novelty measured on selection worlds only; heldouts untouched until selection",
            "Cross-niche scalar descriptor diversity may understate visual differences",
        ],
    )
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=620)
    ap.add_argument("--offspring", type=int, default=4)
    ap.add_argument("--train-episodes", type=int, default=18)
    ap.add_argument("--selection-episodes", type=int, default=8)
    ap.add_argument("--holdout-episodes", type=int, default=12)
    a = ap.parse_args()
    result = experiment_p1(a.p0_output, a.output, a.seed, a.offspring,
                           a.train_episodes, a.selection_episodes, a.holdout_episodes)
    print("P1 trained offspring:", result["trained_offspring"],
          "archive bins:", result["archive_bins"])
    print("Heldout diversity:", result["holdout_descriptor_distance"])
    for niche in NICHES:
        x = result["heldout_comparison"][niche]
        print(niche, "reward", x["reward"]["evaluation"]["mean_reward"],
              "novelty", x["diversity"]["evaluation"]["mean_reward"],
              "nonstop", x["controls"]["nonstop_random"]["mean_reward"],
              "selection", result["survivor_ids"][niche])


if __name__ == "__main__":
    main()
