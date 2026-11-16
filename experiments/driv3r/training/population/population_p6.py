"""P6: multi-generation reward-PBT vs MAP-Elites vs random-parent control.

Trains real small neural policies on external *synthetic* CyberSand-like worlds.
Selection-only feature/topology archives drive mating, not just final ranking.
Not native CyberSand, reinforcement learning, self-play, or an SLM adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .population_lab import NICHES, World, draw_svg, episode, evaluate, softmax, stable_seed, utility
    from .population_p1 import candidate_behaviour, load_policy, model_sha
    from .population_p3 import long_horizon
    from .population_p5 import challenge_eval, challenge_episode, challenge_world
except ImportError:
    from population_lab import NICHES, World, draw_svg, episode, evaluate, softmax, stable_seed, utility
    from population_p1 import candidate_behaviour, load_policy, model_sha
    from population_p3 import long_horizon
    from population_p5 import challenge_eval, challenge_episode, challenge_world

ARMS = ("reward_pbt", "map_elites", "random_parents")


def split_seeds(seed, train, selection, holdout, generations):
    d = {"selection": [stable_seed("p6", seed, "selection", i) for i in range(selection)],
         "holdout": [stable_seed("p6", seed, "holdout", i) for i in range(holdout)]}
    d["train"] = [[stable_seed("p6", seed, "train", g, i) for i in range(train)]
                  for g in range(generations)]
    flat = d["selection"] + d["holdout"] + sum(d["train"], [])
    if len(flat) != len(set(flat)):
        raise AssertionError("P6 split leakage")
    return d


def topology_bin(selection):
    """Predeclared, selection-only descriptors; no holdout-dependent bins."""
    d = selection["mean_vector"]
    if len(d) != 14 or not np.isfinite(d).all():
        raise ValueError("invalid candidate phenotype")
    # Paint coverage, retrace/branch incidence, turn rate, dominant direction.
    # Boundaries are fixed, not tuned by held-out observations.
    return (min(2, int(d[0] * 3)), min(2, int(d[1] * 9)),
            min(2, int(d[2] * 3)), min(2, int(d[3] * 18)),
            int(np.argmax(d[5:9])))


def update_archive(archive, records):
    """One highest-selection-reward occupant per topology cell, cumulatively."""
    updated = dict(archive)
    for record in records:
        cell = tuple(record["selection_bin"])
        old = updated.get(cell)
        if old is None or (record["selection"]["mean_reward"], record["id"]) > (
                old["selection"]["mean_reward"], old["id"]):
            updated[cell] = record
    return updated


def parents_for(arm, archive, best, random_pool, generation, child, seed, niche):
    if generation == 0:
        return None
    if arm == "reward_pbt":
        return best
    rng = np.random.default_rng(stable_seed("p6", seed, niche, arm, generation, child, "parent"))
    if arm == "map_elites":
        cells = sorted(archive)
        return archive[cells[int(rng.integers(len(cells)))]]
    if arm == "random_parents":
        return random_pool[int(rng.integers(len(random_pool)))]
    raise ValueError(arm)


def portfolio_ids(records, archive, arm, size, seed):
    """Exactly size unique policies: archive occupants first for MAP-Elites.

    Others are reward-top-k and uniform sample, respectively. If the archive
    has fewer cells than size, fill from top unarchived policies and report it.
    """
    ranked = sorted(records, key=lambda x: (-x["selection"]["mean_reward"], x["id"]))
    if arm == "map_elites":
        chosen = sorted(archive.values(), key=lambda x: (-x["selection"]["mean_reward"], x["id"]))[:size]
        chosen = [x["id"] for x in chosen]
        chosen += [x["id"] for x in ranked if x["id"] not in chosen][:size-len(chosen)]
        return chosen
    if arm == "reward_pbt":
        return [x["id"] for x in ranked[:size]]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=size, replace=False)
    return [records[int(i)]["id"] for i in indices]


def train_child(policy, niche, seeds):
    """Real Adam updates from on-policy, teacher-augmented state visits.

    Alternates the standard synthetic world and the P5 dead-end/protected
    challenge. This is a fixed common curriculum, *not* claimed self-play.
    """
    start = policy.updates
    losses, actions_total, retraces, stops = [], 0, 0, 0
    for i, seed in enumerate(seeds):
        w = World.generate(seed) if i % 2 == 0 else challenge_world(seed, 30)
        rng = np.random.default_rng(stable_seed("p6", seed, niche, "rollout"))
        while not w.stopped and w.steps < w.max_steps:
            menu = w.legal()
            X = np.stack([w.feature(a) for a in menu])
            target = softmax(np.asarray([utility(niche, x, w) for x in X]) / .42)
            losses.append(policy.teach(X, target))
            chosen = int(rng.choice(len(menu), p=target)) if rng.random() < .25 else policy.choice(X, rng)
            action = menu[chosen]
            retraces += int(action == "RETRACE")
            stops += int(action == "STOP")
            actions_total += 1
            w.step(action)
            if w.painted & (w.protected | w.obstacles):
                raise AssertionError("painted illegal/protected game cell")
    if policy.updates <= start or not losses:
        raise AssertionError("no neural training")
    return dict(adam_steps=policy.updates-start, mean_cross_entropy=round(float(np.mean(losses)), 6),
                curriculum_actions=actions_total, curriculum_retraces=retraces, curriculum_stops=stops)


def run_p6(p0_output, output, seed=634, generations=3, offspring=4,
           train_episodes=8, selection_episodes=5, holdout_episodes=6, horizon=48,
           portfolio=4):
    if not (2 <= generations <= 8 and 3 <= offspring <= 12 and
            4 <= train_episodes <= 100 and 3 <= selection_episodes <= 30 and
            3 <= holdout_episodes <= 30 and 31 <= horizon <= 128 and
            2 <= portfolio <= offspring):
        raise ValueError("invalid bounded P6 campaign")
    p0_output, output = Path(p0_output), Path(output)
    source = json.loads((p0_output / "results.json").read_text())
    if source.get("format") != "population-research-v1" or source.get("language_model") is not False:
        raise ValueError("P6 requires original P0 synthetic provenance")
    splits = split_seeds(seed, train_episodes, selection_episodes, holdout_episodes, generations)
    p0_seeds = {stable_seed(source["seed"], part, i)
                for part, count in (("train", source["train_seed_count"]),
                                    ("selection", source["selection_seed_count"]),
                                    ("test", source["heldout_seed_count"])) for i in range(count)}
    if p0_seeds.intersection(splits["selection"] + splits["holdout"] + sum(splits["train"], [])):
        raise AssertionError("P0/P6 seed overlap")
    parents = {n: load_policy(p0_output / "models" / (n + ".json")) for n in NICHES}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "gallery").mkdir()
    all_records, comparisons, models = [], {}, {}
    for niche in NICHES:
        comparisons[niche] = {}
        for arm in ARMS:
            archive, records, random_pool = {}, [], []
            best = None
            occupancy_history = []
            for g in range(generations):
                generation_records = []
                for child in range(offspring):
                    parent_rec = parents_for(arm, archive, best, random_pool, g, child, seed, niche)
                    parent = parents[niche] if parent_rec is None else models[parent_rec["id"]]
                    mutation_seed = stable_seed("p6", seed, niche, g, child, "mutate")
                    policy = parent.mutate(mutation_seed)
                    ident = f"{niche}-{arm}-g{g}-{child}"
                    initial_sha = model_sha(policy)
                    training = train_child(policy, niche, splits["train"][g])
                    selection = candidate_behaviour(policy, niche, splits["selection"])
                    record = dict(id=ident, niche=niche, arm=arm, generation=g, offspring=child,
                                  parent=None if parent_rec is None else parent_rec["id"],
                                  mutation_seed=mutation_seed, initial_sha256=initial_sha,
                                  final_sha256=model_sha(policy), training=training,
                                  selection=selection, selection_bin=list(topology_bin(selection)))
                    if record["initial_sha256"] == record["final_sha256"]:
                        raise AssertionError("Adam failed to modify weights")
                    generation_records.append(record)
                    records.append(record)
                    all_records.append(record)
                    models[ident] = policy
                    (output / "models" / (ident + ".json")).write_text(
                        json.dumps(policy.serialized(), sort_keys=True) + "\n")
                archive = update_archive(archive, generation_records)
                best = max(records, key=lambda r: (r["selection"]["mean_reward"], r["id"]))
                rng = np.random.default_rng(stable_seed("p6", seed, niche, arm, g, "pool"))
                random_pool = [generation_records[int(i)] for i in rng.permutation(len(generation_records))]
                occupancy_history.append(len(archive))
            selected = portfolio_ids(records, archive, arm, portfolio,
                                     stable_seed("p6", seed, niche, arm, "portfolio"))
            selected_policies = [models[i] for i in selected]
            comparisons[niche][arm] = dict(candidate_ids=[r["id"] for r in records],
                champion_id=best["id"], selected_ids=selected,
                archive_size=len(archive), archive_cells=[list(c) for c in sorted(archive)],
                occupancy_by_generation=occupancy_history,
                selection_champion_reward=best["selection"]["mean_reward"])
            # *** Strict holdout gate: no heldout is used before all parent selection. ***
            comparison = comparisons[niche][arm]
            comparison["heldout_long"] = long_horizon(selected_policies, niche, splits["holdout"], horizon)
            comparison["heldout_champion"] = evaluate(models[best["id"]], niche, splits["holdout"])
            comparison["challenge"] = challenge_eval(selected_policies, niche, splits["holdout"], horizon)
            comparison["heldout_bins"] = len({topology_bin(candidate_behaviour(p, niche, splits["holdout"]))
                                               for p in selected_policies})
            for index, ident in enumerate(selected):
                _, w = challenge_episode(models[ident], niche, splits["holdout"][0], horizon)
                name = f"{niche}-{arm}-{index}"
                (output / "gallery" / (name + ".svg")).write_text(draw_svg(w))
                (output / "gallery" / (name + "-trace.json")).write_text(json.dumps(
                    dict(synthetic=True, native=False, language_model=False, seed=splits["holdout"][0],
                         policy_id=ident, trajectory=w.history, descriptor=w.descriptor(),
                         painted=sorted(w.painted), protected=sorted(w.protected),
                         obstacles=sorted(w.obstacles)), indent=2) + "\n")
    summary = {arm: dict(
        heldout_diversity=round(float(np.mean([comparisons[n][arm]["heldout_long"]["mean_distance"]
                                             for n in NICHES])), 6),
        heldout_reward=round(float(np.mean([comparisons[n][arm]["heldout_champion"]["mean_reward"]
                                          for n in NICHES])), 6),
        heldout_bins=sum(comparisons[n][arm]["heldout_bins"] for n in NICHES),
        selection_archive_cells=sum(comparisons[n][arm]["archive_size"] for n in NICHES),
        challenge_retraces=round(float(np.mean([comparisons[n][arm]["challenge"]["mean_retraces"]
                                              for n in NICHES])), 5)) for arm in ARMS}
    result = dict(format="population-research-p6", kind="MULTIGENERATION_MAP_ELITES_PBT",
                  native=False, language_model=False, reinforcement_learning=False,
                  game_code_modified=False, seed=seed, arms=list(ARMS),
                  generations=generations, offspring=offspring, portfolio=portfolio,
                  candidate_count=len(all_records), adam_updates=sum(x["training"]["adam_steps"]
                                                              for x in all_records),
                  records=all_records, comparison=comparisons, summary=summary,
                  splits_sha256={k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
                                 for k, v in splits.items()}, horizon=horizon,
                  limitations=["Synthetic 22x22 host-masked worlds, not native CyberSand / Workbench",
                               "Actual Adam-trained tiny MLPs, not pretrained SLM adapters",
                               "Handcrafted reward and topology bins, not learned aesthetics",
                               "MAP-Elites and random-parent effects are selection/evolution, not true self-play",
                               "Different ancestral selection changes training state distributions after generation 0",
                               "Selection-only worlds are reused across generations and can overfit",
                               "Legal-action rate is guaranteed by the host catalog, not learned safety",
                               "Only selected portfolios receive held-out evaluation; no holdout tuning"])
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p0-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=634)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--offspring", type=int, default=4)
    p.add_argument("--train-episodes", type=int, default=8)
    p.add_argument("--selection-episodes", type=int, default=5)
    p.add_argument("--holdout-episodes", type=int, default=6)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--portfolio", type=int, default=4)
    a = p.parse_args()
    r = run_p6(a.p0_output, a.output, a.seed, a.generations, a.offspring,
               a.train_episodes, a.selection_episodes, a.holdout_episodes,
               a.horizon, a.portfolio)
    print(json.dumps(dict(seed=a.seed, candidates=r["candidate_count"],
                          adam_updates=r["adam_updates"], summary=r["summary"]), indent=2))


if __name__ == "__main__":
    main()
