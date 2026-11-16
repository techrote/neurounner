"""P7: train-only learned trajectory-embedding evolution with matched controls.

Policies are real Adam-trained tiny MLPs on EXTERNAL SYNTHETIC game worlds;
PCA is a fitted unsupervised linear encoder, not a pretrained neural encoder.
No native CyberSand execution, self-play, SLM adaptation or game edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .population_lab import NICHES, DIRECTIONS, World, episode, draw_svg, evaluate, stable_seed
    from .population_p1 import candidate_behaviour, load_policy, model_sha
    from .population_p3 import long_horizon
    from .population_p5 import challenge_eval, challenge_episode
    from .population_p6 import split_seeds, train_child, topology_bin
except ImportError:
    from population_lab import NICHES, DIRECTIONS, World, episode, draw_svg, evaluate, stable_seed
    from population_p1 import candidate_behaviour, load_policy, model_sha
    from population_p3 import long_horizon
    from population_p5 import challenge_eval, challenge_episode
    from population_p6 import split_seeds, train_child, topology_bin

ARMS = ("reward_pbt", "raw_map", "learned_map", "random_parents")
ACTIONS = tuple(DIRECTIONS) + ("RETRACE", "STOP")


def fingerprint(world):
    """31-dimensional spatial occupancy and action-history, not a style label."""
    out = np.zeros(31, dtype=np.float64)
    for x, y in world.painted:
        out[min(4, 5*x // world.width)*5 + min(4, 5*y // world.height)] += 1
    out[:25] /= max(1, len(world.painted))
    for h in world.history:
        out[25 + ACTIONS.index(h["action"])] += 1
    out[25:] /= max(1, len(world.history))
    if not np.isfinite(out).all() or abs(float(out[:25].sum())-1) > 1e-10:
        raise AssertionError("invalid training trajectory fingerprint")
    return out


def phenotype(policy, niche, seeds):
    return np.mean([fingerprint(episode(policy, niche, s)[1]) for s in seeds], axis=0)


def fit_encoder(parents, seed, probe_count=12, probe_worlds=4):
    """Only independent TRAIN-probe worlds may fit this fixed PCA encoder."""
    seeds = [stable_seed("p7", seed, "encoder_train", i) for i in range(probe_worlds)]
    x = []
    for niche in NICHES:
        x.append(phenotype(parents[niche], niche, seeds))
        for i in range(probe_count):
            probe = parents[niche].mutate(stable_seed("p7", seed, niche, "probe", i))
            x.append(phenotype(probe, niche, seeds))
    x = np.stack(x)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .02)
    u, singular, vt = np.linalg.svd((x-mean) / scale, full_matrices=False)
    projection = vt[:3].T.copy()
    for i in range(3):
        dominant = int(np.argmax(abs(projection[:, i])))
        if projection[dominant, i] < 0:
            projection[:, i] *= -1
    z = ((x-mean) / scale) @ projection
    cuts = np.quantile(z, [.25, .5, .75], axis=0)
    if len(np.unique(x, axis=0)) < 4 or not np.isfinite(cuts).all():
        raise AssertionError("collapsed/invalid train-only embedding")
    enc = dict(mean=mean.tolist(), scale=scale.tolist(), projection=projection.tolist(),
               cuts=cuts.tolist(), train_seeds=seeds, samples=len(x),
               top3_fraction_variance=float((singular[:3]**2).sum() / (singular**2).sum()),
               method="TRAIN-only PCA of 5x5 visit occupancy + 6 action frequencies")
    enc["sha256"] = hashlib.sha256(json.dumps(enc, sort_keys=True).encode()).hexdigest()
    return enc


def learned_cell(vec, encoder):
    x = np.asarray(vec, dtype=float)
    mean = np.asarray(encoder["mean"])
    scale = np.asarray(encoder["scale"])
    projection = np.asarray(encoder["projection"])
    cuts = np.asarray(encoder["cuts"])
    if x.shape != (31,) or not np.isfinite(x).all():
        raise ValueError("invalid trajectory vector")
    z = ((x-mean)/scale) @ projection
    return [int(np.searchsorted(cuts[:, i], z[i], side="right")) for i in range(3)], z.tolist()


def archive_add(archive, records, cell_name):
    out = dict(archive)
    for rec in records:
        cell = tuple(rec[cell_name])
        prev = out.get(cell)
        if prev is None or (rec["selection"]["mean_reward"], rec["id"]) > (
                prev["selection"]["mean_reward"], prev["id"]):
            out[cell] = rec
    return out


def choose_parent(arm, generation, child, seed, niche, archive, best, random_pool):
    if generation == 0:
        return None
    if arm == "reward_pbt":
        return best
    rng = np.random.default_rng(stable_seed("p7", seed, niche, arm, generation, child, "parent"))
    if arm in ("raw_map", "learned_map"):
        cells = sorted(archive)
        return archive[cells[int(rng.integers(len(cells)))]]
    if arm == "random_parents":
        return random_pool[int(rng.integers(len(random_pool)))]
    raise ValueError(arm)


def choose_portfolio(records, archive, arm, size, seed):
    ranked = sorted(records, key=lambda x: (-x["selection"]["mean_reward"], x["id"]))
    if arm in ("raw_map", "learned_map"):
        chosen = [r["id"] for r in sorted(archive.values(),
                    key=lambda r: (-r["selection"]["mean_reward"], r["id"]))[:size]]
        chosen.extend([r["id"] for r in ranked if r["id"] not in chosen][:size-len(chosen)])
        return chosen
    if arm == "reward_pbt":
        return [r["id"] for r in ranked[:size]]
    rng = np.random.default_rng(seed)
    return [records[int(i)]["id"] for i in rng.choice(len(records), size, replace=False)]


def run_p7(p0_output, output, seed=636, generations=3, offspring=4,
           train_episodes=8, selection_episodes=5, holdout_episodes=6,
           horizon=48, portfolio=4, probe_count=12, probe_worlds=4):
    if not (2 <= generations <= 6 and 3 <= offspring <= 10 and
            4 <= train_episodes <= 50 and 3 <= selection_episodes <= 30 and
            3 <= holdout_episodes <= 30 and 31 <= horizon <= 128 and
            2 <= portfolio <= offspring and 4 <= probe_count <= 50 and 3 <= probe_worlds <= 30):
        raise ValueError("P7 invalid bounded campaign parameters")
    p0_output, output = Path(p0_output), Path(output)
    source = json.loads((p0_output / "results.json").read_text())
    if source.get("format") != "population-research-v1" or source.get("language_model") is not False:
        raise ValueError("P7 requires untouched original synthetic P0 provenance")
    splits = split_seeds(seed, train_episodes, selection_episodes, holdout_episodes, generations)
    parents = {n: load_policy(p0_output / "models" / (n + ".json")) for n in NICHES}
    enc = fit_encoder(parents, seed, probe_count, probe_worlds)
    all_seeds = sum(splits["train"], []) + splits["selection"] + splits["holdout"] + enc["train_seeds"]
    p0_seeds = [stable_seed(source["seed"], part, i)
                for part, count in (("train", source["train_seed_count"]),
                                    ("selection", source["selection_seed_count"]),
                                    ("test", source["heldout_seed_count"])) for i in range(count)]
    if len(set(all_seeds)) != len(all_seeds) or set(all_seeds) & set(p0_seeds):
        raise AssertionError("P0/P7 training/encoder/selection/holdout leakage")
    output.mkdir(parents=True, exist_ok=False)
    (output/"models").mkdir()
    (output/"gallery").mkdir()
    (output/"encoder.json").write_text(json.dumps(enc, indent=2) + "\n")
    all_records, comparison, models = [], {}, {}
    for niche in NICHES:
        comparison[niche] = {}
        for arm in ARMS:
            archive, records, random_pool, best = {}, [], [], None
            occupancy = []
            for g in range(generations):
                batch = []
                for child in range(offspring):
                    parent_rec = choose_parent(arm, g, child, seed, niche, archive, best, random_pool)
                    parent = parents[niche] if parent_rec is None else models[parent_rec["id"]]
                    mutation_seed = stable_seed("p7", seed, niche, g, child, "mutate")
                    policy = parent.mutate(mutation_seed)
                    ident = f"{niche}-{arm}-g{g}-{child}"
                    initial = model_sha(policy)
                    training = train_child(policy, niche, splits["train"][g])
                    selection = candidate_behaviour(policy, niche, splits["selection"])
                    vec = phenotype(policy, niche, splits["selection"])
                    cell, z = learned_cell(vec, enc)
                    record = dict(id=ident, niche=niche, arm=arm, generation=g, offspring=child,
                                  parent=None if parent_rec is None else parent_rec["id"],
                                  mutation_seed=mutation_seed, initial_sha256=initial,
                                  final_sha256=model_sha(policy), training=training,
                                  selection=selection, learned_cell=cell,
                                  learned_projection=np.round(z, 6).tolist(),
                                  raw_cell=list(topology_bin(selection)))
                    if record["initial_sha256"] == record["final_sha256"]:
                        raise AssertionError("optimizer did not change weights")
                    records.append(record)
                    batch.append(record)
                    all_records.append(record)
                    models[ident] = policy
                    (output/"models"/(ident+".json")).write_text(json.dumps(policy.serialized(), sort_keys=True)+"\n")
                if arm in ("raw_map", "learned_map"):
                    archive = archive_add(archive, batch, "raw_cell" if arm=="raw_map" else "learned_cell")
                best = max(records, key=lambda r:(r["selection"]["mean_reward"], r["id"]))
                rng = np.random.default_rng(stable_seed("p7", seed, niche, arm, g, "pool"))
                random_pool = [batch[int(i)] for i in rng.permutation(len(batch))]
                occupancy.append(len(archive))
            selected = choose_portfolio(records, archive, arm, portfolio,
                                        stable_seed("p7", seed, niche, arm, "portfolio"))
            chosen = [models[x] for x in selected]
            comp = dict(champion_id=best["id"], selected_ids=selected,
                        selection_archive_cells=len(archive), occupancy_by_generation=occupancy,
                        candidate_count=len(records),
                        selection_learned_cells=len({tuple(r["learned_cell"]) for r in records}),
                        selection_raw_cells=len({tuple(r["raw_cell"]) for r in records}))
            # HOLDOUT FIRST READ: only after all ancestry, archives and portfolio choice.
            comp["heldout_long"] = long_horizon(chosen, niche, splits["holdout"], horizon)
            comp["heldout_reward"] = evaluate(models[best["id"]], niche, splits["holdout"])
            comp["challenge"] = challenge_eval(chosen, niche, splits["holdout"], horizon)
            comparison[niche][arm] = comp
            for i, ident in enumerate(selected):
                _, w = challenge_episode(models[ident], niche, splits["holdout"][0], horizon)
                name = f"{niche}-{arm}-{i}"
                (output/"gallery"/(name+".svg")).write_text(draw_svg(w))
                (output/"gallery"/(name+"-trace.json")).write_text(json.dumps(
                    dict(synthetic=True, native=False, slm=False, policy_id=ident, seed=splits["holdout"][0],
                         trajectory=w.history, painted=sorted(w.painted),
                         protected=sorted(w.protected), obstacles=sorted(w.obstacles)), indent=2)+"\n")
    summary = {arm: dict(
        heldout_diversity=round(float(np.mean([comparison[n][arm]["heldout_long"]["mean_distance"]
                                         for n in NICHES])), 6),
        heldout_reward=round(float(np.mean([comparison[n][arm]["heldout_reward"]["mean_reward"]
                                      for n in NICHES])), 6),
        challenge_retraces=round(float(np.mean([comparison[n][arm]["challenge"]["mean_retraces"]
                                          for n in NICHES])), 5),
        selection_archive_cells=sum(comparison[n][arm]["selection_archive_cells"] for n in NICHES))
        for arm in ARMS}
    result = dict(format="population-research-p7", kind="TRAIN_ONLY_PCA_EVOLUTION_ABLATION",
                  native=False, language_model=False, reinforcement_learning=False, self_play=False,
                  game_code_modified=False, seed=seed, arms=list(ARMS), encoder_sha256=enc["sha256"],
                  generations=generations, offspring=offspring, portfolio=portfolio,
                  candidate_count=len(all_records),
                  adam_updates=sum(r["training"]["adam_steps"] for r in all_records),
                  records=all_records, comparison=comparison, summary=summary,
                  splits_sha256={k:hashlib.sha256(json.dumps(v).encode()).hexdigest() for k,v in splits.items()},
                  horizon=horizon, limitations=[
                      "Only external synthetic 22x22 host-legal worlds, not native CyberSand acceptance",
                      "Trained tiny MLP game policies, not pretrained SLM students",
                      "Learned embedding is unsupervised linear PCA of hand-selected spatial/action features, not a neural aesthetic judge",
                      "All arms use teacher-augmented supervised Adam, not self-play or policy-gradient reinforcement learning",
                      "Selection worlds reused across generations; selection overfit possible",
                      "Ancestry changes visited training distributions; only generation-zero starts are matched",
                      "Host legal-action masking does not demonstrate model-internal safety",
                      "Synthetic reward is handcrafted, not independent human-rated artwork quality"])
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p0-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=636)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--offspring", type=int, default=4)
    ap.add_argument("--train-episodes", type=int, default=8)
    ap.add_argument("--selection-episodes", type=int, default=5)
    ap.add_argument("--holdout-episodes", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--portfolio", type=int, default=4)
    ap.add_argument("--probe-count", type=int, default=12)
    ap.add_argument("--probe-worlds", type=int, default=4)
    args = ap.parse_args()
    r=run_p7(args.p0_output,args.output,args.seed,args.generations,args.offspring,
             args.train_episodes,args.selection_episodes,args.holdout_episodes,
             args.horizon,args.portfolio,args.probe_count,args.probe_worlds)
    print(json.dumps(dict(seed=r["seed"], models=r["candidate_count"], adam_updates=r["adam_updates"],
                          encoder=r["encoder_sha256"], summary=r["summary"]), indent=2))