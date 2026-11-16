"""DG017: on-policy corrective neural donor transfer on random gated mazes.

Research-only synthetic CyberSand-like game. No CyberSand source or assets changed.
Extends DG016 without changing its solver or training artifacts. DG017 records
negative results and explicitly separates history observability from DAgger.
"""
import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.memplant.trajectory_transfer_016 import experiment as prior

SIDE = prior.SIDE
CHANNELS = prior.CHANNELS + 1
HELD = prior.HELD
TRAIN_LAWS = prior.TRAIN_LAWS
MODES = ("legacy", "history_bc", "dagger", "blind_dagger",
         "wrong_wet", "wrong_goal", "oracle")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HistoryGame(prior.Game):
    """Adds host-observable visit counts; no shortest path or teacher potential."""
    def observe(self):
        obs, legal = super().observe()
        visits = np.zeros((1, SIDE, SIDE), dtype=np.float32)
        for (x, y), n in self.visits.items():
            visits[0, y, x] = min(1.0, np.log1p(n) / np.log1p(16))
        return np.concatenate((obs, visits), axis=0), legal


class Receiver(nn.Module):
    def __init__(self, conditioned):
        super().__init__()
        self.conditioned = conditioned
        dimensions = CHANNELS * SIDE * SIDE + (24 if conditioned else 0)
        self.net = nn.Sequential(nn.Linear(dimensions, 192), nn.ReLU(),
                                 nn.Linear(192, 128), nn.ReLU(),
                                 nn.Linear(128, 6))

    def forward(self, obs, legal, signatures=None):
        z = obs.flatten(1)
        if self.conditioned:
            z = torch.cat((z, signatures), -1)
        return self.net(z).masked_fill(~legal, -1e5)


def graft_legacy(legacy):
    """Non-destructive first-layer surgery; preserves exact legacy outputs initially."""
    new = Receiver(legacy.conditioned)
    with torch.no_grad():
        old_first = legacy.net[0]
        new_first = new.net[0]
        new_first.weight.zero_()
        original = prior.CHANNELS * SIDE * SIDE
        new_first.weight[:, :original] = old_first.weight[:, :original]
        if legacy.conditioned:
            new_first.weight[:, CHANNELS * SIDE * SIDE:] = old_first.weight[:, original:]
        new_first.bias.copy_(old_first.bias)
        for idx in (2, 4):
            new.net[idx].load_state_dict(legacy.net[idx].state_dict())
    return new.eval()


def dataset(seed, bank, worlds):
    obs, masks, signatures, labels = [], [], [], []
    for i in range(worlds):
        world = prior.maze(seed + i * 79, "train")
        for law in TRAIN_LAWS:
            game = HistoryGame(world, law)
            sig = prior.base.signatures(bank, "narrow", law)
            for _ in range(96):
                if game.done:
                    break
                x, mask = game.observe()
                action = game.oracle()
                if action is None:
                    break
                assert mask[action]
                obs.append(x)
                masks.append(mask)
                signatures.append(sig)
                labels.append(action)
                game.step(action)
    return (torch.from_numpy(np.stack(obs)), torch.from_numpy(np.stack(masks)),
            torch.from_numpy(np.stack(signatures)), torch.tensor(labels, dtype=torch.long))


def fit(model, base_data, seed, updates, extra=None):
    """Corrective training retains a 50% fresh expert-trajectory anchor."""
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0010, weight_decay=0.0001)
    rng = np.random.default_rng(seed + 39)
    x, masks, sigs, labels = base_data
    for _ in range(updates):
        if extra is None:
            ix = torch.from_numpy(rng.integers(len(labels), size=192))
            xx, mm, ss, yy = x[ix], masks[ix], sigs[ix], labels[ix]
        else:
            a = torch.from_numpy(rng.integers(len(labels), size=96))
            b = torch.from_numpy(rng.integers(len(extra[3]), size=96))
            xx = torch.cat((x[a], extra[0][b]))
            mm = torch.cat((masks[a], extra[1][b]))
            ss = torch.cat((sigs[a], extra[2][b]))
            yy = torch.cat((labels[a], extra[3][b]))
        logits = model(xx, mm, ss if model.conditioned else None)
        loss = F.cross_entropy(logits, yy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


def collect(seed, bank, model, worlds, horizon, teacher_fraction):
    """DAgger-like rollouts; oracle relabels states actually visited by receiver.

    Stores model disagreements and a random sample of agreements, capped per
    episode, to avoid overweighting repeated trapped trajectories.
    """
    rng = np.random.default_rng(seed)
    collected = [[], [], [], []]
    episodes = 0
    completed = 0
    corrections = 0
    samples = 0
    unique_states = set()
    for i in range(worlds):
        world = prior.maze(seed + 10000 + i * 89, "train")
        for law in TRAIN_LAWS:
            game = HistoryGame(world, law)
            signature = prior.base.signatures(bank, "narrow", law)
            kept = 0
            episodes += 1
            for step in range(horizon):
                if game.done:
                    break
                x, legal = game.observe()
                oracle_action = game.oracle()
                if oracle_action is None:
                    break
                with torch.no_grad():
                    action = int(model(torch.from_numpy(x[None]),
                                       torch.from_numpy(legal[None]),
                                       torch.from_numpy(signature[None]) if model.conditioned else None)
                                 .argmax(-1).item())
                assert legal[action] and legal[oracle_action]
                disagreement = action != oracle_action
                corrections += int(disagreement)
                # Preserve salient errors; deduplicate an entire episode's exact observations.
                key = (i, law, x.tobytes())
                if kept < 20 and (disagreement or rng.random() < .14) and key not in unique_states:
                    unique_states.add(key)
                    collected[0].append(x.copy())
                    collected[1].append(legal.copy())
                    collected[2].append(signature.copy())
                    collected[3].append(oracle_action)
                    kept += 1
                    samples += 1
                # A small teacher-action mixture provides opportunities to recover.
                game.step(oracle_action if rng.random() < teacher_fraction else action)
            completed += int(game.done)
    assert samples > 0
    tensor_data = (torch.from_numpy(np.stack(collected[0])),
                   torch.from_numpy(np.stack(collected[1])),
                   torch.from_numpy(np.stack(collected[2])),
                   torch.tensor(collected[3], dtype=torch.long))
    return tensor_data, dict(episodes=episodes, completed=completed,
                             disagreements=corrections, labels=samples,
                             teacher_fraction=teacher_fraction)


def signature_for(bank, law, mode):
    if mode == "wrong_wet":
        donor = (-law[0], law[1], law[2])
    elif mode == "wrong_goal":
        donor = (law[0], -law[1], law[2])
    else:
        donor = law
    return prior.base.signatures(bank, "gated", donor)


def play(world, law, mode, bank, models, horizon):
    if mode == "legacy":
        return prior.play(world, law, "inherited", bank,
                          {"inherited": models["legacy"], "blind": models["legacy_blind"]}, horizon)
    game = HistoryGame(world, law)
    signature = signature_for(bank, law, mode)
    if mode == "history_bc":
        model = models["history_bc"]
    elif mode == "blind_dagger":
        model = models["blind_dagger"]
    else:
        model = models["dagger"]
    trace = [game.pos]
    actions = []
    for _ in range(horizon):
        if game.done:
            break
        x, legal = game.observe()
        if mode == "oracle":
            action = game.oracle()
        else:
            with torch.no_grad():
                action = int(model(torch.from_numpy(x[None]),
                                   torch.from_numpy(legal[None]),
                                   torch.from_numpy(signature[None]) if model.conditioned else None)
                             .argmax(-1).item())
        if action is None:
            break
        assert legal[action]
        game.step(action)
        trace.append(game.pos)
        actions.append(prior.ACTIONS[action])
    return dict(success=int(game.done), cost=round(game.travel_cost, 4),
                steps=game.steps, closure=int(game.closed), retraces=game.retraces,
                illegal=game.illegal, wet=game.wet_steps,
                path=trace, actions=actions, painted=len(game.painted))


def summary(rows, horizon):
    return {
        mode: dict(worlds=len(vals), success=sum(v["success"] for v in vals),
                   illegal=sum(v["illegal"] for v in vals),
                   total_retraces=sum(v["retraces"] for v in vals),
                   failure_penalized_mean_cost=round(float(np.mean([
                       v["cost"] + (0 if v["success"] else 3 * horizon) for v in vals
                   ])), 4),
                   successful_mean_cost=round(float(np.mean([
                       v["cost"] for v in vals if v["success"]
                   ])), 4) if any(v["success"] for v in vals) else None)
        for mode, vals in rows.items()
    }


def evaluate(seed, bank, models, worlds, horizon):
    results, samples = {}, []
    for family in ("iid", "ood"):
        rows = {mode: [] for mode in MODES}
        for i in range(worlds):
            s = seed + i * 103 + (10000000 if family == "ood" else 0)
            world = prior.maze(s, family)
            law = HELD if i % 2 == 0 else TRAIN_LAWS[(i // 2) % len(TRAIN_LAWS)]
            trials = {mode: play(world, law, mode, bank, models, horizon) for mode in MODES}
            for mode in MODES:
                rows[mode].append(trials[mode])
            if i < 4:
                samples.append(dict(family=family, seed=s, laws=law,
                     wall=world["wall"].astype(int).tolist(),
                     wet=world["wet"].astype(int).tolist(), gate=world["gate"],
                     close_after=world["close_after"], start=world["start"],
                     goal=world["goal"], alternate=world["alternate"], trials=trials))
        pair = [i for i in range(worlds)
                if rows["dagger"][i]["success"] and rows["history_bc"][i]["success"]
                and rows["oracle"][i]["success"]]
        regret = [dict(da=rows["dagger"][i]["cost"] / max(.001, rows["oracle"][i]["cost"]) - 1,
                       bc=rows["history_bc"][i]["cost"] / max(.001, rows["oracle"][i]["cost"]) - 1)
                  for i in pair]
        results[family] = dict(trajectories=summary(rows, horizon),
            withheld_worlds=(worlds + 1) // 2,
            withheld_success={mode: sum(row["success"] for j, row in enumerate(rows[mode])
                                         if j % 2 == 0) for mode in MODES},
            paired_cost=dict(joint_success=len(pair),
                dagger_mean_regret=round(float(np.mean([r["da"] for r in regret])), 5) if regret else None,
                history_bc_mean_regret=round(float(np.mean([r["bc"] for r in regret])), 5) if regret else None,
                dagger_cheaper=sum(r["da"] < r["bc"] - 1e-7 for r in regret),
                bc_cheaper=sum(r["bc"] < r["da"] - 1e-7 for r in regret),
                tied=sum(abs(r["da"] - r["bc"]) <= 1e-7 for r in regret)))
    return results, samples


def counterfactual(seed, bank, models, wanted=50, search_limit=2500):
    """Selection must be based on *oracles*, not learned policy outcomes.

    For the same map, opposite wet physics must prescribe different first
    legal navigation actions. Report how often the correct donor and a
    donor-swap model follow the respective laws. No fabricated denominators.
    """
    found = 0
    original = alternate = both = 0
    delta_predictions = 0
    for j in range(search_limit):
        world = prior.maze(seed + j * 173, "ood")
        law = (1, 1, 1)
        opposite = (-1, 1, 1)
        true_game = HistoryGame(world, law)
        cf_game = HistoryGame(world, opposite)
        true_action = true_game.oracle()
        cf_action = cf_game.oracle()
        if true_action is None or cf_action is None or true_action == cf_action:
            continue
        # If first step is painting, compare the first required navigation move.
        if true_action >= 4 or cf_action >= 4:
            continue
        obs, legal = true_game.observe()
        with torch.no_grad():
            sig = torch.from_numpy(signature_for(bank, law, "dagger")[None])
            swapped_sig = torch.from_numpy(signature_for(bank, law, "wrong_wet")[None])
            inp = torch.from_numpy(obs[None])
            mask = torch.from_numpy(legal[None])
            pred = int(models["dagger"](inp, mask, sig).argmax(-1).item())
            swapped = int(models["dagger"](inp, mask, swapped_sig).argmax(-1).item())
        found += 1
        original += int(pred == true_action)
        alternate += int(swapped == cf_action)
        both += int(pred == true_action and swapped == cf_action)
        delta_predictions += int(pred != swapped)
        if found >= wanted:
            break
    return dict(searched=search_limit if found < wanted else j + 1,
                divergent_oracles=found,
                true_law_followed=original,
                swapped_law_followed=alternate,
                both_followed=both,
                receiver_changed_action=delta_predictions)


def gallery(samples, path):
    # Preserve DG016's identical-world SVG map display without changing its source.
    converted = []
    for sample in samples:
        case = dict(sample)
        case["trials"] = {
            "inherited": sample["trials"]["dagger"],
            "wrong_wet": sample["trials"]["wrong_wet"],
            "wrong_goal": sample["trials"]["wrong_goal"],
            "blind": sample["trials"]["blind_dagger"],
            "oracle": sample["trials"]["oracle"],
        }
        converted.append(case)
    prior.svg_gallery(converted, path)
    path.write_text(path.read_text().replace("DG016", "DG017"))


def selftest():
    prior.selftest()
    for law in itertools.product((-1, 1), repeat=3):
        game = HistoryGame(prior.maze(2401, "train"), law)
        obs, legal = game.observe()
        assert obs.shape == (CHANNELS, SIDE, SIDE)
        assert legal.shape == (6,)
        assert np.isfinite(obs).all()
        assert float(obs[-1].max()) > 0
        for _ in range(16):
            action = game.oracle()
            if action is None:
                break
            assert legal[action]
            game.step(action)
            obs, legal = game.observe()
            assert np.isfinite(obs).all()
    for conditioned in (False, True):
        torch.manual_seed(42)
        old = prior.Receiver(conditioned).eval()
        fresh = graft_legacy(old)
        obs = torch.randn(5, prior.CHANNELS, SIDE, SIDE)
        mask = torch.ones(5, 6, dtype=torch.bool)
        with torch.no_grad():
            sig = torch.randn(5, 24) if conditioned else None
            before = old(obs, mask, sig)
            newobs = torch.cat((obs, torch.randn(5, 1, SIDE, SIDE)), 1)
            after = fresh(newobs, mask, sig)
        assert torch.allclose(before, after, atol=1e-5), (conditioned, (before-after).abs().max())
    print("DG017 SELFTEST PASS: host legality, visit history, output-equivalent neural surgery")


def run(seed, out, worlds, updates, sealed, horizon, donor_steps, rounds, correction_worlds):
    torch.set_num_threads(2)
    t0 = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bank, donor_errors = {}, {}
    for kind in ("narrow", "gated"):
        for family in range(3):
            for sign in (-1, 1):
                donor = prior.base.train_donor(
                    seed + 1000 + 100*family + 10*(sign+1) + (500 if kind == "gated" else 0),
                    family, sign, kind == "gated", donor_steps)
                bank[kind, family, sign] = donor
                donor_errors[f"{kind}/{family}/{sign}"] = float(donor["error"])
                torch.save(dict(weights=donor["model"].state_dict(),
                     private_permutation=donor["perm"].tolist(),
                     private_reflection=donor["flip"].tolist(),
                     validation_mse=donor["error"]),
                     out / f"donor-{kind}-{family}-{sign}.pt")
    original_data = prior.train_data(seed + 2000, bank, worlds)
    legacy = prior.fit(seed + 3000, original_data, True, updates)
    legacy_blind = prior.fit(seed + 4000, original_data, False, updates)
    history_data = dataset(seed + 2000, bank, worlds)
    assert len(history_data[3]) == len(original_data[3])
    history_bc = fit(graft_legacy(legacy), history_data, seed + 5000, updates)
    blind_bc = fit(graft_legacy(legacy_blind), history_data, seed + 6000, updates)
    models = dict(legacy=legacy, legacy_blind=legacy_blind,
                  history_bc=history_bc, dagger=copy.deepcopy(history_bc),
                  blind_dagger=copy.deepcopy(blind_bc))
    logs = []
    corrected = {"dagger": None, "blind_dagger": None}
    for round_id in range(rounds):
        for model_name in ("dagger", "blind_dagger"):
            new_data, stats = collect(seed + 1000000 + 10000*round_id
                         + (500000 if model_name == "blind_dagger" else 0),
                         bank, models[model_name], correction_worlds, horizon,
                         teacher_fraction=.20 if round_id == 0 else .08)
            if corrected[model_name] is not None:
                corrected[model_name] = tuple(torch.cat((a, b))
                      for a, b in zip(corrected[model_name], new_data))
            else:
                corrected[model_name] = new_data
            models[model_name] = fit(models[model_name], history_data,
                seed + 7000 + 100*round_id + (1000 if model_name == "blind_dagger" else 0),
                max(100, updates // 2), corrected[model_name])
            logs.append(dict(round=round_id + 1, model=model_name, **stats,
                             cumulative_labels=len(corrected[model_name][3])))
    for name, model in models.items():
        torch.save(model.state_dict(), out / f"receiver-{name}.pt")
    results, samples = evaluate(seed + 100000, bank, models, sealed, horizon)
    cf = counterfactual(seed + 180000, bank, models)
    assert all(row["illegal"] == 0
               for split in results.values() for row in split["trajectories"].values())
    assert all(split["trajectories"]["oracle"]["success"] == sealed
               for split in results.values())
    (out / "trajectories.json").write_text(json.dumps(samples, indent=2) + "\n")
    gallery(samples, out / "gallery.svg")
    report = dict(experiment="DG017", seed=seed, source_sha256=digest(__file__),
        parent_commit="a1ab1ab85d78699368a6af676f9e646edf312bb8",
        pytorch=torch.__version__, numpy=np.__version__, donors=len(bank),
        receiver_models=len(models), original_training_states=len(original_data[3]),
        history_training_states=len(history_data[3]), train_worlds=worlds,
        updates=updates, rounds=rounds, correction_worlds_per_round=correction_worlds,
        correction_log=logs, sealed_worlds_per_split=sealed,
        horizon=horizon, donor_training_steps=donor_steps,
        source_neural_outputs=True, source_hidden_permutation=True,
        input_channels=CHANNELS, visit_history_channel=True,
        initial_legacy_output_equivalence="validated in selftest",
        features="13 visible grid channels including host visit counts; no path distance",
        train_test_seed_overlap=0, withheld_law="+++",
        donor_validation_mse=donor_errors, results=results,
        wet_counterfactual=cf,
        caveats="Synthetic oracle-labeled DAgger; goal/physics rules authored; donor output probes not transplanted KV. No CyberSand source edits or native acceptance. IID/OOD shift topology and gate timing together.",
        duration_seconds=round(time.monotonic()-t0, 2))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    hashes = {p.name: digest(p) for p in out.iterdir()
              if p.is_file() and p.name != "sha256.json"}
    (out / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--seed", type=int, default=26011630)
    parser.add_argument("--out", default="/tmp/dg017")
    parser.add_argument("--train-worlds", type=int, default=200)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--sealed", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--donor-steps", type=int, default=140)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--correction-worlds", type=int, default=65)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.seed, args.out, args.train_worlds, args.updates, args.sealed,
            args.horizon, args.donor_steps, args.rounds, args.correction_worlds)
