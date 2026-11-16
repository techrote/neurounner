"""DG018: sequence-conditioned neural-law inheritance with auxiliary future-cost value.

Research-only synthetic CyberSand-like host-legal game. No CyberSand modifications.
Stacked on DG017: imports the prior sealed world, host action and donor contracts.
Value targets derive from *training* oracle trajectories. Never injected at inference.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.memplant.trajectory_transfer_017 import experiment as prior

SIDE = prior.SIDE
CHANNELS = prior.CHANNELS
HELD = prior.HELD
TRAIN_LAWS = prior.TRAIN_LAWS
MODES = ("value", "bc", "shuffled_value", "blind", "value_reset",
         "wrong_wet", "wrong_goal", "oracle")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SequenceReceiver(nn.Module):
    """Recurrent receiver of canonical game observations and neural donor probes."""
    def __init__(self, conditioned):
        super().__init__()
        self.conditioned = conditioned
        width = CHANNELS * SIDE * SIDE + (24 if conditioned else 0)
        self.encoder = nn.Sequential(nn.Linear(width, 128), nn.ReLU())
        self.memory = nn.GRU(128, 96, batch_first=True)
        self.policy = nn.Linear(96, 6)
        self.value = nn.Linear(96, 1)

    def forward(self, x, legal, sig=None, hidden=None):
        # x[B,T,C,H,W], legal[B,T,6], sig[B,T,24]
        z = x.flatten(2)
        if self.conditioned:
            assert sig is not None
            z = torch.cat((z, sig), dim=-1)
        z, hidden = self.memory(self.encoder(z), hidden)
        logits = self.policy(z).masked_fill(~legal, -1e5)
        return logits, self.value(z).squeeze(-1), hidden


def expert_episodes(seed, bank, worlds):
    """Sequences of legal expert actions, with *remaining observed* expert cost.

    No shortest-path potentials or oracle returns enter model observations.
    A distinct source law is inherited through neural donor probes.
    """
    episodes = []
    for i in range(worlds):
        world = prior.prior.maze(seed + 79 * i, "train")
        for law in TRAIN_LAWS:
            game = prior.HistoryGame(world, law)
            sig = prior.prior.base.signatures(bank, "narrow", law)
            x_list, legal_list, acts, pre_cost = [], [], [], []
            for _ in range(96):
                if game.done:
                    break
                x, legal = game.observe()
                action = game.oracle()
                if action is None:
                    break
                assert legal[action]
                x_list.append(x)
                legal_list.append(legal)
                acts.append(action)
                pre_cost.append(game.travel_cost)
                game.step(action)
            if not game.done or not acts:
                raise RuntimeError("Unfinished training oracle trajectory")
            remaining = (np.float32(game.travel_cost) -
                         np.asarray(pre_cost, dtype=np.float32)) / 40.0
            # Clamp only to reduce outlier dominance, not to inject a path plan.
            remaining = np.minimum(remaining, 8.0).astype(np.float32)
            episodes.append((
                torch.from_numpy(np.stack(x_list)),
                torch.from_numpy(np.stack(legal_list)),
                torch.from_numpy(np.repeat(sig[None], len(acts), axis=0).copy()),
                torch.tensor(acts, dtype=torch.long),
                torch.from_numpy(remaining)))
    return episodes


def minibatch(episodes, rng, batch, window):
    seqs = [[], [], [], [], []]
    valid = []
    for _ in range(batch):
        episode = episodes[int(rng.integers(len(episodes)))]
        n = len(episode[3])
        start = int(rng.integers(n))
        end = min(n, start + window)
        span = end - start
        for k, source in enumerate(episode):
            section = source[start:end]
            if span < window:
                # Mask all padded cells out of both objectives.
                section = torch.cat((section, section[-1:].expand(window-span, *section.shape[1:])))
            seqs[k].append(section)
        valid.append(torch.arange(window) < span)
    return tuple(torch.stack(part) for part in seqs), torch.stack(valid)


def fit(seed, episodes, conditioned, updates, value_mode, window=12):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 33)
    model = SequenceReceiver(conditioned)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.0001)
    model.train()
    trace = []
    for step in range(updates):
        (x, legal, sig, actions, targets), valid = minibatch(
            episodes, rng, batch=32, window=window)
        logits, predicted, _ = model(x, legal, sig if conditioned else None)
        imitation = F.cross_entropy(logits[valid], actions[valid])
        if value_mode == "real":
            value_targets = targets[valid]
            value_loss = F.smooth_l1_loss(predicted[valid], value_targets)
            loss = imitation + 0.20 * value_loss
        elif value_mode == "shuffled":
            # Independent target permutation: same objective scale/compute, wrong world.
            source = targets[valid]
            shuffled = source[torch.randperm(len(source))]
            value_loss = F.smooth_l1_loss(predicted[valid], shuffled)
            loss = imitation + 0.20 * value_loss
        else:
            value_loss = predicted.sum() * 0.0
            loss = imitation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (0, updates-1):
            trace.append(dict(update=step+1, policy_loss=float(imitation.detach()),
                              value_loss=float(value_loss.detach())))
    return model.eval(), trace


def signature(bank, law, mode):
    if mode == "wrong_wet":
        law = (-law[0], law[1], law[2])
    elif mode == "wrong_goal":
        law = (law[0], -law[1], law[2])
    return prior.prior.base.signatures(bank, "gated", law)


def step_policy(game, bank, law, mode, model, hidden):
    obs, legal = game.observe()
    assert bool(legal.any())
    sig = signature(bank, law, mode)
    with torch.no_grad():
        logits, val, next_hidden = model(
            torch.from_numpy(obs[None, None]),
            torch.from_numpy(legal[None, None]),
            torch.from_numpy(sig[None, None]) if model.conditioned else None,
            hidden)
        action = int(logits[0, 0].argmax().item())
    assert legal[action], (mode, action, legal)
    return action, next_hidden, float(val[0, 0])


def play(world, law, mode, bank, models, horizon):
    game = prior.HistoryGame(world, law)
    trajectory = [game.pos]
    actions = []
    predicted_values = []
    hidden = None
    if mode in ("wrong_wet", "wrong_goal", "value_reset"):
        model = models["value"]
    else:
        model = models.get(mode)
    for _ in range(horizon):
        if game.done:
            break
        if mode == "oracle":
            action = game.oracle()
        else:
            if mode == "value_reset":
                hidden = None
            action, hidden, value = step_policy(game, bank, law, mode, model, hidden)
            predicted_values.append(round(value, 3))
        if action is None:
            break
        assert game.legal()[action]
        game.step(action)
        trajectory.append(game.pos)
        actions.append(prior.prior.ACTIONS[action])
    return dict(success=int(game.done), cost=round(game.travel_cost, 4),
                steps=game.steps, closure=int(game.closed), retraces=game.retraces,
                illegal=game.illegal, wet=game.wet_steps,
                path=trajectory, actions=actions, painted=len(game.painted),
                final_predicted_value=predicted_values[-1] if predicted_values else None)


def summarize(rows, horizon):
    result = {}
    for mode, trials in rows.items():
        result[mode] = dict(
            worlds=len(trials), success=sum(t["success"] for t in trials),
            illegal=sum(t["illegal"] for t in trials),
            retraces=sum(t["retraces"] for t in trials),
            failed=sum(not t["success"] for t in trials),
            successful_mean_cost=round(float(np.mean(
                [t["cost"] for t in trials if t["success"]])), 5)
                if any(t["success"] for t in trials) else None,
            failure_penalized_cost=round(float(np.mean(
                [t["cost"] + (0 if t["success"] else 3 * horizon)
                 for t in trials])), 5))
    return result


def eval_sealed(seed, bank, models, count, horizon):
    results, sample = {}, []
    for family in ("iid", "ood"):
        rows = {mode: [] for mode in MODES}
        for i in range(count):
            world_seed = seed + 103*i + (10_000_000 if family == "ood" else 0)
            world = prior.prior.maze(world_seed, family)
            law = HELD if i%2==0 else TRAIN_LAWS[(i//2) % len(TRAIN_LAWS)]
            trials = {mode: play(world, law, mode, bank, models, horizon)
                      for mode in MODES}
            for mode, trial in trials.items():
                rows[mode].append(trial)
            if i < 4:
                # Preserve real, matched-world trajectories (not simulated illustrations).
                sample.append(dict(family=family, seed=world_seed, laws=law,
                    wall=world["wall"].astype(int).tolist(),
                    wet=world["wet"].astype(int).tolist(),
                    gate=world["gate"], close_after=world["close_after"],
                    start=world["start"], goal=world["goal"],
                    alternate=world["alternate"], trials=trials))
        pair = [i for i in range(count) if
                rows["value"][i]["success"] and rows["bc"][i]["success"]
                and rows["oracle"][i]["success"]]
        costs = [(rows["value"][i]["cost"]-rows["oracle"][i]["cost"],
                  rows["bc"][i]["cost"]-rows["oracle"][i]["cost"]) for i in pair]
        results[family] = dict(
            trajectories=summarize(rows, horizon),
            withheld_worlds=(count+1)//2,
            withheld_success={m:sum(v["success"] for j,v in enumerate(rows[m])
                                     if j%2==0) for m in MODES},
            value_vs_recurrent_bc_joint_success=dict(
                worlds=len(pair),
                value_mean_extra_cost=round(float(np.mean([a for a,b in costs])),4) if costs else None,
                bc_mean_extra_cost=round(float(np.mean([b for a,b in costs])),4) if costs else None,
                value_cheaper=sum(a<b-1e-7 for a,b in costs),
                bc_cheaper=sum(b<a-1e-7 for a,b in costs),
                ties=sum(abs(a-b)<=1e-7 for a,b in costs)))
    return results, sample


def counterfactual(seed, bank, models, wanted=50, limit=2500):
    examined, found = 0, 0
    changed, followed_true, followed_swap, both = 0, 0, 0, 0
    for j in range(limit):
        world = prior.prior.maze(seed + j*173, "ood")
        true_law, alt_law = (1, 1, 1), (-1, 1, 1)
        true_game = prior.HistoryGame(world, true_law)
        alt_game = prior.HistoryGame(world, alt_law)
        true_action = true_game.oracle()
        alt_action = alt_game.oracle()
        examined += 1
        if true_action is None or alt_action is None or true_action >= 4 or alt_action >= 4 or true_action == alt_action:
            continue
        a, _, _ = step_policy(true_game, bank, true_law, "value", models["value"], None)
        b, _, _ = step_policy(true_game, bank, true_law, "wrong_wet", models["value"], None)
        changed += int(a != b)
        followed_true += int(a == true_action)
        followed_swap += int(b == alt_action)
        both += int(a == true_action and b == alt_action)
        found += 1
        if found >= wanted:
            break
    return dict(searched=examined, divergent_oracle_cases=found,
                predictions_changed=changed, true_law_followed=followed_true,
                alternate_law_followed=followed_swap, both_followed=both)


def write_gallery(samples, output):
    # Reuse earlier geometry, but display five matched trajectories only.
    converted = []
    for sample in samples:
        case = dict(sample)
        case["trials"] = dict(
            inherited=sample["trials"]["value"],
            wrong_wet=sample["trials"]["wrong_wet"],
            wrong_goal=sample["trials"]["wrong_goal"],
            blind=sample["trials"]["blind"],
            oracle=sample["trials"]["oracle"])
        converted.append(case)
    prior.prior.svg_gallery(converted, output)
    output.write_text(output.read_text().replace("DG016", "DG018"))


def selftest():
    # Validate host legality, finite sensor observations, stateful interface,
    # no privileged future-cost input and gradient flow into shared recurrent state.
    world = prior.prior.maze(12343, "train")
    game = prior.HistoryGame(world, (1, -1, 1))
    obs, legal = game.observe()
    assert obs.shape == (CHANNELS, SIDE, SIDE) and legal.shape == (6,)
    for conditioned in (True, False):
        model = SequenceReceiver(conditioned)
        sig = torch.randn(3, 7, 24) if conditioned else None
        logits, values, hidden = model(torch.randn(3, 7, CHANNELS, SIDE, SIDE),
            torch.ones(3, 7, 6, dtype=torch.bool), sig)
        assert logits.shape == (3, 7, 6) and values.shape == (3, 7)
        assert hidden.shape == (1, 3, 96)
        (values.square().mean() + logits.square().mean()).backward()
        assert model.memory.weight_ih_l0.grad is not None
    for law in TRAIN_LAWS:
        g = prior.HistoryGame(world, law)
        for _ in range(96):
            if g.done:
                break
            _, mask = g.observe()
            a = g.oracle()
            assert mask[a]
            g.step(a)
        assert g.done
    print("DG018 SELFTEST PASS: legal sequence actions, recurrent shapes, value gradients, sealed oracle")


def run(seed, out, train_worlds, updates, sealed, horizon, donor_steps):
    torch.set_num_threads(2)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    bank, donor_errors = {}, {}
    for kind in ("narrow", "gated"):
        for family in range(3):
            for sign in (-1, 1):
                donor = prior.prior.base.train_donor(
                    seed + 1000 + 100*family + 10*(sign+1)
                    + (500 if kind == "gated" else 0),
                    family, sign, kind == "gated", donor_steps)
                bank[kind, family, sign] = donor
                donor_errors[f"{kind}/{family}/{sign}"] = float(donor["error"])
                torch.save(dict(weights=donor["model"].state_dict(),
                    private_permutation=donor["perm"].tolist(),
                    private_reflection=donor["flip"].tolist(),
                    validation_mse=donor["error"]),
                    out/f"donor-{kind}-{family}-{sign}.pt")
    episodes = expert_episodes(seed+2000, bank, train_worlds)
    models, losses = {}, {}
    for index, (name, conditioned, mode) in enumerate((
            ("bc", True, "none"),
            ("value", True, "real"),
            ("shuffled_value", True, "shuffled"),
            ("blind", False, "none"))):
        models[name], losses[name] = fit(seed + (3000 if conditioned else 6000),
                        episodes, conditioned, updates, mode)
        torch.save(models[name].state_dict(), out/f"receiver-{name}.pt")
    results, traces = eval_sealed(seed+100000, bank, models, sealed, horizon)
    cf = counterfactual(seed+180000, bank, models)
    assert all(r["illegal"] == 0 for split in results.values()
               for r in split["trajectories"].values())
    assert all(split["trajectories"]["oracle"]["success"] == sealed
               for split in results.values())
    (out/"trajectories.json").write_text(json.dumps(traces, indent=2)+"\n")
    write_gallery(traces, out/"gallery.svg")
    report = dict(
        experiment="DG018", protocol="sequence-value-v1", seed=seed,
        source_sha256=sha256(__file__), parent_sha="2686158d40a01730509e1a2d2a78ea374d02fdc7",
        pytorch=torch.__version__, numpy=np.__version__,
        train_worlds=train_worlds, training_episodes=len(episodes),
        training_decisions=sum(len(ep[3]) for ep in episodes),
        train_updates_per_receiver=updates, donors=len(bank), receivers=len(models),
        donor_validation_mse=donor_errors, loss_trace=losses,
        sequence_window=12, recurrent_hidden=96,
        supervised_value_target="remaining true weighted cost on TRAINING oracle trajectory, scaled by 40",
        value_auxiliary_coefficient=0.20,
        matched_optim_budget=True, matched_initialization=True,
        matched_training_batches_conditioned=True, shuffled_value_control=True,
        inference_uses_oracle_potential=False, inference_uses_future_cost=False,
        host_legal_actions=True, source_neural_outputs=True,
        private_donor_features=True, train_sealed_seed_overlap=0,
        withheld_law="+++", horizon=horizon, sealed_per_split=sealed,
        counterfactual=cf, results=results,
        caveats=("Synthetic oracle-trained specialists and path labels. Recurrent value branch is a "
                 "shared-representation auxiliary objective, not a standalone planner. "
                 "OOD changes topology and gate timing together; no native CyberSand acceptance; "
                 "not transferable KV cache or spontaneous law discovery."),
        seconds=round(time.monotonic()-started,2))
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    hashes = {p.name: sha256(p) for p in sorted(out.iterdir())
              if p.is_file() and p.name != "sha256.json"}
    (out/"sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--seed", type=int, default=26011632)
    parser.add_argument("--out", default="/tmp/dg018")
    parser.add_argument("--train-worlds", type=int, default=200)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--sealed", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--donor-steps", type=int, default=140)
    a = parser.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.seed, a.out, a.train_worlds, a.updates, a.sealed, a.horizon, a.donor_steps)
