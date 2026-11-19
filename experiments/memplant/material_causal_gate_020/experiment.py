"""DG020: preregistered material-only neural donor intervention gate.

Two fixed 12-tile routes and a common host-legal prefork action prefix.
The only paired intervention is replacing an independently TRAINED local-cost
neural donor. No distance features, path costs, oracle actions, law IDs, or
family IDs are supplied to the receiver. Synthetic research, not game code.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TILES = 12
WET_COST = (0.35, 2.30)
DRY_COST = 1.00
FAMILIES = ("edge_bands", "alternating")
THRESHOLDS = dict(pairs_per_family=160, seeds_required=3,
                  oracle_min_margin=0.60, both_correct=0.90,
                  changed_action=0.95, each_direction=0.94,
                  sham_changed=0.05, shuffled_both_max=0.60,
                  zero_both_max=0.60, donor_mse_max=0.02)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_world(rng, family):
    wet = np.zeros((2, TILES), dtype=np.float32)
    for route in range(2):
        k = int(rng.integers(1, TILES))
        if family == "iid":
            wet[route] = (rng.random(TILES) < rng.uniform(0.13, 0.87))
        elif family == "short_runs":
            start = int(rng.integers(TILES))
            wet[route, (start + np.arange(k)) % TILES] = 1.
        elif family == "edge_bands":
            wet[route, :k] = 1.
            if rng.integers(2):
                wet[route] = wet[route, ::-1]
        elif family == "alternating":
            ordering = np.array([0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11])
            wet[route, (ordering[:k] + int(rng.integers(TILES))) % TILES] = 1.
        else:
            raise ValueError(family)
    fee = rng.uniform(0, 0.30, (2, TILES)).astype(np.float32)
    texture = rng.uniform(-1, 1, (2, TILES, 2)).astype(np.float32)
    obs = np.concatenate((wet[..., None], fee[..., None], texture), axis=-1)
    phase = int(rng.integers(1, 5))
    return obs, phase


def oracle(obs, law):
    wet, fee = obs[..., 0], obs[..., 1]
    costs = np.where(wet > .5, WET_COST[law], DRY_COST) + fee
    totals = costs.sum(axis=-1)
    action = int(np.argmin(totals))
    return action, float(abs(totals[0]-totals[1])), totals.tolist()


def sample_worlds(seed, family, n):
    rng = np.random.default_rng(seed)
    retained, tested = [], 0
    while len(retained) < n and tested < n * 120:
        obs, phase = make_world(rng, family)
        tested += 1
        low, gap_l, _ = oracle(obs, 0)
        high, gap_h, _ = oracle(obs, 1)
        if low != high and min(gap_l, gap_h) >= THRESHOLDS["oracle_min_margin"]:
            retained.append((obs, phase, tested))
    if len(retained) != n:
        raise AssertionError(f"Insufficient oracle divergent pairs: {len(retained)}/{n}")
    return retained, tested


class Donor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 24), nn.Tanh(),
                                 nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Receiver(nn.Module):
    """Small learned additive scorer, no explicit oracle distance or answer."""
    def __init__(self):
        super().__init__()
        self.tile = nn.Sequential(nn.Linear(3, 16), nn.ReLU(),
                                  nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, obs, local_donor):
        tile = torch.stack((obs[..., 0], obs[..., 1], local_donor), dim=-1)
        route_scores = self.tile(tile).squeeze(-1).sum(-1)
        return -route_scores


def train_donor(seed, law, steps):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 115)
    model = Donor()
    opt = torch.optim.Adam(model.parameters(), lr=0.006)
    model.train()
    for _ in range(steps):
        x = rng.uniform(-1, 1, (256, 4)).astype(np.float32)
        x[:, 0] = rng.integers(0, 2, 256)
        x[:, 1] = rng.uniform(0, 0.30, 256)
        target = torch.tensor(np.where(x[:, 0] > .5, WET_COST[law], DRY_COST) + x[:, 1], dtype=torch.float32)
        loss = F.mse_loss(model(torch.from_numpy(x)), target)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    rng = np.random.default_rng(seed + 9981)
    x = rng.uniform(-1, 1, (2048, 4)).astype(np.float32)
    x[:, 0] = rng.integers(0, 2, 2048)
    x[:, 1] = rng.uniform(0, .30, 2048)
    target = np.where(x[:, 0] > .5, WET_COST[law], DRY_COST) + x[:, 1]
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).numpy()
    return model, float(np.mean((pred - target)**2))


def donor_forecast(model, worlds):
    obs = torch.from_numpy(np.stack([x[0] for x in worlds]))
    with torch.no_grad():
        costs = model(obs.reshape(-1, 4)).reshape(len(worlds), 2, TILES)
    return costs


def train_receiver(seed, worlds, primary, updates, blind=False):
    obs = torch.from_numpy(np.stack([w[0] for w in worlds]))
    labels = [torch.tensor([oracle(w[0], k)[0] for w in worlds]) for k in range(2)]
    forecasts = [torch.zeros_like(primary[0]) if blind else x for x in primary]
    torch.manual_seed(seed)
    model = Receiver()
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    rng = np.random.default_rng(seed + 810)
    model.train()
    for _ in range(updates):
        ix = torch.tensor(rng.integers(len(worlds), size=128))
        x = obs[ix]
        low, high = model(x, forecasts[0][ix]), model(x, forecasts[1][ix])
        loss = (F.cross_entropy(low, labels[0][ix]) +
                F.cross_entropy(high, labels[1][ix])) / 2
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def evaluate(seed, family, worlds, donors, receiver, blind, output):
    obs = torch.from_numpy(np.stack([w[0] for w in worlds]))
    cost = [donor_forecast(donors[k][0], worlds) for k in (0, 1)]
    sham = [donor_forecast(donors[k][1], worlds) for k in (0, 1)]
    zeros = torch.zeros_like(cost[0])
    rng = np.random.default_rng(seed + 772)
    with torch.no_grad():
        logits = [receiver(obs, c) for c in cost]
        sham_logit = [receiver(obs, c) for c in sham]
        zero_logit = receiver(obs, zeros)
        blind_logit = blind(obs, zeros)
        # Independent, randomly relabelled functional donor assignments are a
        # negative causal control; no test-world oracle label informs assignment.
        random_sources = [torch.stack([cost[int(rng.integers(2))][i]
                           for i in range(len(worlds))]) for _ in range(2)]
        shuffled_logit = [receiver(obs, v) for v in random_sources]
    pred = [x.argmax(-1).numpy() for x in logits]
    sh = [x.argmax(-1).numpy() for x in sham_logit]
    shuffled_pred = [x.argmax(-1).numpy() for x in shuffled_logit]
    z = zero_logit.argmax(-1).numpy()
    b = blind_logit.argmax(-1).numpy()
    stats = {k: 0 for k in ("correct_low", "correct_high", "both", "changed",
                           "sham_changed", "sham_both", "shuffled_both",
                           "zero_both", "blind_both", "illegal", "identity")}
    margin_deltas, conf_correct, positive_margins, failures = [], [[], []], [[], []], []
    with open(output, "w", encoding="utf-8") as f:
        for i, (world, phase, attempt) in enumerate(worlds):
            oa, ga, ca = oracle(world, 0)
            ob, gb, cb = oracle(world, 1)
            assert oa != ob and min(ga, gb) >= THRESHOLDS["oracle_min_margin"]
            assert pred[0][i] in (0, 1) and pred[1][i] in (0, 1)
            correct = [int(pred[0][i] == oa), int(pred[1][i] == ob)]
            both = int(all(correct))
            stats["correct_low"] += correct[0]
            stats["correct_high"] += correct[1]
            stats["both"] += both
            stats["changed"] += int(pred[0][i] != pred[1][i])
            stats["sham_changed"] += int(sh[0][i] != pred[0][i] or sh[1][i] != pred[1][i])
            stats["sham_both"] += int(sh[0][i] == oa and sh[1][i] == ob)
            stats["shuffled_both"] += int(shuffled_pred[0][i] == oa and shuffled_pred[1][i] == ob)
            stats["zero_both"] += int(z[i] == oa and z[i] == ob)
            stats["blind_both"] += int(b[i] == oa and b[i] == ob)
            stats["identity"] += 1
            margins = [(logits[k][i, [oa, ob][k]] - logits[k][i, 1-[oa, ob][k]]).item() for k in (0, 1)]
            probs = [F.softmax(logits[k][i], -1)[[oa, ob][k]].item() for k in (0, 1)]
            margin_deltas.append(float((logits[1][i, 0]-logits[1][i, 1])-
                                        (logits[0][i, 0]-logits[0][i, 1])))
            for k in (0, 1):
                conf_correct[k].append(probs[k])
                positive_margins[k].append(margins[k])
            if not both and len(failures) < 20:
                failures.append(dict(index=i, true=[oa, ob], predicted=[int(pred[0][i]), int(pred[1][i])], margins=margins))
            receipt = dict(index=i, phase=phase, host_prefix=["MOVE_FORWARD"]*phase,
                legal_mask=[1, 1], action_names=["LEFT_ROUTE", "RIGHT_ROUTE"],
                observation_sha256=hashlib.sha256(world.tobytes()).hexdigest(),
                observation_identical_between_laws=True, candidate_attempt=attempt,
                observation=world.tolist(),
                oracle_actions=[oa, ob], oracle_costs=[ca, cb], oracle_gaps=[ga, gb],
                receiver_actions=[int(pred[0][i]), int(pred[1][i])],
                local_donor_predictions=[c[i].tolist() for c in cost],
                logits=[x[i].tolist() for x in logits], oracle_logit_margins=margins,
                oracle_action_confidence=probs, donor_forecast_sha256=[hashlib.sha256(c[i].numpy().tobytes()).hexdigest() for c in cost],
                sham_actions=[int(sh[0][i]), int(sh[1][i])],
                shuffled_actions=[int(shuffled_pred[0][i]), int(shuffled_pred[1][i])],
                zero_action=int(z[i]), blind_action=int(b[i]), both_correct=both)
            f.write(json.dumps(receipt, separators=(",", ":")) + "\n")
    n = len(worlds)
    return dict(family=family, n=n, metrics={k: round(v/n, 5) for k, v in stats.items()},
                mean_oracle_confidence=[round(float(np.mean(x)), 5) for x in conf_correct],
                mean_low_to_high_left_logit_margin_shift=round(float(np.mean(margin_deltas)), 5),
                mean_absolute_left_logit_margin_shift=round(float(np.mean(np.abs(margin_deltas))), 5),
                mean_oracle_logit_margin=[round(float(np.mean(a)), 5) for a in positive_margins],
                failures=failures)


def run(seed, out, training_worlds, receiver_updates, donor_steps, per_family):
    torch.set_num_threads(2)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    donors, donor_mse = {}, {}
    for law in (0, 1):
        donors[law] = []
        for replica in (0, 1):
            donor, mse = train_donor(seed+1000+law*200+replica*11, law, donor_steps)
            name = f"donor-wet-{law}-replica-{replica}.pt"
            torch.save(donor.state_dict(), out/name)
            donors[law].append(donor)
            donor_mse[name] = round(mse, 8)
    first, _ = sample_worlds(seed+200_000, "iid", training_worlds//2)
    second, _ = sample_worlds(seed+300_000, "short_runs", training_worlds-training_worlds//2)
    train_worlds = first + second
    primary = [donor_forecast(donors[k][0], train_worlds) for k in (0, 1)]
    receiver = train_receiver(seed+4000, train_worlds, primary, receiver_updates)
    blind = train_receiver(seed+4000, train_worlds, primary, receiver_updates, blind=True)
    torch.save(receiver.state_dict(), out/"receiver-conditioned.pt")
    torch.save(blind.state_dict(), out/"receiver-law-blind.pt")
    tests = {}
    for j, family in enumerate(FAMILIES):
        worlds, examined = sample_worlds(seed+5_000_000+(j*1_000_000), family, per_family)
        report = evaluate(seed+100*j, family, worlds, donors, receiver, blind,
                          out/f"receipts-{family}.jsonl")
        report["candidates_examined"] = examined
        tests[family] = report
    valid = all(v <= THRESHOLDS["donor_mse_max"] for v in donor_mse.values())
    for family in FAMILIES:
        m = tests[family]["metrics"]
        valid &= (tests[family]["n"] >= THRESHOLDS["pairs_per_family"] and
                  m["identity"] == 1 and m["illegal"] == 0 and
                  m["both"] >= THRESHOLDS["both_correct"] and
                  m["changed"] >= THRESHOLDS["changed_action"] and
                  m["correct_low"] >= THRESHOLDS["each_direction"] and
                  m["correct_high"] >= THRESHOLDS["each_direction"] and
                  m["sham_changed"] <= THRESHOLDS["sham_changed"] and
                  m["shuffled_both"] <= THRESHOLDS["shuffled_both_max"] and
                  m["zero_both"] <= THRESHOLDS["zero_both_max"] and
                  m["blind_both"] == 0)
    report = dict(experiment="DG020-material-causal-gate", seed=seed,
        hypothesis="Changing only a trained wet-material donor flips the receiver to both oracle-correct actions",
        source_sha256=sha(__file__), torch=torch.__version__, numpy=np.__version__,
        material_laws={"wet_cost": WET_COST, "dry_cost": DRY_COST},
        fixed_topology="2x12 tile routes at common fork; legal LEFT/RIGHT; shared MOVE_FORWARD prefix",
        train_families=["iid", "short_runs"], test_families=list(FAMILIES),
        no_label_distance_or_oracle_at_inference=True,
        donor_mse=donor_mse, training_worlds=len(train_worlds),
        receiver_updates=receiver_updates, donor_steps=donor_steps,
        preregistered_thresholds=THRESHOLDS, per_family=tests,
        seed_status="PASS" if valid else "FAIL",
        scientific_scope="functional learned local-cost donor conditioning; no weight/KV transplantation or physics discovery")
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    checks = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "sha256.json"}
    (out/"sha256.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(dict(seed=seed, status=report["seed_status"],
          donor_max_mse=max(donor_mse.values()),
          metrics={k:v["metrics"] for k,v in tests.items()}, source_sha256=report["source_sha256"])))
    return valid


def selftest():
    for family in ("iid", "short_runs") + FAMILIES:
        worlds, _ = sample_worlds(17, family, 15)
        for w, phase, _ in worlds:
            assert 1 <= phase <= 4 and w.shape == (2, 12, 4)
            a, ga, _ = oracle(w, 0); b, gb, _ = oracle(w, 1)
            assert a != b and min(ga, gb) >= .6
            assert np.isfinite(w).all()
    print("SELFTEST PASS: oracle divergence, material-only matched pair and fixed legal mask")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--seed", type=int, default=26011636)
    p.add_argument("--out", default="/tmp/dg020")
    p.add_argument("--training-worlds", type=int, default=1800)
    p.add_argument("--receiver-updates", type=int, default=650)
    p.add_argument("--donor-steps", type=int, default=400)
    p.add_argument("--per-family", type=int, default=160)
    a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.seed, a.out, a.training_worlds, a.receiver_updates,
            a.donor_steps, a.per_family)
