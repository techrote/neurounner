"""DG015: dynamic obstruction and material regret without supplied graph distances.
Research-only synthetic world. Independently trained neural donors, not KV grafts.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.memplant.trajectory_transfer_014 import experiment as dg014

SIDE = 9
N_ACTIONS = 6
CHANNELS = 11
HELD = (1, 1, 1)
TRAIN_LAWS = [x for x in itertools.product((-1, 1), repeat=3) if x != HELD]
MODES = ("inherited", "wrong_wet", "blind", "oracle")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_world(seed):
    """Reachable wet shortcut, two dry detours and a late-closing central gate."""
    r = np.random.default_rng(seed)
    wall = np.ones((SIDE, SIDE), dtype=bool)
    for y in (2, 4, 6):
        wall[y, 1:8] = False
    wall[2:7, 1] = wall[2:7, 7] = False
    wet = np.zeros_like(wall)
    wet[4, 2:7] = True
    wet[2, 2:7] = r.random(5) < .1
    wet[6, 2:7] = r.random(5) < .1
    protected = (~wall) & (r.random((SIDE, SIDE)) < .18)
    markers = np.zeros((SIDE, SIDE), dtype=np.int8)
    markers[4, 1] = 1
    markers[4, 7] = 2
    markers[6, 7] = 3
    markers[4, 4] = 4
    k = int(r.integers(4))
    wall, wet, protected, markers = [np.rot90(a, k) for a in (wall, wet, protected, markers)]
    if int(r.integers(2)):
        wall, wet, protected, markers = [np.fliplr(a) for a in (wall, wet, protected, markers)]
    out = dict(wall=wall.copy(), wet=wet.copy(), protected=protected.copy())
    for id_, name in ((1, "start"), (2, "goal"), (3, "alternate"), (4, "gate")):
        y, x = np.argwhere(markers == id_)[0]
        out[name] = (int(x), int(y))
        protected[y, x] = False
    return out


class Game(dg014.Game):
    def __init__(self, world, laws):
        w = dict(world)
        w["wall"] = world["wall"].copy()
        super().__init__(w, laws)
        self.moves = 0
        self.closed = False
        self.retraces = 0

    def observe(self):
        """True map channels, no obstacle-distance, goal potential or oracle signal."""
        w = self.world
        z = np.zeros((CHANNELS, SIDE, SIDE), dtype=np.float32)
        z[0] = w["wall"]
        z[1] = w["wet"]
        z[2] = w["protected"]
        previous = self.history[-2] if len(self.history) > 1 else self.pos
        for channel, position in ((3, self.pos), (4, w["goal"]),
                                  (5, w["alternate"]), (6, w["start"]),
                                  (7, previous), (9, w["gate"])):
            z[channel, position[1], position[0]] = 1.
        for x, y in self.painted:
            z[8, y, x] = 1.
        z[10, :, :] = float(self.closed)
        return z, self.legal()

    def oracle(self):
        a = super().oracle()
        if a is not None and a < 4 and self.closed and len(self.history) > 1:
            dx, dy = dg014.DIRS[a]
            if (self.pos[0] + dx, self.pos[1] + dy) == self.history[-2]:
                return 5  # host-legal RETRACE can replace returning one step
        return a

    def step(self, action):
        self.moves += int(action < 4)
        self.retraces += int(action == 5)
        super().step(action)
        if self.moves >= 2 and not self.closed:
            x, y = self.world["gate"]
            if self.pos != (x, y):
                self.world["wall"][y, x] = True
                self.closed = True


class Receiver(nn.Module):
    def __init__(self, conditioned):
        super().__init__()
        self.conditioned = conditioned
        size = CHANNELS * SIDE * SIDE + (24 if conditioned else 0)
        self.net = nn.Sequential(nn.Linear(size, 192), nn.ReLU(),
                                 nn.Linear(192, 128), nn.ReLU(),
                                 nn.Linear(128, N_ACTIONS))
    def forward(self, x, legal, signature=None):
        flat = x.flatten(1)
        if self.conditioned:
            flat = torch.cat((flat, signature), dim=1)
        return self.net(flat).masked_fill(~legal, -1e5)


def samples(seed, bank, worlds):
    x, masks, sigs, labels = [], [], [], []
    for i in range(worlds):
        w = make_world(seed + 53 * i)
        for law in TRAIN_LAWS:
            g = Game(w, law)
            s = dg014.signatures(bank, "narrow", law)
            for _ in range(48):
                ob, legal = g.observe()
                action = g.oracle()
                if g.done or action is None:
                    break
                assert legal[action]
                x.append(ob)
                masks.append(legal)
                sigs.append(s)
                labels.append(action)
                g.step(action)
    return (torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(masks)),
            torch.from_numpy(np.stack(sigs)), torch.tensor(labels, dtype=torch.long))


def train_receiver(seed, data, conditioned, updates):
    torch.manual_seed(seed)
    model = Receiver(conditioned)
    opt = torch.optim.AdamW(model.parameters(), lr=.0018)
    x, legal, sigs, y = data
    rng = np.random.default_rng(seed + 1)
    for _ in range(updates):
        ix = torch.tensor(rng.integers(len(y), size=192))
        pred = model(x[ix], legal[ix], sigs[ix] if conditioned else None)
        loss = F.cross_entropy(pred, y[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def play(world, law, mode, bank, models, horizon):
    game = Game(world, law)
    is_donor = mode not in ("blind", "oracle")
    donor_law = (-law[0], law[1], law[2]) if mode == "wrong_wet" else law
    signature = dg014.signatures(bank, "gated", donor_law)
    points, actions = [game.pos], []
    for _ in range(horizon):
        observation, legal = game.observe()
        if mode == "oracle":
            a = game.oracle()
        else:
            policy = models["inherited" if is_donor else "blind"]
            with torch.no_grad():
                a = int(policy(torch.from_numpy(observation[None]),
                               torch.from_numpy(legal[None]),
                               torch.from_numpy(signature[None]) if is_donor else None).argmax(-1)[0])
        if a is None:
            break
        assert legal[a], (mode, a)
        game.step(a)
        actions.append(dg014.ACTIONS[a])
        points.append(game.pos)
        if game.done:
            break
    return dict(success=int(game.done), steps=game.steps,
                cost=round(game.travel_cost, 4), wet=game.wet_steps,
                retraces=game.retraces, illegal=game.illegal,
                path=points, actions=actions)


def evaluate(seed, bank, models, n, horizon):
    rows = {m: [] for m in MODES}
    galleries = []
    swapped_first_action = 0
    for i in range(n):
        w = make_world(seed + i * 101)
        law = HELD if i % 2 == 0 else TRAIN_LAWS[i % len(TRAIN_LAWS)]
        trials = {m: play(w, law, m, bank, models, horizon) for m in MODES}
        if trials["inherited"]["actions"] and trials["wrong_wet"]["actions"]:
            swapped_first_action += (trials["inherited"]["actions"][0] !=
                                     trials["wrong_wet"]["actions"][0])
        for mode in MODES:
            v = trials[mode]
            v["regret"] = (round(v["cost"] / max(1, trials["oracle"]["cost"]) - 1, 4)
                           if v["success"] and trials["oracle"]["success"] else None)
            rows[mode].append(v)
        if i < 4:
            galleries.append(dict(seed=seed+i*101, laws=law,
                                  wall=w["wall"].astype(int).tolist(),
                                  wet=w["wet"].astype(int).tolist(),
                                  gate=w["gate"], trials=trials))
    summary = {}
    for mode, vals in rows.items():
        regrets = [v["regret"] for v in vals if v["regret"] is not None]
        summary[mode] = dict(worlds=n, success=sum(v["success"] for v in vals),
                             withheld_success=sum(v["success"] for j, v in enumerate(vals) if j%2==0),
                             mean_cost=round(float(np.mean([v["cost"] for v in vals])), 4),
                             matched_success=len(regrets),
                             successful_mean_regret=round(float(np.mean(regrets)), 4) if regrets else None,
                             retraces=sum(v["retraces"] for v in vals),
                             illegal=sum(v["illegal"] for v in vals))
    return summary, swapped_first_action, galleries


def gallery_svg(worlds, path):
    colors = {"inherited": "#3273c5", "wrong_wet": "#bf5555",
              "blind": "#ae8b29", "oracle": "#303030"}
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="905" height="850" viewBox="0 0 905 850">',
           '<rect width="905" height="850" fill="white"/>',
           '<text x="15" y="23" font-size="17" font-family="sans-serif">DG015 · held-out donor-conditioned material routing</text>']
    for i, w in enumerate(worlds):
        for j, mode in enumerate(MODES):
            ox, oy = 18+220*j, 35+201*i
            v = w["trials"][mode]
            out.append(f'<text x="{ox}" y="{oy+13}" font-family="sans-serif" font-size="11">{mode} · {w["laws"]}</text>')
            for y in range(SIDE):
                for x in range(SIDE):
                    fill = ("#37414b" if w["wall"][y][x] else
                            "#a5dae6" if w["wet"][y][x] else "#f5f5f5")
                    out.append(f'<rect x="{ox+18*x}" y="{oy+21+18*y}" width="17" height="17" fill="{fill}" stroke="#ddd"/>')
            x, y = w["gate"]
            out.append(f'<rect x="{ox+18*x+3}" y="{oy+21+18*y+3}" width="11" height="11" stroke="#b72929" stroke-width="2" fill="none"/>')
            points = " ".join(f"{ox+18*x+8},{oy+21+18*y+8}" for x,y in v["path"])
            out.append(f'<polyline points="{points}" fill="none" stroke="{colors[mode]}" stroke-opacity=".9" stroke-width="2.5"/>')
            out.append(f'<text x="{ox}" y="{oy+197}" font-size="10" font-family="sans-serif">done {v["success"]} cost {v["cost"]} retrace {v["retraces"]}</text>')
    path.write_text("\n".join(out+["</svg>"])+"\n")


def selftest():
    for law in itertools.product((-1,1), repeat=3):
        for seed in range(8):
            g = Game(make_world(seed), law)
            for _ in range(64):
                x, legal = g.observe()
                assert x.shape == (CHANNELS, SIDE, SIDE)
                assert np.isfinite(x).all()
                a = g.oracle()
                if a is None:
                    break
                assert legal[a]
                g.step(a)
                if g.done:
                    break
            assert g.done and g.closed, (law, seed)
    print("DG015 SELFTEST PASS: 64 complete oracle worlds; legality and gate verified")


def run(seed, out, worlds, updates, sealed, horizon, donor_steps):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    t = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bank, errors = {}, {}
    for arch in ("narrow", "gated"):
        for family in range(3):
            for sign in (-1, 1):
                donor = dg014.train_donor(
                    seed+1000+100*family+10*(sign+1)+(500 if arch=="gated" else 0),
                    family, sign, arch=="gated", donor_steps)
                bank[arch, family, sign] = donor
                errors[f"{arch}/{family}/{sign}"] = round(donor["error"], 6)
                torch.save(dict(weights=donor["model"].state_dict(),
                                private_permutation=donor["perm"].tolist(),
                                private_reflection=donor["flip"].tolist(),
                                mse=donor["error"]),
                           out/f"donor-{arch}-{family}-{sign}.pt")
    data = samples(seed+2000, bank, worlds)
    models = {"inherited": train_receiver(seed+3000, data, True, updates),
              "blind": train_receiver(seed+4000, data, False, updates)}
    for name, model in models.items():
        torch.save(model.state_dict(), out/f"receiver-{name}.pt")
    train_keys = {seed+2000+53*i for i in range(worlds)}
    sealed_keys = {seed+100000+101*i for i in range(sealed)}
    assert not train_keys & sealed_keys
    summary, first_changes, trajectory = evaluate(seed+100000, bank, models, sealed, horizon)
    assert all(v["illegal"]==0 for v in summary.values())
    (out/"trajectories.json").write_text(json.dumps(trajectory, indent=2)+"\n")
    gallery_svg(trajectory, out/"gallery.svg")
    report = dict(experiment="DG015", seed=seed, source_sha256=sha(__file__),
                  parent_commit="bc95b17fe51f21b2de8b00bee183bf14e0fc74e3",
                  pytorch=torch.__version__, train_worlds=worlds, training_states=len(data[3]),
                  updates=updates, donor_training_steps=donor_steps, sealed_worlds=sealed,
                  horizon=horizon, no_graph_distance_features=True,
                  dynamic_gate=True, trained_retrace=True, withheld_triple="+++",
                  donor_validation_mse=errors, trajectories=summary,
                  first_action_changed_by_wet_donor=first_changes,
                  matched_first_action_comparisons=sealed,
                  train_sealed_seed_overlap=0,
                  limitations="Synthetic authored corridor family, Dijkstra teacher, source outputs not KV or learned unknown physics; success-conditioned regret can select survivors.",
                  duration_seconds=round(time.monotonic()-t, 2))
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    hashes = {p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="sha256.json"}
    (out/"sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--seed",type=int,default=26011626)
    p.add_argument("--out",default="/tmp/dg015")
    p.add_argument("--train-worlds",type=int,default=500)
    p.add_argument("--updates",type=int,default=850)
    p.add_argument("--sealed",type=int,default=100)
    p.add_argument("--horizon",type=int,default=64)
    p.add_argument("--donor-steps",type=int,default=140)
    a=p.parse_args()
    if a.selftest:selftest()
    else:run(a.seed,a.out,a.train_worlds,a.updates,a.sealed,a.horizon,a.donor_steps)
