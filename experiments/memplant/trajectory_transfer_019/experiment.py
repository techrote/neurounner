"""DG019: matched wet-law intervention training and sealed counterfactual tests.

Synthetic CyberSand-like host-legal research only. This deliberately tests causal
policy dependence on independently trained donor *outputs*, not weight/KV grafts.
Stacked on DG018; no CyberSand game or production provider modifications.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.memplant.trajectory_transfer_017 import experiment as prior

LAWS = prior.TRAIN_LAWS
HELD = prior.HELD
MODES = ("paired", "bc", "shuffled", "blind", "wrong_wet", "wrong_goal", "oracle")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def donor_signature(bank, law, architecture):
    return prior.prior.base.signatures(bank, architecture, law)


def build_donors(seed, steps, out):
    bank, validation = {}, {}
    for architecture in ("narrow", "gated"):
        for family in range(3):
            for sign in (-1, 1):
                donor = prior.prior.base.train_donor(
                    seed + 1000 + 100*family + 10*(sign+1)
                    + (500 if architecture == "gated" else 0),
                    family, sign, architecture == "gated", steps)
                bank[architecture, family, sign] = donor
                name = f"donor-{architecture}-{family}-{sign}.pt"
                torch.save(dict(weights=donor["model"].state_dict(),
                                private_permutation=donor["perm"].tolist(),
                                private_reflection=donor["flip"].tolist(),
                                validation_mse=float(donor["error"])), out/name)
                validation[name] = float(donor["error"])
    return bank, validation


def paired_samples(seed, bank, target, limit, family="train"):
    """Select *only* oracle-divergent, SAME-OBSERVATION wet-law pairs.

    The choice of examples is based exclusively on an independent oracle,
    before training any receiver. Neither target action nor oracle value is
    ever present in inference observations. The held +++ triple is excluded.
    """
    x, masks, signatures, labels = [], [], [], []
    selected, searched, after_gate = 0, 0, 0
    for i in range(limit):
        world = prior.prior.maze(seed + i * 173, family)
        searched += 1
        for goal, paint in ((-1, -1), (-1, 1), (1, -1)):
            a_law, b_law = (1, goal, paint), (-1, goal, paint)
            assert a_law in LAWS and b_law in LAWS
            a = prior.HistoryGame(world, a_law)
            b = prior.HistoryGame(world, b_law)
            for step in range(6):
                if a.done or b.done:
                    break
                xa, ma = a.observe()
                xb, mb = b.observe()
                assert np.array_equal(xa, xb) and np.array_equal(ma, mb)
                aa, ab = a.oracle(), b.oracle()
                if aa is None or ab is None:
                    break
                if aa != ab and aa < 4 and ab < 4 and ma[aa] and mb[ab]:
                    x.extend((xa.copy(), xb.copy()))
                    masks.extend((ma.copy(), mb.copy()))
                    signatures.extend((donor_signature(bank, a_law, "narrow"),
                                       donor_signature(bank, b_law, "narrow")))
                    labels.extend((aa, ab))
                    selected += 1
                    after_gate += int(a.closed)
                    if selected >= target:
                        break
                # Identical legal action histories preserve state correspondence.
                common = aa
                if not mb[common]:
                    break
                a.step(common)
                b.step(common)
            if selected >= target:
                break
        if selected >= target:
            break
    if not selected:
        raise RuntimeError("No divergent material-law pairs found; experiment invalid")
    data = (torch.from_numpy(np.stack(x)),
            torch.from_numpy(np.stack(masks)),
            torch.from_numpy(np.stack(signatures)),
            torch.tensor(labels, dtype=torch.long))
    return data, dict(worlds_examined=searched, divergent_pairs=selected,
                      pairs_after_gate_closure=after_gate,
                      selected_from_held_law=0, candidate_limit=limit)


def train(seed, original, pairs, updates, mode):
    """Matched initialization/updates; only causal pairs vs sham differ.

    BC receives the same quantity of ordinary trajectory samples. Sham gets
    the exact paired states and labels but independently permuted donor probes.
    Blind sees the same paired states/labels with no donor probes.
    """
    conditioned = mode != "blind"
    torch.manual_seed(seed)
    model = prior.Receiver(conditioned)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0014, weight_decay=0.0001)
    rng = np.random.default_rng(seed + 73)
    control_rng = np.random.default_rng(seed + 471)
    model.train()
    trace = []
    x, m, s, y = original
    px, pm, ps, py = pairs
    for step in range(updates):
        ix = torch.from_numpy(rng.integers(len(y), size=128))
        jx = torch.from_numpy(rng.integers(len(py), size=64))
        x0, m0, s0, y0 = x[ix], m[ix], s[ix], y[ix]
        if mode == "bc":
            kx = torch.from_numpy(control_rng.integers(len(y), size=64))
            x1, m1, s1, y1 = x[kx], m[kx], s[kx], y[kx]
        else:
            x1, m1, s1, y1 = px[jx], pm[jx], ps[jx], py[jx]
            if mode == "shuffled":
                permutation = torch.from_numpy(control_rng.permutation(len(jx)))
                s1 = s1[permutation]
        full_x = torch.cat((x0, x1), dim=0)
        full_mask = torch.cat((m0, m1), dim=0)
        full_sig = torch.cat((s0, s1), dim=0) if conditioned else None
        full_y = torch.cat((y0, y1), dim=0)
        logits = model(full_x, full_mask, full_sig)
        loss = F.cross_entropy(logits, full_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (0, updates-1):
            trace.append(dict(update=step+1, imitation_loss=float(loss.detach())))
    return model.eval(), trace


def infer(model, observation, legal, signature):
    with torch.no_grad():
        output = model(torch.from_numpy(observation[None]),
                       torch.from_numpy(legal[None]),
                       torch.from_numpy(signature[None]) if model.conditioned else None)
    action = int(output.argmax(dim=-1).item())
    assert legal[action]
    return action


def wet_probe(seed, bank, models, wanted, limit, family, goal=1, paint=1):
    """Selection remains oracle-only; retain per-world evidence and denominators."""
    counts = {key: dict(true=0, swapped=0, both=0, flipped=0, illegal=0)
              for key in ("paired", "bc", "shuffled", "blind")}
    examples = []
    searched, found, after_gate = 0, 0, 0
    for i in range(limit):
        world_seed = seed + i * 191
        world = prior.prior.maze(world_seed, family)
        law_a, law_b = (1, goal, paint), (-1, goal, paint)
        ga, gb = prior.HistoryGame(world, law_a), prior.HistoryGame(world, law_b)
        for t in range(6):
            if ga.done or gb.done:
                break
            xa, legal = ga.observe()
            xb, other_legal = gb.observe()
            assert np.array_equal(xa, xb) and np.array_equal(legal, other_legal)
            true, swap = ga.oracle(), gb.oracle()
            if true is None or swap is None:
                break
            if true < 4 and swap < 4 and true != swap:
                sig_a = donor_signature(bank, law_a, "gated")
                sig_b = donor_signature(bank, law_b, "gated")
                preds = {}
                for key in counts:
                    m = models[key]
                    a = infer(m, xa, legal, sig_a)
                    b = infer(m, xa, legal, sig_b)
                    counts[key]["true"] += int(a == true)
                    counts[key]["swapped"] += int(b == swap)
                    counts[key]["both"] += int(a == true and b == swap)
                    counts[key]["flipped"] += int(a != b)
                    counts[key]["illegal"] += int(not legal[a] or not legal[b])
                    preds[key] = [a, b]
                if len(examples) < 16:
                    examples.append(dict(seed=world_seed, stage=t, true_action=true,
                        opposite_action=swap, after_gate=int(ga.closed),
                        actions=preds, world_sha256=hashlib.sha256(
                            xa.tobytes()).hexdigest()))
                after_gate += int(ga.closed)
                found += 1
                if found >= wanted:
                    break
            action = true
            if not other_legal[action]:
                break
            ga.step(action)
            gb.step(action)
        searched += 1
        if found >= wanted:
            break
    return dict(examined_worlds=searched, divergent_cases=found,
                after_gate_cases=after_gate, held_three_law=(goal == paint == 1),
                family=family, modes=counts, examples=examples, cap=limit)


def play(world, law, mode, bank, models, horizon):
    game = prior.HistoryGame(world, law)
    path, actions = [game.pos], []
    for step in range(horizon):
        if game.done:
            break
        if mode == "oracle":
            action = game.oracle()
        else:
            ob, legal = game.observe()
            inferred = (-law[0], law[1], law[2]) if mode == "wrong_wet" else (
                (law[0], -law[1], law[2]) if mode == "wrong_goal" else law)
            sig = donor_signature(bank, inferred, "gated")
            key = "paired" if mode in ("wrong_wet", "wrong_goal") else mode
            action = infer(models[key], ob, legal, sig)
        if action is None:
            break
        assert game.legal()[action]
        game.step(action)
        path.append(game.pos)
        actions.append(prior.prior.ACTIONS[action])
    return dict(success=int(game.done), steps=game.steps,
        cost=round(game.travel_cost, 5), illegal=game.illegal,
        retraces=game.retraces, wet=game.wet_steps, closure=int(game.closed),
        painted=len(game.painted), path=path, actions=actions)


def trajectory_eval(seed, bank, models, sealed, horizon):
    results, gallery = {}, []
    for family in ("iid", "ood"):
        rows = {mode: [] for mode in MODES}
        for i in range(sealed):
            wseed = seed + i*103 + (10_000_000 if family == "ood" else 0)
            world = prior.prior.maze(wseed, family)
            law = HELD if i%2 == 0 else LAWS[(i//2)%len(LAWS)]
            trials = {mode: play(world, law, mode, bank, models, horizon)
                      for mode in MODES}
            for key, v in trials.items():
                rows[key].append(v)
            if i < 4:
                gallery.append(dict(family=family, seed=wseed, laws=law,
                    wall=world["wall"].astype(int).tolist(),
                    wet=world["wet"].astype(int).tolist(),
                    gate=world["gate"], close_after=world["close_after"],
                    start=world["start"], goal=world["goal"],
                    alternate=world["alternate"], trials=trials))
        summary = {}
        for key, trials in rows.items():
            succ = [v for v in trials if v["success"]]
            summary[key] = dict(worlds=sealed, success=len(succ),
                withheld_success=sum(v["success"] for j, v in enumerate(trials) if j%2==0),
                illegal=sum(v["illegal"] for v in trials),
                retraces=sum(v["retraces"] for v in trials),
                failure_penalized_cost=round(float(np.mean([
                    v["cost"] + (0 if v["success"] else 3*horizon)
                    for v in trials])), 5),
                mean_success_cost=round(float(np.mean([v["cost"] for v in succ])), 5)
                if succ else None)
        matched = [i for i in range(sealed)
                   if all(rows[k][i]["success"] for k in ("paired", "bc", "oracle"))]
        result = dict(trajectories=summary, jointly_successful_worlds=len(matched))
        if matched:
            result["paired_mean_extra_cost"] = round(float(np.mean([
                rows["paired"][i]["cost"] - rows["oracle"][i]["cost"]
                for i in matched])), 5)
            result["bc_mean_extra_cost"] = round(float(np.mean([
                rows["bc"][i]["cost"] - rows["oracle"][i]["cost"]
                for i in matched])), 5)
        results[family] = result
    return results, gallery


def gallery_svg(traces, filename):
    cases = []
    for sample in traces:
        data = dict(sample)
        data["trials"] = {
            "inherited": sample["trials"]["paired"],
            "wrong_wet": sample["trials"]["wrong_wet"],
            "wrong_goal": sample["trials"]["wrong_goal"],
            "blind": sample["trials"]["blind"],
            "oracle": sample["trials"]["oracle"],
        }
        cases.append(data)
    prior.prior.svg_gallery(cases, filename)
    filename.write_text(filename.read_text().replace("DG016", "DG019"))


def selftest():
    prior.selftest()
    a = prior.HistoryGame(prior.prior.maze(1234, "train"), (1,-1,1))
    b = prior.HistoryGame(prior.prior.maze(1234, "train"), (-1,-1,1))
    oa, ma = a.observe()
    ob, mb = b.observe()
    assert np.array_equal(oa, ob) and np.array_equal(ma, mb)
    assert oa.shape == (prior.CHANNELS, prior.SIDE, prior.SIDE)
    torch.manual_seed(3)
    for conditioned in (True, False):
        m = prior.Receiver(conditioned)
        s = torch.randn(2,24) if conditioned else None
        pred = m(torch.randn(2,prior.CHANNELS,prior.SIDE,prior.SIDE),
                 torch.ones(2,6,dtype=torch.bool),s)
        assert pred.shape == (2,6)
        pred.square().mean().backward()
        assert m.net[0].weight.grad is not None
    print("DG019 SELFTEST PASS: matched observable state and legal neural policy")


def run(seed, out, train_worlds, updates, paired_target, paired_limit,
        sealed, horizon, donor_steps, probe_target, probe_limit):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    bank, validation = build_donors(seed, donor_steps, out)
    ordinary = prior.dataset(seed+2000, bank, train_worlds)
    paired, selection = paired_samples(seed+350000, bank, paired_target, paired_limit)
    models, traces = {}, {}
    for name in ("bc", "paired", "shuffled", "blind"):
        models[name], traces[name] = train(seed+3000 if name!="blind" else seed+6000,
                                           ordinary, paired, updates, name)
        torch.save(models[name].state_dict(), out/f"receiver-{name}.pt")
    counter = {}
    for family, family_shift in (("iid", 0), ("ood", 10_000_000)):
        for name, goal, paint in (("held",1,1),("seen",-1,1)):
            counter[f"{family}-{name}"] = wet_probe(
                seed+650000+family_shift+(1000000 if name=="seen" else 0),
                bank, models, probe_target, probe_limit, family, goal, paint)
    results, samples = trajectory_eval(seed+100000, bank, models, sealed, horizon)
    assert all(v["illegal"]==0 for split in results.values()
               for v in split["trajectories"].values())
    assert all(split["trajectories"]["oracle"]["success"]==sealed
               for split in results.values())
    (out/"trajectories.json").write_text(json.dumps(samples,indent=2)+"\n")
    gallery_svg(samples,out/"gallery.svg")
    report = dict(experiment="DG019",protocol="matched-law-contrastive-v1",
        seed=seed,source_sha256=sha256(__file__),
        parent_sha="18d0eab5fe285e8e021bce3fbf180919a99071aa",
        torch=torch.__version__,numpy=np.__version__,donors=len(bank),
        receivers=len(models),donor_validation_mse=validation,
        train_worlds=train_worlds,train_base_decisions=len(ordinary[3]),
        train_paired_selection=selection,updates=updates,training_trace=traces,
        withheld_law=HELD,causal_control="same-observation opposite-wet oracle divergence",
        source_probes="independently trained narrow donors for training; unseen gated for evaluation",
        matched_initialization=True,matched_update_budget=True,
        source_blind_control=True,shuffled_signatures_control=True,
        no_oracle_at_inference=True,no_distance_potential_at_inference=True,
        train_sealed_seed_overlap=0,host_legal_actions=True,
        counterfactual=counter,results=results,horizon=horizon,
        caveats=("Donor signatures are fixed per law, not transplanted state. "
            "Oracle-divergence selection creates a deliberately enriched training distribution. "
            "No native CyberSand integration or autonomous law discovery. "
            "IID vs OOD maze and gate timing are confounded."),
        seconds=round(time.monotonic()-t0,2))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    hashes={p.name:sha256(p) for p in sorted(out.iterdir())
            if p.is_file() and p.name!="sha256.json"}
    (out/"sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--seed",type=int,default=26011634)
    p.add_argument("--out",default="/tmp/dg019")
    p.add_argument("--train-worlds",type=int,default=180)
    p.add_argument("--updates",type=int,default=600)
    p.add_argument("--paired-target",type=int,default=240)
    p.add_argument("--paired-limit",type=int,default=850)
    p.add_argument("--sealed",type=int,default=80)
    p.add_argument("--horizon",type=int,default=96)
    p.add_argument("--donor-steps",type=int,default=140)
    p.add_argument("--probe-target",type=int,default=50)
    p.add_argument("--probe-limit",type=int,default=2500)
    a=p.parse_args()
    if a.selftest:selftest()
    else:run(a.seed,a.out,a.train_worlds,a.updates,a.paired_target,
             a.paired_limit,a.sealed,a.horizon,a.donor_steps,
             a.probe_target,a.probe_limit)
