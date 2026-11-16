#!/usr/bin/env python3
"""QL-012: targeted RETRACE retention during observable paired repair.

Research-only. Free CPU. Immutable QL-003 Q1/XNOR + Q4-middle parent.
Tests whether preserving the parent's RETRACE preference on *real observed*
legal game states can coexist with the QL-011 adaptive-neighbour objective.
Includes independent zero-randomness and host-randomness cohorts, never
changes STOP legality, host masking, or the CyberSand game.
"""
import argparse
import json
import os
import random
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quant_worm import ACTIONS, FAMILIES, WormWorld, digest, draw, timing
from ql002_branch_retrace import examples
from ql003_surgery import morphology
from ql005_bitserial import verified_source
from ql007_mixed_bridges import Hybrid, PARENT_SHA
from ql010_paired_contrastive import legal_pairs, pair_scores
from ql011_observable_repair import RankFour, observable_targets, score, paired_report

R = ACTIONS.index("RETRACE")
S = ACTIONS.index("STOP")
CONFIGS = (("paired-ce", 0., .30),
           ("retrace-anchor-medium", 1., .30),
           ("retrace-anchor-strong", 3., .30))


def train(parent, data, labels, pairs, anchor_ids, updates, weight, prior, seed):
    x, _, legal = data
    torch.manual_seed(seed)
    net = RankFour(parent, seed)
    with torch.no_grad():
        base = parent(x).detach()
    assert len(anchor_ids) >= 32
    counts = torch.bincount(labels, minlength=len(ACTIONS)).float().clamp(min=1)
    sampling = (1. / counts[labels]).double()
    sampling /= sampling.sum()
    opt = torch.optim.AdamW((net.u, net.v), lr=.025, weight_decay=0.)
    trace = []
    for t in range(updates):
        ix = torch.multinomial(sampling, 128, replacement=True)
        logits = (base[ix] + net.correction(x[ix])).masked_fill(~legal[ix], -1e9)
        oracle_loss = F.cross_entropy(logits, labels[ix])
        kl = F.kl_div(F.log_softmax(logits, dim=-1),
                      F.softmax(base[ix].masked_fill(~legal[ix], -1e9), dim=-1),
                      reduction="batchmean")
        i, j = pairs[torch.randint(len(pairs), (96,))].unbind(1)
        residual_gap = pair_scores(net.correction(x[i]), net.correction(x[j]),
                                   labels[i], labels[j])
        paired_loss = F.softplus(.5 - residual_gap).mean()
        ai = anchor_ids[torch.randint(len(anchor_ids), (128,))]
        anchor_logits = (base[ai] + net.correction(x[ai])).masked_fill(~legal[ai], -1e9)
        assert bool(legal[ai, R].all())
        retrace_loss = F.cross_entropy(anchor_logits,
                                      torch.full((len(ai),), R, dtype=torch.long))
        loss = oracle_loss + 1.75 * paired_loss + prior * kl + weight * retrace_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append({"update": t + 1, "loss": float(loss.detach()),
                      "oracle": float(oracle_loss.detach()),
                      "pair": float(paired_loss.detach()),
                      "retrace_anchor": float(retrace_loss.detach()),
                      "parent_kl": float(kl.detach())})
    moment_count = sum(int(bool(torch.count_nonzero(s["exp_avg"])))
                       for s in opt.state.values())
    assert moment_count == 2 and bool(torch.count_nonzero(net.v.detach()))
    return net, opt.state_dict(), trace, {
        "updates": updates, "retrace_weight": weight,
        "parent_kl_weight": prior, "seed": seed,
        "real_parent_retrace_anchors": len(anchor_ids),
        "trainable_params": net.u.numel() + net.v.numel(),
        "optimizer_nonzero_moments": moment_count,
        "first": trace[0], "last": trace[-1],
        "parent_packed_codes_unchanged": True,
    }


def audited_rollout(model, name, seed, family, root, probability, horizon=96):
    """Exactly the W0 action legality/world steps; audit the random host overrides.

    probability .10 matches QL-001 rollout; probability 0 is a *diagnostic*
    counterfactual, not a claim about the shipped host. STOP legal from tick 0.
    """
    w = WormWorld(seed, family)
    rng = random.Random(1000003 * seed)
    steps = []
    for t in range(horizon):
        if w.stopped:
            break
        menu = w.legal()
        assert "STOP" in menu
        feature = torch.from_numpy(w.features(t))[None, :]
        if model is not None:
            with torch.no_grad():
                values = model(feature)[0].tolist()
            raw = ACTIONS[int(np.argmax(values))]
            masked = [v if a in menu else -1e9 for a, v in zip(ACTIONS, values)]
            desired = ACTIONS[int(np.argmax(masked))]
            override = rng.random() < probability
            action = rng.choice(menu) if override else desired
        else:
            values = None
            raw = None
            desired = None
            order = ("STRAIGHT", "RIGHT", "LEFT", "BRANCH", "RETRACE", "STOP")
            desired_wall = next(a for a in order if a in menu)
            override = rng.random() < probability
            action = rng.choice(menu) if override else desired_wall
        assert action in menu
        pos = list(w.pos)
        w.apply(action)
        steps.append({"tick": t, "position": pos, "next_position": list(w.pos),
                      "legal": list(menu), "raw_top": raw,
                      "masked_top": desired, "host_override": override,
                      "chosen": action, "scores": values})
    stem = f"{name}-{family.lower()}-{seed}-{int(probability*100)}pct"
    target = root / "gallery" / (stem + ".png")
    draw(w, target)
    chosen = Counter(w.actions)
    stats = {"seed": seed, "family": family, "policy": name, "override_rate": probability,
             "turns": len(steps), "unique_cells": len(w.ink),
             "branch": chosen["BRANCH"], "retrace": chosen["RETRACE"],
             "stop": chosen["STOP"], "host_override_events": sum(s["host_override"] for s in steps),
             "host_override_stop": sum(s["host_override"] and s["chosen"] == "STOP" for s in steps),
             "raw_top_legal": sum(s["raw_top"] in s["legal"] for s in steps)
                    if model is not None else None,
             "morphology": morphology(steps), "png": str(target.relative_to(root)),
             "png_sha256": digest(target)}
    return stats, steps


def cohort(model, name, seeds, root, probability, raw):
    episodes = []
    for seed in seeds:
        for family in FAMILIES:
            entry, steps = audited_rollout(model, name, seed, family,
                                          root, probability)
            episodes.append(entry)
            raw.append({"policy": name, "seed": seed, "family": family,
                        "override_rate": probability, "first_pass": True,
                        "steps": steps})
    n = len(episodes)
    return {
        "n_episodes": n,
        "mean_unique_cells": round(sum(e["unique_cells"] for e in episodes)/n, 5),
        "mean_turns": round(sum(e["turns"] for e in episodes)/n, 5),
        "total_branches": sum(e["branch"] for e in episodes),
        "total_retraces": sum(e["retrace"] for e in episodes),
        "stop_episodes": sum(bool(e["stop"]) for e in episodes),
        "host_override_stop": sum(e["host_override_stop"] for e in episodes),
        "total_host_overrides": sum(e["host_override_events"] for e in episodes),
        "raw_top_legal": sum(e["raw_top_legal"] for e in episodes)
            if model is not None else None,
        "total_actions": sum(e["turns"] for e in episodes),
        "family_distinct_by_seed": {str(seed):len({e["png_sha256"] for e in episodes
                                     if e["seed"] == seed}) for seed in seeds},
        "episodes": episodes,
    }


def self_test():
    w = WormWorld(10117, "GREEBLES")
    assert "STOP" in w.legal()
    assert "STOP" in ACTIONS and "RETRACE" in ACTIONS
    assert "RETRACE" not in w.legal()
    for step in range(3):
        menu = w.legal()
        a = next((c for c in ("STRAIGHT", "RIGHT", "LEFT") if c in menu), "STOP")
        w.apply(a)
        if w.stopped:
            break
    assert all(w.grid[y, x] == 0 for x, y in w.ink)
    from ql011_observable_repair import self_test as original_test
    original_test()
    print("QL012 PASS: host legality, protected cells, source-pinned parent adapter")


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(12012)
    np.random.seed(12012)
    root = Path(args.out)
    (root / "gallery").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    parent_folder = Path(args.parent)
    _, checked = verified_source(parent_folder, PARENT_SHA)
    q1 = parent_folder / "models" / "bal-q1.npz"
    q4 = parent_folder / "models" / "bal-q4.npz"
    parent = Hybrid(q1, q4, "b")
    train_set, _, train_bad = examples(range(2100, 2180), 33102)
    held_set, _, held_bad = examples(range(9100, 9132), 33103)
    assert train_bad == held_bad == 0 and len(held_set[1]) == 4352
    train_oracle = observable_targets(train_set)
    held_oracle = observable_targets(held_set)
    train_pairs = legal_pairs((train_set[0],train_oracle,train_set[2]),512,12013)
    held_pairs = legal_pairs((held_set[0],held_oracle,held_set[2]),128,12017)
    with torch.no_grad():
        parent_choice = parent(train_set[0]).masked_fill(~train_set[2], -1e9).argmax(1)
        anchors = torch.where(parent_choice == R)[0]
    assert len(anchors) > 100
    names = {"parent-q1-q4b": parent}
    training = {}
    for name, weight, prior in CONFIGS:
        model, state, trace, diag = train(parent,train_set,train_oracle,
                                          train_pairs,anchors,args.steps,
                                          weight,prior,12021)
        names[name] = model
        training[name] = diag
        torch.save({"state_dict": model.state_dict(), "optimizer": state,
                    "trace": trace, "training": diag,
                    "source_sha": os.environ.get("QL_SOURCE_SHA", "local"),
                    "parent_source_sha": PARENT_SHA,
                    "parent_q1_sha256": digest(q1),
                    "parent_q4_sha256": digest(q4)},
                   root/"models"/(name+"-optimizer.pt"))
    raw = []
    seeds = (10011,10023,10047,10051,10067,10079)
    results = {}
    for name, model in names.items():
        results[name] = {
            "heldout": score(model,held_set,held_set[1],held_oracle,parent),
            "matched_pairs":paired_report(model,held_set,held_pairs,held_oracle),
            "host_random_10pct": cohort(model,name,seeds,root,.10,raw),
            "no_host_random_diagnostic": cohort(model,name,seeds,root,0.,raw),
            "working_adapter_bytes":model.memory_bytes() if isinstance(model,RankFour) else 0,
            "same_runner_latency":timing(model,held_set[0],n=96),
        }
    control = {
        "no_host_random_diagnostic":cohort(None,"wall",seeds,root,0.,raw),
        "host_random_10pct":cohort(None,"wall-stable",seeds,root,.10,raw),
    }
    (root/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    output = {
        "schema":"ql012-targeted-retrace-anchor-1",
        "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
        "parent_source_sha":PARENT_SHA,"parent_manifest_entries":checked,
        "parent_q1_sha256":digest(q1),"parent_q4_sha256":digest(q4),
        "parent_retrace_anchor_train_states":len(anchors),
        "train_n":len(train_set[1]),"heldout_n":len(held_set[1]),
        "heldout_pairs":len(held_pairs),"optimizer_updates":len(CONFIGS)*args.steps,
        "training":training,"variants":results,"controls":control,
        "seeds":seeds,"original_png_n":len(list((root/"gallery").glob("*.png"))),
        "first_pass_trajectories":len(raw),
        "raw_decisions_sha256":digest(root/"raw_decisions.json"),
        "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations":[
            "Research synthetic W0, no native CyberSand gameplay or editing",
            "Q1 packed XNOR reference plus decoded Q4 middle, not actual GGUF IQ family",
            "RETRACE anchor labels are parent choices, not aesthetic ground truth",
            "Training and held-out states share exploration regime but disjoint world seeds",
            "No-host-random cohort is a diagnostic intervention, not shipped host semantics",
            "Matched host seed and random streams but policy state trajectories diverge",
            "STOP always host legal, and 10pct random host choices can truncate every policy",
            "Process RSS includes entire Python and PyTorch runtime",
            "Python timings are decisions per second, not SLM tokens per second",
            "No game, provider, production, map or Workbench modifications"
        ],
    }
    (root/"results.json").write_text(json.dumps(output,indent=2)+"\n")
    manifest=[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,
               "sha256":digest(p)} for p in sorted(root.rglob("*"))
              if p.is_file() and p.name!="MANIFEST.json"]
    (root/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"QL012_COMPLETE","source":output["source_sha"],
         "updates":output["optimizer_updates"],"heldout":output["heldout_n"],
         "pngs":output["original_png_n"],"anchors":len(anchors),
         "results":{name:{"oracle":v["heldout"]["observable_oracle_agreement"],
          "adaptive":v["heldout"]["live_vs_constant_adapter_changes"],
          "stochastic_cells":v["host_random_10pct"]["mean_unique_cells"],
          "diagnostic_cells":v["no_host_random_diagnostic"]["mean_unique_cells"],
          "stochastic_retraces":v["host_random_10pct"]["total_retraces"]}
          for name,v in results.items()}},sort_keys=True))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="ql003-source")
    parser.add_argument("--out",default="ql012-out")
    parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    self_test() if args.self_test else run(args)
