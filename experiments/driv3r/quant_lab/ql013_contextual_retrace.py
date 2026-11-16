#!/usr/bin/env python3
"""QL-013: state-conditioned RETRACE repair with explicit novelty counterfactuals.

Research-only free-CPU synthetic CyberSand-style W0 host. Frozen original
QL-003 Q1/XNOR + Q4-middle packed parent. Preserve unusual retracing only
when local novelty is exhausted, instead of globally rewarding backtracking.
Evaluate paired CE, global anchoring, conditional anchoring, and conditional
negative examples, plus an untrained contextual guard. Live host menus always
decide legality; STOP remains legal at turn zero. No game-side changes.
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
# (name, retrace-positive weight, novel-forward negative weight, anchor set, parent KL)
CONFIGS = (
    ("paired-ce", 0., 0., "none", .30),
    ("global-retrace", 1., 0., "global", .30),
    ("exhaustion-retrace", 1., 0., "exhausted", .30),
    ("exhaustion-guarded", 1.5, .75, "exhausted", .30),
)


def novelty_mask(features):
    """Number of *actually observed* fresh legal neighbouring directions.

    The W0 feature contract uses slots 3/7/11 for fresh forward directions.
    This predicate is not derived from policy predictions or an invented state.
    """
    assert features.ndim == 2 and features.shape[1] == 27
    return (features[:, [3, 7, 11]] > .5).sum(1)


class ContextSuppress:
    """Untrained comparator: attenuate only RETRACE when >=2 fresh options."""
    def __init__(self, base, penalty=1.5):
        self.base, self.penalty = base, penalty

    def __call__(self, x):
        logits = self.base(x).clone()
        logits[:, R] -= self.penalty * (novelty_mask(x) >= 2).to(logits.dtype)
        return logits


def conditional_report(model, data, oracle):
    """Held-out subgroup metrics, with support and oracle denominators visible."""
    x, _, legal = data
    fresh = novelty_mask(x)
    with torch.no_grad():
        values = model(x)
        raw = values.argmax(1)
        masked = values.masked_fill(~legal, -1e9).argmax(1)
        retrace_prob = values.masked_fill(~legal, -1e9).softmax(1)[:, R]
    subsets = {
        "exhausted_with_retrace": (fresh == 0) & legal[:, R],
        "one_fresh_with_retrace": (fresh == 1) & legal[:, R],
        "two_plus_fresh_with_retrace": (fresh >= 2) & legal[:, R],
        "oracle_retrace": oracle == R,
        "oracle_nonretrace": oracle != R,
    }
    out = {}
    for key, select in subsets.items():
        n = int(select.sum())
        assert n > 0, (key, n)
        out[key] = {
            "n": n, "oracle_retrace": int((oracle[select] == R).sum()),
            "greedy_retrace": int((masked[select] == R).sum()),
            "greedy_stop": int((masked[select] == S).sum()),
            "mean_retrace_probability": round(float(retrace_prob[select].mean()), 7),
            "oracle_agreement": int((masked[select] == oracle[select]).sum()),
            "host_masked_legal": int(legal[select].gather(1,masked[select,None]).sum()),
        }
    out["total_host_masked_legal"] = int(legal.gather(1,masked[:, None]).sum())
    out["total_raw_top_legal"] = int(legal.gather(1,raw[:,None]).sum())
    out["total"] = len(x)
    return out


def train(parent, data, labels, pairs, parent_anchors, positives, negatives,
          updates, positive_weight, negative_weight, kind, prior, seed):
    x, _, legal = data
    torch.manual_seed(seed)
    net = RankFour(parent, seed)
    with torch.no_grad():
        base = parent(x).detach()
    positive_pool = parent_anchors if kind == "global" else positives
    assert len(positives) >= 32 and len(negatives) >= 32
    assert len(parent_anchors) >= 32
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
        i,j = pairs[torch.randint(len(pairs), (96,))].unbind(1)
        residual_gap = pair_scores(net.correction(x[i]), net.correction(x[j]),
                                   labels[i], labels[j])
        paired_loss = F.softplus(.5 - residual_gap).mean()

        # Always exercise both loss terms to preserve comparable RNG draws.
        ai = positive_pool[torch.randint(len(positive_pool), (128,))]
        pos_logits = (base[ai] + net.correction(x[ai])).masked_fill(~legal[ai], -1e9)
        assert bool(legal[ai, R].all())
        retrace_loss = F.cross_entropy(pos_logits,
                                      torch.full((len(ai),), R, dtype=torch.long))
        ni = negatives[torch.randint(len(negatives), (128,))]
        neg_logits = (base[ni] + net.correction(x[ni])).masked_fill(~legal[ni], -1e9)
        assert bool(legal[ni, R].all()) and bool((labels[ni] != R).all())
        novel_loss = F.cross_entropy(neg_logits, labels[ni])

        loss = (oracle_loss + 1.75 * paired_loss + prior * kl
                + positive_weight * retrace_loss + negative_weight * novel_loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append({"update":t+1, "loss":float(loss.detach()),
                      "oracle":float(oracle_loss.detach()),
                      "pair":float(paired_loss.detach()),
                      "positive_retrace":float(retrace_loss.detach()),
                      "novel_forward":float(novel_loss.detach()),
                      "parent_kl":float(kl.detach())})
    moment_count = sum(int(bool(torch.count_nonzero(s["exp_avg"])))
                       for s in opt.state.values())
    assert moment_count == 2 and bool(torch.count_nonzero(net.v.detach()))
    return net, opt.state_dict(), trace, {
        "updates":updates, "retrace_weight":positive_weight,
        "novel_forward_weight":negative_weight, "positive_anchor_pool":kind,
        "parent_kl_weight":prior, "seed":seed,
        "parent_retrace_anchor_states":len(parent_anchors),
        "exhausted_retrace_anchor_states":len(positives),
        "novel_forward_negative_states":len(negatives),
        "trainable_params":net.u.numel()+net.v.numel(),
        "optimizer_nonzero_moments":moment_count,
        "first":trace[0], "last":trace[-1],
        "parent_packed_codes_unchanged":True,
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
    x = torch.zeros(3,27)
    x[1,3] = 1
    x[2,[3,7]] = 1
    assert novelty_mask(x).tolist() == [0,1,2]
    class TestParent:
        def __call__(self,a):
            return torch.zeros((len(a),len(ACTIONS)))
    effect = ContextSuppress(TestParent())(x)
    assert effect[0,R] == effect[1,R] == 0 and effect[2,R] == -1.5
    print("QL013 PASS: host legality, novelty groups, contextual guard and frozen parent")


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
    fresh = novelty_mask(train_set[0])
    positives = torch.where((fresh == 0) & train_set[2][:,R]
                            & (train_oracle == R))[0]
    negatives = torch.where((fresh >= 2) & train_set[2][:,R]
                            & (train_oracle != R))[0]
    assert len(positives) >= 32 and len(negatives) >= 32
    names = {"parent-q1-q4b": parent, "parent-context-guard": ContextSuppress(parent)}
    training = {}
    for name, positive_weight, negative_weight, kind, prior in CONFIGS:
        model, state, trace, diag = train(parent,train_set,train_oracle,
                                          train_pairs,anchors,positives,negatives,
                                          args.steps,positive_weight,
                                          negative_weight,kind,prior,13021)
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
    seeds = (10211,10223,10247,10251,10267,10279)
    results = {}
    for name, model in names.items():
        results[name] = {
            "heldout": score(model,held_set,held_set[1],held_oracle,parent),
            "conditional_heldout": conditional_report(model, held_set, held_oracle),
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
        "schema":"ql013-contextual-retrace-v1",
        "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
        "parent_source_sha":PARENT_SHA,"parent_manifest_entries":checked,
        "parent_q1_sha256":digest(q1),"parent_q4_sha256":digest(q4),
        "parent_retrace_anchor_train_states":len(anchors),
        "exhausted_retrace_anchor_train_states":len(positives),
        "novel_forward_negative_train_states":len(negatives),
        "novelty_feature_indices":[3,7,11],
        "novelty_threshold":.5,
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
            "Global RETRACE anchors are parent choices, not aesthetic ground truth",
            "Conditional positives and negative examples are observational-oracle labels",
            "The untrained context guard is a causal ablation, not a learned low-bit kernel",
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
    print(json.dumps({"status":"QL013_CONTEXTUAL_COMPLETE","source":output["source_sha"],
         "updates":output["optimizer_updates"],"heldout":output["heldout_n"],
         "pngs":output["original_png_n"],"anchors":len(anchors),"exhausted":len(positives),"novel_negatives":len(negatives),
         "results":{name:{"oracle":v["heldout"]["observable_oracle_agreement"],
          "adaptive":v["heldout"]["live_vs_constant_adapter_changes"],
          "stochastic_cells":v["host_random_10pct"]["mean_unique_cells"],
          "diagnostic_cells":v["no_host_random_diagnostic"]["mean_unique_cells"],
          "stochastic_retraces":v["host_random_10pct"]["total_retraces"]}
          for name,v in results.items()}},sort_keys=True))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="ql003-source")
    parser.add_argument("--out",default="ql013-out")
    parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    self_test() if args.self_test else run(args)
