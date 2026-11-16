#!/usr/bin/env python3
"""QL-017: separate conditional RETRACE preservation from BRANCH-rate control.

Independent, synthetic W0 game-policy research. QL-003 Q1/Q4 parent, QL-014
frontier Q4 and QL-015 STOP-safe Q1 are immutable. The QL-016 sparse-Q1
personality is also reloaded for matched comparisons, never rewritten.
No CyberSand game/runtime modifications, GGUF claims or low-bit speed claims.
"""
import argparse
import json
import os
import resource
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quant_worm import ACTIONS, digest, timing
from ql002_branch_retrace import examples
from ql011_observable_repair import observable_targets
from ql013_contextual_retrace import cohort
from ql015_budget_stop import budget_stage, budget_report
from ql016_contextual_character import (
    B, R, S, SHAPE, CharacterIsland, bucket, load_immutable_models,
    score, verify_archive,
)

SOURCE_016 = "2859264115b97a85a2f8150e661779155051559a"


def reload_ql016(folder, budget, provenance):
    report, n = verify_archive(folder, SOURCE_016, 300)
    assert report["ql015_checkpoint_sha256"] == provenance["ql015_checkpoint_sha256"]
    assert report["ql014_checkpoint_sha256"] == provenance["ql014_checkpoint_sha256"]
    assert report["parent_q1_sha256"] == provenance["parent_q1_sha256"]
    assert report["parent_q4_sha256"] == provenance["parent_q4_sha256"]
    p = folder / "models" / "sparse-q1-no-chain-optimizer.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["source_sha"] == SOURCE_016
    assert ck["bits"] == 1 and ck["sparse"] and ck["updates"] == 48
    assert ck["ql015_checkpoint_sha256"] == provenance["ql015_checkpoint_sha256"]
    assert ck["table_fp32"].shape == SHAPE
    net = CharacterIsland(budget, 1, ck["table_fp32"], suppress_chain=True)
    assert net.payload()["bytes"] == 20
    return net, n, digest(p)


def masks(data, original):
    x, labels, legal = data
    fresh, deep, last = bucket(x)
    eligible = budget_stage(x) < 2
    with torch.no_grad():
        parent_choice = original(x).masked_fill(~legal, -1e9).argmax(1)
    # These are observable states, not label-derived "magic" at inference.
    recovery = eligible & (fresh <= 1) & (deep == 1) & (last == 0) & legal[:, R]
    parent_retrace = recovery & (parent_choice == R)
    new_ground = eligible & (fresh >= 2) & legal[:, B]
    teacher_branch = new_ground & (labels == B)
    unnecessary_retrace = new_ground & legal[:, R] & (labels != R)
    chained_retrace = eligible & (last == 1) & (fresh > 0) & legal[:, R]
    result = dict(recovery=recovery, parent_retrace=parent_retrace,
                  new_ground=new_ground, teacher_branch=teacher_branch,
                  unnecessary_retrace=unnecessary_retrace,
                  chained_retrace=chained_retrace)
    return result


def logistic_margin(z, target_index, margin=.30):
    """Smooth ranking objective against the strongest legal alternative."""
    other = z.clone()
    other[:, target_index] = -1e9
    return F.softplus(other.logsumexp(1) - z[:, target_index] + margin).mean()


def train(net, data, parent, updates, seed, branch_cap):
    x, labels, legal = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    with torch.no_grad():
        frozen = net.base(x).detach()
        prior = frozen.masked_fill(~legal, -1e9).softmax(1)
    m = masks(data, parent)
    support = {name: int(v.sum()) for name, v in m.items()}
    assert min(support.values()) >= 4, support
    indices = {k: torch.where(v)[0] for k, v in m.items()}
    class_count = torch.bincount(labels, minlength=len(ACTIONS)).float().clamp_min(1)
    sampling = (1. / class_count[labels]).double()
    sampling /= sampling.sum()
    opt = torch.optim.AdamW([net.table], lr=.027 if net.bits == 32 else .018,
                            weight_decay=0.)
    trace = []
    for step in range(updates):
        batch = torch.multinomial(sampling, 192, replacement=True)
        scores = (frozen[batch] + net.correction(x[batch])).masked_fill(~legal[batch], -1e9)
        ce = F.cross_entropy(scores, labels[batch])
        kl = F.kl_div(F.log_softmax(scores, dim=1), prior[batch], reduction="batchmean")

        picked = {k: ids[torch.randint(len(ids), (96,))] for k, ids in indices.items()}
        z = {k: (frozen[v] + net.correction(x[v])).masked_fill(~legal[v], -1e9)
             for k, v in picked.items()}
        retrace = logistic_margin(z["parent_retrace"], R, margin=.20)
        suppress_unneeded = F.softplus(
            z["unnecessary_retrace"][:, R] -
            z["unnecessary_retrace"].scatter(1, torch.full(
                (96, 1), R, dtype=torch.long), -1e9).logsumexp(1) + .05).mean()
        branch_teacher = logistic_margin(z["teacher_branch"], B, margin=0.)
        new_ground_branch_probability = z["new_ground"].softmax(1)[:, B].mean()
        # A bounded SOFT conditional rate control, not a hard legal-menu edit.
        # This penalizes only expected BRANCH probability above the ceiling.
        over_branch = F.relu(new_ground_branch_probability - branch_cap).square()
        # Avoid repeatedly retracing after a retrace when fresh ground exists.
        chain = F.softplus(
            z["chained_retrace"][:, R] -
            z["chained_retrace"].scatter(1, torch.full(
                (96, 1), R, dtype=torch.long), -1e9).logsumexp(1) + .1).mean()

        loss = (ce + .23 * kl + 1.50 * retrace + .17 * suppress_unneeded +
                .28 * branch_teacher + 14.0 * over_branch + .15 * chain +
                .001 * net.quant_table().square().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append({"step": step + 1, "loss": float(loss.detach()),
                      "ce": float(ce.detach()), "kl": float(kl.detach()),
                      "recovery_margin": float(retrace.detach()),
                      "conditional_branch_probability": float(
                          new_ground_branch_probability.detach()),
                      "branch_cap_penalty": float(over_branch.detach()),
                      "teacher_branch": float(branch_teacher.detach()),
                      "novelty_retrace_penalty": float(suppress_unneeded.detach()),
                      "chain_penalty": float(chain.detach())})
    state = opt.state_dict()
    assert len(state["state"]) == 1
    moment = next(iter(state["state"].values()))["exp_avg"]
    assert bool(torch.count_nonzero(moment))
    assert torch.isfinite(net.table).all()
    return state, trace, support


def context_score(model, data, parent):
    x, labels, legal = data
    m = masks(data, parent)
    with torch.no_grad():
        s = model(x).masked_fill(~legal, -1e9)
        choice = s.argmax(1)
        p = s.softmax(1)
    report = {}
    for name, mask in m.items():
        n = int(mask.sum())
        report[name] = {"n": n, "retrace": int(((choice == R) & mask).sum()),
                        "branch": int(((choice == B) & mask).sum()),
                        "stop": int(((choice == S) & mask).sum()),
                        "mean_p_retrace": float(p[mask, R].mean()),
                        "mean_p_branch": float(p[mask, B].mean())}
    report["host_legal"] = int(legal.gather(1, choice[:, None]).sum())
    return report


def self_test():
    class Zero:
        def __call__(self, x):
            return torch.zeros((len(x), len(ACTIONS)))
    net = CharacterIsland(Zero(), 1, suppress_chain=True)
    x = torch.zeros((4, 27))
    x[0, 21] = 10 / 64
    x[1, [3, 7, 11]] = 1
    x[2, [3, 7]] = 1
    x[3, 19] = 50 / 64
    legal = torch.ones(4, len(ACTIONS), dtype=torch.bool)
    data = (x, torch.tensor([R, B, B, S]), legal)
    result = masks(data, Zero())
    assert result["recovery"][0] and result["new_ground"][1]
    assert not result["recovery"][1] and not result["new_ground"][0]
    assert net.payload()["bytes"] == 20
    assert (net.correction(x)[:, S] == 0).all()
    assert torch.isfinite(logistic_margin(torch.zeros(3, len(ACTIONS)), R))
    for b in (1, 2, 4, 32):
        n = CharacterIsland(Zero(), b, torch.randn(SHAPE), suppress_chain=True)
        assert n(x).shape == (4, len(ACTIONS))
        assert torch.isfinite(n.quant_table()).all()
        assert n.quant_table()[:, :, 1, :].abs().sum() == 0
    print("QL017 SELF-TEST PASS: conditional recovery, branch context, STOP freeze, packed tables")


def run(args):
    self_test()
    torch.set_num_threads(1)
    torch.manual_seed(17017)
    np.random.seed(17017)
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "gallery").mkdir(exist_ok=True)
    parent, budget, provenance = load_immutable_models(
        Path(args.parent), Path(args.ql014), Path(args.ql015))
    inherited, manifest16, inherited_hash = reload_ql016(
        Path(args.ql016), budget, provenance)
    train_set, _, train_bad = examples(range(2100, 2180), 33102)
    held, _, held_bad = examples(range(9100, 9132), 33103)
    assert train_bad == held_bad == 0 and len(held[1]) == 4352
    y_train, y_held = observable_targets(train_set), observable_targets(held)
    # Train with the independently defined observable target, not the imbalanced
    # source teacher labels stored in the W0 examples.
    train_data = (train_set[0], y_train, train_set[2])
    held_data = (held[0], y_held, held[2])
    models = {"original-q1-q4middle": parent,
              "ql015-budget-q1": budget,
              "ql016-sparse-q1": inherited}
    trained = {}

    def train_one(name, bits, cap, init, steps, seed):
        net = CharacterIsland(budget, bits, init, suppress_chain=True)
        optimizer, trace, support = train(
            net, train_data, parent, steps, seed, cap)
        payload = net.payload()
        checkpoint = {"source_sha": os.environ.get("QL_SOURCE_SHA", "local"),
                      "ql016_source_sha": SOURCE_016,
                      "ql016_checkpoint_sha256": inherited_hash,
                      "ql015_checkpoint_sha256": provenance["ql015_checkpoint_sha256"],
                      "ql014_checkpoint_sha256": provenance["ql014_checkpoint_sha256"],
                      "parent_q1_sha256": provenance["parent_q1_sha256"],
                      "parent_q4_sha256": provenance["parent_q4_sha256"],
                      "bits": bits, "branch_probability_cap": cap,
                      "sparse_no_chain": True, "updates": len(trace),
                      "table_fp32": net.table.detach().clone(),
                      "optimizer": optimizer, "trace": trace,
                      "support": support, "packed_payload": payload}
        torch.save(checkpoint, out / "models" / (name + "-optimizer.pt"))
        if bits < 32:
            np.savez_compressed(out / "models" / (name + "-packed.npz"),
                                bits=np.array(bits), packed=payload["packed"],
                                scales=payload["scales"], sparse=np.array(True))
        trained[name] = (net, trace, cap, support)
        models[name] = net
        return net

    uncapped = train_one("recovery-fp32-no-branch-cap", 32, 1.0, None,
                         args.fp_steps, 17001)
    capped = train_one("recovery-fp32-branch12", 32, .12,
                       uncapped.table.detach().clone(), args.fp_steps, 17002)
    for bits in (4, 2, 1):
        train_one("recovery-q%d-branch12" % bits, bits, .12,
                  capped.table.detach().clone(), args.qat_steps, 17020 + bits)
    train_one("recovery-q1-branch08", 1, .08,
              capped.table.detach().clone(), args.qat_steps, 17031)
    seeds = (10611, 10623, 10647, 10651, 10667, 10679)
    raw, variants = [], {}
    for name, model in models.items():
        result = {"heldout": score(model, held_data, y_held),
                  "contexts": context_score(model, held_data, parent),
                  "stop_budget": budget_report(model, held_data, y_held),
                  "random_10pct": cohort(model, name, seeds, out, .10, raw),
                  "no_random_diagnostic": cohort(model, name, seeds, out, 0., raw),
                  "same_runner_latency": timing(model, held[0], n=96)}
        if name in trained:
            net, trace, cap, support = trained[name]
            result.update({"bits": net.bits, "branch_probability_cap": cap,
                           "updates": len(trace), "support": support,
                           "packed_character_bytes": net.payload()["bytes"],
                           "cumulative_correction_bytes": 41 + net.payload()["bytes"]})
        variants[name] = result

    (out / "raw_decisions.json").write_text(json.dumps(raw, separators=(",", ":")) + "\n")
    report = {"schema": "ql017-separated-recovery-branch-v1",
              "source_sha": os.environ.get("QL_SOURCE_SHA", os.environ.get("GITHUB_SHA", "local")),
              **provenance,
              "ql016_source_sha": SOURCE_016,
              "ql016_manifest": manifest16,
              "ql016_checkpoint_sha256": inherited_hash,
              "train_n": len(train_set[1]), "heldout_n": len(held[1]),
              "optimizer_updates": 2 * args.fp_steps + 4 * args.qat_steps,
              "seeds": seeds, "variants": variants,
              "first_pass_trajectories": len(raw),
              "original_png_n": len(list((out / "gallery").glob("*.png"))),
              "raw_decisions_sha256": digest(out / "raw_decisions.json"),
              "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "limitations": [
                  "Synthetic W0 game-host worlds, not native CyberSand or Workbench.",
                  "Frozen QL-003, QL-014 and QL-015 parent models and QL-016 specimen.",
                  "Conditional RETRACE preservation uses an original-parent action reference, not a human aesthetic judgement.",
                  "BRANCH rate cap is a differentiable training penalty, not a guarantee or host legality rule.",
                  "STOP legality and budget STOP correction are unchanged.",
                  "No GGUF IQ1/IQ2/IQ3 codec: true small-table Q1/Q2/Q4 bitpacks decode for FP32 execution.",
                  "Same seed means same starting world and host RNG, not same states after policies diverge.",
                  "RSS measured for shared process, not independently isolated per-model.",
                  "No paid APIs, GPUs, gameplay code or production provider modifications."]}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    entries = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
                "sha256": digest(p)} for p in sorted(out.rglob("*"))
               if p.is_file() and p.name != "MANIFEST.json"]
    (out / "MANIFEST.json").write_text(json.dumps(entries, indent=2) + "\n")
    summary = {n: {"correct": s["heldout"]["observable_correct"],
                   "recovery_retraces": s["contexts"]["parent_retrace"]["retrace"],
                   "fresh_branches": s["contexts"]["new_ground"]["branch"],
                   "cells": s["random_10pct"]["mean_unique_cells"],
                   "rollout_retrace": s["random_10pct"]["total_retraces"],
                   "rollout_branch": s["random_10pct"]["total_branches"],
                   "early_stop": s["stop_budget"]["nonterminal"]["greedy_stop"]}
               for n, s in variants.items()}
    print(json.dumps({"status": "QL017_COMPLETE", "source_sha": report["source_sha"],
                      "updates": report["optimizer_updates"], "png": report["original_png_n"],
                      "results": summary}, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="ql003-source")
    ap.add_argument("--ql014", default="ql014-source")
    ap.add_argument("--ql015", default="ql015-source")
    ap.add_argument("--ql016", default="ql016-source")
    ap.add_argument("--out", default="ql017-out")
    ap.add_argument("--fp-steps", type=int, default=64)
    ap.add_argument("--qat-steps", type=int, default=48)
    ap.add_argument("--self-test", action="store_true")
    arguments = ap.parse_args()
    self_test() if arguments.self_test else run(arguments)
