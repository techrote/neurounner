#!/usr/bin/env python3
"""QL-015: budget-aware STOP repair on immutable QL-014 Q4/Q1 worm.

QL-014's novelty-only trained Q4 table chose STOP on 560/4352 held-out
states. This research-only continuation trains a 3x2 STOP-vs-continue
residual, in FP32 and actual Q4/Q2/Q1 straight bitpacking with STE QAT.
The two pre-existing low-bit parents stay frozen. STOP remains host-legal
at tick zero; no legality menu is changed. This is not a GGUF IQ codec.
"""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_worm import ACTIONS, digest, timing
from ql002_branch_retrace import examples
from ql005_bitserial import verified_source
from ql007_mixed_bridges import Hybrid, PARENT_SHA
from ql011_observable_repair import observable_targets
from ql013_contextual_retrace import cohort
from ql014_frontier_qat import FrontierTable, pack_codes, unpack_codes, heldout

STOP = ACTIONS.index("STOP")
SOURCE_014 = "eed1514a5b2c93b7a22838134ed40a56a63ee100"
TARGET_014 = "ql014-frontier-bitpack-qat-exact-head"


def budget_stage(x):
    """W0 observed feature[19]=clamped tick/64, feature[20]=clamped ink/64.

    0: early (<24 turns and <32 painted); 1: later pre-terminal;
    2: observed 40-turn or 52-ink deadline. No invented game-side inputs.
    """
    assert x.ndim == 2 and x.shape[1] == 27
    tick, ink = x[:, 19] * 64., x[:, 20] * 64.
    terminal = (tick >= 39.999) | (ink >= 51.999)
    late = (tick >= 24.) | (ink >= 32.)
    return torch.where(terminal, 2, torch.where(late, 1, 0)).long()


def packed_payload(table, bits):
    a = np.asarray(table, dtype=np.float32)
    assert a.shape == (3, 2) and bits in (1, 2, 4)
    scale = np.max(np.abs(a), axis=1).astype(np.float32)
    safe = np.where(scale > 0, scale, 1.).astype(np.float32)
    levels = (1 << bits) - 1
    code = np.clip(np.rint((np.clip(a / safe[:, None], -1., 1.) + 1.) *
                           levels / 2.), 0, levels).astype(np.uint8)
    code[scale == 0.] = 0
    packed = np.frombuffer(pack_codes(code, bits), dtype=np.uint8).copy()
    rebuilt = unpack_codes(packed.tobytes(), bits, (3, 2))
    assert np.array_equal(rebuilt, code)
    decoded = ((rebuilt.astype(np.float32) * (2. / levels)) - 1.) * scale[:, None]
    decoded[scale == 0.] = 0.
    assert len(packed) == (6 * bits + 7) // 8
    return {"bits": bits, "bytes": int(len(packed) + scale.nbytes),
            "packed": packed, "scales": scale, "decoded": decoded}


class BudgetResidual(nn.Module):
    """A 6-weight, stage-conditioned STOP/continue logit island."""
    def __init__(self, base, bits=32, init=None):
        super().__init__()
        self.base = base
        self.bits = bits
        self.table = nn.Parameter(torch.zeros(3, 2))
        if init is not None:
            with torch.no_grad():
                self.table.copy_(init)

    def quant_table(self):
        a = self.table
        if self.bits == 32:
            return a
        scales = a.detach().abs().amax(1, keepdim=True)
        safe = scales.clamp_min(1e-6)
        levels = (1 << self.bits) - 1
        rounded = (((a / safe).clamp(-1., 1.) + 1.) * levels / 2.).round()
        dq = (rounded * (2. / levels) - 1.) * scales
        dq = torch.where(scales > 0, dq, torch.zeros_like(dq))
        return a + (dq - a).detach()

    def correction(self, x):
        chosen = self.quant_table()[budget_stage(x)]
        # One value added to each non-STOP class; second value only to STOP.
        return torch.cat((chosen[:, :1].expand(-1, STOP), chosen[:, 1:2]), dim=1)

    def forward(self, x):
        with torch.no_grad():
            old = self.base(x)
        return old + self.correction(x)

    def payload(self):
        if self.bits == 32:
            a = self.table.detach().numpy().astype(np.float32)
            return {"bits": 32, "bytes": a.nbytes, "table": a}
        p = packed_payload(self.table.detach().numpy(), self.bits)
        assert np.allclose(p["decoded"], self.quant_table().detach().numpy(),
                           rtol=0, atol=4e-7)
        return p


class FixedBudgetGuard:
    """Untrained ablation: shift STOP, retaining the original Q4 action order."""
    def __init__(self, base):
        self.base = base

    def __call__(self, x):
        v = self.base(x).clone()
        stage = budget_stage(x)
        adjustment = torch.where(stage == 2, 3.,
                                 torch.where(stage == 1, -3., -4.))
        v[:, STOP] += adjustment
        return v


def fit(model, data, oracle, updates, lr, seed):
    x, _, legal = data
    stage = budget_stage(x)
    torch.manual_seed(seed)
    with torch.no_grad():
        base = model.base(x).detach()
        base_dist = base.masked_fill(~legal, -1e9).softmax(1)
    stage_counts = torch.bincount(stage, minlength=3).float().clamp_min(1)
    # Stratify across observed budget stages. Report actual support separately.
    p = (1. / stage_counts[stage]).double()
    p /= p.sum()
    opt = torch.optim.AdamW([model.table], lr=lr, weight_decay=0.)
    trace = []
    for n in range(updates):
        ix = torch.multinomial(p, 256, replacement=True)
        z = (base[ix] + model.correction(x[ix])).masked_fill(~legal[ix], -1e9)
        ce = F.cross_entropy(z, oracle[ix])
        kl = F.kl_div(F.log_softmax(z, dim=1), base_dist[ix],
                      reduction="batchmean")
        alternatives = z[:, :STOP].masked_fill(~legal[ix, :STOP], -1e9)
        best_other = alternatives.max(1).values
        have_other = legal[ix, :STOP].any(1)
        too_early = (stage[ix] < 2) & have_other
        deadline = (stage[ix] == 2) & have_other
        early_loss = (F.relu(z[too_early, STOP] - best_other[too_early] + .35).mean()
                      if bool(too_early.any()) else z[:, STOP].sum() * 0.)
        deadline_loss = (F.relu(best_other[deadline] - z[deadline, STOP] + .35).mean()
                         if bool(deadline.any()) else z[:, STOP].sum() * 0.)
        loss = (ce + .22 * kl + .9 * early_loss + 1.1 * deadline_loss
                + .004 * model.quant_table().square().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append({"step": n + 1, "loss": float(loss.detach()),
                      "ce": float(ce.detach()), "kl": float(kl.detach()),
                      "early_stop_margin": float(early_loss.detach()),
                      "deadline_stop_margin": float(deadline_loss.detach())})
    state = opt.state_dict()
    assert len(state["state"]) == 1
    assert bool(torch.count_nonzero(next(iter(state["state"].values()))["exp_avg"]))
    assert torch.isfinite(model.table).all()
    return state, trace


def budget_report(model, data, oracle):
    x, _, legal = data
    with torch.no_grad():
        v = model(x).masked_fill(~legal, -1e9)
        selected = v.argmax(1)
    out = {}
    buckets = budget_stage(x)
    for phase, mask in (("early", buckets == 0), ("later", buckets == 1),
                        ("terminal", buckets == 2), ("nonterminal", buckets < 2)):
        n = int(mask.sum())
        out[phase] = {
            "n": n,
            "oracle_stop": int(((oracle == STOP) & mask).sum()),
            "greedy_stop": int(((selected == STOP) & mask).sum()),
            "stop_true_positive": int(((selected == STOP) & (oracle == STOP) & mask).sum()),
            "observable_correct": int(((selected == oracle) & mask).sum()),
            "host_masked_legal": int(legal[mask].gather(
                1, selected[mask, None]).sum()),
        }
    assert sum(out[k]["n"] for k in ("early", "later", "terminal")) == len(x)
    return out


def verify_ql014(root, q1, q4):
    r = json.loads((root / "results.json").read_text())
    assert r["source_sha"] == SOURCE_014
    assert r["parent_source_sha"] == PARENT_SHA
    assert r["parent_q1_sha256"] == digest(q1)
    assert r["parent_q4_sha256"] == digest(q4)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert len(manifest) >= 225
    for entry in manifest:
        p = root / entry["path"]
        assert p.is_file() and p.stat().st_size == entry["bytes"]
        assert digest(p) == entry["sha256"]
    checkpoint = root / "models" / "qat-q4-table-optimizer.pt"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ck["source_sha"] == SOURCE_014 and ck["updates"] == 48
    assert ck["bits"] == 4 and ck["table_fp32"].shape == (4, len(ACTIONS))
    assert ck["parent_q1_sha256"] == digest(q1)
    assert ck["parent_q4_sha256"] == digest(q4)
    return ck, len(manifest), digest(checkpoint)


def self_test():
    torch.manual_seed(15015)
    x = torch.zeros(4, 27)
    x[1, 19] = 30 / 64
    x[2, 19] = 41 / 64
    x[3, 20] = 53 / 64
    assert budget_stage(x).tolist() == [0, 1, 2, 2]
    class Dummy:
        def __call__(self, features):
            return torch.zeros((len(features), len(ACTIONS)))
    model = BudgetResidual(Dummy())
    with torch.no_grad():
        model.table.copy_(torch.tensor([[.5, -.5], [0., 0.], [-.5, .5]]))
    assert torch.allclose(model.correction(x)[0, :STOP], torch.full((STOP,), .5))
    assert float(model.correction(x)[0, STOP]) == -.5
    assert float(model.correction(x)[2, STOP]) == .5
    for bits in (1, 2, 4):
        for a in (torch.randn(3, 2), torch.zeros(3, 2)):
            net = BudgetResidual(Dummy(), bits, a)
            p = net.payload()
            assert len(p["packed"]) == (6 * bits + 7) // 8
            assert p["bytes"] == len(p["packed"]) + 12
            assert np.isfinite(p["decoded"]).all()
            assert np.allclose(p["decoded"], net.quant_table().detach().numpy(),
                               rtol=0, atol=4e-7)
            assert net(x).shape == (4, len(ACTIONS))
    guard = FixedBudgetGuard(Dummy())(x)
    assert guard[:, STOP].tolist() == [-4., -3., 3., 3.]
    print("QL015 SELF-TEST PASS: observed budgets, STOP split, Q1 Q2 Q4 packs, zero rows")


def run(args):
    self_test()
    torch.set_num_threads(1)
    torch.manual_seed(15015)
    np.random.seed(15015)
    root = Path(args.out)
    (root / "gallery").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    _, checked = verified_source(Path(args.parent), PARENT_SHA)
    q1 = Path(args.parent) / "models" / "bal-q1.npz"
    q4 = Path(args.parent) / "models" / "bal-q4.npz"
    ck, old_manifest, old_ck_sha = verify_ql014(Path(args.ql014), q1, q4)
    parent = Hybrid(q1, q4, "b")
    old = FrontierTable(parent, 4, ck["table_fp32"])
    old.table.requires_grad_(False)
    train, _, bad = examples(range(2100, 2180), 33102)
    held, _, bad_held = examples(range(9100, 9132), 33103)
    assert bad == bad_held == 0 and len(held[1]) == 4352
    oracle_train = observable_targets(train)
    oracle_held = observable_targets(held)
    train_stages = torch.bincount(budget_stage(train[0]), minlength=3).tolist()
    held_stages = torch.bincount(budget_stage(held[0]), minlength=3).tolist()
    assert min(train_stages) > 0 and min(held_stages) > 0, (train_stages, held_stages)
    models = {"original-q1-q4-middle": parent,
              "ql014-frozen-q4-table": old,
              "ql014-q4-fixed-budget": FixedBudgetGuard(old)}
    trained = {}
    fp = BudgetResidual(old)
    state, trace = fit(fp, train, oracle_train, args.fp_steps, .075, 15001)
    trained["budget-fp32"] = (fp, state, trace)
    models["budget-fp32"] = fp
    for bits in (4, 2, 1):
        net = BudgetResidual(old, bits, fp.table.detach().clone())
        opt, log = fit(net, train, oracle_train, args.qat_steps, .035,
                       15000 + bits)
        name = "budget-qat-q" + str(bits)
        trained[name] = (net, opt, log)
        models[name] = net

    for name, (net, optimizer, history) in trained.items():
        payload = net.payload()
        torch.save({"source_sha": os.environ.get("QL_SOURCE_SHA", "local"),
                    "ql014_source_sha": SOURCE_014,
                    "ql014_checkpoint_sha256": old_ck_sha,
                    "parent_source_sha": PARENT_SHA,
                    "parent_q1_sha256": digest(q1),
                    "parent_q4_sha256": digest(q4),
                    "bits": net.bits, "updates": len(history),
                    "table_fp32": net.table.detach().clone(),
                    "optimizer": optimizer, "trace": history,
                    "packed_payload": payload},
                   root / "models" / (name + "-optimizer.pt"))
        if net.bits != 32:
            np.savez_compressed(root / "models" / (name + "-packed.npz"),
                                bits=np.array(net.bits), packed=payload["packed"],
                                scales=payload["scales"])
    seeds = (10411, 10423, 10447, 10451, 10467, 10479)
    raw = []
    scores = {}
    for name, model in models.items():
        info = {"heldout": heldout(model, held, oracle_held, parent),
                "budget_conditional": budget_report(model, held, oracle_held),
                "host_random_10pct": cohort(model, name, seeds, root, .10, raw),
                "no_random_diagnostic": cohort(model, name, seeds, root, 0., raw),
                "same_runner_latency": timing(model, held[0], n=96)}
        if name in trained:
            info["trained_updates"] = len(trained[name][2])
            info["bits"] = model.bits
            info["budget_payload_bytes"] = model.payload()["bytes"]
            info["frozen_ql014_q4_payload_bytes"] = 28
            info["residual_plus_ql014_payload_bytes"] = 28 + model.payload()["bytes"]
        scores[name] = info
    (root / "raw_decisions.json").write_text(
        json.dumps(raw, separators=(",", ":")) + "\n")
    report = {
        "schema": "ql015-budget-conditioned-stop-v1",
        "source_sha": os.environ.get("QL_SOURCE_SHA",
                                      os.environ.get("GITHUB_SHA", "local")),
        "parent_source_sha": PARENT_SHA,
        "parent_q1_sha256": digest(q1),
        "parent_q4_sha256": digest(q4),
        "parent_manifest_entries": checked,
        "ql014_source_sha": SOURCE_014,
        "ql014_manifest_entries": old_manifest,
        "ql014_checkpoint_sha256": old_ck_sha,
        "train_n": len(train[1]), "heldout_n": len(held[1]),
        "train_budget_stage_counts": train_stages,
        "held_budget_stage_counts": held_stages,
        "optimizer_updates": args.fp_steps + 3 * args.qat_steps,
        "seeds": seeds, "variants": scores,
        "original_png_n": len(list((root / "gallery").glob("*.png"))),
        "first_pass_trajectories": len(raw),
        "raw_decisions_sha256": digest(root / "raw_decisions.json"),
        "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": [
            "All W0 examples and game trajectories synthetic, not native Godot.",
            "Budget-stage inputs already exist in the observed W0 27-feature vector.",
            "Only a six-scalar STOP-vs-continue QAT residual; no general model IQ codec.",
            "Bitpacked codes decoded for CPU NumPy/PyTorch reference; no low-bit speedup.",
            "Host legality untouched; STOP always legal even when policy discourages it.",
            "Host random overrides can truncate any model, hence 0%-override control.",
            "The frozen QL-014 Q4 baseline is a distinct preceding experiment.",
            "No user aesthetic ranking or native/Workbench acceptance.",
        ],
    }
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                 "sha256": digest(p)} for p in sorted(root.rglob("*"))
                if p.is_file() and p.name != "MANIFEST.json"]
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "QL015_COMPLETE", "source_sha": report["source_sha"],
                      "updates": report["optimizer_updates"],
                      "pngs": report["original_png_n"],
                      "results": {n: {"correct": r["heldout"]["observable_correct"],
                                    "stop": r["heldout"]["greedy_stop"],
                                    "early_stop": r["budget_conditional"]["nonterminal"]["greedy_stop"],
                                    "cells": r["host_random_10pct"]["mean_unique_cells"],
                                    "retrace": r["host_random_10pct"]["total_retraces"]}
                                  for n, r in scores.items()}}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="ql003-source")
    parser.add_argument("--ql014", default="ql014-source")
    parser.add_argument("--out", default="ql015-out")
    parser.add_argument("--fp-steps", type=int, default=64)
    parser.add_argument("--qat-steps", type=int, default=48)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else run(args)
