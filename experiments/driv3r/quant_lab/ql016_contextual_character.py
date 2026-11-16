#!/usr/bin/env python3
"""QL-016: recover contextual RETRACE/BRANCH character on QL-015's frozen Q1 STOP repair.

Research-only synthetic W0 actions. The existing QL-003 Q1/Q4 middle policy,
QL-014 Q4 frontier and QL-015 Q1 budget STOP corrector remain immutable.
Learn a 4 x 2 x 2 x 2 action-logit island (novelty, path-depth, previous
RETRACE, RETRACE/BRANCH) with FP32, true packed Q4/Q2/Q1 STE and sparse Q1.
No game-side modifications, GGUF codec, or low-bit speedup claims.
"""
import argparse
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
from ql013_contextual_retrace import novelty_mask, cohort
from ql014_frontier_qat import FrontierTable, pack_codes, unpack_codes
from ql015_budget_stop import BudgetResidual, budget_stage, budget_report

R = ACTIONS.index("RETRACE")
B = ACTIONS.index("BRANCH")
S = ACTIONS.index("STOP")
SOURCE_014 = "eed1514a5b2c93b7a22838134ed40a56a63ee100"
SOURCE_015 = "40df4242f22f908f509f808c491598f493fa2222"
CK_015_Q1 = "b722436434f869ffb1a37eec88259daaa24c7ef5c580dfaab4156511271937e0"
# 4 novelty counts, two path-depth strata, previous-RETRACE flag, two actions.
SHAPE = (4, 2, 2, 2)
ROWS = 4
SIZE = 32


def bucket(x):
    """Only existing W0 features: fresh slots 3,7,11; path length 21; last RETRACE 26."""
    fresh = novelty_mask(x).long()
    deep = (x[:, 21] * 64 >= 6 - 1e-4).long()
    prior_retrace = (x[:, 26] > .5).long()
    assert bool(((fresh >= 0) & (fresh <= 3)).all())
    return fresh, deep, prior_retrace


class CharacterIsland(nn.Module):
    """Frozen model + conditional correction to RETRACE/BRANCH only.

    The table is not applied at the observed terminal budget. STOP logits
    are never adjusted. Host masks continue to own all legality.
    """
    def __init__(self, base, bits=32, init=None, suppress_chain=False):
        super().__init__()
        self.base = base
        self.bits = bits
        self.suppress_chain = suppress_chain
        self.table = nn.Parameter(torch.zeros(SHAPE))
        if init is not None:
            with torch.no_grad():
                self.table.copy_(init)
        mask = torch.ones(SHAPE)
        if suppress_chain:
            mask[:, :, 1, :] = 0.
        self.register_buffer("mask", mask, persistent=False)

    def quant_table(self):
        a = self.table * self.mask
        if self.bits == 32:
            return a
        scales = a.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
        safe = scales.clamp_min(1e-6)
        levels = (1 << self.bits) - 1
        codes = (((a / safe).clamp(-1., 1.) + 1.) * levels / 2.).round()
        decoded = (codes * (2. / levels) - 1.) * scales
        decoded = torch.where(scales > 0, decoded, torch.zeros_like(decoded))
        return (a + (decoded - a).detach()) * self.mask

    def correction(self, x):
        fresh, deep, prior = bucket(x)
        chosen = self.quant_table()[fresh, deep, prior]
        active = (budget_stage(x) != 2).to(x.dtype)[:, None]
        out = torch.zeros((len(x), len(ACTIONS)), dtype=x.dtype, device=x.device)
        # cat is differentiable; in-place insertion into a fresh tensor also
        # keeps a differentiable CopySlices path for the two action columns.
        out[:, R] = chosen[:, 0] * active[:, 0]
        out[:, B] = chosen[:, 1] * active[:, 0]
        return out

    def forward(self, x):
        with torch.no_grad():
            base = self.base(x)
        return base + self.correction(x)

    def payload(self):
        a = (self.table.detach() * self.mask).numpy().astype(np.float32)
        if self.bits == 32:
            return {"bits": 32, "bytes": int(a.nbytes), "table": a}
        scales = np.max(np.abs(a), axis=(1, 2, 3)).astype(np.float32)
        safe = np.where(scales == 0, 1., scales)
        levels = (1 << self.bits) - 1
        codes = np.clip(np.rint((np.clip(a / safe[:, None, None, None], -1., 1.) + 1.)
                                 * levels / 2.), 0, levels).astype(np.uint8)
        codes[scales == 0] = 0
        packed = pack_codes(codes, self.bits)
        rebuilt = unpack_codes(packed, self.bits, SHAPE)
        assert np.array_equal(rebuilt, codes)
        decoded = ((rebuilt.astype(np.float32) * (2. / levels)) - 1.) * scales[:, None, None, None]
        decoded[scales == 0] = 0.
        decoded *= self.mask.numpy()
        assert np.allclose(decoded, self.quant_table().detach().numpy(), atol=6e-7, rtol=0)
        return {"bits": self.bits, "bytes": len(packed) + scales.nbytes,
                "packed": np.frombuffer(packed, dtype=np.uint8).copy(),
                "scales": scales, "decoded": decoded}


def verify_archive(folder, expected_source, min_files):
    result = json.loads((folder / "results.json").read_text())
    assert result["source_sha"] == expected_source
    entries = json.loads((folder / "MANIFEST.json").read_text())
    assert len(entries) >= min_files
    for item in entries:
        p = folder / item["path"]
        assert p.is_file() and p.stat().st_size == item["bytes"] and digest(p) == item["sha256"]
    return result, len(entries)


def load_immutable_models(parent_dir, ql014, ql015):
    _, parent_count = verified_source(parent_dir, PARENT_SHA)
    q1 = parent_dir / "models" / "bal-q1.npz"
    q4 = parent_dir / "models" / "bal-q4.npz"
    result14, files14 = verify_archive(ql014, SOURCE_014, 225)
    result15, files15 = verify_archive(ql015, SOURCE_015, 261)
    for record in (result14, result15):
        assert record["parent_q1_sha256"] == digest(q1)
        assert record["parent_q4_sha256"] == digest(q4)
    assert result15["ql014_source_sha"] == SOURCE_014
    original = Hybrid(q1, q4, "b")
    ck14file = ql014 / "models" / "qat-q4-table-optimizer.pt"
    assert digest(ck14file) == result15["ql014_checkpoint_sha256"]
    ck14 = torch.load(ck14file, map_location="cpu", weights_only=False)
    assert ck14["source_sha"] == SOURCE_014 and ck14["bits"] == 4
    frontier = FrontierTable(original, 4, ck14["table_fp32"])
    frontier.table.requires_grad_(False)
    ck15file = ql015 / "models" / "budget-qat-q1-optimizer.pt"
    assert digest(ck15file) == CK_015_Q1
    ck15 = torch.load(ck15file, map_location="cpu", weights_only=False)
    assert ck15["source_sha"] == SOURCE_015 and ck15["bits"] == 1 and ck15["updates"] == 48
    assert ck15["ql014_checkpoint_sha256"] == digest(ck14file)
    budget = BudgetResidual(frontier, 1, ck15["table_fp32"])
    budget.table.requires_grad_(False)
    assert budget.payload()["bytes"] == 13 and frontier.payload()["bytes"] == 28
    return original, budget, {"parent_manifest": parent_count,
                              "ql014_manifest": files14, "ql015_manifest": files15,
                              "parent_q1_sha256": digest(q1),
                              "parent_q4_sha256": digest(q4),
                              "ql014_checkpoint_sha256": digest(ck14file),
                              "ql015_checkpoint_sha256": digest(ck15file)}


def fit(net, data, labels, original, updates, seed, personality):
    x, _, legal = data
    torch.manual_seed(seed)
    with torch.no_grad():
        base = net.base(x).detach()
        orig = original(x).masked_fill(~legal, -1e9).argmax(1)
        prior = base.masked_fill(~legal, -1e9).softmax(1)
    fresh, deep, last = bucket(x)
    eligible_r = (legal[:, R] & (fresh <= 1) & (deep == 1) &
                  (last == 0) & (orig == R) & (budget_stage(x) < 2))
    eligible_b = (legal[:, B] & (fresh >= 2) & (labels == B) &
                  (budget_stage(x) < 2))
    negatives = (legal[:, R] & (fresh >= 2) & (labels != R) &
                 (budget_stage(x) < 2))
    r_ids, b_ids, n_ids = (torch.where(v)[0] for v in
                           (eligible_r, eligible_b, negatives))
    assert len(r_ids) > 0 and len(b_ids) > 0 and len(n_ids) > 0, (
        len(r_ids), len(b_ids), len(n_ids))
    counts = torch.bincount(labels, minlength=len(ACTIONS)).float().clamp_min(1)
    probs = (1. / counts[labels]).double()
    probs /= probs.sum()
    opt = torch.optim.AdamW([net.table], lr=.035 if net.bits == 32 else .02,
                             weight_decay=0.)
    trace = []
    for step in range(updates):
        ids = torch.multinomial(probs, 192, replacement=True)
        z = (base[ids] + net.correction(x[ids])).masked_fill(~legal[ids], -1e9)
        ce = F.cross_entropy(z, labels[ids])
        kl = F.kl_div(F.log_softmax(z, dim=1), prior[ids], reduction="batchmean")
        # Identical RNG draws in both objectives, even when personality disabled.
        ri = r_ids[torch.randint(len(r_ids), (96,))]
        bi = b_ids[torch.randint(len(b_ids), (96,))]
        ni = n_ids[torch.randint(len(n_ids), (96,))]
        rl = (base[ri] + net.correction(x[ri])).masked_fill(~legal[ri], -1e9)
        bl = (base[bi] + net.correction(x[bi])).masked_fill(~legal[bi], -1e9)
        nl = (base[ni] + net.correction(x[ni])).masked_fill(~legal[ni], -1e9)
        keep_retrace = F.cross_entropy(rl, torch.full((len(ri),), R, dtype=torch.long))
        prefer_branch = F.cross_entropy(bl, torch.full((len(bi),), B, dtype=torch.long))
        avoid_retrace = F.relu(nl[:, R] - nl.masked_fill(
            F.one_hot(torch.full((len(ni),), R), len(ACTIONS)).bool(), -1e9).max(1).values
                                    + .2).mean()
        # Unlike the old global RETRACE anchors, these apply only to actual
        # observed recoverable subgroups. Explicit STOP budget policy stays frozen.
        loss = (ce + .28 * kl + (.60 * keep_retrace + .36 * prefer_branch
                + .24 * avoid_retrace if personality else 0.)
                + .002 * net.quant_table().square().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append({"step": step + 1, "loss": float(loss.detach()),
                      "ce": float(ce.detach()), "kl": float(kl.detach()),
                      "keep_retrace": float(keep_retrace.detach()),
                      "prefer_branch": float(prefer_branch.detach()),
                      "avoid_retrace": float(avoid_retrace.detach())})
    state = opt.state_dict()
    assert len(state["state"]) == 1
    assert bool(torch.count_nonzero(next(iter(state["state"].values()))["exp_avg"]))
    assert torch.isfinite(net.table).all()
    return state, trace, {"retrace_anchors": len(r_ids), "branch_targets": len(b_ids),
                          "novelty_negatives": len(n_ids)}


def score(model, data, target):
    x, _, legal = data
    fresh, deep, last = bucket(x)
    with torch.no_grad():
        logits = model(x)
        choice = logits.masked_fill(~legal, -1e9).argmax(1)
        unmasked = logits.argmax(1)
    subsets = {"exhausted_recoverable": (fresh == 0) & legal[:, R],
               "low_novelty_recoverable": (fresh <= 1) & legal[:, R] & (deep == 1),
               "fresh_branch_legal": (fresh >= 2) & legal[:, B],
               "post_retrace_fresh": (last == 1) & (fresh > 0) & legal[:, R],
               "preterminal": budget_stage(x) < 2,
               "terminal": budget_stage(x) == 2}
    counts = {}
    for name, mask in subsets.items():
        n = int(mask.sum())
        counts[name] = {"n": n, "target_correct": int(((choice == target) & mask).sum()),
                        "retrace": int(((choice == R) & mask).sum()),
                        "branch": int(((choice == B) & mask).sum()),
                        "stop": int(((choice == S) & mask).sum())}
    return {"n": len(x), "observable_correct": int((choice == target).sum()),
            "host_legal": int(legal.gather(1, choice[:, None]).sum()),
            "raw_top_legal": int(legal.gather(1, unmasked[:, None]).sum()),
            "retrace": int((choice == R).sum()), "branch": int((choice == B).sum()),
            "stop": int((choice == S).sum()), "subsets": counts}


def self_test():
    class Dummy:
        def __call__(self, x):
            return torch.zeros((len(x), len(ACTIONS)))
    x = torch.zeros(5, 27)
    x[1, 3] = 1
    x[2, [3, 7]] = 1
    x[3, [3, 7, 11]] = 1
    x[1, 21] = 10 / 64
    x[2, 26] = 1
    x[4, 19] = 40 / 64
    fresh, deep, last = bucket(x)
    assert fresh.tolist() == [0, 1, 2, 3, 0]
    assert deep.tolist() == [0, 1, 0, 0, 0]
    assert last.tolist() == [0, 0, 1, 0, 0]
    m = CharacterIsland(Dummy(), 32)
    with torch.no_grad():
        m.table[0, 0, 0] = torch.tensor([2., 3.])
    a = m(x)
    assert a[0, R] == 2 and a[0, B] == 3
    assert a[4].abs().sum() == 0
    assert a[:, S].abs().sum() == 0
    for bits in (1, 2, 4):
        for random_table in (torch.randn(SHAPE), torch.zeros(SHAPE)):
            test = CharacterIsland(Dummy(), bits, random_table, suppress_chain=True)
            payload = test.payload()
            assert payload["bytes"] == (SIZE * bits + 7) // 8 + 16
            assert np.isfinite(payload["decoded"]).all()
            assert unpack_codes(payload["packed"].tobytes(), bits, SHAPE).shape == SHAPE
            assert torch.allclose(test.quant_table()[:, :, 1], torch.zeros(4, 2, 2))
            assert test(x).shape == (5, len(ACTIONS))
    print("QL016 SELF-TEST PASS: observed character contexts, terminal STOP freeze, packed Q1/Q2/Q4")


def run(args):
    self_test()
    torch.set_num_threads(1)
    torch.manual_seed(16016)
    np.random.seed(16016)
    out = Path(args.out)
    (out / "gallery").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    original, base, provenance = load_immutable_models(
        Path(args.parent), Path(args.ql014), Path(args.ql015))
    train, _, bad = examples(range(2100, 2180), 33102)
    held, _, held_bad = examples(range(9100, 9132), 33103)
    assert bad == held_bad == 0 and len(held[1]) == 4352
    target_train, target_held = observable_targets(train), observable_targets(held)
    models = {"original-q1-q4middle": original, "ql015-budget-q1": base}
    trained = {}
    def train_one(name, bits, init, sparse, personality, steps, seed):
        net = CharacterIsland(base, bits, init, suppress_chain=sparse)
        state, log, support = fit(net, train, target_train, original, steps, seed, personality)
        models[name] = net
        trained[name] = (net, state, log, support, personality)
        payload = net.payload()
        torch.save({"source_sha": os.environ.get("QL_SOURCE_SHA", "local"),
                    "ql015_source_sha": SOURCE_015,
                    "ql015_checkpoint_sha256": provenance["ql015_checkpoint_sha256"],
                    "ql014_checkpoint_sha256": provenance["ql014_checkpoint_sha256"],
                    "parent_source_sha": PARENT_SHA, "bits": bits, "sparse": sparse,
                    "personality_loss": personality, "updates": len(log),
                    "table_fp32": net.table.detach().clone(),
                    "optimizer": state, "trace": log, "support": support,
                    "packed_payload": payload},
                   out / "models" / (name + "-optimizer.pt"))
        if bits != 32:
            np.savez_compressed(out / "models" / (name + "-packed.npz"),
                                bits=np.array(bits), packed=payload["packed"],
                                scales=payload["scales"], sparse=np.array(sparse))
        return net

    train_one("ce-fp32", 32, None, False, False, args.fp_steps, 16001)
    personality = train_one("personality-fp32", 32, None, False, True, args.fp_steps, 16002)
    for bits in (4, 2, 1):
        train_one("personality-q" + str(bits), bits,
                  personality.table.detach().clone(), False, True,
                  args.qat_steps, 16020 + bits)
    train_one("sparse-q1-no-chain", 1, personality.table.detach().clone(),
              True, True, args.qat_steps, 16031)

    seeds = (10511, 10523, 10547, 10551, 10567, 10579)
    raw, scored = [], {}
    for name, model in models.items():
        info = {"heldout": score(model, held, target_held),
                "budget": budget_report(model, held, target_held),
                "random_10pct": cohort(model, name, seeds, out, .10, raw),
                "no_random_diagnostic": cohort(model, name, seeds, out, 0., raw),
                "same_runner_latency": timing(model, held[0], n=96)}
        if name in trained:
            m, _, trace, support, personality_loss = trained[name]
            info.update({"bits": m.bits, "sparse": m.suppress_chain,
                         "personality_loss": personality_loss,
                         "updates": len(trace), "support": support,
                         "payload_bytes": m.payload()["bytes"],
                         "frozen_frontier_budget_bytes": 41,
                         "sum_residual_bytes": 41 + m.payload()["bytes"]})
        scored[name] = info

    (out / "raw_decisions.json").write_text(json.dumps(raw, separators=(",", ":")) + "\n")
    report = {"schema": "ql016-contextual-character-v1",
              "source_sha": os.environ.get("QL_SOURCE_SHA", os.environ.get("GITHUB_SHA", "local")),
              **provenance,
              "ql014_source_sha": SOURCE_014, "ql015_source_sha": SOURCE_015,
              "train_n": len(train[1]), "heldout_n": len(held[1]),
              "optimizer_updates": 2 * args.fp_steps + 4 * args.qat_steps,
              "seeds": seeds, "variants": scored, "original_png_n": len(list(
                  (out / "gallery").glob("*.png"))),
              "first_pass_trajectories": len(raw), "raw_decisions_sha256": digest(
                  out / "raw_decisions.json"),
              "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "limitations": [
                  "Synthetic W0 worlds, host menus, obstacles and marked cells, not native CyberSand.",
                  "Observable heuristic target is experimental; higher teacher accuracy is not aesthetic superiority.",
                  "Original packed Q1/Q4, QL-014 frontier, QL-015 budget-Q1 parameters remain frozen.",
                  "Every terminal STOP decision is explicitly insulated from the new correction.",
                  "No GGUF IQ3/IQ2/IQ1 or accelerated packed inference; packed table decoded in PyTorch.",
                  "The sparse Q1 specimen removes all previous-RETRACE correction rows, not backbone weights.",
                  "Host randomness sometimes STOPs early; zero-random diagnostic retained.",
                  "No paid APIs, GPU spend, production integrations, native code or default provider changes.",
              ]}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
                 "sha256": digest(p)} for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "MANIFEST.json"]
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "QL016_COMPLETE", "source_sha": report["source_sha"],
                      "updates": report["optimizer_updates"], "png": report["original_png_n"],
                      "results": {k: {"correct": v["heldout"]["observable_correct"],
                                      "coverage": v["random_10pct"]["mean_unique_cells"],
                                      "retrace": v["random_10pct"]["total_retraces"],
                                      "branch": v["random_10pct"]["total_branches"],
                                      "stop_before_deadline": v["budget"]["nonterminal"]["greedy_stop"]}
                                  for k, v in scored.items()}}, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="ql003-source")
    ap.add_argument("--ql014", default="ql014-source")
    ap.add_argument("--ql015", default="ql015-source")
    ap.add_argument("--out", default="ql016-out")
    ap.add_argument("--fp-steps", type=int, default=64)
    ap.add_argument("--qat-steps", type=int, default=48)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test() if args.self_test else run(args)
