#!/usr/bin/env python3
"""W17: calibrate false branch positives while retaining W16 history motifs.

Synthetic W0 game-policy research ONLY. Frozen root-Instruct/LoRA. Real AdamW
updates affect an auxiliary 142-feature/64x64 policy head; no native game data.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "training"), str(HERE)]
import worm_train_v9 as w9
import worm_train_v13 as w13
import worm_train_v14 as w14
import worm_train_v15 as w15
import worm_train_v16 as w16
import worm_head_sample_v11 as w11
import worm_v4 as objective
from common import profile, sha256_file, write_json
from numerics import configure_cpu

VERSION = "sprocket-w17-hard-negative-branch-history-calibration-v1"
W15_SHA = "a2b521f99a9bda90ad471d980aaf3b3372c69cbe54269c54885e145114addcba"
W16_SHA = "f5dc8482a145e39a0bfb10ba94617efad34dc957e3ff485afab3b13726fe90a9"
W14_SHA = w15.W14_SHA
W0_SHA = w15.W0_SHA
SUPPORT_SHA = w15.SUPPORT_SHA
TRAIN_SEEDS = (367, 373, 379, 383)
HELDOUT = (2707, 2711, 2729)
UPDATES_PER_WAVE = 160
ACTIONS = tuple(w9.ACTIONS)


def snapshot(torch, source):
    m = w9.model_class(torch)()
    m.load_state_dict(source.state_dict(), strict=True)
    m.eval()
    return m


def fidelity(torch, model, rows, pairs):
    return w15.eval_model(torch, model, rows, pairs)


def replay_data(torch, w0, support, parents):
    rows, receipts = [], []
    for seed in TRAIN_SEEDS:
        for family in w9.FAMILIES:
            for name, model, temp in parents:
                rec = w14.rollout(torch, w0, model, seed, family, name, temp)
                assert len(rec["events"]) == len(rec["head_decisions"])
                assert all(e["raw_legal"] for e in rec["events"])
                for e in rec["events"]:
                    obs = e["observation"]
                    target = w15.teacher(obs, support)
                    assert target in obs["legal"]
                    rows.append({"observation": obs, "action": target,
                                 "family": family, "seed": seed,
                                 "source": name, "legal": obs["legal"]})
                receipts.append({"seed": seed, "family": family,
                                 "controller": name, "steps": len(rec["events"]),
                                 "branches": rec["metrics"]["branches"],
                                 "retraces": rec["metrics"]["retraces"]})
    return rows, receipts


def measured_manifest(rows):
    raw = json.dumps([r["observation"] for r in rows],
                     sort_keys=True, separators=(",", ":")).encode()
    return {"rows": len(rows), "labels": dict(Counter(r["action"] for r in rows)),
            "sha256_observations": hashlib.sha256(raw).hexdigest(),
            "all_targets_host_legal": all(r["action"] in r["observation"]["legal"] for r in rows)}


def checkpoint(torch, model, opt, rng, out, label, evaluation):
    path = out / (label + ".pt")
    torch.save({"version": VERSION, "stage": label, "head": model.state_dict(),
                "optimizer": opt.state_dict(), "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.random.get_rng_state(),
                "evaluation": evaluation, "root_frozen": True,
                "root_optimizer_steps": 0}, path)
    return {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def train(torch, w0, support, h14, h15, h16, out):
    torch.manual_seed(171717)
    rng = random.Random(171717)
    rows, train_receipt = w15.worlds(w0, support, TRAIN_SEEDS)
    heldout, heldout_receipt = w15.worlds(w0, support, HELDOUT)
    pairs = w15.contrast_pairs(rows, support)
    heldout_pairs = w15.contrast_pairs(heldout, support)
    assert not set(TRAIN_SEEDS).intersection(HELDOUT)
    x, y = w9.as_tensors(torch, rows)
    mask = w13.legal_mask(torch, rows)
    groups = w13.class_rows(rows)
    pos, neg = w13.eligible_indices(rows)
    write_json(out / "datasets.json", {"version": VERSION,
        "train_worlds": train_receipt, "heldout_worlds": heldout_receipt,
        "train": measured_manifest(rows), "heldout": measured_manifest(heldout),
        "train_history_counterfactual_pairs": len(pairs),
        "heldout_history_counterfactual_pairs": len(heldout_pairs),
        "counterfactual_reachability_verified": False, "native": False,
        "source": "pinned-synthetic-W0, action teacher W15"})
    model = snapshot(torch, h16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.000085, weight_decay=0.003)
    history = []
    stages = [{"name": "frozen-W14", "evaluation": fidelity(torch, h14, heldout, heldout_pairs)},
              {"name": "frozen-W15", "evaluation": fidelity(torch, h15, heldout, heldout_pairs)},
              {"name": "frozen-W16", "evaluation": fidelity(torch, h16, heldout, heldout_pairs)}]
    stage1 = None
    for si, stage in enumerate(("hard-negative-branch-calibration", "closed-loop-history-retention")):
        if si:
            stage1 = snapshot(torch, model)
            replay, replay_receipts = replay_data(torch, w0, support,
                (("W14-sampled-t07", h14, 0.7),
                 ("W16-sampled-t07", h16, 0.7),
                 ("W17-stage1-sampled-t07", stage1, 0.7)))
            rx, ry = w9.as_tensors(torch, replay)
            rm = w13.legal_mask(torch, replay)
            write_json(out / "replay.json", {
                "real_closed_loop_synthetic_host_rollouts": replay_receipts,
                "examples": measured_manifest(replay),
                "labels": "synthetic W15 teacher; not game success feedback",
                "native": False})
        for k in range(UPDATES_PER_WAVE):
            opt.zero_grad(set_to_none=True)
            ix = w13.balanced(groups, rng)
            ce = torch.nn.functional.cross_entropy(
                model(x[ix]).masked_fill(~mask[ix], -1e4), y[ix])
            branch = w14.calibrated_branch_loss(torch, model, x, mask, pos, neg, rng)
            contrast = w15.pair_contrast(torch, model, pairs, rng, n=48)
            # W16's conditional BRANCH-positive margins carry useful knowledge.
            # Protect eligible BRANCH-positive decisions without overwriting
            # the W15 short-history contrastive gradients.
            bp = rng.choices(pos, k=40)
            with torch.no_grad():
                previous = h16(x[bp]).masked_fill(~mask[bp], -1e4)
                previous_branch = previous[:, ACTIONS.index("BRANCH")] - torch.logsumexp(
                    previous.masked_fill(
                        torch.arange(len(ACTIONS)) == ACTIONS.index("BRANCH"), -1e4), dim=1)
            current = model(x[bp]).masked_fill(~mask[bp], -1e4)
            j = ACTIONS.index("BRANCH")
            rival = current.clone()
            rival[:, j] = -1e4
            current_branch = current[:, j] - rival.logsumexp(1)
            retention = torch.nn.functional.softplus(previous_branch - current_branch).mean()
            # W16 has excess BRANCH false positives even with legal alternatives.
            # Penalize only BRANCH-legal hard negatives; preserve positive
            # margins through frozen W16 distillation above.
            ni = rng.choices(neg, k=96)
            ns = model(x[ni]).masked_fill(~mask[ni], -1e4)
            rival_neg = ns.clone()
            rival_neg[:, j] = -1e4
            neg_gap = ns[:, j] - rival_neg.logsumexp(1)
            negative_rejection = torch.nn.functional.softplus(neg_gap + 0.15).mean()
            if si:
                rid = [rng.randrange(len(replay)) for _ in range(64)]
                rce = torch.nn.functional.cross_entropy(
                    model(rx[rid]).masked_fill(~rm[rid], -1e4), ry[rid])
                loss = .16 * ce + .16 * branch + .23 * contrast + .13 * rce + .14 * retention + .18 * negative_rejection
            else:
                loss = .20 * ce + .16 * branch + .25 * contrast + .16 * retention + .23 * negative_rejection
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite W17 optimizer loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            if not bool(torch.isfinite(norm)) or float(norm) <= 0:
                raise ValueError("nonfinite or zero W17 gradient")
            opt.step()
            if (k + 1) % 20 == 0:
                history.append({"stage": stage, "step": k + 1,
                    "total_real_updates": si * UPDATES_PER_WAVE + k + 1,
                    "loss": round(float(loss.detach()), 6),
                    "grad_norm": round(float(norm), 6)})
                write_json(out / "partial-optimizer-history.json", history)
        evaluation = fidelity(torch, model, heldout, heldout_pairs)
        cp = checkpoint(torch, model, opt, rng, out, stage, evaluation)
        stages.append({"name": stage, "real_AdamW_updates": UPDATES_PER_WAVE,
                       "evaluation": evaluation, "checkpoint": cp})
        write_json(out / "stage-progress.json", stages)
    return stage1, model, stages, history


def mixed_head(torch, a, b, weight=0.65):
    class Interpolated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = a, b
        def forward(self, features):
            return weight * self.a(features) + (1 - weight) * self.b(features)
    return Interpolated().eval()


def main(args):
    if args.smoke_only:
        assert ACTIONS == ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
        assert set(TRAIN_SEEDS).isdisjoint(HELDOUT)
        assert not set(TRAIN_SEEDS + HELDOUT).intersection(
            set(w16.TRAIN_SEEDS + w16.HELDOUT + w15.TRAIN_SEEDS + w15.HELDOUT
                + w14.TRAIN_SEEDS + w14.HELDOUT + w13.TRAIN_SEEDS + w13.HELDOUT))
        assert len(W16_SHA) == 64 and len(W15_SHA) == 64 and len(W14_SHA) == 64
        print("W17 source-only preflight PASS, zero optimizer steps", flush=True)
        return
    begin = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_cpu(1)
    import torch
    torch.set_num_threads(1)
    profile_data = profile("sprocket", require_supported=True)
    assert profile_data["model_id"] == "kandivault/sprocket-500m"
    assert profile_data["revision"] == "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
    assert profile_data["subfolder"] == ""
    assert profile_data["framing"]["eos_token_id"] == 5
    for path, digest in ((args.w16, W16_SHA), (args.w15, W15_SHA), (args.w14, W14_SHA),
                         (args.w0, W0_SHA), (args.support, SUPPORT_SHA)):
        assert sha256_file(path) == digest, ("SHA admission failed", str(path), digest)
    support = objective.load_support(args.support, args.w0)
    write_json(args.output / "provenance.json", {
        "version": VERSION, "source_head": os.getenv("GITHUB_SHA", "local"),
        "job_id": os.getenv("GITHUB_RUN_ID", "local"),
        "attempt": os.getenv("GITHUB_RUN_ATTEMPT", "local"),
        "source_script_sha256": sha256_file(__file__),
        "root_profile": profile_data, "root_model_loaded": False,
        "root_forwards": 0, "root_optimizer_steps": 0, "LoRA_updates": 0,
        "W16_parent_sha256": W16_SHA, "W15_parent_sha256": W15_SHA,
        "W14_parent_sha256": W14_SHA,
        "W0_sha256": W0_SHA, "support_sha256": SUPPORT_SHA,
        "planned_real_aux_optimizer_steps": UPDATES_PER_WAVE * 2,
        "synthetic_host": True, "native": False,
        "sealed_final_opened": False, "draft_hold": True})
    h14 = w11.load(torch, args.w14, W14_SHA, w14.VERSION)
    h15 = w11.load(torch, args.w15, W15_SHA, w15.VERSION)
    h16 = w11.load(torch, args.w16, W16_SHA, w16.VERSION)
    stage1, trained, stages, history = train(
        torch, support.worm, support, h14, h15, h16, args.output)
    blended = mixed_head(torch, trained, h14)
    entries = []
    gallery = args.output / "original-png-json"
    for seed in HELDOUT:
        for family in w9.FAMILIES:
            learned = (("W14-greedy", h14, 0.0),
                       ("W15-greedy", h15, 0.0),
                       ("W16-greedy", h16, 0.0),
                       ("W16-t07", h16, 0.7),
                       ("W17-stage1-greedy", stage1, 0.0),
                       ("W17-final-greedy", trained, 0.0),
                       ("W17-final-t07", trained, 0.7),
                       ("W17-mixed-t07", blended, 0.7))
            for name, head, temperature in learned:
                rec = w14.rollout(torch, support.worm, head, seed, family, name, temperature)
                support.worm.write_result(rec, gallery)
                entries.append({"seed": seed, "family": family, "controller": name,
                    "metrics": rec["metrics"], "motifs": w15.motifs(rec),
                    "actual_head_forwards": len(rec["head_decisions"]),
                    "trajectory_sha256": hashlib.sha256(json.dumps(
                        rec["trajectory"], separators=(",", ":")).encode()).hexdigest()})
            for name in ("random", "wall", "fidget"):
                rec = support.worm.run(seed, name, family, 128)
                support.worm.write_result(rec, gallery)
                entries.append({"seed": seed, "family": family, "controller": name,
                    "metrics": rec["metrics"], "motifs": w15.motifs(rec),
                    "actual_head_forwards": 0, "trajectory_sha256": hashlib.sha256(
                        json.dumps(rec["trajectory"], separators=(",", ":")).encode()).hexdigest()})
        write_json(args.output / "partial-original-rollouts.json", entries)
    assert len(entries) == len(HELDOUT) * len(w9.FAMILIES) * 11
    assert all(e["metrics"]["invalid_raw_choices"] == 0 for e in entries)
    checkpoints = [s["checkpoint"] for s in stages if "checkpoint" in s]
    assert len(checkpoints) == 2
    assert all(sha256_file(args.output / c["file"]) == c["sha256"] for c in checkpoints)
    write_json(args.output / "summary.json", {
        "status": "complete", "version": VERSION,
        "genuine_aux_AdamW_updates": UPDATES_PER_WAVE * 2,
        "root_forwards": 0, "root_updates": 0, "LoRA_updates": 0,
        "real_head_forwards": sum(e["actual_head_forwards"] for e in entries),
        "original_png_json_pairs": len(entries), "stages": stages,
        "optimizer_history": history, "results": entries,
        "wall_seconds": time.monotonic() - begin,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_readback_sha_valid": True,
        "synthetic": True, "native": False})
    print(json.dumps({"status": "complete", "genuine_optimizer_steps": UPDATES_PER_WAVE * 2,
                      "original_png_json_pairs": len(entries)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("w16", "w15", "w14", "w0", "support", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    try:
        if not args.smoke_only and any(getattr(args, n) is None for n in
            ("w16", "w15", "w14", "w0", "support", "output")):
            parser.error("all pinned input/output paths required")
        main(args)
    except BaseException as e:
        if args.output and args.output.exists():
            write_json(args.output / "failure.json", {"failed": True,
                "type": type(e).__name__, "error": str(e)})
        raise
