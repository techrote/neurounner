#!/usr/bin/env python3
"""GP350-015: independent, frozen intermediate-GP014 checkpoint recovery selection.

No optimiser, no CyberSand implementation writes, no native success claim.
Evaluate the original full LiquidAI LFM2.5-350M + original archived LoRA
step18/36/54 weights on the source-pinned W0 synthetic host. Reuse—not
relabel as newly inferred—the GP013 parent and GP014 step72 original receipts.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

import lfm350 as c
import gp350_recovery_boost_train as gp

SCHEMA = "driv3r.lfm350.gp350015-early-recovery-selection/1"
SOURCE_RUN = 43933507612
SOURCE_ARTIFACT = 14261975494
SOURCE_SHA = "b384826fca4eb6e5f315e341cd688ec8b3bd29c5"
HASHES = {
    18: "205f6356c25766a5f0664e5e4faf5a5135c22f092c55aa2d50f81293e230335f",
    36: "e37999047767861cfb7c05d64310d503505eeff3795c755d547b5e8aaa81cdf7",
    54: "dc302a1a610b5301ae94d4b8304cf9ae4292cfbd9a209dbb98db3252bf4c9d64",
    72: "d2d53be6fa737dda0e978c31914e3f5c5254af64a7ef3ca4cece7dd4b2878074",
}
STEPS = (18, 36, 54)
DEADLINE = 3100
CANCELLED = False

def sig(_signum, _frame):
    global CANCELLED
    CANCELLED = True
    gp.CANCELLED = True

def original(root):
    p = root / "original-experiment"
    s = json.loads((p / "summary.json").read_text())
    c.check(s["source_sha"] == SOURCE_SHA and s["status"] == "complete",
            "original GP350014 source/status mismatch")
    c.check(s["updates"] == s["backwards"] == 72
            and s["real_gradient_forwards"] == 216
            and s["frozen_full_parent_unchanged"]
            and s["adapter_updated"] and s["original_pngs"] == 60,
            "original optimiser and original W0 evidence missing")
    c.check(s["parent_adapter_sha256"] ==
        "b7790a5fd1c8b048ff94a9ad174fd65f51b871e0c885264a22e350111934ff77",
        "original GP013 parent checkpoint mismatch")
    for n, digest in HASHES.items():
        src = p / f"adapter-step-{n}" / "adapter_model.safetensors"
        c.check(src.is_file() and c.sha(src) == digest,
                "original GP014 checkpoint custody failure: " + str(n))
    for row in s["baseline"] + s["trained"] + s["controls"]:
        c.check(c.sha(p / row["png"]) == row["png_sha256"],
                "original GP014 W0 PNG custody failure")
    return p, s

def sample_checked(w0, original_path):
    observed = gp.observed(w0, gp.DEV_SEEDS)
    general = gp.stratified(observed)
    recovery = gp.recovery_probes(observed)
    c.check(len(general) == 72 and len(recovery) == 48,
            "W0 heldout probe inventory mismatch")
    for rows, old_name in ((general, "probe-initial.json"),
                           (recovery, "recovery-probe-initial.json")):
        old = json.loads((original_path / old_name).read_text())["rows"]
        c.check(len(rows) == len(old), "original heldout row length mismatch")
        for a, b in zip(rows, old):
            obs = a["observation"]
            c.check((a["seed"], a["size"], obs["family"], a["ray2"], a["target"]) ==
                    (b["seed"], b["size"], b["family"], b["marker"], b["target"]),
                    "W0 heldout inventory/teacher mismatch")
    return general, recovery

def local_trapped(obs):
    return ("RETRACE" in obs["legal"] and not any(
        a in obs["legal"] and c.nearby(obs, turn) == "."
        for a, turn in (("STRAIGHT", 0), ("LEFT", -1), ("RIGHT", 1))))

def classify_recovery(probe):
    confusion = collections.Counter()
    groups = collections.defaultdict(collections.Counter)
    for row in probe["rows"]:
        positive = row["target"] == "RETRACE"
        chosen = row["legal"] == "RETRACE"
        confusion["positive" if positive else "negative"] += 1
        result = ("true_positive" if positive and chosen else
                  "false_negative" if positive else
                  "false_positive" if chosen else "true_negative")
        confusion[result] += 1
        groups[row["family"]][result] += 1
    c.check(confusion["positive"] == confusion["negative"] == 24,
            "balanced 24/24 W0 trapped/nontrapped probe drift")
    return {"counts": dict(confusion),
            "family": {k: dict(v) for k, v in groups.items()},
            "exact_teacher_matches": probe["teacher_matches"]}

def action_transfer(path, episodes):
    total = collections.Counter()
    by_family = collections.defaultdict(collections.Counter)
    for row in episodes:
        events = json.loads((path / row["png"].replace(".png", ".json")).read_text())
        scores = events["scores"]
        source_events = events["record"]["events"]
        c.check(len(scores) == len(source_events) == row["decisions"] == 64,
                "missing raw score/host trajectory evidence")
        for decision, e in zip(scores, source_events):
            obs, scores6 = decision["obs"], decision["six_logits"]
            c.check(len(scores6) == 6 and all(math.isfinite(x) for x in scores6),
                    "invalid 6-way model scores")
            raw = c.ACTIONS[max(range(6), key=lambda x: scores6[x])]
            legal = max(obs["legal"], key=lambda a: scores6[c.ACTIONS.index(a)])
            c.check((raw, legal) == (decision["raw"], decision["applied"])
                and legal == e["applied"] == e["raw_choice"]
                and e["raw_legal"] and decision["raw_legal"] == (raw in obs["legal"]),
                "model score -> host legal W0 application mismatch")
            trapped = local_trapped(obs)
            for ctr in (total, by_family[obs["family"]]):
                ctr["decisions"] += 1
                ctr["trapped"] += int(trapped)
                ctr["trapped_retraces"] += int(trapped and legal == "RETRACE")
                ctr["trapped_missed"] += int(trapped and legal != "RETRACE")
                ctr["retrace_with_open_arm"] += int(not trapped and legal == "RETRACE")
                ctr["branches"] += int(legal == "BRANCH")
                ctr["raw_illegal"] += int(raw not in obs["legal"])
    return {"total": dict(total),
            "by_family": {key: dict(value) for key,value in by_family.items()},
            "mean_unique_cells": sum(x["unique_cells"] for x in episodes) / len(episodes)}

def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, snapshot_download

    started = time.monotonic()
    torch.set_num_threads(2)
    c.check(not args.out.exists(), "immutable GP350015 output already exists")
    args.out.mkdir(parents=True)
    original_dir, orig = original(args.source)
    w0 = c.load_w0(args.w0)
    general, recovery = sample_checked(w0, original_dir)
    c.save_json(args.out/"lease.json", {
        "schema": SCHEMA, "source_sha": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "source_run": SOURCE_RUN, "source_artifact": SOURCE_ARTIFACT,
        "source_model_sha": SOURCE_SHA, "adapter_hashes": HASHES,
        "new_optimizer_updates": 0, "no_new_parent_or_step72_forwards": True,
        "model": c.MODEL, "revision": c.REVISION,
        "W0_source_only": True, "native": False,
        "vision": False, "paid_GPU": False, "draft_hold": True})
    c.check(HfApi().model_info(c.MODEL, revision=c.REVISION).sha == c.REVISION,
            "pinned full parent changed upstream")
    snapshot_download(c.MODEL, revision=c.REVISION, local_dir=args.cache,
        max_workers=2, allow_patterns=[
            "*.safetensors", "*.safetensors.index.json", "config.json",
            "generation_config.json", "tokenizer*", "special_tokens_map.json",
            "added_tokens.json", "vocab.*", "merges.txt", "README.md", "LICENSE*"])
    hashes = {str(p.relative_to(args.cache)): c.sha(p)
              for p in sorted(args.cache.rglob("*")) if p.is_file()
              and ".cache" not in p.parts}
    old_hashes = json.loads((original_dir/"model-download.json").read_text())["sha256"]
    c.check(all(hashes.get(k) == value for k,value in old_hashes.items()),
            "original pretrained model weight SHA inventory mismatch")
    c.save_json(args.out/"model-sha256.json", hashes)

    tok = AutoTokenizer.from_pretrained(args.cache, local_files_only=True,
                                       trust_remote_code=False)
    ids = c.id_map(tok)
    base = AutoModelForCausalLM.from_pretrained(args.cache, local_files_only=True,
                                               trust_remote_code=False, dtype=torch.float32)
    c.check(base.config.model_type == "lfm2" and base.config.num_hidden_layers == 16,
            "wrong full LiquidAI LFM350")
    base.requires_grad_(False)
    base.eval()
    model = PeftModel.from_pretrained(
        base, original_dir/"adapter-step-18",
        adapter_name="step18", is_trainable=False)
    for n in (36, 54):
        model.load_adapter(original_dir/f"adapter-step-{n}",
                           adapter_name=f"step{n}", is_trainable=False)
    c.check(not any(p.requires_grad for p in model.parameters()),
            "frozen-only checkpoint comparison accidentally trainable")
    frozen = c.base_fingerprint(model)
    stages = {}
    for step in STEPS:
        c.check(not CANCELLED and time.monotonic() - started < args.deadline - 480,
                "bounded evaluation reached deadline before next checkpoint")
        label = f"gp350014-step{step}"
        model.set_adapter(f"step{step}")
        model.eval()
        began = time.monotonic()
        general_probe = gp.probe(model, tok, ids, general)
        recover_probe = gp.probe(model, tok, ids, recovery)
        c.save_json(args.out/f"{label}-general-probe.json", general_probe)
        c.save_json(args.out/f"{label}-recovery-probe.json", recover_probe)
        rows = []
        # Evaluate exactly the GP350-014 original world/seed/family cells.
        for size in gp.SIZES:
            for seed in gp.EVAL_SEEDS:
                for family in c.FAMILIES:
                    c.check(not CANCELLED and time.monotonic()-started < args.deadline-240,
                            "evaluation cancelled / insufficient cleanup time")
                    rows.append(gp.episode(w0, model, tok, ids, args.out,
                                           label, size, seed, family,
                                           time.monotonic()))
        stats = action_transfer(args.out, rows)
        stages[label] = {
            "checkpoint_step": step,
            "adapter_sha256": HASHES[step],
            "general_teacher_matches": general_probe["teacher_matches"],
            "general_raw_illegal": general_probe["raw_illegal"],
            "recovery": classify_recovery(recover_probe),
            "episodes": rows,
            "on_policy": stats,
            "wall_seconds": time.monotonic() - began,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        c.save_json(args.out/f"{label}-stage.json", stages[label])
        print(c.canonical({"GP350015_STAGE": label,
            "recovery": stages[label]["recovery"]["counts"],
            "on_policy": stats["total"],
            "mean_unique_cells": stats["mean_unique_cells"]}), flush=True)
    c.check(c.base_fingerprint(model) == frozen, "full frozen base modified")
    c.check(sum(len(q["episodes"]) for q in stages.values()) == 36,
            "incomplete matched checkpoint rollouts")
    c.save_json(args.out/"summary.json", {
        "schema": SCHEMA, "status": "complete",
        "source_sha": os.getenv("GITHUB_SHA"),
        "original_run": SOURCE_RUN,
        "original_artifact_id": SOURCE_ARTIFACT,
        "original_source_sha": SOURCE_SHA,
        "revision": c.REVISION,
        "fresh_checkpoint_stages": stages,
        "original_source_parent": {
            "general_teacher_matches": orig["initial_probe"]["teacher_matches"],
            "recovery_probe": orig["recovery_initial_probe"]["teacher_matches"],
            "on_policy": {"mean_unique_cells":
                sum(r["unique_cells"] for r in orig["baseline"])/12,
                "retrace": sum(r["retraces"] for r in orig["baseline"])}},
        "original_source_step72": {
            "general_teacher_matches": orig["final_probe"]["teacher_matches"],
            "recovery_probe": orig["recovery_final_probe"]["teacher_matches"],
            "on_policy": {"mean_unique_cells":
                sum(r["unique_cells"] for r in orig["trained"])/12,
                "retrace": sum(r["retraces"] for r in orig["trained"])}},
        "original_parent_and_step72_not_rescored": True,
        "new_model_decisions": 36*64, "new_heldout_probes": 3*(72+48),
        "new_optimizer_steps": 0, "original_pngs": 36,
        "W0_synthetic_only": True, "d1_matched": "NOT RUN",
        "native": False, "vision": False, "draft_hold": True,
        "wall_seconds": time.monotonic()-started,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

def self_test(w0, source):
    original_dir, s = original(source)
    general, recovery = sample_checked(c.load_w0(w0), original_dir)
    c.check(len(general) == 72 and len(recovery) == 48,
            "original W0 sample shape drift")
    c.check(s["initial_probe"]["teacher_matches"] == 41
        and s["final_probe"]["teacher_matches"] == 38
        and s["recovery_initial_probe"]["teacher_matches"] == 18
        and s["recovery_final_probe"]["teacher_matches"] == 26,
        "original model-score source record mismatch")
    print("GP350015 original adapter custody / W0 heldout self-test PASS", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w0", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deadline", type=int, default=DEADLINE)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    c.check(args.deadline == DEADLINE, "checkpoint deadline altered")
    try:
        if args.self_test:
            self_test(args.w0, args.source)
        else:
            evaluate(args)
    except BaseException as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        f = args.out/"failure.json"
        if not f.exists():
            c.save_json(f, {"error":type(exc).__name__,"message":str(exc),
               "source_sha":os.getenv("GITHUB_SHA"),"no_success_claim":True})
        raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,sig)
    signal.signal(signal.SIGINT,sig)
    main()
