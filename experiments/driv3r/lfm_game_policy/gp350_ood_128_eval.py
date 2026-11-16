#!/usr/bin/env python3
"""GP350-012: independent 128-turn out-of-distribution W0 trajectory stress.

Exactly pinned synthetic W0 host, full frozen LFM350, one existing rank-8
adapter per Actions job. No optimizer, no native gameplay, no vision claim.
Novel *world seed and size* does not imply wholly novel 5x5 local geometry:
explicitly quantify train-state overlap in the evidence.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import resource
import signal
import time

import lfm350 as c
import gp350_paired_margin_train as g
import gp350_selective_branch_train as p

SCHEMA = "driv3r.lfm350.gp350012-ood-128/1"
SIZES = (24, 64)
SEEDS = (9803, 9821)
STEPS = 128
DEADLINE = 2250
ADAPTERS = {
    "stock": None,
    "gp350006": "436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c",
    "gp350010-step24": "047b2bacb61f6f04fb359e4d9839f6c198a17ba7db6dfb451d88e48846c66430",
}
CANCELLED = False


def stop(_signum, _frame):
    global CANCELLED
    CANCELLED = True


def key(obs):
    """Do not count seed or tick as novel geometry."""
    return (tuple(obs["local_5x5"]), obs["heading"],
            tuple(obs["legal"]), obs["family"])


def collect_probe(w0):
    """Real source-host observations, synthetic teacher; no fabricated states."""
    train_keys = set()
    for r in p.observed(w0, p.TRAIN_SEEDS):
        train_keys.add(key(r["observation"]))
    buckets = collections.defaultdict(list)
    seen = set()
    source_counts = collections.Counter()
    for size in SIZES:
        for seed in SEEDS:
            for family in c.FAMILIES:
                for mode in ("random", "fidget"):
                    record = w0.run(seed, mode=mode, family=family,
                                    size=size, steps=128)
                    c.check(record["metrics"]["invalid_raw_choices"] == 0,
                            "source W0 control produced illegal choice")
                    for event in record["events"]:
                        obs = event["observation"]
                        marker = g.ray2(obs)
                        source_counts[f"{size}/{family}/{marker}"] += 1
                        signature = (size, family, marker, key(obs))
                        if signature in seen:
                            continue
                        seen.add(signature)
                        target = p.teacher(obs)
                        c.check(target in obs["legal"], "synthetic teacher illegal")
                        buckets[(size, family, marker)].append({
                            "seed": seed, "size": size, "mode": mode,
                            "observation": obs, "marker": marker,
                            "target": target, "geometry_novel_to_training": key(obs) not in train_keys,
                            "observed_in_source_W0": True, "teacher_is_synthetic": True})
    sample = []
    for size in SIZES:
        for family in c.FAMILIES:
            for marker in (".", "#", "+"):
                group = buckets[(size, family, marker)]
                for i in range(min(2, len(group))):
                    sample.append(group[(i * len(group)) // min(2, len(group))])
    c.check(len(sample) >= 24, "insufficient OOD source-host probe exposure")
    c.check({r["seed"] for r in sample}.isdisjoint(set(p.TRAIN_SEEDS) | set(p.DEV_SEEDS)),
            "new evaluation source seeds overlap training/DEV")
    return sample, dict(source_counts), len(train_keys)


def pressure_and_habits(scores, record):
    rays = collections.Counter()
    actions = collections.Counter()
    branch_bursts = []
    burst = 0
    pressure = collections.Counter()
    revisits = 0
    first_visits = 0
    seen = {tuple(record["trajectory"][0])}
    for decision, event in zip(scores, record["events"]):
        marker = decision["ray2"]
        applied = decision["applied"]
        rays[marker] += 1
        actions[applied] += 1
        if marker in ("#", "+"):
            pressure["encounters"] += 1
            if applied in ("LEFT", "RIGHT"):
                pressure["anticipatory_turns"] += 1
            elif applied == "BRANCH":
                pressure["branches_during_pressure"] += 1
            else:
                pressure["other_choices"] += 1
        if applied == "BRANCH":
            burst += 1
        elif burst:
            branch_bursts.append(burst)
            burst = 0
        current = tuple(event["position_after"])
        if applied not in ("RETRACE", "STOP"):
            if current in seen:
                revisits += 1
            else:
                first_visits += 1
        seen.add(current)
    if burst:
        branch_bursts.append(burst)
    return {"action_counts": dict(actions), "ray2_counts": dict(rays),
            "pressure": dict(pressure), "branch_bursts": branch_bursts,
            "max_branch_burst": max(branch_bursts, default=0),
            "revisit_moves": revisits, "first_visit_moves": first_visits,
            "novel_cells_per_executed_turn": first_visits / max(len(scores), 1)}


def episode(w0, model, tok, ids, args, out, started, size, seed, family):
    import torch
    decisions = []

    def choose(obs):
        c.check(not CANCELLED and time.monotonic()-started < args.deadline-180,
                "bounded 128-turn evaluation exceeded deadline")
        with torch.inference_mode():
            z = g.rich_score(model, tok, ids, obs)
        values = [float(v) for v in z]
        raw = c.ACTIONS[max(range(6), key=lambda i: values[i])]
        selected = max(obs["legal"], key=lambda a: values[c.ACTIONS.index(a)])
        decisions.append({"obs": obs, "ray2": g.ray2(obs), "raw": raw,
                          "raw_legal": raw in obs["legal"], "applied": selected,
                          "six_logits": values})
        return selected

    record = w0.run(seed, mode="callback", family=family, size=size,
                    steps=STEPS, selector=choose)
    c.check(0 < len(decisions) == len(record["events"]) <= STEPS,
            "model score count does not match W0 events")
    for d, event in zip(decisions, record["events"]):
        c.check(d["applied"] in d["obs"]["legal"] and
                d["applied"] == event["applied"] == event["raw_choice"] and
                event["raw_legal"], "host applied a non-model or illegal action")
    replay = w0.run(seed, mode="replay", family=family, size=size,
                    steps=STEPS, raw_actions=[d["applied"] for d in decisions])
    png = w0.png_bytes(record)
    c.check(w0.png_bytes(replay) == png and record["trajectory"] == replay["trajectory"],
            "W0 trajectory replay differs from model episode")
    stem = f"{args.stage}-s{size}-{seed}-{family}"
    (out / (stem + ".png")).write_bytes(png)
    c.save_json(out / (stem + ".json"),
                {"decisions": decisions, "record": record, "frozen": True})
    metrics = record["metrics"]
    return {"stage": args.stage, "size": size, "seed": seed, "family": family,
            "decision_budget": STEPS, "decisions": len(decisions),
            "unique_cells": metrics["unique_trail_cells"],
            "branches": metrics["branches"], "retraces": metrics["retraces"],
            "raw_illegal": sum(not d["raw_legal"] for d in decisions),
            "host_illegal": metrics["invalid_raw_choices"],
            "early_stop": metrics["terminated_by_stop"],
            "host_revisit_contacts": metrics["total_revisit_contacts"],
            "habits": pressure_and_habits(decisions, record),
            "png": stem + ".png", "png_sha256": c.sha_bytes(png)}


def control(w0, out, size, seed, family, mode):
    record = w0.run(seed, mode=mode, family=family, size=size, steps=STEPS)
    c.check(record["metrics"]["invalid_raw_choices"] == 0,
            "procedural control applied illegal action")
    stem = f"control-{mode}-s{size}-{seed}-{family}"
    png = w0.png_bytes(record)
    (out / (stem + ".png")).write_bytes(png)
    c.save_json(out / (stem + ".json"), record)
    return {"mode": mode, "size": size, "seed": seed, "family": family,
            "decisions": len(record["events"]), "unique_cells":
            record["metrics"]["unique_trail_cells"], "branches":
            record["metrics"]["branches"], "retraces":
            record["metrics"]["retraces"], "host_illegal":
            record["metrics"]["invalid_raw_choices"], "png": stem + ".png",
            "png_sha256": c.sha_bytes(png)}


def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, snapshot_download

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(3510012)
    c.check(not args.out.exists(), "immutable stage evidence already exists")
    args.out.mkdir(parents=True)
    if args.stage != "stock":
        c.check(c.sha(args.adapter / "adapter_model.safetensors") == ADAPTERS[args.stage],
                "immutable adapter SHA256 mismatch")
    w0 = c.load_w0(args.w0)
    sample, source_counts, n_train = collect_probe(w0)
    c.save_json(args.out / "preflight.json", {
        "schema": SCHEMA, "source_sha": os.getenv("GITHUB_SHA"),
        "run": os.getenv("GITHUB_RUN_ID"), "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "stage": args.stage, "adapter_sha256": ADAPTERS[args.stage],
        "model": c.MODEL, "revision": c.REVISION,
        "OOD_sizes": SIZES, "OOD_seeds": SEEDS,
        "128_turns_each": True, "heldout_synthetic_teacher": True,
        "model_native": False, "vision": False, "d1_matched": "NOT RUN",
        "optimizers": 0, "draft_hold": True,
        "train_distinct_state_signatures": n_train,
        "source_host_probe_distribution": source_counts,
        "heldout_probes": len(sample)})
    H = HfApi().model_info(c.MODEL, revision=c.REVISION)
    c.check(H.sha == c.REVISION, "upstream full-parent pin drift")
    snapshot_download(c.MODEL, revision=c.REVISION, local_dir=args.cache, max_workers=2,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json",
            "generation_config.json", "tokenizer*", "special_tokens_map.json",
            "added_tokens.json", "vocab.*", "merges.txt", "README.md", "LICENSE*"])
    manifest = {str(f.relative_to(args.cache)): c.sha(f)
                for f in sorted(args.cache.rglob("*"))
                if f.is_file() and ".cache" not in f.parts}
    c.check(any(f.endswith(".safetensors") for f in manifest),
            "full pinned weights absent")
    c.save_json(args.out / "full-model-manifest.json",
                {"source": c.MODEL, "revision": H.sha, "sha256": manifest})
    tok = AutoTokenizer.from_pretrained(args.cache, local_files_only=True,
                                        trust_remote_code=False)
    ids = c.id_map(tok)
    base = AutoModelForCausalLM.from_pretrained(
        args.cache, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32)
    c.check(base.config.model_type == "lfm2" and base.config.num_hidden_layers == 16,
            "not full pinned LFM350")
    base.requires_grad_(False)
    base.eval()
    model = (base if args.stage == "stock" else
             PeftModel.from_pretrained(base, args.adapter, is_trainable=False))
    c.check(not any(v.requires_grad for v in model.parameters()),
            "checkpoint-only stage accidentally trainable")
    model.eval()
    parent_fingerprint = c.base_fingerprint(model)
    episodes = []
    for size in SIZES:
        for seed in SEEDS:
            for family in c.FAMILIES:
                episode_row = episode(w0, model, tok, ids, args, args.out,
                                      started, size, seed, family)
                episodes.append(episode_row)
                print(c.canonical({"GP350012_EPISODE": episode_row}), flush=True)
    probes = []
    with torch.inference_mode():
        for row in sample:
            z = g.rich_score(model, tok, ids, row["observation"])
            vals = [float(v) for v in z]
            raw = c.ACTIONS[max(range(6), key=lambda i: vals[i])]
            legal = row["observation"]["legal"]
            applied = max(legal, key=lambda a: vals[c.ACTIONS.index(a)])
            probes.append({"seed": row["seed"], "size": row["size"],
                "family": row["observation"]["family"], "ray2": row["marker"],
                "target": row["target"], "choice": applied, "raw": raw,
                "raw_legal": raw in legal,
                "match": applied == row["target"], "novel_state":
                    row["geometry_novel_to_training"], "six_logits": vals})
    c.save_json(args.out / "heldout-probes.json", {
        "items": probes, "matches": sum(r["match"] for r in probes),
        "novel_state_count": sum(r["novel_state"] for r in probes),
        "raw_illegal": sum(not r["raw_legal"] for r in probes)})
    controls = []
    for size in SIZES:
        for seed in SEEDS:
            for family in c.FAMILIES:
                for mode in ("random", "wall", "fidget"):
                    controls.append(control(w0, args.out, size, seed, family, mode))
    c.check(parent_fingerprint == c.base_fingerprint(model),
            "frozen parent weights changed during inference")
    for row in episodes + controls:
        c.check(c.sha(args.out / row["png"]) == row["png_sha256"],
                "original PNG SHA mismatch")
    c.check(len(episodes) == 12 and len(controls) == 36
            and all(row["host_illegal"] == 0 for row in episodes + controls),
            "incomplete episodes/controls or host-illegal application")
    c.save_json(args.out / "summary.json", {
        "schema": SCHEMA, "status": "complete", "source_sha": os.getenv("GITHUB_SHA"),
        "stage": args.stage, "model": c.MODEL, "revision": c.REVISION,
        "adapter_sha256": ADAPTERS[args.stage], "optimizer_steps": 0,
        "synthetic_W0_only": True, "native": False, "vision": False,
        "matched_d1": "NOT RUN", "steps_per_episode": STEPS,
        "sizes": SIZES, "seeds": SEEDS, "episodes": episodes, "controls": controls,
        "model_decisions": sum(r["decisions"] for r in episodes),
        "original_pngs": len(episodes) + len(controls),
        "probe_n": len(probes), "probe_matches": sum(r["match"] for r in probes),
        "probe_novel_geometries": sum(r["novel_state"] for r in probes),
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(c.canonical({"GP350012_COMPLETE": args.stage,
          "model_decisions": sum(r["decisions"] for r in episodes),
          "probe_matches": sum(r["match"] for r in probes),
          "wall_seconds": time.monotonic()-started}), flush=True)


def self_test(w0_path):
    w0 = c.load_w0(w0_path)
    c.check(SIZES == (24, 64) and len(SEEDS) == 2 and STEPS == 128,
            "OOD budget drift")
    sample, dist, n_train = collect_probe(w0)
    for size in SIZES:
        for seed in SEEDS:
            for family in c.FAMILIES:
                source = w0.run(seed, family=family, mode="wall",
                                size=size, steps=STEPS)
                replay = w0.run(seed, family=family, mode="replay",
                                size=size, steps=STEPS,
                                raw_actions=[r["applied"] for r in source["events"]])
                c.check(w0.png_bytes(source) == w0.png_bytes(replay),
                        "OOD W0 controls do not replay")
    print(c.canonical({"GP350012_SELF_TEST": "PASS", "probe_count": len(sample),
       "novel_probe_states": sum(r["geometry_novel_to_training"] for r in sample),
       "train_state_signatures": n_train, "source_counts": dist}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w0", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--deadline", type=int, default=DEADLINE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    c.check(args.deadline == DEADLINE, "bounded recipe deadline changed")
    if args.self_test:
        self_test(args.w0)
        return
    try:
        evaluate(args)
    except BaseException as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        if not (args.out / "failure.json").exists():
            c.save_json(args.out / "failure.json", {
                "source_sha": os.getenv("GITHUB_SHA"), "stage": args.stage,
                "kind": type(exc).__name__, "message": str(exc),
                "not_complete": True})
        raise


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
