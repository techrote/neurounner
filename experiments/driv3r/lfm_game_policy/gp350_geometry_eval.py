#!/usr/bin/env python3
"""GP350-004: true full-LFM350 checkpoint-only geometry response and 48-turn W0 galleries.

Read-only synthetic host. Original GP350-003 checkpoint; zero optimizer updates.
Counterfactual outer-ring edits are explicitly SYNTHETIC (the legal menu is unchanged);
model-generated trajectories are real full-model scored inference with W0 host authority.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import lfm350 as c
import gp350_triad as triad

SCHEMA = "driv3r.lfm350.gp350004-geometry-eval/1"
PRODUCER = "5d5e4a5a559f0b3adb340a5a5bde6e3775938500"
ADAPTER_SHA = "57ecb1bb4f0df95c34fc9ddba457094eb06fcb2f34690bb34b9453140ba9b187"
SEEDS = (9503, 9521, 9539)
PROBE_SEEDS = (9103, 9119, 9137, 9503, 9521, 9539, 9551, 9577)
MAX_TURNS = 48


def legal_argmax(scores, obs):
    return max(obs["legal"], key=lambda a: scores[c.ACTIONS.index(a)])


def probs(scores):
    maxi = max(scores)
    exp = [math.exp(x - maxi) for x in scores]
    total = sum(exp)
    return [v / total for v in exp]


def metrics(record):
    m = record["metrics"]
    return {
        "decisions": record["steps_executed"],
        "unique_cells": m["unique_trail_cells"],
        "branches": m["branches"],
        "retraces": m["retraces"],
        "revisit_contacts": m["total_revisit_contacts"],
        "stopped": m["terminated_by_stop"],
        "illegal_applications": m["invalid_raw_choices"],
    }


def outer_variant(obs, replacement):
    # Only edit 2-cell-distance pixels; every immediate neighbour and action
    # legality precondition is unchanged. This is NOT a real W0 world readback.
    lines = [list(x) for x in obs["local_5x5"]]
    candidates = [(y, x) for y in range(5) for x in range(5)
                  if max(abs(x - 2), abs(y - 2)) == 2 and lines[y][x] == "."]
    if not candidates:
        return None
    y, x = candidates[(obs["tick"] + obs["heading"]) % len(candidates)]
    lines[y][x] = replacement
    changed = dict(obs, local_5x5=["".join(row) for row in lines])
    c.check(changed["legal"] == obs["legal"] and
            all(changed[k] == obs[k] for k in obs if k != "local_5x5"),
            "counterfactual leaked non-geometry changes")
    return changed


def sampled_probes(w0):
    # Twelve examples per family, spanning four time bands, with different
    # held-out W0 obstacle worlds. Restrict to genuine observed starting states.
    by_family = {f: {band: [] for band in range(4)} for f in c.FAMILIES}
    for seed in PROBE_SEEDS:
        for family in c.FAMILIES:
            for row in c.collect_world(w0, seed, family, steps=32):
                obs = row["observation"]
                band = min(obs["tick"] // 8, 3)
                if outer_variant(obs, "#") is not None:
                    by_family[family][band].append({"seed": seed, "obs": obs})
    probes = []
    for family in c.FAMILIES:
        for band in range(4):
            pool = by_family[family][band]
            # Different seed/tick combinations, not repeated images pretending
            # to be independent worlds.
            chosen = []
            seen = set()
            for row in sorted(pool, key=lambda r: (r["obs"]["tick"], r["seed"])):
                key = (row["seed"], row["obs"]["tick"])
                if key not in seen:
                    seen.add(key)
                    chosen.append(row)
                if len(chosen) == 3:
                    break
            c.check(len(chosen) == 3, "insufficient observed geometry probes")
            probes.extend(chosen)
    c.check(len(probes) == 36, "bad W0 probe budget")
    return probes


def geometry_eval(w0, model, tok, ids, label, out):
    import torch
    inputs = sampled_probes(w0)
    records = []
    with torch.inference_mode():
        for row in inputs:
            obs = row["obs"]
            old = [float(x) for x in c.score(model, tok, ids, obs)]
            for kind in ("#", "+"):
                altered = outer_variant(obs, kind)
                c.check(altered is not None, "outer-ring variant disappeared")
                other = [float(x) for x in c.score(model, tok, ids, altered)]
                p, q = probs(old), probs(other)
                raw_old = c.ACTIONS[max(range(6), key=lambda i: old[i])]
                raw_new = c.ACTIONS[max(range(6), key=lambda i: other[i])]
                records.append({
                    "seed": row["seed"], "tick": obs["tick"],
                    "family": obs["family"], "substitution": kind,
                    "synthetic_outer_ring_counterfactual": True,
                    "W0_observed_original": True,
                    "original": obs, "counterfactual": altered,
                    "original_scores": old, "changed_scores": other,
                    "original_raw": raw_old, "changed_raw": raw_new,
                    "original_legal": legal_argmax(old, obs),
                    "changed_legal": legal_argmax(other, altered),
                    "raw_flip": raw_old != raw_new,
                    "legal_flip": legal_argmax(old, obs) != legal_argmax(other, altered),
                    "tvd": 0.5 * sum(abs(a - b) for a, b in zip(p, q)),
                })
    c.save_json(out / ("geometry-" + label + ".json"), {
        "schema": SCHEMA, "stage": label, "records": records,
        "n": len(records),
        "raw_flips": sum(x["raw_flip"] for x in records),
        "legal_flips": sum(x["legal_flip"] for x in records),
        "mean_tvd": sum(x["tvd"] for x in records) / len(records),
        "max_tvd": max(x["tvd"] for x in records),
    })
    return {"pairs": len(records),
            "raw_flips": sum(x["raw_flip"] for x in records),
            "legal_flips": sum(x["legal_flip"] for x in records),
            "mean_tvd": sum(x["tvd"] for x in records) / len(records),
            "max_tvd": max(x["tvd"] for x in records)}


def model_episodes(w0, model, tok, ids, stage, out):
    import torch
    rows = []
    for seed in SEEDS:
        for family in c.FAMILIES:
            evidence = []
            def choose(obs):
                with torch.inference_mode():
                    logits = [float(v) for v in c.score(model, tok, ids, obs)]
                raw = c.ACTIONS[max(range(6), key=lambda k: logits[k])]
                action = legal_argmax(logits, obs)
                evidence.append({"observation": obs, "six_logits": logits,
                                 "raw": raw, "raw_legal": raw in obs["legal"],
                                 "host_legal_choice": action})
                return action
            record = w0.run(seed, mode="callback", family=family,
                            steps=MAX_TURNS, selector=choose)
            c.check(len(evidence) == len(record["events"]), "W0 forward/event count drift")
            c.check(all(x["host_legal_choice"] == e["raw_choice"] == e["applied"]
                        and e["raw_legal"] for x, e in zip(evidence, record["events"])),
                    "host authority or model choice discrepancy")
            replay = w0.run(seed, mode="replay", family=family, steps=MAX_TURNS,
                            raw_actions=[x["host_legal_choice"] for x in evidence])
            img = w0.png_bytes(record)
            c.check(img == w0.png_bytes(replay), "original W0 PNG replay changed")
            stem = f"{stage}-{seed}-{family}"
            (out / (stem + ".png")).write_bytes(img)
            c.save_json(out / (stem + ".json"), {"record": record,
                                                "model_forwards": evidence})
            rows.append({
                "stage": stage, "seed": seed, "family": family,
                "first_raw": evidence[0]["raw"] if evidence else None,
                "raw_illegal": sum(not x["raw_legal"] for x in evidence),
                "png": stem + ".png", "png_sha256": c.sha_bytes(img),
                **metrics(record),
            })
    return rows


def controls(w0, out):
    rows = []
    for seed in SEEDS:
        for family in c.FAMILIES:
            for policy in ("random", "wall", "fidget", "triad-teacher"):
                if policy == "triad-teacher":
                    record = w0.run(seed, mode="callback", family=family,
                                    steps=MAX_TURNS, selector=triad.target)
                else:
                    record = w0.run(seed, mode=policy, family=family, steps=MAX_TURNS)
                stem = f"control-{policy}-{seed}-{family}"
                png = w0.png_bytes(record)
                (out / (stem + ".png")).write_bytes(png)
                c.save_json(out / (stem + ".json"), record)
                rows.append({"controller": policy, "seed": seed, "family": family,
                             "png": stem + ".png", "png_sha256": c.sha_bytes(png),
                             **metrics(record)})
    return rows


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, snapshot_download

    torch.set_num_threads(2)
    torch.manual_seed(3504004)
    t0 = time.monotonic()
    out = args.out
    c.check(not out.exists(), "immutable evidence directory already used")
    out.mkdir(parents=True)
    c.check(c.sha(args.adapter / "adapter_model.safetensors") == ADAPTER_SHA,
            "original trained adapter hash mismatch")
    c.save_json(out / "lease.json", {
        "schema": SCHEMA, "source_sha": os.environ.get("GITHUB_SHA"),
        "producer_source": PRODUCER, "producer_run": 43791775878,
        "producer_adapter": ADAPTER_SHA,
        "model": c.MODEL, "revision": c.REVISION,
        "optimizer_updates": 0, "native": False, "draft_hold": True,
        "counterfactuals_are_synthetic": True})
    w0 = c.load_w0(args.w0)
    info = HfApi().model_info(c.MODEL, revision=c.REVISION)
    c.check(info.sha == c.REVISION, "parent pin changed")
    snapshot_download(c.MODEL, revision=c.REVISION, local_dir=args.cache,
        max_workers=2, allow_patterns=[
            "*.safetensors", "*.safetensors.index.json", "config.json",
            "generation_config.json", "tokenizer*", "special_tokens_map.json",
            "added_tokens.json", "vocab.*", "merges.txt", "README.md", "LICENSE*"])
    inventory = {str(p.relative_to(args.cache)): c.sha(p)
                 for p in sorted(args.cache.rglob("*"))
                 if p.is_file() and ".cache" not in p.parts}
    c.check(any(k.endswith(".safetensors") for k in inventory),
            "no full pinned weights")
    c.save_json(out / "model-download.json", {
        "model": c.MODEL, "revision": info.sha, "files": inventory})
    tok = AutoTokenizer.from_pretrained(args.cache, local_files_only=True,
                                        trust_remote_code=False)
    ids = c.id_map(tok)
    base = AutoModelForCausalLM.from_pretrained(args.cache,
        local_files_only=True, trust_remote_code=False, dtype=torch.float32)
    c.check(base.config.model_type == "lfm2" and base.config.num_hidden_layers == 16,
            "not full expected LFM350")
    base.requires_grad_(False)
    base.eval()
    base_identity = c.base_fingerprint(base) if False else None
    stock_probe = geometry_eval(w0, base, tok, ids, "stock", out)
    stock = model_episodes(w0, base, tok, ids, "stock", out)
    # Only one fully resident model. Add the ORIGINAL GP350-003 trained adapter.
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    c.check(not any(p.requires_grad for p in model.parameters()), "model not frozen")
    trained_probe = geometry_eval(w0, model, tok, ids, "trained", out)
    trained = model_episodes(w0, model, tok, ids, "trained", out)
    control_rows = controls(w0, out)
    files = stock + trained + control_rows
    c.check(len(stock) == len(trained) == 9 and len(control_rows) == 36,
            "wrong original image counts")
    for row in files:
        c.check(c.sha(out / row["png"]) == row["png_sha256"],
                "original PNG changed")
    summary = {
        "schema": SCHEMA, "status": "complete", "source_sha": os.environ.get("GITHUB_SHA"),
        "model": c.MODEL, "revision": c.REVISION, "original_adapter_sha256": ADAPTER_SHA,
        "no_additional_optimizer_updates": 0, "native_paint": 0,
        "new_model_score_forwards": sum(x["decisions"] for x in stock + trained)
        + stock_probe["pairs"] * 2 + trained_probe["pairs"] * 2,
        "original_png_count": len(files), "W0_host_legal_applications": True,
        "stock_geometry": stock_probe, "trained_geometry": trained_probe,
        "stock": stock, "trained": trained, "controls": control_rows,
        "wall_seconds": time.monotonic() - t0, "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": [
            "W0 is a synthetic Python host, not native CyberSand",
            "outer-ring geometry edits are explicitly synthetic",
            "trained checkpoint shares synthetic motif teacher bias",
            "the different rollout seeds can begin with identical local observations",
            "d1 matched control not run"],
    }
    c.save_json(out / "summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w0", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        run(args)
    except BaseException as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "failure.json").open("w") as handle:
            json.dump({"kind": type(exc).__name__, "message": str(exc),
                       "source_sha": os.environ.get("GITHUB_SHA"),
                       "no_success_claim": True}, handle)
        raise


if __name__ == "__main__":
    main()
