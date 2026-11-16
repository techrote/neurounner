#!/usr/bin/env python3
"""SPROCKET WORM S2: host-seeded sampling of the W4 checkpoint's actual six-token logits.

This is a deliberately NEW experimental policy, not a W4 greedy/EOS completion
and NOT additional training, native painting, or an Art-v2 provider. The original
W4 greedy first-pass evidence is preserved in its immutable run/artifacts.
The host alone owns legal masking, RNG, state, action application and Stop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1] / "training"
sys.path[:0] = [str(TRAINING), str(HERE)]
from common import profile, sha256_file, verify_checkpoint, write_json
from lane_adapters import adapter_for
from numerics import configure_cpu
import train as shared
import worm_v1 as contract
import worm_v4 as objective

TRAIN_SOURCE = "c14795efe978dadeaa9e0da60f6273e18ab17b4d"
TRAIN_RUN = 43710874563
TEMPERATURES = (1.0, 3.0)
BUDGET = 24


def host_draw(family: str, seed: int, tick: int, temp: float) -> float:
    """Deterministic independent uniform variate, not any model-owned RNG."""
    payload = f"612:sprocket:score-crn-s2:{seed}:{tick}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / (1 << 64)


def sample_scores(logits, legal, temperature: float, draw: float):
    """Six actual learned logits, legal-only softmax, strictly host-seeded action."""
    if temperature <= 0 or not math.isfinite(temperature) or not 0 <= draw < 1:
        raise ValueError("unsafe sampling controls")
    if len(logits) != 6 or not all(math.isfinite(x) for x in logits):
        raise ValueError("nonfinite/missing model logits")
    if not legal or set(legal) - set(contract.ACTIONS) or len(legal) != len(set(legal)):
        raise ValueError("invalid host legality")
    active = [i for i, action in enumerate(contract.ACTIONS) if action in legal]
    scaled = [float(logits[i]) / temperature for i in active]
    high = max(scaled)
    exps = [math.exp(v - high) for v in scaled]
    total = sum(exps)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid softmax denominator")
    probs = {a: 0.0 for a in contract.ACTIONS}
    for i, e in zip(active, exps):
        probs[contract.ACTIONS[i]] = e / total
    cumulative = 0.0
    chosen = contract.ACTIONS[active[-1]]
    for i in active:
        cumulative += probs[contract.ACTIONS[i]]
        if draw < cumulative:
            chosen = contract.ACTIONS[i]
            break
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    effective_count = 1 / sum(p * p for p in probs.values() if p > 0)
    raw_top = contract.ACTIONS[max(range(6), key=lambda i: logits[i])]
    return chosen, {
        "full_six_token_logits": dict(zip(contract.ACTIONS, map(float, logits))),
        "raw_unmasked_top_action": raw_top,
        "raw_top_host_legal": raw_top in legal,
        "masked_probs": probs,
        "temperature": temperature, "host_uniform_draw": draw,
        "entropy_nats": entropy, "effective_actions": effective_count,
        "selected_action": chosen, "selected_host_legal": chosen in legal,
        "is_greedy_decoding": False, "is_eos5_completion": False,
        "is_model_logit_conditioned": True
    }


def action_tokens(tokenizer):
    mapping = {action: tokenizer.encode(code, add_special_tokens=False)
               for action, code in objective.ACTION_TO_CODE.items()}
    if (any(len(v) != 1 or v[0] < 16 for v in mapping.values()) or
        len({v[0] for v in mapping.values()}) != 6 or
        set(mapping) != set(contract.ACTIONS)):
        raise ValueError("W4 fixed one-token action coding violated")
    return [mapping[a][0] for a in contract.ACTIONS], mapping


def rollout(w0, model, tokenizer, adapter, family, seed, temperature, budget, token_ids):
    import torch
    actions, receipts = [], []
    model.eval()
    for tick in range(budget):
        probe = w0.run(seed, "replay", family, steps=tick + 1, raw_actions=actions)
        if len(probe["events"]) != tick + 1:
            raise ValueError("W0 stopped early before local model observation")
        obs = probe["events"][tick]["observation"]
        text = objective.prompt(obs)
        prefix = adapter.encode_prompt(tokenizer, text, objective.SYSTEM)
        ids = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False)
        logits = [float(output.logits[0, -1, t].item()) for t in token_ids]
        draw = host_draw(family, seed, tick, temperature)
        selected, score_receipt = sample_scores(logits, obs["legal"], temperature, draw)
        paired_uniform, _ = sample_scores([0.0] * 6, obs["legal"], temperature, draw)
        actions.append(selected)
        applied = w0.run(seed, "replay", family, steps=tick + 1, raw_actions=actions)
        event = applied["events"][-1]
        if (event["observation"] != obs or event["raw_choice"] != selected or
            not event["raw_legal"] or event["applied"] != selected):
            raise ValueError("W0 host rejected selected action or observation drift")
        receipts.append({"tick": tick, "observation_sha256": hashlib.sha256(
                         json.dumps(obs, sort_keys=True, separators=(",", ":")).encode()
                         ).hexdigest(), "prompt": text, "prefix_ids": prefix,
                         "action_token_ids": token_ids,
                         "host_legal": obs["legal"],
                         "paired_uniform_same_observation": paired_uniform,
                         "paired_differs_from_uniform": selected != paired_uniform,
                         **score_receipt,
                         "applied_by_host": event["applied"]})
        if event["applied"] == "STOP":
            break
    record = w0.run(seed, "replay", family, steps=budget, raw_actions=actions)
    if len(record["events"]) != len(receipts):
        raise ValueError("final W0 source replay differs from recorded decisions")
    return record, receipts


def spatial(record):
    points = record["trajectory"]
    if not points:
        raise ValueError("empty trajectory")
    xs, ys = zip(*points)
    sx, sy = points[0]
    return {"unique_cells": len({tuple(p) for p in points}),
            "bbox_width": max(xs)-min(xs)+1,
            "bbox_height": max(ys)-min(ys)+1,
            "max_manhattan_from_start": max(abs(x-sx)+abs(y-sy) for x, y in points)}


def main(args):
    if args.training_source != TRAIN_SOURCE or args.steps != BUDGET:
        raise ValueError("immutable S2 source/step lease mismatch")
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    numerical, runtime = configure_cpu(1)
    selected = profile("sprocket", require_supported=True)
    w0 = contract.load_w0(args.w0)
    checkpoint = verify_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    if (cfg.get("schema") != objective.SCHEMA or cfg.get("model") != selected or
        cfg.get("source_sha") != TRAIN_SOURCE or
        cfg.get("worm_common_git_commit") != objective.W0_COMMIT or
        cfg.get("worm_common_sha256") != sha256_file(args.w0) or
        cfg.get("worm_support_git_commit") != objective.SUPPORT_COMMIT or
        cfg.get("synthetic_not_native") is not True or cfg.get("final_opened") is not False or
        cfg.get("numerics") != numerical):
        raise ValueError("wrong model/W0/adapter/numerical source or FINAL unsafe")
    import torch
    from peft import PeftModel
    base, tokenizer = shared.load_base(args.model, selected)
    adapter = adapter_for(selected)
    token_ids, token_mapping = action_tokens(tokenizer)
    tuned = PeftModel.from_pretrained(base, args.checkpoint / "adapter",
                                      is_trainable=False, local_files_only=True)
    rows = []
    for family in contract.FAMILIES:
        for seed in contract.EVAL_SEEDS:
            for mode in ("random", "wall", "fidget"):
                record = w0.run(seed, mode, family, steps=args.steps)
                w0.write_result(record, args.output / "png-and-json")
                rows.append({"family": family, "seed": seed, "mode": mode,
                             "model": False, "metrics": record["metrics"],
                             "spatial": spatial(record)})
            for temperature in TEMPERATURES:
                name = f"sprocket-scored-t{temperature:g}"
                record, receipts = rollout(w0, tuned, tokenizer, adapter, family, seed,
                                           temperature, args.steps, token_ids)
                record["mode"] = name
                record["engine"] = "W0-model-logit-host-seeded-sampler-not-native"
                record["score_receipts"] = receipts
                w0.write_result(record, args.output / "png-and-json")
                rows.append({"family": family, "seed": seed, "mode": name,
                             "model": True, "temperature": temperature,
                             "metrics": record["metrics"], "spatial": spatial(record),
                             "sampled_decisions": len(receipts),
                             "mean_effective_actions": sum(x["effective_actions"] for x in receipts) / len(receipts),
                             "raw_top_distribution": {a: sum(x["raw_unmasked_top_action"] == a
                                             for x in receipts) for a in contract.ACTIONS},
                             "selected_distribution": {a: sum(x["selected_action"] == a
                                             for x in receipts) for a in contract.ACTIONS},
                             "paired_diff_from_uniform": sum(x["paired_differs_from_uniform"] for x in receipts),
                             "active_decisions": len(receipts),
                             "early_stop": len(receipts) < args.steps,
                             "first_step_logits": receipts[0]["full_six_token_logits"],
                             "non_greedy_sampled": sum(x["selected_action"] != x["raw_unmasked_top_action"]
                                                       for x in receipts)})
                write_json(args.output / "partial-progress.json",
                           {"status": "partial-until-comparison", "rows": rows,
                            "training_steps_this_job": 0})
    write_json(args.output / "comparison.json", {
        "schema": "cybersand.worm.sprocket-live-crn-ablation-s2",
        "policy": "live W4 trained logits T1/T3 with common world-tick RNG",
        "common_rng": "SHA256(612:sprocket:score-crn-s2:world_seed:tick)",
        "same_observation_uniform_ablation": True,
        "no_off_policy_score_tape": True,
        "not_greedy_eos5": True, "original_greedy_w4_run": 43721058235,
        "training_run": TRAIN_RUN, "training_source": TRAIN_SOURCE,
        "checkpoint_inventory_sha256": sha256_file(args.checkpoint / "inventory.json"),
        "model": selected["model_id"], "revision": selected["revision"],
        "w0_commit": objective.W0_COMMIT, "token_mapping": token_mapping,
        "seed_groups": list(contract.EVAL_SEEDS), "families": list(contract.FAMILIES),
        "steps_per_trajectory": args.steps, "temperatures": list(TEMPERATURES),
        "model_rollouts": sum(r["model"] for r in rows),
        "controls": sum(not r["model"] for r in rows),
        "paired_uniform_action_count": sum(r.get("active_decisions",0) for r in rows if r["model"]),
        "completed": True, "actual_optimizer_steps_this_job": 0,
        "native": False, "full_art_v2": False, "sealed_final_opened": False,
        "rows": rows, "wall_seconds": time.monotonic()-started,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "numerics": numerical, "numerical_runtime": runtime})
    print(json.dumps({"result": "source-faithful W4 model-score sampler",
                      "model_rollouts": sum(r["model"] for r in rows),
                      "controls": sum(not r["model"] for r in rows)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--w0", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--training-source", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        main(args)
    except BaseException as exc:
        if args.output.exists():
            write_json(args.output / "failure.json", {
                "type": type(exc).__name__, "message": str(exc),
                "status": "FAILED/PARTIAL", "not_success": True})
        raise
