#!/usr/bin/env python3
"""WORM SPROCKET W8 E5 — genuine root / W8 / frozen-prior / flat W0 rollouts.

Checkpoint-only, source-fenced synthetic host experiment, not native CyberSand.
All scored arms share the same legal action policy, STOP threshold and host RNG.
Historical W6, whole-program step96 and failed evidence remain unmodified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "training"), str(HERE)]
from common import profile, sha256_file, verify_checkpoint, write_json
from lane_adapters import adapter_for
from numerics import configure_cpu
import train as shared
import worm_v1 as contract
import worm_v4 as objective
import worm_score_v6_e3_eval as prior
from worm_score_v5_eval import action_tokens, host_draw, sample_scores, spatial

TRAIN_SOURCE = "de37e65b56bc55b8c6a224478c6a98d5b9828304"
TRAIN_RUN = 43789416142
ARTIFACT_SHA = "bb6b1d5cc2d9ced5a2d051eedd5fcec5fc29aad0fef92157ff7c8256cb46a516"
WORLD_SEEDS = (521, 541)
BUDGET = 48
STOP_DEFER_BEFORE = 40
TEMPERATURE = 2.25


def frozen_first_logits_rollout(w0, model, tokenizer, adapter, family, seed, token_ids):
    """Closed-loop frozen-first *trained* logits: actual W0 states, not off-policy."""
    import torch
    initial = w0.run(seed, "replay", family, steps=1, raw_actions=[])
    obs0 = initial["events"][0]["observation"]
    prefix = adapter.encode_prompt(tokenizer, objective.prompt(obs0), objective.SYSTEM)
    inputs = torch.tensor([prefix], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    first = [float(output.logits[0, -1, i].item()) for i in token_ids]
    assert len(first) == 6
    actions, receipts = [], []
    for tick in range(BUDGET):
        trial = w0.run(seed, "replay", family, steps=tick + 1, raw_actions=actions)
        obs = trial["events"][tick]["observation"]
        legal = prior.policy_legal(obs)
        draw = host_draw(family, seed, tick, TEMPERATURE)
        chosen, scores = sample_scores(first, legal, TEMPERATURE, draw)
        flat, _ = sample_scores([0.0] * 6, legal, TEMPERATURE, draw)
        actions.append(chosen)
        applied = w0.run(seed, "replay", family, steps=tick + 1, raw_actions=actions)
        event = applied["events"][-1]
        if event["observation"] != obs or not event["raw_legal"] or event["applied"] != chosen:
            raise RuntimeError("frozen policy violated real W0 host receipt")
        receipts.append({"tick": tick, "host_selected": chosen, "actual_W0_applied": chosen,
                         "original_W0_legal": obs["legal"], "host_policy_legal": legal,
                         "same_observation_flat": flat, "differs_from_flat": flat != chosen,
                         "trained_first_logit_vector_frozen": True, "live_model_forward": tick == 0,
                         "reused_first_logits": first, "raw_greedy_EOS_completion": False,
                         **prior.attributed_score(scores, "trained")})
        if chosen == "STOP":
            break
    record = w0.run(seed, "replay", family, steps=BUDGET, raw_actions=actions)
    if len(record["events"]) != len(receipts):
        raise RuntimeError("frozen W0 replay differs")
    record["mode"] = "sprocket-w8-e5-frozen"
    record["engine"] = "pinned-Python-W0-trained-tickzero-logits-frozen-not-native"
    record["score_receipts"] = receipts
    return record, receipts


def main(args):
    if args.training_source != TRAIN_SOURCE or args.train_run != TRAIN_RUN or args.checkpoint.name != "checkpoint-000036":
        raise ValueError("wrong W8 original source/run/checkpoint")
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    numerical, runtime = configure_cpu(1)
    selected = profile("sprocket", require_supported=True)
    checkpoint = verify_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    if (cfg.get("schema") != "cybersand.worm.sprocket-local-frontier-contrast-v8"
            or cfg.get("source_sha") != TRAIN_SOURCE or cfg.get("model") != selected
            or cfg.get("worm_common_git_commit") != objective.W0_COMMIT
            or cfg.get("worm_common_sha256") != sha256_file(args.w0)
            or cfg.get("worm_support_git_commit") != objective.SUPPORT_COMMIT
            or cfg.get("synthetic_not_native") is not True or cfg.get("final_opened") is not False
            or cfg.get("numerics") != numerical or cfg.get("rank") != 8
            or cfg.get("threads") != 1):
        raise ValueError("W8 checkpoint source, recipe, processor, W0 or LoRA admission mismatch")
    receipt = json.loads((args.checkpoint / "receipt.json").read_text())
    if (receipt.get("global_step") != 36 or receipt.get("sample_cursor") != 36
            or len(receipt.get("history", [])) != 36):
        raise ValueError("W8 original 36 optimizer updates not admitted")

    from peft import PeftModel
    base, tokenizer = shared.load_base(args.model, selected)
    adapter = adapter_for(selected)
    token_ids, token_map = action_tokens(tokenizer)
    model = PeftModel.from_pretrained(base, args.checkpoint / "adapter",
                                      is_trainable=False, local_files_only=True)
    w0 = contract.load_w0(args.w0)
    prior.BUDGET = BUDGET
    prior.STOP_DEFER_BEFORE = STOP_DEFER_BEFORE
    prior.TEMPERATURE = TEMPERATURE
    rows = []
    for family in contract.FAMILIES:
        for seed in WORLD_SEEDS:
            for mode in ("random", "wall", "fidget"):
                record = w0.run(seed, mode, family, BUDGET)
                w0.write_result(record, args.output / "png-and-json")
                rows.append({"family": family, "world_seed": seed, "mode": mode,
                             "model_forwards": 0, "steps": record["steps_executed"],
                             "metrics": record["metrics"], "spatial": spatial(record)})
            for mode in ("flat", "untuned", "trained", "frozen"):
                if mode == "frozen":
                    record, decisions = frozen_first_logits_rollout(
                        w0, model, tokenizer, adapter, family, seed, token_ids)
                else:
                    record, decisions = prior.rollout(
                        w0, model, tokenizer, adapter, family, seed, mode, token_ids)
                    record["mode"] = "sprocket-w8-e5-" + mode
                w0.write_result(record, args.output / "png-and-json")
                rows.append({
                    "family": family, "world_seed": seed, "mode": mode,
                    "model_forwards": len(decisions) if mode in ("trained", "untuned") else
                                      (1 if mode == "frozen" else 0),
                    "steps": record["steps_executed"], "metrics": record["metrics"],
                    "spatial": spatial(record), "early_stop": record["steps_executed"] < BUDGET,
                    "decision_count": len(decisions),
                    "invalid_applied": sum(r["host_selected"] != r["actual_W0_applied"] for r in decisions),
                    "differs_same_observation_flat": sum(r["differs_from_flat"] for r in decisions),
                    "action_histogram": {a: sum(r["host_selected"] == a for r in decisions)
                                         for a in contract.ACTIONS},
                })
                write_json(args.output / "partial-progress.json",
                           {"status": "partial", "rows": rows, "optimizer_updates_here": 0})
    write_json(args.output / "comparison.json", {
        "schema": "cybersand.worm.sprocket-v8-e5-live-vs-frozen-original-W0",
        "train_source": TRAIN_SOURCE, "train_run": TRAIN_RUN,
        "training_artifact_digest_sha256": ARTIFACT_SHA,
        "checkpoint": "checkpoint-000036",
        "checkpoint_inventory_sha256": sha256_file(args.checkpoint / "inventory.json"),
        "model": selected["model_id"], "model_revision": selected["revision"],
        "W0_common_commit": objective.W0_COMMIT,
        "world_seeds": list(WORLD_SEEDS), "families": list(contract.FAMILIES),
        "steps_per_trajectory": BUDGET, "temperature": TEMPERATURE,
        "stop_deferred_before_tick": STOP_DEFER_BEFORE,
        "comparability": "W8/root/flat/frozen use identical host legal and STOP deferral, with same SHA host draws per seed/tick. Controls use original W0 rules.",
        "frozen_arm": "one genuine trained forward on tick-zero observation; those same logits sampled in all later evolving W0 observations",
        "flat_arm": "uniform six logits, no neural inference",
        "conditioned_scored_arms": ["trained", "untuned"],
        "status": "complete", "optimizer_updates_here": 0, "native": False,
        "sealed_final_opened": False, "rows": rows,
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "token_mapping": token_map, "numerical": numerical, "runtime": runtime})
    print(json.dumps({"status": "complete", "worlds": 6, "total_original_W0_pngs": len(rows),
                      "genuine_trained_forwards": sum(r["model_forwards"] for r in rows if r["mode"] == "trained")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--w0", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--training-source", required=True)
    p.add_argument("--train-run", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        main(args)
    except BaseException as exc:
        if args.output.exists():
            write_json(args.output / "failure.json",
                       {"type": type(exc).__name__, "message": str(exc),
                        "failed_or_partial": True, "not_success": True})
        raise
