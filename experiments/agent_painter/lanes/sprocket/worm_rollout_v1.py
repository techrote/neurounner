#!/usr/bin/env python3
"""WORM-001: independent model-in-the-loop W0 replay and original PNGs.

Each choice is generated once from the CURRENT compact W0 observation. W0 is
reexecuted deterministically to obtain the next observation; no patched host,
teacher-policy fallback, JSON repair, or future-state leak is allowed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1] / "training"
sys.path[:0] = [str(TRAINING), str(HERE)]
import train as shared
from common import profile, sha256_file, verify_checkpoint, write_json
from lane_adapters import adapter_for
from numerics import configure_cpu
import worm_v1 as contract


def choose(model, tokenizer, adapter, observation):
    import torch
    text = contract.prompt(observation)
    prefix = adapter.encode_prompt(tokenizer, text, contract.SYSTEM)
    with torch.no_grad():
        output = model.generate(input_ids=torch.tensor([prefix], dtype=torch.long),
                                do_sample=False, max_new_tokens=12, eos_token_id=5,
                                pad_token_id=1, use_cache=True)
    ids = output[0].tolist()[len(prefix):]
    eos = bool(ids) and ids[-1] == 5 and ids.count(5) == 1
    payload = ids[:-1] if eos else ids
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    choice = tokenizer.decode(payload, skip_special_tokens=False)
    permitted = eos and choice in observation["legal"] and choice in contract.ACTIONS and all(i >= 16 for i in payload)
    host_choice = choice if permitted else raw
    return host_choice, {"observation": observation, "prompt": text, "prefix_ids": prefix,
                         "completion_ids": ids, "raw_output": raw,
                         "eos5": eos, "parsed_action_exact": choice if permitted else None,
                         "legal_raw": bool(permitted), "host_choice": host_choice}


def roll(w0, model, tokenizer, adapter, seed, family, budget):
    # This replay is the actual pinned W0 state machine, not a locally reimplemented approximation.
    choices = []
    receipts = []
    model.eval()
    for tick in range(budget):
        probe = w0.run(seed, "replay", family, steps=tick + 1, raw_actions=choices)
        if len(probe["events"]) != tick + 1:
            raise ValueError("W0 deterministic replay ended before the current observation")
        observation = probe["events"][tick]["observation"]
        raw, receipt = choose(model, tokenizer, adapter, observation)
        choices.append(raw)
        applied = w0.run(seed, "replay", family, steps=tick + 1, raw_actions=choices)
        event = applied["events"][-1]
        if event["raw_choice"] != raw or event["observation"] != observation:
            raise ValueError("W0 policy replay or observation consistency lost")
        receipt["tick"] = tick
        receipt["applied"] = event["applied"]
        receipt["invalid_host_action"] = not event["raw_legal"]
        receipts.append(receipt)
        if event["applied"] == "STOP":
            break
    record = w0.run(seed, "replay", family, steps=budget, raw_actions=choices)
    if len(record["events"]) != len(receipts):
        raise ValueError("W0 final replay length drift")
    return record, receipts


def main(args):
    start = time.monotonic()
    if not 1 <= args.steps <= 64:
        raise ValueError("must bound expensive model-in-the-loop steps")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    numerical, _ = configure_cpu(1)
    selected = profile("sprocket", require_supported=True)
    w0 = contract.load_w0(args.w0)
    source_receipt = verify_checkpoint(args.checkpoint)
    cfg = source_receipt["config"]
    if (cfg.get("schema") != contract.SCHEMA or
        cfg.get("model") != selected or
        cfg.get("worm_common_git_commit") != contract.W0_COMMIT or
        cfg.get("worm_common_sha256") != sha256_file(args.w0) or
        cfg.get("source_sha") != args.training_source or
        cfg.get("synthetic_not_native") is not True or
        cfg.get("final_opened") is not False):
        raise ValueError("WORM adapter objective/model/source checkpoint contract mismatch")
    if cfg["numerics"] != numerical:
        raise ValueError("CPU numerical recipe mismatched training source")
    import torch
    from peft import PeftModel
    base, tokenizer = shared.load_base(args.model, selected)
    adapter = adapter_for(selected)
    run_summaries = []
    for family in contract.FAMILIES:
        for seed in contract.EVAL_SEEDS:
            for mode in ("random", "wall", "fidget"):
                record = w0.run(seed, mode, family, steps=args.steps)
                w0.write_result(record, out / "png-and-json")
                run_summaries.append({"family": family, "seed": seed, "mode": mode,
                                      "metrics": record["metrics"], "model": False})
    for label, policy in (("untuned", base),):
        for family in contract.FAMILIES:
            for seed in contract.EVAL_SEEDS:
                record, receipts = roll(w0, policy, tokenizer, adapter, seed, family, args.steps)
                record["mode"] = "sprocket-" + label
                record["engine"] = "W0-source-faithful-model-first-pass-not-native"
                record["model_first_passes"] = receipts
                w0.write_result(record, out / "png-and-json")
                run_summaries.append({"family": family, "seed": seed, "mode": label,
                                      "metrics": record["metrics"], "model": True,
                                      "eos5": sum(x["eos5"] for x in receipts),
                                      "legal_raw": sum(x["legal_raw"] for x in receipts),
                                      "decisions": len(receipts)})
    # Separate immutable trained adapter is loaded only AFTER unmodified base evidence.
    tuned = PeftModel.from_pretrained(base, args.checkpoint / "adapter", is_trainable=False, local_files_only=True)
    for family in contract.FAMILIES:
        for seed in contract.EVAL_SEEDS:
            record, receipts = roll(w0, tuned, tokenizer, adapter, seed, family, args.steps)
            record["mode"] = "sprocket-trained"
            record["engine"] = "W0-source-faithful-model-first-pass-not-native"
            record["model_first_passes"] = receipts
            w0.write_result(record, out / "png-and-json")
            run_summaries.append({"family": family, "seed": seed, "mode": "trained",
                                  "metrics": record["metrics"], "model": True,
                                  "eos5": sum(x["eos5"] for x in receipts),
                                  "legal_raw": sum(x["legal_raw"] for x in receipts),
                                  "decisions": len(receipts)})
    write_json(out / "comparison.json", {"schema": contract.SCHEMA + "/rollout1",
                "pinned_w0": contract.W0_COMMIT, "source_training": args.training_source,
                "adapter_inventory_sha256": sha256_file(args.checkpoint / "inventory.json"),
                "model": selected["model_id"], "revision": selected["revision"],
                "budget_per_trajectory": args.steps, "seed_pairs": list(contract.EVAL_SEEDS),
                "families": list(contract.FAMILIES), "rollouts": run_summaries,
                "actual_native_art": False, "full_art_v2": False, "final_opened": False,
                "wall_seconds": time.monotonic() - start,
                "cpu_seconds": time.process_time(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(json.dumps({"status": "model-rollouts-complete", "count": len(run_summaries),
                      "model_rollouts": sum(x["model"] for x in run_summaries)}))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--w0", required=True, type=Path)
    a.add_argument("--model", required=True, type=Path)
    a.add_argument("--checkpoint", required=True, type=Path)
    a.add_argument("--training-source", required=True)
    a.add_argument("--steps", type=int, default=24)
    a.add_argument("--output", required=True, type=Path)
    args = a.parse_args()
    try:
        main(args)
    except BaseException as error:
        if args.output.exists():
            write_json(args.output / "failure.json", {"type": type(error).__name__,
                    "error": str(error), "status": "failed-or-partial", "do_not_claim_success": True})
        raise
