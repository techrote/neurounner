#!/usr/bin/env python3
"""WORM-001 SPROCKET W7: within-family local-frontier six-action steering.

WORM synthetic steering does NOT impersonate Art v2 native-admitted TRAIN. The
pinned loader, model bytes, CPU numerical profile, LoRA boundary, atomic checkpoint
mechanics, and optimizer checks are reused without changing shared source.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(HERE))
import train as shared
from common import digest_json, profile, sha256_file, write_json
from lane_adapters import adapter_for, adapter_identity
from numerics import configure_cpu
import worm_v1 as contract
import worm_v4 as objective

STOP = False


def stop_handler(_signum, _frame):
    global STOP
    STOP = True


def first_pass(model, tokenizer, adapter, rows, label, out):
    """Unmodified greedy token decisions, true EOS5 required, no retry or repair."""
    import torch
    model.eval()
    results = []
    for row in rows:
        prefix = adapter.encode_prompt(tokenizer, row["prompt"], objective.SYSTEM)
        with torch.no_grad():
            generated = model.generate(input_ids=torch.tensor([prefix], dtype=torch.long),
                                       do_sample=False, max_new_tokens=12, pad_token_id=1,
                                       eos_token_id=5, use_cache=True)
        ids = generated[0].tolist()[len(prefix):]
        has_eos = 5 in ids and ids[-1] == 5
        content_ids = ids[:-1] if has_eos else ids
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        choice = tokenizer.decode(content_ids, skip_special_tokens=False)
        decoded = objective.CODE_TO_ACTION.get(choice)
        legal = has_eos and decoded in row["legal"] and all(i >= 16 for i in content_ids)
        results.append({"id": row["id"], "family": row["family"], "seed": row["seed"],
                        "prefix_ids": prefix, "completion_ids": ids, "raw": raw,
                        "content_code": choice, "decoded_action": decoded, "eos5": has_eos, "raw_legal": legal,
                        "applied_if_hosted": decoded if legal else ("RETRACE" if "RETRACE" in row["legal"] else "STOP"),
                        "legal_actions": row["legal"], "teacher_reference_not_quality_gate": row["action"],
                        "teacher_code": row["completion"]})
    write_json(out / (label + "-raw-first-pass.json"), results)
    return {"decisions": len(results), "eos": sum(x["eos5"] for x in results),
            "legal": sum(x["raw_legal"] for x in results)}


def first_target_position(labels):
    """Precommitted first action token after a fully masked root-Instruct prefix."""
    idx = next((i for i, v in enumerate(labels) if v != -100), None)
    if idx is None or idx < 1 or labels[idx] < 16 or labels[-1] != 5:
        raise ValueError("target framing, action token or EOS5 corrupted")
    if any(v != -100 for v in labels[:idx]):
        raise ValueError("prompt unmasked, would leak teacher into loss")
    return idx


def score_probe(model, tokenizer, adapter, rows, class_ids, label, out):
    """Actual held-out six-code logits; NOT raw EOS completion or full-vocabulary probabilities."""
    import torch
    from collections import Counter
    model.eval()
    results = []
    for row in rows:
        prefix = adapter.encode_prompt(tokenizer, row["prompt"], objective.SYSTEM)
        with torch.no_grad():
            ids = torch.tensor([prefix], dtype=torch.long)
            values = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False).logits[0, -1, class_ids].float()
            probabilities = torch.softmax(values, dim=-1)
        if not torch.isfinite(values).all().item() or not torch.isfinite(probabilities).all().item():
            raise ValueError("nonfinite class probe output")
        raw_index = int(values.argmax().item())
        masked = [i for i, a in enumerate(contract.ACTIONS) if a in row["legal"]]
        chosen_index = max(masked, key=lambda i: float(values[i].item()))
        results.append({"id": row["id"], "family": row["family"],
                        "seed": row["seed"], "prefix_ids": prefix,
                        "model_class_logits": dict(zip(contract.ACTIONS, [float(x) for x in values])),
                        "conditional_six_way_probabilities": dict(zip(contract.ACTIONS, [float(x) for x in probabilities])),
                        "unmasked_top": contract.ACTIONS[raw_index],
                        "unmasked_top_legal": contract.ACTIONS[raw_index] in row["legal"],
                        "host_masked_top": contract.ACTIONS[chosen_index],
                        "reference_teacher": row["action"],
                        "host_masked_teacher_match": contract.ACTIONS[chosen_index] == row["action"],
                        "not_raw_eos_generation": True})
    write_json(out / (label + "-six-way-first-scores.json"), results)
    return {"count": len(results), "masked_correct": sum(x["host_masked_teacher_match"] for x in results),
            "distinct_unmasked_classes": len({x["unmasked_top"] for x in results}),
            "unmasked_counts": dict(Counter(x["unmasked_top"] for x in results)),
            "host_legal_raw_top": sum(x["unmasked_top_legal"] for x in results)}



def counterfactual_pairs(rows, support, scope):
    """Anchor to executed W0 observations; change one local frontier mark only.

    The second observation is a deliberately synthetic occupancy counterfactual,
    NOT a W0 replay, native admission or an observed state. Family, time, world
    seed, heading, location and action history are identical within each pair.
    The old teacher is independently re-run after the one-cell edit. Host BRANCH
    legality is only *removed* when its fresh-arm prerequisite ceases to hold.
    """
    from collections import defaultdict
    if scope not in ("train", "eval"):
        raise ValueError("unknown W7 split")
    buckets = defaultdict(list)
    seen = set()
    rotations = {"STRAIGHT": 0, "LEFT": -1, "RIGHT": 1}
    for row in sorted(rows, key=lambda r: r["id"]):
        original_obs = row["observation"]
        original_action = support.teacher_choice(original_obs)
        if original_action != row["action"]:
            raise ValueError("pinned teacher source drift: " + row["id"])
        if original_action not in rotations:
            continue
        direction = (original_obs["heading"] + rotations[original_action]) % 4
        dx, dy = support.worm.DIRECTIONS[direction]
        grid = [list(line) for line in original_obs["local_5x5"]]
        if grid[2 + dy][2 + dx] != ".":
            continue
        altered = deepcopy(original_obs)
        grid[2 + dy][2 + dx] = "+"
        altered["local_5x5"] = ["".join(line) for line in grid]
        fresh = 0
        for move, rotation in rotations.items():
            direction = (altered["heading"] + rotation) % 4
            fx, fy = support.worm.DIRECTIONS[direction]
            if move in altered["legal"] and grid[2 + fy][2 + fx] == ".":
                fresh += 1
        if "BRANCH" in altered["legal"] and fresh < 2:
            altered["legal"] = [a for a in altered["legal"] if a != "BRANCH"]
        support.validate_observation(altered)
        target = support.teacher_choice(altered)
        if target == original_action or target not in altered["legal"]:
            continue
        old_prompt = objective.prompt(original_obs)
        new_prompt = objective.prompt(altered)
        key = (old_prompt, new_prompt)
        if key in seen:
            continue
        seen.add(key)
        pair = (
            {"id": row["id"], "prompt": old_prompt,
             "completion": objective.ACTION_TO_CODE[original_action],
             "action": original_action, "family": original_obs["family"],
             "seed": row["seed"], "legal": original_obs["legal"],
             "observation": original_obs, "origin": "actual-pinned-W0-support-row"},
            {"id": row["id"] + "-w7-one-neighbour-visited", "prompt": new_prompt,
             "completion": objective.ACTION_TO_CODE[target],
             "action": target, "family": altered["family"], "seed": row["seed"],
             "legal": altered["legal"], "observation": altered,
             "origin": "synthetic-one-local-frontier-mark-not-W0-replayed"}
        )
        buckets[(original_obs["family"], original_action, target)].append(pair)
    # Avoid repeating one favourite action: deterministic family/transition round
    # robin; hash-order within each class. No labels or heldout rows are invented.
    for items in buckets.values():
        items.sort(key=lambda pair: hashlib.sha256(
            ("worm-v7:" + pair[0]["prompt"] + pair[1]["prompt"]).encode()).digest())
    categories = sorted(buckets)
    result = []
    limit = 72 if scope == "train" else 36
    for round_index in range(max((len(buckets[k]) for k in categories), default=0)):
        for category in categories:
            items = buckets[category]
            if round_index < len(items):
                result.append(items[round_index])
                if len(result) == limit:
                    break
        if len(result) == limit:
            break
    if len(result) < 6:
        raise ValueError("insufficient original-W0-anchored local W7 contrast pairs")
    return result

def pair_probe(model, tokenizer, adapter, pairs, class_ids, label, out):
    """Same-family one-local-mark opposing-label six-action scores; synthetic foil."""
    import torch
    model.eval()
    results = []
    with torch.no_grad():
        for left, right in pairs[:9]:
            item = {"left_id": left["id"], "right_id": right["id"],
                    "reference": [left["action"], right["action"]], "pairs": []}
            for own, opponent in ((left, right), (right, left)):
                prefix = adapter.encode_prompt(tokenizer, own["prompt"], objective.SYSTEM)
                inputs = torch.tensor([prefix], dtype=torch.long)
                logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                               use_cache=False).logits[0, -1, class_ids].float()
                if not torch.isfinite(logits).all().item():
                    raise ValueError("nonfinite paired score")
                a = contract.ACTIONS.index(own["action"])
                b = contract.ACTIONS.index(opponent["action"])
                item["pairs"].append({"id": own["id"], "family": own["family"],
                    "prefix_ids": prefix, "teacher": own["action"],
                    "contrasting_teacher": opponent["action"],
                    "class_logits": [float(v) for v in logits],
                    "teacher_minus_other_logit": float((logits[a] - logits[b]).item()),
                    "unmasked_top": contract.ACTIONS[int(logits.argmax())],
                    "host_legal": contract.ACTIONS[int(logits.argmax())] in own["legal"]})
            results.append(item)
    write_json(out / (label + "-local-frontier-pair-scores.json"), results)
    return {"pairs": len(results),
            "both_correct_unmasked": sum(
              all(p["unmasked_top"] == p["teacher"] for p in r["pairs"]) for r in results),
            "both_positive_target_margin": sum(
              all(p["teacher_minus_other_logit"] > 0 for p in r["pairs"]) for r in results)}


def main(args):
    if args.steps < 1 or args.steps > 18 or args.deadline < 90 or args.deadline > 1800:
        raise ValueError("WORM v7 must remain a finite 1-18 step paired campaign")
    started = time.monotonic()
    selected = profile("sprocket", require_supported=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "lease.json", {"source_sha": shared.source_sha(),
              "model": selected["model_id"], "revision": selected["revision"],
              "experiment": "worm-sprocket-local-frontier-v7-fresh", "steps": args.steps,
              "deadline_seconds": args.deadline, "run_id": os.environ.get("GITHUB_RUN_ID"),
              "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "native": False})
    numeric, numerical_runtime = configure_cpu(1)
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(612)
    random.seed(612)

    support = objective.load_support(args.support, args.w0)
    w0 = support.worm
    datasets, generated = objective.dataset(support, args.output / "dataset")
    train_rows = datasets["train"]
    heldout_rows = datasets["eval"]
    train_pairs = counterfactual_pairs(train_rows, support, "train")
    heldout_pairs = counterfactual_pairs(heldout_rows, support, "eval")
    if len(train_pairs) < 6:
        raise ValueError("too few structurally distinct training counterfactual pairs")
    if {r["seed"] for pair in train_pairs for r in pair if r["seed"] is not None} & {
            r["seed"] for pair in heldout_pairs for r in pair if r["seed"] is not None}:
        raise ValueError("WORM v7 paired world split overlap")

    splits = {name: {"sha256": generated[name]["sha256"], "rows": generated[name]["rows"],
                     "actions": generated[name]["by_action"]} for name in ("train", "eval")}
    adapter = adapter_for(selected)
    base, tokenizer = shared.load_base(args.model, selected)
    # W5 digit coding must be one root-Instruct noncontrol lexical token each.
    action_token_ids = {code: tokenizer.encode(code, add_special_tokens=False)
                        for code in objective.CODE_TO_ACTION}
    if any(len(ids) != 1 or ids[0] < 16 for ids in action_token_ids.values()):
        raise ValueError("W5 code action must encode as exactly one noncontrol token")
    if len({ids[0] for ids in action_token_ids.values()}) != 6:
        raise ValueError("W5 code token collision")
    encoded_pairs = [tuple(adapter.encode_row(tokenizer, r, objective.SYSTEM, 2048)
                            for r in pair) for pair in train_pairs]
    encoded = [item for pair in encoded_pairs for item in pair]
    if max(len(x["input_ids"]) for x in encoded) > 2048 or any(x["target_tokens"] < 2 for x in encoded):
        raise ValueError("unsafe synthetic framing or truncated action target")
    # Six exactly pinned action tokens, trained as alternatives at the action position.
    class_token_ids = [action_token_ids[objective.ACTION_TO_CODE[a]][0]
                       for a in contract.ACTIONS]
    # Exactly paired and contrasting TRAIN families per optimizer step; source-hash
    # ordering yields deterministic identities even if upstream row order changes.
    order = [index % len(train_pairs) for index in range(args.steps)]
    packages = {name: importlib.metadata.version(name) for name in shared.PACKAGES}
    w0_hash = sha256_file(args.w0)
    cfg = {"schema": "cybersand.worm.sprocket-local-frontier-contrast-v7", "model": selected, "source_sha": shared.source_sha(),
           "objective": "v7 within-family one-frontier-mark synthetic counterfactual six-way CE+contrastive margin; no EOS optimizer target",
           "action_to_code": objective.ACTION_TO_CODE, "action_code_token_ids": action_token_ids,
           "class_token_ids": class_token_ids, "loss": "per-pair mean 0.7*(six-way CE label_smoothing=.15) + .3*softplus(.5-(teacher_logit-other_teacher_logit))",
           "sampler": "18 optimizer updates from 72 deterministic within-family W0-anchored one-visited-mark pairs; category round-robin",
           "worm_common_git_commit": objective.W0_COMMIT, "worm_common_sha256": w0_hash,
           "worm_support_git_commit": objective.SUPPORT_COMMIT,
           "worm_support_sha256": sha256_file(args.support),
           "model_files": json.loads((args.model / "download_receipt.json").read_text())["files"],
           "synthetic_train_sha256": splits["train"]["sha256"],
           "synthetic_eval_sha256": splits["eval"]["sha256"],
           "synthetic_not_native": True, "final_opened": False,
           "training_code_sha256": sha256_file(__file__), "data_code_sha256": sha256_file(objective.__file__),
           "shared_trainer_sha256": sha256_file(TRAINING / "train.py"),
           "shared_checkpoint_contract_sha256": sha256_file(TRAINING / "common.py"),
           "adapter_contract": adapter_identity(selected),
           "chat_template_hash": digest_json(tokenizer.chat_template),
           "system_hash": digest_json(objective.SYSTEM), "system": objective.SYSTEM,
           "sample_order": [[r["id"] for r in train_pairs[i]] for i in order],
           "training_pair_count": len(train_pairs), "heldout_pair_count": len(heldout_pairs),
           "training_pair_hash": digest_json([[r["prompt"], r["action"]] for pair in train_pairs for r in pair]),
           "development_pair_hash": digest_json([[r["prompt"], r["action"]] for pair in heldout_pairs for r in pair]),
           "literal_train_eval_overlap": generated["literal_prompt_intersection"],
           "numerics": numeric, "packages": packages, "python": platform.python_version(),
           "seed": 612, "learning_rate": 0.00015, "rank": 8, "alpha": 16, "dropout": 0.05,
           "optimizer": "AdamW betas=(0.9,0.95) weight_decay=0.01",
           "max_length": 2048, "threads": 1, "dtype": "float32", "device": "cpu"}
    write_json(args.output / "preflight.json",
               {"config": cfg, "train_rows": len(train_rows), "heldout_rows": len(heldout_rows),
                "train_action_counts": splits["train"]["actions"], "eval_action_counts": splits["eval"]["actions"],
                "max_train_tokens": max(len(x["input_ids"]) for x in encoded),
                "action_code_token_ids": action_token_ids,
                "train_eos_masks_pass": all(x["input_ids"][-1] == 5 and x["labels"][-1] == 5 for x in encoded),
                "numerical_runtime": numerical_runtime,
                "baseline_w0": "source-faithful, not native"})
    write_json(args.output / "paired-curriculum-manifest.json", {
        "schema": "cybersand.worm.sprocket-local-frontier-v7",
        "train_pair_count": len(train_pairs), "development_pair_count": len(heldout_pairs),
        "train_pairs": [[{"id": r["id"], "family": r["family"], "action": r["action"], "origin": r["origin"],
                         "seed": r["seed"], "prompt_sha256": digest_json(r["prompt"])}
                        for r in pair] for pair in train_pairs],
        "development_pairs": [[{"id": r["id"], "family": r["family"], "action": r["action"], "origin": r["origin"],
                               "seed": r["seed"], "prompt_sha256": digest_json(r["prompt"])}
                              for r in pair] for pair in heldout_pairs],
        "synthetic_counterfactual_not_native": True, "one_local_visited_mark_per_pair": True, "sealed_final_opened": False})
    # Select deterministic disjoint heldout probes before training and preserve originals.
    samples = objective.probes(heldout_rows, 18)
    if len(samples) < 9:
        raise ValueError("too few precommitted heldout action probes")
    write_json(args.output / "precommitted_probes.json",
               [{"id": r["id"], "family": r["family"], "target": r["action"], "target_code": r["completion"]} for r in samples])
    untuned = first_pass(base, tokenizer, adapter, samples, "untuned", args.output)
    untuned_scores = score_probe(base, tokenizer, adapter, samples, class_token_ids, "untuned", args.output)
    untuned_pairs = pair_probe(base, tokenizer, adapter, heldout_pairs, class_token_ids, "untuned", args.output)
    if time.monotonic() - started > args.deadline - 120:
        raise TimeoutError("insufficient remaining time for a real optimizer update")
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                 lora_dropout=0.05, bias="none", target_modules=selected["lora_targets"]))
    adapter.set_training_mode(model)
    model_contract = adapter.validate_trainable(model, selected)
    if model_contract["trainable_parameters"] != 4739072 or model_contract["lora_modules"] != 182:
        raise ValueError("reviewed LoRA whitelist diverged")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.00015, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    frozen_before = shared.fingerprint(model, False)
    adapter_before = shared.fingerprint(model, True)
    step, cursor, history = 0, 0, []
    last_checkpoint = None
    write_json(args.output / "model_contract.json", model_contract)
    while step < args.steps and not STOP and time.monotonic() - started < args.deadline - 90:
        pair_index = order[cursor % len(order)]
        pair = train_pairs[pair_index]
        encoded_pair = encoded_pairs[pair_index]
        started_step = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        # Two actual forwards/backwards per update; same family/tick/heading,
        # only the local visited frontier mark and consequent legal BRANCH differ. No target token at logit position.
        for member_idx in (0, 1):
            row = encoded_pair[member_idx]
            own, foil = pair[member_idx], pair[1-member_idx]
            ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            position = first_target_position(row["labels"])
            class_id = contract.ACTIONS.index(own["action"])
            foil_id = contract.ACTIONS.index(foil["action"])
            if class_id == foil_id:
                raise AssertionError("paired teacher labels must contrast")
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            scores = output.logits[0, position - 1, class_token_ids].float()
            ce = torch.nn.functional.cross_entropy(
                scores.unsqueeze(0), torch.tensor([class_id], dtype=torch.long),
                label_smoothing=0.15)
            margin = torch.nn.functional.softplus(0.5 - (scores[class_id] - scores[foil_id]))
            loss = 0.7 * ce + 0.3 * margin
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite local-frontier contrastive loss")
            (loss / 2).backward()
            losses.append(float(loss.item()))
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients) or not any(torch.count_nonzero(g).item() for g in gradients):
            raise ValueError("zero, absent, or nonfinite LoRA pair gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        optimizer.step()
        scheduler.step()
        step += 1
        cursor += 1
        event = {"step": step, "ids": [r["id"] for r in pair], "origins": [r["origin"] for r in pair],
                 "targets": [r["action"] for r in pair],
                 "loss": sum(losses)/2, "member_losses": losses,
                 "grad_norm": norm, "loss_tokens": 2,
                 "sequence_tokens": [len(r["input_ids"]) for r in encoded_pair],
                 "objective": "paired_local_frontier_ce_margin_v7",
                 "seconds": time.monotonic() - started_step}
        history.append(event)
        with (args.output / "progress.jsonl").open("a") as fp:
            fp.write(contract.canonical(event) + "\n")
        print(contract.canonical({"optimizer": event}), flush=True)
        if step % 9 == 0 or step == args.steps:
            last_checkpoint = shared.save_checkpoint(args.output, model, tokenizer,
                     shared.state_for(optimizer, scheduler, step, cursor, cfg, frozen_before, history), numerical_runtime)
    if step < 1:
        raise RuntimeError("no real optimizer update")
    if last_checkpoint is None or step % 9:
        last_checkpoint = shared.save_checkpoint(args.output, model, tokenizer,
                 shared.state_for(optimizer, scheduler, step, cursor, cfg, frozen_before, history), numerical_runtime)
    frozen_after, adapter_after = shared.fingerprint(model, False), shared.fingerprint(model, True)
    if frozen_after != frozen_before or adapter_after == adapter_before:
        raise ValueError("frozen base altered or LoRA unchanged")
    trained = None
    trained_scores = None
    trained_pairs = None
    if time.monotonic() - started < args.deadline - 60:
        trained = first_pass(model, tokenizer, adapter, samples, "trained", args.output)
        trained_scores = score_probe(model, tokenizer, adapter, samples, class_token_ids, "trained", args.output)
        trained_pairs = pair_probe(model, tokenizer, adapter, heldout_pairs, class_token_ids, "trained", args.output)
    write_json(args.output / "summary.json",
               {"actual_training": True, "status": "complete" if step == args.steps else "partial",
                "steps": step, "target_steps": args.steps, "checkpoint": last_checkpoint.name,
                "checkpoint_inventory_sha256": sha256_file(last_checkpoint / "inventory.json"),
                "frozen_base_equal": True, "adapter_changed": True,
                "frozen_base_sha256": frozen_after, "adapter_before_sha256": adapter_before,
                "adapter_after_sha256": adapter_after, "train": splits["train"], "eval": splits["eval"],
                "untuned_static_first_pass": untuned, "trained_static_first_pass": trained,
                "untuned_six_way_scores": untuned_scores, "trained_six_way_scores": trained_scores,
                "untuned_heldout_pairs": untuned_pairs, "trained_heldout_pairs": trained_pairs,
                "native_execution": False, "full_art_program_evaluation": False,
                "novel_only_probes": True, "raw_support_manifest": generated["support_manifest"],
                "sealed_final_opened": False, "wall_seconds": time.monotonic()-started,
                "cpu_seconds": time.process_time(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    return 0 if step == args.steps else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--w0", type=Path, required=True)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=18)
    p.add_argument("--deadline", type=int, default=1650)
    a = p.parse_args()
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    # sys.exit(0) raises SystemExit; do not label a completed campaign failed.
    # A nonzero main() result still exits nonzero, preserving partial-run CI failure.
    try:
        exit_code = main(a)
    except BaseException as exc:
        if a.output.exists():
            write_json(a.output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                      "status": "FAILED_OR_INTERRUPTED",
                      "not_training_success": True})
        raise
    raise SystemExit(exit_code)
