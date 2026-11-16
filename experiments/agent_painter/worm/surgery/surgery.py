#!/usr/bin/env python3
"""CyberSand WORM-001: reversible SPROCKET LoRA neural-surgery research trial.

THIS IS NOT TRAINING, NATIVE WORKBENCH, OR AN ACCEPTED MODEL CANDIDATE.
Loads the exact frozen SPROCKET W2 adapter and public root Instruct model once,
then reverts the adapter byte-for-byte between all interventions. Host W0 remains
source-pinned; the raw, unfixed generation controls the next host decision.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINING = ROOT / "training"
SPROCKET = ROOT / "lanes" / "sprocket"
sys.path[:0] = [str(HERE), str(TRAINING), str(SPROCKET)]
import core
import train as shared
from common import profile, verify_checkpoint, sha256_file, write_json
from lane_adapters import adapter_for
from numerics import configure_cpu
import worm_v1 as worm_contract
import worm_rollout_v2 as rollout

W2_SOURCE = "7683d06d9ffb26417ffc9694a07e8cf25e165fc6"
W2_RUN = 43697235301
W2_ARTIFACT = "sprocket-worm-v2-7683d06d9ffb26417ffc9694a07e8cf25e165fc6"
W2_INVENTORY_SHA256 = "2ef02292b9dbd290cd053b7fcdf54c0ebb4bcef91affacd133d7cd714bce3266"
MODEL_REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
W0_COMMIT = worm_contract.W0_COMMIT
SEEDS = (211, 223)
MAX_STEP = 12


def now_stats(started):
    return {"elapsed_wall_seconds": time.monotonic() - started,
            "elapsed_cpu_seconds": time.process_time(),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def probe_observations(worm):
    """Precommitted public W0 observations; selection does not use test outcomes."""
    contexts = []
    for family in worm_contract.FAMILIES:
        for seed in SEEDS:
            for tick in (0, 1):
                prefix = ["RIGHT"] * tick
                record = worm.run(seed, "replay", family, steps=tick+1, raw_actions=prefix)
                if len(record["events"]) != tick + 1:
                    continue
                obs = record["events"][tick]["observation"]
                if "RIGHT" in obs["legal"] and len(obs["legal"]) > 1:
                    contexts.append(obs)
    if len(contexts) < 6:
        raise ValueError("Not enough legitimate precommitted RIGHT contexts")
    return contexts


def action_first_ids(tokenizer, framing, observation):
    prompt = worm_contract.prompt(observation)
    prefix = framing.encode_prompt(tokenizer, prompt, worm_contract.SYSTEM)
    result = {}
    for action in worm_contract.ACTIONS:
        row = {"id": "surgery-first-token-" + action, "prompt": prompt, "completion": action}
        encoded = framing.encode_row(tokenizer, row, worm_contract.SYSTEM, 2048)
        if encoded["input_ids"][:len(prefix)] != prefix:
            raise ValueError("Model and action first-token prefixes diverged")
        token_id = encoded["input_ids"][len(prefix)]
        result[action] = token_id
    # This specific first-token margin is only valid if the initial action IDs
    # differ. Never silently conflate RIGHT with a label sharing its token.
    if len(set(result.values())) != len(result):
        raise ValueError("Action-first-token IDs collide; requires a separately versioned full-sequence attribution")
    return prefix, result


def add_activation_hooks(model, per_layer):
    """Layer-output gradient times activation at the final prompt position.

    These are attribution candidates, not proven unique neuron circuits.
    Both attention and MLP contributions are inspected per block.
    """
    handles = []
    for name, module in model.named_modules():
        if not (name.endswith(".self_attn") or name.endswith(".mlp")):
            continue
        def hook(_mod, _inputs, output, path=name):
            import torch
            t = output[0] if isinstance(output, tuple) else output
            if not isinstance(t, torch.Tensor) or t.ndim != 3 or not t.requires_grad:
                return
            activation = t[0, -1, :].detach().clone()
            def capture(gradient):
                values = (activation * gradient[0, -1, :].detach()).cpu()
                if path not in per_layer:
                    per_layer[path] = values.clone()
                else:
                    per_layer[path] += values
                return None
            t.register_hook(capture)
        handles.append(module.register_forward_hook(hook))
    if len(handles) < 26:
        for handle in handles:
            handle.remove()
        raise ValueError("Expected at least 26 attention/MLP source paths")
    return handles


def right_attribution(model, tokenizer, framing, parameters, observations, report):
    """Score RIGHT preference versus other legal action first tokens."""
    import torch
    for _, param in parameters:
        param.requires_grad_(True)
    model.eval()
    cumulative = {name: torch.zeros_like(p, device="cpu") for name, p in parameters}
    layers = {}
    hooks = add_activation_hooks(model, layers)
    trials = []
    try:
        for obs in observations[:6]:
            model.zero_grad(set_to_none=True)
            prefix, token_ids = action_first_ids(tokenizer, framing, obs)
            legal_others = [action for action in obs["legal"] if action != "RIGHT"]
            if not legal_others:
                raise ValueError("No legal contrast candidates")
            inputs = torch.tensor([prefix], dtype=torch.long)
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0, -1]
            if not torch.isfinite(logits).all().item():
                raise ValueError("Unmodified source checkpoint yielded nonfinite logits")
            right_logit = logits[token_ids["RIGHT"]]
            alternatives = logits[[token_ids[action] for action in legal_others]]
            margin = right_logit - (torch.logsumexp(alternatives, dim=0) - math.log(len(legal_others)))
            margin.backward()
            for name, p in parameters:
                if p.grad is not None:
                    cumulative[name] += (p.grad.detach() * p.detach()).clamp_min(0).cpu()
            trials.append({"observation_sha256": hashlib.sha256(
                          json.dumps(obs, sort_keys=True).encode()).hexdigest(),
                          "family": obs["family"], "tick": obs["tick"],
                          "position": [obs["x"], obs["y"]],
                          "right_vs_others_logit_margin": float(margin.detach()),
                          "legal_alternatives": legal_others, "first_action_token_ids": token_ids})
    finally:
        for handle in hooks:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for _, p in parameters:
            p.requires_grad_(False)
    flat = torch.cat([cumulative[name].reshape(-1) for name, _ in parameters])
    positive = int((flat > 0).sum().item())
    if positive <= 10:
        raise ValueError("Too few sign-positive RIGHT attribution weights to rank")
    top = torch.topk(flat, k=min(80, positive))
    spans = []
    cursor = 0
    for name, p in parameters:
        spans.append((cursor, cursor+p.numel(), name))
        cursor += p.numel()
    nodes = []
    for index, value in zip(top.indices.tolist(), top.values.tolist()):
        name = next(name for first, last, name in spans if first <= index < last)
        nodes.append({"tensor": name, "flat_parameter_index": index,
                      "positive_grad_weight_attribution": value})
    paths = []
    for path, values in sorted(layers.items()):
        strongest = torch.topk(values.abs(), k=min(6, values.numel()))
        paths.append({"path": path, "signed_sum": float(values.sum().item()),
                      "abs_sum": float(values.abs().sum().item()),
                      "most_active_channels": [
                          {"channel": int(i), "gradient_x_activation": float(values[i].item())}
                          for i in strongest.indices.tolist()]})
    paths.sort(key=lambda row: row["abs_sum"], reverse=True)
    report["attribution"] = {
        "type": "contrastive first-action-logit gradient times LoRA weight; experimental candidate ranking",
        "not_mechanistic_causal_proof": True,
        "positive_right_attributed_parameters": positive,
        "candidate_fraction_of_all_lora": positive / flat.numel(),
        "input_probes": trials, "top_candidate_weights": nodes,
        "most_influential_module_paths": paths[:52],
        "right_attribution_sha256": core.finite_tensor_sha256(flat)}
    return flat


def first_token_margin(model, tokenizer, framing, observation):
    import torch
    prefix, token_ids = action_first_ids(tokenizer, framing, observation)
    with torch.no_grad():
        inputs = torch.tensor([prefix], dtype=torch.long)
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=True).logits[0, -1]
    if not torch.isfinite(logits).all().item():
        return {"finite_logits": False, "right_margin": None,
                "first_token_winner": None}
    labels = list(worm_contract.ACTIONS)
    action_logits = logits[[token_ids[action] for action in labels]]
    normalized = torch.softmax(action_logits, dim=-1)
    winner = labels[int(torch.argmax(action_logits).item())]
    return {"finite_logits": True,
            "right_action_mass_first_token": float(normalized[labels.index("RIGHT")]),
            "first_token_winner": winner,
            "action_probabilities_conditional_on_six_first_tokens": {
                action: float(normalized[i]) for i, action in enumerate(labels)}}


def scenario(model, tokenizer, framing, worm, observations, out, arm_name, budget):
    out.mkdir(parents=True, exist_ok=True)
    # An extreme min-float corruption can make logits nonfinite. Record it as
    # a failed intervention rather than repairing the model or hiding it.
    preliminary = [first_token_margin(model, tokenizer, framing, obs) for obs in observations[:6]]
    if not all(item["finite_logits"] for item in preliminary):
        return {"status": "nonfinite-or-degenerate-before-rollout",
                "numerical_failure": True, "finite_preflight": preliminary,
                "rollouts": [], "raw_actions_generated": 0}
    summaries = []
    for family in worm_contract.FAMILIES:
        for seed in SEEDS:
            record, receipts = rollout.roll(worm, model, tokenizer, framing, seed, family, budget)
            record["mode"] = "sprocket-surgery-" + arm_name
            record["engine"] = "W0-source-faithful-model-first-pass-not-native"
            record["model_first_passes"] = receipts
            worm.write_result(record, out)
            summaries.append({"family": family, "seed": seed, "metrics": record["metrics"],
                              "raw_actions": [item["raw_output"] for item in receipts],
                              "applied_actions": [item["applied"] for item in receipts],
                              "legal_raw": sum(item["legal_raw"] for item in receipts),
                              "eos5": sum(item["eos5"] for item in receipts),
                              "decisions": len(receipts),
                              "png_sha256": sha256_file(
                                  out / f"{family}-sprocket-surgery-{arm_name}-{seed}.png")})
    return {"status": "executed", "numerical_failure": False,
            "finite_preflight": preliminary, "rollouts": summaries,
            "raw_actions_generated": sum(item["decisions"] for item in summaries)}


def main(args):
    start = time.monotonic()
    if args.mode not in ("targeted", "chaos") or not 1 <= args.steps <= MAX_STEP:
        raise ValueError("Bounded WORM surgery experiment only")
    if args.output.exists():
        raise ValueError("Immutable output lease already exists")
    args.output.mkdir(parents=True)
    run = {"schema": "cybersand.worm.sprocket-neural-surgery/1", "mode": args.mode,
           "research": "reversible, destructive, intentionally silly neural surgery",
           "training_performed": False, "original_model_unchanged": True,
           "native_execution": False, "sealed_FINAL_opened": False,
           "surgery_scope": "W2 rank-8 LoRA adapter weights only; not full 500M frozen base",
           "W2_training_source": W2_SOURCE, "W2_training_run": W2_RUN,
           "W2_training_artifact": W2_ARTIFACT, "W0_source": W0_COMMIT,
           "model_revision": MODEL_REVISION, "seeds": list(SEEDS),
           "steps": args.steps,
           "host_policy_on_illegal_raw": "W0 standard retrace-or-stop; never repair model output",
           "interventions": [], "source_sha": os.environ.get("GITHUB_SHA")}
    write_json(args.output / "lease.json", run)
    spec = profile("sprocket", require_supported=True)
    numerics, runtime = configure_cpu(1)
    from peft import PeftModel
    import torch
    model_base, tokenizer = shared.load_base(args.model, spec)
    framing = adapter_for(spec)
    checkpoint = verify_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    if (cfg.get("schema") != "cybersand.worm.sprocket-steering-v2"
        or cfg.get("source_sha") != W2_SOURCE or cfg.get("model") != spec
        or cfg.get("worm_common_git_commit") != W0_COMMIT
        or cfg.get("worm_common_sha256") != sha256_file(args.w0)
        or cfg.get("numerics") != numerics or cfg.get("final_opened") is not False
        or cfg.get("synthetic_not_native") is not True):
        raise ValueError("W2 original model/adapter/native source custody mismatch")
    if sha256_file(args.checkpoint / "inventory.json") != W2_INVENTORY_SHA256:
        raise ValueError("W2 immutable checkpoint inventory SHA mismatch")
    model = PeftModel.from_pretrained(model_base, args.checkpoint / "adapter",
                                      is_trainable=False, local_files_only=True)
    model.eval()
    parameters = core.lora_parameters(model)
    original = core.flatten(parameters)
    if original.numel() != 4_739_072:
        raise ValueError("Sprocket trained LoRA parameter count changed")
    source_hash = core.finite_tensor_sha256(original)
    model_frozen_before = shared.fingerprint(model, False)
    worm = worm_contract.load_w0(args.w0)
    observations = probe_observations(worm)
    run.update({"trainable_lora_scalar_count": original.numel(),
                "original_lora_bytes_sha256": source_hash,
                "frozen_base_sha256": model_frozen_before,
                "numerics": numerics, "environment_runtime": runtime})
    write_json(args.output / "lease.json", run)
    if args.mode == "targeted":
        scores = right_attribution(model, tokenizer, framing, parameters, observations, run)
        write_json(args.output / "attribution.json", run["attribution"])
    else:
        scores = None
    original_dir = args.output / "original-nonmodel-baselines"
    for family in worm_contract.FAMILIES:
        for seed in SEEDS:
            for policy in ("random", "wall", "fidget"):
                worm.write_result(worm.run(seed, policy, family, steps=args.steps), original_dir)
    arms = core.TARGETED if args.mode == "targeted" else core.CHAOS
    failures = []
    try:
        for arm in arms:
            started_arm = time.monotonic()
            outcome = {"arm": arm.name, "operator": arm.method,
                       "fraction": arm.fraction, "seed": arm.seed,
                       "source": "immutable W2 LoRA clone, no optimizer update"}
            if time.monotonic() - start > args.deadline - 170:
                outcome["status"] = "not-run-deadline-budget"
                failures.append(arm.name)
                run["interventions"].append(outcome)
                write_json(args.output / "results.json", run)
                continue
            try:
                altered, receipt = core.mutate(original, parameters, arm, scores)
                core.apply_flat(parameters, altered)
                outcome["mutation"] = receipt
                outcome["result"] = scenario(model, tokenizer, framing, worm, observations,
                                              args.output / arm.name, arm.name, args.steps)
            except Exception as exc:
                outcome["status"] = "exception-during-candidate-evaluation"
                outcome["error"] = {"type": type(exc).__name__, "message": str(exc),
                                    "traceback": traceback.format_exc()[-5500:]}
                failures.append(arm.name)
            finally:
                core.apply_flat(parameters, original)
                if core.finite_tensor_sha256(core.flatten(parameters)) != source_hash:
                    raise RuntimeError("Surgery did not restore original LoRA bytes!")
                outcome["adapter_restored_exactly"] = True
                outcome["elapsed_arm_seconds"] = time.monotonic() - started_arm
                run["interventions"].append(outcome)
                write_json(args.output / "results.json", run)
                print(json.dumps({"event": "surgery-arm-done", "arm": arm.name,
                                  "result": outcome.get("result", {}).get("status", outcome.get("status", "unknown")),
                                  "elapsed_seconds": outcome["elapsed_arm_seconds"]}), flush=True)
    finally:
        core.apply_flat(parameters, original)
        if shared.fingerprint(model, False) != model_frozen_before:
            raise RuntimeError("Frozen Sprocket base changed in surgical experiment")
    run["result"] = "complete" if not failures else "partial-with-labelled-failures"
    run["failed_or_unexecuted_arms"] = failures
    run["final_adapter_restored_exactly"] = core.finite_tensor_sha256(core.flatten(parameters)) == source_hash
    run["full_base_frozen_exactly"] = True
    run["timing"] = now_stats(start)
    write_json(args.output / "results.json", run)
    print(json.dumps({"event": "worm-surgery-complete", "mode": args.mode,
                      "arms": len(run["interventions"]), "failures": failures,
                      "wall_seconds": run["timing"]["elapsed_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--w0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("targeted", "chaos"), required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--deadline", type=int, default=2300)
    main(parser.parse_args())
