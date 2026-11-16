#!/usr/bin/env python3
"""WORM W7 zero-optimizer, source-fenced held-out sensitivity experiment.

This experiment intentionally does NOT overwrite the failed W7 producer or
pretend that a complete 120-step optimizer implies successful model painting.
It scores genuine unmodified model completions against LFM support's frozen
96-stimulus challenge and separately measures six-code next-token scores.
The latter are diagnostics, NOT model-emitted legal actions or native painting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import worm_train_w7_v1 as w7
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for, encode_prompt
from train import fingerprint, load_base, verify_checkpoint

PRODUCER_SHA = "7b3a39cbc27e12f41b5cc1f232a9a09ec4a8be93"
PRODUCER_RUN = 43721719510
PRODUCER_ARTIFACT = 14166720580
PRODUCER_ZIP_SHA256 = "259d40640392e4ccd922b8d51e754795ea9d95503d26c6429c8fe4ac05cce1d9"
CHALLENGE_SHA256 = "f07fae41a0416d13d587a2be02bec54e23178fc46fc6bb66b3ed0aa710d20faf"
CHALLENGE_SOURCE = "fa5b3de4e779b6e0ac564abac809fe580627a2f6"
MODEL_REV = "fc6221ca597f3315e4f82fc2df606783267b34ba"


def validate_sources(original, model_root, challenge, receipt, selected):
    if receipt["global_step"] != 120 or receipt["sample_cursor"] != 120:
        raise ValueError("W7 original checkpoint not optimizer step120")
    if len(receipt["history"]) != 120:
        raise ValueError("W7 original optimizer history incomplete")
    if [e["global_step"] for e in receipt["history"]] != list(range(1, 121)):
        raise ValueError("W7 optimizer history nonconsecutive")
    if any(e["gradient_norm"] <= 0 or e["loss"] < 0 for e in receipt["history"]):
        raise ValueError("W7 optimizer history invalid")
    config = receipt["config"]
    if (config["source_sha"] != PRODUCER_SHA
            or config["objective"] != "lfm-worm-action-selection-v7-balanced-codes-fresh"
            or config["profile"] != "lfm25_vl"
            or config["model"] != selected
            or config["model"]["revision"] != MODEL_REV):
        raise ValueError("wrong source, model, or W7 optimizer recipe")
    manifest = config["corpus_manifest"]
    if (manifest["W0_source"] != w7.WORM_SHA
            or manifest["support_source"] != w7.SUPPORT_SHA
            or manifest["TRAIN_count"] != 1099
            or manifest["DEV_count"] != 683
            or manifest["sealed_FINAL_opened"] is not False):
        raise ValueError("wrong source-fenced W7 corpus")
    if (sha256_file(original / "worm_train.jsonl") != manifest["WORM_TRAIN_sha256"]
            or sha256_file(original / "worm_dev.jsonl") != manifest["WORM_DEV_sha256"]):
        raise ValueError("W7 saved corpus bytes differ")
    if config["model_files"] != json.loads((model_root / "download_receipt.json").read_text())["files"]:
        raise ValueError("official multimodal model file inventory differs")
    if config["recipe"]["action_codes"] != w7.ACTION_CODES or config["final_targets_opened"]:
        raise ValueError("W7 action/FINAL custody mismatch")
    if sha256_file(challenge) != CHALLENGE_SHA256:
        raise ValueError("frozen four-world challenge bytes differ")


def load_challenge_module(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing source-pinned support evaluator")
    spec = importlib.util.spec_from_file_location("w7_frozen_support_challenge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCHEMA != "worm-lfm-w7-heldout-observation-sensitivity-v1":
        raise ValueError("support challenge schema changed")
    return module


def score_next_token(model, processor, selected, prompt):
    """Six unmasked candidate-token scores, NEVER a silently repaired generation."""
    import torch
    from lane_adapters import adapter_for
    adapter = adapter_for(selected)
    candidate_ids = {}
    for action, digit in w7.ACTION_CODES.items():
        encoded = adapter.encode_row(processor, {"id": "probe", "prompt": prompt,
                                      "completion": digit}, w7.SYSTEM, 2048)
        target = [i for i, label in zip(encoded["input_ids"], encoded["labels"])
                  if label != -100]
        if len(target) != 2 or target[-1] != processor.eos_token_id:
            raise ValueError("digit is not a single token followed by EOS7")
        if processor.decode(target[:-1], skip_special_tokens=True) != digit:
            raise ValueError("digit next-token identity changed")
        candidate_ids[action] = target[0]
    if len(set(candidate_ids.values())) != 6:
        raise ValueError("digit candidate token identities overlap")
    prefix = encode_prompt(processor, prompt, w7.SYSTEM, selected)
    inputs = torch.tensor([prefix], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                       use_cache=False).logits[0, -1].float()
    selected_logits = torch.stack([logits[candidate_ids[a]] for a in w7.ACTIONS])
    conditional = torch.softmax(selected_logits, dim=0)
    probabilities = {a: float(p.item()) for a, p in zip(w7.ACTIONS, conditional)}
    return {"candidate_token_ids": candidate_ids,
            "candidate_logits": {a: float(v.item()) for a, v in zip(w7.ACTIONS, selected_logits)},
            "six_code_conditional_probabilities": probabilities,
            "six_code_top_action": max(probabilities, key=probabilities.get),
            "six_code_entropy_nats": -sum(p * __import__("math").log(p)
                                         for p in probabilities.values() if p > 0)}


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "worm_root", "original", "challenge", "support_challenge", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("immutable evaluation output already exists")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    write_json(args.output / "lease.json",
               {"producer_sha": PRODUCER_SHA, "producer_run": PRODUCER_RUN,
                "producer_artifact": PRODUCER_ARTIFACT,
                "producer_zip_sha256": PRODUCER_ZIP_SHA256,
                "challenge_sha256": CHALLENGE_SHA256,
                "challenge_source": CHALLENGE_SOURCE, "new_optimizer_updates": 0,
                "sealed_FINAL_opened": False, "native_execution": False})
    configure_cpu(1)
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    selected = profile("lfm25_vl", require_supported=True)
    checkpoint = args.original / "checkpoint-000120"
    receipt = verify_checkpoint(checkpoint)
    validate_sources(args.original, args.model, args.challenge, receipt, selected)
    worm = w7.worm_source(args.worm_root)
    # Only the W0 module identity is used for the support challenge scorer.
    support = load_challenge_module(args.support_challenge)
    if support.worm is not worm:
        raise ValueError("support challenge imported a different W0 host")
    rows = support.read_jsonl(args.challenge)
    if (len(rows) != 96 or len({r["id"] for r in rows}) != 96
            or {r["kind"] for r in rows} != set(support.KINDS)
            or {r["source_seed"] for r in rows} != set(support.SEEDS)):
        raise ValueError("unsupported held-out stimulus manifest")

    base, processor = load_base(args.model, selected)
    model = PeftModel.from_pretrained(
        base, checkpoint / "adapter", is_trainable=True, local_files_only=True)
    adapter = adapter_for(selected)
    adapter.set_training_mode(model)
    contract = adapter.validate_trainable(model, selected)
    if fingerprint(model, False) != receipt["frozen_base_sha256"]:
        raise ValueError("base fingerprint does not match original W7")
    stored = load_file(str(checkpoint / "adapter" / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model)
    if stored.keys() != loaded.keys() or any(
            not torch.equal(stored[k], loaded[k].detach().cpu()) for k in stored):
        raise ValueError("W7 LoRA weights do not match checkpoint")
    model.eval()

    answers = []
    scores = []
    for i, row in enumerate(rows):
        if sha256_file_bytes(row["prompt"].encode()) != row["prompt_sha256"]:
            raise ValueError("support prompt identity mismatch")
        first = w7.generate(model, processor, selected,
                            row["prompt"], row["observation"]["legal"], max_tokens=10)
        item = {"id": row["id"], "prompt_sha256": row["prompt_sha256"],
                "kind": row["kind"], "source_seed": row["source_seed"],
                "raw_text": first["raw_text"], "token_ids": first["token_ids"],
                "eos_terminated": first["eos_terminated"],
                "raw_action": first["raw_action"],
                "raw_legal": first["raw_legal"],
                "valid_first_pass": first["valid_first_pass"],
                "seconds": first["seconds"]}
        score = {"id": row["id"], "prompt_sha256": row["prompt_sha256"],
                 "kind": row["kind"], **score_next_token(model, processor, selected,
                                                        row["prompt"])}
        answers.append(item)
        scores.append(score)
        with (args.output / "raw-first-passes.jsonl").open("a") as f:
            f.write(canonical(item) + "\n")
        with (args.output / "six-code-logits.jsonl").open("a") as f:
            f.write(canonical(score) + "\n")
        print(canonical({"event": "heldout-W7-model", "case": i + 1,
                         "of": len(rows), "raw": item["raw_text"],
                         "raw_legal": item["raw_legal"],
                         "top_six": score["six_code_top_action"]}), flush=True)

    scored = support.score(rows, answers)
    write_json(args.output / "challenge-score.json", scored)
    grouped = {}
    index = {x["id"]: x for x in scores}
    for kind in support.KINDS[1:]:
        differences = []
        for row in rows:
            if row["kind"] != "observed":
                continue
            a = index[row["id"]]["six_code_conditional_probabilities"]
            b = index[row["pair_id"] + ":" + kind]["six_code_conditional_probabilities"]
            differences.append(sum(abs(a[k] - b[k]) for k in w7.ACTIONS) / 2)
        grouped[kind] = {"paired_groups": len(differences),
                         "mean_six_code_total_variation": sum(differences) / len(differences),
                         "max_six_code_total_variation": max(differences)}
    write_json(args.output / "summary.json", {
        "status": "complete", "new_optimizer_updates": 0,
        "original_producer_optimizer_steps": 120,
        "original_producer_workflow_status": "failure-after-step120",
        "source_model": "complete pinned Lfm2VlForConditionalGeneration",
        "checkpoint": "checkpoint-000120", "frozen_base_verified": True,
        "loaded_adapter_bytes_verified": True, "contract": contract,
        "model_revision": selected["revision"],
        "challenge_sha256": CHALLENGE_SHA256, "requests": len(rows),
        "independent_obstacle_seeds": len(support.SEEDS),
        "raw_valid_eos": sum(x["eos_terminated"] and x["raw_text"] in w7.CODE_ACTIONS
                             for x in answers),  # grammar/EOS independent of host legality
        "raw_legal": sum(x["raw_legal"] for x in answers),
        "raw_action_codes": {digit: sum(x["raw_text"] == digit for x in answers)
                             for digit in "012345"},
        "paired_raw_changes": scored["by_perturbation"],
        "conditional_six_code_total_variation": grouped,
        "wall_seconds": time.monotonic() - start,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "interpretation": "unmasked first-pass generation and separate six-code next-token diagnostics only; no native or stochastic model painting",
        "sealed_FINAL_opened": False, "native_Godot_execution": False})
    print(canonical({"event": "W7-heldout-eval-complete",
                     "raw": len(answers), "optimizer_steps": 0}), flush=True)


def sha256_file_bytes(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
