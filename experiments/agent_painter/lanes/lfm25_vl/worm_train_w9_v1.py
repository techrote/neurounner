#!/usr/bin/env python3
"""WORM W9: fresh full-LFM six-way contrastive steering on actual W0 TRAIN pairs.

Distinct from W8's full-vocabulary digit+EOS language-model objective: score
only the SIX qualified next-digit tokens on each complete official-processor
prompt, then optimize paired six-way conditional cross-entropy. Neither the
legal mask nor an external teacher repairs model output. Host W0 still owns
permission, legality, STOP and application. Strict raw EOS7 is evaluated
separately; W9 score-driven actions are never represented as raw completions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time

import worm_train_w7_v1 as w7
import worm_train_w8_v1 as w8
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for, encode_prompt
from train import fingerprint, load_base, save_checkpoint, state_for

VERSION = "lfm-worm-w9-six-way-paired-CE-fresh-v1"
PAIRS = 45
STEPS = 45
SEED = 612901
DEADLINE = 1700
STOP = False


def stop(_signal, _frame):
    global STOP
    STOP = True


def code_ids(adapter, processor, selected, prompt):
    """Only reviewed single-token digit alternatives + exact EOS7 are permitted."""
    tokens = {}
    for action, digit in w7.ACTION_CODES.items():
        row = adapter.encode_row(processor, {
            "id": "W9-action-token-probe", "prompt": prompt,
            "completion": digit}, w7.SYSTEM, 2048)
        labels = [label for label in row["labels"] if label != -100]
        if (len(labels) != 2 or labels[-1] != 7 or
                labels[-1] != processor.eos_token_id or
                processor.decode([labels[0]], skip_special_tokens=True) != digit):
            raise ValueError("W9 official processor candidate/EOS7 identity failed")
        tokens[action] = labels[0]
    if len(set(tokens.values())) != 6:
        raise ValueError("six candidate code tokens must be distinct")
    return tokens


def encode_item(adapter, processor, selected, row, ids):
    encoded = adapter.encode_row(processor, {
        **row, "completion": w7.ACTION_CODES[row["completion"]]},
        w7.SYSTEM, 2048)
    labels = encoded["labels"]
    target_start = next((i for i, value in enumerate(labels) if value != -100), None)
    if target_start is None or target_start < 1:
        raise ValueError("missing supervised suffix boundary")
    if (labels[target_start:] != [ids[row["completion"]], 7] or
            encoded["input_ids"][target_start:] != labels[target_start:]):
        raise ValueError("action/EOS suffix mutated or unexpected template tail")
    prefix = encoded["input_ids"][:target_start]
    if prefix != encode_prompt(processor, row["prompt"], w7.SYSTEM, selected):
        raise ValueError("training/inference prompt framing differs")
    if not prefix or len(prefix) + 2 > selected["max_sequence_tokens"]:
        raise ValueError("W9 context exceeds qualified profile")
    return {"id": row["id"], "family": row["family"],
            "target": w7.ACTIONS.index(row["completion"]),
            "prompt_ids": prefix, "input_tokens": len(prefix),
            "origin": row["origin"]}


def six_scores(model, ids, action_ids):
    import torch
    tensor = torch.tensor([ids], dtype=torch.long)
    logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                   use_cache=False).logits[0, -1].float()
    return logits.index_select(0, torch.tensor(action_ids, dtype=torch.long))


def probes(model, processor, selected, adapter, dev, codebook, label):
    """Held-out actual local observations; syntax, legality and logits separate."""
    import torch
    chosen = [next(r for r in dev if r["family"] == family and
                   r["completion"] == action)
              for family in w7.FAMILIES for action in w7.ACTIONS]
    records = []
    model.eval()
    with torch.inference_mode():
        for row in chosen:
            enc = encode_item(adapter, processor, selected, row, codebook)
            scores = six_scores(model, enc["prompt_ids"],
                                [codebook[a] for a in w7.ACTIONS])
            top = w7.ACTIONS[int(scores.argmax().item())]
            permitted = [a for a in w7.ACTIONS if a in row["legal"]]
            masked_top = max(permitted, key=lambda a: float(scores[w7.ACTIONS.index(a)]))
            changed = None
            if len(permitted) > 1 and row["completion"] in permitted:
                altered = dict(json.loads(row["prompt"]))
                altered["legal"] = [a for a in altered["legal"] if a != row["completion"]]
                modified = dict(row, id=row["id"] + ":synthetic-heldout-exclusion",
                                prompt=canonical(altered), legal=altered["legal"],
                                completion=altered["legal"][0])
                mod = encode_item(adapter, processor, selected, modified, codebook)
                shifted = six_scores(model, mod["prompt_ids"],
                                     [codebook[a] for a in w7.ACTIONS])
                altered_top = w7.ACTIONS[int(shifted.argmax().item())]
                changed = {"excluded": row["completion"], "raw_score_top": altered_top,
                           "unmasked_score_top_changed": altered_top != top,
                           "new_legal_top": max(altered["legal"],
                               key=lambda a: float(shifted[w7.ACTIONS.index(a)])),
                           "scores": [float(z) for z in shifted],
                           "synthetic_not_W0_observed": True}
            records.append({"id": row["id"], "family": row["family"],
                "origin": row["origin"], "world_group": row["seed_group"],
                "expected": row["completion"], "legal": row["legal"],
                "raw_score_top": top, "legal_projected_top": masked_top,
                "scores": [float(z) for z in scores],
                "teacher_raw_match": top == row["completion"],
                "teacher_legal_projected_match": masked_top == row["completion"],
                "synthetic_exclusion": changed})
    # Raw digit+EOS inference is a distinct model process, not relabelled as
    # the six-way scorer. No hidden host repair/retry on raw model completions.
    raw = w7.evaluation(model, processor, selected, dev)
    if [x["id"] for x in records] != [x["id"] for x in raw]:
        raise ValueError("raw/score DEVELOPMENT ordering differs")
    write = {"stage": label, "source": "novel-WORM-DEVELOPMENT-not-FINAL",
             "samples": records, "raw_first_passes": raw,
             "denominator": len(records),
             "observed_denominator": sum(x["origin"] == "W0-observed" for x in records),
             "synthetic_dev_denominator": sum(x["origin"] != "W0-observed" for x in records),
             "score_argmax": dict(Counter(x["raw_score_top"] for x in records)),
             "score_legal_top": dict(Counter(x["legal_projected_top"] for x in records)),
             "score_teacher": sum(x["teacher_raw_match"] for x in records),
             "legal_score_teacher": sum(x["teacher_legal_projected_match"] for x in records),
             "raw_true_EOS7_syntax": sum(x["raw_syntactic_eos"] for x in raw),
             "raw_host_legal": sum(x["valid_first_pass"] for x in raw),
             "raw_teacher": sum(x["decoded_action"] == x["expected"] for x in raw),
             "paired_exclusions": sum(x["synthetic_exclusion"] is not None for x in records),
             "paired_unmasked_changes": sum(bool(x["synthetic_exclusion"] and
                                            x["synthetic_exclusion"]["unmasked_score_top_changed"])
                                             for x in records)}
    return write


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "worm_root", "support_root", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--deadline", type=int, default=DEADLINE)
    a = p.parse_args()
    if a.steps != STEPS or not 1300 <= a.deadline <= 2050:
        raise ValueError("W9 one finite 45-step paired-CE pilot only")
    if a.output.exists():
        raise ValueError("immutable W9 job lease already exists")
    a.output.mkdir(parents=True)
    start = time.monotonic()
    source = w7.git("rev-parse", "HEAD", cwd=w7.ROOT.parents[1])
    write_json(a.output / "lease.json", {
        "experiment": VERSION, "source_sha": source, "job": os.getenv("GITHUB_JOB"),
        "run_id": os.getenv("GITHUB_RUN_ID"), "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "fresh_from_pinned_full_model": True, "not_W8_resume": True,
        "W0_source": w7.WORM_SHA, "support_source": w7.SUPPORT_SHA,
        "maximum_optimizer_steps": STEPS, "sealed_FINAL_opened": False,
        "native_execution": False, "provider_changed": False})
    worm = w7.worm_source(a.worm_root)
    support = w7.support_source(a.support_root, worm)
    splits, provenance = w7.dataset(support)
    pairs = w8.paired_training_rows(splits)
    if len(pairs) != PAIRS or len({r["prompt"] for pair in pairs for r in pair}) != 90:
        raise ValueError("W9 pair data or prompt uniqueness mismatch")
    samples = [r for pair in pairs for r in pair]
    (a.output / "optimizer-pairs.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in samples), encoding="utf-8")
    pair_sha = sha256_file(a.output / "optimizer-pairs.jsonl")
    manifest = {"schema": "cybersand.worm.lfm-w9-paired-six-way-CE/1",
        "W0_source": w7.WORM_SHA, "support_source": w7.SUPPORT_SHA,
        "TRAIN_W0_original": 45, "TRAIN_synthetic_exclusions": 45,
        "DEV_optimizer_rows": 0, "FINAL_opened": False,
        "pair_sha256": pair_sha, "TRAIN_DEV_prompt_overlap": 0,
        "corpus_provenance": provenance}
    write_json(a.output / "corpus-manifest.json", manifest)
    selected = profile("lfm25_vl", require_supported=True)
    numerical, runtime = configure_cpu(1)
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    random.seed(SEED)
    torch.manual_seed(SEED)
    base, processor = load_base(a.model, selected)
    adapter = adapter_for(selected)
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=8,
        lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=selected["lora_targets"]))
    adapter.set_training_mode(model)
    contract = adapter.validate_trainable(model, selected)
    ids = code_ids(adapter, processor, selected, pairs[0][0]["prompt"])
    encoded = [encode_item(adapter, processor, selected, row, ids)
               for row in samples]
    recipe = {"objective": "paired-six-code-first-next-token-conditional-CE",
        "source_parent": "complete official full multimodal LFM2.5-VL parent",
        "W8_language_model_loss_used": False, "rank": 8, "alpha": 16,
        "dropout": 0.05, "lr": 0.0005, "weight_decay": 0.01,
        "betas": [0.9, 0.95], "threads": 1, "precision": "fp32",
        "seed": SEED, "steps": STEPS, "pairs": PAIRS,
        "two_backward_passes_per_AdamW_update": True,
        "code_token_ids": ids, "teacher_without_masked_loss": True,
        "training_legal_masks_from_source_not_network_output": True}
    config = {"source_sha": source, "objective": VERSION,
        "profile": "lfm25_vl", "model": selected,
        "model_files": json.loads((a.model / "download_receipt.json").read_text())["files"],
        "numerics": numerical, "system": w7.SYSTEM, "recipe": recipe,
        "corpus_manifest": manifest, "final_targets_opened": False,
        "native_execution": False,
        "packages": {x: importlib.metadata.version(x)
                     for x in ("torch", "transformers", "peft", "torchvision", "Pillow")}}
    write_json(a.output / "preflight.json", {"config": config,
        "adapter_contract": contract,
        "trainable_parameters": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        "max_prefix_tokens": max(x["input_tokens"] for x in encoded),
        "runtime": runtime})
    before = probes(model, processor, selected, adapter,
                    splits["WORM_DEV"], ids, "fresh-untuned")
    write_json(a.output / "untuned-probes.json", before)
    adapter.set_training_mode(model)
    frozen_before = fingerprint(model, False)
    adapter_before = fingerprint(model, True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.0005,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ordering = list(range(PAIRS))
    random.Random(SEED + 1).shuffle(ordering)
    write_json(a.output / "sampler-receipt.json", {"pair_order": ordering,
        "pair_ids": [[samples[2*i]["id"],samples[2*i+1]["id"]] for i in ordering],
        "seed": SEED + 1, "steps": STEPS})
    hist = []
    checkpoint = None
    candidate_token_tensor = torch.tensor([ids[action] for action in w7.ACTIONS],
                                          dtype=torch.long)
    for step, pair_index in enumerate(ordering, 1):
        if STOP or time.monotonic()-start > a.deadline-200:
            break
        adapter.set_training_mode(model)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        tick = time.monotonic()
        for idx in (2*pair_index, 2*pair_index+1):
            row = encoded[idx]
            logits = six_scores(model, row["prompt_ids"],
                                candidate_token_tensor)
            label = torch.tensor([row["target"]],dtype=torch.long)
            loss = F.cross_entropy(logits.unsqueeze(0),label)
            if not torch.isfinite(loss).item():
                raise ValueError("W9 nonfinite six-way CE")
            (loss / 2).backward()
            losses.append(float(loss.item()))
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients) or not any(
                torch.count_nonzero(g).item() for g in gradients):
            raise ValueError("W9 no real finite adapter gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(trainable,1.0).item())
        optimizer.step()
        scheduler.step()
        event = {"global_step": step,
            "pair_id": pairs[pair_index][0]["pair_id"],
            "original_id": encoded[2*pair_index]["id"],
            "synthetic_id": encoded[2*pair_index+1]["id"],
            "two_genuine_backward_passes": 2, "loss_original": losses[0],
            "loss_synthetic": losses[1], "gradient_norm": norm,
            "elapsed_seconds": time.monotonic()-tick}
        hist.append(event)
        with (a.output/"progress.jsonl").open("a") as f:
            f.write(canonical(event)+"\n")
        print(canonical({"event": "W9-real-paired-AdamW",**event}),flush=True)
        if step%15==0 or step==STEPS:
            checkpoint = save_checkpoint(a.output,model,processor,
                state_for(optimizer,scheduler,step,step,config,frozen_before,hist),
                runtime)
    if not hist:
        raise ValueError("no W9 optimizer update")
    if checkpoint is None or int(checkpoint.name.split("-")[-1])!=len(hist):
        checkpoint = save_checkpoint(a.output,model,processor,
            state_for(optimizer,scheduler,len(hist),len(hist),config,frozen_before,hist),
            runtime)
    frozen_after, adapter_after = fingerprint(model,False),fingerprint(model,True)
    if frozen_before!=frozen_after or adapter_before==adapter_after:
        raise ValueError("frozen base mutated or LoRA failed to update")
    after = None
    if time.monotonic()-start < a.deadline-190:
        after = probes(model, processor, selected, adapter,
                       splits["WORM_DEV"], ids, "trained")
        write_json(a.output/"trained-probes.json",after)
    summary = {"status": "complete" if len(hist)==STEPS else "partial",
        "model_class": "Lfm2VlForConditionalGeneration",
        "source_sha": source, "revision": selected["revision"],
        "actual_optimizer_updates": len(hist), "requested_updates": STEPS,
        "real_backward_passes": len(hist)*2,
        "checkpoint": checkpoint.name, "pair_sha256": pair_sha,
        "frozen_base_unchanged": True, "adapter_changed": True,
        "untuned_score_teacher": before["score_teacher"],
        "trained_score_teacher": after["score_teacher"] if after else None,
        "untuned_paired_unmasked_changes": before["paired_unmasked_changes"],
        "trained_paired_unmasked_changes": after["paired_unmasked_changes"] if after else None,
        "strict_raw_first_passes": after["raw_first_passes"] if after else None,
        "gallery_model_rollouts": 0, "gallery_deferred": True,
        "wall_seconds": time.monotonic()-start,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sealed_FINAL_opened": False, "native_execution": False}
    write_json(a.output/"summary.json",summary)
    if len(hist)!=STEPS:
        raise ValueError("W9 partial deadline/Stop; checkpoint retained")
    print(canonical({"event": "W9-complete","updates":len(hist),
                     "teacher":summary["trained_score_teacher"]}),flush=True)


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT,stop)
        signal.signal(signal.SIGTERM,stop)
        main()
    except Exception as error:
        if "--output" in sys.argv:
            root=Path(sys.argv[sys.argv.index("--output")+1])
            root.mkdir(parents=True,exist_ok=True)
            write_json(root/"failure.json",{"experiment": VERSION,
                "type":type(error).__name__, "message":str(error)})
        raise
