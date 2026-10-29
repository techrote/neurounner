#!/usr/bin/env python3
"""DETECTIVE-001C2 40-step pairwise LoRA continuation from exact C1 parents."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

import peft
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-001/training-c2.json"


class TrainError(ValueError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha_file(item)))
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def load_rows(dataset_root: Path, member: str, cfg: dict) -> tuple[dict, list[dict]]:
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    if (manifest.get("schema") != "detective-witness-pair-dataset-v1"
            or manifest.get("experiment") != "DETECTIVE-001C2"
            or manifest.get("final_suite_materialized") is not False
            or manifest.get("c1_final_used_as_training") is not False):
        raise TrainError("pair dataset custody mismatch")
    path = dataset_root / manifest["member_files"][member]["path"]
    if sha_file(path) != manifest["member_files"][member]["sha256"]:
        raise TrainError("pair shard hash mismatch")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows) != cfg["dataset"]["pairs_per_member"]:
        raise TrainError("pair count mismatch")
    if len({x["pair_id"] for x in rows}) != len(rows):
        raise TrainError("duplicate pair IDs")
    if any(x["member_id"] != member for x in rows):
        raise TrainError("cross-member pair")
    return manifest, rows


def encode_response(tok, system: str, user: str, response: str,
                    max_length: int) -> dict:
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    full = prompt + [{"role": "assistant", "content": response}]
    prompt_ids = tok.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False)
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    if common >= len(full_ids):
        raise TrainError("no assistant supervised tokens")
    if len(full_ids) > max_length:
        raise TrainError(f"sequence length {len(full_ids)} > {max_length}")
    labels = [-100] * common + full_ids[common:]
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sequence_tokens": len(full_ids),
        "supervised_tokens": len(full_ids) - common,
    }


def pair_batch(chosen: dict, rejected: dict, pad_id: int):
    ids = pad_sequence(
        [chosen["input_ids"], rejected["input_ids"]],
        batch_first=True, padding_value=pad_id)
    labels = pad_sequence(
        [chosen["labels"], rejected["labels"]],
        batch_first=True, padding_value=-100)
    attention = (ids != pad_id).long()
    return ids, labels, attention


def assistant_mean_logps(model, input_ids, labels, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = labels[:, 1:] != -100
    logp = torch.log_softmax(logits, dim=-1)
    token = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    counts = mask.sum(dim=1).clamp_min(1)
    return (token * mask).sum(dim=1) / counts


@torch.inference_mode()
def preference_eval(model, pairs, pad_id: int) -> dict:
    model.eval()
    margins = []
    chosen_values, rejected_values = [], []
    for chosen, rejected in pairs:
        ids, labels, attention = pair_batch(chosen, rejected, pad_id)
        values = assistant_mean_logps(model, ids, labels, attention)
        c, r = float(values[0]), float(values[1])
        chosen_values.append(c); rejected_values.append(r)
        margins.append(c - r)
    return {
        "mean_chosen_logp": sum(chosen_values) / len(chosen_values),
        "mean_rejected_logp": sum(rejected_values) / len(rejected_values),
        "mean_margin_nats_per_token": sum(margins) / len(margins),
        "preference_accuracy": sum(x > 0 for x in margins) / len(margins),
        "margin_min": min(margins), "margin_max": max(margins),
        "margins": margins,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--parent-adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    if args.member not in cfg["members"]:
        raise SystemExit("unknown C2 member " + args.member)
    meta = cfg["members"][args.member]
    train = cfg["training"]
    if train["optimizer_steps_per_member"] != 40 or train["epochs"] != 1:
        raise TrainError("training schedule drift")
    if train["paid_api_calls"] != 0 or train["gpu_required"]:
        raise TrainError("unapproved compute/provider budget")

    manifest, rows = load_rows(args.dataset_root, args.member, cfg)
    parent_hash = directory_hash(args.parent_adapter)
    if parent_hash != meta["parent_directory_sha256"]:
        raise TrainError("C1 parent adapter directory hash mismatch")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(train["max_threads"], os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(meta["seed"]); torch.manual_seed(meta["seed"])

    weight = Path(hf_hub_download(
        cfg["base_model"]["id"], "model.safetensors",
        revision=cfg["base_model"]["revision"]))
    if sha_file(weight) != cfg["base_model"]["weights_sha256"]:
        raise TrainError("base model SHA mismatch")
    started_load = time.monotonic()
    tok = AutoTokenizer.from_pretrained(
        cfg["base_model"]["id"], revision=cfg["base_model"]["revision"],
        trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["id"], revision=cfg["base_model"]["revision"],
        torch_dtype=torch.float32, trust_remote_code=False)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, args.parent_adapter, is_trainable=True)
    model.config.use_cache = False
    load_seconds = time.monotonic() - started_load

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise TrainError("no trainable LoRA parameters")
    offenders = [
        name for name in trainable_names
        if "lora_" not in name.casefold() and "modules_to_save" not in name.casefold()
    ]
    if offenders:
        raise TrainError("non-LoRA parameter trainable: " + ",".join(offenders[:8]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    pairs = []
    for row in rows:
        chosen = encode_response(
            tok, cfg["system"], row["input"], row["chosen"],
            train["max_length"])
        rejected = encode_response(
            tok, cfg["system"], row["input"], row["rejected"],
            train["max_length"])
        pairs.append((chosen, rejected))

    before = preference_eval(model, pairs, tok.pad_token_id)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train["learning_rate"], betas=(0.9, 0.95), weight_decay=0.01)
    order = list(range(len(pairs)))
    random.Random(meta["seed"]).shuffle(order)
    if len(order) != train["optimizer_steps_per_member"]:
        raise TrainError("optimizer schedule/cardinality mismatch")

    beta = train["beta"]
    target_margin = train["target_margin_nats"]
    anchor_weight = train["chosen_nll_weight"]
    clip = train["gradient_clip"]
    losses, pref_losses, anchor_losses, margins = [], [], [], []
    seq_tokens = sup_tokens = 0
    model.train()
    started = time.monotonic()
    for pos in order:
        chosen, rejected = pairs[pos]
        ids, labels, attention = pair_batch(
            chosen, rejected, tok.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        values = assistant_mean_logps(model, ids, labels, attention)
        chosen_lp, rejected_lp = values[0], values[1]
        margin = chosen_lp - rejected_lp
        pref_loss = -torch.nn.functional.logsigmoid(
            beta * (margin - target_margin))
        anchor_loss = -chosen_lp
        loss = pref_loss + anchor_weight * anchor_loss
        if not torch.isfinite(loss):
            raise TrainError("nonfinite pairwise loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        pref_losses.append(float(pref_loss.detach()))
        anchor_losses.append(float(anchor_loss.detach()))
        margins.append(float(margin.detach()))
        seq_tokens += chosen["sequence_tokens"] + rejected["sequence_tokens"]
        sup_tokens += chosen["supervised_tokens"] + rejected["supervised_tokens"]
    train_seconds = time.monotonic() - started
    if len(losses) != 40:
        raise TrainError("incomplete pairwise training")

    after = preference_eval(model, pairs, tok.pad_token_id)
    args.out.mkdir(parents=True, exist_ok=True)
    adapter = args.out / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    trained_hash = directory_hash(adapter)

    result = {
        "schema": "detective-witness-preference-result-v1",
        "experiment": "DETECTIVE-001C2", "issue": 115,
        "member_id": args.member, "policy_id": meta["policy_id"],
        "base_model": cfg["base_model"]["id"],
        "base_revision": cfg["base_model"]["revision"],
        "base_weight_sha256": cfg["base_model"]["weights_sha256"],
        "parent_release_tag": cfg["parent_release"]["tag"],
        "parent_archive_sha256": meta["parent_archive_sha256"],
        "parent_directory_sha256": parent_hash,
        "pair_dataset_manifest_sha256":
            sha_file(args.dataset_root / "manifest.json"),
        "training_shard_sha256":
            manifest["member_files"][args.member]["sha256"],
        "pair_ids": [x["pair_id"] for x in rows],
        "preference_before": before,
        "training": {
            "optimizer_steps": len(losses), "epochs": train["epochs"],
            "learning_rate": train["learning_rate"],
            "beta": beta, "target_margin_nats": target_margin,
            "chosen_nll_weight": anchor_weight,
            "max_length": train["max_length"], "gradient_clip": clip,
            "seconds": train_seconds,
            "sequence_tokens": seq_tokens,
            "supervised_tokens": sup_tokens,
            "sequence_tokens_per_second":
                seq_tokens / train_seconds if train_seconds else None,
            "supervised_tokens_per_second":
                sup_tokens / train_seconds if train_seconds else None,
            "first_loss": losses[0], "last_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses),
            "mean_preference_loss": sum(pref_losses) / len(pref_losses),
            "mean_anchor_nll": sum(anchor_losses) / len(anchor_losses),
            "mean_training_margin": sum(margins) / len(margins),
            "losses": losses,
        },
        "preference_after": after,
        "model": {
            "total_parameters": total, "trainable_parameters": trainable,
            "trainable_fraction": trainable / total,
            "only_lora_parameters_trainable": True,
        },
        "adapter": {
            "path": "adapter", "directory_sha256": trained_hash,
            "bytes": sum(
                p.stat().st_size for p in adapter.rglob("*") if p.is_file()),
        },
        "runtime": {
            "cpu_model": cpu_model(), "cpu_count": os.cpu_count(),
            "platform": platform.platform(), "threads": torch.get_num_threads(),
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "load_seconds": load_seconds,
        },
    }
    write_json(args.out / "result.json", result)
    print(json.dumps({
        "member": args.member, "steps": len(losses),
        "loss": [losses[0], losses[-1]],
        "preference_margin": [
            before["mean_margin_nats_per_token"],
            after["mean_margin_nats_per_token"],
        ],
        "preference_accuracy": [
            before["preference_accuracy"], after["preference_accuracy"]],
        "parent_sha256": parent_hash, "trained_sha256": trained_hash,
        "seconds": train_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
