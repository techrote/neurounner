#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import shutil
import sys
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from village_001_manifest import build_manifest
from village_001d_validate import validate_file

ROOT = Path(__file__).resolve().parents[1]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(item)))
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def member_contract(campaign: dict, member_id: str) -> tuple[dict, dict]:
    population = build_manifest(campaign)
    for row in population["members"]:
        if row["member_id"] == member_id:
            return population, row
    raise ValueError(f"unknown member {member_id}")


def encode_response(tok, system: str, user: str, response: str, max_length: int) -> dict:
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    full = prompt + [{"role": "assistant", "content": response}]
    prompt_ids = tok.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tok.apply_chat_template(
        full,
        tokenize=True,
        add_generation_prompt=False,
    )
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    if len(full_ids) > max_length:
        raise RuntimeError(
            f"sequence length {len(full_ids)} exceeds max_length {max_length}"
        )
    if common >= len(full_ids):
        raise RuntimeError("response has no supervised assistant tokens")
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
        batch_first=True,
        padding_value=pad_id,
    )
    labels = pad_sequence(
        [chosen["labels"], rejected["labels"]],
        batch_first=True,
        padding_value=-100,
    )
    attention = (ids != pad_id).long()
    return ids, labels, attention


def assistant_mean_logps(model, input_ids, labels, attention_mask):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = labels[:, 1:] != -100
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    counts = mask.sum(dim=1).clamp_min(1)
    return token_logp.sum(dim=1) / counts


def schedule(n: int, epochs: int, seed: int) -> list[int]:
    order = []
    for epoch in range(epochs):
        indices = list(range(n))
        random.Random(seed + epoch * 100003).shuffle(indices)
        order.extend(indices)
    return order


@torch.inference_mode()
def preference_eval(model, pairs: list[tuple[dict, dict]], pad_id: int) -> dict:
    model.eval()
    margins = []
    chosen_logps = []
    rejected_logps = []
    for chosen, rejected in pairs:
        ids, labels, attention = pair_batch(chosen, rejected, pad_id)
        values = assistant_mean_logps(model, ids, labels, attention)
        chosen_lp = float(values[0])
        rejected_lp = float(values[1])
        chosen_logps.append(chosen_lp)
        rejected_logps.append(rejected_lp)
        margins.append(chosen_lp - rejected_lp)
    return {
        "mean_chosen_logp": sum(chosen_logps) / len(chosen_logps),
        "mean_rejected_logp": sum(rejected_logps) / len(rejected_logps),
        "mean_margin_nats_per_token": sum(margins) / len(margins),
        "preference_accuracy": (
            sum(1 for margin in margins if margin > 0) / len(margins)
        ),
        "margin_min": min(margins),
        "margin_max": max(margins),
        "margins": margins,
    }


@torch.inference_mode()
def run_heldout(model, tok, eval_cfg: dict) -> list[dict]:
    old_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    model.eval()
    rows = []
    for item in eval_cfg["scenarios"]:
        messages = [
            {"role": "system", "content": eval_cfg["system"]},
            {"role": "user", "content": item["prompt"]},
        ]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        started = time.monotonic()
        out = model.generate(
            ids,
            max_new_tokens=int(eval_cfg["max_new_tokens"]),
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        elapsed = time.monotonic() - started
        generated = out[0, ids.shape[-1] :]
        rows.append(
            {
                "scenario_id": item["id"],
                "family": item["family"],
                "prompt": item["prompt"],
                "response": tok.decode(
                    generated, skip_special_tokens=True
                ).strip(),
                "generated_tokens": int(generated.numel()),
                "elapsed_seconds": elapsed,
                "tokens_per_second": (
                    int(generated.numel()) / elapsed if elapsed else None
                ),
            }
        )
    model.config.use_cache = old_cache
    return rows


def save_checkpoint(
    *,
    root: Path,
    model,
    optimizer,
    next_step: int,
    losses: list[float],
    preference_losses: list[float],
    anchor_losses: list[float],
    train_margins: list[float],
    sequence_tokens: int,
    supervised_tokens: int,
    training_seconds: float,
    source_run_id: str | None,
    parent_adapter_sha256: str,
    pair_dataset_sha256: str,
    member_contract_sha256: str,
    base_model: str,
    base_revision: str,
    parent_release: str,
    training_config: dict,
) -> None:
    tmp = root.with_name(root.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter", safe_serialization=True)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save(torch.get_rng_state(), tmp / "torch-rng.pt")
    write_json(
        tmp / "progress.json",
        {
            "schema_version": 1,
            "next_step": next_step,
            "losses": losses,
            "preference_losses": preference_losses,
            "anchor_losses": anchor_losses,
            "train_margins": train_margins,
            "sequence_tokens": sequence_tokens,
            "supervised_tokens": supervised_tokens,
            "training_seconds": training_seconds,
            "source_run_id": source_run_id,
            "parent_adapter_sha256": parent_adapter_sha256,
            "pair_dataset_sha256": pair_dataset_sha256,
            "member_contract_sha256": member_contract_sha256,
            "base_model": base_model,
            "base_revision": base_revision,
            "parent_release": parent_release,
            "training_config": training_config,
        },
    )
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)


def load_checkpoint(root: Path) -> dict | None:
    progress = root / "progress.json"
    if not progress.is_file():
        return None
    for required in (
        root / "adapter" / "adapter_config.json",
        root / "adapter" / "adapter_model.safetensors",
        root / "optimizer.pt",
        root / "torch-rng.pt",
    ):
        if not required.is_file():
            raise RuntimeError(f"incomplete checkpoint: {required} missing")
    return json.loads(progress.read_text())


def copy_resume(resume: Path, out: Path) -> None:
    if not resume.is_dir():
        raise FileNotFoundError(resume)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(resume, out)


def validate_complete(
    out: Path,
    member: dict,
    population: dict,
    parent_sha: str,
    pair_sha: str,
    campaign: dict,
    pref: dict,
) -> bool:
    result_path = out / "result.json"
    adapter = out / "adapter"
    if not result_path.is_file() or not adapter.is_dir():
        return False
    result = json.loads(result_path.read_text())
    checks = {
        "experiment": "VILLAGE-001D",
        "stage": "preference-continuation",
        "member_id": member["member_id"],
        "lineage": member["lineage"],
        "sibling": member["sibling"],
        "member_contract_sha256": population["member_contract_sha256"],
        "base_model": campaign["base_model"]["id"],
        "base_revision": campaign["base_model"]["revision"],
        "parent_release": pref["parent_release_tag"],
        "parent_adapter_sha256": parent_sha,
        "preference_dataset_sha256": pair_sha,
        "training_config": pref["preference_training"],
    }
    for key, expected in checks.items():
        if result.get(key) != expected:
            raise RuntimeError(
                f"resume result {key} {result.get(key)!r} != {expected!r}"
            )
    recorded = (result.get("adapter") or {}).get("sha256")
    if recorded != directory_hash(adapter):
        raise RuntimeError("resume result adapter directory hash mismatch")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/preference-config.json",
    )
    parser.add_argument("--member", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    pref = json.loads(args.config.read_text())
    campaign = json.loads((ROOT / pref["campaign_config"]).read_text())
    population, member = member_contract(campaign, args.member)
    train_cfg = pref["preference_training"]

    sft_result_path = (
        ROOT / pref["sft_evidence_root"] / "results" / f"{member['member_id']}.json"
    )
    if not sft_result_path.is_file():
        raise RuntimeError("VILLAGE-001C SFT evidence is missing")
    sft = json.loads(sft_result_path.read_text())
    if sft.get("member_contract_sha256") != population["member_contract_sha256"]:
        raise RuntimeError("SFT/member population contract mismatch")
    expected_parent_sha = sft["adapter"]["sha256"]
    actual_parent_sha = directory_hash(args.parent_adapter)
    if actual_parent_sha != expected_parent_sha:
        raise RuntimeError(
            f"parent adapter directory hash {actual_parent_sha} != "
            f"SFT evidence {expected_parent_sha}"
        )

    pairs_path = args.pairs
    if pairs_path.is_dir():
        pairs_path = pairs_path / "pairs.jsonl"
    validation = validate_file(
        pairs_path,
        member["member_id"],
        int(pref["pair_generation"]["pair_count"]),
    )
    pair_sha = sha256_file(pairs_path)
    pair_rows = [
        json.loads(line)
        for line in pairs_path.read_text().splitlines()
        if line.strip()
    ]
    if any(
        row["provenance"]["parent_adapter_sha256"] != expected_parent_sha
        for row in pair_rows
    ):
        raise RuntimeError("preference pair parent adapter hash mismatch")
    if any(
        row["provenance"]["member_contract_sha256"]
        != population["member_contract_sha256"]
        for row in pair_rows
    ):
        raise RuntimeError("preference pair member contract mismatch")

    eval_cfg = json.loads(
        (ROOT / campaign["corpus"]["common_heldout_config"]).read_text()
    )["eval"]

    out = args.out
    if args.resume:
        copy_resume(args.resume, out)
    else:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

    if validate_complete(
        out,
        member,
        population,
        expected_parent_sha,
        pair_sha,
        campaign,
        pref,
    ):
        print(
            json.dumps(
                {
                    "member": member["member_id"],
                    "resume": "complete_artifact_reused",
                    "preference_dataset_sha256": pair_sha,
                },
                indent=2,
            )
        )
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    seed = int(member["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    load_started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(
        campaign["base_model"]["id"],
        revision=campaign["base_model"]["revision"],
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        campaign["base_model"]["id"],
        revision=campaign["base_model"]["revision"],
        torch_dtype=torch.float32,
    )
    base.config.use_cache = False

    checkpoint_dir = out / "checkpoint"
    checkpoint = load_checkpoint(checkpoint_dir)
    if checkpoint:
        checkpoint_checks = {
            "parent_adapter_sha256": expected_parent_sha,
            "pair_dataset_sha256": pair_sha,
            "member_contract_sha256": population["member_contract_sha256"],
            "base_model": campaign["base_model"]["id"],
            "base_revision": campaign["base_model"]["revision"],
            "parent_release": pref["parent_release_tag"],
            "training_config": train_cfg,
        }
        for key, expected in checkpoint_checks.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(
                    f"checkpoint {key} {checkpoint.get(key)!r} != {expected!r}"
                )
        model = PeftModel.from_pretrained(
            base,
            checkpoint_dir / "adapter",
            is_trainable=True,
        )
    else:
        model = PeftModel.from_pretrained(
            base,
            args.parent_adapter,
            is_trainable=True,
        )
    model.config.use_cache = False
    load_seconds = time.monotonic() - load_started

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    max_length = int(train_cfg["max_length"])
    encoded_pairs = []
    for row in pair_rows:
        chosen = encode_response(
            tok,
            eval_cfg["system"],
            row["input"],
            row["chosen"],
            max_length,
        )
        rejected = encode_response(
            tok,
            eval_cfg["system"],
            row["input"],
            row["rejected"],
            max_length,
        )
        encoded_pairs.append((chosen, rejected))

    before = preference_eval(model, encoded_pairs, tok.pad_token_id)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    order = schedule(
        len(encoded_pairs),
        int(train_cfg["epochs"]),
        seed,
    )

    start_step = 0
    losses: list[float] = []
    preference_losses: list[float] = []
    anchor_losses: list[float] = []
    train_margins: list[float] = []
    sequence_tokens = 0
    supervised_tokens = 0
    training_seconds = 0.0
    resumed_from_run_id = None

    if checkpoint:
        start_step = int(checkpoint["next_step"])
        losses = [float(x) for x in checkpoint["losses"]]
        preference_losses = [
            float(x) for x in checkpoint["preference_losses"]
        ]
        anchor_losses = [float(x) for x in checkpoint["anchor_losses"]]
        train_margins = [float(x) for x in checkpoint["train_margins"]]
        sequence_tokens = int(checkpoint["sequence_tokens"])
        supervised_tokens = int(checkpoint["supervised_tokens"])
        training_seconds = float(checkpoint["training_seconds"])
        resumed_from_run_id = checkpoint.get("source_run_id")
        lengths = {
            len(losses),
            len(preference_losses),
            len(anchor_losses),
            len(train_margins),
        }
        if lengths != {start_step}:
            raise RuntimeError(
                f"checkpoint trace lengths {lengths} != next_step {start_step}"
            )
        if start_step < 0 or start_step > len(order):
            raise RuntimeError("checkpoint next_step outside schedule")
        optimizer.load_state_dict(
            torch.load(
                checkpoint_dir / "optimizer.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
        torch.set_rng_state(
            torch.load(
                checkpoint_dir / "torch-rng.pt",
                map_location="cpu",
                weights_only=False,
            )
        )

    beta = float(train_cfg["beta"])
    target_margin = float(train_cfg["target_margin_nats"])
    anchor_weight = float(train_cfg["chosen_nll_weight"])
    clip = float(train_cfg["gradient_clip"])
    checkpoint_interval = int(train_cfg["checkpoint_interval_steps"])

    model.train()
    segment_started = time.monotonic()
    for step in range(start_step, len(order)):
        chosen, rejected = encoded_pairs[order[step]]
        ids, labels, attention = pair_batch(
            chosen,
            rejected,
            tok.pad_token_id,
        )
        optimizer.zero_grad(set_to_none=True)
        values = assistant_mean_logps(model, ids, labels, attention)
        chosen_lp, rejected_lp = values[0], values[1]
        margin = chosen_lp - rejected_lp
        preference_loss = -torch.nn.functional.logsigmoid(
            beta * (margin - target_margin)
        )
        anchor_loss = -chosen_lp
        loss = preference_loss + anchor_weight * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            clip,
        )
        optimizer.step()

        losses.append(float(loss.detach()))
        preference_losses.append(float(preference_loss.detach()))
        anchor_losses.append(float(anchor_loss.detach()))
        train_margins.append(float(margin.detach()))
        sequence_tokens += (
            chosen["sequence_tokens"] + rejected["sequence_tokens"]
        )
        supervised_tokens += (
            chosen["supervised_tokens"] + rejected["supervised_tokens"]
        )

        next_step = step + 1
        if (
            next_step % checkpoint_interval == 0
            or next_step == len(order)
        ):
            training_seconds += time.monotonic() - segment_started
            save_checkpoint(
                root=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                next_step=next_step,
                losses=losses,
                preference_losses=preference_losses,
                anchor_losses=anchor_losses,
                train_margins=train_margins,
                sequence_tokens=sequence_tokens,
                supervised_tokens=supervised_tokens,
                training_seconds=training_seconds,
                source_run_id=os.environ.get("GITHUB_RUN_ID"),
                parent_adapter_sha256=expected_parent_sha,
                pair_dataset_sha256=pair_sha,
                member_contract_sha256=population["member_contract_sha256"],
                base_model=campaign["base_model"]["id"],
                base_revision=campaign["base_model"]["revision"],
                parent_release=pref["parent_release_tag"],
                training_config=train_cfg,
            )
            segment_started = time.monotonic()

    after = preference_eval(model, encoded_pairs, tok.pad_token_id)
    heldout = run_heldout(model, tok, eval_cfg)

    adapter_dir = out / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_bytes = sum(
        path.stat().st_size
        for path in adapter_dir.rglob("*")
        if path.is_file()
    )
    adapter_sha = directory_hash(adapter_dir)

    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "stage": "preference-continuation",
        "issue": 13,
        "member_id": member["member_id"],
        "lineage": member["lineage"],
        "sibling": member["sibling"],
        "specialization": member["specialization"],
        "member_seed": seed,
        "member_contract_sha256": population["member_contract_sha256"],
        "base_model": campaign["base_model"]["id"],
        "base_revision": campaign["base_model"]["revision"],
        "parent_release": pref["parent_release_tag"],
        "parent_adapter_sha256": expected_parent_sha,
        "preference_dataset": str(pairs_path),
        "preference_dataset_sha256": pair_sha,
        "preference_pair_ids": [row["pair_id"] for row in pair_rows],
        "preference_validation": validation,
        "training_config": train_cfg,
        "runner": {
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
        "packages": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "load_seconds": load_seconds,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": (
            trainable_parameters / total_parameters
            if total_parameters
            else 0
        ),
        "preference_before": before,
        "training": {
            "epochs": int(train_cfg["epochs"]),
            "optimizer_steps": len(order),
            "pair_count": len(encoded_pairs),
            "sequence_tokens": sequence_tokens,
            "supervised_tokens": supervised_tokens,
            "seconds": training_seconds,
            "sequence_tokens_per_second": (
                sequence_tokens / training_seconds
                if training_seconds
                else None
            ),
            "supervised_tokens_per_second": (
                supervised_tokens / training_seconds
                if training_seconds
                else None
            ),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "mean_loss": (
                sum(losses) / len(losses) if losses else None
            ),
            "mean_preference_loss": (
                sum(preference_losses) / len(preference_losses)
                if preference_losses
                else None
            ),
            "mean_anchor_nll": (
                sum(anchor_losses) / len(anchor_losses)
                if anchor_losses
                else None
            ),
            "mean_train_margin": (
                sum(train_margins) / len(train_margins)
                if train_margins
                else None
            ),
            "losses": losses,
            "preference_losses": preference_losses,
            "anchor_losses": anchor_losses,
            "train_margins": train_margins,
        },
        "preference_after": after,
        "recovery": {
            "checkpoint_interval_steps": checkpoint_interval,
            "resumed": bool(checkpoint),
            "resumed_from_step": start_step,
            "resumed_from_run_id": resumed_from_run_id,
            "final_checkpoint_step": len(order),
        },
        "peak_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "adapter": {
            "path": "adapter",
            "bytes": adapter_bytes,
            "sha256": adapter_sha,
        },
        "evaluation": {
            "system": eval_cfg["system"],
            "hidden_member_prompt_used": False,
            "max_new_tokens": int(eval_cfg["max_new_tokens"]),
            "responses": heldout,
        },
    }
    write_json(out / "result.json", result)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    print(
        json.dumps(
            {
                "member": member["member_id"],
                "cpu": result["runner"]["cpu_model"],
                "steps": result["training"]["optimizer_steps"],
                "margin": [
                    before["mean_margin_nats_per_token"],
                    after["mean_margin_nats_per_token"],
                ],
                "preference_accuracy": [
                    before["preference_accuracy"],
                    after["preference_accuracy"],
                ],
                "loss": [
                    result["training"]["first_loss"],
                    result["training"]["last_loss"],
                ],
                "adapter_sha256": adapter_sha,
                "held_out_responses": len(heldout),
                "resumed": result["recovery"]["resumed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
