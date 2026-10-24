#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import inspect
import json
import os
import platform
import random
import re
import resource
import sys
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["trigger", "state", "update", "action", "failure"]
STOP = {
    "the","a","an","and","or","to","of","in","on","with","while","when","if","is","are",
    "be","from","for","it","its","this","that","after","only","can","will","into","by",
}

def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = p.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()

def load_lane(config_path: Path, lane_id: str) -> tuple[dict, dict]:
    cfg = json.loads(config_path.read_text())
    lane = next((x for x in cfg["lanes"] if x["id"] == lane_id), None)
    if lane is None:
        raise SystemExit(f"unknown lane {lane_id}")
    return cfg, lane

def prompt(system: str, item: dict) -> str:
    return f"""Task: {system}
Input behaviour idea:
{item['input']}
Output JSON:
"""

def target(item: dict) -> str:
    return json.dumps(item["output"], ensure_ascii=False, separators=(",", ":"))

def parse_first_json(text: str):
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None

def words(text: str) -> set[str]:
    return {
        x for x in re.findall(r"[a-z0-9_]+", text.lower())
        if len(x) >= 3 and x not in STOP
    }

def field_similarity(actual: str, expected: str) -> float:
    a = " ".join(str(actual).lower().split())
    e = " ".join(str(expected).lower().split())
    seq = difflib.SequenceMatcher(None, a, e).ratio()
    aw, ew = words(a), words(e)
    if not aw and not ew:
        f1 = 1.0
    elif not aw or not ew:
        f1 = 0.0
    else:
        inter = len(aw & ew)
        precision = inter / len(aw)
        recall = inter / len(ew)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return 0.5 * seq + 0.5 * f1

def score_output(text: str, expected: dict) -> dict:
    obj = parse_first_json(text)
    exact = bool(obj is not None and set(obj) == set(KEYS) and len(obj) == len(KEYS))
    key_count = sum(k in obj for k in KEYS) if obj else 0
    field_scores = {}
    if obj:
        for key in KEYS:
            field_scores[key] = field_similarity(str(obj.get(key, "")), str(expected[key])) if key in obj else 0.0
    else:
        field_scores = {k: 0.0 for k in KEYS}
    return {
        "json": obj is not None,
        "exact_schema": exact,
        "expected_key_count": key_count,
        "field_similarity": field_scores,
        "mechanical_similarity_proxy": sum(field_scores.values()) / len(KEYS),
        "parsed": obj,
    }

def encode(tok, system: str, item: dict, maxlen: int = 1024):
    p = prompt(system, item)
    tgt = target(item)
    plain = p + tgt
    pi = tok(p, add_special_tokens=True)["input_ids"]
    fi = tok(plain + (tok.eos_token or ""), add_special_tokens=True)["input_ids"]
    common = 0
    for a, b in zip(pi, fi):
        if a != b:
            break
        common += 1
    if len(fi) > maxlen:
        raise RuntimeError(f"example length {len(fi)} exceeds {maxlen}")
    labels = [-100] * common + fi[common:]
    if all(x == -100 for x in labels):
        raise RuntimeError("assistant target fully masked")
    return {
        "input_ids": torch.tensor([fi], dtype=torch.long),
        "attention_mask": torch.ones((1, len(fi)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "sequence_units": len(fi),
        "supervised_units": len(fi) - common,
        "utf8_bytes": len(plain.encode("utf-8")),
        "characters": len(plain),
    }

def materialize_items(base: list[dict], lane: dict) -> list[tuple[int, dict]]:
    items = [(i, x) for i, x in enumerate(base)]
    if lane["material"] == "failure_weighted":
        for idx in lane.get("material_repeat_indices", []):
            items.append((idx, base[idx]))
    elif lane["material"] != "baseline":
        raise RuntimeError(f"unknown material profile {lane['material']}")
    return items

def training_schedule(n: int, steps: int, seed: int) -> list[int]:
    schedule = []
    epoch = 0
    while len(schedule) < steps:
        order = list(range(n))
        rng = random.Random(seed + epoch * 1000003)
        rng.shuffle(order)
        schedule.extend(order)
        epoch += 1
    return schedule[:steps]

@torch.inference_mode()
def evaluate(model, tok, system: str, items: list[dict], max_new_tokens: int) -> dict:
    old_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    model.eval()
    rows = []
    for idx, item in enumerate(items):
        p = prompt(system, item)
        ids = tok(p, return_tensors="pt", add_special_tokens=True)
        t0 = time.monotonic()
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        elapsed = time.monotonic() - t0
        gen = out[0, ids["input_ids"].shape[-1]:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        raw_bytes = text.encode("utf-8")
        eos_reached = bool(tok.eos_token_id is not None and int(tok.eos_token_id) in [int(x) for x in gen.tolist()])
        score = score_output(text, item["output"])
        rows.append({
            "case_id": f"E{idx+1:02d}",
            "input": item["input"],
            "expected": item["output"],
            "output": text,
            "score": score,
            "generated_tokenizer_units": int(gen.numel()),
            "generated_utf8_bytes": len(raw_bytes),
            "generated_characters": len(text),
            "elapsed_seconds": elapsed,
            "tokenizer_units_per_second": int(gen.numel()) / elapsed if elapsed else None,
            "utf8_bytes_per_second": len(raw_bytes) / elapsed if elapsed else None,
            "characters_per_second": len(text) / elapsed if elapsed else None,
            "eos_reached": eos_reached,
            "budget_exhausted": bool(int(gen.numel()) >= max_new_tokens and not eos_reached),
        })
    model.config.use_cache = old_cache
    return {
        "max_new_tokens": max_new_tokens,
        "rows": rows,
        "json_rate": sum(r["score"]["json"] for r in rows) / len(rows),
        "exact_schema_rate": sum(r["score"]["exact_schema"] for r in rows) / len(rows),
        "mean_expected_key_count": sum(r["score"]["expected_key_count"] for r in rows) / len(rows),
        "mean_mechanical_similarity_proxy": sum(r["score"]["mechanical_similarity_proxy"] for r in rows) / len(rows),
        "budget_exhaustion_rate": sum(r["budget_exhausted"] for r in rows) / len(rows),
        "mean_characters_per_second": sum(r["characters_per_second"] for r in rows) / len(rows),
        "mean_utf8_bytes_per_second": sum(r["utf8_bytes_per_second"] for r in rows) / len(rows),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--lane", required=True)
    args = ap.parse_args()

    cfg, lane = load_lane(args.config, args.lane)
    base_model = cfg["base_model"]
    base_revision = cfg["base_revision"]
    data_path = ROOT / cfg["dataset"]
    data = json.loads(data_path.read_text())
    system = data["system"]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(lane["seed"])
    torch.manual_seed(lane["seed"])

    t_load = time.monotonic()
    tok = AutoTokenizer.from_pretrained(
        base_model,
        revision=base_revision,
        trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.pad_token_id = 0
    tokenizer_source = inspect.getsourcefile(tok.__class__)
    tokenizer_source_sha = sha256_file(Path(tokenizer_source)) if tokenizer_source else None

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_revision,
        torch_dtype=torch.float32,
    )
    base.config.use_cache = False

    targets = []
    for name, module in base.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"} and isinstance(module, torch.nn.Linear):
            targets.append(suffix)
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError("no expected LoRA target modules found")

    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lane["lora_r"]),
        lora_alpha=int(lane["lora_alpha"]),
        lora_dropout=float(lane["lora_dropout"]),
        bias="none",
        target_modules=targets,
    ))
    load_seconds = time.monotonic() - t_load

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lane["lr"]),
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    before = evaluate(model, tok, system, data["eval"], int(cfg["eval_max_new_tokens"]))

    material = materialize_items(data["train"], lane)
    encoded = [(src_idx, encode(tok, system, item)) for src_idx, item in material]
    schedule = training_schedule(len(encoded), int(cfg["steps"]), int(lane["seed"]))

    losses = []
    sequence_units = supervised_units = utf8_bytes = characters = 0
    source_indices = []
    model.train()
    model.config.use_cache = False
    t_train = time.monotonic()
    for pos in schedule:
        source_idx, ex = encoded[pos]
        source_indices.append(source_idx)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=ex["input_ids"],
            attention_mask=ex["attention_mask"],
            labels=ex["labels"],
        )
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        losses.append(float(loss.detach()))
        sequence_units += ex["sequence_units"]
        supervised_units += ex["supervised_units"]
        utf8_bytes += ex["utf8_bytes"]
        characters += ex["characters"]
    train_seconds = time.monotonic() - t_train

    after = evaluate(model, tok, system, data["eval"], int(cfg["eval_max_new_tokens"]))

    out_dir = ROOT / "train002b" / lane["id"] / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    adapter_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    adapter_sha256 = directory_hash(out_dir)

    material_identity = [
        {"source_index": src_idx, "input": data["train"][src_idx]["input"], "output": data["train"][src_idx]["output"]}
        for src_idx, _ in material
    ]
    material_sha256 = sha256_bytes(json.dumps(material_identity, sort_keys=True, ensure_ascii=False).encode())

    result = {
        "experiment": "TRAIN-002b",
        "issue": 1,
        "lane": lane,
        "base_model": base_model,
        "base_revision": base_revision,
        "dataset": cfg["dataset"],
        "dataset_sha256": sha256_file(data_path),
        "tokenizer": {
            "class": tok.__class__.__name__,
            "source_file": Path(tokenizer_source).name if tokenizer_source else None,
            "source_sha256": tokenizer_source_sha,
            "unit_note": "void.1 tokenizer units are byte-oriented and must not be compared directly with ordinary subword-token throughput.",
        },
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
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "lora_targets": targets,
        "training": {
            "steps": len(schedule),
            "material_profile": lane["material"],
            "material_item_count": len(material),
            "material_sha256": material_sha256,
            "schedule_source_indices": source_indices,
            "sequence_tokenizer_units": sequence_units,
            "supervised_tokenizer_units": supervised_units,
            "utf8_bytes": utf8_bytes,
            "characters": characters,
            "seconds": train_seconds,
            "sequence_units_per_second": sequence_units / train_seconds,
            "supervised_units_per_second": supervised_units / train_seconds,
            "utf8_bytes_per_second": utf8_bytes / train_seconds,
            "characters_per_second": characters / train_seconds,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses),
            "losses": losses,
        },
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "adapter": {
            "bytes": adapter_bytes,
            "sha256": adapter_sha256,
        },
        "evaluation_before": before,
        "evaluation_after": after,
    }

    results = ROOT / "results" / "train002b"
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{lane['id']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "lane": lane["id"],
        "cpu": result["runner"]["cpu_model"],
        "train_seconds": train_seconds,
        "train_utf8_bytes_per_second": result["training"]["utf8_bytes_per_second"],
        "loss": [losses[0], losses[-1]],
        "before_exact_schema": before["exact_schema_rate"],
        "after_exact_schema": after["exact_schema_rate"],
        "after_budget_exhaustion": after["budget_exhaustion_rate"],
        "after_mechanical_proxy": after["mean_mechanical_similarity_proxy"],
        "adapter_bytes": adapter_bytes,
        "adapter_sha256": adapter_sha256,
    }, indent=2))

if __name__ == "__main__":
    main()
