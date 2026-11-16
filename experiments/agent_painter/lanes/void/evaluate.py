#!/usr/bin/env python3
"""Strict VOID base-vs-adapter DEVELOPMENT evaluation. NO native writes or FINAL use."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
from common import (ROOT, compiler_module, profile, read_json, relative_file,
                    sha256_file, strict_completion, strict_json, verify_checkpoint,
                    verify_file, write_json)
from lane_adapters import adapter_identity, encode_prompt
from numerics import configure_cpu
from train import SYSTEM, load_base, source_sha

FAMILIES = ("architecture", "greebles", "pipework")


def development(directory):
    manifest = read_json(relative_file(directory, "admitted-corpus.json"))
    groups = manifest.get("structural_groups", {})
    if manifest.get("state") != "native-admitted" or set(groups) != {"TRAIN", "DEVELOPMENT", "FINAL"}:
        raise ValueError("positive native admission and three structural partitions required")
    if set(groups["DEVELOPMENT"]) & (set(groups["TRAIN"]) | set(groups["FINAL"])):
        raise ValueError("DEVELOPMENT split leakage")
    record = manifest["split_files"]["DEVELOPMENT"]
    path = verify_file(directory, record["path"], record["sha256"])
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = strict_json(line)
        group = data.get("group_id", data.get("structural_group"))
        focus = data.get("compiled", {}).get("focus")
        material = data.get("compiled", {}).get("material")
        if (data.get("split") != "DEVELOPMENT" or group not in groups["DEVELOPMENT"]
                or data.get("family") not in FAMILIES
                or not isinstance(data.get("id"), str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", data["id"]) is None
                or not isinstance(data.get("prompt"), str) or not data["prompt"].strip()
                or type(data.get("seed")) is not int or not 0 <= data["seed"] <= 2**32 - 1
                or not isinstance(focus, list) or len(focus) != 4
                or not all(type(x) is int for x in focus)
                or not (96 <= focus[2] <= 512 and 96 <= focus[3] <= 512)
                or type(material) is not int):
            raise ValueError("malformed DEVELOPMENT host contract")
        # Do NOT carry the target program, completion or expected operations forward.
        rows.append({"id": data["id"], "family": data["family"],
                     "prompt": data["prompt"], "seed": data["seed"],
                     "focus": focus, "material": material, "group": group})
    if len(rows) != record["count"] or len({x["id"] for x in rows}) != len(rows):
        raise ValueError("DEVELOPMENT file count or sample IDs differ")
    if {x["family"] for x in rows} != set(FAMILIES):
        raise ValueError("DEVELOPMENT missing a family")
    return sorted(rows, key=lambda x: (x["family"], x["id"])), manifest


def evaluate_case(model, tokenizer, selected, compiler, row, arm, limit, folder, cached=None):
    import torch
    prompt_ids = encode_prompt(tokenizer, row["prompt"], SYSTEM, selected)
    if len(prompt_ids) + limit > selected["max_sequence_tokens"]:
        raise ValueError("untruncated prompt + generation would exceed VOID context")
    if cached is None:
        inputs = torch.tensor([prompt_ids], dtype=torch.long)
        begin = time.monotonic()
        with model.disable_adapter() if arm == "base" else nullcontext():
            with torch.inference_mode():
                ids = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=limit, do_sample=False, use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id)[0, len(prompt_ids):].tolist()
        elapsed = time.monotonic() - begin
    else:
        # One actual language request yields one immutable first pass; only host
        # seeds/focus/material are varied during independently checked compile.
        ids = list(cached)
        elapsed = 0.0
    verdict = strict_completion(tokenizer, ids, compiler.validate_program)
    accepted = bool(verdict["first_pass_accepted"])
    compiled = None
    if accepted:
        try:
            compiled = compiler.compile_program(
                strict_json(verdict["raw_text"]), seed=row["seed"],
                focus=tuple(row["focus"]), material=row["material"],
                allowed_materials=(row["material"],))
        except ValueError as exc:
            accepted = False
            verdict["compile_failure"] = str(exc)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "completion.txt").write_text(verdict["raw_text"], encoding="utf-8")
    write_json(folder / "raw-output.json", verdict)
    write_json(folder / "art_batch.json", compiled["operations"] if accepted else [])
    if accepted:
        write_json(folder / "compiled.json", compiled)
    receipt = {
        "case": row["id"], "family": row["family"], "group": row["group"],
        "arm": arm, "original_prompt": row["prompt"], "host_seed": row["seed"],
        "host_focus": row["focus"], "host_material": row["material"],
        "first_pass_accepted": accepted, "eos_terminated": verdict["eos_terminated"],
        "json_valid": verdict["json_valid"], "program_valid": verdict["program_valid"],
        "compiler_accepted": accepted, "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(ids), "generation_seconds": elapsed,
        "actual_model_generation": cached is None,
        "reused_first_pass_for_host_seed": cached is not None,
        "completion_sha256": sha256_file(folder / "completion.txt"),
        "operations_sha256": compiled["operations_sha256"] if accepted else None,
        "decoding": "single greedy first pass; no retries or repair",
        "native_execution": False, "native_acknowledged_cells": None,
        "visual_quality_claim": False}
    write_json(folder / "receipt.json", receipt)
    print(json.dumps({"case": row["id"], "arm": arm, "accepted": accepted,
                      "eos": verdict["eos_terminated"], "tokens": len(ids),
                      "seconds": round(elapsed, 3)}), flush=True)
    return receipt, ids


def verify_development_pairs(rows):
    """Keep six structural language requests and both native host-seed views."""
    groups = {}
    for row in rows:
        key = row["group"]
        if key not in groups:
            groups[key] = {"prompt": row["prompt"], "family": row["family"], "seeds": set()}
        record = groups[key]
        if (record["prompt"] != row["prompt"] or record["family"] != row["family"]
                or row["seed"] in record["seeds"]):
            raise ValueError("structural group changes its language request or repeats a seed")
        record["seeds"].add(row["seed"])
    if (len(rows) != 12 or len(groups) != 6
            or any(record["seeds"] != {104729, 130363} for record in groups.values())):
        raise ValueError("frozen DEVELOPMENT requires six requests with two host-seed views")
    return len(groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "checkpoint", "corpus", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=768)
    args = p.parse_args()
    if args.output.exists() or not 256 <= args.max_new_tokens <= 1024:
        p.error("output must be new; generation budget must be 256–1024")
    receipt = verify_checkpoint(args.checkpoint)
    cfg = receipt["config"]
    selected = profile("void", require_supported=True)
    if (cfg["profile"] != "void" or cfg["model"] != selected
            or cfg["source_sha"] != source_sha()
            or cfg["adapter_contract"] != adapter_identity(selected)
            or cfg["compiler_sha256"] != sha256_file(ROOT / "compiler.py")
            or cfg["system"] != SYSTEM):
        raise ValueError("exact source/model/framing checkpoint mismatch")
    rows, admission = development(args.corpus)
    request_count = verify_development_pairs(rows)
    if cfg["corpus_manifest_sha256"] != sha256_file(args.corpus / "admitted-corpus.json"):
        raise ValueError("DEVELOPMENT manifest differs from training corpus")
    numeric, runtime = configure_cpu(1)
    if cfg["numerics"] != numeric:
        raise ValueError("CPU numerical profile mismatch")
    from peft import PeftModel
    base, tokenizer = load_base(args.model, selected)
    model = PeftModel.from_pretrained(base, args.checkpoint / "adapter",
                                     is_trainable=False, local_files_only=True)
    model.eval()
    compiler = compiler_module()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "identity.json", {
        "source_sha": cfg["source_sha"], "native_source_sha": admission["source_sha"],
        "corpus_sha256": cfg["corpus_manifest_sha256"],
        "model_id": selected["model_id"], "model_revision": selected["revision"],
        "adapter_sha256": sha256_file(args.checkpoint / "adapter" / "adapter_model.safetensors"),
        "inventory_sha256": sha256_file(args.checkpoint / "inventory.json"),
        "split": "DEVELOPMENT", "cases": len(rows), "arms": ["base", "adapter"],
        "max_new_tokens": args.max_new_tokens, "native_execution": False,
        "visual_quality_claim": False})
    results = []
    first_passes = {}
    for row in rows:
        for arm in ("base", "adapter"):
            key = (arm, row["group"], row["prompt"])
            cached = first_passes.get(key)
            receipt, ids = evaluate_case(
                model, tokenizer, selected, compiler, row, arm, args.max_new_tokens,
                args.output / arm / row["id"], cached=cached)
            if cached is None:
                first_passes[key] = tuple(ids)
            results.append(receipt)
            write_json(args.output / "partial-summary.json", {
                "attempts": len(results), "expected": 2 * len(rows),
                "actual_model_generations": len(first_passes),
                "accepted": sum(x["first_pass_accepted"] for x in results),
                "native_execution": False})
    summary = {"cases": len(rows), "distinct_language_requests": request_count,
               "host_seed_views_per_arm": len(rows), "arms": {},
               "native_execution": False, "visual_quality_claim": False, "complete": True}
    for arm in ("base", "adapter"):
        subset = [r for r in results if r["arm"] == arm]
        summary["arms"][arm] = {
            "requests": len(subset),
            "actual_model_generations": sum(r["actual_model_generation"] for r in subset),
            "reused_first_pass_views": sum(r["reused_first_pass_for_host_seed"] for r in subset),
            "first_pass_accepted": sum(r["first_pass_accepted"] for r in subset),
            "eos_terminated": sum(r["eos_terminated"] for r in subset),
            "json_valid": sum(r["json_valid"] for r in subset),
            "generation_seconds": sum(r["generation_seconds"] for r in subset),
            "per_family": {family: {
                "requests": sum(r["family"] == family for r in subset),
                "accepted": sum(r["first_pass_accepted"] for r in subset if r["family"] == family)}
                for family in FAMILIES}}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
