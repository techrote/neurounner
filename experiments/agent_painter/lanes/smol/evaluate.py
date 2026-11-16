#!/usr/bin/env python3
"""Strict untuned-vs-SMOL-adapter DEVELOPMENT diagnosis; never executes native Art.

One original greedy completion per distinct linguistic prompt per arm. Repeated
host-seed rows reuse those exact bytes and are reported as seed variants, not
independent language predictions. FINAL targets/files are never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import sys
import time

ART = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ART / "training"))
from common import (compiler_module, profile, read_json, relative_file, sha256_file,
                    strict_completion, strict_json, verify_checkpoint, write_json)
from lane_adapters import adapter_identity, encode_prompt
from numerics import configure_cpu
from train import load_base


def development_rows(directory: Path, expected_source: str, compiler_sha256: str):
    """Open only the DEVELOPMENT split and integrity-check its manifest and rows."""
    manifest = read_json(relative_file(directory, "admitted-corpus.json"))
    if (manifest.get("schema_version") != 1 or manifest.get("state") != "native-admitted"
            or manifest.get("source_sha") != expected_source
            or manifest.get("compiler_sha256") != compiler_sha256):
        raise ValueError("qualified native corpus identity mismatch")
    groups = manifest.get("structural_groups", {})
    if set(groups) != {"TRAIN", "DEVELOPMENT", "FINAL"}:
        raise ValueError("invalid structural split metadata")
    if set(groups["DEVELOPMENT"]) & (set(groups["TRAIN"]) | set(groups["FINAL"])):
        raise ValueError("DEVELOPMENT structural group leakage")
    info = manifest["split_files"]["DEVELOPMENT"]
    path = relative_file(directory, info["path"])
    if sha256_file(path) != info["sha256"]:
        raise ValueError("DEVELOPMENT cohort custody hash mismatch")
    rows = [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != info["count"] or len(set(row["id"] for row in rows)) != len(rows):
        raise ValueError("DEVELOPMENT row count or ID mismatch")
    for row in rows:
        group = row.get("group_id", row.get("structural_group"))
        if (row.get("split") != "DEVELOPMENT" or group not in groups["DEVELOPMENT"]
                or row.get("structural_group", group) != group
                or row.get("family") not in {"greebles", "pipework", "architecture"}
                or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                or type(row.get("seed")) is not int or not 0 <= row["seed"] < 2**32
                or not isinstance(row["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["id"])):
            raise ValueError("invalid DEVELOPMENT row; no silent cohort repair")
    return rows, manifest


def run_arm(model, tokenizer, system, rows, compiler, root: Path, max_tokens: int):
    import torch
    cache = {}
    records = []
    for row in rows:
        prompt = row["prompt"]
        cached = prompt in cache
        if not cached:
            ids = encode_prompt(tokenizer, prompt, system, profile("smol", require_supported=True))
            if len(ids) + max_tokens > 2048:
                raise ValueError("DEVELOPMENT request exceeds model context, no truncation")
            inputs = torch.tensor([ids], dtype=torch.long)
            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=max_tokens, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True)[0, len(ids):].tolist()
            elapsed = time.monotonic() - started
            cache[prompt] = (ids, generated, elapsed)
        ids, generated, elapsed = cache[prompt]
        verdict = strict_completion(tokenizer, generated, compiler.validate_program)
        accepted = bool(verdict["first_pass_accepted"])
        compiled = None
        if accepted:
            try:
                compiled = compiler.compile_program(
                    strict_json(verdict["raw_text"]), seed=row["seed"],
                    focus=(0, 0, 96, 96), material=58, allowed_materials=(58,))
            except (TypeError, KeyError, ValueError) as exc:
                accepted = False
                verdict["compile_failure"] = str(exc)
        directory = root / row["id"]
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "completion.txt").write_text(verdict["raw_text"], encoding="utf-8")
        write_json(directory / "raw-output.json", verdict)
        if accepted:
            write_json(directory / "compiled-proposal.json", compiled)
        record = {
            "id": row["id"], "family": row["family"],
            "structural_group": row.get("group_id", row.get("structural_group")),
            "host_seed": row["seed"],
            "prompt": prompt, "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
            "completion_sha256": sha256_file(directory / "completion.txt"),
            "input_tokens": len(ids), "output_tokens": len(generated),
            "eos_terminated": verdict["eos_terminated"],
            "json_valid": verdict["json_valid"],
            "program_valid": verdict["program_valid"],
            "compiler_accepted": accepted,
            "native_executed": False, "native_acknowledged_cells": None,
            "native_image": None, "visual_quality": "NOT_ASSESSED",
            "model_generation_reused_for_host_seed_variant": cached,
            "inference_seconds_for_unique_prompt": elapsed if not cached else 0.0,
            "proposed_operations": len(compiled["operations"]) if accepted else 0,
            "rejection": verdict.get("failure", verdict.get("compile_failure")),
        }
        write_json(directory / "receipt.json", record)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--training-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.max_new_tokens <= 768:
        parser.error("output must be new and token generation must be bounded")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = verify_checkpoint(args.checkpoint)
        config = receipt["config"]
        selected = profile("smol", require_supported=True)
        if (config.get("profile") != "smol" or config.get("model") != selected
                or config.get("source_sha") != args.training_source
                or config.get("trainer_sha256") != sha256_file(ART / "training/train.py")
                or config.get("contract_sha256") != sha256_file(ART / "training/common.py")
                or config.get("compiler_sha256") != sha256_file(ART / "compiler.py")
                or config.get("adapter_contract") != adapter_identity(selected)):
            raise ValueError("trained checkpoint is not compatible with this evaluation source")
        numerics, runtime = configure_cpu(1)
        if config.get("numerics") != numerics:
            raise ValueError("CPU numerical profile differs from trained checkpoint")
        rows, manifest = development_rows(args.corpus, config["native_source_sha"], config["compiler_sha256"])
        compiler = compiler_module()
        from peft import PeftModel
        base, tokenizer = load_base(args.model, selected)
        base.eval()
        baseline = run_arm(base, tokenizer, config["system"], rows, compiler,
                           args.output / "untuned-base", args.max_new_tokens)
        tuned = PeftModel.from_pretrained(
            base, args.checkpoint / "adapter", is_trainable=False, local_files_only=True)
        tuned.eval()
        candidate = run_arm(tuned, tokenizer, config["system"], rows, compiler,
                            args.output / "trained-candidate", args.max_new_tokens)
        def scores(records):
            return {
                family: {
                    "host_seed_cases": len(family_rows),
                    "unique_language_prompts": len({r["prompt_sha256"] for r in family_rows}),
                    "eos": sum(r["eos_terminated"] for r in family_rows),
                    "json": sum(r["json_valid"] for r in family_rows),
                    "compiler_accepted": sum(r["compiler_accepted"] for r in family_rows),
                    "native_accepted": None,
                }
                for family in ("greebles", "pipework", "architecture")
                if (family_rows := [r for r in records if r["family"] == family])
            }
        summary = {
            "schema": "cybersand.agent-painter.smol-development-diagnosis/1",
            "qualification": "raw model proposals and deterministic compiler only; NO NATIVE ART",
            "model_id": selected["model_id"], "model_revision": selected["revision"],
            "training_source_sha": args.training_source,
            "evaluation_source_sha": os.getenv("GITHUB_SHA"),
            "training_checkpoint": args.checkpoint.name,
            "adapter_sha256": sha256_file(args.checkpoint / "adapter/adapter_model.safetensors"),
            "native_source_sha": manifest["source_sha"],
            "corpus_manifest_sha256": sha256_file(args.corpus / "admitted-corpus.json"),
            "development_file_sha256": manifest["split_files"]["DEVELOPMENT"]["sha256"],
            "development_seed_cases": len(rows),
            "development_unique_prompt_count": len({row["prompt"] for row in rows}),
            "development_structural_groups": len({row.get("group_id", row.get("structural_group")) for row in rows}),
            "max_new_tokens": args.max_new_tokens,
            "numerical_profile": numerics, "numerical_runtime": runtime,
            "untuned_base": scores(baseline), "trained_candidate": scores(candidate),
            "final_targets_accessed": False, "native_execution": False,
            "model_scores_are_not_visual_or_native_success": True,
        }
        write_json(args.output / "summary.json", summary)
        print(__import__("json").dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        write_json(args.output / "failure.json", {"error_type": type(exc).__name__,
                                                  "reason": str(exc),
                                                  "native_execution": False,
                                                  "final_targets_accessed": False})
        raise


if __name__ == "__main__":
    main()
