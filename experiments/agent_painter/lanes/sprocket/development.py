#!/usr/bin/env python3
"""SPROCKET development-only untouched first-pass parent/adapter comparison.

This intentionally opens DEVELOPMENT rows only. It never accesses FINAL targets,
does not repair generations and never claims a compiled proposal was native paint.
"""
from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from common import (compiler_module, profile, read_json, sha256_file,
                    strict_completion, strict_json, verify_checkpoint,
                    verify_file, write_json)
from lane_adapters import adapter_identity, encode_prompt
from numerics import configure_cpu
from train import SYSTEM, load_base

NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
TRAINING_SOURCE = "591b83bc383b31a0ae2e3031b4a501b0a88ddc39"
SEEDS = {104729, 130363}
FAMILIES = {"greebles", "pipework", "architecture"}


def development_groups(directory):
    directory = Path(directory)
    manifest_file = directory / "admitted-corpus.json"
    if sha256_file(manifest_file) != CORPUS_SHA:
        raise ValueError("DEVELOPMENT admission manifest differs from the pinned native source")
    manifest = read_json(manifest_file)
    if (manifest.get("state") != "native-admitted" or
            manifest.get("source_sha") != NATIVE_SOURCE or
            manifest.get("compiler_sha256") != sha256_file(ROOT / "compiler.py")):
        raise ValueError("native runtime/source or compiler mismatch")
    groups = manifest["structural_groups"]
    if set(groups) != {"TRAIN", "DEVELOPMENT", "FINAL"}:
        raise ValueError("missing structural partitions")
    if len(set().union(*(set(value) for value in groups.values()))) != sum(len(value) for value in groups.values()):
        raise ValueError("structural partition leakage")
    # Only metadata for FINAL is touched; FINAL JSONL, targets and images are never opened.
    entry = manifest["split_files"]["DEVELOPMENT"]
    file = verify_file(directory, entry["path"], entry["sha256"])
    rows = [strict_json(x) for x in file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != entry["count"] or len(rows) != 12 or len({x["id"] for x in rows}) != len(rows):
        raise ValueError("fixed DEVELOPMENT cohort not exactly 12 original admitted rows")
    result = collections.defaultdict(list)
    for row in rows:
        group = row.get("structural_group")
        admission = row.get("native_admission", {})
        if (row.get("split") != "DEVELOPMENT" or group not in groups["DEVELOPMENT"] or
                row.get("family") not in FAMILIES or row.get("seed") not in SEEDS or
                not isinstance(row.get("prompt"), str) or not row["prompt"].strip() or
                admission.get("native_execution") is not True or
                admission.get("verified") is not True or
                admission.get("state") != "PASS" or
                admission.get("source_sha") != NATIVE_SOURCE):
            raise ValueError("DEVELOPMENT row native admission/identity mismatch: " + str(row.get("id")))
        for kind in ("png", "readback", "receipt"):
            verify_file(directory, admission["paths"][kind], admission[kind + "_sha256"])
        result[group].append(row)
    if len(result) != 6 or len(groups["DEVELOPMENT"]) != 6:
        raise ValueError("fixed DEVELOPMENT requires six independent structural groups")
    for group, variants in result.items():
        if (len(variants) != 2 or {r["seed"] for r in variants} != SEEDS or
                len({r["prompt"] for r in variants}) != 1 or
                len({r["family"] for r in variants}) != 1):
            raise ValueError("development group is not one language request with two native seeds: " + group)
    if {r["family"] for variants in result.values() for r in variants} != FAMILIES:
        raise ValueError("one of the three target families is missing")
    return dict(sorted(result.items()))


def evaluate(model, tokenizer, groups, compiler, output, arm, token_budget, torch):
    model.eval()
    records = []
    started = time.monotonic()
    for group, variants in groups.items():
        exemplar = variants[0]
        prompt = exemplar["prompt"]
        prefix = encode_prompt(tokenizer, prompt, SYSTEM, profile("sprocket", require_supported=True))
        if len(prefix) + token_budget > 2048:
            raise ValueError("context budget would truncate original DEVELOPMENT request")
        ids = torch.tensor([prefix], dtype=torch.long)
        begin = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=token_budget, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                use_cache=True)[0, len(prefix):].tolist()
        verdict = strict_completion(tokenizer, generated, compiler.validate_program)
        accepted = 0
        seed_results = []
        for row in variants:
            proposed = None
            compile_error = None
            if verdict["first_pass_accepted"]:
                try:
                    proposed = compiler.compile_program(
                        strict_json(verdict["raw_text"]), seed=row["seed"],
                        focus=(0, 0, 96, 96), material=58, allowed_materials=(58,))
                except (ValueError, TypeError, KeyError) as exc:
                    compile_error = str(exc)
            valid = proposed is not None
            accepted += int(valid)
            seed_record = {
                "row_id": row["id"], "seed": row["seed"], "admitted_teacher_sha256": row["native_admission"]["program_sha256"],
                "first_pass_compiler_accepted": valid, "compile_error": compile_error,
                "operations": proposed["operations"] if valid else [],
                "operations_sha256": proposed["operations_sha256"] if valid else None,
                "native_execution": False, "native_acknowledged_cells": None,
                "native_png": None, "visual_quality_claim": False,
            }
            seed_results.append(seed_record)
        records.append({
            "arm": arm, "family": exemplar["family"], "structural_group": group,
            "prompt": prompt, "prefix_ids": prefix, "output_token_ids": generated,
            "raw": verdict, "input_tokens": len(prefix), "output_tokens": len(generated),
            "seconds": time.monotonic() - begin,
            "admitted_seed_programs": accepted, "seed_results": seed_results,
        })
        print(arm, group, "EOS", verdict["eos_terminated"],
              "program", verdict["program_valid"], "compiled", accepted, "/2", flush=True)
    accepted_requests = sum(x["raw"]["first_pass_accepted"] for x in records)
    accepted_seed_programs = sum(x["admitted_seed_programs"] for x in records)
    summary = {
        "arm": arm, "independent_language_requests": len(records), "seed_variants": sum(len(x["seed_results"]) for x in records),
        "eos_terminated": sum(x["raw"]["eos_terminated"] for x in records),
        "schema_first_pass_accepted": accepted_requests, "compiler_seed_accepted": accepted_seed_programs,
        "wall_seconds": time.monotonic() - started, "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "native_execution_attempts": 0, "native_confirmed_paintings": 0,
        "no_retries_repairs_fallback": True, "final_targets_opened": False,
    }
    write_json(output / (arm + "-raw-development.json"), records)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    if not 1 <= args.max_new_tokens <= 1024 or args.output.exists():
        raise ValueError("invalid bounded output/token request or output already exists")
    checkpoint = verify_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    selected = profile("sprocket", require_supported=True)
    # A standalone DEV evaluator was introduced AFTER the immutable step16
    # checkpoint. The shared adapter_identity() fingerprints every lane .py file,
    # including DEV-only instruments, so exclude EXACTLY this evaluator from
    # the training checkpoint contract and strictly retain every original hash.
    current_identity = adapter_identity(selected)
    original_files = dict(current_identity["source_files"])
    if original_files.pop("lanes/sprocket/development.py", None) != sha256_file(__file__):
        raise ValueError("Sprocket DEV evaluator source identity missing")
    original_training_adapter = {**current_identity, "source_files": original_files}
    checks = {
        "profile": config.get("profile") == "sprocket",
        "trainer_source": config.get("source_sha") == TRAINING_SOURCE,
        "model": config.get("model") == selected,
        "native_source": config.get("native_source_sha") == NATIVE_SOURCE,
        "corpus_manifest": config.get("corpus_manifest_sha256") == CORPUS_SHA,
        "system": config.get("system") == SYSTEM,
        "compiler": config.get("compiler_sha256") == sha256_file(ROOT / "compiler.py"),
        "trainer": config.get("trainer_sha256") == sha256_file(ROOT / "training" / "train.py"),
        "original_training_adapter": config.get("adapter_contract") == original_training_adapter,
        "checkpoint_step": checkpoint.get("global_step") == 16,
    }
    failures = [name for name, valid in checks.items() if not valid]
    if failures:
        raise ValueError("checkpoint identity differs from frozen SPROCKET 16-step pilot: "
                         + ", ".join(failures))
    groups = development_groups(args.corpus)
    numerics, runtime = configure_cpu(1)
    if numerics != config["numerics"]:
        raise ValueError("portable CPU recipe is not source-matched")
    import torch
    from peft import PeftModel
    torch.manual_seed(568)
    args.output.mkdir(parents=True)
    base, tokenizer = load_base(args.model, selected)
    before = evaluate(base, tokenizer, groups, compiler_module(), args.output,
                      "untuned", args.max_new_tokens, torch)
    tuned = PeftModel.from_pretrained(
        base, args.checkpoint / "adapter", is_trainable=False, local_files_only=True)
    after = evaluate(tuned, tokenizer, groups, compiler_module(), args.output,
                     "step16", args.max_new_tokens, torch)
    write_json(args.output / "comparison.json", {
        "schema": "cybersand.sprocket-development-first-pass/1",
        "model": selected["model_id"], "model_revision": selected["revision"],
        "trainer_source": config["source_sha"], "native_source": NATIVE_SOURCE,
        "corpus_manifest_sha256": CORPUS_SHA,
        "checkpoint_inventory_sha256": sha256_file(args.checkpoint / "inventory.json"),
        "max_new_tokens": args.max_new_tokens, "decoding": "greedy first pass, no repair/retry/fallback",
        "arms": [before, after], "numerics": numerics, "runtime": runtime,
        "primary_target": "Greebles, Pipework, orthographic Architecture",
        "comparison_scope": "six fixed DEVELOPMENT language requests, each at two native host seeds",
        "no_native_execution_claim": True, "final_targets_opened": False,
    })
    print("DEVELOPMENT summary", before, after, flush=True)


if __name__ == "__main__":
    main()
