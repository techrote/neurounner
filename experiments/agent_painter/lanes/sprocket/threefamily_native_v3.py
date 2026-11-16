#!/usr/bin/env python3
"""Native Art research replay for four SPROCKET Greeble/Architecture proposals.

This is a source-built, separately versioned teacher-prototype test, NOT model
inference or shared TRAIN admission. Original FINAL is never accessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "native"))
from compiler import compile_program
from threefamily_v3 import (
    HOST_SEEDS, NATIVE_SOURCE, NATIVE_MANIFEST_SHA, ORIGINAL_TRAIN_SHA,
    ORIGINAL_DEV_SHA, ORIGINAL_COMPILER_SHA, STUDIES, program_for, evaluate,
)
from run import native_cases, execute, write_json
from runtime import ROOT as REPO, digest as file_digest, verify as runtime_verify
from verify import verify as verify_native

SCREEN_SOURCE_SHA = "af673d4497f199d2ce10a6aa7ff1b89dffc23c37"
SCREEN_RUN_ID = 43587356114
SCREEN_ARTIFACT_ID = 14110113977

def require(value, message):
    if not value:
        raise ValueError(message)

def source_guard(source):
    env = os.environ
    require(
        env.get("GITHUB_ACTIONS") == "true"
        and env.get("RUNNER_ENVIRONMENT") == "github-hosted"
        and env.get("GITHUB_REPOSITORY") == "techrote/cybersand"
        and env.get("RUNNER_OS") == "Linux"
        and env.get("GITHUB_REF") == "refs/heads/codex/agent-painter-sprocket-568"
        and env.get("GITHUB_SHA") == source
        and bool(re.fullmatch("[0-9a-f]{40}", source)),
        "execute only from own exact GitHub-hosted research source",
    )

def candidate_cases(corpus: Path, screen_path: Path):
    actual = evaluate(corpus)
    original = json.loads(screen_path.read_text())
    require(original == actual and actual["candidate_compositions"] == 4
            and actual["compiler_seed_views"] == 8
            and actual["shared_train_groups_admitted"] == 0
            and not actual["final_targets_opened"],
            "immutable original source-screen program/split identity differs")
    examples = []
    for identifier, row in STUDIES.items():
        program = program_for(identifier)
        for seed in HOST_SEEDS:
            compiled = compile_program(program, seed=seed)
            require(compiled["proposal_only"] is True
                    and compiled["native_qualified"] is False,
                    "compiler cannot claim native authority")
            examples.append({
                "id": f"sprocket-{identifier}-s{seed}",
                "family": row["family"], "split": "PROPOSAL_ONLY",
                "structural_group": f"proposal/sprocket/threefamily/{identifier}/v3",
                "prompt": row["description"], "seed": seed,
                "program": program, "compiled": compiled,
                "origin": "human-authored-nonmodel-threefamily-research",
            })
    require(len(examples) == 8
            and len({r["structural_group"] for r in examples}) == 4,
            "exact 4 independent designs, 8 seeded cases required")
    return {"examples": examples}, actual

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("preflight", "execute"), required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--original-screen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--godot", type=Path)
    p.add_argument("--runtime-manifest", type=Path)
    args = p.parse_args()
    source_guard(args.source_sha)
    require(not args.output.exists(), "refuse overwrite of existing native proof")
    cases, screen = candidate_cases(args.corpus, args.original_screen)
    args.output.mkdir(parents=True)
    write_json(args.output / "input-cases.json", native_cases(cases))
    write_json(args.output / "candidates.json", cases)
    write_json(args.output / "compiler-screen.json", screen)
    summary = {
        "schema": "cybersand.sprocket-threefamily-v3-native-prospective/1",
        "source_sha": args.source_sha,
        "original_source_screen_sha": SCREEN_SOURCE_SHA,
        "original_source_screen_run": SCREEN_RUN_ID,
        "original_source_screen_artifact": SCREEN_ARTIFACT_ID,
        "original_screen_sha256": file_digest(args.original_screen),
        "native_corpus_source_sha": NATIVE_SOURCE,
        "native_corpus_manifest_sha256": NATIVE_MANIFEST_SHA,
        "original_train_sha256": ORIGINAL_TRAIN_SHA,
        "original_development_sha256": ORIGINAL_DEV_SHA,
        "compiler_sha256": ORIGINAL_COMPILER_SHA,
        "candidate_compositions": 4, "requested_native_views": 8,
        "native_verified": False, "actual_native_positive_views": 0,
        "shared_train_groups_admitted": 0, "model_training": False,
        "model_generated_views": 0, "final_targets_opened": False,
        "reviewer_approved": False,
        "claim": "Human-authored research, not model artwork or TRAIN",
    }
    write_json(args.output / "qualification-summary.json", summary)
    if args.stage == "preflight":
        print(json.dumps({"preflight": "PASS", "candidate_groups": 4,
                          "native_views": 0, "train_groups": 0}, sort_keys=True))
        return

    require(args.godot is not None and args.runtime_manifest is not None,
            "native execution needs source-matched actual Godot build")
    godot = args.godot.resolve()
    custody = runtime_verify(REPO, args.runtime_manifest.resolve(), args.source_sha, godot)
    custody.update({
        "claim": "Source-built prospective Art painting; no model or TRAIN",
        "harness_sha256": file_digest(REPO / "godot/tests/run_agent_painter_foundation.gd"),
        "input_manifest_sha256": file_digest(args.output / "input-cases.json"),
        "candidates_sha256": file_digest(args.output / "candidates.json"),
        "workers": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(args.output / "runtime-custody.json", custody)

    env = os.environ.copy()
    env.update({
        "AP_NATIVE_CASES": str((args.output / "input-cases.json").resolve()),
        "AP_NATIVE_OUTPUT": str(args.output.resolve()),
        "AP_NATIVE_WORKERS": "1",
        "AP_NATIVE_SOURCE_SHA": args.source_sha,
        "XDG_DATA_HOME": str((args.output / "profile/data").resolve()),
        "XDG_CONFIG_HOME": str((args.output / "profile/config").resolve()),
        "XDG_CACHE_HOME": str((args.output / "profile/cache").resolve()),
    })
    imported = execute([str(godot), "--headless", "--path", str(REPO / "godot"),
                        "--editor", "--quit"], env, args.output / "import.log", 600)
    write_json(args.output / "import-result.json", imported)
    require(imported["exit_code"] == 0 and imported["timed_out"] is False,
            "source-matched Godot import failed")
    result = execute([str(godot), "--headless", "--path", str(REPO / "godot"),
                      "--script", "res://tests/run_agent_painter_foundation.gd"],
                     env, args.output / "engine.log", 1800)
    try:
        refreshed = runtime_verify(REPO, args.runtime_manifest.resolve(),
                                   args.source_sha, godot)
        result["runtime_unchanged"] = all(custody.get(k) == v for k,v in refreshed.items())
    except (OSError, ValueError) as error:
        result["runtime_unchanged"] = False
        result["custody_error"] = str(error)
    result["harness_unchanged"] = (
        file_digest(REPO / "godot/tests/run_agent_painter_foundation.gd")
        == custody["harness_sha256"]
    )
    result["input_unchanged"] = (
        file_digest(args.output / "input-cases.json") == custody["input_manifest_sha256"]
        and file_digest(args.output / "candidates.json") == custody["candidates_sha256"]
    )
    write_json(args.output / "process-result.json", result)
    try:
        report = verify_native(args.output)
    except (ValueError, TypeError, KeyError, OSError) as error:
        report = {"passed": False, "failures": [f"Incomplete native verification: {error}"],
                  "cases": []}
    write_json(args.output / "native-verification.json", report)
    positives = [row for row in report["cases"] if row.get("success") is True]
    summary.update({
        "native_verified": bool(report.get("passed") is True and len(positives) == 8),
        "actual_native_positive_views": len(positives),
        "native_verifier_sha256": file_digest(args.output / "native-verification.json"),
        "cases": [
            {"id": row["id"], "seed": row["seed"],
             "family": row["family"], "affected_cells": row["affected_cells"],
             "png_sha256": row["image_sha256"],
             "readback_sha256": row["readback_sha256"],
             "receipt_sha256": row["receipt_sha256"],
             "history_and_reopen_checked": row.get("history_and_reopen_checked", False)}
            for row in positives
        ],
        "failures": report.get("failures", []),
    })
    write_json(args.output / "qualification-summary.json", summary)
    require(summary["native_verified"], "native Art proof not complete; preserve failure artifact")
    print(json.dumps({"status": "native-prospective-only", "views": len(positives),
                      "admitted_TRAIN": 0, "model_generated": 0, "FINAL": False},
                     sort_keys=True))

if __name__ == "__main__":
    main()
