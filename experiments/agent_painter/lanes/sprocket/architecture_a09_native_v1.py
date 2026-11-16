#!/usr/bin/env python3
"""Source-built native Art research for SPROCKET A09, without TRAIN admission.

Consumes a separately qualified source screen and selectively extracted
original visible TRAIN/DEVELOPMENT split. Does not inspect sealed FINAL.
One human-authored composition, two host seeds, zero model inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

PAINTER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER_ROOT))
sys.path.insert(0, str(PAINTER_ROOT / "native"))
from compiler import compile_program
from architecture_revision_v1 import (
    IDENTIFIER, DESCRIPTION, HOST_SEEDS, NATIVE_SOURCE,
    NATIVE_MANIFEST_SHA, evaluate, program_for,
)
from run import native_cases, execute, write_json
from runtime import ROOT as REPO, digest as file_digest, verify as runtime_verify
from verify import verify as verify_native

SCREEN_RUN = 43678612406
SCREEN_SOURCE_SHA = "63375c2263a33bfa3cd35f9d470c46d5dfc6cc26"
SCREEN_ARTIFACT = 14147236676
SCREEN_ZIP_SHA256 = "96f90940c28a80e55c6a17119d15fccbdcb595d084807cb50d2c912bb14668e7"

def must(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)

def source_guard(sha: str) -> None:
    e = os.environ
    must(
        bool(re.fullmatch(r"[0-9a-f]{40}", sha))
        and e.get("GITHUB_ACTIONS") == "true"
        and e.get("RUNNER_ENVIRONMENT") == "github-hosted"
        and e.get("GITHUB_REPOSITORY") == "techrote/cybersand"
        and e.get("RUNNER_OS") == "Linux"
        and e.get("GITHUB_REF") == "refs/heads/codex/agent-painter-sprocket-568"
        and e.get("GITHUB_SHA") == sha,
        "A09 native research must execute exact own hosted HEAD",
    )

def cases_from_source(corpus: Path, original_screen: Path):
    actual = evaluate(corpus)
    saved = json.loads(original_screen.read_text())
    must(saved == actual, "independent A09 original source screen disagrees")
    must(actual["design_count"] == 1
         and actual["compiler_seed_views"] == 2
         and actual["sealed_final_members_accessed"] == 0
         and actual["native_executions"] == 0
         and actual["new_shared_train_groups"] == 0,
         "A09 original source/split/claim identity drift")
    cases = []
    program = program_for()
    for seed in HOST_SEEDS:
        compiled = compile_program(program, seed=seed, material=58,
                                   allowed_materials=(58,),
                                   max_operations=512, max_cell_work=100000)
        must(compiled["structural_preflight"]["passed"]
             and compiled["proposal_only"] and not compiled["native_qualified"],
             "compiler cannot grant native Art authority")
        cases.append({
            "id": f"sprocket-{IDENTIFIER}-s{seed}",
            "family": "architecture",
            "split": "PROPOSAL_ONLY",
            "structural_group": f"proposal/sprocket/architecture/{IDENTIFIER}/v1",
            "prompt": DESCRIPTION,
            "seed": seed,
            "program": program,
            "compiled": compiled,
            "origin": "human-authored-architecture-revision-not-model-output",
        })
    must(len(cases) == 2 and len({x["structural_group"] for x in cases}) == 1,
         "A09 one whole structural group with two seed replays")
    return {"examples": cases}, actual

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
    must(not args.output.exists(), "A09 immutable native evidence path already exists")
    data, screen = cases_from_source(args.corpus, args.original_screen)
    args.output.mkdir(parents=True)
    write_json(args.output / "input-cases.json", native_cases(data))
    write_json(args.output / "candidates.json", data)
    write_json(args.output / "compiler-screen.json", screen)
    summary = {
        "schema": "cybersand.sprocket-a09-positive-native-research/1",
        "status": "SOURCE_ONLY" if args.stage == "preflight" else "NATIVE_NOT_YET_VERIFIED",
        "source_sha": args.source_sha,
        "original_source_screen_sha": SCREEN_SOURCE_SHA,
        "original_source_screen_run": SCREEN_RUN,
        "original_source_screen_artifact": SCREEN_ARTIFACT,
        "original_source_screen_archive_sha256": SCREEN_ZIP_SHA256,
        "original_screen_json_sha256": file_digest(args.original_screen),
        "original_native_corpus_source_sha": NATIVE_SOURCE,
        "original_native_corpus_manifest_sha256": NATIVE_MANIFEST_SHA,
        "candidate_id": IDENTIFIER,
        "whole_designs": 1, "host_seed_views": 2,
        "native_verified": False, "positive_native_views": 0,
        "shared_train_groups_admitted": 0, "model_generated_views": 0,
        "model_training": False, "final_members_accessed": 0,
        "reviewer_approved": False,
        "claim": "Human-authored prospective native research, NOT learned art or TRAIN",
    }
    write_json(args.output / "qualification-summary.json", summary)
    if args.stage == "preflight":
        print(json.dumps({"preflight": "PASS", "candidate": IDENTIFIER,
                          "structures": 1, "native_views": 0, "TRAIN": 0},
                         sort_keys=True))
        return

    must(args.godot is not None and args.runtime_manifest is not None,
         "source-built native requires pinned Godot and native manifest")
    godot = args.godot.resolve()
    custody = runtime_verify(REPO, args.runtime_manifest.resolve(),
                             args.source_sha, godot)
    custody.update({
        "claim": "source-built prospective native Art; no model/FINAL/TRAIN",
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
    must(imported["exit_code"] == 0 and imported["timed_out"] is False,
         "pinned Godot import failed")
    result = execute([str(godot), "--headless", "--path", str(REPO / "godot"),
                      "--script", "res://tests/run_agent_painter_foundation.gd"],
                     env, args.output / "engine.log", 1800)
    try:
        again = runtime_verify(REPO, args.runtime_manifest.resolve(),
                               args.source_sha, godot)
        result["runtime_unchanged"] = all(custody.get(k) == v for k,v in again.items())
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
        report = {"passed": False, "failures": [f"Native verification incomplete: {error}"],
                  "cases": []}
    write_json(args.output / "native-verification.json", report)
    positive = [x for x in report["cases"] if x.get("success") is True]
    summary.update({
        "status": "NATIVE_POSITIVE_REVIEW_ONLY" if report.get("passed") and len(positive)==2 else "NATIVE_FAILED",
        "native_verified": bool(report.get("passed") and len(positive) == 2),
        "positive_native_views": len(positive),
        "verification_sha256": file_digest(args.output / "native-verification.json"),
        "receipts": [{
            "id": x["id"], "seed": x["seed"],
            "affected_cells": x["affected_cells"],
            "readback_sha256": x["readback_sha256"],
            "png_sha256": x["image_sha256"],
            "receipt_sha256": x["receipt_sha256"],
            "history_and_reopen_checked": x.get("history_and_reopen_checked", False),
        } for x in positive],
        "failures": report.get("failures", []),
    })
    write_json(args.output / "qualification-summary.json", summary)
    must(summary["native_verified"],
         "A09 source-built Art receipt/readback failed; retain actual failure artifact")
    print(json.dumps({"native_research": "PASS", "designs": 1,
                      "seed_views": len(positive), "shared_TRAIN": 0,
                      "model_generated": 0, "FINAL": 0}, sort_keys=True))

if __name__ == "__main__":
    main()
