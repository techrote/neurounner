#!/usr/bin/env python3
"""SMOL G06/P01/A01 source-built prospective native Art research.

Reviewer approved only bounded native research in issue #568 comment 6877658992.
No TRAIN admission, FINAL access, model evaluation or training occurs here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

LANE = Path(__file__).resolve().parent
PAINTER = LANE.parents[1]
sys.path.insert(0, str(LANE))
sys.path.insert(0, str(PAINTER))
sys.path.insert(0, str(PAINTER / "native"))

from curriculum_preflight import run as source_screen, SEEDS, MANIFEST, SOURCE
from compiler import compile_program
from run import execute, native_cases, write_json
from runtime import ROOT, digest, verify as verify_runtime
from verify import verify as verify_native

PROPOSALS = LANE / "curriculum_proposals_v2.json"
EXPECTED = {
    "smol_g06_stepped_absence": "greebles",
    "smol_p01_nested_branch_tree": "pipework",
    "smol_a01_unequal_alcove_hall": "architecture",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def host_guard(source: str) -> None:
    e = os.environ
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", source))
        and e.get("GITHUB_ACTIONS") == "true"
        and e.get("RUNNER_ENVIRONMENT") == "github-hosted"
        and e.get("GITHUB_REPOSITORY") == "techrote/cybersand"
        and e.get("GITHUB_REF") == "refs/heads/codex/agent-painter-smol-568"
        and e.get("GITHUB_SHA") == source
        and e.get("RUNNER_OS") == "Linux",
        "SMOL native research requires source-identical GitHub-hosted Linux job on own branch",
    )


def prepare(corpus: Path, output: Path) -> tuple[dict, dict]:
    # Reads original TRAIN/DEVELOPMENT only; FINAL metadata stays sealed.
    screened = source_screen(corpus, PROPOSALS)
    require(screened["status"].startswith("COMPILER_PREFLIGHT_ONLY")
            and screened["admitted_teacher_rows"] == 0
            and screened["native_execution"] is False
            and screened["final_targets_opened"] is False,
            "source or holdout screening refused")
    doc = json.loads(PROPOSALS.read_text(encoding="utf-8"))
    require(doc["fixed_seeds"] == list(SEEDS) and len(SEEDS) == 2,
            "fixed two-seed source contract changed")
    entries = doc["proposals"]
    require({x["id"]: x["family"] for x in entries} == EXPECTED
            and len(entries) == 3, "reviewer selected exactly G06/P01/A01")
    cases = []
    for entry in entries:
        for seed in SEEDS:
            program = entry["program"]
            compiled = compile_program(program, seed=seed)
            require(compiled["proposal_only"] is True
                    and compiled["native_qualified"] is False
                    and compiled["structural_preflight"]["passed"] is True,
                    "source compiler is not a native authorization")
            cases.append({
                "id": entry["id"] + "-s" + str(seed),
                "family": entry["family"],
                "seed": seed, "split": "PROPOSAL_ONLY_UNADMITTED",
                "structural_group": "research/smol/v2/" + entry["id"],
                "prompt": entry["request"], "program": program,
                "compiled": compiled,
                "origin": "human-authored-prospective-native-research",
                "training_approved": False,
            })
    require(len(cases) == 6 and len({x["structural_group"] for x in cases}) == 3,
            "two seed views are NOT independent designs")
    document = {"schema_version": 1, "purpose": "research-only-unadmitted",
                "examples": cases, "final_targets_opened": False}
    write_json(output / "candidates.json", document)
    write_json(output / "input-cases.json", native_cases(document))
    write_json(output / "source-screen.json", screened)
    write_json(output / "native-research-request.json", {
        "status": "PROSPECTIVE_NOT_ADMITTED",
        "source633_native_manifest_sha256": MANIFEST,
        "source633_sha": SOURCE,
        "proposal_sha256": digest(PROPOSALS),
        "compiler_sha256": screened["compiler_sha256"],
        "host_seeds": list(SEEDS),
        "structural_groups": list(EXPECTED),
        "requested_native_views": 6, "training_steps": 0,
        "model_generated_views": 0, "shared_train_admissions": 0,
        "final_targets_opened": False,
        "reviewer_disposition": "APPROVE_FOR_NATIVE_RESEARCH_ONLY",
        "review_comment": "https://github.com/techrote/cybersand/issues/568#issuecomment-6877658992",
    })
    return document, screened


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("preflight", "execute"), required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--godot", type=Path)
    p.add_argument("--runtime-manifest", type=Path)
    a = p.parse_args()
    host_guard(a.source_sha)
    output = a.output.resolve()
    require(not output.exists(), "research output already exists; refuse overwrite")
    output.mkdir(parents=True, exist_ok=False)
    document, screen = prepare(a.corpus.resolve(), output)
    if a.stage == "preflight":
        print(json.dumps({"status": "SOURCE_ONLY_PASS", "native_views": 0,
                          "candidate_structures": 3, "final_opened": False}, sort_keys=True))
        return

    require(a.godot is not None and a.runtime_manifest is not None,
            "native execution requires source-built Godot, World and stamp")
    godot = a.godot.resolve()
    stamp = a.runtime_manifest.resolve()
    harness = ROOT / "godot/tests/run_agent_painter_foundation.gd"
    custody = verify_runtime(ROOT, stamp, a.source_sha, godot)
    custody.update({
        "harness_sha256": digest(harness),
        "input_manifest_sha256": digest(output / "input-cases.json"),
        "candidates_sha256": digest(output / "candidates.json"),
        "proposal_sha256": digest(PROPOSALS),
        "source_screen_sha256": digest(output / "source-screen.json"),
        "claim": "Source-built native Art research, never TRAIN or model inference",
        "workers": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(output / "runtime-custody.json", custody)
    env = os.environ.copy()
    env.update({
        "AP_NATIVE_CASES": str(output / "input-cases.json"),
        "AP_NATIVE_OUTPUT": str(output),
        "AP_NATIVE_WORKERS": "1",
        "AP_NATIVE_SOURCE_SHA": a.source_sha,
        "XDG_DATA_HOME": str(output / "profile/data"),
        "XDG_CONFIG_HOME": str(output / "profile/config"),
        "XDG_CACHE_HOME": str(output / "profile/cache"),
    })
    imported = execute([str(godot), "--headless", "--path", str(ROOT / "godot"),
                        "--editor", "--quit"], env, output / "import.log", 600)
    write_json(output / "import-result.json", imported)
    if imported["exit_code"] == 0 and not imported["timed_out"]:
        process = execute([str(godot), "--headless", "--path", str(ROOT / "godot"),
                           "--script", "res://tests/run_agent_painter_foundation.gd"],
                          env, output / "engine.log", 1800)
    else:
        process = {**imported, "stage": "godot-import"}
    try:
        after = verify_runtime(ROOT, stamp, a.source_sha, godot)
        process["runtime_unchanged"] = all(custody.get(k) == v for k, v in after.items())
    except (ValueError, OSError) as exc:
        process["runtime_unchanged"] = False
        process["custody_error"] = str(exc)
    process["harness_unchanged"] = digest(harness) == custody["harness_sha256"]
    process["input_unchanged"] = (
        digest(output / "input-cases.json") == custody["input_manifest_sha256"]
        and digest(output / "candidates.json") == custody["candidates_sha256"]
        and digest(PROPOSALS) == custody["proposal_sha256"]
        and digest(output / "source-screen.json") == custody["source_screen_sha256"]
    )
    write_json(output / "process-result.json", process)
    try:
        report = verify_native(output)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        report = {"passed": False,
                  "failures": ["Native verification incomplete: " + str(exc)],
                  "cases": [], "global_gates": {}}
    write_json(output / "native-verification.json", report)
    positives = [x for x in report.get("cases", []) if x.get("success") is True]
    ok = (report.get("passed") is True and len(positives) == 6
          and {x["family"] for x in positives} == set(EXPECTED.values())
          and len(report.get("global_gates", {})) >= 16
          and all(report["global_gates"].values()))
    result = {
        "schema": "cybersand.agent-painter.smol-v2-native-research/1",
        "native_source_sha": a.source_sha,
        "original_corpus_source_sha": SOURCE,
        "original_corpus_manifest_sha256": MANIFEST,
        "program_proposals_sha256": digest(PROPOSALS),
        "source_screen_sha256": digest(output / "source-screen.json"),
        "native_verifier_sha256": digest(output / "native-verification.json"),
        "native_passed": ok, "verified_native_views": len(positives),
        "distinct_structural_groups": 3, "host_seeds": list(SEEDS),
        "total_material_cells": sum(x.get("filled_cells", 0) for x in positives),
        "cases": [
            {"id": x["id"], "family": x["family"], "seed": x["seed"],
             "filled_cells": x["filled_cells"],
             "png_sha256": x["image_sha256"],
             "readback_sha256": x["readback_sha256"],
             "receipt_sha256": x["receipt_sha256"],
             "history_and_reopen_checked": x.get("history_and_reopen_checked", False)}
            for x in positives
        ],
        "failures": report.get("failures", []),
        "admitted_train_groups": 0, "model_outputs": 0, "optimizer_steps": 0,
        "final_targets_opened": False,
        "next_gate": "Independent reviewer original PNG/material inspection and shepherd shared split/admission decision",
    }
    write_json(output / "smol-v2-native-summary.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    require(ok, "native research incomplete/failed; do not admit, preserve partial artifact")


if __name__ == "__main__":
    main()
