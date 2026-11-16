#!/usr/bin/env python3
"""Source-built native Art research for LFM contrastive v1; never shared TRAIN admission.

All six actual host-seed views derive from three original hand-authored programs,
not model completions. Positive native evidence is NOT teacher curriculum
selection; independent reviewer and shepherd retain that authority.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(LANE))
from contrastive_preflight import screen, SEEDS, FILE, MANIFEST, sha

PAINTER = LANE.parents[1]
sys.path.insert(0, str(PAINTER))
from compiler import compile_program

sys.path.insert(0, str(PAINTER / "native"))
from run import execute, native_cases, write_json
from runtime import ROOT, digest, verify as verify_runtime
from verify import verify as verify_native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "godot", "runtime-manifest", "source-sha", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--proposals", default=str(FILE))
    parser.add_argument("--disposition", required=True)
    args = parser.parse_args()
    corpus = Path(args.corpus)
    proposals = Path(args.proposals)
    godot = Path(args.godot).resolve()
    runtime_manifest = Path(args.runtime_manifest).resolve()
    out = Path(args.output).resolve()
    if out.exists() or len(args.source_sha) != 40:
        parser.error("full source identity and fresh, never-overwritten output required")
    out.mkdir(parents=True, exist_ok=False)

    screened = screen(corpus, proposals)
    if (screened["status"] !=
            "COMPILER_SCREEN_ONLY_PENDING_SHEPHERD_REVIEW_NATIVE_AND_TRAIN_ADMISSION"
            or screened["original_manifest_sha256"] != MANIFEST
            or len(screened["proposals"]) != 3):
        raise ValueError("original TRAIN/DEVELOPMENT source screen required before native")
    write_json(out / "pre-native-source-screen.json", screened)

    document = json.loads(proposals.read_text(encoding="utf-8"))
    disposition_path = Path(args.disposition)
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    if (disposition.get("schema") != "cybersand.lfm25_vl.contrastive-review-disposition/1"
            or disposition.get("source_screen_producer_sha")
                != "44721f3b116cb0eedfea36965be4c20274d4a5e3"
            or disposition.get("native_research_producer_sha")
                != "abef683ceaa0fb2782e2d4b65db2c4c3c475e52d"
            or disposition.get("not_common_train") is not True
            or disposition.get("final_targets_opened") is not False):
        raise ValueError("exact previous failed native source and reviewer disposition required")
    status = {x["id"]: x for x in disposition["entries"]}
    if set(status) != {x["id"] for x in document["proposals"]}:
        raise ValueError("reviewer disposition missing or adding original candidates")
    quarantined = [key for key, value in status.items()
                   if value["disposition"].startswith("QUARANTINED_")]
    if quarantined != ["lfm-c3-gantry-room-and-underdeck-stairs"]:
        raise ValueError("expected explicit C3 P2 quarantine, no waiver")
    active = [item for item in document["proposals"] if item["id"] not in quarantined]
    if len(active) != 2 or {x["family"] for x in active} != {"greebles", "pipework"}:
        raise ValueError("only Greebles and Pipework are cleared for native research retry")
    cases = []
    for candidate in active:
        p = candidate["program"]
        for seed in SEEDS:
            compiled = compile_program(p, seed=seed, focus=(0, 0, 96, 96),
                                       material=58, allowed_materials=(58,),
                                       max_operations=512, max_cell_work=100000)
            cases.append({
                "id": candidate["id"] + "-s" + str(seed),
                "family": candidate["family"], "seed": seed,
                "split": "UNADMITTED_RESEARCH_PROPOSAL", "proposal_group_id": candidate["id"],
                "program": p, "compiled": compiled, "training_approved": False,
            })
    if len(cases) != 4:
        raise ValueError("expected exactly two non-quarantined designs and two host seeds each")

    candidate_document = {
        "schema_version": 1, "purpose": "unadmitted-contrastive-native-research",
        "examples": cases, "final_targets_opened": False,
    }
    write_json(out / "candidates.json", candidate_document)
    write_json(out / "input-cases.json", native_cases(candidate_document))
    harness = ROOT / "godot/tests/run_agent_painter_foundation.gd"
    custody = verify_runtime(ROOT, runtime_manifest, args.source_sha, godot)
    custody.update({
        "claim": "Source-built native Art teacher research ONLY; NOT LFM model-generated paints, native-approved shared TRAIN, or accepted vN",
        "harness_sha256": digest(harness),
        "proposal_sha256": sha(proposals),
        "original_manifest_sha256": MANIFEST,
        "original_split_screen_sha256": digest(out / "pre-native-source-screen.json"),
        "review_disposition_sha256": sha(disposition_path),
        "input_manifest_sha256": digest(out / "input-cases.json"),
        "candidates_sha256": digest(out / "candidates.json"),
        "workers": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    write_json(out / "runtime-custody.json", custody)
    env = os.environ.copy()
    env.update({
        "AP_NATIVE_CASES": str((out / "input-cases.json").resolve()),
        "AP_NATIVE_OUTPUT": str(out),
        "AP_NATIVE_WORKERS": "1",
        "AP_NATIVE_SOURCE_SHA": args.source_sha,
        "XDG_DATA_HOME": str(out / "profile/data"),
        "XDG_CONFIG_HOME": str(out / "profile/config"),
        "XDG_CACHE_HOME": str(out / "profile/cache"),
    })
    imported = execute(
        [str(godot), "--headless", "--path", str(ROOT / "godot"), "--editor", "--quit"],
        env, out / "import.log", 600)
    write_json(out / "import-result.json", imported)
    if imported["exit_code"] == 0:
        result = execute(
            [str(godot), "--headless", "--path", str(ROOT / "godot"),
             "--script", "res://tests/run_agent_painter_foundation.gd"],
            env, out / "engine.log", 1800)
    else:
        result = {**imported, "stage": "Godot import"}
    try:
        after = verify_runtime(ROOT, runtime_manifest, args.source_sha, godot)
        result["runtime_unchanged"] = all(custody.get(k) == v for k, v in after.items())
    except (ValueError, OSError) as error:
        result["runtime_unchanged"] = False
        result["runtime_error"] = str(error)
    result["harness_unchanged"] = digest(harness) == custody["harness_sha256"]
    result["input_unchanged"] = all((
        digest(out / "input-cases.json") == custody["input_manifest_sha256"],
        digest(out / "candidates.json") == custody["candidates_sha256"],
        digest(out / "pre-native-source-screen.json") == custody["original_split_screen_sha256"],
        sha(proposals) == custody["proposal_sha256"],
        sha(disposition_path) == custody["review_disposition_sha256"],
    ))
    write_json(out / "process-result.json", result)
    try:
        verified = verify_native(out)
    except (ValueError, KeyError, OSError, TypeError) as error:
        verified = {"passed": False, "failures": [str(error)], "cases": []}
    write_json(out / "native-verification.json", verified)
    good = [item for item in verified.get("cases", []) if item.get("success") is True]
    summary = {
        "actual_native_source_sha": args.source_sha,
        "original_native_corpus_source_sha": screened["original_native_source_sha"],
        "original_manifest_sha256": MANIFEST,
        "proposal_sha256": sha(proposals),
        "original_split_screen_sha256": custody["original_split_screen_sha256"],
        "native_verifier_passed": verified.get("passed") is True,
        "runtime_unchanged": result["runtime_unchanged"],
        "harness_unchanged": result["harness_unchanged"],
        "input_unchanged": result["input_unchanged"],
        "independent_hand_authored_compositions": 2,
        "host_seed_variants": 4,
        "quarantined_candidate_ids": quarantined,
        "review_disposition_sha256": custody["review_disposition_sha256"],
        "actual_native_verified_views": len(good),
        "verified_case_ids": [r["id"] for r in good],
        "verified_affected_material_cells": [r.get("affected_cells") for r in good],
        "new_common_train_admissions": 0,
        "new_model_completions": 0,
        "optimizer_updates": 0,
        "final_targets_opened": False,
        "review_and_visual_quality": "PENDING",
        "failure_reasons": verified.get("failures", []),
    }
    write_json(out / "lfm-contrastive-native-summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not all((
        summary["native_verifier_passed"], len(good) == 4,
        summary["runtime_unchanged"], summary["harness_unchanged"],
        summary["input_unchanged"],
    )):
        raise SystemExit("Native research failed or incomplete; retain evidence, no TRAIN admission")


if __name__ == "__main__":
    main()
