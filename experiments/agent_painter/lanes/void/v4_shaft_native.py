#!/usr/bin/env python3
"""Real source-built Art research replay of VOID v4 A02 vertical-shaft hypothesis.

Outputs are native-positive research evidence IF the independent native
verifier passes. This never calls shared corpus admission, never produces
TRAIN JSONL, never trains a model, and never accesses sealed FINAL data.
Shepherd/reviewer alone decide if proposals enter the common curriculum.
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
from v4_shaft_probe import verify as screen, MANIFEST

PAINTER = LANE.parents[1]
sys.path.insert(0, str(PAINTER))
from compiler import compile_program

sys.path.insert(0, str(PAINTER / "native"))
from run import execute, native_cases, write_json
from runtime import ROOT, digest, verify as verify_runtime
from verify import verify as verify_native

SEEDS = (104729, 130363)


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("proposals", "previous", "corpus", "godot", "runtime-manifest", "source-sha", "output"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    proposals, corpus, godot, stamp, out = (
        Path(a.proposals), Path(a.corpus), Path(a.godot).resolve(),
        Path(a.runtime_manifest).resolve(), Path(a.output).resolve())
    if out.exists() or not a.source_sha or len(a.source_sha) != 40:
        p.error("new output directory and full source SHA required")
    out.mkdir(parents=True, exist_ok=False)
    # Fail before launching native if original immutable TRAIN/DEV metadata,
    # proposal topology, or both compiler seed views no longer match.
    screened = screen(corpus, proposals, Path(a.previous))
    write_json(out / "pre-native-source-screen.json", screened)
    if screened["status"] != "SOURCE_ONLY_PASS" or screened["seed_preflights"] != 2:
        raise ValueError("strict frozen reference and graph preflight required")
    reference_manifest_sha256 = screened["original_manifest_sha256"]
    if reference_manifest_sha256 != MANIFEST:
        raise ValueError("native refused missing/foreign original manifest digest")
    doc = json.loads(proposals.read_text(encoding="utf-8"))
    cases = []
    for entry in doc["candidates"]:
        if entry["training_admitted"] is not False or entry["native_qualified"] is not False:
            raise ValueError("research proposal already admits TRAIN/native claim")
        for seed in SEEDS:
            program = entry["program"]
            compiled = compile_program(program, seed=seed, focus=(0,0,96,96),
                                       material=58, allowed_materials=(58,),
                                       max_operations=512, max_cell_work=100000)
            cases.append({"id": entry["id"]+"-s"+str(seed), "family": program["family"],
                          "seed": seed, "split": "UNADMITTED_TRAIN_PROPOSAL",
                          "proposal_group_id": entry["id"], "program": program,
                          "compiled": compiled, "training_approved": False})
    if len(cases) != 2:
        raise ValueError("exactly one unadmitted A02 composition/two host seeds required")
    candidate_doc = {"schema_version":1, "purpose":"native-research-unadmitted",
                     "examples":cases, "sealed_final_opened":False}
    write_json(out / "candidates.json", candidate_doc)
    write_json(out / "input-cases.json", native_cases(candidate_doc))
    harness = ROOT / "godot/tests/run_agent_painter_foundation.gd"
    custody = verify_runtime(ROOT, stamp, a.source_sha, godot)
    custody.update({"claim":"Real native Art A02 one-composition/two-seed research; not model inference and no TRAIN",
                    "harness_sha256": digest(harness),
                    "input_manifest_sha256": digest(out / "input-cases.json"),
                    "candidates_sha256": digest(out / "candidates.json"),
                    "proposal_sha256": digest(proposals),
                    "source_screen_sha256": digest(out / "pre-native-source-screen.json"),
                    "workers":1, "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    write_json(out / "runtime-custody.json", custody)
    env = os.environ.copy()
    env.update(AP_NATIVE_CASES=str((out / "input-cases.json").resolve()),
               AP_NATIVE_OUTPUT=str(out), AP_NATIVE_WORKERS="1", AP_NATIVE_SOURCE_SHA=a.source_sha,
               XDG_DATA_HOME=str(out / "profile/data"), XDG_CONFIG_HOME=str(out / "profile/config"),
               XDG_CACHE_HOME=str(out / "profile/cache"))
    imported = execute([str(godot),"--headless","--path",str(ROOT / "godot"),
                        "--editor","--quit"],env,out / "import.log",600)
    write_json(out / "import-result.json", imported)
    if imported["exit_code"] == 0:
        process = execute([str(godot),"--headless","--path",str(ROOT / "godot"),
                           "--script","res://tests/run_agent_painter_foundation.gd"],
                          env,out / "engine.log",1800)
    else:
        process = {**imported, "stage":"Godot import"}
    try:
        new_custody = verify_runtime(ROOT, stamp, a.source_sha, godot)
        process["runtime_unchanged"] = all(custody.get(k)==v for k,v in new_custody.items())
    except (ValueError,OSError) as exc:
        process["runtime_unchanged"] = False
        process["runtime_error"] = str(exc)
    process["harness_unchanged"] = digest(harness)==custody["harness_sha256"]
    process["input_unchanged"] = (
        digest(out / "input-cases.json")==custody["input_manifest_sha256"] and
        digest(out / "candidates.json")==custody["candidates_sha256"] and
        digest(proposals)==custody["proposal_sha256"] and
        digest(out / "pre-native-source-screen.json")==custody["source_screen_sha256"])
    write_json(out / "process-result.json",process)
    try:
        report = verify_native(out)
    except (ValueError,KeyError,OSError,TypeError) as exc:
        report = {"passed":False,"failures":["Native verifier incomplete: "+str(exc)],"cases":[]}
    write_json(out / "native-verification.json",report)
    completed = [r for r in report.get("cases",[]) if r.get("success") is True]
    outcome = {"native_source_sha":a.source_sha,
               "proposal_sha256":digest(proposals), "reference_manifest_sha256":reference_manifest_sha256,
               "quarantined_previous_id":screened["quarantined_old_architecture"],
               "review_status":screened["review_status"],
               "original_source_screen_signature":screened["signature_sha256"],
               "native_verifier_passed":report.get("passed") is True,
               "independent_source_compilations":2,
               "distinct_structural_proposals":1, "nonindependent_style_variants":0, "actual_verified_native_views":len(completed),
               "admitted_common_train_groups":0, "optimizer_steps":0, "sealed_final_opened":False,
               "native_case_ids":[r["id"] for r in completed],
               "verifier_failures":report.get("failures",[]),
               "readback_original_pngs_in_evidence":len(completed)==2,
               "next_gate":"independent reviewer visual/split approval and shepherd-owned common admission"}
    write_json(out / "void-v4-shaft-native-summary.json",outcome)
    print(json.dumps(outcome,sort_keys=True),flush=True)
    if not report.get("passed") or len(completed)!=2:
        raise SystemExit("Research native prototype FAILED/partial; preserved receipts; no admission")


if __name__ == "__main__":
    main()
