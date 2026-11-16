#!/usr/bin/env python3
"""Native Art verification of six prospective SPROCKET Pipework compositions.

Source-built Godot native receipts/readback/PNG prove a prospective teacher
drawing physically exists; NO shared TRAIN admission, model inference,
optimization, benchmark freeze or sealed FINAL target access occurs here.
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
from proposal_pipework import GRAPHS, HOST_SEEDS, evaluate, program_for
from run import native_cases, execute, write_json
from runtime import ROOT as REPO, digest as file_digest, verify as runtime_verify
from verify import verify as verify_native

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def source_guard(source):
    env = os.environ
    require(env.get("GITHUB_ACTIONS") == "true"
            and env.get("RUNNER_ENVIRONMENT") == "github-hosted"
            and env.get("GITHUB_REPOSITORY") == "techrote/cybersand"
            and env.get("RUNNER_OS") == "Linux"
            and env.get("GITHUB_REF") == "refs/heads/codex/agent-painter-sprocket-568"
            and env.get("GITHUB_SHA") == source
            and bool(re.fullmatch(r"[0-9a-f]{40}", source)),
            "must run exact source on dedicated GitHub-hosted lane")
    
def proposed_cases(corpus):
    screen = evaluate(corpus)
    require(screen["reviewable_structures"] == 6
            and screen["host_seed_views_compiled"] == 12
            and screen["final_targets_opened"] is False
            and all(row["train_eligible"] is False for row in screen["candidates"]),
            "original split/source/compiler screen not qualified")
    examples = []
    for identifier in GRAPHS:
        program = program_for(identifier)
        for seed in HOST_SEEDS:
            compiled = compile_program(program, seed=seed)
            require(compiled["native_qualified"] is False and compiled["proposal_only"] is True,
                    "compiler is proposal-only authority")
            examples.append({
                "id": f"sprocket-{identifier}-s{seed}",
                "family": "pipework", "split": "PROPOSAL_ONLY",
                "structural_group": f"proposal/sprocket/pipework/{identifier}/v0",
                "prompt": GRAPHS[identifier]["description"],
                "seed": seed, "program": program, "compiled": compiled,
                "origin": "human-authored-structural-prototype-not-model-output",
            })
    require(len(examples) == 12 and len({r["structural_group"] for r in examples}) == 6,
            "one composition per group, two host-seed views")
    return {"examples": examples}, screen

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("preflight","execute"),required=True)
    parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-sha",required=True)
    parser.add_argument("--godot",type=Path)
    parser.add_argument("--runtime-manifest",type=Path)
    args=parser.parse_args()
    source_guard(args.source_sha)
    require(not args.output.exists(),"refuse overwrite of native proof")
    cases, screen = proposed_cases(args.corpus)
    args.output.mkdir(parents=True)
    write_json(args.output/"candidates.json",cases)
    write_json(args.output/"input-cases.json",native_cases(cases))
    write_json(args.output/"compiler-screen.json",screen)
    summary = {
        "schema": "cybersand.sprocket-prospective-native-pipework/1",
        "source_sha":args.source_sha, "native_source":"633920373a3d406a2ede61ce4137c324f76be23b",
        "compiler_only": True, "native_verified": False,
        "actual_native_positive_views": 0, "shared_train_groups_admitted":0,
        "development_inference_requests":0, "final_targets_opened":False,
        "model_training":False, "reviewer_approved":False,
        "note":"This is a non-model structural teacher prototype, not a learned painting.",
    }
    write_json(args.output/"qualification-summary.json",summary)
    if args.stage=="preflight":
        print(json.dumps({"preflight":"PASS", "candidate_groups":6,"seed_views":12,
                          "native_verified":False,"train_admitted":False},sort_keys=True))
        return
    require(args.godot is not None and args.runtime_manifest is not None,
            "native stage requires verified current-source Godot build")
    godot=args.godot.resolve()
    custody=runtime_verify(REPO,args.runtime_manifest.resolve(),args.source_sha,godot)
    custody.update({
        "claim":"Actual source-built native Art/readback of prospective nonmodel compositions; no TRAIN admission",
        "harness_sha256":file_digest(REPO/"godot/tests/run_agent_painter_foundation.gd"),
        "input_manifest_sha256":file_digest(args.output/"input-cases.json"),
        "candidates_sha256":file_digest(args.output/"candidates.json"),
        "workers":1, "started_utc":datetime.now(timezone.utc).isoformat(),
    })
    write_json(args.output/"runtime-custody.json",custody)
    env=os.environ.copy()
    env.update({
        "AP_NATIVE_CASES":str((args.output/"input-cases.json").resolve()),
        "AP_NATIVE_OUTPUT":str(args.output.resolve()),
        "AP_NATIVE_WORKERS":"1", "AP_NATIVE_SOURCE_SHA":args.source_sha,
        "XDG_DATA_HOME":str((args.output/"profile/data").resolve()),
        "XDG_CONFIG_HOME":str((args.output/"profile/config").resolve()),
        "XDG_CACHE_HOME":str((args.output/"profile/cache").resolve()),
    })
    imported=execute([str(godot),"--headless","--path",str(REPO/"godot"),"--editor","--quit"],
                     env,args.output/"import.log",600)
    write_json(args.output/"import-result.json",imported)
    require(imported["exit_code"]==0 and imported["timed_out"] is False,"Godot import failed")
    result=execute([str(godot),"--headless","--path",str(REPO/"godot"),
                    "--script","res://tests/run_agent_painter_foundation.gd"],
                   env,args.output/"engine.log",1800)
    try:
        verified_again=runtime_verify(REPO,args.runtime_manifest.resolve(),args.source_sha,godot)
        result["runtime_unchanged"]=all(custody.get(k)==v for k,v in verified_again.items())
    except (OSError,ValueError) as error:
        result["runtime_unchanged"]=False
        result["custody_error"]=str(error)
    result["harness_unchanged"]=file_digest(REPO/"godot/tests/run_agent_painter_foundation.gd")==custody["harness_sha256"]
    result["input_unchanged"]=(file_digest(args.output/"input-cases.json")==custody["input_manifest_sha256"]
                               and file_digest(args.output/"candidates.json")==custody["candidates_sha256"])
    write_json(args.output/"process-result.json",result)
    try:
        report=verify_native(args.output)
    except (ValueError,KeyError,TypeError,OSError) as error:
        report={"passed":False,"failures":["Native independent verification incomplete: "+str(error)],"cases":[]}
    write_json(args.output/"native-verification.json",report)
    successes=[row for row in report["cases"] if row.get("success") is True]
    summary.update({
        "compiler_only":False,
        "native_verified":bool(report.get("passed") is True and len(successes)==12),
        "actual_native_positive_views":len(successes),
        "native_verifier_sha256":file_digest(args.output/"native-verification.json"),
        "native_case_receipts":[
            {"id":x["id"],"seed":x["seed"],"changed":x["affected_cells"],
             "png_sha256":x["image_sha256"],"readback_sha256":x["readback_sha256"],
             "receipt_sha256":x["receipt_sha256"]} for x in successes],
        "failures":report.get("failures",[]),
        "claim":"Physical painting by a hand-authored prototype, NOT a trained model or shared TRAIN member.",
    })
    write_json(args.output/"qualification-summary.json",summary)
    require(summary["native_verified"],"native source-built positive gate failed; artifact retained")
    print(json.dumps({"status":"native-prospective-positive-not-train-admitted",
                      "groups":6,"seed_views":len(successes),
                      "shared_train_groups_admitted":0,"final_targets_opened":False},sort_keys=True))

if __name__=="__main__":
    main()
