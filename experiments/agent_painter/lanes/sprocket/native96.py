#!/usr/bin/env python3
"""Source-locked native replay of Sprocket's ONLY raw step96 accepted program.

Only the original six DEVELOPMENT language completions and native-admitted
DEVELOPMENT rows are read. FINAL and optimizer training are not accessed.
This is a measured negative semantic example, never a useful-painter freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "native"))
from compiler import compile_program, parse_program
from runtime import ROOT as REPO, digest as file_digest, verify as runtime_verify
from run import execute, native_cases, write_json
from verify import verify as verify_native

EVAL_SHA = "22c7870990569e013ec1fe71308940d95e613496"
TRAIN_SHA = "25b24bf80c593b115321387a2a0374dd5ff02548"
NATIVE_SHA = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
MODEL_SHA = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
SEEDS = (104729, 130363)
GROUP = "architecture/stacked_rooms_and_shaft/v1"


def require(test, reason):
    if not test:
        raise ValueError(reason)


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_guard(source):
    env = os.environ
    require(env.get("GITHUB_ACTIONS") == "true" and env.get("RUNNER_ENVIRONMENT") == "github-hosted"
            and env.get("GITHUB_REPOSITORY") == "techrote/cybersand"
            and env.get("RUNNER_OS") == "Linux"
            and env.get("GITHUB_REF") == "refs/heads/codex/agent-painter-sprocket-568"
            and env.get("GITHUB_SHA") == source
            and re.fullmatch(r"[a-f0-9]{40}", source),
            "source-bound GitHub-hosted Sprocket branch required")


def prepare(evaluation, corpus):
    comparison = json_file(evaluation / "comparison.json")
    require(comparison["schema"] == "cybersand.sprocket-development-first-pass/2"
            and comparison["evaluation_source"] == EVAL_SHA
            and comparison["trainer_source"] == TRAIN_SHA
            and comparison["trainer_actions_run"] == 43529482591
            and comparison["checkpoint_step"] == comparison["actual_optimizer_steps"] == 96
            and comparison["checkpoint_readback_verified"] is True
            and comparison["model"] == "kandivault/sprocket-500m"
            and comparison["revision"] == MODEL_SHA
            and comparison["native_source"] == NATIVE_SHA
            and comparison["corpus_manifest_sha256"] == CORPUS_SHA
            and comparison["native_execution"] is False
            and comparison["native_confirmed_paintings"] == 0
            and comparison["final_targets_opened"] is False,
            "immutable evaluator/model/native identity mismatch")
    arms = {x["arm"]: x for x in comparison["arms"]}
    require(set(arms) == {"untuned", "step96"}
            and arms["untuned"]["independent_language_requests"] == 6
            and arms["untuned"]["compiler_seed_accepted"] == 0
            and arms["step96"]["independent_language_requests"] == 6
            and arms["step96"]["schema_first_pass_accepted"] == 1
            and arms["step96"]["compiler_seed_accepted"] == 2
            and all(x["native_execution_attempts"] == 0 and x["final_targets_opened"] is False
                    for x in arms.values()),
            "evaluator acceptance counts differ from original first pass")
    all_raw = json_file(evaluation / "step96-raw-development.json")
    require(len(all_raw) == 6 and len({x["structural_group"] for x in all_raw}) == 6,
            "source DEVELOPMENT generations not six independent requests")
    accepted = [x for x in all_raw if x["raw"]["first_pass_accepted"]]
    require(len(accepted) == 1 and accepted[0]["structural_group"] == GROUP
            and accepted[0]["family"] == "architecture"
            and accepted[0]["raw"]["eos_terminated"] is True
            and accepted[0]["raw"]["program_valid"] is True
            and accepted[0]["raw"]["token_ids"] == accepted[0]["output_token_ids"]
            and accepted[0]["admitted_seed_programs"] == 2,
            "unexpected accepted model output; refuse replay")
    manifest_file = corpus / "admitted-corpus.json"
    require(file_digest(manifest_file) == CORPUS_SHA, "native corpus manifest hash mismatch")
    manifest = json_file(manifest_file)
    require(manifest.get("source_sha") == NATIVE_SHA and manifest.get("state") == "native-admitted"
            and manifest.get("compiler_sha256") == file_digest(ROOT / "compiler.py"),
            "native corpus source/compiler mismatch")
    entry = manifest["split_files"]["DEVELOPMENT"]
    dev_file = corpus / entry["path"]
    require(file_digest(dev_file) == entry["sha256"] and entry["count"] == 12,
            "DEVELOPMENT bytes/count differ from immutable admission")
    rows = [json.loads(s) for s in dev_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    require(len(rows) == 12 and all(r["split"] == "DEVELOPMENT"
            and r["native_admission"]["source_sha"] == NATIVE_SHA
            and r["native_admission"]["verified"] is True for r in rows),
            "DEVELOPMENT rows are not originally native admitted")
    selected = [r for r in rows if r["structural_group"] == GROUP]
    require(len(selected) == 2 and {r["seed"] for r in selected} == set(SEEDS)
            and all(r["prompt"] == accepted[0]["prompt"] for r in selected),
            "model request differs from original DEVELOPMENT prompt")
    program = parse_program(accepted[0]["raw"]["raw_text"])
    require(program["family"] == "architecture"
            and len(program["components"]) == 2
            and all(c.get("kind") == "room" for c in program["components"]),
            "unexpected candidate structure; no silent substitution")
    # Only a diagnostic of the fixed prompt's explicit vertical-shaft requirement.
    rooms = program["components"]
    horizontally_aligned = rooms[0]["box"][1] == rooms[1]["box"][1]
    vertical_opening = any(o["side"] in ("top", "bottom")
                           for c in rooms for o in c["openings"])
    semantic_ok = not horizontally_aligned and vertical_opening
    examples = []
    seed_views = {x["seed"]: x for x in accepted[0]["seed_results"]}
    require(set(seed_views) == set(SEEDS), "original host seed views missing")
    for r in sorted(selected, key=lambda x: x["seed"]):
        seed = r["seed"]
        original = seed_views[seed]
        compiled = compile_program(program, seed=seed)
        require(original["row_id"] == r["id"]
                and original["first_pass_compiler_accepted"] is True
                and original["native_execution"] is False
                and original["native_acknowledged_cells"] is None
                and original["operations"] == compiled["operations"]
                and original["operations_sha256"] == compiled["operations_sha256"]
                and original["admitted_teacher_sha256"] == r["native_admission"]["program_sha256"],
                "original model proposal/corpus/compiled operations differ")
        examples.append({
            "id": "sprocket-step96-stacked-shaft-s" + str(seed),
            "family": "architecture", "split": "DEVELOPMENT", "seed": seed,
            "structural_group": GROUP, "prompt": r["prompt"],
            "program": program, "compiled": compiled,
            "original_language_output_sha256": hashlib.sha256(
                accepted[0]["raw"]["raw_text"].encode("utf-8")).hexdigest(),
            "origin": "unmodified-step96-model-output-not-teacher",
        })
    assessment = {
        "schema": "cybersand.agent-painter.sprocket-step96-native-assessment/1",
        "evaluation_source": EVAL_SHA, "training_source": TRAIN_SHA,
        "native_teacher_source": NATIVE_SHA,
        "program": program, "request": accepted[0]["prompt"],
        "accepted_unique_requests": 1, "accepted_host_seed_views": 2,
        "semantic_success": semantic_ok,
        "semantic_failure": None if semantic_ok else (
            "Requested stacked rooms, shared floor shaft and supported lower platform; "
            "model drew adjacent rooms with left/right openings and no shaft/platform"),
        "claim": "Compiler acceptance is not requested-layout success; native execution pending.",
        "final_targets_opened": False,
    }
    return {"examples": examples}, assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "execute"), required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--godot", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    args = parser.parse_args()
    source_guard(args.source_sha)
    require(not args.output.exists(), "refuse to overwrite existing run evidence")
    candidates, assessment = prepare(args.evaluation, args.corpus)
    args.output.mkdir(parents=True)
    write_json(args.output / "candidates.json", candidates)
    write_json(args.output / "input-cases.json", native_cases(candidates))
    assessment["replay_source"] = args.source_sha
    assessment["native_confirmed_paintings"] = 0
    assessment["native_execution"] = False
    write_json(args.output / "semantic-assessment.json", assessment)
    if args.stage == "preflight":
        print(json.dumps({"preflight": "PASS", "source": args.source_sha,
                          "original_valid_requests": 1, "host_seed_views": 2,
                          "semantic_success": assessment["semantic_success"]}, sort_keys=True))
        return
    require(args.godot is not None and args.runtime_manifest is not None,
            "source-built Godot/runtime evidence required")
    godot = args.godot.resolve()
    custody = runtime_verify(REPO, args.runtime_manifest.resolve(), args.source_sha, godot)
    custody.update({
        "claim": "Actual source-built native Art/readback, not model semantic success or framebuffer",
        "harness_sha256": file_digest(REPO / "godot/tests/run_agent_painter_foundation.gd"),
        "input_manifest_sha256": file_digest(args.output / "input-cases.json"),
        "candidates_sha256": file_digest(args.output / "candidates.json"),
        "workers": 1,
        "evaluation_source": EVAL_SHA,
        "started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    })
    write_json(args.output / "runtime-custody.json", custody)
    env = os.environ.copy()
    env.update({
        "AP_NATIVE_CASES": str((args.output / "input-cases.json").resolve()),
        "AP_NATIVE_OUTPUT": str(args.output.resolve()),
        "AP_NATIVE_WORKERS": "1", "AP_NATIVE_SOURCE_SHA": args.source_sha,
        "XDG_DATA_HOME": str((args.output / "profile/data").resolve()),
        "XDG_CONFIG_HOME": str((args.output / "profile/config").resolve()),
        "XDG_CACHE_HOME": str((args.output / "profile/cache").resolve()),
    })
    imported = execute([str(godot), "--headless", "--path", str(REPO / "godot"),
                        "--editor", "--quit"], env, args.output / "import.log", 600)
    write_json(args.output / "import-result.json", imported)
    require(imported["exit_code"] == 0 and not imported["timed_out"], "Godot import failed")
    result = execute([str(godot), "--headless", "--path", str(REPO / "godot"),
                      "--script", "res://tests/run_agent_painter_foundation.gd"],
                     env, args.output / "engine.log", 1800)
    verified_again = runtime_verify(REPO, args.runtime_manifest.resolve(), args.source_sha, godot)
    result["runtime_unchanged"] = all(custody.get(k) == v for k, v in verified_again.items())
    result["harness_unchanged"] = file_digest(REPO / "godot/tests/run_agent_painter_foundation.gd") == custody["harness_sha256"]
    result["input_unchanged"] = (file_digest(args.output / "input-cases.json") == custody["input_manifest_sha256"]
                               and file_digest(args.output / "candidates.json") == custody["candidates_sha256"])
    write_json(args.output / "process-result.json", result)
    try:
        verified = verify_native(args.output)
    except (ValueError, TypeError, OSError, KeyError, AssertionError) as exc:
        verified = {"passed": False, "cases": [], "failures": ["independent verifier: " + str(exc)]}
    write_json(args.output / "native-verification.json", verified)
    passed = verified.get("passed") is True and len(verified.get("cases", [])) == 2 and all(
        x.get("success") is True for x in verified["cases"])
    assessment["native_execution"] = True
    assessment["native_confirmed_paintings"] = 2 if passed else 0
    assessment["native_independent_verification_pass"] = passed
    assessment["native_case_count"] = len(verified.get("cases", []))
    assessment["claim"] = (
        "Source-built native receipts and material PNGs verified; requested topology FAILED"
        if passed and not assessment["semantic_success"] else
        "Native verification failed or semantics unmet; no useful-painter freeze")
    write_json(args.output / "semantic-assessment.json", assessment)
    if not passed:
        print(json.dumps({"native_pass": False, "failures": verified.get("failures", []),
                          "global_gates": verified.get("global_gates", {})}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"native_pass": True, "cases": 2, "semantic_success": assessment["semantic_success"],
                      "material_pngs": [x["id"] + ".png" for x in verified["cases"]]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("SPROCKET NATIVE REFUSED: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
