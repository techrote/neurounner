"""Replay strict SMOL DEVELOPMENT outputs through the unchanged native Art owner.

The evaluator supplies raw predictions, never native success. This adapter binds
each accepted JSON object to its original bytes, frozen host seed and compiler,
then invokes the foundation's real Workbench harness and independent verifier.
Rejected predictions remain in the comparison. Zero accepted predictions runs
the native safety probes and produces exactly zero model drawing images.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

LANE = Path(__file__).resolve().parent
PAINTER = LANE.parents[1]
sys.path.insert(0, str(PAINTER))
sys.path.insert(0, str(PAINTER / "native"))
from compiler import compile_program, parse_program
from native.runtime import ROOT, digest, verify as verify_runtime
from native.verify import verify as verify_native
from run import execute, native_cases, write_json
from training.common import digest_json, read_json, relative_file, verify_file

FOUNDATION_SHA = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA256 = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
DEVELOPMENT_SHA256 = "8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572"
COMPILER_SHA256 = "4d8559805df1ff5f3f23277a7bca98e47fa8feb53337aa79e8546d0bb89132f6"
ARMS = ("baseline", "candidate")
SEEDS = {104729, 130363}
HOST_FIELDS = ("id", "group_id", "structural_group", "family", "design", "split", "prompt", "seed", "focus", "material")
RAW_FILES = {"completion", "raw_output", "receipt"}


def cohort(directory: Path) -> tuple[dict, list[dict]]:
    """Read the frozen manifest and DEVELOPMENT only; never open FINAL targets."""
    manifest_path = verify_file(directory, "admitted-corpus.json", CORPUS_SHA256)
    manifest = read_json(manifest_path)
    item = manifest["split_files"]["DEVELOPMENT"]
    if (manifest["source_sha"] != FOUNDATION_SHA or manifest["compiler_sha256"] != COMPILER_SHA256
            or item != {"path": "development.jsonl", "sha256": DEVELOPMENT_SHA256, "count": 12}):
        raise ValueError("Frozen DEVELOPMENT corpus identity changed")
    path = verify_file(directory, item["path"], DEVELOPMENT_SHA256)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        row = {key: value[key] for key in HOST_FIELDS if key not in ("focus", "material")}
        row.update(focus=value["compiled"]["focus"], material=value["compiled"]["material"])
        if row["split"] != "DEVELOPMENT" or row["focus"] != [0, 0, 96, 96] or row["material"] != 58:
            raise ValueError("Only frozen DEVELOPMENT host requests are permitted")
        rows.append(row)
    groups = {}
    for row in rows:
        groups.setdefault(row["structural_group"], []).append(row)
    if (len(rows) != 12 or len({r["id"] for r in rows}) != 12 or len(groups) != 6
            or any(len(pair) != 2 or {r["seed"] for r in pair} != SEEDS for pair in groups.values())
            or set(groups) != set(manifest["structural_groups"]["DEVELOPMENT"])):
        raise ValueError("DEVELOPMENT must contain six whole structures and twelve seed views")
    return {"source_sha": FOUNDATION_SHA, "manifest_sha256": CORPUS_SHA256,
            "development_sha256": DEVELOPMENT_SHA256, "compiler_sha256": COMPILER_SHA256}, rows


def validate_evaluation(directory: Path, arm: str, requests: list[dict], source_sha: str) -> dict:
    """Independently reject substitution, repaired JSON and host-field relabeling."""
    document = read_json(relative_file(directory, "evaluation.json"))
    accepted_document = read_json(relative_file(directory, "accepted-candidates.json"))
    if (document.get("schema_version") != 1 or document.get("arm") != arm
            or document.get("source_sha") != source_sha
            or document.get("compiler_sha256") != COMPILER_SHA256
            or accepted_document.get("source_sha") != source_sha
            or accepted_document.get("compiler_sha256") != COMPILER_SHA256):
        raise ValueError(arm + ": evaluation source/compiler identity mismatch")
    comparison = document.get("comparison_identity")
    if (not isinstance(comparison, dict)
            or digest_json(comparison) != document.get("comparison_identity_sha256")
            or digest_json({"comparison": comparison, "arm": arm, "checkpoint": document.get("checkpoint")})
            != document.get("evaluation_identity_sha256")
            or document.get("model_id") != "HuggingFaceTB/SmolLM2-135M-Instruct"
            or document.get("model_revision") != "12fd25f77366fa6b3b4b768ec3050bf629380bac"
            or document.get("accepted_candidates_sha256") != digest(directory / "accepted-candidates.json")
            or accepted_document.get("evaluation_identity_sha256") != document["evaluation_identity_sha256"]):
        raise ValueError(arm + ": model/comparison identity or accepted inventory changed")
    if arm == "baseline" and document.get("checkpoint") is not None:
        raise ValueError("Baseline must retain the untuned parent")
    if arm == "candidate":
        checkpoint = document.get("checkpoint", {})
        if (checkpoint.get("source_sha") != source_sha or type(checkpoint.get("global_step")) is not int
                or not 1 <= checkpoint["global_step"] <= 120):
            raise ValueError("Candidate must bind an actual bounded current-source checkpoint")
    cases = document.get("cases")
    expected = {row["id"]: row for row in requests}
    if (not isinstance(cases, list) or len(cases) != 12
            or len({row.get("id") for row in cases}) != 12
            or {row.get("id") for row in cases} != set(expected)):
        raise ValueError(arm + ": all twelve frozen DEVELOPMENT views must remain visible")
    accepted = accepted_document.get("examples")
    if not isinstance(accepted, list) or len({row.get("id") for row in accepted}) != len(accepted):
        raise ValueError(arm + ": invalid accepted-candidate inventory")
    by_id = {row["id"]: row for row in accepted}
    inputs = {name: digest(relative_file(directory, name)) for name in ("evaluation.json", "accepted-candidates.json")}
    outcomes, examples = [], []
    generations = {}
    for row in cases:
        identifier = row["id"]
        host = expected[identifier]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,90}", identifier):
            raise ValueError("Unsafe native case ID")
        if any(row.get(key) != host[key] for key in HOST_FIELDS):
            raise ValueError(arm + ": frozen request/host fields changed for " + identifier)
        if type(row.get("accepted")) is not bool:
            raise ValueError("Explicit first-pass acceptance is required")
        paths, hashes = row.get("paths", {}), row.get("sha256", {})
        required = RAW_FILES | ({"program", "compiled"} if row["accepted"] else set())
        if set(paths) != required or set(hashes) != required:
            raise ValueError(arm + ": raw output file inventory mismatch for " + identifier)
        files = {}
        for key in required:
            path = verify_file(directory, paths[key], hashes[key])
            if paths[key] in inputs:
                raise ValueError("Each evaluation case must retain its own files")
            inputs[paths[key]] = hashes[key]
            files[key] = path
        raw_text = files["completion"].read_text(encoding="utf-8")
        raw = read_json(files["raw_output"])
        receipt = read_json(files["receipt"])
        expected_receipt = {**row, "sha256": {key: value for key, value in hashes.items() if key != "receipt"}}
        if receipt != expected_receipt or raw != row.get("verdict"):
            raise ValueError("Per-case receipt/verdict differs from the evaluation inventory")
        tokens = raw.get("token_ids")
        eos = raw.get("eos_token_id", document.get("eos_token_id"))
        if (raw.get("raw_text") != raw_text or not isinstance(tokens, list)
                or len(tokens) > 1024 or any(type(token) is not int or token < 0 for token in tokens)
                or type(eos) is not int or eos < 0):
            raise ValueError(arm + ": raw token/text custody mismatch for " + identifier)
        terminated = bool(tokens) and tokens[-1] == eos and eos not in tokens[:-1]
        if raw.get("eos_terminated") != terminated or raw.get("accepted") != row["accepted"]:
            raise ValueError("Raw verdict differs from terminal EOS/acceptance facts")
        if raw.get("generation_failure") != row.get("generation_failure"):
            raise ValueError("Raw generation failure differs from the case receipt")
        generation = row.get("generation_id")
        if not isinstance(generation, str) or not generation:
            raise ValueError("A real generation identity or explicit attempted-generation identity is required")
        if type(row.get("generation_attempted")) is not bool or (not row["generation_attempted"] and tokens):
            raise ValueError("Unattempted generation cannot have model output tokens")
        generation_facts = (row["prompt"], tuple(tokens), raw_text, row["generation_attempted"], row.get("generation_failure"))
        if generation in generations and generations[generation] != generation_facts:
            raise ValueError("Reused generation has different prompt or raw output")
        generations[generation] = generation_facts
        program, compiled, refusal = None, None, None
        try:
            program = parse_program(raw_text)
            compiled = compile_program(program, seed=host["seed"], focus=host["focus"],
                                       material=host["material"], allowed_materials=(host["material"],))
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError) as error:
            refusal = str(error)
        if not terminated:
            refusal = (refusal + "; " if refusal else "") + "missing unique terminal EOS"
        if row.get("generation_failure"):
            refusal = (refusal + "; " if refusal else "") + str(row["generation_failure"])
        independently_accepted = compiled is not None and terminated and not row.get("generation_failure")
        if row["accepted"] != independently_accepted:
            raise ValueError(arm + ": first-pass verdict differs from untouched output for " + identifier)
        native_id = arm + "-" + identifier
        outcome = {**host, "arm": arm, "native_id": native_id,
                   "accepted": row["accepted"], "generation_id": generation,
                   "generation_reused": row.get("generation_reused", False),
                   "generation_attempted": row["generation_attempted"],
                   "eos_terminated": terminated, "refusal": refusal,
                   "predicted_family": program["family"] if compiled is not None else None,
                   "family_agreement": program["family"] == host["family"] if compiled is not None else False,
                   "raw_text": raw_text, "paths": paths, "sha256": hashes}
        if row["accepted"]:
            if (read_json(files["program"]) != program or read_json(files["compiled"]) != compiled
                    or identifier not in by_id):
                raise ValueError(arm + ": accepted output is repaired, substituted or missing")
            example = by_id[identifier]
            if (example.get("program") != program or example.get("compiled") != compiled
                    or example.get("seed") != host["seed"] or example.get("split") != "DEVELOPMENT"
                    or example.get("family") != program["family"]
                    or any(example.get(key) != host[key] for key in ("group_id", "structural_group", "prompt"))):
                raise ValueError(arm + ": accepted candidate does not bind the original request and program")
            examples.append({**example, "id": native_id, "request_id": identifier,
                             "request_family": host["family"], "arm": arm})
        elif identifier in by_id:
            raise ValueError("Rejected output cannot supply a native candidate")
        outcomes.append(outcome)
    if set(by_id) != {row["id"] for row in cases if row["accepted"]}:
        raise ValueError("Unexpected accepted candidate outside DEVELOPMENT")
    if len(generations) != 6:
        raise ValueError("This comparison requires six raw prompt generations per arm")
    recorded_generations = document.get("generations", [])
    if (len(recorded_generations) != 6 or {row.get("id") for row in recorded_generations} != set(generations)):
        raise ValueError("Actual generation inventory differs from the case receipts")
    for generation in recorded_generations:
        prompt, tokens, _text, attempted, failure = generations[generation["id"]]
        expected_ids = [row["id"] for row in cases if row["generation_id"] == generation["id"]]
        if (generation.get("prompt") != prompt or generation.get("request_ids") != expected_ids
                or generation.get("attempted") != attempted or generation.get("failure") != failure
                or generation.get("token_ids_sha256") != digest_json(list(tokens))
                or generation.get("output_tokens") != len(tokens)):
            raise ValueError("Actual generation records changed")
    attempted = sum(facts[3] for facts in generations.values())
    expected_counts = {"case_views": 12, "structural_groups": 6, "distinct_prompts": 6,
                       "model_generations_attempted": attempted,
                       "attempted_case_views": sum(row["generation_attempted"] for row in cases),
                       "accepted_case_views": len(examples),
                       "accepted_generations": len({row["generation_id"] for row in cases if row["accepted"]}),
                       "family_matching_accepted_case_views": sum(row["accepted"] and row["family_agreement"] for row in outcomes)}
    if document.get("counts") != expected_counts:
        raise ValueError("Evaluation denominators differ from retained generation/case receipts")
    return {"document": document, "inputs": inputs, "outcomes": outcomes, "examples": examples,
            "actual_generations": attempted}


def copy_evaluation(source: Path, destination: Path, verified: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in verified["inputs"].items():
        source_file = verify_file(source, name, expected)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        if digest(target) != expected:
            raise ValueError("Evaluation bytes changed while copying")


def run_native(args, packet: Path, examples: list[dict], evaluation_inputs: dict) -> dict:
    directory = packet / "native"
    directory.mkdir()
    check_runtime = lambda: verify_runtime(ROOT, args.runtime_manifest.resolve(), args.source_sha, args.godot.resolve())
    custody = check_runtime()
    document = {"schema_version": 1, "source_sha": args.source_sha, "examples": examples}
    write_json(directory / "candidates.json", document)
    write_json(directory / "input-cases.json", native_cases(document) if examples else {"schema_version": 1, "cases": []})
    harness = ROOT / "godot/tests/run_agent_painter_foundation.gd"
    helper_paths = [LANE / "native_replay.py", LANE / "native_gallery.py"]
    helper_hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in helper_paths}
    custody.update({"harness_sha256": digest(harness), "input_manifest_sha256": digest(directory / "input-cases.json"),
                    "candidates_sha256": digest(directory / "candidates.json"), "workers": args.workers,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model_evaluation_inputs": evaluation_inputs, "lane_helpers": helper_hashes,
                    "claim": "Actual native Art replay of strict first-pass SMOL DEVELOPMENT outputs; PNGs visualize stored material readback, not a framebuffer"})
    write_json(directory / "runtime-custody.json", custody)
    env = os.environ.copy()
    env.update(AP_NATIVE_CASES=str(directory / "input-cases.json"), AP_NATIVE_OUTPUT=str(directory),
               AP_NATIVE_WORKERS=str(args.workers), AP_NATIVE_SOURCE_SHA=args.source_sha,
               XDG_DATA_HOME=str(directory / "profile/data"), XDG_CONFIG_HOME=str(directory / "profile/config"),
               XDG_CACHE_HOME=str(directory / "profile/cache"))
    command = [str(args.godot.resolve()), "--headless", "--path", str(ROOT / "godot"), "--editor", "--quit"]
    imported = execute(command, env, directory / "import.log", min(args.timeout, 600))
    write_json(directory / "import-result.json", imported)
    if imported["exit_code"] == 0:
        command = [str(args.godot.resolve()), "--headless", "--path", str(ROOT / "godot"),
                   "--script", "res://tests/run_agent_painter_foundation.gd"]
        result = execute(command, env, directory / "engine.log", args.timeout)
    else:
        result = {**imported, "stage": "Godot import"}
    try:
        after = check_runtime()
        result["runtime_unchanged"] = all(custody.get(key) == value for key, value in after.items())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        result.update(runtime_unchanged=False, custody_error=str(error))
    result["harness_unchanged"] = digest(harness) == custody["harness_sha256"]
    result["input_unchanged"] = (digest(directory / "input-cases.json") == custody["input_manifest_sha256"]
                                 and digest(directory / "candidates.json") == custody["candidates_sha256"]
                                 and all(digest(ROOT / name) == value for name, value in helper_hashes.items())
                                 and all(digest(packet / "evaluations" / arm / name) == value
                                         for arm, inventory in evaluation_inputs.items() for name, value in inventory.items()))
    write_json(directory / "process-result.json", result)
    try:
        report = verify_native(directory)
    except (ValueError, KeyError, OSError, TypeError) as error:
        report = {"passed": False, "cases": [], "global_gates": {},
                  "failures": ["Native verification incomplete: " + str(error)]}
    write_json(directory / "native-verification.json", report)
    return report


def summarize(source_sha: str, corpus_identity: dict, arms: dict, report: dict, packet: Path) -> dict:
    results = {row["id"]: row for row in report["cases"]}
    summaries, cases = {}, []
    for arm, verified in arms.items():
        successes = 0
        for outcome in verified["outcomes"]:
            row = {key: value for key, value in outcome.items() if key != "raw_text"}
            result = results.get(row["native_id"])
            success = bool(result and result.get("success") is True)
            if success and not row["accepted"]:
                raise ValueError("Native success cannot belong to a rejected model output")
            row.update(native_success=success,
                       status="native-verified" if success else "native-failed" if row["accepted"] else "raw-output-rejected",
                       native_failures=result.get("failures", []) if result else (["Native case did not complete"] if row["accepted"] else []))
            cases.append(row)
            successes += success
        summaries[arm] = {"actual_generations": verified["actual_generations"], "seed_views": 12,
                          "accepted_program_views": len(verified["examples"]), "native_verified_views": successes,
                          "family_agreement_views": sum(row["accepted"] and row["family_agreement"] for row in verified["outcomes"]),
                          "evaluation_sha256": verified["inputs"]["evaluation.json"]}
    custody = read_json(packet / "native/runtime-custody.json")
    return {"schema_version": 1, "kind": "smol-native-development-comparison", "source_sha": source_sha,
            "compiler_sha256": digest(PAINTER / "compiler.py"), "corpus": corpus_identity,
            "runtime": {key: custody[key] for key in ("source_sha", "source_tree", "native_sha256", "godot_version", "godot_sha256", "rapier_sha256")},
            "evaluation_sha256": {arm: item["inputs"]["evaluation.json"] for arm, item in arms.items()},
            "evaluation_inputs": {arm: item["inputs"] for arm, item in arms.items()},
            "native_evidence": "native", "native_verification_sha256": digest(packet / "native/native-verification.json"),
            "native_passed": report["passed"], "native_global_gates": report.get("global_gates", {}),
            "native_failures": report.get("failures", []),
            "counts": {"structural_requests": 6, "seed_views_per_arm": 12, "arms": 2,
                       "accepted_program_views": sum(item["accepted_program_views"] for item in summaries.values()),
                       "native_verified_views": sum(item["native_verified_views"] for item in summaries.values())},
            "arms": summaries, "cases": cases,
            "claim_limits": ["Six structural DEVELOPMENT requests, each displayed with two host seeds; not twelve independent prompts",
                             "Strict syntax/native structure does not establish prompt fulfillment or visual quality",
                             "Original PNGs are Godot native material-readback visualizations, not framebuffer screenshots",
                             "Rejected outputs have no substitute teacher program, raster or generated image",
                             "Original corpus qualification remains attributed to source " + FOUNDATION_SHA,
                             "No FINAL targets, images or evaluation are consumed"]}


def verify_packet(directory: Path, source_sha: str | None = None, *, details: bool = False) -> dict:
    """Recheck a portable packet without executing Godot or loading any model."""
    directory = Path(directory)
    saved = read_json(relative_file(directory, "model-native-summary.json"))
    expected_source = source_sha or saved["source_sha"]
    if saved.get("source_sha") != expected_source:
        raise ValueError("Native comparison source differs from requested publication source")
    corpus_identity, requests = cohort(directory / "cohort")
    arms = {arm: validate_evaluation(directory / "evaluations" / arm, arm, requests, expected_source) for arm in ARMS}
    if arms["baseline"]["document"]["comparison_identity"] != arms["candidate"]["document"]["comparison_identity"]:
        raise ValueError("Baseline and candidate did not use the same frozen evaluation recipe")
    native = directory / "native"
    custody = read_json(relative_file(native, "runtime-custody.json"))
    expected_inputs = {arm: value["inputs"] for arm, value in arms.items()}
    if custody.get("model_evaluation_inputs") != expected_inputs or custody.get("source_sha") != expected_source:
        raise ValueError("Native execution is not bound to these exact evaluation inputs/source")
    for path in (LANE / "native_replay.py", LANE / "native_gallery.py"):
        relative = path.relative_to(ROOT).as_posix()
        if custody.get("lane_helpers", {}).get(relative) != digest(path):
            raise ValueError("Lane helper differs from recorded native execution: " + relative)
    examples = [example for arm in ARMS for example in arms[arm]["examples"]]
    expected_candidates = {"schema_version": 1, "source_sha": expected_source, "examples": examples}
    expected_native = native_cases(expected_candidates) if examples else {"schema_version": 1, "cases": []}
    if (read_json(relative_file(native, "candidates.json")) != expected_candidates
            or read_json(relative_file(native, "input-cases.json")) != expected_native):
        raise ValueError("Native cases are not the original accepted model programs")
    recorded_report = read_json(relative_file(native, "native-verification.json"))
    try:
        report = verify_native(native)
    except (ValueError, KeyError, OSError, TypeError) as error:
        if recorded_report.get("passed") is not False or recorded_report.get("cases"):
            raise ValueError("Incomplete native evidence cannot claim model success") from error
        # A failed import/process may have no native summary. Preserve that
        # explicit negative result and raw predictions, with no image admission.
        report = recorded_report
    if report != recorded_report:
        raise ValueError("Recorded native verification differs from independent readback recheck")
    expected_summary = summarize(expected_source, corpus_identity, arms, report, directory)
    if expected_summary != saved:
        raise ValueError("Native comparison summary differs from raw-output/native evidence")
    proof = {"summary": saved, "arms": arms, "native_report": report}
    if details:
        return proof
    from native_gallery import verify_gallery
    gallery = verify_gallery(directory, directory / "gallery", proof)
    return {**saved, "runtime_sha256": saved["runtime"]["native_sha256"], "gallery": gallery}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "corpus", "godot", "runtime-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--workers", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.source_sha) or not 30 <= args.timeout <= 3600:
        parser.error("Full source SHA and 30–3600 second native deadline required")
    if digest(PAINTER / "compiler.py") != COMPILER_SHA256:
        raise ValueError("This comparison is frozen to the qualified v2 compiler")
    corpus_identity, requests = cohort(args.corpus)
    inputs = {arm: validate_evaluation(getattr(args, arm), arm, requests, args.source_sha) for arm in ARMS}
    packet = args.output.resolve()
    packet.mkdir(parents=True, exist_ok=False)
    (packet / "cohort").mkdir()
    for name in ("admitted-corpus.json", "development.jsonl"):
        shutil.copyfile(relative_file(args.corpus, name), packet / "cohort" / name)
    if cohort(packet / "cohort") != (corpus_identity, requests):
        raise ValueError("Cohort changed while copying")
    for arm in ARMS:
        copy_evaluation(getattr(args, arm), packet / "evaluations" / arm, inputs[arm])
    arms = {arm: validate_evaluation(packet / "evaluations" / arm, arm, requests, args.source_sha) for arm in ARMS}
    if arms["baseline"]["document"]["comparison_identity"] != arms["candidate"]["document"]["comparison_identity"]:
        raise ValueError("Baseline and candidate did not use the same frozen evaluation recipe")
    examples = [example for arm in ARMS for example in arms[arm]["examples"]]
    report = run_native(args, packet, examples, {arm: value["inputs"] for arm, value in arms.items()})
    summary = summarize(args.source_sha, corpus_identity, arms, report, packet)
    write_json(packet / "model-native-summary.json", summary)
    from native_gallery import build
    build(packet, packet / "gallery")
    print(json.dumps({"native_passed": summary["native_passed"], "counts": summary["counts"],
                      "arms": summary["arms"], "output": str(packet)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print("SMOL native replay refused: " + str(error), file=sys.stderr)
        raise SystemExit(2)
