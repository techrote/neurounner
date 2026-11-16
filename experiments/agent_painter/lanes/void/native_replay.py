#!/usr/bin/env python3
"""Source-built, fail-closed native replay of immutable VOID DEVELOPMENT first passes.

No model inference, training, JSON repair, FINAL targets, or teacher substitution.
The original positive corpus and model execution retain their separate source IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

PAINTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER))
from compiler import compile_program, parse_program
sys.path.insert(0, str(PAINTER / "native"))
from run import execute, native_cases, write_json
from native.runtime import ROOT, digest, verify as verify_runtime
from native.verify import verify as verify_native

MODEL_SOURCE = "7d90bf5294d16a8644694a4c8a6a36a5227fa3c0"
NATIVE_CORPUS_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
MODEL_REVISION = "9c2be5087772867d45961abd11f6b35110617563"
MODEL_ID = "dotlabs/void.1"
SEEDS = {104729, 130363}
ARMS = ("base", "adapter")


def load(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked evidence: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def safe_child(root: Path, *names: str):
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", name):
            raise ValueError("Unsafe filename")
    result = root.joinpath(*names)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes root")
    return result


def read_cohort(corpus: Path):
    if digest(corpus / "admitted-corpus.json") != CORPUS_SHA:
        raise ValueError("Native-admitted corpus SHA differs")
    manifest = load(corpus / "admitted-corpus.json")
    if (manifest.get("state") != "native-admitted"
            or manifest.get("source_sha") != NATIVE_CORPUS_SOURCE):
        raise ValueError("Native source/positive admission mismatch")
    spec = manifest["split_files"]["DEVELOPMENT"]
    if spec["count"] != 12 or spec["path"] != "development.jsonl":
        raise ValueError("Frozen DEVELOPMENT cohort differs")
    dev = corpus / "development.jsonl"
    if digest(dev) != spec["sha256"]:
        raise ValueError("Frozen DEVELOPMENT file hash differs")
    rows = [json.loads(line) for line in dev.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {}
    for row in rows:
        if row["split"] != "DEVELOPMENT":
            raise ValueError("Non-DEVELOPMENT target in cohort")
        groups.setdefault(row["group_id"], []).append(row)
    if (len(rows) != 12 or len(groups) != 6
            or set(groups) != set(manifest["structural_groups"]["DEVELOPMENT"])
            or any(len(items) != 2 or {x["seed"] for x in items} != SEEDS for items in groups.values())):
        raise ValueError("Six disjoint groups and both host seeds required")
    # Never access program, target completion, or original expected geometry.
    host = {row["id"]: {"id": row["id"], "group": row["group_id"],
                       "family": row["family"], "prompt": row["prompt"],
                       "seed": row["seed"], "focus": row["compiled"]["focus"],
                       "material": row["compiled"]["material"]} for row in rows}
    if len(host) != 12:
        raise ValueError("Duplicate native cohort case ID")
    return host, spec["sha256"]


def validate_model_output(development: Path, host: dict):
    identity = load(development / "identity.json")
    if (identity.get("source_sha") != MODEL_SOURCE
            or identity.get("native_source_sha") != NATIVE_CORPUS_SOURCE
            or identity.get("corpus_sha256") != CORPUS_SHA
            or identity.get("model_id") != MODEL_ID
            or identity.get("model_revision") != MODEL_REVISION
            or identity.get("cases") != 12 or identity.get("split") != "DEVELOPMENT"):
        raise ValueError("Producer model/source/split contract mismatch")
    summary = load(development / "summary.json")
    if (summary.get("distinct_language_requests") != 6
            or summary.get("host_seed_views_per_arm") != 12
            or not summary.get("complete")
            or summary.get("native_execution") is not False):
        raise ValueError("Original developmental accounting mismatch")
    rows, examples, first_passes, input_hashes = [], [], {}, {}
    for arm in ARMS:
        for case_id in sorted(host):
            expected = host[case_id]
            folder = safe_child(development, arm, case_id)
            raw_path = folder / "raw-output.json"
            receipt_path = folder / "receipt.json"
            text_path = folder / "completion.txt"
            batch_path = folder / "art_batch.json"
            raw, receipt, ops = load(raw_path), load(receipt_path), load(batch_path)
            completion = text_path.read_bytes().decode("utf-8", errors="strict")
            for f in (raw_path, receipt_path, text_path, batch_path):
                input_hashes[f.relative_to(development).as_posix()] = digest(f)
            if (receipt["case"] != case_id or receipt["arm"] != arm
                    or receipt["family"] != expected["family"]
                    or receipt["group"] != expected["group"]
                    or receipt["original_prompt"] != expected["prompt"]
                    or receipt["host_seed"] != expected["seed"]
                    or receipt["host_focus"] != expected["focus"]
                    or receipt["host_material"] != expected["material"]
                    or receipt["completion_sha256"] != digest(text_path)
                    or receipt.get("native_execution") is not False
                    or raw.get("raw_text") != completion):
                raise ValueError("Source/prompt/native host receipt mismatch: " + case_id)
            ids = raw.get("token_ids")
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 1024
                    or any(type(i) is not int or not 0 <= i <= 258 for i in ids)):
                raise ValueError("Missing/invalid raw byte token IDs")
            terminal = ids[-1] == 258 and 258 not in ids[:-1]
            if terminal != raw.get("eos_terminated") or terminal != receipt.get("eos_terminated"):
                raise ValueError("Terminal EOS differs from raw tokens")
            if raw.get("text_decoded") is True:
                if (258 in ids[:-1] or any(i > 255 for i in ids[:(-1 if terminal else None)])
                        or bytes(ids[:(-1 if terminal else None)]).decode("utf-8") != completion):
                    raise ValueError("Original raw bytes differ from reported completion")
            else:
                if receipt.get("first_pass_accepted"):
                    raise ValueError("Malformed UTF-8 cannot be admitted")
            key = (arm, expected["group"])
            fact = (completion, tuple(ids), expected["prompt"])
            if key in first_passes and first_passes[key] != fact:
                raise ValueError("Host-seed pair has different model first pass")
            if key not in first_passes:
                first_passes[key] = fact
            try:
                program = parse_program(completion)
                compiled = compile_program(program, seed=expected["seed"],
                                           focus=expected["focus"],
                                           material=expected["material"],
                                           allowed_materials=(expected["material"],))
                accepted = terminal and raw.get("text_decoded") is True
            except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
                program, compiled, accepted = None, None, False
            if accepted != receipt["first_pass_accepted"] or accepted != raw["first_pass_accepted"]:
                raise ValueError("Raw replay disagrees with first-pass verdict")
            if receipt["compiler_accepted"] != accepted or (ops != (compiled["operations"] if accepted else [])):
                raise ValueError("Native operation payload does not match raw compilation")
            compiled_path = folder / "compiled.json"
            if accepted:
                if load(compiled_path) != compiled or receipt["operations_sha256"] != compiled["operations_sha256"]:
                    raise ValueError("Compiled output was repaired or substituted")
                input_hashes[compiled_path.relative_to(development).as_posix()] = digest(compiled_path)
                examples.append({"id": f"void-{arm}-{case_id}", "family": program["family"],
                                 "split": "DEVELOPMENT", "seed": expected["seed"],
                                 "program": program, "compiled": compiled})
            elif compiled_path.exists():
                raise ValueError("Rejected output carries compiled substitution")
            rows.append({"case": case_id, "arm": arm, "group": expected["group"],
                         "family": expected["family"], "accepted": accepted,
                         "eos": terminal, "json_valid": receipt["json_valid"],
                         "native_case_id": f"void-{arm}-{case_id}" if accepted else None,
                         "completion_sha256": digest(text_path),
                         "operations_sha256": compiled["operations_sha256"] if accepted else None})
    if len(first_passes) != 12 or len(rows) != 24:
        raise ValueError("Incorrect independent generation/view denominators")
    if {arm: sum(r["accepted"] for r in rows if r["arm"] == arm) for arm in ARMS} != {
            arm: summary["arms"][arm]["first_pass_accepted"] for arm in ARMS}:
        raise ValueError("Original A/B summary differs from independent compilation")
    if len(examples) != 2 or {x["family"] for x in examples} != {"pipework"}:
        raise ValueError("Recorded accepted-candidate cohort differs from immutable producer")
    return identity, rows, examples, input_hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "corpus", "godot", "runtime-manifest", "output", "source-sha"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    dev, corpus, output = Path(args.development), Path(args.corpus), Path(args.output)
    godot, stamp = Path(args.godot), Path(args.runtime_manifest)
    if output.exists() or not re.fullmatch("[a-f0-9]{40}", args.source_sha):
        ap.error("new output and full source SHA required")
    host, development_sha = read_cohort(corpus)
    identity, rows, examples, inputs = validate_model_output(dev, host)
    output.mkdir(parents=True)
    custody = verify_runtime(ROOT, stamp.resolve(), args.source_sha, godot.resolve())
    document = {"schema_version": 1, "source_sha": args.source_sha, "examples": examples}
    write_json(output / "candidates.json", document)
    write_json(output / "input-cases.json", native_cases(document))
    harness = ROOT / "godot/tests/run_agent_painter_foundation.gd"
    my_script = Path(__file__).resolve()
    custody.update({"claim": "Source-built Godot native Art execution of unrepaired VOID first-pass DEVELOPMENT predictions; no TRAIN admission",
                    "harness_sha256": digest(harness),
                    "input_manifest_sha256": digest(output / "input-cases.json"),
                    "candidates_sha256": digest(output / "candidates.json"),
                    "model_development_sha256": digest(dev / "summary.json"),
                    "model_source_sha": MODEL_SOURCE, "model_checkpoint": identity["adapter_sha256"],
                    "native_corpus_sha256": CORPUS_SHA, "development_split_sha256": development_sha,
                    "first_pass_files": inputs, "lane_script_sha256": digest(my_script),
                    "workers": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_json(output / "runtime-custody.json", custody)
    env = os.environ.copy()
    env.update(AP_NATIVE_CASES=str((output / "input-cases.json").resolve()),
               AP_NATIVE_OUTPUT=str(output.resolve()), AP_NATIVE_WORKERS="1",
               AP_NATIVE_SOURCE_SHA=args.source_sha,
               XDG_DATA_HOME=str((output / "profile/data").resolve()),
               XDG_CONFIG_HOME=str((output / "profile/config").resolve()),
               XDG_CACHE_HOME=str((output / "profile/cache").resolve()))
    import_command = [str(godot.resolve()), "--headless", "--path", str(ROOT / "godot"), "--editor", "--quit"]
    imported = execute(import_command, env, output / "import.log", 600)
    write_json(output / "import-result.json", imported)
    if imported["exit_code"] == 0:
        command = [str(godot.resolve()), "--headless", "--path", str(ROOT / "godot"),
                   "--script", "res://tests/run_agent_painter_foundation.gd"]
        result = execute(command, env, output / "engine.log", 1800)
    else:
        result = {**imported, "stage": "Godot import"}
    try:
        after = verify_runtime(ROOT, stamp.resolve(), args.source_sha, godot.resolve())
        result["runtime_unchanged"] = all(custody.get(k) == v for k, v in after.items())
    except (ValueError, OSError) as error:
        result["runtime_unchanged"] = False
        result["custody_error"] = str(error)
    result["harness_unchanged"] = digest(harness) == custody["harness_sha256"]
    result["input_unchanged"] = (
        digest(output / "input-cases.json") == custody["input_manifest_sha256"]
        and digest(output / "candidates.json") == custody["candidates_sha256"]
        and digest(my_script) == custody["lane_script_sha256"]
        and all(digest(dev / name) == sha for name, sha in inputs.items()))
    write_json(output / "process-result.json", result)
    try:
        report = verify_native(output)
    except (ValueError, KeyError, OSError, TypeError) as error:
        report = {"passed": False, "cases": [], "failures": ["Verification incomplete: " + str(error)]}
    write_json(output / "native-verification.json", report)
    print(json.dumps({"native_verifier_failures": report.get("failures", []),
                      "native_verifier_global_gates": report.get("global_gates", {}),
                      "native_verifier_cases": len(report.get("cases", []))}, sort_keys=True), flush=True)
    passed = {x["id"]: x for x in report.get("cases", []) if x.get("success") is True}
    for row in rows:
        row["native_verified"] = row["native_case_id"] in passed if row["accepted"] else False
        row["native_affected_cells"] = passed[row["native_case_id"]].get("affected_cells") if row["native_verified"] else None
    trained = [x for x in examples if x["id"].startswith("void-adapter-")]
    topologies = []
    for x in trained:
        nodes = x["program"]["components"][0]["nodes"]
        edges = x["program"]["components"][0]["edges"]
        degrees = [sum(i in e for e in edges) for i in range(len(nodes))]
        topologies.append({"case": x["id"], "endpoints": degrees.count(1),
                           "degree_three_junctions": degrees.count(3),
                           "cycles": len(edges) - len(nodes) + 1,
                           "diameter": x["program"]["components"][0]["diameter"],
                           "requested_five_endpoints_three_upward_branches": False
                           if degrees.count(1) != 5 else None})
    summary = {"producer": MODEL_SOURCE, "native_source": args.source_sha,
               "original_corpus_source": NATIVE_CORPUS_SOURCE, "original_corpus_sha256": CORPUS_SHA,
               "model_revision": MODEL_REVISION, "original_model_generations_per_arm": 6,
               "original_host_seed_views_per_arm": 12, "accepted_seed_views": len(examples),
               "independent_accepted_generations": 1, "native_verified_views": sum(r["native_verified"] for r in rows),
               "native_verifier_passed": report.get("passed"), "topology": topologies,
               "visual_usefulness": "NOT QUALIFIED: one TRAIN-like tee, not held-out three-branch manifold",
               "sealed_final_opened": False, "cases": rows}
    write_json(output / "void-native-summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, sort_keys=True), flush=True)
    if not report.get("passed") or summary["native_verified_views"] != 2:
        raise SystemExit("Native verification failed; retain all original evidence and partial receipts")


if __name__ == "__main__":
    main()
