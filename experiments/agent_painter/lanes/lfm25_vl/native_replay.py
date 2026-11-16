#!/usr/bin/env python3
"""Source-built native Art replay of immutable LFM 48/120 DEVELOPMENT first passes.

Only original raw completions are admitted. No model inference, repair,
teacher substitution, FINAL targets, or training occurs in this module.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PAINTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER))
from compiler import compile_program, parse_program
sys.path.insert(0, str(PAINTER / "native"))
from run import execute, native_cases, write_json
from native.runtime import ROOT, digest, verify as verify_runtime
from verify import verify as verify_native

EVAL = "984b595b3b612341adc9ca0a4112698b5a7f6743"
TRAIN = "1c9cf2ef8760177e0c1d095052c27fcf200a5cb3"
CORPUS = "633920373a3d406a2ede61ce4137c324f76be23b"
MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
MODEL = "LiquidAI/LFM2.5-VL-450M"
REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"
ARMS = {"fresh48": 48, "fresh120": 120}


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing/symlinked evidence: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(corpus, evidence):
    manifest = read(corpus / "admitted-corpus.json")
    if (digest(corpus / "admitted-corpus.json") != MANIFEST
            or manifest["source_sha"] != CORPUS or manifest["state"] != "native-admitted"):
        raise ValueError("Original native admission/source identity mismatch")
    spec = manifest["split_files"]["DEVELOPMENT"]
    devfile = corpus / "development.jsonl"
    if spec["path"] != "development.jsonl" or spec["count"] != 12 or digest(devfile) != spec["sha256"]:
        raise ValueError("Frozen DEVELOPMENT hash/count changed")
    rows = [json.loads(x) for x in devfile.read_text(encoding="utf-8").splitlines() if x.strip()]
    groups = {r["group_id"] for r in rows}
    if (len(rows) != 12 or len(groups) != 6 or len({r["id"] for r in rows}) != 12
            or any(r["split"] != "DEVELOPMENT"
                   or r["group_id"] not in manifest["structural_groups"]["DEVELOPMENT"] for r in rows)
            or any({r["seed"] for r in rows if r["group_id"] == g} != {104729, 130363} for g in groups)):
        raise ValueError("Structural split, group and host seed accounting mismatch")
    # Host request/seed/focus/material only; no teacher target/program or FINAL files are consulted.
    records, examples, topology, original_hashes, totals = [], [], [], {}, {}
    for arm, step in ARMS.items():
        root = evidence / arm / arm
        prov, summary = read(root / "provenance.json"), read(root / "summary.json")
        if (prov.get("evaluation_source_sha") != EVAL
                or prov.get("training_source_sha") != TRAIN
                or prov.get("model") != MODEL or prov.get("model_revision") != REVISION
                or prov.get("native_source_sha") != CORPUS
                or prov.get("corpus_manifest_sha256") != MANIFEST
                or prov.get("expected_optimizer_step") != step
                or prov.get("independent_language_requests") != 6
                or prov.get("seed_views") != 12
                or prov.get("native_execution") is not False
                or prov.get("final_targets_opened") is not False
                or summary.get("source_sha") != EVAL
                or summary.get("training_source_sha") != TRAIN
                or summary.get("first_pass_only") is not True
                or summary.get("final_targets_opened") is not False
                or summary.get("native_execution") is not False
                or summary.get("status") != "complete"
                or (summary["global"]["requests"], summary["global"]["case_views"],
                    summary["global"]["compiled"], summary["global"]["program"]) != (6, 12, 2, 1)):
            raise ValueError("Original model/evaluation/checkpoint contract mismatch: " + arm)
        pairs = {}
        totals[arm] = {"unique_requests": 6, "host_views": 12,
                       "compiler_accepted_views": 0, "eos": summary["global"]["eos"],
                       "json": summary["global"]["json"]}
        for index, row in enumerate(rows):
            folder = root / "cases" / f"{index:02d}"
            raw, receipt = read(folder / "raw-output.json"), read(folder / "receipt.json")
            text = (folder / "completion.txt").read_text(encoding="utf-8")
            for name in ("raw-output.json", "receipt.json", "completion.txt"):
                file = folder / name
                if file.is_symlink():
                    raise ValueError("Symlink original model evidence")
                original_hashes[f"{arm}/{index:02d}/{name}"] = sha(file)
            if (text != raw.get("raw_text") or sha(folder / "completion.txt") != receipt["completion_sha256"]
                    or receipt.get("id") != row["id"] or receipt.get("prompt") != row["prompt"]
                    or receipt.get("family") != row["family"]
                    or receipt.get("structural_group") != row["group_id"]
                    or receipt.get("seed") != row["seed"] or receipt.get("variant") != "tuned"
                    or receipt.get("native_execution") is not False
                    or receipt.get("max_new_tokens") != 384
                    or receipt.get("generation_id") != row["group_id"]):
                raise ValueError("Original raw text/prompt/host receipt mismatch: " + str(folder))
            tokens = raw.get("token_ids")
            if (not isinstance(tokens, list) or not tokens or len(tokens) > 384
                    or any(type(t) is not int or not 0 <= t < 65536 for t in tokens)):
                raise ValueError("Unqualified model token IDs")
            terminal = tokens[-1] == 7 and 7 not in tokens[:-1]
            if terminal != raw["eos_terminated"] or terminal != receipt["eos_terminated"]:
                raise ValueError("EOS differs from original raw token receipts")
            key = row["group_id"]
            if key in pairs:
                if pairs[key] != (text, tuple(tokens)) or receipt["generation_reused"] is not True:
                    raise ValueError("Host-seed views must reuse identical original model output")
            else:
                pairs[key] = (text, tuple(tokens))
                if receipt["generation_reused"] is not False:
                    raise ValueError("First model-view evidence incorrectly marked reused")
            try:
                program = parse_program(text)
                compiled = compile_program(program, seed=row["seed"],
                                           focus=row["compiled"]["focus"],
                                           material=row["compiled"]["material"],
                                           allowed_materials=(row["compiled"]["material"],))
                accepted = terminal and raw.get("text_decoded") is True
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                program, compiled, accepted = None, None, False
            if (accepted != receipt.get("compile_accepted")
                    or accepted != receipt.get("first_pass_accepted")
                    or raw.get("first_pass_accepted") != receipt.get("first_pass_accepted")):
                raise ValueError("Original first-pass verdict differs from independent raw compilation")
            source_compiled = folder / "compiled-only.json"
            case_id = f"lfm-{arm}-{row['id']}" if accepted else None
            if accepted:
                if read(source_compiled) != compiled:
                    raise ValueError("Original compiled output substituted or altered")
                original_hashes[f"{arm}/{index:02d}/compiled-only.json"] = sha(source_compiled)
                examples.append({"id": case_id, "family": program["family"],
                                 "seed": row["seed"], "split": "DEVELOPMENT",
                                 "program": program, "compiled": compiled})
                totals[arm]["compiler_accepted_views"] += 1
                net = program["components"][0]
                edges, nodes = net["edges"], net["nodes"]
                degrees = [sum(i in edge for edge in edges) for i in range(len(nodes))]
                up = 0
                for i, degree in enumerate(degrees):
                    if degree != 1:
                        continue
                    for a, b in edges:
                        if i not in (a, b):
                            continue
                        other = b if i == a else a
                        up += int(nodes[i][0] == nodes[other][0] and nodes[i][1] < nodes[other][1])
                shape = (program["family"] == "pipework" and net["diameter"] == 3
                         and degrees.count(1) == 5 and degrees.count(3) == 3
                         and up == 3 and len(edges) - len(nodes) + 1 == 0)
                topology.append({"case": case_id, "endpoints": degrees.count(1),
                                 "three_way_junctions": degrees.count(3),
                                 "upward_terminal_branches": up, "diameter": net["diameter"],
                                 "cycles": len(edges) - len(nodes) + 1,
                                 "requested_manifold_topology": shape})
            elif source_compiled.exists():
                raise ValueError("Rejected model response has replacement compiled operations")
            records.append({"arm": arm, "id": row["id"], "accepted": accepted,
                            "native_case_id": case_id,
                            "raw_completion_sha256": sha(folder / "completion.txt")})
        if len(pairs) != 6 or totals[arm]["compiler_accepted_views"] != 2:
            raise ValueError("Original independent-requests/accepted-view counts changed")
    if len(examples) != 4 or {x["family"] for x in examples} != {"pipework"}:
        raise ValueError("Source-bound accepted cohort differs: native replay refused")
    return examples, records, topology, original_hashes, totals


def main():
    ap = argparse.ArgumentParser()
    for name in ("corpus", "development", "godot", "runtime-manifest", "source-sha", "output"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite native evidence")
    examples, cases, topology, hashes, totals = collect(Path(args.corpus), Path(args.development))
    output.mkdir(parents=True)
    candidate = {"schema_version": 1, "source_sha": args.source_sha, "examples": examples}
    write_json(output / "candidates.json", candidate)
    write_json(output / "input-cases.json", native_cases(candidate))
    godot, stamp = Path(args.godot).resolve(), Path(args.runtime_manifest).resolve()
    before = verify_runtime(ROOT, stamp, args.source_sha, godot)
    harness = ROOT / "godot/tests/run_agent_painter_foundation.gd"
    script = Path(__file__).resolve()
    before.update({"harness_sha256": digest(harness), "script_sha256": digest(script),
                   "candidate_sha256": digest(output / "candidates.json"),
                   "candidates_sha256": digest(output / "candidates.json"),
                   "input_sha256": digest(output / "input-cases.json"),
                   "input_manifest_sha256": digest(output / "input-cases.json"),
                   "claim": "Actual source-built native Art replay of original LFM DEVELOPMENT completions; model output unchanged; no teacher substitution",
                   "model_training_source": TRAIN, "model_evaluation_source": EVAL,
                   "original_native_corpus_source": CORPUS, "original_corpus_sha256": MANIFEST,
                   "original_model_output_hashes": hashes})
    write_json(output / "runtime-custody.json", before)
    env = os.environ.copy()
    env.update(AP_NATIVE_CASES=str((output / "input-cases.json").resolve()),
               AP_NATIVE_OUTPUT=str(output.resolve()), AP_NATIVE_WORKERS="1",
               AP_NATIVE_SOURCE_SHA=args.source_sha,
               XDG_DATA_HOME=str((output / "profile/data").resolve()),
               XDG_CONFIG_HOME=str((output / "profile/config").resolve()),
               XDG_CACHE_HOME=str((output / "profile/cache").resolve()))
    imported = execute([str(godot), "--headless", "--path", str(ROOT / "godot"),
                        "--editor", "--quit"], env, output / "import.log", 600)
    write_json(output / "import-result.json", imported)
    if imported["exit_code"] == 0:
        result = execute([str(godot), "--headless", "--path", str(ROOT / "godot"),
                          "--script", "res://tests/run_agent_painter_foundation.gd"],
                         env, output / "engine.log", 1800)
    else:
        result = {**imported, "stage": "Godot import"}
    after = verify_runtime(ROOT, stamp, args.source_sha, godot)
    result["runtime_unchanged"] = all(before.get(k) == v for k, v in after.items())
    result["harness_unchanged"] = digest(harness) == before["harness_sha256"]
    result["input_unchanged"] = (
        digest(output / "input-cases.json") == before["input_sha256"]
        and digest(output / "candidates.json") == before["candidate_sha256"]
        and digest(script) == before["script_sha256"]
        and all(sha(Path(args.development) / arm / arm / "cases" / idx / name) == value
                for path, value in hashes.items() for arm, idx, name in (path.split("/", 2),)))
    write_json(output / "process-result.json", result)
    try:
        report = verify_native(output)
    except (ValueError, KeyError, OSError, TypeError) as error:
        report = {"passed": False, "cases": [], "failures": [str(error)]}
    write_json(output / "native-verification.json", report)
    good = {item["id"]: item for item in report.get("cases", []) if item.get("success") is True}
    for item in cases:
        item["native_verified"] = item["native_case_id"] in good if item["accepted"] else False
    summary = {"schema_version": 1, "model": MODEL, "model_revision": REVISION,
               "training_source": TRAIN, "evaluation_source": EVAL,
               "native_execution_source": args.source_sha,
               "original_native_corpus_source": CORPUS,
               "original_manifest_sha256": MANIFEST,
               "independent_model_generations_per_arm": 6, "host_seed_views_per_arm": 12,
               "counts": totals, "compiler_accepted_host_views": 4,
               "native_verified_views": sum(x["native_verified"] for x in cases),
               "native_verifier_passed": report.get("passed") is True,
               "topology": topology, "visual_review": "PENDING",
               "final_targets_opened": False, "cases": cases}
    write_json(output / "lfm-native-summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, sort_keys=True), flush=True)
    if (not report.get("passed") or summary["native_verified_views"] != 4
            or not all(result.get(k) for k in ("runtime_unchanged",
                                               "harness_unchanged", "input_unchanged"))):
        raise SystemExit("Actual native Art/readback not fully qualified; preserve all evidence")


if __name__ == "__main__":
    main()
