#!/usr/bin/env python3
"""Recompute the descriptive Sprocket supplement from published F evidence.

Requires Python 3.11+ and only standard-library dependencies. Native read-only
validators reconstruct archived requests; no model, tokenizer, or provider runs.
The supplement does not change frozen syntax flags, rubrics, or thresholds.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
REL = Path("experiments/village-001/diagnostic-20261119")
FREEZE = "78a72238a8403c4cddb9e8651242a89e17ee21be"
HARNESS_SHA = "8066e05169f3fb10048945b6d9158ecffd1c3992360d1d058d688ee823702050"
ARMS = ("original", "compact_schema_first")
SCORES = ("response_grounding", "scenario_grounding", "corrective_usefulness")
FIELDS = {"keep", "mutate", "failure_guess", "genericness"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    require(path.is_file() and not path.is_symlink(), "missing or nonregular source: " + str(path))
    return digest(path.read_bytes())


def read(path):
    sha(path)
    return json.loads(path.read_text())


def inventory(root):
    require(root.is_dir() and not root.is_symlink(), "missing or nonregular source directory")
    found = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "nonregular source entry")
        if path.is_file():
            found[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return found


def forbidden(*args, **kwargs):
    raise RuntimeError("network/model/provider execution is forbidden during supplemental recomputation")


def accepted_report(path, *, final=False):
    value = read(path)
    require(value.get("audit_script_sha256") == HARNESS_SHA and value.get("source_freeze_commit") == FREEZE,
            "acceptance receipt does not identify the frozen independent harness/source")
    require(value.get("source_acceptance") == value.get("input_acceptance") == "pass" and not value.get("failed"),
            "accepted actual input evidence is required")
    if final:
        require(value.get("final_acceptance") == "pass", "final acceptance is required before semantic recomputation")
    return value


def counts(rows):
    objects = [row for row in rows if row["whole_string_json_object"]]
    result = {"responses": len(rows), **{key: sum(row[key] for row in rows) for key in
        ("legacy_schema_valid", "strict_json_schema_valid", "empty_response", "whole_string_json_parse",
         "whole_string_json_object", "starts_with_open_brace")},
        "completion_tokens_equal_160": sum(row["completion_tokens"] == 160 for row in rows),
        "completion_token_sum": sum(row["completion_tokens"] for row in rows),
        "mean_completion_tokens": sum(row["completion_tokens"] for row in rows) / len(rows),
        "finish_reasons": dict(sorted(Counter(row["finish_reason"] for row in rows).items())),
        "whole_object_shape_denominator": len(objects),
        "whole_object_shape_status": "descriptive" if objects else "not_evaluable_no_whole_string_objects",
        "whole_objects_exactly_one_keep_and_one_mutate": sum(row["keep_items"] == row["mutate_items"] == 1 for row in objects),
        "whole_objects_with_extra_fields": sum(bool(row["extra_fields"]) for row in objects),
        "whole_object_extra_field_names": dict(sorted(Counter(key for row in objects for key in row["extra_fields"]).items()))}
    if rows[0]["arm"] == "original":
        result.update({key: sum(row[key] for row in rows) for key in
            ("historical_raw_exact_match", "historical_tokens_match", "historical_finish_match")})
    if "rating" in rows[0]:
        result["semantic"] = {
            **{key: sum(row["rating"][key] for row in rows) for key in
               ("unsupported_claims", "grounded", "strict_and_grounded", "strict_grounded_and_actionable")},
            "score_sums": {key: sum(row["rating"][key] for row in rows) for key in SCORES},
            "means": {key: sum(row["rating"][key] for row in rows) / len(rows) for key in SCORES}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output file; never a native input/result file")
    parser.add_argument("--mode", choices=("inputs", "final"), default="inputs")
    parser.add_argument("--acceptance", type=Path, help="Default: repository review/input-acceptance.json")
    parser.add_argument("--final-acceptance", type=Path, help="Default: repository review/final-acceptance.json")
    parser.add_argument("--custody", type=Path, help="Default: sprocket-custody.json beside this script")
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    work = repo / REL
    require(not out.exists() and not args.out.is_symlink(), "output must be a new file")
    require(not any(source == out or source in out.parents for source in
                    (work / "inputs", work / "results")), "cannot write into native evidence")
    acceptance_path = args.acceptance or work / "review/input-acceptance.json"
    custody_path = args.custody or Path(__file__).with_name("sprocket-custody.json")
    accepted, public_custody = accepted_report(acceptance_path), read(custody_path)
    require(sha(acceptance_path) == public_custody["shared_review"]["input_acceptance_sha256"],
            "input acceptance differs from public custody")
    require(accepted["input_release"]["tag_commit"] == public_custody["input_publication"]["commit"],
            "input publication differs from custody")
    for relative, expected in accepted["source"]["executable_sources_sha256"].items():
        require(sha(repo / relative) == expected, "frozen executable source changed: " + relative)
    for name, expected in accepted["source"]["contract_sha256"].items():
        require(sha(work / name) == expected, "frozen contract changed: " + name)
    for name in ("input-custody.json", "input-release-assets.json"):
        require(sha(work / name) == public_custody["published_receipts_sha256"][name], "published receipt changed")
    native = work / "inputs/sprocket"
    native_inventory = inventory(native)
    recorded_inputs = read(work / "input-custody.json")["files"]
    require(native_inventory == {name.removeprefix("sprocket/"): item for name, item in recorded_inputs.items()
                                 if name.startswith("sprocket/")}, "Sprocket/input-custody inventory mismatch")
    for name in ("plan.json", "summary.json", "evidence-hashes.json"):
        require(sha(native / name) == public_custody["native_sha256"][name], "published native source changed: " + name)
    require(len(native_inventory) == public_custody["native_files"] == 658, "native file inventory count changed")

    # Only archived exchanges are needed. Guard accidental live paths as well.
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    subprocess.Popen = forbidden
    sys.path.insert(0, str(repo / "scripts"))
    judge = importlib.import_module("village_001f_judge")
    judge.call_api = forbidden
    judge.sprocket.execute = forbidden
    config = read(work / "judge-config.json")
    native_rows, public, sources = judge.load_sprocket(native, config)
    cases = {row["case_id"]: row for row in public["plan"]["cases"]}
    require(len(native_rows) == 36 and len(cases) == 6, "fixed population changed")
    rows = []
    for native_row in native_rows:
        try:
            obj = json.loads(native_row["raw_response"])
            json_valid = True
        except ValueError:
            obj, json_valid = None, False
        whole = json_valid and isinstance(obj, dict)
        row = {key: native_row[key] for key in ("id", "key", "case_id", "arm", "worker", "seed",
            "legacy_schema_valid", "strict_json_schema_valid", "empty_response", "completion_tokens", "finish_reason",
            "historical_raw_exact_match", "historical_tokens_match", "historical_finish_match")}
        row.update(whole_string_json_parse=json_valid, whole_string_json_object=whole,
            starts_with_open_brace=native_row["raw_response"].lstrip().startswith("{"),
            keep_items=len(obj["keep"]) if whole and isinstance(obj.get("keep"), list) else None,
            mutate_items=len(obj["mutate"]) if whole and isinstance(obj.get("mutate"), list) else None,
            extra_fields=sorted(set(obj) - FIELDS) if whole else None,
            source_row="inputs/sprocket/rows/" + row["key"] + ".json",
            source_row_sha256=sha(native / "rows" / (row["key"] + ".json")))
        rows.append(row)
    semantic_sources, final_receipt_sha, final_summary = {}, None, None
    if args.mode == "final":
        final_path = args.final_acceptance or work / "review/final-acceptance.json"
        final = accepted_report(final_path, final=True)
        require(all(final["input_release"][key] == accepted["input_release"][key]
                    for key in ("tag", "tag_commit", "verified_assets")), "final/input publication acceptance differs")
        final_receipt_sha = sha(final_path)
        results = work / "results"
        for name in ("summary.json", "judgments.json", "sprocket-grounding.json"):
            asset = final["final_release"]["verified_assets"][name]
            require(sha(results / name) == asset["sha256"] and (results / name).stat().st_size == asset["bytes"],
                    "accepted final JSON asset changed: " + name)
        identity = read(results / "provider-identity.json")
        judgments = read(results / "judgments.json")
        mapping = read(results / "private-mapping.json")
        blinded = read(results / "blinded-inputs.json")
        require(judgments["provider_identity_sha256"] == sha(results / "provider-identity.json"), "judgment identity changed")
        require(digest(canonical(mapping)) == identity["private_mapping_sha256"]
                and digest(canonical(blinded)) == identity["blinded_inputs_sha256"], "blinded source hashes changed")
        require(mapping["sprocket"] == public["mapping"] and blinded["sprocket"] == public["items"],
                "Sprocket blind identity/text differs from native inputs")
        ratings = judgments["ratings"]["sprocket"]
        ids = sorted(row["id"] for row in rows)
        require(sorted(ratings) == ids, "grounding review population differs")
        reviewed = judge.validate_reviews(json.dumps({"reviews": [{"id": ident, **ratings[ident]} for ident in ids]}),
            ids, kind="sprocket", items={row["id"]: row for row in public["items"]})
        require(reviewed == ratings, "native grounding validator changes archived ratings")
        final_summary = read(results / "sprocket-grounding.json")
        require(read(results / "summary.json")["sprocket"] == final_summary, "standalone/summary Sprocket results differ")
        by_id = {row["review_id"]: row for row in final_summary["reviews"]}
        require(sorted(by_id) == ids, "reported Sprocket review population differs")
        for row in rows:
            rating = ratings[row["id"]]
            grounded = rating["response_grounding"] >= 1 and rating["scenario_grounding"] >= 1 and not rating["unsupported_claims"]
            derived = {**rating, "grounded": grounded, "strict_and_grounded": row["strict_json_schema_valid"] and grounded,
                "strict_grounded_and_actionable": row["strict_json_schema_valid"] and grounded and rating["corrective_usefulness"] >= 1}
            reported = {"review_id": row["id"], **public["mapping"][row["id"]], **derived,
                "legacy_schema_valid": row["legacy_schema_valid"], "strict_json_schema_valid": row["strict_json_schema_valid"],
                "empty_response": row["empty_response"]}
            require(by_id[row["id"]] == reported, "reported per-row Sprocket derivation differs")
            row["rating"] = derived
        semantic_sources = {name: sha(results / name) for name in ("summary.json", "sprocket-grounding.json", "judgments.json",
                            "provider-identity.json", "private-mapping.json", "blinded-inputs.json")}

    arm_summary = {arm: counts([row for row in rows if row["arm"] == arm]) for arm in ARMS}
    require(all(arm_summary[arm]["responses"] == 18 for arm in ARMS), "arm population differs")
    matches = []
    for case in sorted(cases):
        for worker in (1, 2, 3):
            pair = {row["arm"]: row for row in rows if row["case_id"] == case and row["worker"] == worker}
            require(set(pair) == set(ARMS) and pair[ARMS[0]]["seed"] == pair[ARMS[1]]["seed"], "matched seed pair differs")
            record = {"case_id": case, "worker": worker, "seed": pair[ARMS[0]]["seed"],
                      "strict_syntax_delta": int(pair[ARMS[1]]["strict_json_schema_valid"]) - int(pair[ARMS[0]]["strict_json_schema_valid"])}
            if final_summary is not None:
                record.update(semantic_delta={key: pair[ARMS[1]]["rating"][key] - pair[ARMS[0]]["rating"][key] for key in SCORES},
                    strict_grounded_actionable_delta=int(pair[ARMS[1]]["rating"]["strict_grounded_and_actionable"])
                        - int(pair[ARMS[0]]["rating"]["strict_grounded_and_actionable"]))
            matches.append(record)
    classification = "pending_blinded_review"
    if final_summary is not None:
        for arm in ARMS:
            ours, reported = arm_summary[arm], final_summary["arms"][arm]
            for key in ("responses", "legacy_schema_valid", "strict_json_schema_valid", "empty_response"):
                require(ours[key] == reported[key], "reported arm count differs")
            for key in ("unsupported_claims", "grounded", "strict_and_grounded", "strict_grounded_and_actionable"):
                require(ours["semantic"][key] == reported[key], "reported semantic count differs")
            for key in SCORES:
                require(math.isclose(ours["semantic"]["means"][key], reported["means"][key], rel_tol=0, abs_tol=1e-12),
                        "reported semantic mean differs")
        syntax_gain = sum(row["strict_syntax_delta"] for row in matches)
        useful_gain = sum(row["strict_grounded_actionable_delta"] for row in matches)
        classification = "syntax_gain_without_grounded_actionable_gain" if syntax_gain > 0 and useful_gain <= 0 else (
            "grounded_actionable_count_gain" if useful_gain > 0 else "no_grounded_actionable_count_gain")
        require((syntax_gain, useful_gain, classification) == (final_summary["strict_syntax_count_gain"],
            final_summary["strict_grounded_actionable_count_gain"], final_summary["classification"]), "frozen classification differs")

    runtime = read(native / "runtime-001.json")
    plan = public["plan"]
    require(runtime["model_sha256"] == plan["model"]["sha256"]
            and runtime["runtime_archive_sha256"] == plan["runtime"]["archive_sha256"]
            and runtime["actual_template_sha256"] == plan["wire"]["gguf_chat_template_sha256"], "runtime receipt identity changed")
    report = {"schema": "village-001f-portable-sprocket-supplement-v1", "mode": args.mode, "descriptive_only": True,
        "analysis_script_sha256": sha(Path(__file__)), "public_custody_sha256": sha(custody_path),
        "input_acceptance_sha256": sha(acceptance_path), "final_acceptance_sha256": final_receipt_sha,
        "scientific_source_commit": FREEZE, "input_publication": public_custody["input_publication"],
        "native_files": len(native_inventory), "native_sha256": public_custody["native_sha256"],
        "native_source_reconstruction": "pass", "semantic_sources_sha256": semantic_sources,
        "semantic_status": "accepted_archived_blind_reviews" if final_summary is not None else "pending_blinded_review",
        "cases": 6, "seeds_per_case": 3, "matched_case_seed_pairs": 18, "responses": 36,
        "arms": arm_summary, "classification": classification, "matched_pairs": matches,
        "per_case": [{"case_id": case, "scenario_id": cases[case]["scenario_id"],
            "arms": {arm: counts([row for row in rows if row["case_id"] == case and row["arm"] == arm]) for arm in ARMS}}
            for case in sorted(cases)], "rows": rows,
        "runtime_receipt": {key: runtime[key] for key in ("server_version", "model_sha256", "runtime_archive_sha256",
                            "effective_context", "actual_template_sha256")},
        "native_prompt_token_range": [min(len(row["input_token_ids"]) for row in native_rows),
                                      max(len(row["input_token_ids"]) for row in native_rows)],
        "maximum_native_prompt_plus_output_budget": max(len(row["input_token_ids"]) + 160 for row in native_rows),
        "measurement_definition": "Frozen strict_json_schema_valid means the whole raw response parses as a JSON object and passes the historical field validator. Keep/mutate permit zero to three nonempty strings; extra fields are ignored. One-item/extra-field counts supplement and do not alter the frozen metric.",
        "limits": ["Six selected cases are task units; seeds are repeated observations within case.",
            "Syntax, grounding, unsupported claims, and corrective usefulness remain separate measurements.",
            "Token-limit hits do not identify the cause of malformed feedback.",
            "Historical raw equality is measured, not assumed, and this pilot is not a replay of all 3456 historical outputs.",
            "These prompt-arm outputs do not train workers or establish benefit to worker learning.",
            "Remote critic ratings are subjective and its backend is not version-pinned.",
            "This supplement checks accepted evidence; independent release/provider acceptance remains in the shared review receipts."],
        "new_model_generations": 0, "new_provider_calls": 0, "new_optimizer_steps": 0,
        "new_tokenizer_calls": 0, "new_model_forward_batches": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        stream.write(canonical(report))
    print(json.dumps({"mode": args.mode, "sha256": sha(out), "native_reconstruction": "pass",
                      "semantic_status": report["semantic_status"], "classification": classification}, indent=2))


if __name__ == "__main__":
    main()
