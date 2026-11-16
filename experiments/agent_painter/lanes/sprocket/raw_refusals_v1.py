#!/usr/bin/env python3
"""Source-only triage of immutable first-pass SPROCKET DEVELOPMENT generations.

Never read FINAL, regenerate tokens, repair completions, train, or claim native Art.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

CORPUS = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
NATIVE = "633920373a3d406a2ede61ce4137c324f76be23b"
MODEL = "kandivault/sprocket-500m"
FAMILIES = {"greebles", "pipework", "architecture"}
SEEDS = {104729, 130363}
PIN = {
    "early": {"run": 43523253619, "artifact": 14083354475, "sha256": "686f67190766d71667b8e78c9a58bdcd121070eff1b75bad21327ac63f3d6b2a"},
    "late": {"run": 43536717867, "artifact": 14089867633, "sha256": "991d2f2c0a578d935c839d5e5b4650ad8b9b9116348a54e873340f59f115ff23"}
}

def require(value, reason):
    if not value:
        raise ValueError(reason)

def digest(s):
    return sha256(s.encode("utf-8")).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def _strict_json_valid(text):
    """Derive JSON validity from preserved raw text; refuse duplicate keys/NaN/Infinity."""
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON object key")
            obj[key] = value
        return obj

    def reject_constant(value):
        raise ValueError("non-finite JSON constant: " + value)

    try:
        json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return False
    return True


def classify(row):
    raw = row["raw"]
    ids = row["output_token_ids"]
    require(type(ids) is list and ids and all(type(i) is int and i >= 0 for i in ids), "missing exact token IDs")
    require(ids == raw["token_ids"] and len(ids) == row["output_tokens"], "raw/generated token mismatch")
    require(type(row["prompt"]) is str and bool(row["prompt"]) and type(raw["raw_text"]) is str,
            "missing unchanged prompt/raw text")
    for k in ("text_decoded", "eos_terminated", "json_valid", "program_valid", "first_pass_accepted"):
        require(type(raw[k]) is bool, "invalid strict first-pass flag " + k)
    expected_eos = ids[-1] == 5 and 5 not in ids[:-1]
    require(raw["eos_terminated"] == expected_eos,
            "terminal EOS flag/token mismatch")
    require(raw["json_valid"] == _strict_json_valid(raw["raw_text"]),
            "strict JSON flag/raw text mismatch")
    require(not raw["first_pass_accepted"] or
            (raw["eos_terminated"] and raw["json_valid"] and raw["program_valid"]),
            "first-pass assertion without EOS, JSON and schema")
    require(not raw["program_valid"] or raw["json_valid"], "schema claimed without JSON")
    views = row["seed_results"]
    require(type(views) is list and len(views) == 2 and {x["seed"] for x in views} == SEEDS,
            "missing/duplicate original seed variant")
    for v in views:
        require(type(v["first_pass_compiler_accepted"]) is bool and
                v["native_execution"] is False and v["native_acknowledged_cells"] is None and
                v["native_png"] is None and v["visual_quality_claim"] is False,
                "compiler-only candidate misrepresented as native result")
    compiled = sum(v["first_pass_compiler_accepted"] for v in views)
    require(compiled == row["admitted_seed_programs"] and
            (compiled == 0 or raw["first_pass_accepted"]), "unqualified compiler acceptance")
    if not raw["text_decoded"]:
        category = "decode_refusal"
    elif not raw["eos_terminated"]:
        category = "no_terminal_eos"
    elif not raw["json_valid"]:
        category = "invalid_json"
    elif not raw["program_valid"]:
        category = "invalid_program"
    elif compiled < 2:
        category = "compile_refusal"
    else:
        category = "compiled_not_semantic_or_native_pass"
    return {
        "family": row["family"], "group": row["structural_group"],
        "prompt_sha256": digest(row["prompt"]), "raw_text_sha256": digest(raw["raw_text"]),
        "token_ids_sha256": digest(json.dumps(ids, separators=(",", ":"))),
        "raw_tokens": len(ids), "eos": raw["eos_terminated"], "json": raw["json_valid"],
        "schema_valid": raw["program_valid"], "first_pass": raw["first_pass_accepted"],
        "compiler_seed_passes": compiled, "category": category, "native_paint": False
    }

def checked(rows, arm):
    require(type(rows) is list and len(rows) == 6, "six independent DEVELOPMENT requests required")
    require(all(x["arm"] == arm for x in rows), "arm mislabel")
    require(len({x["structural_group"] for x in rows}) == 6, "duplicate structural group")
    require({x["family"] for x in rows} == FAMILIES, "required three families absent")
    return {x["structural_group"]: classify(x) for x in rows}

def analyze(early, late):
    arms = {
        "untuned_step16_run": checked(early["untuned"], "untuned"),
        "step16": checked(early["step16"], "step16"),
        "untuned_step96_run": checked(late["untuned"], "untuned"),
        "step96": checked(late["step96"], "step96")
    }
    groups = set(arms["untuned_step16_run"])
    require(all(set(x) == groups for x in arms.values()), "changed DEVELOPMENT structural groups")
    for g in groups:
        identities = {(a[g]["family"], a[g]["prompt_sha256"]) for a in arms.values()}
        require(len(identities) == 1, "different DEVELOPMENT prompt/family across source runs")
    summary = {}
    for arm, rows in arms.items():
        values = list(rows.values())
        summary[arm] = {
            "requests": 6, "seed_variants": 12,
            "eos_requests": sum(v["eos"] for v in values),
            "json_requests": sum(v["json"] for v in values),
            "program_valid_requests": sum(v["schema_valid"] for v in values),
            "first_pass_requests": sum(v["first_pass"] for v in values),
            "compiled_seed_variants": sum(v["compiler_seed_passes"] for v in values),
            "category_counts": dict(sorted(Counter(v["category"] for v in values).items())),
            "per_family": {f: {"requests": sum(v["family"] == f for v in values),
                               "compiler_passed_requests": sum(v["family"] == f and
                                                               v["compiler_seed_passes"] == 2 for v in values)}
                           for f in sorted(FAMILIES)},
            "native_paintings": 0, "semantic_success": "NOT_EVALUATED"
        }
    return {
        "schema": "cybersand.sprocket-raw-refusal-taxonomy/1",
        "status": "RAW_DEVELOPMENT_ONLY_NO_NATIVE_NO_TRAIN",
        "original_artifacts": PIN, "native_source_sha": NATIVE,
        "corpus_manifest_sha256": CORPUS, "pinned_model": MODEL,
        "unique_requests": 6, "arms": summary,
        "untuned_cross_run_identical_token_sequences": sum(
            arms["untuned_step16_run"][g]["token_ids_sha256"] ==
            arms["untuned_step96_run"][g]["token_ids_sha256"] for g in groups),
        "untuned_cross_run_denominator": 6,
        "raw_records": [{"group": g, "by_arm": {k: v[g] for k, v in arms.items()}}
                        for g in sorted(groups)],
        "new_model_inference": False, "new_optimizer_updates": 0,
        "new_native_paintings": 0, "new_train_groups": 0, "final_opened": False,
        "caveat": "First-pass syntax and compiler evidence is neither request conformance nor native painting."
    }

def bundle(root, stage):
    root = Path(root)
    names = ("untuned", "step16" if stage == "early" else "step96")
    result = {}
    for arm in names:
        matches = list(root.rglob(arm + "-raw-development.json"))
        require(len(matches) == 1, "missing/ambiguous original raw arm: " + arm)
        result[arm] = read(matches[0])
    matches = list(root.rglob("comparison.json"))
    require(len(matches) == 1, "missing/ambiguous original comparison receipt")
    comp = read(matches[0])
    require(comp["model"] == MODEL and comp["native_source"] == NATIVE and
            comp["corpus_manifest_sha256"] == CORPUS and
            comp["max_new_tokens"] == 512 and comp["final_targets_opened"] is False,
            "original data/source/model/FINAL contract mismatch")
    if stage == "early":
        require(comp["schema"] == "cybersand.sprocket-development-first-pass/1" and
                comp["trainer_source"] == "591b83bc383b31a0ae2e3031b4a501b0a88ddc39",
                "historical step16 source differs")
    else:
        require(comp["schema"] == "cybersand.sprocket-development-first-pass/2" and
                comp["trainer_source"] == "25b24bf80c593b115321387a2a0374dd5ff02548" and
                comp["checkpoint_step"] == comp["actual_optimizer_steps"] == 96 and
                comp["native_execution"] is False,
                "historical step96 source or training receipt differs")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refuse overwrite of evidence")
    report = analyze(bundle(args.early, "early"), bundle(args.late, "late"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "arms": report["arms"],
                      "untuned_equal": report["untuned_cross_run_identical_token_sequences"]},
                     sort_keys=True))

if __name__ == "__main__":
    main()
