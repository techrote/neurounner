#!/usr/bin/env python3
"""Describe context-versus-paired ratings of identical F worker texts.

This is an after-the-fact calibration description, never a quality gate or a
causal estimate of presentation effects. Read saved native text and exact raw
integer ratings only; do not load a model, retokenize or call a provider.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

PANEL = ["V-G01-A", "V-G03-B", "V-G05-C", "V-G07-A", "V-G09-B", "V-G11-C"]
METRICS = ("task_relevance", "coherence", "genericness", "groundedness")
REL = Path("experiments/village-001/diagnostic-20261119")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), "Missing/nonregular source")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    sha(path)
    return json.loads(path.read_text())


def fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator, "value": float(value)}


def summarize(rows):
    require(rows, "Empty matched cohort")
    result = {"identical_response_pairs": len(rows), "metrics": {}}
    for metric in METRICS:
        deltas = [row["context_minus_paired_reproduced"][metric] for row in rows]
        result["metrics"][metric] = {
            "context_score_sum": sum(row["context_original_D"][metric] for row in rows),
            "paired_reproduced_score_sum": sum(row["paired_reproduced_D"][metric] for row in rows),
            "context_mean": fraction(Fraction(sum(row["context_original_D"][metric] for row in rows), len(rows))),
            "paired_reproduced_mean": fraction(Fraction(sum(row["paired_reproduced_D"][metric] for row in rows), len(rows))),
            "difference_sum": sum(deltas), "mean_difference": fraction(Fraction(sum(deltas), len(rows))),
            "mean_absolute_difference": fraction(Fraction(sum(map(abs, deltas)), len(rows))),
            "score_equal_pairs": sum(value == 0 for value in deltas),
            "context_higher_pairs": sum(value > 0 for value in deltas),
            "context_lower_pairs": sum(value < 0 for value in deltas),
            "difference_distribution": {str(k): v for k, v in sorted(Counter(deltas).items())}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists() and args.repo.resolve() not in args.out.resolve().parents, "Use a new scratch output path")
    root = args.repo / REL
    reconstructed = read(args.reconstruction)
    for name, expected in reconstructed["source_sha256"].items():
        require(sha(root / "results" / name) == expected, "Primary reconstruction source changed: " + name)
    require(reconstructed["case_pairs"] == 72 and reconstructed["member_replication_units"] == 6,
            "Complete frozen primary reconstruction required")
    pairs = {(row["member_id"], row["scenario_id"]): row for row in reconstructed["paired_cases"]}
    contexts = {(row["member_id"], row["scenario_id"]): row for row in reconstructed["context_cases"]
                if row["stage"] == "D_ORIGINAL"}
    require(len(pairs) == len(contexts) == 72 and pairs.keys() == contexts.keys(), "Context/paired population mismatch")
    inputs = root / "inputs"
    custody = read(root / "input-custody.json")
    source_rows, native = {}, {}
    for member in PANEL:
        for stage in ("D_ORIGINAL", "D_REPRODUCED"):
            paths = sorted((inputs / "worker" / ("village001f-infer-" + member) / stage / "rows").glob("*.json"))
            require(len(paths) == 12, "Complete native original/reproduced responses required")
            for path in paths:
                relative = path.relative_to(inputs).as_posix()
                require(sha(path) == custody["files"][relative]["sha256"]
                        and path.stat().st_size == custody["files"][relative]["bytes"], "Native response custody differs")
                checkpoint = read(path)
                require(isinstance(checkpoint.get("row"), dict), "Missing native checkpoint row payload")
                row = checkpoint["row"]
                key = (member, stage, row["scenario_id"])
                require(key not in native, "Duplicate native response")
                native[key] = row
                source_rows[key] = {"repository_path": path.relative_to(args.repo).as_posix(), "sha256": sha(path)}
    matched, excluded = [], []
    for key in sorted(pairs):
        member, scenario = key
        original = native[(member, "D_ORIGINAL", scenario)]
        reproduced = native[(member, "D_REPRODUCED", scenario)]
        require(original["prompt"] == reproduced["prompt"], "Native prompt differs between stages")
        primary, contextual = pairs[key], contexts[key]
        require(primary["family"] == contextual["family"] == original["family"] == reproduced["family"], "Joined case family differs")
        record = {"member_id": member, "scenario_id": scenario, "family": primary["family"],
            "context_review_id": contextual["review_id"], "paired_review_id": primary["review_id"],
            "native_original_D": source_rows[(member, "D_ORIGINAL", scenario)],
            "native_reproduced_D": source_rows[(member, "D_REPRODUCED", scenario)],
            "prompt_sha256": hashlib.sha256(original["prompt"].encode()).hexdigest(),
            "exact_prompt_match": True, "exact_response_text_match": original["response"] == reproduced["response"],
            "exact_generated_token_match": original["generated_token_ids"] == reproduced["generated_token_ids"]}
        if not record["exact_response_text_match"]:
            record["exclusion_reason"] = "Native original-D/reproduced-D text differs; excluded from identical-text description."
            excluded.append(record)
            continue
        require(record["exact_generated_token_match"], "Actual identical-text cohort unexpectedly differs in generated tokens")
        require(all(type(contextual[k]) is int and 0 <= contextual[k] <= 4
                    and type(primary["reproduced"][k]) is int and 0 <= primary["reproduced"][k] <= 4 for k in METRICS),
                "Expected raw integer ratings0..4")
        record.update({"response_sha256": hashlib.sha256(original["response"].encode()).hexdigest(),
            "context_original_D": {k: contextual[k] for k in METRICS},
            "paired_reproduced_D": primary["reproduced"],
            "context_minus_paired_reproduced": {k: contextual[k] - primary["reproduced"][k] for k in METRICS}})
        matched.append(record)
    published = read(root / "results/summary.json")
    require(len(matched) == published["worker"]["original_D_reproduced_exact_responses"], "Matched count differs from published native comparison")
    overall = summarize(matched)
    report = {"schema": "village-001f-identical-text-rating-context-description-v1",
        "analysis_script_sha256": sha(Path(__file__)), "primary_reconstruction_sha256": sha(args.reconstruction),
        "input_custody_sha256": sha(root / "input-custody.json"), "final_source_sha256": reconstructed["source_sha256"],
        "direction": "Individually rated context original-D minus paired-review historical-mask reproduced-D, for the same prompt and exactly identical response text and generated tokens.",
        "candidate_member_case_pairs": 72, "member_units": 6, "overall": overall,
        "per_member": [{"member_id": member, **summarize([row for row in matched if row["member_id"] == member])} for member in PANEL],
        "matched_cases": matched, "excluded_nonidentical_cases": excluded,
        "interpretation_limits": ["This is a descriptive calibration check requested after inspecting final results. It is not a new gate, adjusted primary estimate or significance analysis.",
            "Identical text received ratings in different individual and paired requests, batches and judging contexts; the paired request also contained a comparison response.",
            "The remote backend is not version-pinned and may be stochastic. This observational contrast does not isolate a causal effect of presentation, order, batch or backend variation.",
            "Contextual BASE/C/original-D score levels must not be joined with paired-arm levels as a single causal training trajectory.",
            "Genericness uses the frozen0..4 score; its direction is distinct from relevance/coherence/groundedness. These score differences describe calibration, not text-quality changes.",
            "The71 case pairs are within six members and repeated prompts, not71 independent model replications."],
        "raw_evidence_modified": False, "frozen_primary_result_modified": False, "new_model_tokenizer_or_provider_calls": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.out), "sha256": sha(args.out), "overall": overall,
        "excluded": [{k: row[k] for k in ("member_id", "scenario_id")} for row in excluded]}, indent=2))


if __name__ == "__main__":
    main()
