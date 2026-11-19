#!/usr/bin/env python3
"""Offline F worker interpretation from complete published native evidence.

Native read-only validators check execution/source custody. Mask arithmetic,
fixed-forward differences, adapter/response comparisons, blind orientation and
primary effect arithmetic are recomputed here without calling the model or the
production aggregation function. This supplements the separate release/weight
acceptance harness; it does not authenticate GitHub releases or reread tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from audit_native_masks import PANEL, REL, audit, read, require, sha

ARMS = ("historical_token_id_mask", "length_derived_mask")
STAGES = ("C", "D_ORIGINAL", "D_REPRODUCED", "D_CORRECTED")
METRICS = ("task_relevance", "coherence", "genericness", "groundedness")
PRELIMINARY_AUDIT_SHA = "3566f8a5279ba5acf367f019fbdb23c32cbad611a5942e8b159d655f1d8c82f5"


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def exact_mean(values):
    values = list(values)
    require(values, "Cannot average an empty population")
    return Fraction(sum(values), len(values))


def fraction_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator, "value": float(value)}


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-12), label)


def verify_inventory(root, inventory):
    require(isinstance(inventory, dict) and inventory, "Missing source inventory")
    physical = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    require(set(physical) == set(inventory), "Published input inventory has missing/extra files")
    for name, path in physical.items():
        require(not path.is_symlink() and ".." not in Path(name).parts, "Unsafe published input path")
        require(path.stat().st_size == inventory[name]["bytes"] and sha(path) == inventory[name]["sha256"], "Published input bytes changed: " + name)


def fingerprints(left, right):
    for fp in (left, right):
        digest = hashlib.sha256()
        require(len(fp["tensors"]) == fp["tensor_count"], "Tensor fingerprint count differs")
        for name, row in sorted(fp["tensors"].items()):
            digest.update(canonical({"name": name, **row}))
        require(digest.hexdigest() == fp["canonical_tensor_sha256"], "Tensor fingerprint framing mismatch")
    names = sorted(set(left["tensors"]) | set(right["tensors"]))
    changed = [name for name in names if left["tensors"].get(name) != right["tensors"].get(name)]
    return {"canonical_tensor_hash_equal": left["canonical_tensor_sha256"] == right["canonical_tensor_sha256"],
        "safetensors_file_hash_equal": left["safetensors_file_sha256"] == right["safetensors_file_sha256"],
        "union_tensor_count": len(names), "changed_tensor_count": len(changed), "changed_tensor_names": changed,
        "weight_bytes_independently_verified_here": False}


def fixed_forward(inputs, audited, plan):
    panel = {row["member_id"]: row for row in plan["panel"]}
    selected = {member: row["forward_only_pair_ids"] for member, row in panel.items()}
    files = sorted((inputs / "worker").rglob("score.json"))
    require(len(files) == 18, "All18 fixed-forward lanes are required")
    seen, lanes = set(), []
    for path in files:
        doc = read(path)
        member, stage = doc["identity"]["member_id"], doc["identity"]["stage"]
        key = (member, stage)
        require(member in PANEL and stage in ("BASE", "C", "D_ORIGINAL") and key not in seen, "Unexpected/duplicate fixed-forward lane")
        seen.add(key)
        expected_identity = {**plan["identity"], "kind": "forward-only-paired-mask-scores", "member_id": member, "stage": stage,
            "adapter_directory_sha256": None if stage == "BASE" else panel[member]["C_adapter_sha256" if stage == "C" else "original_D_adapter_sha256"]}
        require(doc["identity"] == expected_identity and doc["pair_dataset_sha256"] == panel[member]["pair_dataset_sha256"],
                "Fixed-forward source/model/pair identity changed")
        require(doc["model_forward_batches"] == 16 and doc["optimizer_steps"] == doc["new_generations"] == 0,
                "Fixed-forward workload changed")
        require([row["pair_id"] for row in doc["rows"]] == selected[member] and len(doc["rows"]) == 8, "Forward pair selection changed")
        differences = []
        for row in doc["rows"]:
            sequences = [audited[(member, row["pair_id"], choice)] for choice in ("chosen", "rejected")]
            lengths = [len(sequence["input_token_ids"]) for sequence in sequences]
            width = max(lengths)
            ids = [sequence["input_token_ids"] + [2] * (width - n) for sequence, n in zip(sequences, lengths)]
            labels = [sequence["labels"] + [-100] * (width - n) for sequence, n in zip(sequences, lengths)]
            require(row["training_diagnostic"] is True and row["input_ids"] == ids and row["labels"] == labels,
                    "Forward native IDs/labels differ from tokenizer audit")
            require(row["historical_attention_mask"] == [[int(token != 2) for token in seq] for seq in ids]
                    and row["length_attention_mask"] == [[int(i < n) for i in range(width)] for n in lengths],
                    "Forward mask construction differs")
            require(row["real_positions_wrongly_masked"] == [[i for i in range(n) if ids[j][i] == 2] for j, n in enumerate(lengths)],
                    "Forward wrongly masked positions differ")
            values = row["scores"]
            require(set(values) == set(ARMS), "Both fixed masks required")
            for score in values.values():
                require(set(score) == {"chosen_mean_logp", "rejected_mean_logp", "margin"}
                        and all(type(v) in (float, int) and math.isfinite(v) for v in score.values()), "Invalid fixed-forward score")
                require(score["chosen_mean_logp"] <= 1e-6 and score["rejected_mean_logp"] <= 1e-6
                        and math.isclose(score["margin"], score["chosen_mean_logp"] - score["rejected_mean_logp"], rel_tol=1e-5, abs_tol=1e-6),
                        "Forward log-probability arithmetic differs")
            differences.append({"pair_id": row["pair_id"], "scenario_id": row["scenario_id"],
                "historical": values[ARMS[0]], "length_derived": values[ARMS[1]],
                "length_minus_historical": {k: values[ARMS[1]][k] - values[ARMS[0]][k] for k in values[ARMS[0]]},
                "positive_margin_flipped": (values[ARMS[0]]["margin"] > 0) != (values[ARMS[1]]["margin"] > 0)})
        lanes.append({"member_id": member, "stage": stage, "pairs": 8, "model_forward_batches": 16,
            "source_sha256": sha(path), "mean_length_minus_historical": {k: math.fsum(r["length_minus_historical"][k] for r in differences) / 8
                for k in ("chosen_mean_logp", "rejected_mean_logp", "margin")},
            "positive_margin_flips": sum(r["positive_margin_flipped"] for r in differences), "pair_differences": differences})
    require(seen == {(m, s) for m in PANEL for s in ("BASE", "C", "D_ORIGINAL")}, "Forward population incomplete")
    return {"lanes": 18, "pair_rows": 144, "model_forward_batches": 288, "optimizer_steps": 0, "generated_responses": 0,
        "interpretation": "Same fixed weights, IDs and labels; observed likelihood changes isolate mask semantics on existing training diagnostics, not fresh-task generalization.",
        "per_lane": lanes}


def input_analysis(repo, froot):
    inputs = froot / "inputs"
    custody = read(froot / "input-custody.json")
    verify_inventory(inputs, custody["files"])
    for name, expected in custody["contract_sha256"].items():
        require(sha(froot / name) == expected, "Published scientific contract changed: " + name)
    sys.path.insert(0, str(repo / "scripts"))
    import village_001f_judge as native
    jcfg, cfg, suite, campaign, pref = native.load_contract(froot / "judge-config.json")
    training, _ = native.load_training(inputs / "training", jcfg, cfg, campaign, pref)
    workers, _ = native.load_workers(inputs / "worker", training, jcfg, cfg, suite)
    require(len(training) == 12 and len(workers) == 25 and sum(len(x["rows"]) for x in workers.values()) == 300,
            "Matched pilot inputs incomplete")
    audit_paths = list((inputs / "token-audit").rglob("token-mask-audit.json"))
    require(len(audit_paths) == 1 and sha(audit_paths[0]) == PRELIMINARY_AUDIT_SHA, "Published token audit differs from preserved preliminary bytes")
    mask, indexed_audit = audit(repo, audit_paths[0])
    plan = read(froot / "worker-plan.json")
    forward = fixed_forward(inputs, indexed_audit, plan)
    native_prompts, generation, trajectories, reproduction = {}, [], [], []
    for (member, stage), lane in workers.items():
        for row in lane["rows"]:
            signature = (row["prompt"], row["rendered_prompt"], tuple(row["input_token_ids"]), tuple(row["attention_mask"]))
            require(row["attention_mask"] == [1] * len(row["input_token_ids"]), "Q inference used a non-native padding mask")
            require(native_prompts.setdefault(row["scenario_id"], signature) == signature, "Q prompt/token wire differs across model stages")
        generation.append({"member_id": member, "stage": stage, "responses": len(lane["rows"]),
            "empty_responses": sum(not row["response"].strip() for row in lane["rows"]),
            "generated_tokens": sum(len(row["generated_token_ids"]) for row in lane["rows"]),
            "token_limit_without_terminal_eos": sum(len(row["generated_token_ids"]) == 160 and row["generated_token_ids"][-1] != 2 for row in lane["rows"])})
    for (member, stage), lane in training.items():
        provenance = lane["execution_provenance"]
        known = sum(segment["end_step"] - segment["start_step"] for segment in provenance["optimizer_segments"])
        unknown = sum(segment["end_step"] - segment["start_step"] for segment in provenance["unattributed_optimizer_segments"])
        require(known + unknown == 120 and lane["result"]["training"]["optimizer_steps"] == 120, "Retained optimizer trajectory mismatch")
        trajectories.append({"member_id": member, "stage": stage, "retained_steps": known + unknown,
            "attributed_steps": known, "unattributed_steps": unknown, "zero_step_copy_invocations": provenance["zero_step_copies"],
            "resume_invocations": provenance["resume_invocations"], "optimizer_segments": provenance["optimizer_segments"],
            "unattributed_optimizer_segments": provenance["unattributed_optimizer_segments"]})
    require(sum(r["retained_steps"] for r in trajectories) == 1440, "Total retained optimizer count differs")
    for member in PANEL:
        original, reproduced, corrected = [workers[(member, s)] for s in ("D_ORIGINAL", "D_REPRODUCED", "D_CORRECTED")]
        by_case = [{r["scenario_id"]: r for r in lane["rows"]} for lane in (original, reproduced, corrected)]
        comparisons = {}
        for label, left, right in (("original_D_vs_reproduced_D", 0, 1), ("reproduced_D_vs_corrected_D", 1, 2)):
            a, b = (original, reproduced, corrected)[left], (original, reproduced, corrected)[right]
            comparisons[label] = {**fingerprints(a["result"]["adapter_weight_fingerprint"], b["result"]["adapter_weight_fingerprint"]),
                "adapter_directory_hash_equal": a["result"]["adapter_directory_sha256"] == b["result"]["adapter_directory_sha256"],
                "exact_text_matches": sum(by_case[left][sid]["response"] == by_case[right][sid]["response"] for sid in by_case[left]),
                "exact_generated_token_matches": sum(by_case[left][sid]["generated_token_ids"] == by_case[right][sid]["generated_token_ids"] for sid in by_case[left]),
                "cases": 12}
        reproduction.append({"member_id": member, **comparisons})
    report = {"schema": "village-001f-independent-worker-input-analysis-v1", "input_custody_sha256": sha(froot / "input-custody.json"),
        "source_head": custody["source_head"], "native_read_only_validators": ["load_training", "load_workers"],
        "workload": {"training_lanes": 12, "retained_optimizer_steps": 1440, "response_lanes": 25, "responses": 300,
            "mask_sequences": 720, "fixed_forward_batches": 288}, "mask": mask, "fixed_forward": forward,
        "training_trajectories": trajectories, "native_generation": generation, "reproduction": reproduction,
        "native_Q_prompt_wire_identical_across_all_25_cohorts": True,
        "limits": ["Adapter fingerprints are compared from hash-bound native receipts; separate acceptance rereads the actual weight bytes.",
                   "Retained optimizer steps exclude any lost uncheckpointed work; completed-copy invocations contribute zero steps.",
                   "No fresh task-quality conclusion is available until complete blinded judgments exist."], "new_model_or_provider_calls": 0}
    return report, cfg, suite, training, workers


def final_analysis(results, cfg, suite, workers):
    judgments, mapping, blinded, summary = [read(results / name) for name in
        ("judgments.json", "private-mapping.json", "blinded-inputs.json", "summary.json")]
    identity = read(results / "provider-identity.json")
    for name, value in (("private_mapping", mapping), ("blinded_inputs", blinded)):
        require(hashlib.sha256(canonical(value)).hexdigest() == identity[name + "_sha256"], "Final blind-source hash differs")
    require(judgments["provider_identity_sha256"] == sha(results / "provider-identity.json"), "Judgment provider identity differs")
    ratings = judgments["ratings"]["primary"]
    require(len(ratings) == 72 and len(mapping["primary"]) == 72 and len(blinded["primary"]) == 72, "Complete primary72 required")
    source = {(m, s): {r["scenario_id"]: r for r in lane["rows"]} for (m, s), lane in workers.items()}
    by_id = {row["id"]: row for row in blinded["primary"]}
    ordering = sorted(((m, row["id"]) for m in PANEL for row in suite["scenarios"]),
        key=lambda item: hashlib.sha256(("F-PRIMARY-ORDER:" + ":".join(item)).encode()).hexdigest())
    pairs = []
    for index, (member, sid) in enumerate(ordering, 1):
        ident = f"P{index:03d}"
        first = hashlib.sha256(f"F-PAIR-ORIENTATION:{member}:{sid}".encode()).digest()[0] % 2 == 0
        a, b = ("D_CORRECTED", "D_REPRODUCED") if first else ("D_REPRODUCED", "D_CORRECTED")
        left, right = source[(member, a)][sid], source[(member, b)][sid]
        private = {"member_id": member, "scenario_id": sid, "family": left["family"], "A": a, "B": b}
        require(mapping["primary"][ident] == private, "Primary blind orientation/case identity differs")
        require(by_id[ident] == {"id": ident, "situation": left["prompt"], "A": left["response"], "B": right["response"]},
                "Blinded primary text differs from native source")
        rating = ratings[ident]
        require(rating["winner"] in ("A", "B", "tie"), "Invalid paired winner")
        for side in ("A", "B"):
            require(set(rating[side]) == set(METRICS) and all(type(v) is int and 0 <= v <= 4 for v in rating[side].values()), "Primary ratings must be integer0..4")
        same = left["response"] == right["response"]
        require(not same or (rating["A"] == rating["B"] and rating["winner"] == "tie"), "Identical outputs received differing ratings")
        corrected, reproduced = (rating["A"], rating["B"]) if first else (rating["B"], rating["A"])
        pairs.append({"review_id": ident, "member_id": member, "scenario_id": sid, "family": left["family"],
            "corrected": corrected, "reproduced": reproduced,
            "delta": {k: corrected[k] - reproduced[k] for k in METRICS},
            "winner": "tie" if rating["winner"] == "tie" else private[rating["winner"]],
            "identical_response_text": same, "reason": rating["reason"]})
    pairs.sort(key=lambda r: (r["member_id"], r["scenario_id"]))
    means = {k: exact_mean(row["delta"][k] for row in pairs) for k in METRICS}
    member_means, members = [], []
    for member in PANEL:
        rows = [row for row in pairs if row["member_id"] == member]
        require(len(rows) == 12, "Each prespecified member needs all12 cases")
        delta = {k: exact_mean(row["delta"][k] for row in rows) for k in METRICS}
        member_means.append(delta["task_relevance"])
        members.append({"member_id": member, "cases": 12, "mean_delta": {k: fraction_record(v) for k, v in delta.items()},
            "corrected_mean": {k: fraction_record(exact_mean(row["corrected"][k] for row in rows)) for k in METRICS},
            "reproduced_mean": {k: fraction_record(exact_mean(row["reproduced"][k] for row in rows)) for k in METRICS},
            "preference_winners": dict(Counter(row["winner"] for row in rows))})
    require(sum(member_means, Fraction()) / 6 == means["task_relevance"], "Equal-member and all72 integer estimates differ")
    thresholds = cfg["primary_analysis"]
    require((thresholds["working_effect_threshold"], thresholds["minimum_members_with_positive_mean_gain"],
             thresholds["maximum_member_mean_relevance_regression"]) == (0.5, 4, 0.5), "Frozen primary thresholds changed")
    criteria = {"mean_gain": means["task_relevance"] >= Fraction("0.5"),
        "members_positive": sum(x > 0 for x in member_means) >= 4,
        "maximum_member_regression": min(member_means) >= -Fraction("0.5")}
    classification = "working_effect_criteria_met" if all(criteria.values()) else (
        "member_harm_exceeds_limit" if not criteria["maximum_member_regression"] else "working_effect_criteria_not_met")
    published = summary["worker"]
    require(published["paired_cases"] == pairs, "Published per-case primary reconstruction differs")
    require(published["working_effect_criteria"] == criteria and published["working_effect_signal"] is all(criteria.values())
            and published["classification"] == classification, "Published primary decision differs")
    require(published["positive_member_relevance_gains"] == sum(x > 0 for x in member_means), "Positive-member count differs")
    close(published["worst_member_relevance_delta"], min(member_means), "Worst-member mean differs")
    for key, value in means.items():
        close(published["mean_delta"][key], value, "Published primary mean differs: " + key)
    for member in members:
        original = next(row for row in published["per_member"] if row["member_id"] == member["member_id"])
        for key in METRICS:
            close(original["mean_delta"][key], member["mean_delta"][key]["value"], "Published member delta differs")
    context_order = [(m, stage, row["id"]) for m, stage in workers if stage in ("BASE", "C", "D_ORIGINAL") for row in suite["scenarios"]]
    context_order.sort(key=lambda item: hashlib.sha256(("F-CONTEXT-ORDER:" + ":".join(x or "BASE" for x in item)).encode()).hexdigest())
    context_inputs = {row["id"]: row for row in blinded["context"]}
    require(len(context_order) == len(context_inputs) == len(mapping["context"]) == len(judgments["ratings"]["context"]) == 156,
            "Complete context156 required")
    context = []
    for index, (member, stage, sid) in enumerate(context_order, 1):
        ident, row = f"R{index:03d}", source[(member, stage)][sid]
        private = {"member_id": member, "stage": stage, "scenario_id": sid, "family": row["family"]}
        require(mapping["context"][ident] == private and context_inputs[ident] == {
            "id": ident, "situation": row["prompt"], "response": row["response"]}, "Context mapping/native text differs")
        rating = judgments["ratings"]["context"][ident]
        require(all(type(rating[k]) is int and 0 <= rating[k] <= 4 for k in METRICS), "Invalid contextual integer scores")
        context.append({"review_id": ident, **private, **rating})
    context_summary = {}
    for stage, expected_count in (("BASE", 12), ("C", 72), ("D_ORIGINAL", 72)):
        rows = [row for row in context if row["stage"] == stage]
        require(len(rows) == expected_count, "Context stage count differs")
        values = {k: exact_mean(row[k] for row in rows) for k in METRICS}
        require(published["context_summary"][stage]["responses"] == expected_count, "Published context denominator differs")
        for k, value in values.items():
            close(published["context_summary"][stage]["means"][k], value, "Published context mean differs")
        context_summary[stage] = {"responses": expected_count, "means": {k: fraction_record(v) for k, v in values.items()}}
    return {"schema": "village-001f-independent-primary-effect-reconstruction-v1", "source_sha256": {
        name: sha(results / name) for name in ("judgments.json", "private-mapping.json", "blinded-inputs.json", "summary.json", "provider-identity.json")},
        "case_pairs": 72, "member_replication_units": 6, "mean_delta": {k: fraction_record(v) for k, v in means.items()},
        "positive_member_relevance_gains": sum(x > 0 for x in member_means), "worst_member_relevance_delta": fraction_record(min(member_means)),
        "working_effect_criteria": criteria, "classification": classification, "per_member": members, "paired_cases": pairs,
        "context_summary": context_summary, "context_cases": context,
        "arithmetic": "Exact integer score deltas and fractions; 72case mean equals six equally weighted member means.",
        "production_aggregate_called": False, "context_vs_paired_score_level_contrasts_are_causal": False,
        "automatic_rollout": False, "new_model_or_provider_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--f-root", type=Path, help="Defaults to the repository's frozen F diagnostic directory")
    parser.add_argument("--phase", choices=("inputs", "final"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    froot = args.f_root or args.repo / REL
    require(not args.out.exists() and args.repo.resolve() not in args.out.resolve().parents, "Use a new scratch analysis directory")
    report, cfg, suite, training, workers = input_analysis(args.repo, froot)
    final = final_analysis(froot / "results", cfg, suite, workers) if args.phase == "final" else None
    args.out.mkdir(parents=True)
    for name, value in (("worker-input-analysis.json", report), ("worker-primary-reconstruction.json", final)):
        if value is not None:
            (args.out / name).write_bytes(canonical(value))
    receipt = {"phase": args.phase, "script_sha256": sha(Path(__file__)), "mask_audit_script_sha256": sha(Path(__file__).with_name("audit_native_masks.py")),
        "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in args.out.iterdir()}, "new_model_or_provider_calls": 0}
    (args.out / "analysis-receipt.json").write_bytes(canonical(receipt))
    print(json.dumps({"out": str(args.out), "workload": report["workload"], "mask_totals": report["mask"]["totals"],
        "primary": {k: final[k] for k in ("mean_delta", "working_effect_criteria", "classification")} if final else "not_evaluated"}, indent=2))


if __name__ == "__main__":
    main()
