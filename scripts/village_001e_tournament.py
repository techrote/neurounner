#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from babble_002_expand import extract_json, packet_for
from village_001_manifest import build_manifest
from village_001b_generate_member import call_api

ROOT = Path(__file__).resolve().parents[1]
LINEAGES = [f"G{i:02d}" for i in range(1, 13)]
SIBLINGS = ["A", "B", "C"]
SPECIALIZATION_GUIDANCE = {
    "A": (
        "failure-hardened: characteristic failure and awkward recovery, conflicting "
        "motives, physical reactions, stubborn imperfect settling"
    ),
    "B": (
        "local tinkerer: bounded local memory, mundane habits, maintenance choices "
        "and practical physical interaction"
    ),
    "C": (
        "social scavenger: bargaining, opportunism, social friction, embodied "
        "decisions and conflicting motives"
    ),
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return obj


def load_contract(config_path: Path) -> tuple[dict, dict, dict]:
    final_cfg = load_json(config_path)
    if final_cfg.get("experiment") != "VILLAGE-001E" or final_cfg.get("issue") != 14:
        raise RuntimeError("unexpected final evaluation config identity")
    campaign_path = ROOT / final_cfg["campaign_config"]
    campaign = load_json(campaign_path)
    population = build_manifest(campaign)
    if len(population["members"]) != 36:
        raise RuntimeError("VILLAGE-001 population must contain exactly 36 members")
    return final_cfg, campaign, population


def load_results(final_cfg: dict, population: dict) -> tuple[dict[str, dict], dict]:
    root = ROOT / final_cfg["preference_evidence_root"]
    results: dict[str, dict] = {}
    for member in population["members"]:
        member_id = member["member_id"]
        path = root / "results" / f"{member_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing VILLAGE-001D result {path}")
        obj = load_json(path)
        checks = {
            "experiment": "VILLAGE-001D",
            "member_id": member_id,
            "lineage": member["lineage"],
            "sibling": member["sibling"],
            "member_contract_sha256": population["member_contract_sha256"],
        }
        for key, expected in checks.items():
            if obj.get(key) != expected:
                raise RuntimeError(
                    f"{member_id}: result {key} {obj.get(key)!r} != {expected!r}"
                )
        if (obj.get("evaluation") or {}).get("hidden_member_prompt_used") is not False:
            raise RuntimeError(f"{member_id}: held-out result used hidden member prompt")
        responses = (obj.get("evaluation") or {}).get("responses")
        if not isinstance(responses, list) or len(responses) != 24:
            raise RuntimeError(f"{member_id}: expected 24 held-out responses")
        results[member_id] = obj

    base_path = ROOT / "experiments/persona-001/evidence/models/BASE.json"
    if not base_path.is_file():
        raise RuntimeError("PERSONA-001 untuned BASE control result is missing")
    base = load_json(base_path)
    if base.get("persona") != "BASE" or base.get("base_only") is not True:
        raise RuntimeError("unexpected untuned BASE control identity")
    return results, base


def scenario_table(results: dict[str, dict], base: dict) -> list[dict]:
    first = next(iter(results.values()))
    reference = first["evaluation"]["responses"]
    expected_ids = [f"T{i:02d}" for i in range(1, 25)]
    if [row["scenario_id"] for row in reference] != expected_ids:
        raise RuntimeError("held-out scenario IDs are not canonical T01..T24")

    table = []
    for ref in reference:
        sid = ref["scenario_id"]
        prompt = ref["prompt"]
        family = ref["family"]
        member_responses = {}
        for member_id, result in results.items():
            matches = [
                row for row in result["evaluation"]["responses"]
                if row["scenario_id"] == sid
            ]
            if len(matches) != 1:
                raise RuntimeError(f"{member_id}/{sid}: response multiplicity")
            row = matches[0]
            if row["prompt"] != prompt or row["family"] != family:
                raise RuntimeError(f"{member_id}/{sid}: held-out prompt/family drift")
            member_responses[member_id] = row["response"]

        base_matches = [
            row for row in base["evaluation"]["responses"]
            if row["scenario_id"] == sid
        ]
        if len(base_matches) != 1:
            raise RuntimeError(f"BASE/{sid}: response multiplicity")
        base_row = base_matches[0]
        if base_row["prompt"] != prompt or base_row["family"] != family:
            raise RuntimeError(f"BASE/{sid}: held-out prompt/family drift")

        table.append(
            {
                "scenario_id": sid,
                "family": family,
                "prompt": prompt,
                "responses": member_responses,
                "base_response": base_row["response"],
            }
        )
    return table


def shuffled_aliases(values: list[str], seed_text: str, prefix: str) -> tuple[dict[str, str], dict[str, str]]:
    ordered = list(values)
    digest = hashlib.sha256(seed_text.encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    random.Random(seed).shuffle(ordered)
    width = max(2, len(str(len(values))))
    forward = {value: f"{prefix}{i:0{width}d}" for i, value in enumerate(ordered, 1)}
    reverse = {alias: value for value, alias in forward.items()}
    return forward, reverse


def lineage_descriptors() -> dict[str, dict]:
    out = {}
    for lineage in LINEAGES:
        packet = packet_for(lineage)
        out[lineage] = {
            "summary": packet["persona_summary"],
            "temperament_axes": packet.get("temperament_axes", {}),
            "embodied_habits": packet.get("embodied_habits", []),
            "motivational_biases": packet.get("motivational_biases", []),
            "interesting_broken": packet.get("interesting_broken", []),
        }
    return out


def normalize_prompt(records: list[dict]) -> str:
    public = [
        {
            "record_id": row["record_id"],
            "scenario": row["prompt"],
            "response": row["response"],
        }
        for row in records
    ]
    return """Convert every response into a style-neutral behavioral summary.

Preserve only observable choices, priorities, motives, hesitation, persistence,
risk appetite, territoriality, information sharing, failure/recovery behavior,
and concrete decision logic. Remove prose voice, catchphrases, cadence,
metaphors, decorative nouns, and any identity hints.

Each summary must be 1-3 plain sentences. Do not classify or score it.

Return JSON only:
{"summaries":[{"record_id":"...","summary":"..."}]}

RECORDS:
""" + json.dumps(public, ensure_ascii=False)


def normalize_all(
    *,
    key: str,
    final_cfg: dict,
    campaign: dict,
    table: list[dict],
    outdir: Path,
) -> tuple[dict[str, str], list[dict]]:
    records = []
    for scenario in table:
        sid = scenario["scenario_id"]
        for member_id, response in scenario["responses"].items():
            records.append(
                {
                    "record_id": f"{sid}:{member_id}",
                    "prompt": scenario["prompt"],
                    "response": response,
                }
            )
        records.append(
            {
                "record_id": f"{sid}:BASE",
                "prompt": scenario["prompt"],
                "response": scenario["base_response"],
            }
        )

    nv = campaign["nvidia"]
    model = nv["models"]["critic"]
    cfg = final_cfg["tournament"]
    batch_size = int(cfg["normalization_batch_size"])
    summaries: dict[str, str] = {}
    calls = []
    for index, start in enumerate(range(0, len(records), batch_size), 1):
        chunk = records[start : start + batch_size]
        label = f"normalize-{index:03d}"
        payload = {
            "model": model["id"],
            "messages": [
                {
                    "role": "system",
                    "content": "You neutrally summarize behavior and never infer hidden identities. Return JSON only.",
                },
                {"role": "user", "content": normalize_prompt(chunk)},
            ],
            "max_tokens": int(cfg["max_tokens"]["normalizer"]),
            "temperature": 0.15,
            "top_p": float(cfg["top_p"]),
            "stream": False,
        }
        content, obj, attempts = call_api(
            key=key,
            endpoint=nv["endpoint"],
            payload=payload,
            outdir=outdir,
            label=label,
            attempts_limit=int(cfg["attempts"]),
            role="final-behavior-normalizer",
        )
        parsed = extract_json(content)
        vals = parsed.get("summaries") if isinstance(parsed, dict) else None
        if not isinstance(vals, list):
            raise RuntimeError(f"{label}: summaries_not_list")
        expected = {row["record_id"] for row in chunk}
        got = {}
        for row in vals:
            if not isinstance(row, dict):
                continue
            rid = row.get("record_id")
            summary = row.get("summary")
            if rid in expected and isinstance(summary, str) and len(summary.strip()) >= 8:
                if rid in got:
                    raise RuntimeError(f"{label}: duplicate summary {rid}")
                got[rid] = summary.strip()
        if set(got) != expected:
            raise RuntimeError(f"{label}: summary IDs mismatch")
        summaries.update(got)
        calls.append(
            {
                "label": label,
                "records": len(chunk),
                "attempts": attempts,
                "usage": obj.get("usage"),
            }
        )
    if len(summaries) != 24 * 37:
        raise RuntimeError(f"normalization cardinality {len(summaries)} != 888")
    return summaries, calls


def lineage_prompt(
    *,
    scenario: dict,
    mode: str,
    summaries: dict[str, str],
    member_alias: dict[str, str],
    lineage_alias: dict[str, str],
    descriptors: dict[str, dict],
) -> str:
    desc_rows = [
        {"lineage_id": lineage_alias[lineage], **descriptors[lineage]}
        for lineage in LINEAGES
    ]
    candidates = []
    sid = scenario["scenario_id"]
    for member_id, response in scenario["responses"].items():
        text = summaries[f"{sid}:{member_id}"] if mode == "behavior" else response
        candidates.append({"candidate_id": member_alias[member_id], "text": text})
    random.Random(int.from_bytes(hashlib.sha256(f"{sid}:{mode}:candidates".encode()).digest()[:8], "big")).shuffle(candidates)
    random.Random(int.from_bytes(hashlib.sha256(f"{sid}:{mode}:descriptors".encode()).digest()[:8], "big")).shuffle(desc_rows)
    control_text = summaries[f"{sid}:BASE"] if mode == "behavior" else scenario["base_response"]
    mode_note = (
        "Candidate texts are style-neutral behavioral summaries. Use underlying actions and priorities only."
        if mode == "behavior"
        else "Candidate texts are raw outputs. Behavior outranks wording or stylistic resemblance."
    )
    return f"""Blindly classify 36 anonymous responses into one of 12 ancestral behavioral lineages.

There are three independently trained siblings from each lineage, but DO NOT assume
the predictions must contain exactly three of each lineage. Classify each candidate
independently from evidence.

For every candidate return:
- lineage_id: exactly one supplied opaque lineage_id;
- task_relevance: integer 0..4;
- genericness: integer 0..4, where 0 is distinctive/local behavior and 4 is generic assistant behavior;
- reason: a short concrete behavioral rationale.

Also score the unclassified BASE control for task_relevance and genericness only.
{mode_note}

Return JSON only:
{{"predictions":[{{"candidate_id":"C01","lineage_id":"L01","task_relevance":0,"genericness":0,"reason":"..."}}],
"control":{{"task_relevance":0,"genericness":0}}}}

SCENARIO:
{json.dumps({"scenario_id": sid, "family": scenario["family"], "prompt": scenario["prompt"]}, ensure_ascii=False)}

LINEAGE DESCRIPTORS:
{json.dumps(desc_rows, ensure_ascii=False)}

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}

UNCLASSIFIED BASE CONTROL:
{json.dumps({"text": control_text}, ensure_ascii=False)}
"""


def validate_lineage_output(
    raw: str,
    member_alias: dict[str, str],
    lineage_alias: dict[str, str],
) -> dict:
    obj = extract_json(raw)
    rows = obj.get("predictions") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("lineage predictions missing")
    expected_candidates = set(member_alias.values())
    expected_lineages = set(lineage_alias.values())
    got = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = row.get("candidate_id")
        lid = row.get("lineage_id")
        if cid not in expected_candidates or lid not in expected_lineages or cid in got:
            raise RuntimeError("invalid/duplicate lineage prediction identity")
        clean = {"candidate_id": cid, "lineage_id": lid}
        for key in ("task_relevance", "genericness"):
            value = row.get(key)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4:
                raise RuntimeError(f"{cid}: invalid {key}")
            clean[key] = value
        reason = row.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 3:
            raise RuntimeError(f"{cid}: invalid reason")
        clean["reason"] = reason.strip()
        got[cid] = clean
    if set(got) != expected_candidates:
        raise RuntimeError("lineage prediction candidate set mismatch")
    control = obj.get("control")
    if not isinstance(control, dict):
        raise RuntimeError("lineage control score missing")
    clean_control = {}
    for key in ("task_relevance", "genericness"):
        value = control.get(key)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4:
            raise RuntimeError(f"invalid BASE control {key}")
        clean_control[key] = value
    return {"predictions": list(got.values()), "control": clean_control}


def sibling_prompt(
    *,
    scenario: dict,
    mode: str,
    summaries: dict[str, str],
    member_alias: dict[str, str],
    population_by_id: dict[str, dict],
    descriptors: dict[str, dict],
) -> tuple[str, dict[str, dict[str, str]]]:
    sid = scenario["scenario_id"]
    groups = []
    alias_maps: dict[str, dict[str, str]] = {}
    for lineage in LINEAGES:
        members = [
            member_id for member_id, meta in population_by_id.items()
            if meta["lineage"] == lineage
        ]
        if len(members) != 3:
            raise RuntimeError(f"{lineage}: expected exactly three siblings")
        sibling_alias, reverse = shuffled_aliases(
            SIBLINGS, f"{sid}:{lineage}:sibling-descriptors", "S"
        )
        alias_maps[lineage] = reverse
        candidates = []
        for member_id in members:
            raw = scenario["responses"][member_id]
            text = summaries[f"{sid}:{member_id}"] if mode == "behavior" else raw
            candidates.append({"candidate_id": member_alias[member_id], "text": text})
        random.Random(int.from_bytes(hashlib.sha256(f"{sid}:{lineage}:{mode}".encode()).digest()[:8], "big")).shuffle(candidates)
        packet = descriptors[lineage]
        groups.append(
            {
                "group_id": lineage,
                "ancestral_lineage_summary": packet["summary"],
                "sibling_descriptors": [
                    {
                        "sibling_id": sibling_alias[sibling],
                        "description": SPECIALIZATION_GUIDANCE[sibling],
                    }
                    for sibling in SIBLINGS
                ],
                "candidates": candidates,
            }
        )
    random.Random(int.from_bytes(hashlib.sha256(f"{sid}:{mode}:groups".encode()).digest()[:8], "big")).shuffle(groups)
    mode_note = (
        "Texts are style-neutral behavioral summaries."
        if mode == "behavior"
        else "Texts are raw outputs; do not use wording alone."
    )
    prompt = f"""Given the correct ancestral lineage for each three-candidate group,
blindly match its three independently trained siblings to the three sibling
specializations. Use each sibling_id exactly once within every group.

{mode_note}

Return JSON only:
{{"groups":[{{"group_id":"G01","assignments":[{{"candidate_id":"C01","sibling_id":"S1","reason":"..."}}]}}]}}

GROUPS:
{json.dumps(groups, ensure_ascii=False)}
"""
    return prompt, alias_maps


def validate_sibling_output(
    raw: str,
    member_alias: dict[str, str],
    population_by_id: dict[str, dict],
    alias_maps: dict[str, dict[str, str]],
) -> list[dict]:
    obj = extract_json(raw)
    groups = obj.get("groups") if isinstance(obj, dict) else None
    if not isinstance(groups, list):
        raise RuntimeError("sibling groups missing")
    got = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        lineage = group.get("group_id")
        if lineage not in LINEAGES or lineage in got:
            raise RuntimeError("invalid/duplicate sibling group")
        expected_members = {
            member_alias[mid]
            for mid, meta in population_by_id.items()
            if meta["lineage"] == lineage
        }
        expected_sibling_aliases = set(alias_maps[lineage])
        assignments = group.get("assignments")
        if not isinstance(assignments, list) or len(assignments) != 3:
            raise RuntimeError(f"{lineage}: expected three sibling assignments")
        by_candidate = {}
        used_siblings = set()
        for row in assignments:
            if not isinstance(row, dict):
                raise RuntimeError(f"{lineage}: malformed sibling assignment")
            cid = row.get("candidate_id")
            sid = row.get("sibling_id")
            reason = row.get("reason")
            if (
                cid not in expected_members
                or sid not in expected_sibling_aliases
                or cid in by_candidate
                or sid in used_siblings
            ):
                raise RuntimeError(f"{lineage}: invalid/non-bijective sibling assignment")
            if not isinstance(reason, str) or len(reason.strip()) < 3:
                raise RuntimeError(f"{lineage}/{cid}: invalid sibling reason")
            by_candidate[cid] = {
                "candidate_id": cid,
                "sibling_id": sid,
                "reason": reason.strip(),
            }
            used_siblings.add(sid)
        if set(by_candidate) != expected_members or used_siblings != expected_sibling_aliases:
            raise RuntimeError(f"{lineage}: sibling assignment set mismatch")
        got[lineage] = list(by_candidate.values())
    if set(got) != set(LINEAGES):
        raise RuntimeError("sibling group set mismatch")
    return [{"group_id": lineage, "assignments": got[lineage]} for lineage in LINEAGES]


def run_mode(
    *,
    key: str,
    mode: str,
    final_cfg: dict,
    campaign: dict,
    population: dict,
    table: list[dict],
    summaries: dict[str, str],
    descriptors: dict[str, dict],
    outdir: Path,
) -> tuple[list[dict], list[dict]]:
    cfg = final_cfg["tournament"]
    nv = campaign["nvidia"]
    model = nv["models"]["critic"]
    population_by_id = {row["member_id"]: row for row in population["members"]}
    lineage_records = []
    sibling_records = []

    for scenario in table:
        sid = scenario["scenario_id"]
        member_alias, member_reverse = shuffled_aliases(
            list(population_by_id), f"{sid}:members", "C"
        )
        lineage_alias, lineage_reverse = shuffled_aliases(
            LINEAGES, f"{sid}:lineages", "L"
        )

        lprompt = lineage_prompt(
            scenario=scenario,
            mode=mode,
            summaries=summaries,
            member_alias=member_alias,
            lineage_alias=lineage_alias,
            descriptors=descriptors,
        )
        llabel = f"{mode}-{sid}-lineage"
        lpayload = {
            "model": model["id"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict blind behavioral evaluator. Return JSON only.",
                },
                {"role": "user", "content": lprompt},
            ],
            "max_tokens": int(cfg["max_tokens"]["lineage"]),
            "temperature": float(cfg["temperature"]),
            "top_p": float(cfg["top_p"]),
            "stream": False,
        }
        lcontent, lobj, lattempts = call_api(
            key=key,
            endpoint=nv["endpoint"],
            payload=lpayload,
            outdir=outdir,
            label=llabel,
            attempts_limit=int(cfg["attempts"]),
            role="final-lineage-judge",
        )
        lclean = validate_lineage_output(lcontent, member_alias, lineage_alias)
        mapped = []
        for row in lclean["predictions"]:
            member_id = member_reverse[row["candidate_id"]]
            mapped.append(
                {
                    **row,
                    "member_id": member_id,
                    "true_lineage": population_by_id[member_id]["lineage"],
                    "predicted_lineage": lineage_reverse[row["lineage_id"]],
                }
            )
        lineage_records.append(
            {
                "scenario_id": sid,
                "family": scenario["family"],
                "predictions": mapped,
                "control": lclean["control"],
                "call": {
                    "label": llabel,
                    "attempts": lattempts,
                    "usage": lobj.get("usage"),
                },
            }
        )

        sprompt, sibling_alias_maps = sibling_prompt(
            scenario=scenario,
            mode=mode,
            summaries=summaries,
            member_alias=member_alias,
            population_by_id=population_by_id,
            descriptors=descriptors,
        )
        slabel = f"{mode}-{sid}-sibling"
        spayload = {
            "model": model["id"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict blind behavioral evaluator. Return JSON only.",
                },
                {"role": "user", "content": sprompt},
            ],
            "max_tokens": int(cfg["max_tokens"]["sibling"]),
            "temperature": float(cfg["temperature"]),
            "top_p": float(cfg["top_p"]),
            "stream": False,
        }
        scontent, sobj, sattempts = call_api(
            key=key,
            endpoint=nv["endpoint"],
            payload=spayload,
            outdir=outdir,
            label=slabel,
            attempts_limit=int(cfg["attempts"]),
            role="final-sibling-judge",
        )
        sclean = validate_sibling_output(
            scontent, member_alias, population_by_id, sibling_alias_maps
        )
        smapped = []
        for group in sclean:
            lineage = group["group_id"]
            for row in group["assignments"]:
                member_id = member_reverse[row["candidate_id"]]
                smapped.append(
                    {
                        **row,
                        "member_id": member_id,
                        "true_sibling": population_by_id[member_id]["sibling"],
                        "predicted_sibling": sibling_alias_maps[lineage][row["sibling_id"]],
                    }
                )
        sibling_records.append(
            {
                "scenario_id": sid,
                "family": scenario["family"],
                "predictions": smapped,
                "call": {
                    "label": slabel,
                    "attempts": sattempts,
                    "usage": sobj.get("usage"),
                },
            }
        )
    return lineage_records, sibling_records


def zero_matrix(labels: list[str]) -> dict[str, dict[str, int]]:
    return {true: {pred: 0 for pred in labels} for true in labels}


def summarize_mode(
    mode: str,
    population: dict,
    lineage_records: list[dict],
    sibling_records: list[dict],
) -> dict:
    member_ids = [row["member_id"] for row in population["members"]]
    meta = {row["member_id"]: row for row in population["members"]}
    lineage_conf = zero_matrix(LINEAGES)
    sibling_conf = zero_matrix(SIBLINGS)
    member_conf = zero_matrix(member_ids)
    by_member: dict[str, dict[str, list[float] | int]] = {
        member_id: {
            "lineage_correct": 0,
            "sibling_correct": 0,
            "member_correct": 0,
            "task_relevance": [],
            "genericness": [],
        }
        for member_id in member_ids
    }
    sibling_by_scenario = {
        row["scenario_id"]: {
            pred["member_id"]: pred["predicted_sibling"]
            for pred in row["predictions"]
        }
        for row in sibling_records
    }
    family_counts = defaultdict(lambda: Counter(
        lineage_correct=0, sibling_correct=0, member_correct=0, total=0
    ))
    control_genericness = []
    control_relevance = []

    for record in lineage_records:
        sid = record["scenario_id"]
        family = record["family"]
        control_genericness.append(record["control"]["genericness"])
        control_relevance.append(record["control"]["task_relevance"])
        siblings = sibling_by_scenario[sid]
        for row in record["predictions"]:
            member_id = row["member_id"]
            true_lineage = meta[member_id]["lineage"]
            true_sibling = meta[member_id]["sibling"]
            pred_lineage = row["predicted_lineage"]
            pred_sibling = siblings[member_id]
            pred_member = f"V-{pred_lineage}-{pred_sibling}"
            lineage_ok = pred_lineage == true_lineage
            sibling_ok = pred_sibling == true_sibling
            member_ok = pred_member == member_id
            lineage_conf[true_lineage][pred_lineage] += 1
            sibling_conf[true_sibling][pred_sibling] += 1
            member_conf[member_id][pred_member] += 1
            state = by_member[member_id]
            state["lineage_correct"] += int(lineage_ok)
            state["sibling_correct"] += int(sibling_ok)
            state["member_correct"] += int(member_ok)
            state["task_relevance"].append(row["task_relevance"])
            state["genericness"].append(row["genericness"])
            family_counts[family]["lineage_correct"] += int(lineage_ok)
            family_counts[family]["sibling_correct"] += int(sibling_ok)
            family_counts[family]["member_correct"] += int(member_ok)
            family_counts[family]["total"] += 1

    total = 24 * 36
    per_member = {}
    nearest = {}
    for member_id in member_ids:
        state = by_member[member_id]
        row = member_conf[member_id]
        ranked = sorted(
            (
                {"member_id": candidate, "count": count}
                for candidate, count in row.items()
                if candidate != member_id
            ),
            key=lambda item: (-item["count"], item["member_id"]),
        )[:5]
        nearest[member_id] = ranked
        per_member[member_id] = {
            "lineage": meta[member_id]["lineage"],
            "sibling": meta[member_id]["sibling"],
            "lineage_accuracy": state["lineage_correct"] / 24,
            "sibling_accuracy": state["sibling_correct"] / 24,
            "member_accuracy": state["member_correct"] / 24,
            "mean_task_relevance": mean(state["task_relevance"]),
            "mean_genericness": mean(state["genericness"]),
            "nearest_confusions": ranked,
        }

    return {
        "mode": mode,
        "lineage_accuracy": sum(
            lineage_conf[t][t] for t in LINEAGES
        ) / total,
        "sibling_accuracy": sum(
            sibling_conf[t][t] for t in SIBLINGS
        ) / total,
        "member_accuracy": sum(
            member_conf[t][t] for t in member_ids
        ) / total,
        "lineage_confusion": lineage_conf,
        "sibling_confusion": sibling_conf,
        "member_confusion": member_conf,
        "per_member": per_member,
        "nearest_confusions": nearest,
        "population": {
            "mean_task_relevance": mean(
                [x for state in by_member.values() for x in state["task_relevance"]]
            ),
            "mean_genericness": mean(
                [x for state in by_member.values() for x in state["genericness"]]
            ),
        },
        "control": {
            "mean_task_relevance": mean(control_relevance),
            "mean_genericness": mean(control_genericness),
        },
        "family_accuracy": {
            family: {
                "lineage_accuracy": counts["lineage_correct"] / counts["total"],
                "sibling_accuracy": counts["sibling_correct"] / counts["total"],
                "member_accuracy": counts["member_correct"] / counts["total"],
                "responses": counts["total"],
            }
            for family, counts in sorted(family_counts.items())
        },
    }


def self_test(config_path: Path) -> None:
    final_cfg, campaign, population = load_contract(config_path)
    assert int(final_cfg["tournament"]["scenario_count"]) == 24
    assert int(final_cfg["tournament"]["cloud_max_parallel"]) == 1
    assert campaign["evaluation"]["lineage_classes"] == 12
    assert campaign["evaluation"]["sibling_classes"] == 3
    member_ids = [row["member_id"] for row in population["members"]]
    assert len(member_ids) == 36 and len(set(member_ids)) == 36
    assert {row["lineage"] for row in population["members"]} == set(LINEAGES)
    assert {row["sibling"] for row in population["members"]} == set(SIBLINGS)
    fwd, rev = shuffled_aliases(member_ids, "self-test", "C")
    assert len(fwd) == len(rev) == 36
    assert all(rev[fwd[value]] == value for value in member_ids)
    descriptors = lineage_descriptors()
    assert set(descriptors) == set(LINEAGES)
    print("VILLAGE-001E tournament self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/evaluation-config.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/village001e-tournament")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.config)
        return

    final_cfg, campaign, population = load_contract(args.config)
    results, base = load_results(final_cfg, population)
    table = scenario_table(results, base)
    if len(table) != int(final_cfg["tournament"]["scenario_count"]):
        raise RuntimeError("scenario count mismatch")

    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY/NVKEY required")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    provider = out / "provider"
    descriptors = lineage_descriptors()

    summaries, normalization_calls = normalize_all(
        key=key,
        final_cfg=final_cfg,
        campaign=campaign,
        table=table,
        outdir=provider,
    )
    write_json(out / "normalized-behaviors.json", summaries)

    raw_lineage, raw_sibling = run_mode(
        key=key,
        mode="raw",
        final_cfg=final_cfg,
        campaign=campaign,
        population=population,
        table=table,
        summaries=summaries,
        descriptors=descriptors,
        outdir=provider,
    )
    behavior_lineage, behavior_sibling = run_mode(
        key=key,
        mode="behavior",
        final_cfg=final_cfg,
        campaign=campaign,
        population=population,
        table=table,
        summaries=summaries,
        descriptors=descriptors,
        outdir=provider,
    )
    write_json(
        out / "raw-judgements.json",
        {"lineage": raw_lineage, "sibling": raw_sibling},
    )
    write_json(
        out / "behavior-judgements.json",
        {"lineage": behavior_lineage, "sibling": behavior_sibling},
    )

    raw_summary = summarize_mode(
        "raw", population, raw_lineage, raw_sibling
    )
    behavior_summary = summarize_mode(
        "behavior", population, behavior_lineage, behavior_sibling
    )
    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "stage": "hierarchical-blind-tournament",
        "issue": 14,
        "member_contract_sha256": population["member_contract_sha256"],
        "members": 36,
        "scenarios": 24,
        "responses_judged_per_mode": 864,
        "base_control_responses_per_mode": 24,
        "hidden_member_prompt_used": False,
        "judge_provider": "nvidia",
        "judge_model": campaign["nvidia"]["models"]["critic"]["id"],
        "remote_unpinned_backend": True,
        "normalization_calls": normalization_calls,
        "raw": raw_summary,
        "behavior": behavior_summary,
        "style_accuracy_gap": {
            "lineage": raw_summary["lineage_accuracy"] - behavior_summary["lineage_accuracy"],
            "sibling": raw_summary["sibling_accuracy"] - behavior_summary["sibling_accuracy"],
            "member": raw_summary["member_accuracy"] - behavior_summary["member_accuracy"],
        },
    }
    write_json(out / "tournament.json", result)
    print(
        json.dumps(
            {
                "experiment": "VILLAGE-001E",
                "raw_lineage_accuracy": raw_summary["lineage_accuracy"],
                "behavior_lineage_accuracy": behavior_summary["lineage_accuracy"],
                "raw_sibling_accuracy": raw_summary["sibling_accuracy"],
                "behavior_sibling_accuracy": behavior_summary["sibling_accuracy"],
                "raw_member_accuracy": raw_summary["member_accuracy"],
                "behavior_member_accuracy": behavior_summary["member_accuracy"],
                "behavior_genericness": behavior_summary["population"]["mean_genericness"],
                "base_genericness": behavior_summary["control"]["mean_genericness"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
