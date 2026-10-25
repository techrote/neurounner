#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from babble_002_expand import extract_json, packet_for
from village_001_manifest import build_manifest
from village_001b_generate_member import call_api

ROOT = Path(__file__).resolve().parents[1]
SPECIALIZATION_GUIDANCE = {
    "A": (
        "failure-hardened: awkward failure/recovery, conflicting motives, physical "
        "reaction, imperfect settling"
    ),
    "B": (
        "local tinkerer: bounded local memory, mundane habits, maintenance choices, "
        "practical physical interaction"
    ),
    "C": (
        "social scavenger: bargaining, opportunism, social friction, embodied choices, "
        "conflicting motives"
    ),
}
SCORE_KEYS = (
    "task_relevance",
    "genericness",
    "lineage_fit",
    "style_dependence",
    "behavior_fit",
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def member_contract(campaign: dict, member_id: str) -> tuple[dict, dict]:
    population = build_manifest(campaign)
    for row in population["members"]:
        if row["member_id"] == member_id:
            return population, row
    raise ValueError(f"unknown member {member_id}")


def orientation(member_id: str, scenario_id: str) -> str:
    digest = hashlib.sha256(f"{member_id}:{scenario_id}".encode()).digest()
    return "parent-x" if digest[0] % 2 == 0 else "child-x"


def build_cases(member: dict, parent: list[dict], child: list[dict]) -> list[dict]:
    parent_by_id = {row["scenario_id"]: row for row in parent}
    child_by_id = {row["scenario_id"]: row for row in child}
    expected = [f"T{i:02d}" for i in range(1, 25)]
    if sorted(parent_by_id) != expected or sorted(child_by_id) != expected:
        raise RuntimeError("parent/child held-out scenario sets are incomplete")

    cases = []
    for scenario_id in expected:
        p = parent_by_id[scenario_id]
        c = child_by_id[scenario_id]
        if p["prompt"] != c["prompt"]:
            raise RuntimeError(f"{scenario_id}: parent/child prompt drift")
        orient = orientation(member["member_id"], scenario_id)
        if orient == "parent-x":
            x_text, y_text = p["response"], c["response"]
        else:
            x_text, y_text = c["response"], p["response"]
        cases.append(
            {
                "scenario_id": scenario_id,
                "family": p["family"],
                "prompt": p["prompt"],
                "X": x_text,
                "Y": y_text,
                "_orientation": orient,
            }
        )
    return cases


def judge_prompt(member: dict, packet: dict, cases: list[dict]) -> str:
    public_cases = [
        {k: value for k, value in case.items() if not k.startswith("_")}
        for case in cases
    ]
    sibling_defs = "\n".join(
        f"- {key}: {value}" for key, value in SPECIALIZATION_GUIDANCE.items()
    )
    return f"""Blindly compare two responses per scenario.

Target ancestral lineage summary:
{packet['persona_summary']}

The three sibling ecologies are:
{sibling_defs}

The actual target sibling is NOT stated here. Infer it from behavior.

For every response label X and Y, return integer 0..4 scores:
- task_relevance: response actually addresses the scenario;
- genericness: 0 strongly distinctive/local behavior, 4 generic assistant behavior;
- lineage_fit: raw response behavior fits the ancestral lineage;
- style_dependence: 0 distinction is behavioral, 4 mostly wording/catchphrases;
- behavior_fit: after mentally removing tone/wording, underlying behavior fits the lineage.

Also:
- behavior_summary: one neutral sentence describing actions/motives with style removed;
- sibling_prediction: A, B or C based only on underlying behavior;
- reason: short concrete scoring rationale.

Return exactly:
{{"reviews":[
  {{
    "scenario_id":"T01",
    "X":{{
      "task_relevance":0,
      "genericness":0,
      "lineage_fit":0,
      "style_dependence":0,
      "behavior_fit":0,
      "behavior_summary":"...",
      "sibling_prediction":"A",
      "reason":"..."
    }},
    "Y":{{ "...":"same fields" }}
  }}
]}}

Do not guess which response is earlier/later training. Judge X and Y independently.
Return JSON only, one review per supplied scenario, no extras.

CASES:
{json.dumps(public_cases, ensure_ascii=False)}
"""


def validate_side(side: dict, scenario_id: str, label: str) -> dict:
    if not isinstance(side, dict):
        raise ValueError(f"{scenario_id}:{label}: side not object")
    out = {}
    for key in SCORE_KEYS:
        value = side.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 4
        ):
            raise ValueError(f"{scenario_id}:{label}:{key}")
        out[key] = value
    summary = side.get("behavior_summary")
    reason = side.get("reason")
    sibling = side.get("sibling_prediction")
    if not isinstance(summary, str) or len(summary.strip()) < 8:
        raise ValueError(f"{scenario_id}:{label}:behavior_summary")
    if not isinstance(reason, str) or len(reason.strip()) < 4:
        raise ValueError(f"{scenario_id}:{label}:reason")
    if sibling not in {"A", "B", "C"}:
        raise ValueError(f"{scenario_id}:{label}:sibling_prediction")
    out.update(
        {
            "behavior_summary": summary.strip(),
            "sibling_prediction": sibling,
            "reason": reason.strip(),
        }
    )
    return out


def validate_reviews(raw: str, cases: list[dict]) -> list[dict]:
    obj = extract_json(raw)
    reviews = obj.get("reviews") if isinstance(obj, dict) else None
    expected = [case["scenario_id"] for case in cases]
    if not isinstance(reviews, list) or len(reviews) != len(expected):
        raise ValueError("review count mismatch")
    out = {}
    for row in reviews:
        if not isinstance(row, dict):
            raise ValueError("review not object")
        scenario_id = row.get("scenario_id")
        if scenario_id not in expected or scenario_id in out:
            raise ValueError("scenario ID mismatch")
        out[scenario_id] = {
            "scenario_id": scenario_id,
            "X": validate_side(row.get("X"), scenario_id, "X"),
            "Y": validate_side(row.get("Y"), scenario_id, "Y"),
        }
    if set(out) != set(expected):
        raise ValueError("review scenario IDs incomplete")
    return [out[scenario_id] for scenario_id in expected]


def score_summary(rows: list[dict], correct_sibling: str) -> dict:
    result = {
        key: statistics.mean(row[key] for row in rows)
        for key in SCORE_KEYS
    }
    result["sibling_accuracy"] = (
        sum(1 for row in rows if row["sibling_prediction"] == correct_sibling)
        / len(rows)
    )
    result["raw_vs_behavior_gap"] = (
        result["lineage_fit"] - result["behavior_fit"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/preference-config.json",
    )
    parser.add_argument("--member", required=True)
    parser.add_argument("--preference-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pref = json.loads(args.config.read_text())
    campaign = json.loads((ROOT / pref["campaign_config"]).read_text())
    population, member = member_contract(campaign, args.member)
    packet = packet_for(member["lineage"])

    parent_path = (
        ROOT / pref["sft_evidence_root"] / "results" / f"{member['member_id']}.json"
    )
    parent_result = json.loads(parent_path.read_text())
    child_result = json.loads(args.preference_result.read_text())

    if parent_result.get("member_contract_sha256") != population[
        "member_contract_sha256"
    ]:
        raise RuntimeError("parent member contract mismatch")
    if child_result.get("member_contract_sha256") != population[
        "member_contract_sha256"
    ]:
        raise RuntimeError("preference member contract mismatch")
    if child_result.get("member_id") != member["member_id"]:
        raise RuntimeError("preference result member mismatch")
    if child_result.get("parent_adapter_sha256") != parent_result["adapter"][
        "sha256"
    ]:
        raise RuntimeError("preference parent adapter hash mismatch")

    parent_responses = parent_result["evaluation"]["responses"]
    child_responses = child_result["evaluation"]["responses"]
    cases = build_cases(member, parent_responses, child_responses)

    key = __import__("os").environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY/NVKEY is required")

    out = args.out
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rawdir = out / "provider"
    model = campaign["nvidia"]["models"]["critic"]
    all_reviews = []

    for chunk_index in range(0, len(cases), 6):
        chunk = cases[chunk_index : chunk_index + 6]
        label = (
            f"{member['member_id']}-before-after-"
            f"{chunk_index // 6 + 1:02d}-bolt"
        )
        payload = {
            "model": model["id"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a blind behavioral evaluator. Separate behavior from "
                        "wording and score both candidate responses independently. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": judge_prompt(member, packet, chunk),
                },
            ],
            "max_tokens": 6144,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
        }
        raw, _, _ = call_api(
            key=key,
            endpoint=campaign["nvidia"]["endpoint"],
            payload=payload,
            outdir=rawdir,
            label=label,
            attempts_limit=int(pref["pair_generation"]["attempts"]),
            role="before-after-critic",
        )
        try:
            reviews = validate_reviews(raw, chunk)
        except ValueError as exc:
            repair_label = label + "-json-repair"
            repair_payload = {
                "model": model["id"],
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Repair the blind evaluator JSON only. Return complete JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            judge_prompt(member, packet, chunk)
                            + "\n\nMALFORMED OUTPUT:\n"
                            + raw
                            + f"\n\nPARSER PROBLEM: {exc}"
                        ),
                    },
                ],
                "max_tokens": 6144,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False,
            }
            repaired, _, _ = call_api(
                key=key,
                endpoint=campaign["nvidia"]["endpoint"],
                payload=repair_payload,
                outdir=rawdir,
                label=repair_label,
                attempts_limit=int(pref["pair_generation"]["attempts"]),
                role="before-after-critic-repair",
            )
            reviews = validate_reviews(repaired, chunk)
        all_reviews.extend(reviews)

    if len(all_reviews) != 24:
        raise RuntimeError("before/after review count mismatch")

    parent_rows = []
    child_rows = []
    mapped = []
    reviews_by_id = {row["scenario_id"]: row for row in all_reviews}
    for case in cases:
        review = reviews_by_id[case["scenario_id"]]
        if case["_orientation"] == "parent-x":
            parent_side, child_side = review["X"], review["Y"]
            parent_label, child_label = "X", "Y"
        else:
            parent_side, child_side = review["Y"], review["X"]
            parent_label, child_label = "Y", "X"
        parent_rows.append(parent_side)
        child_rows.append(child_side)
        mapped.append(
            {
                "scenario_id": case["scenario_id"],
                "family": case["family"],
                "orientation": case["_orientation"],
                "parent_label": parent_label,
                "child_label": child_label,
                "parent": parent_side,
                "child": child_side,
            }
        )

    before = score_summary(parent_rows, member["sibling"])
    after = score_summary(child_rows, member["sibling"])
    delta = {
        key: after[key] - before[key]
        for key in (
            *SCORE_KEYS,
            "sibling_accuracy",
            "raw_vs_behavior_gap",
        )
    }

    relevance_gate = delta["task_relevance"] >= -0.5
    style_only_regression = (
        delta["lineage_fit"] > 0
        and delta["behavior_fit"] < 0
        and delta["raw_vs_behavior_gap"] > 0.25
    )
    behavior_gate = not style_only_regression

    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "stage": "before-after-evaluation",
        "issue": 13,
        "member_id": member["member_id"],
        "lineage": member["lineage"],
        "sibling": member["sibling"],
        "member_contract_sha256": population["member_contract_sha256"],
        "parent_adapter_sha256": parent_result["adapter"]["sha256"],
        "preference_adapter_sha256": child_result["adapter"]["sha256"],
        "judge_model": model["id"],
        "blind_orientation_rule": "sha256(member_id:scenario_id) first-byte parity",
        "scenario_count": 24,
        "before": before,
        "after": after,
        "delta": delta,
        "gates": {
            "task_relevance_delta_at_least_minus_0_5": relevance_gate,
            "no_raw_only_behavior_regression": behavior_gate,
            "style_only_regression_detected": style_only_regression,
            "qualified_for_population_tournament": (
                relevance_gate and behavior_gate
            ),
        },
        "reviews": mapped,
    }
    write_json(out / "evaluation.json", result)
    print(
        json.dumps(
            {
                "member": member["member_id"],
                "before": before,
                "after": after,
                "delta": delta,
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
