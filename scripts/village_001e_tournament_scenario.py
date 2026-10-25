#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from babble_002_expand import extract_json, packet_for
from village_001_manifest import build_manifest
from village_001b_generate_member import call_api

ROOT = Path(__file__).resolve().parents[1]
SIBLING_CATALOG = {
    "A": (
        "failure-hardened: awkward failure and recovery, conflicting motives, "
        "physical reaction, stubbornness, imperfect settling"
    ),
    "B": (
        "local tinkerer: bounded local memory, mundane habits, maintenance choices, "
        "practical manipulation and local repair"
    ),
    "C": (
        "social scavenger: bargaining, opportunism, social friction, embodied choices, "
        "conflicting motives and resource-seeking"
    ),
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lineage_catalog() -> dict[str, str]:
    return {
        f"G{i:02d}": packet_for(f"G{i:02d}")["persona_summary"]
        for i in range(1, 13)
    }


def load_cases(
    cfg: dict,
    scenario_id: str,
    responses_root: Path,
) -> tuple[dict, list[dict]]:
    campaign = json.loads((ROOT / cfg["campaign_config"]).read_text())
    population = build_manifest(campaign)
    members = population["members"]
    member_meta = {
        row["member_id"]: row
        for row in members
    }

    final_eval_path = ROOT / cfg["final_eval_config"]
    final_eval = json.loads(final_eval_path.read_text())
    final_eval_sha = sha256_file(final_eval_path)
    by_scenario = {
        row["id"]: row
        for row in final_eval["scenarios"]
    }
    if scenario_id not in by_scenario:
        raise RuntimeError(f"final heldout scenario {scenario_id} not found")
    expected_scenario = by_scenario[scenario_id]

    release_path = (
        ROOT
        / cfg["preference_evidence_root"]
        / "adapter-release.json"
    )
    release = json.loads(release_path.read_text())
    if release.get("member_contract_sha256") != population[
        "member_contract_sha256"
    ]:
        raise RuntimeError("preference release/member contract mismatch")
    release_assets = {
        row["member_id"]: row
        for row in release.get("assets", [])
    }
    if set(release_assets) != set(member_meta):
        raise RuntimeError("preference release member set incomplete")

    found: dict[str, Path] = {}
    for result_path in responses_root.rglob("result.json"):
        try:
            result = json.loads(result_path.read_text())
        except Exception:
            continue
        if (
            result.get("experiment") != "VILLAGE-001E"
            or result.get("stage") != "fresh-final-heldout-responses"
        ):
            continue
        member_id = result.get("member_id")
        if member_id not in member_meta:
            raise RuntimeError(
                f"unexpected final-response member {member_id!r}"
            )
        if member_id in found:
            raise RuntimeError(
                f"duplicate final-response artifact for {member_id}"
            )
        found[member_id] = result_path

    missing = sorted(set(member_meta) - set(found))
    extra = sorted(set(found) - set(member_meta))
    if missing or extra:
        raise RuntimeError(
            f"final-response artifacts mismatch missing={missing} extra={extra}"
        )

    rows = []
    for member in members:
        member_id = member["member_id"]
        result = json.loads(found[member_id].read_text())
        checks = {
            "member_id": member_id,
            "lineage": member["lineage"],
            "sibling": member["sibling"],
            "member_contract_sha256": population["member_contract_sha256"],
            "base_model": campaign["base_model"]["id"],
            "base_revision": campaign["base_model"]["revision"],
            "preference_release_tag": release["release_tag"],
            "preference_adapter_sha256": release_assets[member_id][
                "adapter_directory_sha256"
            ],
            "final_eval_config_sha256": final_eval_sha,
            "fresh_after_preference_training": True,
            "hidden_member_prompt_used": False,
            "exact_training_input_collision_count": 0,
        }
        for key, expected in checks.items():
            if result.get(key) != expected:
                raise RuntimeError(
                    f"{member_id}: final-response {key} "
                    f"{result.get(key)!r} != {expected!r}"
                )

        response_by_id = {
            item["scenario_id"]: item
            for item in result.get("responses", [])
        }
        if set(response_by_id) != set(by_scenario):
            raise RuntimeError(
                f"{member_id}: final-response scenario set mismatch"
            )
        response = response_by_id[scenario_id]
        if (
            response.get("prompt") != expected_scenario["prompt"]
            or response.get("family") != expected_scenario["family"]
        ):
            raise RuntimeError(
                f"{member_id}/{scenario_id}: final prompt/family drift"
            )
        rows.append(
            {
                "member_id": member_id,
                "lineage": member["lineage"],
                "sibling": member["sibling"],
                "response": response["response"],
                "adapter_sha256": result["preference_adapter_sha256"],
            }
        )

    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{scenario_id}:{row['member_id']}".encode()
        ).hexdigest()
    )
    for index, row in enumerate(rows, 1):
        row["opaque_id"] = f"R{index:03d}"

    return {
        "scenario_id": scenario_id,
        "family": expected_scenario["family"],
        "prompt": expected_scenario["prompt"],
        "member_contract_sha256": population["member_contract_sha256"],
        "final_eval_config_sha256": final_eval_sha,
        "fresh_after_preference_training": True,
    }, rows


def normalize_prompt(scenario: dict, rows: list[dict]) -> str:
    public = [
        {
            "id": row["opaque_id"],
            "response": row["response"],
        }
        for row in rows
    ]
    return f"""Convert each response into a neutral behavior-only summary.

Scenario:
{scenario['prompt']}

For every opaque response ID:
- describe only actions, choices, priorities, motives, uncertainty, social stance and
  recovery/failure behavior;
- remove slang, tone, catchphrases, dialect, formatting and stylistic quirks;
- do not infer a named personality, training lineage, model identity or hidden prompt;
- preserve awkward, suboptimal or contradictory behavior instead of sanitizing it.

Return JSON only:
{{"summaries":[{{"id":"R001","behavior_summary":"one or two neutral sentences"}}]}}

One result per supplied ID, no extras.

RESPONSES:
{json.dumps(public, ensure_ascii=False)}
"""


def validate_summaries(raw: str, expected: list[str]) -> dict[str, str]:
    obj = extract_json(raw)
    rows = obj.get("summaries") if isinstance(obj, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("summary count mismatch")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("summary row is not an object")
        opaque_id = row.get("id")
        summary = row.get("behavior_summary")
        if opaque_id not in expected or opaque_id in out:
            raise ValueError("summary ID mismatch")
        if not isinstance(summary, str) or len(summary.strip()) < 8:
            raise ValueError(f"{opaque_id}: invalid behavior summary")
        out[opaque_id] = summary.strip()
    if set(out) != set(expected):
        raise ValueError("summary IDs incomplete")
    return out


def classifier_prompt(
    scenario: dict,
    rows: list[dict],
    *,
    mode: str,
) -> str:
    catalog = lineage_catalog()
    public = []
    for row in rows:
        value = (
            row["response"]
            if mode == "raw"
            else row["behavior_summary"]
        )
        public.append({"id": row["opaque_id"], "text": value})

    return f"""Blindly classify the supplied {mode} responses.

Scenario:
{scenario['prompt']}

Candidate ancestral lineages:
{json.dumps(catalog, ensure_ascii=False, sort_keys=True)}

Candidate sibling ecologies:
{json.dumps(SIBLING_CATALOG, ensure_ascii=False, sort_keys=True)}

For each opaque ID, infer from behavior only:
- predicted_lineage: one of G01..G12;
- predicted_sibling: A, B or C;
- lineage_confidence: integer 0..4;
- sibling_confidence: integer 0..4;
- task_relevance: integer 0..4, whether the response/behavior addresses the scenario;
- genericness: integer 0..4, 0 strongly distinctive/local and 4 generic-default behavior;
- reason: short concrete behavioral reason.

Do not infer which model or training stage produced the text. Do not use prose style
as sufficient evidence for lineage or sibling.

Return JSON only:
{{"classifications":[{{
  "id":"R001",
  "predicted_lineage":"G01",
  "predicted_sibling":"A",
  "lineage_confidence":0,
  "sibling_confidence":0,
  "task_relevance":0,
  "genericness":0,
  "reason":"..."
}}]}}

One result per supplied ID, no extras.

ITEMS:
{json.dumps(public, ensure_ascii=False)}
"""


def validate_classifications(
    raw: str,
    expected: list[str],
) -> dict[str, dict]:
    obj = extract_json(raw)
    rows = obj.get("classifications") if isinstance(obj, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("classification count mismatch")
    out = {}
    lineages = {f"G{i:02d}" for i in range(1, 13)}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("classification is not object")
        opaque_id = row.get("id")
        if opaque_id not in expected or opaque_id in out:
            raise ValueError("classification ID mismatch")
        lineage = row.get("predicted_lineage")
        sibling = row.get("predicted_sibling")
        if lineage not in lineages:
            raise ValueError(f"{opaque_id}: invalid lineage")
        if sibling not in {"A", "B", "C"}:
            raise ValueError(f"{opaque_id}: invalid sibling")
        scores = {}
        for key in (
            "lineage_confidence",
            "sibling_confidence",
            "task_relevance",
            "genericness",
        ):
            value = row.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 4
            ):
                raise ValueError(f"{opaque_id}:{key}")
            scores[key] = value
        reason = row.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 4:
            raise ValueError(f"{opaque_id}: invalid reason")
        out[opaque_id] = {
            "predicted_lineage": lineage,
            "predicted_sibling": sibling,
            **scores,
            "reason": reason.strip(),
        }
    if set(out) != set(expected):
        raise ValueError("classification IDs incomplete")
    return out


def sibling_prompt(
    scenario: dict,
    rows: list[dict],
    *,
    mode: str,
) -> str:
    catalog = lineage_catalog()
    public = []
    for row in rows:
        value = row["response"] if mode == "raw" else row["behavior_summary"]
        public.append(
            {
                "id": row["opaque_id"],
                "correct_lineage": row["lineage"],
                "correct_lineage_summary": catalog[row["lineage"]],
                "text": value,
            }
        )
    return f"""The correct ancestral lineage is supplied for each opaque response.
Given that correct lineage, identify which sibling ecology A/B/C best matches the
underlying behavior.

Sibling ecologies:
{json.dumps(SIBLING_CATALOG, ensure_ascii=False, sort_keys=True)}

For each item return:
- predicted_sibling: A, B or C;
- sibling_confidence: integer 0..4;
- reason: short behavioral reason.

This is the second level of a hierarchical tournament. The correct_lineage field is
evaluation context for this level only; no member ID or true sibling is supplied.
Do not use prose style as sufficient evidence.

Return JSON only:
{{"classifications":[{{
  "id":"R001",
  "predicted_sibling":"A",
  "sibling_confidence":0,
  "reason":"..."
}}]}}

SCENARIO:
{scenario['prompt']}

ITEMS:
{json.dumps(public, ensure_ascii=False)}
"""


def validate_sibling_classifications(
    raw: str,
    expected: list[str],
) -> dict[str, dict]:
    obj = extract_json(raw)
    rows = obj.get("classifications") if isinstance(obj, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("sibling classification count mismatch")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("sibling classification is not object")
        opaque_id = row.get("id")
        if opaque_id not in expected or opaque_id in out:
            raise ValueError("sibling classification ID mismatch")
        sibling = row.get("predicted_sibling")
        confidence = row.get("sibling_confidence")
        reason = row.get("reason")
        if sibling not in {"A", "B", "C"}:
            raise ValueError(f"{opaque_id}: invalid sibling")
        if (
            not isinstance(confidence, int)
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 4
        ):
            raise ValueError(f"{opaque_id}: invalid sibling_confidence")
        if not isinstance(reason, str) or len(reason.strip()) < 4:
            raise ValueError(f"{opaque_id}: invalid sibling reason")
        out[opaque_id] = {
            "predicted_sibling": sibling,
            "sibling_confidence": confidence,
            "reason": reason.strip(),
        }
    if set(out) != set(expected):
        raise ValueError("sibling classification IDs incomplete")
    return out


def adjudicator_prompt(
    scenario: dict,
    rows: list[dict],
) -> str:
    catalog = lineage_catalog()
    public = [
        {
            "id": row["opaque_id"],
            "behavior_summary": row["behavior_summary"],
            "bolt_prediction": row["behavior_prediction"],
        }
        for row in rows
    ]
    return f"""Independently audit low-confidence behavior-only classifications.

Scenario:
{scenario['prompt']}

Lineage catalog:
{json.dumps(catalog, ensure_ascii=False, sort_keys=True)}

Sibling ecologies:
{json.dumps(SIBLING_CATALOG, ensure_ascii=False, sort_keys=True)}

For each item, classify the behavior summary independently. Bolt's earlier prediction
is supplied only so you can state whether you agree; it is not authority.

Return JSON only:
{{"audits":[{{
  "id":"R001",
  "predicted_lineage":"G01",
  "predicted_sibling":"A",
  "agree_with_bolt":true,
  "reason":"..."
}}]}}

ITEMS:
{json.dumps(public, ensure_ascii=False)}
"""


def validate_audits(raw: str, expected: list[str]) -> dict[str, dict]:
    obj = extract_json(raw)
    rows = obj.get("audits") if isinstance(obj, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("audit count mismatch")
    lineages = {f"G{i:02d}" for i in range(1, 13)}
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("audit row not object")
        opaque_id = row.get("id")
        if opaque_id not in expected or opaque_id in out:
            raise ValueError("audit ID mismatch")
        lineage = row.get("predicted_lineage")
        sibling = row.get("predicted_sibling")
        agree = row.get("agree_with_bolt")
        reason = row.get("reason")
        if lineage not in lineages or sibling not in {"A", "B", "C"}:
            raise ValueError(f"{opaque_id}: invalid audit class")
        if not isinstance(agree, bool):
            raise ValueError(f"{opaque_id}: invalid audit agreement")
        if not isinstance(reason, str) or len(reason.strip()) < 4:
            raise ValueError(f"{opaque_id}: invalid audit reason")
        out[opaque_id] = {
            "predicted_lineage": lineage,
            "predicted_sibling": sibling,
            "agree_with_bolt": agree,
            "reason": reason.strip(),
        }
    return out


def call_json_with_repair(
    *,
    key: str,
    campaign: dict,
    model: dict,
    rawdir: Path,
    label: str,
    prompt: str,
    system: str,
    max_tokens: int,
    validator,
    expected: list[str],
    role: str,
) -> tuple[dict, list[str]]:
    payload = {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
    }
    if "reasoning_effort" in model:
        payload["reasoning_effort"] = model["reasoning_effort"]
    raw, _, _ = call_api(
        key=key,
        endpoint=campaign["nvidia"]["endpoint"],
        payload=payload,
        outdir=rawdir,
        label=label,
        attempts_limit=int(campaign["nvidia"]["generation"]["attempts"]),
        role=role,
    )
    try:
        return validator(raw, expected), [label]
    except ValueError as exc:
        repair_label = label + "-json-repair"
        repair_payload = {
            "model": model["id"],
            "messages": [
                {
                    "role": "system",
                    "content": "Repair the requested JSON only. Return complete JSON.",
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                        + "\n\nMALFORMED OUTPUT:\n"
                        + raw
                        + f"\n\nPARSER PROBLEM: {exc}"
                    ),
                },
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False,
        }
        if "reasoning_effort" in model:
            repair_payload["reasoning_effort"] = "low"
        repaired, _, _ = call_api(
            key=key,
            endpoint=campaign["nvidia"]["endpoint"],
            payload=repair_payload,
            outdir=rawdir,
            label=repair_label,
            attempts_limit=int(campaign["nvidia"]["generation"]["attempts"]),
            role=role + "-repair",
        )
        return validator(repaired, expected), [label, repair_label]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/tournament-config.json",
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    campaign = json.loads((ROOT / cfg["campaign_config"]).read_text())
    final_eval = json.loads(
        (ROOT / cfg["final_eval_config"]).read_text()
    )
    valid_scenarios = {
        row["id"] for row in final_eval["scenarios"]
    }
    if len(valid_scenarios) != int(cfg["tournament"]["scenario_count"]):
        raise RuntimeError("final heldout scenario count/config mismatch")
    if args.scenario not in valid_scenarios:
        raise SystemExit(f"invalid scenario {args.scenario}")

    key = __import__("os").environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY/NVKEY is required")

    scenario, rows = load_cases(
        cfg,
        args.scenario,
        args.responses_root,
    )
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rawdir = out / "provider"
    rawdir.mkdir(parents=True)

    normalizer = campaign["nvidia"]["models"]["compiler"]
    classifier = campaign["nvidia"]["models"]["critic"]
    adjudicator = campaign["nvidia"]["models"]["adjudicator"]
    norm_chunk = int(cfg["tournament"]["normalizer_chunk_size"])
    class_chunk = int(cfg["tournament"]["classifier_chunk_size"])

    summaries: dict[str, str] = {}
    call_refs = []
    for offset in range(0, len(rows), norm_chunk):
        chunk = rows[offset : offset + norm_chunk]
        expected = [row["opaque_id"] for row in chunk]
        label = f"{args.scenario}-normalize-{offset // norm_chunk + 1:02d}"
        result, refs = call_json_with_repair(
            key=key,
            campaign=campaign,
            model=normalizer,
            rawdir=rawdir,
            label=label,
            prompt=normalize_prompt(scenario, chunk),
            system=(
                "You are a behavior neutralizer. Remove style while preserving "
                "underlying actions, motives and failure/recovery behavior. Return JSON."
            ),
            max_tokens=6144,
            validator=validate_summaries,
            expected=expected,
            role="tournament-normalizer",
        )
        summaries.update(result)
        call_refs.extend(refs)

    for row in rows:
        row["behavior_summary"] = summaries[row["opaque_id"]]

    for mode in ("raw", "behavior"):
        predictions = {}
        for offset in range(0, len(rows), class_chunk):
            chunk = rows[offset : offset + class_chunk]
            expected = [row["opaque_id"] for row in chunk]
            label = (
                f"{args.scenario}-{mode}-classify-"
                f"{offset // class_chunk + 1:02d}"
            )
            result, refs = call_json_with_repair(
                key=key,
                campaign=campaign,
                model=classifier,
                rawdir=rawdir,
                label=label,
                prompt=classifier_prompt(scenario, chunk, mode=mode),
                system=(
                    "You are a blind behavioral classifier. Use behavioral evidence "
                    "rather than prose flair. Return JSON only."
                ),
                max_tokens=6144,
                validator=validate_classifications,
                expected=expected,
                role=f"tournament-{mode}-classifier",
            )
            predictions.update(result)
            call_refs.extend(refs)
        for row in rows:
            row[f"{mode}_prediction"] = predictions[row["opaque_id"]]

    # Level 2 is deliberately separate: the correct lineage is supplied to the
    # judge, as required by the hierarchical VILLAGE-001E contract.
    for mode in ("raw", "behavior"):
        sibling_predictions = {}
        for offset in range(0, len(rows), class_chunk):
            chunk = rows[offset : offset + class_chunk]
            expected = [row["opaque_id"] for row in chunk]
            label = (
                f"{args.scenario}-{mode}-sibling-"
                f"{offset // class_chunk + 1:02d}"
            )
            result, refs = call_json_with_repair(
                key=key,
                campaign=campaign,
                model=classifier,
                rawdir=rawdir,
                label=label,
                prompt=sibling_prompt(scenario, chunk, mode=mode),
                system=(
                    "You are the second-level blind sibling classifier in a hierarchical "
                    "behavior tournament. The correct ancestral lineage is supplied; "
                    "infer only sibling A/B/C. Return JSON only."
                ),
                max_tokens=4096,
                validator=validate_sibling_classifications,
                expected=expected,
                role=f"tournament-{mode}-sibling-classifier",
            )
            sibling_predictions.update(result)
            call_refs.extend(refs)
        for row in rows:
            row[f"{mode}_sibling_prediction"] = sibling_predictions[
                row["opaque_id"]
            ]

    uncertain = [
        row
        for row in rows
        if row["behavior_prediction"]["lineage_confidence"] <= 1
        or row["behavior_prediction"]["sibling_confidence"] <= 1
    ]
    audits = {}
    for offset in range(0, len(uncertain), class_chunk):
        chunk = uncertain[offset : offset + class_chunk]
        expected = [row["opaque_id"] for row in chunk]
        label = (
            f"{args.scenario}-behavior-audit-"
            f"{offset // class_chunk + 1:02d}"
        )
        result, refs = call_json_with_repair(
            key=key,
            campaign=campaign,
            model=adjudicator,
            rawdir=rawdir,
            label=label,
            prompt=adjudicator_prompt(scenario, chunk),
            system=(
                "You are a sparse independent adjudicator for low-confidence blind "
                "behavior classification. Return JSON only."
            ),
            max_tokens=6144,
            validator=validate_audits,
            expected=expected,
            role="tournament-behavior-audit",
        )
        audits.update(result)
        call_refs.extend(refs)

    records = []
    raw_lineage_correct = 0
    behavior_lineage_correct = 0
    raw_sibling_correct_given_lineage = 0
    behavior_sibling_correct_given_lineage = 0
    for row in rows:
        raw_pred = row["raw_prediction"]
        behavior_pred = row["behavior_prediction"]
        raw_sibling = row["raw_sibling_prediction"]
        behavior_sibling = row["behavior_sibling_prediction"]
        raw_correct = raw_pred["predicted_lineage"] == row["lineage"]
        behavior_correct = (
            behavior_pred["predicted_lineage"] == row["lineage"]
        )
        raw_lineage_correct += int(raw_correct)
        behavior_lineage_correct += int(behavior_correct)
        raw_sibling_correct_given_lineage += int(
            raw_sibling["predicted_sibling"] == row["sibling"]
        )
        behavior_sibling_correct_given_lineage += int(
            behavior_sibling["predicted_sibling"] == row["sibling"]
        )
        records.append(
            {
                "opaque_id": row["opaque_id"],
                "member_id": row["member_id"],
                "lineage": row["lineage"],
                "sibling": row["sibling"],
                "adapter_sha256": row["adapter_sha256"],
                "response": row["response"],
                "behavior_summary": row["behavior_summary"],
                "raw_prediction": raw_pred,
                "behavior_prediction": behavior_pred,
                "raw_sibling_prediction": raw_sibling,
                "behavior_sibling_prediction": behavior_sibling,
                "super_audit": audits.get(row["opaque_id"]),
            }
        )

    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "stage": "hierarchical-blind-tournament-scenario",
        "issue": 14,
        "scenario": scenario,
        "fresh_after_preference_training": True,
        "final_eval_config_sha256": scenario[
            "final_eval_config_sha256"
        ],
        "providers_saw_member_ids": False,
        "opaque_order": "sha256(scenario_id:member_id)",
        "normalizer_model": normalizer["id"],
        "classifier_model": classifier["id"],
        "adjudicator_model": adjudicator["id"],
        "counts": {
            "members": len(records),
            "super_audited_low_confidence": len(audits),
        },
        "primary_metrics": {
            "raw_lineage_accuracy": raw_lineage_correct / len(records),
            "behavior_lineage_accuracy": (
                behavior_lineage_correct / len(records)
            ),
            "raw_sibling_accuracy_given_correct_lineage": (
                raw_sibling_correct_given_lineage / len(records)
            ),
            "behavior_sibling_accuracy_given_correct_lineage": (
                behavior_sibling_correct_given_lineage / len(records)
            ),
        },
        "provider_call_ids": call_refs,
        "records": records,
    }
    write_json(out / "scenario.json", result)
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "metrics": result["primary_metrics"],
                "super_audits": len(audits),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
