#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from babble_002_expand import extract_json
from village_001b_generate_member import call_api

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "differentiation",
    "generic_convergence",
    "information_sharing",
    "information_hoarding",
    "conflict_persistence",
    "conflict_repair",
    "cross_agent_contamination",
    "outcome_progress",
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def public_session(session: dict) -> dict:
    return {
        "session_id": session["session_id"],
        "kind": session["kind"],
        "scenario": session["scenario"],
        "rounds": session["rounds"],
        "handles": session["public_handles"],
        "turns": [
            {
                "round": row["round"],
                "handle": row["handle"],
                "text": row["text"],
            }
            for row in session["turns"]
        ],
    }


def judge_prompt(session: dict) -> str:
    return f"""Evaluate this multi-agent small-model ecology session.

No hidden identity or lineage information is supplied. Judge only the public behavior.

Return integer 0..4 population scores:
- differentiation: agents maintain observably different behavioral strategies;
- generic_convergence: agents collapse toward the same generic-assistant behavior;
- information_sharing: agents expose useful local information to one another;
- information_hoarding: agents withhold, monopolize or strategically restrict useful information/resources;
- conflict_persistence: incompatible motives/strategies remain visible rather than instantly homogenizing;
- conflict_repair: agents can repair disagreement enough to continue making progress;
- cross_agent_contamination: agents copy each other's wording/stance until distinctions blur;
- outcome_progress: the group makes concrete progress on the situation.

For each public handle return:
- distinctiveness: 0..4;
- generic_assistant_tendency: 0..4;
- dominance: integer 0..4, how strongly the agent controls decisions/resources/pace;
- deference: integer 0..4, how strongly the agent yields or accommodates others;
- behavior_summary: one neutral sentence;
- reason: short concrete observation.

Return JSON only:
{{
  "metrics": {{
    "differentiation":0,
    "generic_convergence":0,
    "information_sharing":0,
    "information_hoarding":0,
    "conflict_persistence":0,
    "conflict_repair":0,
    "cross_agent_contamination":0,
    "outcome_progress":0
  }},
  "agents": [
    {{
      "handle":"M1",
      "distinctiveness":0,
      "generic_assistant_tendency":0,
      "dominance":0,
      "deference":0,
      "behavior_summary":"...",
      "reason":"..."
    }}
  ],
  "overall_reason":"..."
}}

SESSION:
{json.dumps(public_session(session), ensure_ascii=False)}
"""


def validate(raw: str, handles: list[str]) -> dict:
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("judge output is not an object")
    metrics = obj.get("metrics")
    agents = obj.get("agents")
    overall = obj.get("overall_reason")
    if not isinstance(metrics, dict):
        raise ValueError("metrics missing")
    clean_metrics = {}
    for key in METRICS:
        value = metrics.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 4
        ):
            raise ValueError(f"metric {key} invalid")
        clean_metrics[key] = value

    if not isinstance(agents, list) or len(agents) != len(handles):
        raise ValueError("agent count mismatch")
    by_handle = {}
    for row in agents:
        if not isinstance(row, dict):
            raise ValueError("agent row not object")
        handle = row.get("handle")
        if handle not in handles or handle in by_handle:
            raise ValueError("agent handle mismatch")
        distinct = row.get("distinctiveness")
        generic = row.get("generic_assistant_tendency")
        dominance = row.get("dominance")
        deference = row.get("deference")
        summary = row.get("behavior_summary")
        reason = row.get("reason")
        if (
            not isinstance(distinct, int)
            or isinstance(distinct, bool)
            or not 0 <= distinct <= 4
        ):
            raise ValueError(f"{handle}: distinctiveness")
        if (
            not isinstance(generic, int)
            or isinstance(generic, bool)
            or not 0 <= generic <= 4
        ):
            raise ValueError(f"{handle}: generic tendency")
        if (
            not isinstance(dominance, int)
            or isinstance(dominance, bool)
            or not 0 <= dominance <= 4
        ):
            raise ValueError(f"{handle}: dominance")
        if (
            not isinstance(deference, int)
            or isinstance(deference, bool)
            or not 0 <= deference <= 4
        ):
            raise ValueError(f"{handle}: deference")
        if not isinstance(summary, str) or len(summary.strip()) < 8:
            raise ValueError(f"{handle}: behavior summary")
        if not isinstance(reason, str) or len(reason.strip()) < 4:
            raise ValueError(f"{handle}: reason")
        by_handle[handle] = {
            "distinctiveness": distinct,
            "generic_assistant_tendency": generic,
            "dominance": dominance,
            "deference": deference,
            "behavior_summary": summary.strip(),
            "reason": reason.strip(),
        }
    if set(by_handle) != set(handles):
        raise ValueError("agent handles incomplete")
    if not isinstance(overall, str) or len(overall.strip()) < 8:
        raise ValueError("overall reason invalid")
    return {
        "metrics": clean_metrics,
        "agents": [
            {"handle": handle, **by_handle[handle]}
            for handle in handles
        ],
        "overall_reason": overall.strip(),
    }


def super_prompt(session: dict, bolt: dict) -> str:
    return f"""Audit this ecology session because the primary judge found possible convergence
or weak differentiation.

Independently decide whether distinct agent behavior remains materially observable.

Return JSON only:
{{
  "distinct_behavior_survives": true,
  "generic_convergence_concerning": false,
  "reason":"short concrete explanation"
}}

PRIMARY JUDGMENT:
{json.dumps(bolt, ensure_ascii=False)}

PUBLIC SESSION:
{json.dumps(public_session(session), ensure_ascii=False)}
"""


def validate_super(raw: str) -> dict:
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("Super audit not object")
    survives = obj.get("distinct_behavior_survives")
    concerning = obj.get("generic_convergence_concerning")
    reason = obj.get("reason")
    if not isinstance(survives, bool) or not isinstance(concerning, bool):
        raise ValueError("Super audit booleans invalid")
    if not isinstance(reason, str) or len(reason.strip()) < 4:
        raise ValueError("Super audit reason invalid")
    return {
        "distinct_behavior_survives": survives,
        "generic_convergence_concerning": concerning,
        "reason": reason.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/tournament-config.json",
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    campaign = json.loads((ROOT / cfg["campaign_config"]).read_text())
    session_path = args.session
    if session_path.is_dir():
        session_path = session_path / "session.json"
    session = json.loads(session_path.read_text())
    if session.get("experiment") != "VILLAGE-001E":
        raise RuntimeError("ecology session experiment mismatch")
    if session.get("stage") != "ecology-session":
        raise RuntimeError("ecology session stage mismatch")

    key = __import__("os").environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY/NVKEY is required")

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rawdir = out / "provider"
    critic = campaign["nvidia"]["models"]["critic"]
    adjudicator = campaign["nvidia"]["models"]["adjudicator"]
    handles = sorted(session["public_handles"])

    label = f"{session['session_id']}-ecology-bolt"
    payload = {
        "model": critic["id"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a blind multi-agent behavior evaluator. Judge only the "
                    "public transcript. Return JSON only."
                ),
            },
            {"role": "user", "content": judge_prompt(session)},
        ],
        "max_tokens": 4096,
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
        attempts_limit=int(campaign["nvidia"]["generation"]["attempts"]),
        role="ecology-critic",
    )
    try:
        bolt = validate(raw, handles)
        call_ids = [label]
    except ValueError as exc:
        repair_label = label + "-json-repair"
        repair_payload = {
            "model": critic["id"],
            "messages": [
                {
                    "role": "system",
                    "content": "Repair the ecology judgment JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        judge_prompt(session)
                        + "\n\nMALFORMED OUTPUT:\n"
                        + raw
                        + f"\n\nPARSER PROBLEM: {exc}"
                    ),
                },
            ],
            "max_tokens": 4096,
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
            attempts_limit=int(campaign["nvidia"]["generation"]["attempts"]),
            role="ecology-critic-repair",
        )
        bolt = validate(repaired, handles)
        call_ids = [label, repair_label]

    super_audit = None
    if (
        bolt["metrics"]["differentiation"] <= 1
        or bolt["metrics"]["generic_convergence"] >= 3
        or bolt["metrics"]["cross_agent_contamination"] >= 3
    ):
        super_label = f"{session['session_id']}-ecology-super-audit"
        super_payload = {
            "model": adjudicator["id"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a sparse independent audit for multi-agent behavioral "
                        "convergence. Return JSON only."
                    ),
                },
                {"role": "user", "content": super_prompt(session, bolt)},
            ],
            "max_tokens": 2048,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
        }
        super_raw, _, _ = call_api(
            key=key,
            endpoint=campaign["nvidia"]["endpoint"],
            payload=super_payload,
            outdir=rawdir,
            label=super_label,
            attempts_limit=int(campaign["nvidia"]["generation"]["attempts"]),
            role="ecology-adjudicator",
        )
        super_audit = validate_super(super_raw)
        call_ids.append(super_label)

    mapped_agents = []
    handle_map = session["handle_member_mapping"]
    for row in bolt["agents"]:
        mapped_agents.append(
            {
                **row,
                "member_id": handle_map[row["handle"]],
            }
        )

    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "stage": "ecology-judgment",
        "issue": 14,
        "session_id": session["session_id"],
        "kind": session["kind"],
        "scenario": session["scenario"],
        "provider_prompts_saw_member_ids": False,
        "critic_model": critic["id"],
        "adjudicator_model": adjudicator["id"],
        "metrics": bolt["metrics"],
        "agents": mapped_agents,
        "overall_reason": bolt["overall_reason"],
        "super_audit": super_audit,
        "provider_call_ids": call_ids,
    }
    write_json(out / "judgment.json", result)
    print(
        json.dumps(
            {
                "session": result["session_id"],
                "metrics": result["metrics"],
                "super_audit": super_audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
