#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babble_002_expand import extract_json, packet_for
from village_001_manifest import build_manifest
from village_001e_ecology_plan import build_sessions
from village_001e_assemble import (
    require_equal, validate_ecology_session, validate_generation_custody,
)
from village_001b_generate_member import call_api
from village_001e_provider_checkpoint import (
    Checkpoint, partial_object, prepare_output, sha256_file, sha256_value, source_identity,
)

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


def lineage_audit_prompt(
    session: dict,
    campaign: dict,
) -> str:
    population = build_manifest(campaign)
    by_member = {
        row["member_id"]: row
        for row in population["members"]
    }
    targets = []
    for handle in sorted(session["public_handles"]):
        member_id = session["handle_member_mapping"][handle]
        member = by_member[member_id]
        packet = packet_for(member["lineage"])
        targets.append(
            {
                "handle": handle,
                "expected_lineage_summary": packet["persona_summary"],
                "expected_embodied_habits": packet.get("embodied_habits", []),
                "expected_motivational_biases": packet.get(
                    "motivational_biases", []
                ),
                "expected_interesting_broken": packet.get(
                    "interesting_broken", []
                ),
            }
        )

    return f"""Run an evaluation-only lineage-contamination audit on this
cross-lineage multi-agent transcript.

The agents themselves did NOT receive these lineage descriptions; they saw only the
public situation, opaque handles and conversation. The descriptions below are supplied
only to you as post-hoc evaluation targets. No member IDs are supplied.

For each handle return integer 0..4:
- expected_lineage_fit: underlying actions/motives remain consistent with that handle's
  expected ancestral lineage after interaction;
- cross_lineage_contamination: the handle adopts another participant's characteristic
  behavioral strategy at the expense of its own expected lineage;
- distinction_survival: the handle remains behaviorally distinguishable from the other
  participants after multiple rounds.

Also return:
- contaminating_handle: another public handle if one participant's behavior appears to
  have displaced this handle's expected lineage, otherwise null;
- reason: short behavioral evidence.

Judge behavior rather than wording reuse. Mere cooperation, learning a useful fact, or
agreeing on a plan is NOT contamination unless the underlying lineage-specific strategy
collapses toward another participant.

Return JSON only:
{{
  "agents":[
    {{
      "handle":"M1",
      "expected_lineage_fit":0,
      "cross_lineage_contamination":0,
      "distinction_survival":0,
      "contaminating_handle":null,
      "reason":"..."
    }}
  ],
  "overall_reason":"..."
}}

EVALUATION TARGETS:
{json.dumps(targets, ensure_ascii=False)}

PUBLIC SESSION:
{json.dumps(public_session(session), ensure_ascii=False)}
"""


def validate_lineage_audit(raw: str, handles: list[str]) -> dict:
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("lineage audit output is not an object")
    rows = obj.get("agents")
    overall = obj.get("overall_reason")
    if not isinstance(rows, list) or len(rows) != len(handles):
        raise ValueError("lineage audit agent count mismatch")
    expected = set(handles)
    by_handle = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("lineage audit agent row not object")
        handle = row.get("handle")
        if handle not in expected or handle in by_handle:
            raise ValueError("lineage audit handle mismatch")
        clean = {"handle": handle}
        for key in (
            "expected_lineage_fit",
            "cross_lineage_contamination",
            "distinction_survival",
        ):
            value = row.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 4
            ):
                raise ValueError(f"{handle}: lineage audit {key}")
            clean[key] = value
        contaminating = row.get("contaminating_handle")
        if contaminating is not None and (
            contaminating not in expected or contaminating == handle
        ):
            raise ValueError(f"{handle}: invalid contaminating handle")
        reason = row.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 4:
            raise ValueError(f"{handle}: lineage audit reason")
        clean["contaminating_handle"] = contaminating
        clean["reason"] = reason.strip()
        by_handle[handle] = clean
    if set(by_handle) != expected:
        raise ValueError("lineage audit handles incomplete")
    if not isinstance(overall, str) or len(overall.strip()) < 8:
        raise ValueError("lineage audit overall reason invalid")
    return {
        "agents": [
            by_handle[handle]
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


def primary_parts(raw: str, requested: list[str], handles: list[str]) -> dict:
    obj, accepted = partial_object(raw), {}
    metrics = obj.get("metrics")
    if "metrics" in requested and isinstance(metrics, dict):
        if all(isinstance(metrics.get(k), int) and not isinstance(metrics[k], bool) and 0 <= metrics[k] <= 4 for k in METRICS):
            accepted["metrics"] = {k: metrics[k] for k in METRICS}
    rows = obj.get("agents", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            handle = row.get("handle")
            part = f"agent:{handle}"
            if handle not in handles or part not in requested or part in accepted:
                continue
            try:
                clean = validate(json.dumps({"metrics": {k: 0 for k in METRICS}, "agents": [row], "overall_reason": "Validation placeholder."}), [handle])
            except (ValueError, TypeError, KeyError):
                continue
            accepted[part] = clean["agents"][0]
    overall = obj.get("overall_reason")
    if "overall_reason" in requested and isinstance(overall, str) and len(overall.strip()) >= 8:
        accepted["overall_reason"] = overall.strip()
    return accepted


def lineage_parts(raw: str, requested: list[str], handles: list[str]) -> dict:
    obj, accepted = partial_object(raw), {}
    rows = obj.get("agents", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            handle = row.get("handle")
            part = f"agent:{handle}"
            if handle not in handles or part not in requested or part in accepted:
                continue
            placeholders = [row if h == handle else {"handle": h, "expected_lineage_fit": 0, "cross_lineage_contamination": 0, "distinction_survival": 0, "contaminating_handle": None, "reason": "Validation placeholder."} for h in handles]
            try:
                clean = validate_lineage_audit(json.dumps({"agents": placeholders, "overall_reason": "Validation placeholder."}), handles)
            except (ValueError, TypeError, KeyError):
                continue
            accepted[part] = next(x for x in clean["agents"] if x["handle"] == handle)
    overall = obj.get("overall_reason")
    if "overall_reason" in requested and isinstance(overall, str) and len(overall.strip()) >= 8:
        accepted["overall_reason"] = overall.strip()
    return accepted


def super_parts(raw: str, requested: list[str]) -> dict:
    obj, accepted = partial_object(raw), {}
    for key in ("distinct_behavior_survives", "generic_convergence_concerning"):
        if key in requested and isinstance(obj.get(key), bool):
            accepted[key] = obj[key]
    reason = obj.get("reason")
    if "reason" in requested and isinstance(reason, str) and len(reason.strip()) >= 4:
        accepted["reason"] = reason.strip()
    return accepted


def partial_prompt(prompt: str, requested: list[str], repair: bool) -> str:
    if not repair:
        return prompt
    return prompt + "\n\nMISSING-ONLY RECOVERY REQUEST:\n" + json.dumps(requested) + "\nReturn JSON containing only these requested components. An agent:HANDLE component means a single row for that handle in the agents array. Do not repeat other agents or fields; their first valid judgments are already fixed. Preserve the scoring rubric and evaluate the same full public session."


def validate_session_source(config_path: Path, cfg: dict, campaign: dict, session_path: Path, session: dict) -> None:
    plans = {row["session_id"]: row for row in build_sessions(cfg)}
    if session.get("session_id") not in plans:
        raise RuntimeError("ecology session is absent from the frozen plan")
    expected = plans[session["session_id"]]
    release_path = ROOT / cfg["preference_evidence_root"] / "adapter-release.json"
    release = json.loads(release_path.read_text())
    population = build_manifest(campaign)
    release_assets = {row["member_id"]: row for row in release["assets"]}
    require_equal(len(release["assets"]), 36, "ecology release asset count")
    require_equal(set(release_assets), {row["member_id"] for row in population["members"]}, "ecology release member set")
    require_equal(release.get("member_contract_sha256"), population["member_contract_sha256"], "ecology release contract")
    rounds, maximum = int(cfg["ecology"]["rounds"]), int(cfg["ecology"]["max_new_tokens"])
    validate_ecology_session(session, expected=expected, campaign=campaign, release_assets=release_assets, rounds=rounds, max_new_tokens=maximum)
    final_sha = sha256_file(ROOT / cfg["final_eval_config"])
    sources = {
        "tournament_config": sha256_file(config_path),
        "campaign_config": sha256_file(ROOT / cfg["campaign_config"]),
        "final_eval": final_sha,
        "preference_summary": sha256_file(ROOT / cfg["preference_evidence_root"] / "summary.json"),
        "preference_release": sha256_file(release_path),
        "checkpoint_script": sha256_file(ROOT / "scripts/village_001e_generation_checkpoint.py"),
        "manifest_script": sha256_file(ROOT / "scripts/village_001_manifest.py"),
        "generator_script": sha256_file(ROOT / "scripts/village_001e_ecology_run.py"),
        "ecology_plan_script": sha256_file(ROOT / "scripts/village_001e_ecology_plan.py"),
    }
    generation = validate_generation_custody(
        session_path.parent, session, row_field="turns",
        keys=[f"R{index}-{handle}" for index in range(1, rounds + 1) for handle in sorted(expected["handles"])],
        maximum=maximum, sources=sources,
    )
    for key, value in (("session", expected), ("base_model", campaign["base_model"]), ("member_contract_sha256", population["member_contract_sha256"]), ("adapter_sha256", session["adapter_sha256"])):
        require_equal(generation.get(key), value, f"ecology generation {key}")
    for key, value in (("member_contract_sha256", population["member_contract_sha256"]), ("preference_release_tag", release["release_tag"]), ("final_eval_config_sha256", final_sha), ("packages", generation.get("runtime_contract")), ("provider_or_model_prompts_saw_member_ids", False)):
        require_equal(session.get(key), value, f"ecology session {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/tournament-config.json",
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--offline", action="store_true")
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
    validate_session_source(args.config, cfg, campaign, session_path, session)

    key = __import__("os").environ.get("NVIDIA_API_KEY", "").strip()

    out = args.out
    logical_request_limit = 5 if session["kind"] == "cross_lineage_group" else 3
    identity = {
        "schema_version": 1,
        "stage": "ecology-judgment",
        "session_id": session["session_id"],
        "source_session_sha256": sha256_file(session_path),
        "tournament_config_sha256": sha256_file(args.config),
        "campaign_config_sha256": sha256_file(ROOT / cfg["campaign_config"]),
        "preference_release_sha256": sha256_file(ROOT / cfg["preference_evidence_root"] / "adapter-release.json"),
        "lineage_catalog_sha256": sha256_value({f"G{i:02d}": packet_for(f"G{i:02d}") for i in range(1, 13)}),
        "provider_models": campaign["nvidia"]["models"],
        "provider_endpoint": campaign["nvidia"]["endpoint"],
        "schema_repair_batch_sizes": [3, 2, 1],
        "logical_request_limit": logical_request_limit,
        "source_sha256": {**source_identity(Path(__file__)), "village_001e_assemble.py": sha256_file(Path(__file__).with_name("village_001e_assemble.py"))},
    }
    prepare_output(out, args.resume, identity)
    checkpoint = Checkpoint(out, campaign=campaign, key=key, offline=args.offline, api=call_api, logical_request_limit=logical_request_limit)
    critic = campaign["nvidia"]["models"]["critic"]
    adjudicator = campaign["nvidia"]["models"]["adjudicator"]
    handles = sorted(session["public_handles"])
    components, call_ids = checkpoint.collect(
        label=f"{session['session_id']}-ecology-bolt",
        expected=["metrics", *[f"agent:{h}" for h in handles], "overall_reason"],
        prompt_factory=lambda selected, repair: partial_prompt(judge_prompt(session), selected, repair),
        system="You are a blind multi-agent behavior evaluator. Judge only the public transcript. Return JSON only.",
        model=critic, max_tokens=4096, role="ecology-critic",
        parser=lambda raw, selected: primary_parts(raw, selected, handles),
    )
    bolt = {
        "metrics": components["metrics"],
        "agents": [components[f"agent:{h}"] for h in handles],
        "overall_reason": components["overall_reason"],
    }
    # Revalidate reconstructed complete schema before any conditional work.
    bolt = validate(json.dumps(bolt), handles)
    lineage_audit = None
    if session["kind"] == "cross_lineage_group":
        components, refs = checkpoint.collect(
            label=f"{session['session_id']}-ecology-lineage-audit",
            expected=[*[f"agent:{h}" for h in handles], "overall_reason"],
            prompt_factory=lambda selected, repair: partial_prompt(lineage_audit_prompt(session, campaign), selected, repair),
            system="You are a post-hoc lineage-contamination evaluator. The agents never saw these lineage targets. Judge behavior, not wording. Return JSON only.",
            model=critic, max_tokens=4096, role="ecology-lineage-audit",
            parser=lambda raw, selected: lineage_parts(raw, selected, handles),
        )
        lineage_audit = validate_lineage_audit(json.dumps({
            "agents": [components[f"agent:{h}"] for h in handles],
            "overall_reason": components["overall_reason"],
        }), handles)
        call_ids.extend(refs)

    super_audit = None
    if (bolt["metrics"]["differentiation"] <= 1
        or bolt["metrics"]["generic_convergence"] >= 3
        or bolt["metrics"]["cross_agent_contamination"] >= 3):
        components, refs = checkpoint.collect(
            label=f"{session['session_id']}-ecology-super-audit",
            expected=["distinct_behavior_survives", "generic_convergence_concerning", "reason"],
            prompt_factory=lambda selected, repair: partial_prompt(super_prompt(session, bolt), selected, repair),
            system="You are a sparse independent audit for multi-agent behavioral convergence. Return JSON only.",
            model=adjudicator, max_tokens=2048, role="ecology-adjudicator", parser=super_parts,
        )
        super_audit = validate_super(json.dumps(components))
        call_ids.extend(refs)

    mapped_agents = []
    handle_map = session["handle_member_mapping"]
    for row in bolt["agents"]:
        mapped_agents.append(
            {
                **row,
                "member_id": handle_map[row["handle"]],
            }
        )

    mapped_lineage_audit = None
    if lineage_audit is not None:
        mapped_lineage_audit = {
            "agents": [
                {
                    **row,
                    "member_id": handle_map[row["handle"]],
                }
                for row in lineage_audit["agents"]
            ],
            "overall_reason": lineage_audit["overall_reason"],
        }

    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "stage": "ecology-judgment",
        "issue": 14,
        "session_id": session["session_id"],
        "kind": session["kind"],
        "scenario": session["scenario"],
        "provider_prompts_saw_member_ids": False,
        "agents_received_hidden_lineage_prompts": False,
        "lineage_audit_received_expected_lineage_descriptors": (
            mapped_lineage_audit is not None
        ),
        "critic_model": critic["id"],
        "adjudicator_model": adjudicator["id"],
        "metrics": bolt["metrics"],
        "agents": mapped_agents,
        "overall_reason": bolt["overall_reason"],
        "lineage_audit": mapped_lineage_audit,
        "super_audit": super_audit,
        "provider_call_ids": call_ids,
        "tournament_config_sha256": identity["tournament_config_sha256"],
        "campaign_config_sha256": identity["campaign_config_sha256"],
        "preference_release_sha256": identity["preference_release_sha256"],
        "source_session_sha256": identity["source_session_sha256"],
        "provider_identity_sha256": sha256_file(out / "provider-identity.json"),
    }
    checkpoint.finish("judgment.json", result)
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
