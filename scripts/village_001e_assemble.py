#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from village_001_manifest import build_manifest
from village_001e_ecology_plan import build_sessions

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def hash_inventory(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "hashes.sha256":
            continue
        rows.append(f"{sha256_file(path)}  {rel}")
    return "\n".join(rows) + "\n"


def find_unique(
    root: Path,
    filename: str,
    *,
    experiment: str,
    stage: str,
    id_key: str,
) -> dict[str, Path]:
    found: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob(filename):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if obj.get("experiment") != experiment or obj.get("stage") != stage:
            continue
        key = obj.get(id_key)
        if isinstance(key, str):
            found[key].append(path)
    duplicates = {key: paths for key, paths in found.items() if len(paths) != 1}
    if duplicates:
        raise RuntimeError(f"{filename} multiplicity error: {duplicates}")
    return {key: paths[0] for key, paths in found.items()}


def empty_matrix(labels: list[str]) -> dict[str, dict[str, int]]:
    return {
        true: {pred: 0 for pred in labels}
        for true in labels
    }


def metric(values: list[float]) -> dict:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament-artifacts", type=Path, required=True)
    parser.add_argument("--ecology-artifacts", type=Path, required=True)
    parser.add_argument("--ecology-judgment-artifacts", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/tournament-config.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/village-001/final-evidence",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    campaign_path = ROOT / cfg["campaign_config"]
    campaign = json.loads(campaign_path.read_text())
    preference_cfg_path = ROOT / cfg["preference_config"]
    preference_cfg = json.loads(preference_cfg_path.read_text())
    population = build_manifest(campaign)
    members = population["members"]
    member_meta = {row["member_id"]: row for row in members}
    expected_members = list(member_meta)
    lineages = [f"G{i:02d}" for i in range(1, 13)]
    siblings = ["A", "B", "C"]
    scenarios = [
        f"T{i:02d}"
        for i in range(1, int(cfg["tournament"]["scenario_count"]) + 1)
    ]

    preference_root = ROOT / cfg["preference_evidence_root"]
    preference_summary_path = preference_root / "summary.json"
    preference_release_path = preference_root / "adapter-release.json"
    if not preference_summary_path.is_file() or not preference_release_path.is_file():
        raise RuntimeError("VILLAGE-001D preference evidence is missing")
    preference_summary = json.loads(preference_summary_path.read_text())
    preference_release = json.loads(preference_release_path.read_text())
    if preference_summary.get("members") != 36:
        raise RuntimeError("preference summary member count mismatch")
    if len(preference_release.get("assets", [])) != 36:
        raise RuntimeError("preference release manifest is incomplete")

    base_summary_path = ROOT / "experiments/persona-001/evidence/summary.json"
    if not base_summary_path.is_file():
        raise RuntimeError("PERSONA-001 untuned BASE evidence is missing")
    base_summary = json.loads(base_summary_path.read_text())
    measured_base_genericness = float(
        base_summary["raw_blind"]["control_scores"]["genericness"]
    )
    declared_base_genericness = float(
        cfg["tournament"]["base_genericness_reference"]
    )
    if abs(measured_base_genericness - declared_base_genericness) > 0.01:
        raise RuntimeError(
            "predeclared BASE genericness reference drift: "
            f"{declared_base_genericness} vs measured {measured_base_genericness}"
        )

    tournament = find_unique(
        args.tournament_artifacts,
        "scenario.json",
        experiment="VILLAGE-001E",
        stage="hierarchical-blind-tournament-scenario",
        id_key="scenario",
    )
    missing_scenarios = sorted(set(scenarios) - set(tournament))
    extra_scenarios = sorted(set(tournament) - set(scenarios))
    if missing_scenarios or extra_scenarios:
        raise RuntimeError(
            f"tournament scenario mismatch missing={missing_scenarios} "
            f"extra={extra_scenarios}"
        )

    raw_lineage = empty_matrix(lineages)
    behavior_lineage = empty_matrix(lineages)
    raw_sibling = empty_matrix(siblings)
    behavior_sibling = empty_matrix(siblings)
    raw_member = empty_matrix(expected_members)
    behavior_member = empty_matrix(expected_members)
    raw_lineage_correct = 0
    behavior_lineage_correct = 0
    raw_sibling_correct = 0
    behavior_sibling_correct = 0
    raw_member_correct = 0
    behavior_member_correct = 0
    total_records = 0
    family_stats = defaultdict(Counter)
    member_stats = {
        member_id: {
            "records": 0,
            "raw_lineage_hits": 0,
            "behavior_lineage_hits": 0,
            "raw_sibling_hits_given_correct_lineage": 0,
            "behavior_sibling_hits_given_correct_lineage": 0,
            "raw_member_hits": 0,
            "behavior_member_hits": 0,
            "raw_predicted_lineages": Counter(),
            "behavior_predicted_lineages": Counter(),
            "raw_predicted_members": Counter(),
            "behavior_predicted_members": Counter(),
        }
        for member_id in expected_members
    }
    super_audit_count = 0

    for scenario_id in scenarios:
        doc = json.loads(tournament[scenario_id].read_text())
        if doc["providers_saw_member_ids"] is not False:
            raise RuntimeError(f"{scenario_id}: tournament provider identity leak flag")
        records = doc.get("records")
        if not isinstance(records, list) or len(records) != 36:
            raise RuntimeError(f"{scenario_id}: tournament record count != 36")
        seen_members = set()
        for row in records:
            member_id = row.get("member_id")
            if member_id not in member_meta or member_id in seen_members:
                raise RuntimeError(f"{scenario_id}: invalid/duplicate member {member_id}")
            seen_members.add(member_id)
            meta = member_meta[member_id]
            if row.get("lineage") != meta["lineage"] or row.get("sibling") != meta["sibling"]:
                raise RuntimeError(f"{scenario_id}:{member_id}: truth identity mismatch")
            if not isinstance(row.get("opaque_id"), str):
                raise RuntimeError(f"{scenario_id}:{member_id}: missing opaque ID")
            raw = row["raw_prediction"]
            behavior = row["behavior_prediction"]
            raw_sibling_pred = row.get("raw_sibling_prediction")
            behavior_sibling_pred = row.get("behavior_sibling_prediction")
            if raw["predicted_lineage"] not in lineages:
                raise RuntimeError("invalid raw predicted lineage")
            if behavior["predicted_lineage"] not in lineages:
                raise RuntimeError("invalid behavior predicted lineage")
            if (
                not isinstance(raw_sibling_pred, dict)
                or raw_sibling_pred.get("predicted_sibling") not in siblings
            ):
                raise RuntimeError("invalid conditioned raw sibling prediction")
            if (
                not isinstance(behavior_sibling_pred, dict)
                or behavior_sibling_pred.get("predicted_sibling") not in siblings
            ):
                raise RuntimeError("invalid conditioned behavior sibling prediction")

            raw_lineage[meta["lineage"]][raw["predicted_lineage"]] += 1
            behavior_lineage[meta["lineage"]][behavior["predicted_lineage"]] += 1
            raw_correct = raw["predicted_lineage"] == meta["lineage"]
            behavior_correct = behavior["predicted_lineage"] == meta["lineage"]
            raw_lineage_correct += int(raw_correct)
            behavior_lineage_correct += int(behavior_correct)

            stats = member_stats[member_id]
            stats["records"] += 1
            stats["raw_lineage_hits"] += int(raw_correct)
            stats["behavior_lineage_hits"] += int(behavior_correct)
            stats["raw_predicted_lineages"][raw["predicted_lineage"]] += 1
            stats["behavior_predicted_lineages"][behavior["predicted_lineage"]] += 1

            raw_sibling_value = raw_sibling_pred["predicted_sibling"]
            behavior_sibling_value = behavior_sibling_pred["predicted_sibling"]
            raw_sibling[meta["sibling"]][raw_sibling_value] += 1
            behavior_sibling[meta["sibling"]][behavior_sibling_value] += 1
            raw_sibling_hit = raw_sibling_value == meta["sibling"]
            behavior_sibling_hit = behavior_sibling_value == meta["sibling"]
            raw_sibling_correct += int(raw_sibling_hit)
            behavior_sibling_correct += int(behavior_sibling_hit)
            stats["raw_sibling_hits_given_correct_lineage"] += int(raw_sibling_hit)
            stats["behavior_sibling_hits_given_correct_lineage"] += int(
                behavior_sibling_hit
            )

            raw_predicted_member = (
                f"V-{raw['predicted_lineage']}-{raw_sibling_value}"
            )
            behavior_predicted_member = (
                f"V-{behavior['predicted_lineage']}-{behavior_sibling_value}"
            )
            if (
                raw_predicted_member not in member_meta
                or behavior_predicted_member not in member_meta
            ):
                raise RuntimeError("derived predicted member outside canonical population")
            raw_member[member_id][raw_predicted_member] += 1
            behavior_member[member_id][behavior_predicted_member] += 1
            raw_member_hit = raw_predicted_member == member_id
            behavior_member_hit = behavior_predicted_member == member_id
            raw_member_correct += int(raw_member_hit)
            behavior_member_correct += int(behavior_member_hit)
            stats["raw_member_hits"] += int(raw_member_hit)
            stats["behavior_member_hits"] += int(behavior_member_hit)
            stats["raw_predicted_members"][raw_predicted_member] += 1
            stats["behavior_predicted_members"][behavior_predicted_member] += 1

            family = doc["scenario"]["family"]
            family_stats[family]["records"] += 1
            family_stats[family]["raw_lineage_hits"] += int(raw_correct)
            family_stats[family]["behavior_lineage_hits"] += int(behavior_correct)
            family_stats[family]["raw_sibling_hits"] += int(raw_sibling_hit)
            family_stats[family]["behavior_sibling_hits"] += int(
                behavior_sibling_hit
            )
            family_stats[family]["raw_member_hits"] += int(raw_member_hit)
            family_stats[family]["behavior_member_hits"] += int(
                behavior_member_hit
            )

            if row.get("super_audit") is not None:
                super_audit_count += 1
            total_records += 1

        if seen_members != set(expected_members):
            raise RuntimeError(f"{scenario_id}: incomplete member set")

    if total_records != 36 * 24:
        raise RuntimeError(f"tournament record count {total_records} != 864")

    raw_lineage_accuracy = raw_lineage_correct / total_records
    behavior_lineage_accuracy = behavior_lineage_correct / total_records
    raw_sibling_accuracy = raw_sibling_correct / total_records
    behavior_sibling_accuracy = behavior_sibling_correct / total_records
    raw_member_accuracy = raw_member_correct / total_records
    behavior_member_accuracy = behavior_member_correct / total_records
    family_accuracy = {
        family: {
            "records": counts["records"],
            "raw_lineage_accuracy": counts["raw_lineage_hits"] / counts["records"],
            "behavior_lineage_accuracy": (
                counts["behavior_lineage_hits"] / counts["records"]
            ),
            "raw_sibling_accuracy_given_correct_lineage": (
                counts["raw_sibling_hits"] / counts["records"]
            ),
            "behavior_sibling_accuracy_given_correct_lineage": (
                counts["behavior_sibling_hits"] / counts["records"]
            ),
            "raw_member_accuracy": counts["raw_member_hits"] / counts["records"],
            "behavior_member_accuracy": (
                counts["behavior_member_hits"] / counts["records"]
            ),
        }
        for family, counts in sorted(family_stats.items())
    }

    before_after = {}
    for member_id in expected_members:
        path = preference_root / "before-after" / f"{member_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing before-after evidence {path}")
        doc = json.loads(path.read_text())
        if doc.get("member_id") != member_id:
            raise RuntimeError(f"{member_id}: before-after member mismatch")
        before_after[member_id] = doc

    maturation_qualified = set(
        preference_summary["qualified_for_population_tournament"]
    )
    task_relevance_values = []
    genericness_values = []
    individual_pass = []
    individual_fail = []
    member_table = []

    for member_id in expected_members:
        meta = member_meta[member_id]
        comparison = before_after[member_id]
        after = comparison["after"]
        task_relevance = float(after["task_relevance"])
        genericness = float(after["genericness"])
        task_relevance_values.append(task_relevance)
        genericness_values.append(genericness)
        stats = member_stats[member_id]
        behavior_hits = int(stats["behavior_lineage_hits"])
        behavior_sibling_hits = int(
            stats["behavior_sibling_hits_given_correct_lineage"]
        )
        passes = (
            member_id in maturation_qualified
            and behavior_hits
            >= int(cfg["tournament"]["individual_min_behavior_lineage_hits"])
            and behavior_sibling_hits
            >= int(cfg["tournament"]["individual_min_behavior_sibling_hits"])
            and task_relevance
            >= float(cfg["tournament"]["individual_min_task_relevance"])
            and genericness
            <= float(cfg["tournament"]["individual_max_genericness"])
        )
        (individual_pass if passes else individual_fail).append(member_id)

        wrong_raw_lineage = Counter(stats["raw_predicted_lineages"])
        wrong_raw_lineage.pop(meta["lineage"], None)
        wrong_behavior_lineage = Counter(stats["behavior_predicted_lineages"])
        wrong_behavior_lineage.pop(meta["lineage"], None)
        wrong_raw_member = Counter(stats["raw_predicted_members"])
        wrong_raw_member.pop(member_id, None)
        wrong_behavior_member = Counter(stats["behavior_predicted_members"])
        wrong_behavior_member.pop(member_id, None)

        member_table.append(
            {
                "member_id": member_id,
                "lineage": meta["lineage"],
                "sibling": meta["sibling"],
                "raw_lineage_hits": stats["raw_lineage_hits"],
                "behavior_lineage_hits": behavior_hits,
                "raw_lineage_accuracy": stats["raw_lineage_hits"] / 24,
                "behavior_lineage_accuracy": behavior_hits / 24,
                "raw_sibling_accuracy_given_correct_lineage": (
                    stats["raw_sibling_hits_given_correct_lineage"] / 24
                ),
                "behavior_sibling_accuracy_given_correct_lineage": (
                    stats["behavior_sibling_hits_given_correct_lineage"] / 24
                ),
                "raw_member_accuracy": stats["raw_member_hits"] / 24,
                "behavior_member_accuracy": stats["behavior_member_hits"] / 24,
                "task_relevance": task_relevance,
                "genericness": genericness,
                "maturation_gate_pass": member_id in maturation_qualified,
                "individual_quality_pass": passes,
                "top_raw_lineage_confusions": wrong_raw_lineage.most_common(5),
                "top_behavior_lineage_confusions": (
                    wrong_behavior_lineage.most_common(5)
                ),
                "nearest_raw_member_confusions": wrong_raw_member.most_common(5),
                "nearest_behavior_member_confusions": (
                    wrong_behavior_member.most_common(5)
                ),
            }
        )

    mean_genericness = statistics.mean(genericness_values)
    population_gates = {
        "behavior_lineage_accuracy": {
            "value": behavior_lineage_accuracy,
            "threshold": float(cfg["tournament"]["lineage_gate"]),
            "pass": behavior_lineage_accuracy
            >= float(cfg["tournament"]["lineage_gate"]),
        },
        "behavior_sibling_accuracy_given_correct_lineage": {
            "value": behavior_sibling_accuracy,
            "threshold": float(
                cfg["tournament"]["sibling_gate_given_correct_lineage"]
            ),
            "pass": behavior_sibling_accuracy
            >= float(cfg["tournament"]["sibling_gate_given_correct_lineage"]),
        },
        "mean_genericness_below_base": {
            "value": mean_genericness,
            "threshold": float(
                cfg["tournament"]["base_genericness_reference"]
            ),
            "pass": mean_genericness
            < float(cfg["tournament"]["base_genericness_reference"]),
        },
        "individual_quality_members": {
            "value": len(individual_pass),
            "threshold": int(
                cfg["tournament"]["individual_quality_required"]
            ),
            "pass": len(individual_pass)
            >= int(cfg["tournament"]["individual_quality_required"]),
        },
    }
    usable_village = all(row["pass"] for row in population_gates.values())

    expected_sessions = {
        row["session_id"]: row for row in build_sessions(cfg)
    }
    sessions = find_unique(
        args.ecology_artifacts,
        "session.json",
        experiment="VILLAGE-001E",
        stage="ecology-session",
        id_key="session_id",
    )
    judgments = find_unique(
        args.ecology_judgment_artifacts,
        "judgment.json",
        experiment="VILLAGE-001E",
        stage="ecology-judgment",
        id_key="session_id",
    )
    if set(sessions) != set(expected_sessions) or set(judgments) != set(
        expected_sessions
    ):
        raise RuntimeError("ecology session/judgment set mismatch")

    ecology_metric_values: dict[str, list[float]] = defaultdict(list)
    ecology_by_kind: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ecology_agent: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ecology_super_audits = 0
    ecology_concerns = []
    ecology_rows = []

    for session_id in sorted(expected_sessions):
        session = json.loads(sessions[session_id].read_text())
        judgment = json.loads(judgments[session_id].read_text())
        if session.get("provider_or_model_prompts_saw_member_ids") is not False:
            raise RuntimeError(f"{session_id}: ecology runner identity leak flag")
        if judgment.get("provider_prompts_saw_member_ids") is not False:
            raise RuntimeError(f"{session_id}: ecology judge identity leak flag")
        if judgment["kind"] != session["kind"]:
            raise RuntimeError(f"{session_id}: ecology kind mismatch")

        for key, value in judgment["metrics"].items():
            ecology_metric_values[key].append(float(value))
            ecology_by_kind[session["kind"]][key].append(float(value))

        if judgment.get("super_audit") is not None:
            ecology_super_audits += 1
            audit = judgment["super_audit"]
            if audit.get("generic_convergence_concerning"):
                ecology_concerns.append(session_id)

        handle_map = session["handle_member_mapping"]
        if set(handle_map.values()) != set(session["members"]):
            raise RuntimeError(f"{session_id}: ecology handle/member mismatch")
        for agent in judgment["agents"]:
            member_id = agent["member_id"]
            if member_id not in member_meta:
                raise RuntimeError(f"{session_id}: unknown ecology member {member_id}")
            ecology_agent[member_id]["distinctiveness"].append(
                float(agent["distinctiveness"])
            )
            ecology_agent[member_id]["generic_assistant_tendency"].append(
                float(agent["generic_assistant_tendency"])
            )

        ecology_rows.append(
            {
                "session_id": session_id,
                "kind": session["kind"],
                "scenario_id": session["scenario"]["id"],
                "members": session["members"],
                "metrics": judgment["metrics"],
                "super_audit": judgment.get("super_audit"),
            }
        )

    ecology_summary = {
        "sessions": len(ecology_rows),
        "same_lineage_sessions": sum(
            1 for row in ecology_rows if row["kind"] == "same_lineage"
        ),
        "cross_lineage_sessions": sum(
            1 for row in ecology_rows if row["kind"] == "cross_lineage"
        ),
        "metrics": {
            key: metric(values)
            for key, values in ecology_metric_values.items()
        },
        "by_kind": {
            kind: {
                key: metric(values)
                for key, values in metrics.items()
            }
            for kind, metrics in ecology_by_kind.items()
        },
        "super_audit_sessions": ecology_super_audits,
        "super_convergence_concerns": ecology_concerns,
        "member_agent_metrics": {
            member_id: {
                key: metric(values)
                for key, values in metrics.items()
            }
            for member_id, metrics in ecology_agent.items()
        },
        "sessions_table": ecology_rows,
    }

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    tournament_dest = out / "tournament-scenarios"
    ecology_dest = out / "ecology"
    for scenario_id, path in tournament.items():
        shutil.copytree(path.parent, tournament_dest / scenario_id)
    for session_id in expected_sessions:
        session_dest = ecology_dest / "sessions" / session_id
        judge_dest = ecology_dest / "judgments" / session_id
        shutil.copytree(sessions[session_id].parent, session_dest)
        shutil.copytree(judgments[session_id].parent, judge_dest)

    confusion = {
        "lineage_labels": lineages,
        "sibling_labels": siblings,
        "member_labels": expected_members,
        "raw_lineage": raw_lineage,
        "behavior_lineage": behavior_lineage,
        "raw_sibling_given_correct_lineage": raw_sibling,
        "behavior_sibling_given_correct_lineage": behavior_sibling,
        "raw_member": raw_member,
        "behavior_member": behavior_member,
    }
    write_json(out / "confusion-matrices.json", confusion)
    write_json(out / "ecology-summary.json", ecology_summary)

    summary = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "issue": 14,
        "members": 36,
        "scenarios": 24,
        "blind_response_records": total_records,
        "providers_saw_member_ids": False,
        "primary_tournament_model": campaign["nvidia"]["models"]["critic"]["id"],
        "behavior_normalizer_model": campaign["nvidia"]["models"]["compiler"]["id"],
        "sparse_audit_model": campaign["nvidia"]["models"]["adjudicator"]["id"],
        "super_audited_low_confidence_records": super_audit_count,
        "raw": {
            "lineage_accuracy": raw_lineage_accuracy,
            "sibling_accuracy_given_correct_lineage": raw_sibling_accuracy,
            "member_accuracy": raw_member_accuracy,
        },
        "behavior_only": {
            "lineage_accuracy": behavior_lineage_accuracy,
            "sibling_accuracy_given_correct_lineage": behavior_sibling_accuracy,
            "member_accuracy": behavior_member_accuracy,
            "family_accuracy": family_accuracy,
        },
        "raw_vs_behavior_gap": {
            "lineage_accuracy": raw_lineage_accuracy
            - behavior_lineage_accuracy,
            "sibling_accuracy_given_correct_lineage": raw_sibling_accuracy
            - behavior_sibling_accuracy,
        },
        "final_quality": {
            "mean_task_relevance": statistics.mean(task_relevance_values),
            "mean_genericness": mean_genericness,
            "base_genericness_reference": float(
                cfg["tournament"]["base_genericness_reference"]
            ),
            "base_genericness_measured": measured_base_genericness,
            "base_genericness_source": cfg["tournament"][
                "base_genericness_source"
            ],
        },
        "population_gates": population_gates,
        "individual_quality_pass": individual_pass,
        "individual_quality_fail": individual_fail,
        "individual_quality_pass_count": len(individual_pass),
        "usable_village": usable_village,
        "members_table": member_table,
        "ecology": {
            "sessions": ecology_summary["sessions"],
            "metrics": ecology_summary["metrics"],
            "super_convergence_concerns": ecology_concerns,
        },
        "final_adapter_release": cfg["final_release_tag"],
    }
    write_json(out / "summary.json", summary)

    run_id = __import__("os").environ.get("GITHUB_RUN_ID")
    provenance = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "issue": 14,
        "workflow_run_id": int(run_id) if run_id else None,
        "workflow_run_url": (
            "https://github.com/techrote/neurounner/actions/runs/" + run_id
            if run_id
            else None
        ),
        "source_commit": __import__("os").environ.get("GITHUB_SHA"),
        "tournament_config_sha256": sha256_file(args.config),
        "campaign_config_sha256": sha256_file(campaign_path),
        "preference_config_sha256": sha256_file(preference_cfg_path),
        "preference_summary_sha256": sha256_file(preference_summary_path),
        "preference_release_manifest_sha256": sha256_file(preference_release_path),
        "untuned_base_summary_sha256": sha256_file(base_summary_path),
        "member_contract_sha256": population["member_contract_sha256"],
        "qualification_thresholds_predeclared": {
            key: value["threshold"]
            for key, value in population_gates.items()
        },
        "ecology_used_as_hard_population_gate": False,
    }
    write_json(out / "provenance.json", provenance)

    report = [
        "# VILLAGE-001E — final village qualification",
        "",
        "The final preference-matured cartridges were evaluated without prompt-time "
        "member/persona descriptions. Tournament providers received opaque response IDs; "
        "ecology models/judges received opaque participant handles.",
        "",
        "## Population gate",
        "",
        f"- usable village: **{'PASS' if usable_village else 'FAIL'}**",
        f"- behavior-only lineage accuracy: {behavior_lineage_accuracy:.3f} "
        f"(gate {float(cfg['tournament']['lineage_gate']):.3f})",
        f"- behavior-only sibling accuracy with correct lineage supplied: "
        f"{behavior_sibling_accuracy:.3f} "
        f"(gate {float(cfg['tournament']['sibling_gate_given_correct_lineage']):.3f})",
        f"- exact behavior-only 36-member accuracy: {behavior_member_accuracy:.3f}",
        f"- mean final genericness: {mean_genericness:.3f} "
        f"(must remain below untuned reference "
        f"{float(cfg['tournament']['base_genericness_reference']):.2f})",
        f"- individual quality checks: {len(individual_pass)}/36 "
        f"(gate {int(cfg['tournament']['individual_quality_required'])})",
        "",
        "## Raw versus behavior-only",
        "",
        f"- raw lineage accuracy: {raw_lineage_accuracy:.3f}",
        f"- behavior-only lineage accuracy: {behavior_lineage_accuracy:.3f}",
        f"- lineage style gap: "
        f"{raw_lineage_accuracy - behavior_lineage_accuracy:+.3f}",
        f"- raw sibling accuracy given correct lineage: {raw_sibling_accuracy:.3f}",
        f"- behavior-only sibling accuracy given correct lineage: "
        f"{behavior_sibling_accuracy:.3f}",
        "",
        "## Ecology",
        "",
        f"- sessions: {ecology_summary['sessions']} "
        "(12 same-lineage, 12 cross-lineage)",
    ]
    for key in (
        "differentiation",
        "generic_convergence",
        "information_sharing",
        "conflict_persistence",
        "conflict_repair",
        "cross_agent_contamination",
        "outcome_progress",
    ):
        if key in ecology_summary["metrics"]:
            report.append(
                f"- {key}: mean "
                f"{ecology_summary['metrics'][key]['mean']:.3f}/4"
            )
    report.extend(
        [
            f"- Super convergence concerns: {len(ecology_concerns)}/24 sessions",
            "",
            "Ecology is diagnostic evidence, not a retroactively invented hard gate. "
            "The population gate is exactly the predeclared lineage/sibling/genericness/"
            "individual-quality gate above.",
            "",
            "## Member summary",
            "",
            "| Member | Behavior lineage | Task relevance | Genericness | "
            "Maturation | Individual gate |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in member_table:
        report.append(
            f"| {row['member_id']} | {row['behavior_lineage_hits']}/24 | "
            f"{row['task_relevance']:.3f} | {row['genericness']:.3f} | "
            f"{'PASS' if row['maturation_gate_pass'] else 'FAIL'} | "
            f"{'PASS' if row['individual_quality_pass'] else 'FAIL'} |"
        )
    report.extend(
        [
            "",
            "Full confusion matrices, per-scenario raw provider evidence, ecology "
            "transcripts/judgments and hashes are retained beside this report.",
            "",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(report))
    (out / "hashes.sha256").write_text(hash_inventory(out))

    print(
        json.dumps(
            {
                "usable_village": usable_village,
                "behavior_lineage_accuracy": behavior_lineage_accuracy,
                "behavior_sibling_accuracy_given_correct_lineage": (
                    behavior_sibling_accuracy
                ),
                "mean_genericness": mean_genericness,
                "individual_quality_pass": len(individual_pass),
                "ecology_sessions": ecology_summary["sessions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
