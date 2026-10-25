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


def dataset_hash_inventory(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "final-hashes.sha256":
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


def top_five_confusions(
    counts: Counter,
    labels: list[str],
    true_label: str,
) -> list[list]:
    ranked = sorted(
        (
            (label, int(counts.get(label, 0)))
            for label in labels
            if label != true_label
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return [[label, count] for label, count in ranked[:5]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-artifacts", type=Path, required=True)
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
    final_eval_path = ROOT / cfg["final_eval_config"]
    final_eval = json.loads(final_eval_path.read_text())
    final_eval_sha = sha256_file(final_eval_path)
    final_scenarios = final_eval.get("scenarios")
    if (
        final_eval.get("experiment") != "VILLAGE-001E"
        or final_eval.get("stage") != "fresh-final-heldout"
        or not isinstance(final_scenarios, list)
        or len(final_scenarios) != int(cfg["tournament"]["scenario_count"])
    ):
        raise RuntimeError("fresh final heldout config is invalid")
    scenarios = [row["id"] for row in final_scenarios]
    if scenarios != [f"F{i:02d}" for i in range(1, 25)]:
        raise RuntimeError("fresh final heldout IDs must be F01..F24")
    final_scenario_by_id = {
        row["id"]: row
        for row in final_scenarios
    }

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
    release_assets = {
        row["member_id"]: row
        for row in preference_release["assets"]
    }
    if set(release_assets) != set(expected_members):
        raise RuntimeError("preference release member set mismatch")

    final_responses = find_unique(
        args.response_artifacts,
        "result.json",
        experiment="VILLAGE-001E",
        stage="fresh-final-heldout-responses",
        id_key="member_id",
    )
    expected_response_ids = set(expected_members) | {"BASE"}
    if set(final_responses) != expected_response_ids:
        raise RuntimeError(
            "fresh final response set mismatch "
            f"missing={sorted(expected_response_ids-set(final_responses))} "
            f"extra={sorted(set(final_responses)-expected_response_ids)}"
        )

    for response_id in sorted(expected_response_ids):
        doc = json.loads(final_responses[response_id].read_text())
        common_checks = {
            "member_contract_sha256": population["member_contract_sha256"],
            "base_model": campaign["base_model"]["id"],
            "base_revision": campaign["base_model"]["revision"],
            "preference_release_tag": preference_release["release_tag"],
            "final_eval_config_sha256": final_eval_sha,
            "fresh_after_preference_training": True,
            "exact_training_input_collision_count": 0,
            "hidden_member_prompt_used": False,
        }
        for key, expected in common_checks.items():
            if doc.get(key) != expected:
                raise RuntimeError(
                    f"{response_id}: final response {key} "
                    f"{doc.get(key)!r} != {expected!r}"
                )
        if response_id == "BASE":
            base_checks = {
                "base_control": True,
                "lineage": None,
                "sibling": None,
                "preference_adapter_sha256": None,
            }
            for key, expected in base_checks.items():
                if doc.get(key) != expected:
                    raise RuntimeError(
                        f"BASE: final response {key} mismatch"
                    )
        else:
            meta = member_meta[response_id]
            member_checks = {
                "base_control": False,
                "lineage": meta["lineage"],
                "sibling": meta["sibling"],
                "preference_adapter_sha256": release_assets[
                    response_id
                ]["adapter_directory_sha256"],
            }
            for key, expected in member_checks.items():
                if doc.get(key) != expected:
                    raise RuntimeError(
                        f"{response_id}: final response {key} mismatch"
                    )

        response_rows = doc.get("responses")
        if not isinstance(response_rows, list) or len(response_rows) != 24:
            raise RuntimeError(
                f"{response_id}: final response cardinality mismatch"
            )
        if [row.get("scenario_id") for row in response_rows] != scenarios:
            raise RuntimeError(
                f"{response_id}: final response scenario order mismatch"
            )
        for row in response_rows:
            expected = final_scenario_by_id[row["scenario_id"]]
            if (
                row.get("prompt") != expected["prompt"]
                or row.get("family") != expected["family"]
            ):
                raise RuntimeError(
                    f"{response_id}/{row['scenario_id']}: "
                    "final prompt/family drift"
                )

    base_final_result = json.loads(final_responses["BASE"].read_text())
    base_final_by_id = {
        row["scenario_id"]: row
        for row in base_final_result["responses"]
    }

    base_summary_path = ROOT / "experiments/persona-001/evidence/summary.json"
    if not base_summary_path.is_file():
        raise RuntimeError("PERSONA-001 untuned BASE evidence is missing")
    base_summary = json.loads(base_summary_path.read_text())
    measured_base_genericness = float(
        base_summary["raw_blind"]["control_scores"]["genericness"]
    )
    measured_base_behavior_genericness = float(
        base_summary["behavior_blind"]["control_scores"]["genericness"]
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
        id_key="scenario_id",
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
            "raw_task_relevance": [],
            "behavior_task_relevance": [],
            "raw_genericness": [],
            "behavior_genericness": [],
        }
        for member_id in expected_members
    }
    base_raw_task_relevance = []
    base_behavior_task_relevance = []
    base_raw_genericness = []
    base_behavior_genericness = []
    super_audit_count = 0

    for scenario_id in scenarios:
        doc = json.loads(tournament[scenario_id].read_text())
        expected_scenario = final_scenario_by_id[scenario_id]
        if (
            doc.get("scenario_id") != scenario_id
            or doc.get("fresh_after_preference_training") is not True
            or doc.get("final_eval_config_sha256") != final_eval_sha
            or doc.get("scenario", {}).get("prompt")
            != expected_scenario["prompt"]
            or doc.get("scenario", {}).get("family")
            != expected_scenario["family"]
        ):
            raise RuntimeError(
                f"{scenario_id}: tournament final-heldout identity mismatch"
            )
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
            stats["raw_task_relevance"].append(float(raw["task_relevance"]))
            stats["behavior_task_relevance"].append(
                float(behavior["task_relevance"])
            )
            stats["raw_genericness"].append(float(raw["genericness"]))
            stats["behavior_genericness"].append(float(behavior["genericness"]))

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
            family_stats[family]["raw_task_relevance_sum"] += float(
                raw["task_relevance"]
            )
            family_stats[family]["behavior_task_relevance_sum"] += float(
                behavior["task_relevance"]
            )
            family_stats[family]["raw_genericness_sum"] += float(
                raw["genericness"]
            )
            family_stats[family]["behavior_genericness_sum"] += float(
                behavior["genericness"]
            )

            if row.get("super_audit") is not None:
                super_audit_count += 1
            total_records += 1

        control = doc.get("untuned_base_control")
        if not isinstance(control, dict):
            raise RuntimeError(
                f"{scenario_id}: untuned BASE control is missing"
            )
        if control.get("response") != base_final_by_id[scenario_id]["response"]:
            raise RuntimeError(
                f"{scenario_id}: BASE control response drift"
            )
        if not isinstance(control.get("behavior_summary"), str) or len(
            control["behavior_summary"].strip()
        ) < 8:
            raise RuntimeError(
                f"{scenario_id}: BASE behavior summary missing"
            )
        for mode, task_bucket, generic_bucket in (
            (
                "raw",
                base_raw_task_relevance,
                base_raw_genericness,
            ),
            (
                "behavior",
                base_behavior_task_relevance,
                base_behavior_genericness,
            ),
        ):
            score = control.get(mode)
            if not isinstance(score, dict):
                raise RuntimeError(
                    f"{scenario_id}: BASE {mode} score missing"
                )
            for key in ("task_relevance", "genericness"):
                value = score.get(key)
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 0 <= value <= 4
                ):
                    raise RuntimeError(
                        f"{scenario_id}: BASE {mode} {key} invalid"
                    )
            task_bucket.append(float(score["task_relevance"]))
            generic_bucket.append(float(score["genericness"]))

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
            "raw_mean_task_relevance": (
                counts["raw_task_relevance_sum"] / counts["records"]
            ),
            "behavior_mean_task_relevance": (
                counts["behavior_task_relevance_sum"] / counts["records"]
            ),
            "raw_mean_genericness": (
                counts["raw_genericness_sum"] / counts["records"]
            ),
            "behavior_mean_genericness": (
                counts["behavior_genericness_sum"] / counts["records"]
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
    preference_task_relevance_values = []
    preference_genericness_values = []
    final_raw_task_relevance_values = []
    final_behavior_task_relevance_values = []
    final_raw_genericness_values = []
    final_behavior_genericness_values = []
    individual_pass = []
    individual_fail = []
    member_table = []

    for member_id in expected_members:
        meta = member_meta[member_id]
        comparison = before_after[member_id]
        after = comparison["after"]
        preference_task_relevance = float(after["task_relevance"])
        preference_genericness = float(after["genericness"])
        preference_task_relevance_values.append(preference_task_relevance)
        preference_genericness_values.append(preference_genericness)
        stats = member_stats[member_id]
        behavior_hits = int(stats["behavior_lineage_hits"])
        behavior_sibling_hits = int(
            stats["behavior_sibling_hits_given_correct_lineage"]
        )
        final_raw_task_relevance = statistics.mean(
            stats["raw_task_relevance"]
        )
        final_behavior_task_relevance = statistics.mean(
            stats["behavior_task_relevance"]
        )
        final_raw_genericness = statistics.mean(stats["raw_genericness"])
        final_behavior_genericness = statistics.mean(
            stats["behavior_genericness"]
        )
        final_raw_task_relevance_values.append(final_raw_task_relevance)
        final_behavior_task_relevance_values.append(
            final_behavior_task_relevance
        )
        final_raw_genericness_values.append(final_raw_genericness)
        final_behavior_genericness_values.append(
            final_behavior_genericness
        )
        passes = (
            member_id in maturation_qualified
            and behavior_hits
            >= int(cfg["tournament"]["individual_min_behavior_lineage_hits"])
            and behavior_sibling_hits
            >= int(cfg["tournament"]["individual_min_behavior_sibling_hits"])
            and final_raw_task_relevance
            >= float(cfg["tournament"]["individual_min_task_relevance"])
            and final_raw_genericness
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
                "task_relevance": final_raw_task_relevance,
                "genericness": final_raw_genericness,
                "raw_task_relevance": final_raw_task_relevance,
                "behavior_task_relevance": final_behavior_task_relevance,
                "raw_genericness": final_raw_genericness,
                "behavior_genericness": final_behavior_genericness,
                "preference_stage_task_relevance": preference_task_relevance,
                "preference_stage_genericness": preference_genericness,
                "maturation_gate_pass": member_id in maturation_qualified,
                "individual_quality_pass": passes,
                "top_raw_lineage_confusions": wrong_raw_lineage.most_common(5),
                "top_behavior_lineage_confusions": (
                    wrong_behavior_lineage.most_common(5)
                ),
                "nearest_raw_member_confusions": top_five_confusions(
                    wrong_raw_member, expected_members, member_id
                ),
                "nearest_behavior_member_confusions": top_five_confusions(
                    wrong_behavior_member, expected_members, member_id
                ),
            }
        )

    mean_genericness = statistics.mean(final_raw_genericness_values)
    mean_behavior_genericness = statistics.mean(
        final_behavior_genericness_values
    )
    mean_task_relevance = statistics.mean(
        final_raw_task_relevance_values
    )
    mean_behavior_task_relevance = statistics.mean(
        final_behavior_task_relevance_values
    )
    fresh_base_raw_task_relevance = statistics.mean(
        base_raw_task_relevance
    )
    fresh_base_behavior_task_relevance = statistics.mean(
        base_behavior_task_relevance
    )
    fresh_base_raw_genericness = statistics.mean(
        base_raw_genericness
    )
    fresh_base_behavior_genericness = statistics.mean(
        base_behavior_genericness
    )
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
            "threshold": fresh_base_raw_genericness,
            "comparison": "same fresh F-suite untuned BASE control",
            "historical_persona001_reference": float(
                cfg["tournament"]["base_genericness_reference"]
            ),
            "pass": mean_genericness < fresh_base_raw_genericness,
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
    ecology_agent_population: dict[str, list[float]] = defaultdict(list)
    ecology_lineage_metric_values: dict[str, list[float]] = defaultdict(list)
    ecology_lineage_agent: dict[str, dict[str, list[float]]] = defaultdict(
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
            ecology_agent[member_id]["dominance"].append(
                float(agent["dominance"])
            )
            ecology_agent[member_id]["deference"].append(
                float(agent["deference"])
            )
            for key in (
                "distinctiveness",
                "generic_assistant_tendency",
                "dominance",
                "deference",
            ):
                ecology_agent_population[key].append(float(agent[key]))

        lineage_audit = judgment.get("lineage_audit")
        if session["kind"] == "cross_lineage_group":
            if not isinstance(lineage_audit, dict):
                raise RuntimeError(
                    f"{session_id}: missing cross-lineage audit"
                )
            audit_agents = lineage_audit.get("agents")
            if not isinstance(audit_agents, list) or len(audit_agents) != len(
                session["members"]
            ):
                raise RuntimeError(
                    f"{session_id}: lineage audit agent cardinality mismatch"
                )
            audit_members = set()
            for agent in audit_agents:
                member_id = agent.get("member_id")
                if member_id not in member_meta or member_id in audit_members:
                    raise RuntimeError(
                        f"{session_id}: invalid lineage-audit member"
                    )
                audit_members.add(member_id)
                for key in (
                    "expected_lineage_fit",
                    "cross_lineage_contamination",
                    "distinction_survival",
                ):
                    value = float(agent[key])
                    ecology_lineage_metric_values[key].append(value)
                    ecology_lineage_agent[member_id][key].append(value)
            if audit_members != set(session["members"]):
                raise RuntimeError(
                    f"{session_id}: lineage audit member set mismatch"
                )
        elif lineage_audit is not None:
            raise RuntimeError(
                f"{session_id}: same-lineage pair unexpectedly has lineage audit"
            )

        ecology_rows.append(
            {
                "session_id": session_id,
                "kind": session["kind"],
                "scenario_id": session["scenario"]["id"],
                "members": session["members"],
                "metrics": judgment["metrics"],
                "lineage_audit": judgment.get("lineage_audit"),
                "super_audit": judgment.get("super_audit"),
            }
        )

    ecology_summary = {
        "sessions": len(ecology_rows),
        "pair_sessions": sum(
            1 for row in ecology_rows if row["kind"] == "same_lineage_pair"
        ),
        "small_group_sessions": sum(
            1 for row in ecology_rows if row["kind"] == "cross_lineage_group"
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
        "lineage_audit_metrics": {
            key: metric(values)
            for key, values in ecology_lineage_metric_values.items()
        },
        "cross_lineage_contamination": metric(
            ecology_lineage_metric_values["cross_lineage_contamination"]
        ),
        "member_lineage_audit_metrics": {
            member_id: {
                key: metric(values)
                for key, values in metrics.items()
            }
            for member_id, metrics in ecology_lineage_agent.items()
        },
        "agent_metrics": {
            key: metric(values)
            for key, values in ecology_agent_population.items()
        },
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

    final_responses_dest = out / "final-responses"
    tournament_dest = out / "tournament-scenarios"
    ecology_dest = out / "ecology"
    for response_id, path in final_responses.items():
        shutil.copytree(
            path.parent,
            final_responses_dest / response_id,
        )
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

    release_assets = {
        row["member_id"]: row
        for row in preference_release["assets"]
    }
    if set(release_assets) != set(expected_members):
        raise RuntimeError("final adapter release member set mismatch")
    verdict_by_member = {
        row["member_id"]: row["individual_quality_pass"]
        for row in member_table
    }
    model_registry = {
        "schema_version": 1,
        "experiment": "VILLAGE-001",
        "qualification_stage": "VILLAGE-001E",
        "issue": 14,
        "release_tag": preference_release["release_tag"],
        "release_url": preference_release["release_url"],
        "member_contract_sha256": population["member_contract_sha256"],
        "models": [
            {
                "member_id": member_id,
                "lineage": member_meta[member_id]["lineage"],
                "sibling": member_meta[member_id]["sibling"],
                "individual_quality_pass": verdict_by_member[member_id],
                "asset_file": release_assets[member_id]["file"],
                "asset_sha256": release_assets[member_id]["sha256"],
                "adapter_directory_sha256": release_assets[member_id][
                    "adapter_directory_sha256"
                ],
            }
            for member_id in expected_members
        ],
    }
    write_json(out / "model-registry.json", model_registry)

    dataset_root = ROOT / campaign["archive"]["dataset_root"]
    dataset_lineage = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "dataset_root": str(dataset_root.relative_to(ROOT)),
        "member_contract_sha256": population["member_contract_sha256"],
        "files": {},
    }
    for rel in (
        "manifest.json",
        "preference-manifest.json",
        "provenance/hashes.sha256",
        "preference-hashes.sha256",
    ):
        source = dataset_root / rel
        if not source.is_file():
            raise RuntimeError(f"required durable dataset lineage file missing: {source}")
        dataset_lineage["files"][rel] = {
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        }
    write_json(out / "dataset-provenance.json", dataset_lineage)

    summary = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "issue": 14,
        "members": 36,
        "scenarios": 24,
        "blind_response_records": total_records,
        "fresh_final_heldout": {
            "config": cfg["final_eval_config"],
            "config_sha256": final_eval_sha,
            "scenario_ids": scenarios,
            "fresh_after_preference_training": True,
            "trained_members": 36,
            "untuned_base_control": True,
            "exact_training_input_collision_count": 0,
        },
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
            "mean_task_relevance": mean_task_relevance,
            "mean_genericness": mean_genericness,
            "raw_mean_task_relevance": mean_task_relevance,
            "behavior_mean_task_relevance": mean_behavior_task_relevance,
            "raw_mean_genericness": mean_genericness,
            "behavior_mean_genericness": mean_behavior_genericness,
            "fresh_base_raw_task_relevance": (
                fresh_base_raw_task_relevance
            ),
            "fresh_base_behavior_task_relevance": (
                fresh_base_behavior_task_relevance
            ),
            "fresh_base_raw_genericness": fresh_base_raw_genericness,
            "fresh_base_behavior_genericness": (
                fresh_base_behavior_genericness
            ),
            "historical_persona001_base_genericness_reference": float(
                cfg["tournament"]["base_genericness_reference"]
            ),
            "historical_persona001_base_genericness_measured": (
                measured_base_genericness
            ),
            "historical_persona001_base_behavior_genericness_measured": (
                measured_base_behavior_genericness
            ),
            "historical_base_genericness_source": cfg["tournament"][
                "base_genericness_source"
            ],
            "preference_stage_mean_task_relevance": statistics.mean(
                preference_task_relevance_values
            ),
            "preference_stage_mean_genericness": statistics.mean(
                preference_genericness_values
            ),
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
            "lineage_audit_metrics": ecology_summary[
                "lineage_audit_metrics"
            ],
            "cross_lineage_contamination": ecology_summary[
                "cross_lineage_contamination"
            ],
            "super_convergence_concerns": ecology_concerns,
        },
        "final_adapter_release": cfg["final_release_tag"],
        "qualification_evidence_release": cfg["qualification_release_tag"],
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "per-member-verdicts.json",
        {
            "schema_version": 1,
            "experiment": "VILLAGE-001E",
            "issue": 14,
            "members": member_table,
        },
    )

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
        "final_eval_config_sha256": final_eval_sha,
        "campaign_config_sha256": sha256_file(campaign_path),
        "preference_config_sha256": sha256_file(preference_cfg_path),
        "preference_summary_sha256": sha256_file(preference_summary_path),
        "preference_release_manifest_sha256": sha256_file(preference_release_path),
        "untuned_base_summary_sha256": sha256_file(base_summary_path),
        "dataset_provenance_sha256": sha256_file(out / "dataset-provenance.json"),
        "model_registry_sha256": sha256_file(out / "model-registry.json"),
        "member_contract_sha256": population["member_contract_sha256"],
        "qualification_contract_predeclared": {
            "behavior_lineage_accuracy_min": float(
                cfg["tournament"]["lineage_gate"]
            ),
            "behavior_sibling_accuracy_given_correct_lineage_min": float(
                cfg["tournament"]["sibling_gate_given_correct_lineage"]
            ),
            "genericness_rule": (
                "population raw mean < same fresh F-suite untuned BASE raw mean"
            ),
            "individual_quality_members_min": int(
                cfg["tournament"]["individual_quality_required"]
            ),
        },
        "qualification_gate_results": population_gates,
        "ecology_used_as_hard_population_gate": False,
        "qualification_evidence_release": cfg["qualification_release_tag"],
    }
    write_json(out / "provenance.json", provenance)

    report = [
        "# VILLAGE-001E — final village qualification",
        "",
        "The final preference-matured cartridges were evaluated on a fresh F01-F24 "
        "suite generated only after preference training, without prompt-time member/persona "
        "descriptions. Tournament providers received opaque response IDs; ecology "
        "models/judges received opaque participant handles.",
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
        f"- mean final raw-tournament genericness: {mean_genericness:.3f} "
        f"(same F-suite untuned BASE {fresh_base_raw_genericness:.3f})",
        f"- mean style-neutral genericness: {mean_behavior_genericness:.3f} "
        f"(same F-suite untuned BASE {fresh_base_behavior_genericness:.3f})",
        f"- mean final raw task relevance: {mean_task_relevance:.3f}/4 "
        f"(BASE {fresh_base_raw_task_relevance:.3f})",
        f"- mean final style-neutral task relevance: "
        f"{mean_behavior_task_relevance:.3f}/4 "
        f"(BASE {fresh_base_behavior_task_relevance:.3f})",
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
        "(12 same-lineage pairs, 12 cross-lineage small groups)",
    ]
    for key in (
        "differentiation",
        "generic_convergence",
        "information_sharing",
        "information_hoarding",
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
            f"- cross-lineage contamination: mean "
            f"{ecology_summary['cross_lineage_contamination']['mean']:.3f}/4",
            f"- cross-lineage expected-lineage fit: mean "
            f"{ecology_summary['lineage_audit_metrics']['expected_lineage_fit']['mean']:.3f}/4",
            f"- cross-lineage distinction survival: mean "
            f"{ecology_summary['lineage_audit_metrics']['distinction_survival']['mean']:.3f}/4",
            f"- participant dominance: mean "
            f"{ecology_summary['agent_metrics']['dominance']['mean']:.3f}/4",
            f"- participant deference: mean "
            f"{ecology_summary['agent_metrics']['deference']['mean']:.3f}/4",
            f"- Super convergence concerns: {len(ecology_concerns)}/24 sessions",
            "",
            "Ecology is diagnostic evidence, not a retroactively invented hard gate. "
            "The population gate is exactly the predeclared lineage/sibling/genericness/"
            "individual-quality gate above.",
            "",
            "## Transfer families",
            "",
            f"- abstract_unfamiliar behavior lineage accuracy: "
            f"{family_accuracy['abstract_unfamiliar']['behavior_lineage_accuracy']:.3f}",
            f"- abstract_unfamiliar behavior sibling accuracy: "
            f"{family_accuracy['abstract_unfamiliar']['behavior_sibling_accuracy_given_correct_lineage']:.3f}",
            f"- failure_recovery behavior lineage accuracy: "
            f"{family_accuracy['failure_recovery']['behavior_lineage_accuracy']:.3f}",
            f"- failure_recovery behavior sibling accuracy: "
            f"{family_accuracy['failure_recovery']['behavior_sibling_accuracy_given_correct_lineage']:.3f}",
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
    population_report = "\n".join(report) + "\n"
    (out / "REPORT.md").write_text(population_report)
    (out / "POPULATION_REPORT.md").write_text(population_report)

    ecology_report = [
        "# VILLAGE-001E — ecology report",
        "",
        "Fixed prompt-neutral interactions used 12 same-lineage pairs and 12 "
        "cross-lineage three-agent groups. Models and provider judges saw only "
        "opaque handles, never member IDs.",
        "",
        "## Population ecology metrics",
        "",
    ]
    for key, values in sorted(ecology_summary["metrics"].items()):
        ecology_report.append(
            f"- {key}: mean {values['mean']:.3f}/4 "
            f"(min {values['min']:.1f}, max {values['max']:.1f})"
        )
    ecology_report.extend(
        [
            f"- cross-lineage contamination: mean "
            f"{ecology_summary['cross_lineage_contamination']['mean']:.3f}/4",
            f"- participant dominance: mean "
            f"{ecology_summary['agent_metrics']['dominance']['mean']:.3f}/4",
            f"- participant deference: mean "
            f"{ecology_summary['agent_metrics']['deference']['mean']:.3f}/4",
            "",
            "## Per-member participant behavior",
            "",
            "| Member | appearances | distinctiveness | generic tendency | dominance | deference | lineage fit | contamination | lineage distinction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for member_id in expected_members:
        values = ecology_summary["member_agent_metrics"].get(member_id, {})
        if not values:
            raise RuntimeError(f"{member_id}: missing ecology participant metrics")
        appearances = len(ecology_agent[member_id]["distinctiveness"])
        lineage_values = ecology_summary[
            "member_lineage_audit_metrics"
        ].get(member_id)
        if not lineage_values:
            raise RuntimeError(
                f"{member_id}: missing cross-lineage audit metrics"
            )
        ecology_report.append(
            f"| {member_id} | {appearances} | "
            f"{values['distinctiveness']['mean']:.2f} | "
            f"{values['generic_assistant_tendency']['mean']:.2f} | "
            f"{values['dominance']['mean']:.2f} | "
            f"{values['deference']['mean']:.2f} | "
            f"{lineage_values['expected_lineage_fit']['mean']:.2f} | "
            f"{lineage_values['cross_lineage_contamination']['mean']:.2f} | "
            f"{lineage_values['distinction_survival']['mean']:.2f} |"
        )
    ecology_report.append("")
    (out / "ECOLOGY_REPORT.md").write_text("\n".join(ecology_report) + "\n")
    (out / "hashes.sha256").write_text(hash_inventory(out))

    # Promote the complete final qualification tree into the durable VILLAGE-001
    # dataset archive. Actions artifacts remain staging; this committed tree is the
    # authoritative raw tournament/ecology evidence for the completed campaign.
    dataset_root = ROOT / campaign["archive"]["dataset_root"]
    evaluation_root = dataset_root / "evaluation"
    if evaluation_root.exists():
        shutil.rmtree(evaluation_root)
    shutil.copytree(out, evaluation_root)

    final_manifest = {
        "schema_version": 1,
        "experiment": "VILLAGE-001",
        "stage": "VILLAGE-001E-final-qualification",
        "issue": 14,
        "member_contract_sha256": population["member_contract_sha256"],
        "evaluation_root": "evaluation",
        "final_evidence_root": (
            out.relative_to(ROOT).as_posix()
            if out.is_absolute()
            else out.as_posix()
        ),
        "preference_release_tag": preference_release["release_tag"],
        "qualification_evidence_release": cfg["qualification_release_tag"],
        "members": 36,
        "scenarios": 24,
        "ecology_sessions": ecology_summary["sessions"],
        "usable_village": usable_village,
        "individual_quality_pass_count": len(individual_pass),
        "summary_sha256": sha256_file(out / "summary.json"),
        "confusion_matrices_sha256": sha256_file(
            out / "confusion-matrices.json"
        ),
        "ecology_summary_sha256": sha256_file(out / "ecology-summary.json"),
        "model_registry_sha256": sha256_file(out / "model-registry.json"),
    }
    write_json(dataset_root / "final-manifest.json", final_manifest)
    (dataset_root / "final-hashes.sha256").write_text(
        dataset_hash_inventory(dataset_root)
    )

    # Verify the promoted copy and the final dataset inventory before allowing the
    # workflow to commit either authority surface.
    if sha256_file(evaluation_root / "summary.json") != final_manifest[
        "summary_sha256"
    ]:
        raise RuntimeError("promoted final summary hash mismatch")
    if sha256_file(evaluation_root / "model-registry.json") != final_manifest[
        "model_registry_sha256"
    ]:
        raise RuntimeError("promoted model registry hash mismatch")

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
