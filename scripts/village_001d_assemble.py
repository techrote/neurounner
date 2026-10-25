#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from village_001_manifest import build_manifest
from village_001_validate import validate_feedback_row
from village_001d_validate import validate_file as validate_pair_file

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(item)))
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def deterministic_tar_gz(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(
                fileobj=gz,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as tf:
                for path in sorted(source.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(source).as_posix()
                    info = tf.gettarinfo(str(path), arcname=rel)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as fh:
                        tf.addfile(info, fh)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def find_by_marker(
    root: Path,
    marker: str,
    experiment: str,
    stage: str | None = None,
) -> dict[str, Path]:
    found: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob(marker):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if obj.get("experiment") != experiment:
            continue
        if stage is not None and obj.get("stage") != stage:
            continue
        member = obj.get("member_id")
        if isinstance(member, str):
            found[member].append(path.parent)
    duplicates = {
        member: paths for member, paths in found.items() if len(paths) != 1
    }
    if duplicates:
        raise RuntimeError(
            f"{marker} artifact multiplicity error: {duplicates}"
        )
    return {
        member: paths[0]
        for member, paths in found.items()
    }


def hash_inventory(root: Path, exclude: set[str]) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in exclude:
            continue
        rows.append(f"{sha256_file(path)}  {rel}")
    return "\n".join(rows) + "\n"


def terminal_pair_curation(pair_root: Path, pair_ids: set[str]) -> dict[str, str]:
    critic_path = pair_root / "quality" / "critic.json"
    adjudicator_path = pair_root / "quality" / "adjudicator.json"
    if not critic_path.is_file() or not adjudicator_path.is_file():
        raise RuntimeError(f"{pair_root}: missing pair curation evidence")
    critic = json.loads(critic_path.read_text())
    adjudicator = json.loads(adjudicator_path.read_text())

    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in critic:
        if not isinstance(row, dict):
            raise RuntimeError("malformed pair critic record")
        pair_id = row.get("pair_id")
        round_index = row.get("round")
        if pair_id not in pair_ids or not isinstance(round_index, int):
            raise RuntimeError("invalid pair critic identity/round")
        by_pair[pair_id].append(row)

    adjudicator_by_key = {}
    for row in adjudicator:
        if not isinstance(row, dict):
            raise RuntimeError("malformed pair adjudicator record")
        key = (row.get("pair_id"), row.get("round"))
        if (
            key[0] not in pair_ids
            or not isinstance(key[1], int)
            or key in adjudicator_by_key
        ):
            raise RuntimeError("invalid/duplicate pair adjudicator record")
        adjudicator_by_key[key] = row

    terminal = {}
    for pair_id in sorted(pair_ids):
        history = by_pair.get(pair_id, [])
        if not history:
            raise RuntimeError(f"{pair_id}: no pair critic history")
        latest_round = max(row["round"] for row in history)
        latest = [
            row for row in history
            if row["round"] == latest_round
        ]
        if len(latest) != 1:
            raise RuntimeError(
                f"{pair_id}: ambiguous final critic round {latest_round}"
            )
        review = latest[0]
        verdict = review.get("derived_verdict")
        if verdict == "pass":
            terminal[pair_id] = "critic_pass"
        elif verdict == "borderline":
            decision = adjudicator_by_key.get((pair_id, latest_round))
            if not decision or decision.get("decision") != "accept":
                raise RuntimeError(
                    f"{pair_id}: unresolved terminal borderline verdict"
                )
            terminal[pair_id] = "adjudicator_accept"
        else:
            raise RuntimeError(
                f"{pair_id}: terminal critic verdict {verdict!r}"
            )
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-artifacts", type=Path, required=True)
    parser.add_argument("--pair-artifacts", type=Path, required=True)
    parser.add_argument("--preference-artifacts", type=Path, required=True)
    parser.add_argument("--comparison-artifacts", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/preference-config.json",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=ROOT / "experiments/village-001/preference-evidence",
    )
    parser.add_argument("--release-out", type=Path, required=True)
    args = parser.parse_args()

    pref = json.loads(args.config.read_text())
    feedback_per_member = (
        int(pref["feedback"]["scenario_count"])
        + int(pref["feedback"]["member_probe_count"])
    ) * int(pref["feedback"]["workers_per_scenario"])
    feedback_scenario_ids = [
        *[
            f"T{i:02d}"
            for i in range(1, int(pref["feedback"]["scenario_count"]) + 1)
        ],
        *[
            f"P{i:02d}"
            for i in range(1, int(pref["feedback"]["member_probe_count"]) + 1)
        ],
    ]
    campaign_path = ROOT / pref["campaign_config"]
    campaign = json.loads(campaign_path.read_text())
    population = build_manifest(campaign)
    member_meta = {
        row["member_id"]: row
        for row in population["members"]
    }
    expected_members = list(member_meta)

    feedback_found = find_by_marker(
        args.feedback_artifacts,
        "manifest.json",
        "VILLAGE-001D",
        "sprocket-feedback",
    )
    pair_found = find_by_marker(
        args.pair_artifacts,
        "manifest.json",
        "VILLAGE-001D",
        "preference-pairs",
    )
    preference_found = find_by_marker(
        args.preference_artifacts,
        "result.json",
        "VILLAGE-001D",
        "preference-continuation",
    )
    comparison_found = find_by_marker(
        args.comparison_artifacts,
        "evaluation.json",
        "VILLAGE-001D",
        "before-after-evaluation",
    )

    for label, found in (
        ("feedback", feedback_found),
        ("pairs", pair_found),
        ("preference", preference_found),
        ("comparison", comparison_found),
    ):
        missing = sorted(set(expected_members) - set(found))
        extra = sorted(set(found) - set(expected_members))
        if missing or extra:
            raise RuntimeError(
                f"{label} artifacts mismatch missing={missing} extra={extra}"
            )

    evidence = args.evidence_out
    release = args.release_out
    if evidence.exists():
        shutil.rmtree(evidence)
    if release.exists():
        shutil.rmtree(release)
    evidence.mkdir(parents=True)
    release.mkdir(parents=True)

    dataset_root = ROOT / campaign["archive"]["dataset_root"]
    dataset_manifest = dataset_root / "manifest.json"
    if not dataset_manifest.is_file():
        raise RuntimeError("VILLAGE-001B durable dataset is missing")
    stage = json.loads(dataset_manifest.read_text())
    if stage.get("stage") != "VILLAGE-001B-member-corpora":
        raise RuntimeError("unexpected VILLAGE-001 durable dataset stage")
    if stage.get("member_contract_sha256") != population[
        "member_contract_sha256"
    ]:
        raise RuntimeError("durable dataset/member contract mismatch")

    parent_release = json.loads(
        (
            ROOT
            / pref["sft_evidence_root"]
            / "adapter-release.json"
        ).read_text()
    )
    parent_assets = {
        row["member_id"]: row
        for row in parent_release["assets"]
    }
    if set(parent_assets) != set(expected_members):
        raise RuntimeError("SFT parent release manifest is incomplete")

    member_rows = []
    release_assets = []
    qualified = []
    regressed = []
    invalid_feedback_total = 0
    valid_feedback_total = 0
    terminal_counts = Counter()
    pref_margin_before = []
    pref_margin_after = []
    pref_accuracy_before = []
    pref_accuracy_after = []
    relevance_delta = []
    behavior_delta = []
    genericness_delta = []
    sibling_delta = []
    training_seconds = []
    peak_rss = []

    for member_id in expected_members:
        meta = member_meta[member_id]
        feedback_root = feedback_found[member_id]
        pair_root = pair_found[member_id]
        preference_root = preference_found[member_id]
        comparison_root = comparison_found[member_id]

        feedback_manifest = json.loads(
            (feedback_root / "manifest.json").read_text()
        )
        feedback_rows = read_jsonl(feedback_root / "feedback.jsonl")
        if len(feedback_rows) != feedback_per_member:
            raise RuntimeError(
                f"{member_id}: feedback count {len(feedback_rows)} "
                f"!= {feedback_per_member}"
            )
        feedback_ids = set()
        per_scenario = Counter()
        for row in feedback_rows:
            validate_feedback_row(row)
            if row["member_id"] != member_id:
                raise RuntimeError(f"{member_id}: feedback member mismatch")
            if row["feedback_id"] in feedback_ids:
                raise RuntimeError(f"{member_id}: duplicate feedback ID")
            feedback_ids.add(row["feedback_id"])
            per_scenario[row["scenario_id"]] += 1
        if set(per_scenario) != set(feedback_scenario_ids):
            raise RuntimeError(f"{member_id}: feedback scenario ID set mismatch")
        if any(
            per_scenario[scenario_id]
            != int(pref["feedback"]["workers_per_scenario"])
            for scenario_id in feedback_scenario_ids
        ):
            raise RuntimeError(f"{member_id}: feedback scenario coverage mismatch")
        valid_feedback = sum(1 for row in feedback_rows if row["valid"])
        invalid_feedback = len(feedback_rows) - valid_feedback
        valid_feedback_total += valid_feedback
        invalid_feedback_total += invalid_feedback

        pair_path = pair_root / "pairs.jsonl"
        validate_pair_file(pair_path, member_id, 60)
        pair_rows = read_jsonl(pair_path)
        feedback_dataset_sha = sha256_file(
            feedback_root / "feedback.jsonl"
        )
        if any(
            row["provenance"]["feedback_dataset_sha256"]
            != feedback_dataset_sha
            for row in pair_rows
        ):
            raise RuntimeError(
                f"{member_id}: preference pair feedback dataset hash mismatch"
            )
        pair_manifest = json.loads((pair_root / "manifest.json").read_text())
        if pair_manifest.get("feedback_dataset_sha256") != feedback_dataset_sha:
            raise RuntimeError(
                f"{member_id}: pair manifest feedback dataset hash mismatch"
            )
        pair_ids = {row["pair_id"] for row in pair_rows}
        if len(pair_ids) != 60:
            raise RuntimeError(f"{member_id}: pair IDs not unique")
        if any(
            set(row["source_feedback_ids"]) != set(
                item["feedback_id"]
                for item in feedback_rows
                if item["scenario_id"] == row["scenario_id"]
            )
            for row in pair_rows
        ):
            raise RuntimeError(
                f"{member_id}: preference pair feedback lineage mismatch"
            )
        terminal = terminal_pair_curation(pair_root, pair_ids)
        terminal_counts.update(terminal.values())

        result_path = preference_root / "result.json"
        result = json.loads(result_path.read_text())
        checks = {
            "member_id": member_id,
            "lineage": meta["lineage"],
            "sibling": meta["sibling"],
            "specialization": meta["specialization"],
            "member_seed": meta["seed"],
            "member_contract_sha256": population["member_contract_sha256"],
            "base_model": campaign["base_model"]["id"],
            "base_revision": campaign["base_model"]["revision"],
            "parent_release": pref["parent_release_tag"],
            "parent_adapter_sha256": feedback_manifest[
                "parent_adapter_sha256"
            ],
            "preference_dataset_sha256": sha256_file(pair_path),
        }
        for key, expected in checks.items():
            if result.get(key) != expected:
                raise RuntimeError(
                    f"{member_id}: {key} {result.get(key)!r} != {expected!r}"
                )
        if result.get("training_config") != pref["preference_training"]:
            raise RuntimeError(f"{member_id}: preference training config drift")
        if result["training"]["optimizer_steps"] != 120:
            raise RuntimeError(f"{member_id}: optimizer steps != 120")
        if len(result["training"]["losses"]) != 120:
            raise RuntimeError(f"{member_id}: preference loss trace incomplete")
        if len(result["preference_pair_ids"]) != 60:
            raise RuntimeError(f"{member_id}: preference pair IDs incomplete")
        responses = result["evaluation"]["responses"]
        if len(responses) != 24:
            raise RuntimeError(f"{member_id}: held-out responses != 24")
        if result["evaluation"]["hidden_member_prompt_used"] is not False:
            raise RuntimeError(f"{member_id}: hidden member prompt flag not false")

        adapter_dir = preference_root / "adapter"
        if not adapter_dir.is_dir():
            raise RuntimeError(f"{member_id}: preference adapter missing")
        actual_adapter_sha = directory_hash(adapter_dir)
        if result["adapter"]["sha256"] != actual_adapter_sha:
            raise RuntimeError(f"{member_id}: adapter directory hash mismatch")

        comparison = json.loads(
            (comparison_root / "evaluation.json").read_text()
        )
        if comparison.get("member_id") != member_id:
            raise RuntimeError(f"{member_id}: comparison member mismatch")
        if comparison.get("parent_adapter_sha256") != result[
            "parent_adapter_sha256"
        ]:
            raise RuntimeError(f"{member_id}: comparison parent hash mismatch")
        if comparison.get("preference_adapter_sha256") != actual_adapter_sha:
            raise RuntimeError(
                f"{member_id}: comparison preference hash mismatch"
            )
        gates = comparison["gates"]
        member_qualified = (
            gates["task_relevance_delta_at_least_minus_0_5"]
            and gates["no_raw_only_behavior_regression"]
        )
        if member_qualified:
            qualified.append(member_id)
        else:
            regressed.append(member_id)

        feedback_dest = (
            dataset_root / "sprocket-feedback" / f"{member_id}.jsonl"
        )
        feedback_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(feedback_root / "feedback.jsonl", feedback_dest)

        feedback_evidence_dest = (
            dataset_root
            / "sprocket-feedback-evidence"
            / member_id
        )
        if feedback_evidence_dest.exists():
            shutil.rmtree(feedback_evidence_dest)
        feedback_evidence_dest.mkdir(parents=True)
        for name in ("manifest.json", "sprocket-server.log"):
            source = feedback_root / name
            if source.is_file():
                shutil.copy2(source, feedback_evidence_dest / name)

        preference_dest = (
            dataset_root / "preference" / f"{member_id}.jsonl"
        )
        preference_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pair_path, preference_dest)

        provider_dest = (
            dataset_root / "provider" / "preference" / member_id
        )
        if provider_dest.exists():
            shutil.rmtree(provider_dest)
        shutil.copytree(pair_root / "provider", provider_dest)

        quality_dest = (
            dataset_root / "preference-quality" / member_id
        )
        if quality_dest.exists():
            shutil.rmtree(quality_dest)
        shutil.copytree(pair_root / "quality", quality_dest)
        shutil.copy2(
            pair_root / "manifest.json",
            quality_dest / "manifest.json",
        )

        result_dest = evidence / "results" / f"{member_id}.json"
        result_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, result_dest)
        compare_dest = evidence / "before-after" / f"{member_id}.json"
        compare_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            comparison_root / "evaluation.json",
            compare_dest,
        )

        archive_name = (
            f"village-001-preference-{member_id}-"
            "smollm2-135m-lora.tar.gz"
        )
        archive = release / archive_name
        deterministic_tar_gz(adapter_dir, archive)
        asset = {
            "member_id": member_id,
            "qualified_for_population_tournament": member_qualified,
            "file": archive_name,
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
            "adapter_directory_sha256": actual_adapter_sha,
            "adapter_directory_bytes": result["adapter"]["bytes"],
            "parent_adapter_directory_sha256": result[
                "parent_adapter_sha256"
            ],
        }
        release_assets.append(asset)

        before = result["preference_before"]
        after = result["preference_after"]
        compare_delta = comparison["delta"]
        pref_margin_before.append(before["mean_margin_nats_per_token"])
        pref_margin_after.append(after["mean_margin_nats_per_token"])
        pref_accuracy_before.append(before["preference_accuracy"])
        pref_accuracy_after.append(after["preference_accuracy"])
        relevance_delta.append(compare_delta["task_relevance"])
        behavior_delta.append(compare_delta["behavior_fit"])
        genericness_delta.append(compare_delta["genericness"])
        sibling_delta.append(compare_delta["sibling_accuracy"])
        training_seconds.append(result["training"]["seconds"])
        peak_rss.append(result["peak_rss_kib"])

        member_rows.append(
            {
                "member_id": member_id,
                "lineage": meta["lineage"],
                "sibling": meta["sibling"],
                "valid_feedback": valid_feedback,
                "invalid_feedback": invalid_feedback,
                "feedback_total": feedback_per_member,
                "preference_margin_before": before[
                    "mean_margin_nats_per_token"
                ],
                "preference_margin_after": after[
                    "mean_margin_nats_per_token"
                ],
                "preference_accuracy_before": before[
                    "preference_accuracy"
                ],
                "preference_accuracy_after": after[
                    "preference_accuracy"
                ],
                "task_relevance_delta": compare_delta["task_relevance"],
                "genericness_delta": compare_delta["genericness"],
                "lineage_fit_delta": compare_delta["lineage_fit"],
                "behavior_fit_delta": compare_delta["behavior_fit"],
                "style_dependence_delta": compare_delta["style_dependence"],
                "raw_vs_behavior_gap_delta": compare_delta[
                    "raw_vs_behavior_gap"
                ],
                "sibling_accuracy_delta": compare_delta[
                    "sibling_accuracy"
                ],
                "qualified_for_population_tournament": member_qualified,
                "adapter_directory_sha256": actual_adapter_sha,
                "adapter_archive_sha256": asset["sha256"],
                "resumed": result["recovery"]["resumed"],
            }
        )

    if len(member_rows) != 36 or len(release_assets) != 36:
        raise RuntimeError("VILLAGE-001D population cardinality mismatch")

    preference_manifest = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "issue": 13,
        "member_contract_sha256": population["member_contract_sha256"],
        "feedback_records": 36 * feedback_per_member,
        "valid_feedback_records": valid_feedback_total,
        "invalid_feedback_records": invalid_feedback_total,
        "preference_pairs": 36 * 60,
        "terminal_pair_curation": dict(terminal_counts),
        "member_feedback": [
            f"sprocket-feedback/{member_id}.jsonl"
            for member_id in expected_members
        ],
        "member_preference_pairs": [
            f"preference/{member_id}.jsonl"
            for member_id in expected_members
        ],
        "provider_root": "provider/preference",
        "quality_root": "preference-quality",
        "evidence_root": "experiments/village-001/preference-evidence",
        "release_tag": pref["final_release_tag"],
    }
    write_json(
        dataset_root / "preference-manifest.json",
        preference_manifest,
    )

    release_manifest = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "issue": 13,
        "release_tag": pref["final_release_tag"],
        "release_url": (
            "https://github.com/techrote/neurounner/releases/tag/"
            + pref["final_release_tag"]
        ),
        "parent_release": pref["parent_release_tag"],
        "base_model": campaign["base_model"]["id"],
        "base_revision": campaign["base_model"]["revision"],
        "member_contract_sha256": population["member_contract_sha256"],
        "qualified_count": len(qualified),
        "regressed_count": len(regressed),
        "assets": release_assets,
    }
    write_json(evidence / "adapter-release.json", release_manifest)

    def metric(values: list[float]) -> dict:
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }

    summary = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "issue": 13,
        "members": 36,
        "feedback_records": 36 * feedback_per_member,
        "valid_feedback_records": valid_feedback_total,
        "invalid_feedback_records": invalid_feedback_total,
        "preference_pairs": 36 * 60,
        "preference_lanes": 36,
        "qualified_for_population_tournament": qualified,
        "regressed_or_gate_failed": regressed,
        "qualified_count": len(qualified),
        "regressed_count": len(regressed),
        "terminal_pair_curation": dict(terminal_counts),
        "metrics": {
            "preference_margin_before": metric(pref_margin_before),
            "preference_margin_after": metric(pref_margin_after),
            "preference_accuracy_before": metric(pref_accuracy_before),
            "preference_accuracy_after": metric(pref_accuracy_after),
            "task_relevance_delta": metric(relevance_delta),
            "behavior_fit_delta": metric(behavior_delta),
            "genericness_delta": metric(genericness_delta),
            "sibling_accuracy_delta": metric(sibling_delta),
            "training_seconds": metric(training_seconds),
            "peak_rss_kib": metric([float(x) for x in peak_rss]),
        },
        "members_table": member_rows,
    }
    write_json(evidence / "summary.json", summary)

    run_id = os.environ.get("GITHUB_RUN_ID")
    provenance = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "issue": 13,
        "workflow_run_id": int(run_id) if run_id else None,
        "workflow_run_url": (
            "https://github.com/techrote/neurounner/actions/runs/"
            + run_id
            if run_id
            else None
        ),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "preference_config_sha256": sha256_file(args.config),
        "campaign_config_sha256": sha256_file(campaign_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "sft_release_manifest_sha256": sha256_file(
            ROOT / pref["sft_evidence_root"] / "adapter-release.json"
        ),
        "member_contract_sha256": population["member_contract_sha256"],
        "provider_models": {
            role: campaign["nvidia"]["models"][role]["id"]
            for role in ("compiler", "critic", "adjudicator")
        },
    }
    write_json(evidence / "provenance.json", provenance)

    report = [
        "# VILLAGE-001D — Sprocket-feedback preference maturation",
        "",
        "Every SFT villager was exposed to fresh pinned-Sprocket feedback on its own "
        "held-out responses, converted into independently curated preference pairs, "
        "and continued with the PERSONA-002 pairwise objective.",
        "",
        "## Population",
        "",
        "- members: 36",
        f"- Sprocket feedback records: {36 * feedback_per_member}",
        f"- valid parsed Sprocket feedback: {valid_feedback_total}",
        f"- malformed-but-preserved Sprocket feedback: {invalid_feedback_total}",
        f"- preference pairs: {36 * 60}",
        f"- qualified for final population tournament: {len(qualified)}/36",
        f"- preserved as regressed/gate-failed evidence: {len(regressed)}/36",
        "",
        "Qualification here is deliberately narrower than VILLAGE-001E: it blocks "
        "large task-relevance regressions and raw-style-only gains, but it does not "
        "replace the final 36-member blind lineage/sibling tournament.",
        "",
        "## Members",
        "",
        "| Member | Feedback valid | Margin before→after | Pref acc before→after | "
        "Relevance Δ | Behavior-fit Δ | Sibling acc Δ | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in member_rows:
        report.append(
            f"| {row['member_id']} | {row['valid_feedback']}/72 | "
            f"{row['preference_margin_before']:.3f}→"
            f"{row['preference_margin_after']:.3f} | "
            f"{row['preference_accuracy_before']:.3f}→"
            f"{row['preference_accuracy_after']:.3f} | "
            f"{row['task_relevance_delta']:+.3f} | "
            f"{row['behavior_fit_delta']:+.3f} | "
            f"{row['sibling_accuracy_delta']:+.3f} | "
            f"{'PASS' if row['qualified_for_population_tournament'] else 'REGRESSED'} |"
        )
    report.extend(
        [
            "",
            "Raw feedback, provider calls, pair-quality records, full training traces, "
            "blind before/after reviews, and adapter hashes remain durable evidence.",
            "",
        ]
    )
    (evidence / "REPORT.md").write_text("\n".join(report))

    # This inventory is additive to the VILLAGE-001B hashes; do not rewrite the
    # earlier stage's provenance inventory.
    (
        dataset_root / "preference-hashes.sha256"
    ).write_text(
        hash_inventory(
            dataset_root,
            {
                "provenance/hashes.sha256",
                "preference-hashes.sha256",
            },
        )
    )
    (evidence / "hashes.sha256").write_text(
        hash_inventory(evidence, {"hashes.sha256"})
    )

    print(
        json.dumps(
            {
                "experiment": "VILLAGE-001D",
                "members": 36,
                "feedback_records": 36 * feedback_per_member,
                "preference_pairs": 36 * 60,
                "qualified": len(qualified),
                "regressed": len(regressed),
                "release_assets": len(release_assets),
                "release_bytes": sum(
                    row["bytes"] for row in release_assets
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
