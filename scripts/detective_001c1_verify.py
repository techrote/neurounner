#!/usr/bin/env python3
"""Independent standard-library readback for committed DETECTIVE-001C1 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_001c1_dataset as dataset  # noqa: E402

CONFIG = ROOT / "experiments/detective-001/training-c1.json"


class VerifyError(ValueError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise VerifyError(message)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_inventory(root: Path) -> list[str]:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel != "hashes.sha256":
            rows.append(f"{sha_file(path)}  {rel}")
    return rows


def verify(dataset_root: Path, evidence: Path) -> dict:
    cfg = json.loads(CONFIG.read_text())
    dmanifest = dataset.validate_existing(dataset_root)

    expected = {
        "REPORT.md", "adapter-release.json", "campaign-summary.json",
        "final-suite.json", "hashes.sha256", "provenance.json",
        "raw-results.json", "summary.json",
        "training/V-G02-A.json", "training/V-G09-C.json",
    }
    present = {p.relative_to(evidence).as_posix()
               for p in evidence.rglob("*") if p.is_file()}
    need(present == expected, "evidence file set mismatch")
    recorded = (evidence / "hashes.sha256").read_text().splitlines()
    need(recorded == hash_inventory(evidence),
         "evidence hash inventory mismatch")

    final = json.loads((evidence / "final-suite.json").read_text())
    summary = json.loads((evidence / "summary.json").read_text())
    raw = json.loads((evidence / "raw-results.json").read_text())
    campaign = json.loads((evidence / "campaign-summary.json").read_text())
    release = json.loads((evidence / "adapter-release.json").read_text())
    provenance = json.loads((evidence / "provenance.json").read_text())

    need(final.get("schema") == "detective-witness-final-v1"
         and final.get("experiment") == "DETECTIVE-001C1"
         and final.get("generated_after_training") is True
         and len(final.get("scenarios", [])) == 12
         and final.get("exact_training_input_collision_count") == 0,
         "final-suite identity/custody mismatch")
    need({x["case_family"] for x in final["scenarios"]}
         == set(cfg["final_suite_recipe"]["family_names"]),
         "final family set mismatch")
    need(not ({x["case_family"] for x in final["scenarios"]}
              & set(cfg["dataset"]["training_families"])),
         "training/final family overlap")
    need(all(x["mandatory_clue_fallback"]["available"]
             and x["mandatory_clue_fallback"]["kind"] == "public_record"
             for x in final["scenarios"]),
         "mandatory fallback route missing")

    relation_counts = {}
    for row in final["scenarios"]:
        relation_counts[row["relation_label"]] = (
            relation_counts.get(row["relation_label"], 0) + 1)
    for rel in cfg["dataset"]["relation_labels"]:
        need(relation_counts.get(rel) == 2,
             "unbalanced final relation " + rel)

    need(summary.get("schema") == "detective-witness-evaluation-summary-v1"
         and summary.get("final_suite_sha256")
             == sha_file(evidence / "final-suite.json")
         and summary.get("dataset_manifest_sha256")
             == sha_file(dataset_root / "manifest.json")
         and summary.get("larger_cast_expansion_admitted") is False,
         "evaluation summary mismatch or improper expansion admission")
    need(summary.get("reference_relevance_gate")
         == "UNSCORED_NO_INDEPENDENT_0_TO_4_JUDGE",
         "unscored relevance gate was silently relabeled")

    lanes = raw.get("lanes", [])
    need(raw.get("schema") == "detective-witness-evaluation-raw-v1"
         and len(lanes) == 4, "raw evaluation lane count mismatch")
    keys = [(x["member_id"], x["lane_kind"]) for x in lanes]
    need(keys == [
        ("V-G02-A", "parent"), ("V-G02-A", "trained"),
        ("V-G09-C", "parent"), ("V-G09-C", "trained"),
    ], "raw lane identity/order mismatch")
    canonical_prompts = [x["prompt"] for x in final["scenarios"]]
    for lane in lanes:
        responses = lane.get("final_responses", [])
        need(len(responses) == 12, "final response count mismatch")
        need([x["prompt"] for x in responses] == canonical_prompts,
             "parent/trained final prompt mismatch")
        need([x["scenario_id"] for x in responses]
             == [f"F{i:02d}" for i in range(1, 13)],
             "final scenario sequence mismatch")
        need(len(lane.get("reference_responses", [])) == 4,
             "reference response count mismatch")

    assets = release.get("assets", [])
    need(release.get("schema") == "detective-witness-release-v1"
         and release.get("release_tag") == cfg["release"]["tag"]
         and len(assets) == 2, "release manifest identity mismatch")
    by_asset = {x["member_id"]: x for x in assets}
    need(set(by_asset) == {"V-G02-A", "V-G09-C"},
         "release member set mismatch")

    for member in ("V-G02-A", "V-G09-C"):
        result = json.loads(
            (evidence / "training" / f"{member}.json").read_text())
        asset = by_asset[member]
        need(result.get("experiment") == "DETECTIVE-001C1"
             and result.get("member_id") == member
             and result.get("training", {}).get("optimizer_steps") == 64,
             "training result identity/step mismatch " + member)
        need(result["parent_directory_sha256"]
             == cfg["members"][member]["parent_directory_sha256"],
             "training parent hash mismatch " + member)
        need(result["adapter"]["directory_sha256"]
             == asset["adapter_directory_sha256"]
             == final["trained_directory_sha256"][member],
             "trained adapter hash chain mismatch " + member)
        need(asset["parent_directory_sha256"]
             == cfg["members"][member]["parent_directory_sha256"],
             "release parent ancestry mismatch " + member)
        need(re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is not None
             and asset["bytes"] > 0,
             "release archive metadata invalid " + member)

    need(campaign.get("schema") == "detective-witness-c1-summary-v1"
         and campaign.get("dataset_manifest_sha256")
             == sha_file(dataset_root / "manifest.json")
         and campaign.get("evaluation") == summary
         and campaign.get("larger_cast_expansion_admitted") is False,
         "campaign summary mismatch")
    need(re.fullmatch(r"[0-9a-f]{40}", campaign.get("source_head", ""))
         is not None and str(campaign.get("workflow_run_id", "")).isdigit(),
         "campaign source/run identity missing")

    need(provenance.get("config_sha256") == sha_file(CONFIG)
         and provenance.get("dataset_manifest_sha256")
             == sha_file(dataset_root / "manifest.json")
         and provenance.get("final_suite_sha256")
             == sha_file(evidence / "final-suite.json")
         and provenance.get("external_provider_calls") == 0
         and provenance.get("training_members")
             == ["V-G02-A", "V-G09-C"],
         "provenance mismatch")

    return {
        "verdict": "DETECTIVE_001C1_SOURCE_READBACK_PASS",
        "members": 2,
        "training_steps_each": 64,
        "final_scenarios": 12,
        "relation_policy_gate": summary["relation_policy_gate"],
        "reference_relevance_gate": summary["reference_relevance_gate"],
        "larger_cast_expansion_admitted": False,
        "release_tag": release["release_tag"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(verify(args.dataset_root, args.evidence),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
