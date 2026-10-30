#!/usr/bin/env python3
"""Independent standard-library readback for committed DETECTIVE-001C2 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import detective_001c2_dataset as dataset  # noqa: E402
from scripts import detective_001c2_evaluator as proposition  # noqa: E402

CONFIG = ROOT / "experiments/detective-001/training-c2.json"


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


def summarize_raw_lane(lane: dict) -> dict:
    relations = {}
    protected = evidence = truncated = 0
    rows = lane["final_responses"]
    for row in rows:
        score = proposition.score_response({
            "relation_label": row["relation_label"],
            "evaluation": row["evaluation"],
        }, row["response"])
        need(score == row["score"], "raw stored proposition score does not replay")
        rel = row["relation_label"]
        relations.setdefault(rel, {"pass": 0, "total": 0})
        relations[rel]["total"] += 1
        relations[rel]["pass"] += int(score["pass"])
        evidence += len(score["checks"]["evidence_id_leaks"])
        truncated += int(row["generated_tokens"] >= 96)
        if rel == "intentional_lie":
            protected += int(score["checks"]["truth_hit"])
        elif rel == "withheld_fact":
            protected += int(score["checks"]["protected_truth_hit"])
    passes = sum(x["score"]["pass"] for x in rows)
    return {
        "policy_passes": passes, "policy_total": len(rows),
        "policy_score": passes / len(rows),
        "protected_fact_leaks": protected,
        "evidence_id_leaks": evidence, "truncated": truncated,
        "relation": dict(sorted(relations.items())),
        "lie_passes": relations["intentional_lie"]["pass"],
        "lie_total": relations["intentional_lie"]["total"],
        "withholding_passes": relations["withheld_fact"]["pass"],
        "withholding_total": relations["withheld_fact"]["total"],
        "reference_relevance_0_to_4": None,
        "reference_relevance_status": "UNSCORED_NO_INDEPENDENT_RATINGS",
    }


def verify(dataset_root: Path, evidence: Path) -> dict:
    cfg = json.loads(CONFIG.read_text())
    dmanifest = dataset.validate_existing(dataset_root)
    proposition.load_contract()

    expected = {
        "REPORT.md", "adapter-release.json", "campaign-summary.json",
        "final-suite.json", "hashes.sha256", "provenance.json",
        "raw-results.json", "relevance-packet.json", "summary.json",
        "training/V-G02-A.json", "training/V-G09-C.json",
    }
    present = {p.relative_to(evidence).as_posix()
               for p in evidence.rglob("*") if p.is_file()}
    need(present == expected, "C2 evidence file set mismatch")
    need((evidence / "hashes.sha256").read_text().splitlines()
         == hash_inventory(evidence), "C2 evidence hash inventory mismatch")

    final = json.loads((evidence / "final-suite.json").read_text())
    summary = json.loads((evidence / "summary.json").read_text())
    raw = json.loads((evidence / "raw-results.json").read_text())
    relevance = json.loads((evidence / "relevance-packet.json").read_text())
    campaign = json.loads((evidence / "campaign-summary.json").read_text())
    release = json.loads((evidence / "adapter-release.json").read_text())
    provenance = json.loads((evidence / "provenance.json").read_text())

    need(final.get("schema") == "detective-c2-final-v1"
         and final.get("experiment") == "DETECTIVE-001C2"
         and final.get("generated_after_training") is True
         and final.get("c1_final_used_as_training") is False
         and len(final.get("scenarios", [])) == 18
         and final.get("exact_historical_input_collision_count") == 0,
         "C2 final identity/custody mismatch")
    families = {x["case_family"] for x in final["scenarios"]}
    need(families == set(cfg["final_suite_recipe"]["family_names"]),
         "C2 final family set mismatch")
    need(all(x["mandatory_clue_fallback"]["available"]
             and x["mandatory_clue_fallback"]["kind"] == "public_record"
             for x in final["scenarios"]),
         "C2 final public fallback missing")
    counts = {}
    for row in final["scenarios"]:
        counts[row["relation_label"]] = counts.get(row["relation_label"], 0) + 1
    need(counts == {
        "truthful_known_fact": 3, "corrupted_belief": 3,
        "intentional_lie": 3, "withheld_fact": 3,
        "contradiction_repair": 3, "uninformed_speculation": 3,
    }, "C2 final relation balance mismatch")

    need(summary.get("schema") == "detective-c2-evaluation-summary-v1"
         and summary.get("final_suite_sha256")
             == sha_file(evidence / "final-suite.json")
         and summary.get("pair_dataset_manifest_sha256")
             == sha_file(dataset_root / "manifest.json")
         and summary.get("independent_relevance_gate")
             == "UNSCORED_NO_INDEPENDENT_RATINGS"
         and summary.get("larger_cast_expansion_admitted") is False,
         "C2 summary identity/relevance/expansion mismatch")

    lanes = raw.get("lanes", [])
    need(raw.get("schema") == "detective-c2-evaluation-raw-v1"
         and len(lanes) == 4, "C2 raw lane count mismatch")
    need([(x["member_id"], x["lane_kind"]) for x in lanes] == [
        ("V-G02-A", "parent"), ("V-G02-A", "trained"),
        ("V-G09-C", "parent"), ("V-G09-C", "trained"),
    ], "C2 raw lane identity/order mismatch")
    canonical_prompts = [x["prompt"] for x in final["scenarios"]]
    by_lane = {}
    for lane in lanes:
        need(len(lane.get("final_responses", [])) == 18,
             "C2 final response count mismatch")
        need([x["prompt"] for x in lane["final_responses"]] == canonical_prompts,
             "C2 parent/trained prompt mismatch")
        need([x["scenario_id"] for x in lane["final_responses"]]
             == [f"F{i:02d}" for i in range(1, 19)],
             "C2 final scenario order mismatch")
        need(len(lane.get("reference_responses", [])) == 4,
             "C2 reference response count mismatch")
        replay = summarize_raw_lane(lane)
        need(replay == lane["summary"], "C2 lane summary does not replay")
        by_lane[(lane["member_id"], lane["lane_kind"])] = replay

    # Recompute every decision gate rather than trusting summary prose.
    gate = True
    for member in ("V-G02-A", "V-G09-C"):
        p = by_lane[(member, "parent")]
        t = by_lane[(member, "trained")]
        score_improved = t["policy_score"] > p["policy_score"]
        leakage_ok = (
            t["protected_fact_leaks"] <= p["protected_fact_leaks"]
            and t["evidence_id_leaks"] == 0)
        lie_ok = t["lie_passes"] >= cfg["evaluation"]["minimum_lie_passes_each"]
        withholding_ok = (
            t["withholding_passes"]
            >= cfg["evaluation"]["minimum_withholding_passes_each"])
        recomputed = score_improved and leakage_ok and lie_ok and withholding_ok
        gate &= recomputed
        stored = summary["comparisons"][member]
        need(stored["parent"] == p and stored["trained"] == t,
             "C2 comparison lane summary mismatch " + member)
        need(stored["policy_score_improved"] == score_improved
             and stored["protected_leakage_not_increased"] == leakage_ok
             and stored["lie_gate"] == lie_ok
             and stored["withholding_gate"] == withholding_ok
             and stored["member_proposition_gate"] == recomputed,
             "C2 stored gate does not replay " + member)
    need(summary["proposition_gate"] == gate,
         "C2 aggregate proposition gate mismatch")

    need(relevance.get("schema") == "detective-c2-post-blind-relevance-v1"
         and relevance.get("identity_mapping_committed") is False
         and len(relevance.get("items", [])) == 16
         and all(x["score_0_to_4"] is None for x in relevance["items"]),
         "C2 blind relevance packet invalid/pre-scored")
    rendered = json.dumps(relevance, sort_keys=True)
    for secret in ("V-G02-A", "V-G09-C", '"member_id"', '"lane_kind"'):
        need(secret not in rendered, "C2 relevance packet identity leak")

    assets = release.get("assets", [])
    need(release.get("schema") == "detective-witness-preference-release-v1"
         and release.get("experiment") == "DETECTIVE-001C2"
         and release.get("release_tag") == cfg["release"]["tag"]
         and len(assets) == 2, "C2 release manifest mismatch")
    by_asset = {x["member_id"]: x for x in assets}
    need(set(by_asset) == {"V-G02-A", "V-G09-C"},
         "C2 release member set mismatch")
    for member in ("V-G02-A", "V-G09-C"):
        result = json.loads(
            (evidence / "training" / f"{member}.json").read_text())
        asset = by_asset[member]
        need(result.get("schema") == "detective-witness-preference-result-v1"
             and result.get("experiment") == "DETECTIVE-001C2"
             and result.get("member_id") == member
             and result.get("training", {}).get("optimizer_steps") == 40,
             "C2 training result identity/step mismatch " + member)
        need(result["parent_directory_sha256"]
             == cfg["members"][member]["parent_directory_sha256"],
             "C2 parent hash mismatch " + member)
        need(result["adapter"]["directory_sha256"]
             == asset["adapter_directory_sha256"]
             == final["trained_directory_sha256"][member],
             "C2 trained adapter hash chain mismatch " + member)
        need(asset["parent_directory_sha256"]
             == cfg["members"][member]["parent_directory_sha256"],
             "C2 release parent ancestry mismatch " + member)
        need(re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is not None
             and asset["bytes"] > 0, "C2 release archive metadata invalid")

    need(campaign.get("schema") == "detective-witness-c2-summary-v1"
         and campaign.get("pair_dataset_manifest_sha256")
             == sha_file(dataset_root / "manifest.json")
         and campaign.get("evaluation") == summary
         and campaign.get("larger_cast_expansion_admitted") is False,
         "C2 campaign summary mismatch")
    need(re.fullmatch(r"[0-9a-f]{40}", campaign.get("source_head", ""))
         is not None and str(campaign.get("workflow_run_id", "")).isdigit(),
         "C2 source/run identity missing")
    need(provenance.get("config_sha256") == sha_file(CONFIG)
         and provenance.get("frozen_evaluator_merge_commit")
             == cfg["evaluator"]["merge_commit"]
         and provenance.get("pair_dataset_manifest_sha256")
             == sha_file(dataset_root / "manifest.json")
         and provenance.get("final_suite_sha256")
             == sha_file(evidence / "final-suite.json")
         and provenance.get("external_provider_calls") == 0
         and provenance.get("c1_final_used_as_training") is False,
         "C2 provenance mismatch")
    return {
        "verdict": "DETECTIVE_001C2_SOURCE_READBACK_PASS",
        "members": 2, "training_steps_each": 40,
        "final_scenarios": 18, "proposition_gate": gate,
        "independent_relevance_gate":
            summary["independent_relevance_gate"],
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
