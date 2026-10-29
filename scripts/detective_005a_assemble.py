#!/usr/bin/env python3
"""Assemble DETECTIVE-005A lineage artifacts into durable raw/review evidence.

Standard-library only. Requires exactly 12 source-matched lineage manifests
and 144 raw responses. It never rates fun, reconstructs identities into the
review packet, trains, infers or admits a production population.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_005a_fun_screen as screen
from scripts import detective_005a_live as live

LINEAGES = [f"G{i:02d}" for i in range(1, 13)]
SHA_RX = re.compile(r"^[0-9a-f]{64}$")
HEAD_RX = re.compile(r"^[0-9a-f]{40}$")
RUN_RX = re.compile(r"^[0-9]+$")


def source_run_identity(rows: list[dict]) -> tuple[str, str]:
    """Reject mixed, unpinned, or foreign Actions-run responses."""
    identities = set()
    for row in rows:
        runner = row.get("runner", {})
        head = runner.get("repository_head_sha")
        run_id = runner.get("actions_run_id")
        screen.require(
            isinstance(head, str) and HEAD_RX.fullmatch(head) is not None and
            isinstance(run_id, str) and RUN_RX.fullmatch(run_id) is not None,
            "source/run identity missing or malformed in raw response")
        identities.add((head, run_id))
    screen.require(len(identities) == 1, "mixed source heads or Actions runs")
    head, run_id = next(iter(identities))
    expected_head = os.environ.get("SOURCE_HEAD_SHA")
    if expected_head:
        screen.require(head == expected_head,
                       "raw response head differs from workflow dispatch")
        expected_run = os.environ.get("GITHUB_RUN_ID")
        if expected_run:
            screen.require(run_id == expected_run,
                           "raw response run differs from workflow execution")
    return head, run_id


def discover_lineages(root: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = {}
    for path in root.rglob("manifest.json"):
        try:
            obj = screen.read_json(path)
        except Exception:
            continue
        if obj.get("schema") != "detective-005a-live-lineage-manifest-v1":
            continue
        lineage = obj.get("lineage")
        if lineage in LINEAGES:
            found.setdefault(lineage, []).append(path.parent)
    duplicates = {k: v for k, v in found.items() if len(v) != 1}
    screen.require(not duplicates, "duplicate lineage artifacts")
    screen.require(set(found) == set(LINEAGES),
                   "all 12 lineage artifacts are required")
    return {k: v[0] for k, v in found.items()}


def read_lineage(cfg: dict, lineage: str, folder: Path) -> tuple[dict, list[dict]]:
    manifest = screen.read_json(folder / "manifest.json")
    results_path = folder / "results.jsonl"
    journal_path = folder / "results.partial.jsonl"
    screen.require(results_path.is_file(), "lineage results.jsonl missing")
    screen.require(journal_path.is_file(), "lineage fsynced journal missing")
    screen.require(journal_path.read_bytes() == results_path.read_bytes(),
                   "lineage fsynced journal/final bytes mismatch")
    screen.require(manifest.get("schema") ==
                   "detective-005a-live-lineage-manifest-v1" and
                   manifest.get("experiment") == "DETECTIVE-005A" and
                   manifest.get("issue") == 176 and
                   manifest.get("lineage") == lineage,
                   "lineage manifest identity drift")
    screen.require(manifest.get("source_plan") == live.source_plan(lineage),
                   "lineage source plan drift")
    screen.require(manifest.get("runner_script_sha256") ==
                   screen.sha256_file(Path(live.__file__)),
                   "lineage runner source hash differs from checked-out code")
    screen.require(manifest.get("results_jsonl_sha256") ==
                   screen.sha256_file(results_path),
                   "lineage raw-results hash mismatch")
    screen.require(manifest.get("results") == 12 and
                   manifest.get("optimizer_steps") == 0 and
                   manifest.get("paid_api_calls") == 0 and
                   manifest.get("model_judge_calls") == 0 and
                   manifest.get("production_admission") is False and
                   manifest.get("population_expansion_admitted") is False,
                   "lineage authority/resource drift")
    rows = screen.read_results(results_path)
    screen.require(len(rows) == 12, "lineage raw result count drift")
    expected_members = live.lineage_member_ids(cfg, lineage)
    screen.require(
        {(r.get("member_id"), r.get("prompt_id")) for r in rows} ==
        {(m, p["id"]) for m in expected_members for p in cfg["prompts"]},
        "lineage member/prompt matrix drift")
    for row in rows:
        screen.validate_result(cfg, row)
    return manifest, rows


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n",
                    encoding="utf-8")


def hash_inventory(root: Path) -> str:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == "provenance/hashes.sha256":
            continue
        rows.append(f"{screen.sha256_file(path)}  {rel}")
    return "\n".join(rows) + "\n"


def assemble(artifacts: Path, out: Path) -> dict:
    cfg = screen.load_contract()
    screen.require(not out.exists(), "output dataset already exists")
    found = discover_lineages(artifacts)
    manifests, rows = {}, []
    for lineage in LINEAGES:
        manifest, lineage_rows = read_lineage(cfg, lineage, found[lineage])
        manifests[lineage] = manifest
        rows.extend(lineage_rows)
    observations = screen.validate_complete_results(cfg, rows)
    source_head, run_id = source_run_identity(rows)

    member_order = {m["member_id"]: i for i, m in enumerate(
        cfg["source_population"]["members"])}
    prompt_order = {p["id"]: i for i, p in enumerate(cfg["prompts"])}
    rows.sort(key=lambda r: (
        member_order[r["member_id"]], prompt_order[r["prompt_id"]]))

    packet, mapping = screen.build_review_packet(cfg, rows)
    screen.require(len(mapping) == 144, "review mapping cardinality drift")
    template = screen.rating_template(packet)

    (out / "raw" / "lineage-manifests").mkdir(parents=True)
    (out / "raw" / "lineage-journals").mkdir(parents=True)
    (out / "review").mkdir(parents=True)
    (out / "provenance").mkdir(parents=True)
    results_path = out / "raw" / "results.jsonl"
    results_path.write_text(
        "".join(screen.canonical(row) + "\n" for row in rows),
        encoding="utf-8")
    for lineage in LINEAGES:
        write_json(out / "raw" / "lineage-manifests" /
                   f"{lineage}.json", manifests[lineage])
        # Retain ORIGINAL fsynced journal bytes, not a reconstructed copy.
        source_journal = found[lineage] / "results.partial.jsonl"
        (out / "raw" / "lineage-journals" / f"{lineage}.jsonl").write_bytes(
            source_journal.read_bytes())
    write_json(out / "review" / "fun-packet.json", packet)
    write_json(out / "review" / "fun-ratings-template.json", template)
    write_json(out / "provenance" / "source-artifacts.json", {
        "schema": "detective-005a-source-artifacts-v1",
        "repository_head_sha": source_head,
        "actions_run_id": run_id,
        "actions_run_url":
            f"https://github.com/techrote/neurounner/actions/runs/{run_id}",
        "source_artifacts": [
            {
                "lineage": lineage,
                "artifact_name": f"detective005a-{lineage}",
                "results_jsonl_sha256": manifests[lineage]["results_jsonl_sha256"],
                "incremental_journal_sha256": screen.sha256_file(
                    out / "raw" / "lineage-journals" / f"{lineage}.jsonl"),
                "runner_script_sha256": manifests[lineage]["runner_script_sha256"],
            }
            for lineage in LINEAGES
        ],
        "artifact_ids_unavailable_at_assembly": True,
    })

    floor_failures = [
        {"member_id": x["member_id"], "prompt_id": x["prompt_id"],
         "hidden_canary_hits": x["hidden_canary_hits"],
         "forbidden_control_hits": x["forbidden_control_hits"],
         "nonempty": x["nonempty"]}
        for x in observations if not x["hard_floor_pass"]
    ]
    manifest = {
        "schema": "detective-005a-dataset-manifest-v1",
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "repository_head_sha": source_head,
        "actions_run_id": run_id,
        "source_config_sha256": screen.sha256_file(screen.CONFIG),
        "release_tag": cfg["source_population"]["release_tag"],
        "release_manifest_git_blob_sha1":
            cfg["source_population"]["release_manifest"]["git_blob_sha1"],
        "member_contract_sha256":
            cfg["source_population"]["release_manifest"]
               ["member_contract_sha256"],
        "members": 36,
        "prompts_per_member": 4,
        "raw_results": 144,
        "raw_results_sha256": screen.sha256_file(results_path),
        "review_packet_sha256": template["packet_sha256"],
        "hard_floor_failures": floor_failures,
        "hard_floor_clean_results": 144 - len(floor_failures),
        "objective_relevance_threshold": None,
        "subjective_fun_ratings_present": False,
        "model_judge_calls": 0,
        "optimizer_steps": 0,
        "production_admission": False,
        "training_admission": False,
        "population_expansion_admitted": False,
    }
    write_json(out / "manifest.json", manifest)
    (out / "README.md").write_text(
        "# DETECTIVE-005A raw fun-first cast screen\n\n"
        "Complete source-bound raw outputs for the 36 VILLAGE-001C SFT "
        "villagers on four matched detective gameplay prompts. This archive "
        "contains no subjective scores yet. The review packet deliberately "
        "hides member identity; the separately generated rating template asks "
        "how much the reviewer wants the NPC to remain in the game. Wrong or "
        "incoherent answers are not automatically bad. Hard authority/leakage "
        "floors are recorded separately and cannot be overridden by a high fun "
        "score. No training, model judge or production admission occurs here.\n",
        encoding="utf-8")
    (out / "provenance" / "hashes.sha256").write_text(
        hash_inventory(out), encoding="utf-8")
    return manifest


def verify_dataset(path: Path) -> dict:
    cfg = screen.load_contract()
    manifest = screen.read_json(path / "manifest.json")
    screen.require(manifest.get("schema") ==
                   "detective-005a-dataset-manifest-v1",
                   "wrong dataset manifest")
    results = screen.read_results(path / "raw" / "results.jsonl")
    screen.validate_complete_results(cfg, results)
    head, run_id = source_run_identity(results)
    screen.require(
        (manifest.get("repository_head_sha"),
         manifest.get("actions_run_id")) == (head, run_id),
        "dataset source/run identity drift")
    provenance = screen.read_json(path / "provenance" / "source-artifacts.json")
    screen.require(
        provenance.get("schema") == "detective-005a-source-artifacts-v1" and
        provenance.get("repository_head_sha") == head and
        provenance.get("actions_run_id") == run_id,
        "dataset source-artifact provenance drift")
    expected_artifacts = []
    for lineage in LINEAGES:
        lineage_manifest = screen.read_json(
            path / "raw" / "lineage-manifests" / f"{lineage}.json")
        lineage_rows = [r for r in results
                        if r["member_id"].split("-")[1] == lineage]
        recreated_sha = screen.sha256_bytes(
            "".join(screen.canonical(r) + "\n"
                    for r in lineage_rows).encode("utf-8"))
        journal_path = path / "raw" / "lineage-journals" / f"{lineage}.jsonl"
        screen.require(
            journal_path.is_file() and
            journal_path.read_bytes() ==
            "".join(screen.canonical(r) + "\n"
                    for r in lineage_rows).encode("utf-8"),
            "archived fsynced lineage journal missing or mismatched")
        screen.require(
            len(lineage_rows) == 12 and
            lineage_manifest.get("lineage") == lineage and
            lineage_manifest.get("source_plan") == live.source_plan(lineage) and
            lineage_manifest.get("results_jsonl_sha256") == recreated_sha,
            "archived lineage raw/source digest drift")
        expected_artifacts.append({
            "lineage": lineage,
            "artifact_name": f"detective005a-{lineage}",
            "results_jsonl_sha256": recreated_sha,
            "incremental_journal_sha256": screen.sha256_file(journal_path),
            "runner_script_sha256":
                lineage_manifest["runner_script_sha256"],
        })
    screen.require(
        provenance.get("source_artifacts") == expected_artifacts,
        "dataset source-artifact manifest/hash drift")
    packet, _ = screen.build_review_packet(cfg, results)
    screen.require(screen.read_json(path / "review" / "fun-packet.json") ==
                   packet, "committed fun packet drift")
    screen.require(screen.read_json(
        path / "review" / "fun-ratings-template.json") ==
        screen.rating_template(packet), "rating template drift")
    screen.require(manifest["raw_results_sha256"] ==
                   screen.sha256_file(path / "raw" / "results.jsonl"),
                   "dataset raw hash drift")
    screen.require(manifest["review_packet_sha256"] ==
                   screen.rating_template(packet)["packet_sha256"],
                   "dataset packet hash drift")
    inventory = {}
    for line in (path / "provenance" / "hashes.sha256").read_text(
            encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        screen.require(rel not in inventory, "duplicate hash inventory path")
        inventory[rel] = digest
    expected_paths = [
        x.relative_to(path).as_posix()
        for x in sorted(path.rglob("*"))
        if x.is_file() and
        x.relative_to(path).as_posix() != "provenance/hashes.sha256"
    ]
    screen.require(sorted(inventory) == sorted(expected_paths),
                   "hash inventory coverage drift")
    for rel, digest in inventory.items():
        screen.require(screen.sha256_file(path / rel) == digest,
                       "hash inventory mismatch: " + rel)
    return {
        "raw_results": len(results),
        "packet_items": len(packet["items"]),
        "hard_floor_failures": len(manifest["hard_floor_failures"]),
        "subjective_fun_ratings_present":
            manifest["subjective_fun_ratings_present"],
        "production_admission": manifest["production_admission"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", type=Path)
    args = ap.parse_args()
    if args.verify:
        print(json.dumps(verify_dataset(args.verify),
                         indent=2, sort_keys=True))
    else:
        screen.require(args.artifacts is not None and args.out is not None,
                       "--artifacts and --out required")
        print(json.dumps(assemble(args.artifacts, args.out),
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
