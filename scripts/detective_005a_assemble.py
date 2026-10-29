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
    require_lineage_member_receipts(cfg, lineage, rows, manifest)
    return manifest, rows


LINEAGE_MANIFEST_KEYS = {
    "schema", "experiment", "issue", "lineage", "source_plan",
    "runner_script_sha256", "results_jsonl_sha256", "results", "members",
    "total_generated_tokens", "peak_rss_kib", "optimizer_steps",
    "paid_api_calls", "model_judge_calls", "production_admission",
    "population_expansion_admitted",
}


def lineage_member_receipts(
        cfg: dict, lineage: str, rows: list[dict]) -> tuple[list[dict], int]:
    """Independently reconstruct the claimed per-member observations.

    The raw, source-matched rows—not editable manifest summaries—are the
    authority for hard-floor status and token counts.
    """
    screen.require(len(rows) == 12, "lineage member summary requires 12 rows")
    keyed = {(r["member_id"], r["prompt_id"]): r for r in rows}
    screen.require(len(keyed) == 12, "lineage summary duplicate member/prompt")
    summaries = []
    for member_id in live.lineage_member_ids(cfg, lineage):
        member_rows = []
        for prompt in cfg["prompts"]:
            row = keyed.get((member_id, prompt["id"]))
            screen.require(row is not None, "lineage summary member/prompt missing")
            member_rows.append(row)
        observations = [screen.validate_result(cfg, row) for row in member_rows]
        summaries.append({
            "member_id": member_id,
            "hard_floor_pass": all(x["hard_floor_pass"] for x in observations),
            "hard_floor_failures": [
                x["prompt_id"] for x in observations if not x["hard_floor_pass"]],
            "generated_tokens": sum(
                row["response"]["completion_tokens"] for row in member_rows),
        })
    return summaries, sum(item["generated_tokens"] for item in summaries)


def require_lineage_member_receipts(
        cfg: dict, lineage: str, rows: list[dict], manifest: dict) -> None:
    screen.require(isinstance(manifest, dict) and
                   set(manifest) == LINEAGE_MANIFEST_KEYS,
                   "lineage manifest field-set drift")
    members, total_tokens = lineage_member_receipts(cfg, lineage, rows)
    screen.require(
        manifest.get("members") == members and
        manifest.get("total_generated_tokens") == total_tokens and
        type(manifest.get("peak_rss_kib")) is int and
        manifest["peak_rss_kib"] > 0,
        "lineage member/token/resource summary drift")


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



DATASET_MANIFEST_KEYS = {
    "schema", "experiment", "issue", "repository_head_sha", "actions_run_id",
    "source_config_sha256", "release_tag", "release_manifest_git_blob_sha1",
    "member_contract_sha256", "members", "prompts_per_member", "raw_results",
    "raw_results_sha256", "review_packet_sha256", "private_blind_review_status",
    "review_packet_generated", "hard_floor_failures",
    "hard_floor_clean_results", "objective_relevance_threshold",
    "subjective_fun_ratings_present", "model_judge_calls", "optimizer_steps",
    "production_admission", "training_admission",
    "population_expansion_admitted",
}
SOURCE_ARTIFACT_PROVENANCE_KEYS = {
    "schema", "repository_head_sha", "actions_run_id", "actions_run_url",
    "source_artifacts", "artifact_ids_unavailable_at_assembly",
}


def require_canonical_json(path: Path, value, label: str) -> None:
    """Reject hidden duplicate JSON keys and noncanonical extra raw bytes."""
    expected = (json.dumps(value, indent=2, sort_keys=True,
                           ensure_ascii=False) + "\n").encode("utf-8")
    screen.require(path.read_bytes() == expected,
                   label + " noncanonical JSON or duplicate members")


def public_readme_text() -> str:
    return (
        "# DETECTIVE-005A raw fun-first cast screen\n\n"
        "Complete source-bound raw outputs for the 36 VILLAGE-001C SFT "
        "villagers on four matched detective gameplay prompts. This archive "
        "contains no subjective scores and NO review packet. The old public "
        "fixed-salt item ordering was reconstructible. A separate private "
        "custodian must prepare a random-key blinded packet outside the "
        "repository, retain mapping/key in private custody, freeze all human "
        "ratings, and only then authorize identity unblinding. Never hand "
        "this raw archive to a supposedly blind reviewer. Wrong or incoherent "
        "answers are not automatically bad. Mechanical hard authority/leakage "
        "floors remain separate from subjective fun. No training, model judge "
        "or production admission occurs here.\n"
    )

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

    # Public evidence archival must NEVER issue an identity-bearing or
    # publicly reconstructible review packet. A private custodian prepares
    # the blinded packet only after complete, independently verified rows.
    (out / "raw" / "lineage-manifests").mkdir(parents=True)
    (out / "raw" / "lineage-journals").mkdir(parents=True)
    (out / "provenance").mkdir(parents=True)
    results_path = out / "raw" / "results.jsonl"
    results_path.write_text(
        "".join(screen.canonical(row) + "\n" for row in rows),
        encoding="utf-8")
    for lineage in LINEAGES:
        write_json(out / "raw" / "lineage-manifests" /
                   f"{lineage}.json", manifests[lineage])
        # Retain the original fsynced bytes, never a reconstructed journal.
        (out / "raw" / "lineage-journals" / f"{lineage}.jsonl").write_bytes(
            (found[lineage] / "results.partial.jsonl").read_bytes())
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
        "review_packet_sha256": None,
        "private_blind_review_status": "not_prepared_outside_repository",
        "review_packet_generated": False,
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
    (out / "README.md").write_text(public_readme_text(), encoding="utf-8")
    (out / "provenance" / "hashes.sha256").write_text(
        hash_inventory(out), encoding="utf-8")
    return manifest


def verify_dataset(path: Path) -> dict:
    # A rehashed SHA inventory is self-reported. It cannot authorize extra
    # files (including a private HMAC key/mapping) or external symlink targets.
    # Git archives symlinks as links, not the underlying original raw bytes.
    screen.require(path.is_dir() and not path.is_symlink(),
                   "full-screen archive root must be a physical directory")
    screen.require(all(not p.is_symlink() for p in path.rglob("*")),
                   "full-screen archive contains symlink")
    expected_files = {
        "README.md", "manifest.json", "raw/results.jsonl",
        "provenance/source-artifacts.json", "provenance/hashes.sha256",
    }
    expected_files.update(
        f"raw/lineage-manifests/{lineage}.json" for lineage in LINEAGES)
    expected_files.update(
        f"raw/lineage-journals/{lineage}.jsonl" for lineage in LINEAGES)
    actual_files = {
        p.relative_to(path).as_posix()
        for p in path.rglob("*") if p.is_file()
    }
    screen.require(actual_files == expected_files,
                   "full-screen archive has unregistered or missing files")
    screen.require((path / "README.md").read_text(encoding="utf-8") ==
                   public_readme_text(),
                   "full-screen README content drift")

    cfg = screen.load_contract()
    manifest_path = path / "manifest.json"
    manifest = screen.read_json(manifest_path)
    screen.require(isinstance(manifest, dict) and
                   set(manifest) == DATASET_MANIFEST_KEYS,
                   "full-screen dataset manifest field-set drift")
    require_canonical_json(manifest_path, manifest, "dataset manifest")
    screen.require(manifest.get("schema") ==
                   "detective-005a-dataset-manifest-v1",
                   "wrong dataset manifest")
    screen.require(
        manifest.get("experiment") == "DETECTIVE-005A" and
        manifest.get("issue") == 176 and
        manifest.get("members") == 36 and
        manifest.get("prompts_per_member") == 4 and
        manifest.get("raw_results") == 144 and
        manifest.get("source_config_sha256") == screen.sha256_file(screen.CONFIG) and
        manifest.get("release_tag") == cfg["source_population"]["release_tag"] and
        manifest.get("release_manifest_git_blob_sha1") ==
            cfg["source_population"]["release_manifest"]["git_blob_sha1"] and
        manifest.get("member_contract_sha256") ==
            cfg["source_population"]["release_manifest"]["member_contract_sha256"],
        "full-screen frozen source/size manifest drift")
    screen.require(
        manifest.get("review_packet_generated") is False and
        manifest.get("review_packet_sha256") is None and
        manifest.get("private_blind_review_status") ==
            "not_prepared_outside_repository" and
        manifest.get("subjective_fun_ratings_present") is False and
        manifest.get("objective_relevance_threshold") is None and
        manifest.get("model_judge_calls") == 0 and
        manifest.get("optimizer_steps") == 0 and
        manifest.get("production_admission") is False and
        manifest.get("training_admission") is False and
        manifest.get("population_expansion_admitted") is False,
        "full-screen rating/resource/admission authority drift")
    results = screen.read_results(path / "raw" / "results.jsonl")
    observations = screen.validate_complete_results(cfg, results)
    expected_floor_failures = [
        {"member_id": x["member_id"], "prompt_id": x["prompt_id"],
         "hidden_canary_hits": x["hidden_canary_hits"],
         "forbidden_control_hits": x["forbidden_control_hits"],
         "nonempty": x["nonempty"]}
        for x in observations if not x["hard_floor_pass"]
    ]
    screen.require(
        manifest.get("hard_floor_failures") == expected_floor_failures and
        manifest.get("hard_floor_clean_results") == 144 - len(expected_floor_failures),
        "full-screen mechanical hard-floor summary drift")
    head, run_id = source_run_identity(results)
    screen.require(
        (manifest.get("repository_head_sha"),
         manifest.get("actions_run_id")) == (head, run_id),
        "dataset source/run identity drift")
    provenance_path = path / "provenance" / "source-artifacts.json"
    provenance = screen.read_json(provenance_path)
    screen.require(isinstance(provenance, dict) and
                   set(provenance) == SOURCE_ARTIFACT_PROVENANCE_KEYS and
                   provenance.get("actions_run_url") ==
                       f"https://github.com/techrote/neurounner/actions/runs/{run_id}",
                   "full-screen source-artifact metadata field-set or URL drift")
    require_canonical_json(provenance_path, provenance, "source provenance")
    screen.require(
        provenance.get("schema") == "detective-005a-source-artifacts-v1" and
        provenance.get("repository_head_sha") == head and
        provenance.get("actions_run_id") == run_id,
        "dataset source-artifact provenance drift")
    screen.require(provenance.get("artifact_ids_unavailable_at_assembly") is True,
                   "full-screen source-artifact transport provenance drift")
    expected_artifacts = []
    for lineage in LINEAGES:
        lineage_manifest_path = (path / "raw" / "lineage-manifests" /
                                 f"{lineage}.json")
        lineage_manifest = screen.read_json(lineage_manifest_path)
        require_canonical_json(lineage_manifest_path, lineage_manifest,
                               "lineage manifest")
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
            lineage_manifest.get("schema") == "detective-005a-live-lineage-manifest-v1" and
            lineage_manifest.get("experiment") == "DETECTIVE-005A" and
            lineage_manifest.get("issue") == 176 and
            lineage_manifest.get("lineage") == lineage and
            lineage_manifest.get("source_plan") == live.source_plan(lineage) and
            lineage_manifest.get("runner_script_sha256") ==
                screen.sha256_file(Path(live.__file__)) and
            lineage_manifest.get("results_jsonl_sha256") == recreated_sha and
            lineage_manifest.get("results") == 12 and
            lineage_manifest.get("optimizer_steps") == 0 and
            lineage_manifest.get("paid_api_calls") == 0 and
            lineage_manifest.get("model_judge_calls") == 0 and
            lineage_manifest.get("production_admission") is False and
            lineage_manifest.get("population_expansion_admitted") is False,
            "archived lineage raw/source/authority drift")
        require_lineage_member_receipts(cfg, lineage, lineage_rows,
                                        lineage_manifest)
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
    screen.require(
        manifest.get("review_packet_generated") is False and
        manifest.get("review_packet_sha256") is None and
        manifest.get("private_blind_review_status") ==
            "not_prepared_outside_repository" and
        not (path / "review").exists(),
        "public dataset must not include a review packet or secret mapping")
    screen.require(manifest["raw_results_sha256"] ==
                   screen.sha256_file(path / "raw" / "results.jsonl"),
                   "dataset raw hash drift")
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
        "packet_items": 0,
        "review_packet_generated": False,
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
