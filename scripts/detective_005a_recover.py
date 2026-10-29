#!/usr/bin/env python3
"""DETECTIVE-005A failed-workflow raw custody (NO TRAINING).

Archives whatever lineage artifacts actually survived a failed/cancelled
matrix or failed 144-row assembly. Never assembles a fun packet, restarts
inference, repairs source rows, or treats partials as accepted behavior.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_005a_fun_screen as screen
from scripts import detective_005a_live as live
from scripts import detective_005a_assemble as assemble

HEAD_RX = re.compile(r"^[0-9a-f]{40}$")
RUN_RX = re.compile(r"^[1-9][0-9]*$")
ALLOWED = {
    "results.partial.jsonl", "results.jsonl", "manifest.json",
    "interrupted.json",
}


def ensure_identity(head: str, run_id: str) -> None:
    screen.require(isinstance(head, str) and HEAD_RX.fullmatch(head) is not None,
                   "recovery source head must be a 40-character commit SHA")
    screen.require(isinstance(run_id, str) and RUN_RX.fullmatch(run_id) is not None,
                   "recovery Actions run ID must be a positive integer string")


def artifact_files(root: Path, lineage: str) -> dict[str, Path]:
    """A per-name artifact must not smuggle weights, unexpected or duplicate files."""
    source = root / f"detective005a-{lineage}"
    if not source.exists():
        return {}
    screen.require(source.is_dir() and not source.is_symlink(),
                   "lineage artifact path is not a directory")
    result = {}
    for path in source.rglob("*"):
        screen.require(not path.is_symlink(), "symlink in lineage artifact")
        if path.is_dir():
            screen.require("work" not in path.relative_to(source).parts,
                           "unpacked adapter work folder in artifact")
            continue
        screen.require(path.is_file() and path.name in ALLOWED,
                       "unexpected file in lineage artifact")
        screen.require(path.name not in result,
                       "ambiguous duplicate artifact filename")
        result[path.name] = path
    return result


def inspect_raw(cfg: dict, lineage: str, source: Path,
                head: str, run_id: str) -> tuple[int, str | None, list[str]]:
    """Validate only the intact canonical prefix; NEVER modify the original bytes."""
    syntactic, tail_problem = live.intact_partial_raw_rows(source)
    errors: list[str] = []
    valid = 0
    expected = [(f"V-{lineage}-{s}", prompt["id"])
                for s in "ABC" for prompt in cfg["prompts"]]
    if syntactic > 12:
        errors.append("too_many_jsonl_records")
    with source.open("rb") as fh:
        for offset in range(min(syntactic, 12)):
            line = fh.readline()
            try:
                row = json.loads(line)
                screen.validate_result(cfg, row)
                identity = row["member_id"], row["prompt_id"]
                screen.require(identity == expected[offset],
                               "member/prompt ordering mismatch")
                runner = row["runner"]
                frozen = row["source"]
                screen.require(
                    runner.get("repository_head_sha") == head and
                    runner.get("actions_run_id") == run_id,
                    "response source/run identity mismatch")
                screen.require(
                    runner.get("base_weights_sha256_verified") ==
                    frozen["base_weights_sha256"] and
                    runner.get("adapter_archive_sha256_verified") ==
                    frozen["asset_sha256"] and
                    runner.get("adapter_directory_sha256_verified") ==
                    frozen["adapter_directory_sha256"],
                    "response verified source hash mismatch")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                errors.append(f"invalid_response_{offset + 1}:{type(exc).__name__}")
                break
            valid += 1
    return valid, tail_problem, errors


def inspect_artifact(cfg: dict, lineage: str, files: dict[str, Path],
                     head: str, run_id: str) -> dict:
    """Retain failed/foreign data but never certify it as qualified output."""
    errors: list[str] = []
    preferred = files.get("results.partial.jsonl") or files.get("results.jsonl")
    valid, tail_problem = 0, None
    if preferred:
        valid, tail_problem, row_errors = inspect_raw(
            cfg, lineage, preferred, head, run_id)
        errors.extend(row_errors)
    if not files:
        status = "artifact_missing"
    else:
        status = "recovery_only"
        if not preferred:
            errors.append("raw_results_missing")
        if "interrupted.json" in files:
            try:
                receipt = screen.read_json(files["interrupted.json"])
                count = live.intact_partial_raw_rows(preferred)[0] if preferred else 0
                source_sha = screen.sha256_file(preferred) if preferred else None
                screen.require(
                    receipt.get("schema") == "detective-005a-interrupted-lineage-v1"
                    and receipt.get("lineage") == lineage
                    and receipt.get("repository_head_sha") == head
                    and receipt.get("actions_run_id") == run_id
                    and receipt.get("completed_raw_results") == count
                    and receipt.get("partial_results_sha256") == source_sha
                    and receipt.get("optimizer_steps") == 0
                    and receipt.get("model_judge_calls") == 0
                    and receipt.get("durably_promoted") is False
                    and receipt.get("production_admission") is False
                    and receipt.get("population_expansion_admitted") is False,
                    "interruption receipt source/hash/authority mismatch")
                if preferred and "results.partial.jsonl" not in files:
                    errors.append("interrupted_receipt_without_partial_journal")
            except (ValueError, KeyError, TypeError, AttributeError):
                errors.append("invalid_interruption_receipt")
        else:
            if valid < 12 or tail_problem:
                errors.append("missing_interruption_receipt")
        if "manifest.json" in files and "results.jsonl" in files:
            # Classify lost or corrupted fsynced journals explicitly.
            # The strict assembler now refuses both cases; do not collapse a
            # demonstrable transport gap into a generic bad-manifest warning.
            if "results.partial.jsonl" not in files:
                errors.append("complete_lineage_missing_incremental_journal")
            elif (files["results.partial.jsonl"].read_bytes() !=
                  files["results.jsonl"].read_bytes()):
                errors.append("complete_lineage_incremental_journal_mismatch")
            else:
                try:
                    manifest, rows = assemble.read_lineage(
                        cfg, lineage, files["manifest.json"].parent)
                    # Complete lineage is not complete full-matrix evidence.
                    assemble.source_run_identity(rows)
                    screen.require(
                        rows[0]["runner"]["repository_head_sha"] == head and
                        rows[0]["runner"]["actions_run_id"] == run_id,
                        "complete lineage source/run mismatch")
                    if "interrupted.json" not in files and not errors:
                        status = "complete_lineage_recovered"
                except (ValueError, KeyError, TypeError, AttributeError,
                        OSError):
                    errors.append("invalid_complete_lineage_manifest")
        elif "manifest.json" in files or "results.jsonl" in files:
            errors.append("incomplete_final_lineage_pair")
    return {
        "lineage": lineage,
        "status": status,
        "validated_raw_prefix_rows": valid,
        "journal_tail_problem": tail_problem,
        "custody_warnings": errors,
        "files": [
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": screen.sha256_file(path),
            }
            for name, path in sorted(files.items())
        ],
    }


def metadata_by_name(metadata_path: Path | None) -> dict[str, dict]:
    if metadata_path is None:
        return {}
    value = screen.read_json(metadata_path)
    screen.require(isinstance(value, dict) and
                   isinstance(value.get("artifacts"), list),
                   "invalid GitHub Actions artifact API response")
    found = {}
    for entry in value["artifacts"]:
        name = entry.get("name")
        if name and re.fullmatch(r"detective005a-G(0[1-9]|1[0-2])", name):
            screen.require(name not in found, "duplicate artifact API identity")
            source_run = entry.get("workflow_run")
            found[name] = {
                "id": entry.get("id"),
                "name": name,
                "size_in_bytes": entry.get("size_in_bytes"),
                "digest": entry.get("digest"),
                "expired": entry.get("expired"),
                "workflow_run": {
                    "id": source_run.get("id"),
                    "head_sha": source_run.get("head_sha"),
                    "repository_id": source_run.get("repository_id"),
                } if isinstance(source_run, dict) else None,
            }
    return found


def load_api_for_custody(metadata_path: Path | None) -> tuple[dict[str, dict], bool]:
    """Keep malformed *original* API bytes instead of aborting raw recovery.

    Syntactic/shape failures are quarantined; an unreadable file still fails
    because its original bytes cannot be preserved and verified.
    """
    if metadata_path is None:
        return {}, False
    try:
        return metadata_by_name(metadata_path), False
    except (ValueError, TypeError, AttributeError, KeyError):
        return {}, True


def bind_artifact_api(report: dict, name: str, api: dict[str, dict],
                      metadata_present: bool, head: str,
                      run_id: str, metadata_invalid: bool = False) -> dict:
    """Bind artifact ID/digest and prevent foreign/unknown API source admission.

    Keep the original raw files even when API identity fails. A damaged or
    foreign metadata record is a custody warning, never completed evidence.
    """
    entry = api.get(name)
    report["artifact_name"] = name
    report["artifact_api"] = entry
    issues = report["custody_warnings"]
    if metadata_invalid and report["files"]:
        issues.append("artifact_api_metadata_invalid")
    elif entry is None and report["files"] and metadata_present:
        issues.append("downloaded_artifact_not_listed")
    if not metadata_present and report["status"] == "complete_lineage_recovered":
        # Local source integrity != independently verified Actions artifact
        # lineage. Preserve bytes but quarantine the custody claim.
        issues.append("artifact_api_metadata_unavailable")
    if entry is not None and not report["files"]:
        issues.append("listed_artifact_not_downloaded")
    if entry is not None:
        source = entry.get("workflow_run")
        if not isinstance(source, dict) or (
            source.get("id") != int(run_id) or
            source.get("head_sha") != head
        ):
            issues.append("artifact_api_source_run_mismatch")
        if type(entry.get("id")) is not int or entry["id"] <= 0:
            issues.append("artifact_api_id_missing")
        if (type(entry.get("size_in_bytes")) is not int or
                entry["size_in_bytes"] <= 0):
            issues.append("artifact_api_size_missing")
        digest = entry.get("digest")
        if (not isinstance(digest, str) or
                re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None):
            issues.append("artifact_api_digest_missing")
        if entry.get("expired") is not False:
            issues.append("artifact_api_expired_or_unknown")
    if report["status"] == "complete_lineage_recovered" and issues:
        report["status"] = "recovery_only"
    return report


TRANSPORT_CUSTODY_WARNINGS = frozenset({
    "listed_artifact_not_downloaded",
    "downloaded_artifact_not_listed",
    "artifact_api_source_run_mismatch",
    "artifact_api_id_missing",
    "artifact_api_size_missing",
    "artifact_api_digest_missing",
    "artifact_api_expired_or_unknown",
    "artifact_api_metadata_unavailable",
    "artifact_api_metadata_invalid",
})


def artifact_transport_complete(records: list[dict], api: dict[str, dict],
                                metadata_present: bool,
                                api_outcome: str, download_outcome: str,
                                metadata_invalid: bool = False) -> bool:
    """Complete *retrieval* custody only; never a complete 144-row screen."""
    return (metadata_present and not metadata_invalid and bool(api) and
            api_outcome == "success" and download_outcome == "success" and
            not any(TRANSPORT_CUSTODY_WARNINGS.intersection(
                r["custody_warnings"]) for r in records))


def recover(artifacts: Path, out: Path, head: str, run_id: str,
            metadata_path: Path | None = None,
            artifact_api_outcome: str = "unknown",
            download_outcome: str = "unknown") -> dict:
    """Archive immutable original per-lineage bytes, including negative evidence."""
    ensure_identity(head, run_id)
    screen.require(
        artifact_api_outcome in {"success", "failure", "skipped", "cancelled", "unknown"} and
        download_outcome in {"success", "failure", "skipped", "cancelled", "unknown"},
        "invalid artifact-transport outcome")
    cfg = screen.load_contract()
    screen.require(not out.exists(), "refuse overwrite of recovery archive")
    screen.require(artifacts.is_dir(), "downloaded artifact root missing")
    api, metadata_invalid = load_api_for_custody(metadata_path)
    records = []
    staged: list[tuple[str, dict[str, Path]]] = []
    for lineage in assemble.LINEAGES:
        name = f"detective005a-{lineage}"
        files = artifact_files(artifacts, lineage)
        report = bind_artifact_api(
            inspect_artifact(cfg, lineage, files, head, run_id),
            name, api, metadata_path is not None, head, run_id,
            metadata_invalid=metadata_invalid)
        staged.append((lineage, files))
        records.append(report)

    (out / "raw").mkdir(parents=True)
    (out / "provenance").mkdir(parents=True)
    for lineage, files in staged:
        for name, source in files.items():
            destination = out / "raw" / lineage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            screen.require(source.read_bytes() == destination.read_bytes(),
                           "recovered original raw byte drift")
    if metadata_path:
        shutil.copyfile(metadata_path, out / "provenance" / "actions-artifacts-api.json")

    completed = sum(r["status"] == "complete_lineage_recovered" for r in records)
    count = sum(r["validated_raw_prefix_rows"] for r in records)
    transport_complete = artifact_transport_complete(
        records, api, metadata_path is not None,
        artifact_api_outcome, download_outcome, metadata_invalid)
    manifest = {
        "schema": "detective-005a-partial-recovery-v1",
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "status": "raw_recovery_only_not_a_complete_screen",
        "repository_head_sha": head,
        "actions_run_id": run_id,
        "actions_run_url":
            f"https://github.com/techrote/neurounner/actions/runs/{run_id}",
        "source_config_sha256": screen.sha256_file(screen.CONFIG),
        "runner_script_sha256": screen.sha256_file(Path(live.__file__)),
        "recovery_script_sha256": screen.sha256_file(Path(__file__)),
        "lineages_reported": len(records),
        "lineages_with_artifacts":
            sum(bool(r["files"]) for r in records),
        "complete_lineage_artifacts": completed,
        "validated_raw_prefix_rows": count,
        "artifact_api_outcome": artifact_api_outcome,
        "artifact_download_outcome": download_outcome,
        "artifact_api_metadata_invalid": metadata_invalid,
        "artifact_transport_complete": transport_complete,
        "review_packet_generated": False,
        "training_admission": False,
        "production_admission": False,
        "population_expansion_admitted": False,
        "optimizer_steps": 0,
        "model_judge_calls": 0,
    }
    assemble.write_json(out / "manifest.json", manifest)
    assemble.write_json(out / "provenance" / "source-artifacts.json", {
        "schema": "detective-005a-recovery-source-artifacts-v1",
        "repository_head_sha": head,
        "actions_run_id": run_id,
        "metadata_available": metadata_path is not None,
        "artifact_api_metadata_invalid": metadata_invalid,
        "lineages": records,
    })
    (out / "README.md").write_text(
        "# DETECTIVE-005A failed-run raw custody\n\n"
        f"Source commit: `{head}`. GitHub Actions run: `{run_id}`.\n\n"
        "This is a **recovery archive, not a complete experiment**. "
        "Each available lineage artifact is copied byte-for-byte, including "
        "torn JSONL suffixes, error receipts and successful full lineages. "
        "Missing artifacts or invalid rows are identified without fabricating "
        "outputs or altering original bytes. Incomplete data never produces "
        "a fun-rating packet, model admission, training or cast expansion.\n\n"
        "The archive must be reviewed before retrying a failed lineage. "
        "The artifact_transport_complete flag describes only same-run API "
        "retrieval/download reconciliation, not semantic output validity. "
        "A false value is a BLOCKING custody gap even when this best-effort "
        "archive branch was pushed; recover the remaining original bytes "
        "before any retry. Entire-workflow cancellation may prevent this "
        "job from starting; inspect the original Actions artifacts.\n",
        encoding="utf-8")
    (out / "provenance" / "hashes.sha256").write_text(
        assemble.hash_inventory(out), encoding="utf-8")
    return manifest


def verify(path: Path) -> dict:
    # Git stores symlinks as links, not the bytes they reference. A digest
    # read through a link can match while the pushed dataset lacks its raw
    # evidence. Refuse all links before trusting any readback or SHA inventory.
    screen.require(path.is_dir() and not path.is_symlink(),
                   "recovery archive root must be a physical directory")
    screen.require(all(not entry.is_symlink() for entry in path.rglob("*")),
                   "recovery archive contains symlink; not self-contained custody")
    manifest = screen.read_json(path / "manifest.json")
    screen.require(manifest.get("schema") == "detective-005a-partial-recovery-v1" and
                   manifest.get("status") == "raw_recovery_only_not_a_complete_screen" and
                   manifest.get("review_packet_generated") is False and
                   manifest.get("optimizer_steps") == 0 and
                   manifest.get("model_judge_calls") == 0 and
                   manifest.get("production_admission") is False and
                   manifest.get("training_admission") is False and
                   manifest.get("population_expansion_admitted") is False,
                   "recovery archive must never claim admission")
    head = manifest["repository_head_sha"]
    run_id = manifest["actions_run_id"]
    ensure_identity(head, run_id)
    screen.require(manifest.get("source_config_sha256") ==
                   screen.sha256_file(screen.CONFIG) and
                   manifest.get("runner_script_sha256") ==
                   screen.sha256_file(Path(live.__file__)) and
                   manifest.get("recovery_script_sha256") ==
                   screen.sha256_file(Path(__file__)),
                   "recovery source hash mismatch")
    provenance = screen.read_json(path / "provenance" / "source-artifacts.json")
    screen.require(provenance.get("schema") ==
                   "detective-005a-recovery-source-artifacts-v1" and
                   provenance.get("repository_head_sha") == head and
                   provenance.get("actions_run_id") == run_id and
                   len(provenance.get("lineages", [])) == 12,
                   "recovery provenance drift")
    cfg = screen.load_contract()
    metadata_path = path / "provenance" / "actions-artifacts-api.json"
    screen.require(
        metadata_path.exists() == provenance.get("metadata_available"),
        "original artifact API availability drift")
    api, metadata_invalid = load_api_for_custody(
        metadata_path if metadata_path.exists() else None)
    screen.require(
        manifest.get("artifact_api_metadata_invalid") is metadata_invalid and
        provenance.get("artifact_api_metadata_invalid") is metadata_invalid,
        "recovery artifact API parse status drift")
    total = 0
    complete = 0
    with_artifacts = 0
    for lineage, saved in zip(assemble.LINEAGES, provenance["lineages"]):
        raw_dir = path / "raw" / lineage
        names = saved["files"]
        screen.require(saved["lineage"] == lineage and
                       saved["artifact_name"] == f"detective005a-{lineage}" and
                       [x["name"] for x in names] == sorted(x["name"] for x in names),
                       "lineage identity/order drift")
        actual = {p.name: p for p in raw_dir.iterdir()} if raw_dir.exists() else {}
        screen.require(set(actual) == {x["name"] for x in names},
                       "recovered raw file inventory drift")
        for record in names:
            target = actual[record["name"]]
            screen.require(target.is_file() and
                           target.stat().st_size == record["bytes"] and
                           screen.sha256_file(target) == record["sha256"],
                           "recovered original artifact bytes drift")
        artifact_name = f"detective005a-{lineage}"
        refreshed = bind_artifact_api(
            inspect_artifact(cfg, lineage, actual, head, run_id),
            artifact_name, api, metadata_path.exists(), head, run_id,
            metadata_invalid=metadata_invalid)
        with_artifacts += bool(actual)
        for field in ("lineage", "status", "validated_raw_prefix_rows",
                      "journal_tail_problem", "custody_warnings", "files",
                      "artifact_name", "artifact_api"):
            screen.require(refreshed[field] == saved[field],
                           "recovery lineage readback drift: " + field)
        total += refreshed["validated_raw_prefix_rows"]
        complete += refreshed["status"] == "complete_lineage_recovered"
    screen.require(total == manifest["validated_raw_prefix_rows"] and
                   complete == manifest["complete_lineage_artifacts"] and
                   with_artifacts == manifest["lineages_with_artifacts"],
                   "recovery aggregate row/count drift")
    screen.require(
        manifest.get("artifact_transport_complete") is
        artifact_transport_complete(
            provenance["lineages"], api, metadata_path.exists(),
            manifest.get("artifact_api_outcome"),
            manifest.get("artifact_download_outcome"), metadata_invalid),
        "recovery artifact transport gate drift")
    inventory = (path / "provenance" / "hashes.sha256").read_text(
        encoding="utf-8")
    screen.require(inventory == assemble.hash_inventory(path),
                   "recovery hash inventory mismatch")
    return {"validated_raw_prefix_rows": total,
            "complete_lineage_artifacts": complete,
            "artifact_transport_complete": manifest["artifact_transport_complete"],
            "review_packet_generated": False,
            "production_admission": False}


def require_complete_transport(path: Path) -> dict:
    """Fail CI after best-effort promotion if any original retrieval is unverified."""
    report = verify(path)
    screen.require(report["artifact_transport_complete"],
                   "incomplete original lineage artifact transport custody")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--source-head")
    ap.add_argument("--run-id")
    ap.add_argument("--artifact-metadata", type=Path)
    ap.add_argument("--artifact-api-outcome", default="unknown")
    ap.add_argument("--download-outcome", default="unknown")
    ap.add_argument("--verify", type=Path)
    ap.add_argument("--require-complete-transport", action="store_true")
    args = ap.parse_args()
    if args.verify:
        print(json.dumps(
            require_complete_transport(args.verify)
            if args.require_complete_transport else verify(args.verify),
            indent=2, sort_keys=True))
    else:
        screen.require(not args.require_complete_transport,
                       "complete transport gate requires --verify")
        screen.require(
            args.artifacts is not None and args.out is not None and
            args.source_head is not None and args.run_id is not None,
            "recovery requires artifacts, output, exact source and Actions run")
        print(json.dumps(recover(
            args.artifacts, args.out, args.source_head, args.run_id,
            args.artifact_metadata, args.artifact_api_outcome,
            args.download_outcome), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
