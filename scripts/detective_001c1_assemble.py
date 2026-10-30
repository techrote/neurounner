#!/usr/bin/env python3
"""Assemble DETECTIVE-001C1 training/evaluation evidence and neutral adapter release."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from detective_001c1_dataset import validate_existing
from detective_001c1_train import directory_hash, sha_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-001/training-c1.json"


class AssembleError(ValueError):
    pass


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def deterministic_tar_gz(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w",
                              format=tarfile.PAX_FORMAT) as tf:
                for path in sorted(source.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(source).as_posix()
                    info = tf.gettarinfo(str(path), arcname=rel)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as fh:
                        tf.addfile(info, fh)


def hash_inventory(root: Path) -> str:
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix()
        if rel != "hashes.sha256":
            rows.append(f"{sha_file(p)}  {rel}")
    return "\n".join(rows) + "\n"


def result_path(artifacts: Path, member: str) -> Path:
    matches = list(artifacts.rglob(f"{member}/result.json"))
    if len(matches) != 1:
        raise AssembleError(
            f"{member}: expected one result.json, found {len(matches)}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--evidence-out", type=Path, required=True)
    ap.add_argument("--release-out", type=Path, required=True)
    ap.add_argument("--source-head", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-url", required=True)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    dataset_manifest = validate_existing(args.dataset_root)
    evaluation_summary = json.loads(
        (args.evaluation / "summary.json").read_text())
    final_suite = json.loads(
        (args.evaluation / "final-suite.json").read_text())

    if args.evidence_out.exists():
        shutil.rmtree(args.evidence_out)
    if args.release_out.exists():
        shutil.rmtree(args.release_out)
    args.evidence_out.mkdir(parents=True)
    args.release_out.mkdir(parents=True)

    training_rows = {}
    assets = []
    for member in ("V-G02-A", "V-G09-C"):
        src_result = result_path(args.artifacts, member)
        src_dir = src_result.parent
        adapter = src_dir / "adapter"
        if not adapter.is_dir():
            raise AssembleError(member + ": trained adapter missing")
        result = json.loads(src_result.read_text())
        expected = cfg["members"][member]
        if (result.get("experiment") != "DETECTIVE-001C1"
            or result.get("member_id") != member
            or result.get("parent_directory_sha256")
                != expected["parent_directory_sha256"]
            or result.get("training", {}).get("optimizer_steps") != 64):
            raise AssembleError(member + ": training result source drift")
        actual_dir_hash = directory_hash(adapter)
        if result["adapter"]["directory_sha256"] != actual_dir_hash:
            raise AssembleError(member + ": trained adapter hash mismatch")
        if (final_suite["trained_directory_sha256"].get(member)
            != actual_dir_hash):
            raise AssembleError(member + ": final suite not bound to trained hash")
        dest = args.evidence_out / "training" / f"{member}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_result, dest)

        archive_name = (
            f"detective-001c1-{member}-smollm2-135m-lora.tar.gz")
        archive = args.release_out / archive_name
        deterministic_tar_gz(adapter, archive)
        assets.append({
            "member_id": member,
            "policy_id": expected["policy_id"],
            "file": archive_name,
            "sha256": sha_file(archive),
            "bytes": archive.stat().st_size,
            "adapter_directory_sha256": actual_dir_hash,
            "parent_archive_sha256": expected["parent_archive_sha256"],
            "parent_directory_sha256": expected["parent_directory_sha256"],
        })
        training_rows[member] = {
            "first_loss": result["training"]["first_loss"],
            "last_loss": result["training"]["last_loss"],
            "mean_loss": result["training"]["mean_loss"],
            "seconds": result["training"]["seconds"],
            "sequence_tokens_per_second":
                result["training"]["sequence_tokens_per_second"],
            "peak_rss_kib": result["runtime"]["peak_rss_kib"],
            "trained_directory_sha256": actual_dir_hash,
        }

    # Copy the exact post-training generated evaluation evidence.
    for name in ("final-suite.json", "raw-results.json", "summary.json",
                 "REPORT.md"):
        shutil.copy2(args.evaluation / name, args.evidence_out / name)

    release = {
        "schema": "detective-witness-release-v1",
        "experiment": "DETECTIVE-001C1",
        "issue": 111,
        "release_tag": cfg["release"]["tag"],
        "base_model": cfg["base_model"]["id"],
        "base_revision": cfg["base_model"]["revision"],
        "dataset_manifest_sha256": sha_file(
            args.dataset_root / "manifest.json"),
        "final_suite_sha256": sha_file(args.evaluation / "final-suite.json"),
        "assets": assets,
    }
    write_json(args.evidence_out / "adapter-release.json", release)

    summary = {
        "schema": "detective-witness-c1-summary-v1",
        "experiment": "DETECTIVE-001C1",
        "issue": 111,
        "source_head": args.source_head,
        "workflow_run_id": args.run_id,
        "workflow_run_url": args.run_url,
        "dataset_manifest_sha256": sha_file(
            args.dataset_root / "manifest.json"),
        "dataset_counts": dataset_manifest["counts"],
        "training": training_rows,
        "evaluation": evaluation_summary,
        "release_tag": cfg["release"]["tag"],
        "larger_cast_expansion_admitted":
            evaluation_summary["larger_cast_expansion_admitted"],
    }
    write_json(args.evidence_out / "campaign-summary.json", summary)
    write_json(args.evidence_out / "provenance.json", {
        "source_head": args.source_head,
        "workflow_run_id": args.run_id,
        "workflow_run_url": args.run_url,
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha_file(CONFIG),
        "dataset_manifest_sha256": sha_file(
            args.dataset_root / "manifest.json"),
        "final_suite_sha256": sha_file(
            args.evaluation / "final-suite.json"),
        "release_assets": assets,
        "external_provider_calls": 0,
        "training_members": ["V-G02-A", "V-G09-C"],
    })
    (args.evidence_out / "hashes.sha256").write_text(
        hash_inventory(args.evidence_out), encoding="utf-8")

    print(json.dumps({
        "training_members": list(training_rows),
        "release_assets": len(assets),
        "relation_policy_gate":
            evaluation_summary["relation_policy_gate"],
        "larger_cast_expansion_admitted":
            evaluation_summary["larger_cast_expansion_admitted"],
        "evidence_hashes": len(
            (args.evidence_out / "hashes.sha256").read_text().splitlines()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
