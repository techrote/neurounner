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


def find_artifacts(root: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = defaultdict(list)
    for result_path in root.rglob("result.json"):
        try:
            obj = json.loads(result_path.read_text())
        except Exception:
            continue
        if obj.get("experiment") != "VILLAGE-001C":
            continue
        member = obj.get("member_id")
        if isinstance(member, str):
            found[member].append(result_path.parent)
    duplicates = {member: paths for member, paths in found.items() if len(paths) != 1}
    if duplicates:
        raise RuntimeError(f"training artifact multiplicity error: {duplicates}")
    return {member: paths[0] for member, paths in found.items()}


def deterministic_tar_gz(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/config.json",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=ROOT / "experiments/village-001/sft-evidence",
    )
    parser.add_argument("--release-out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    population = build_manifest(cfg)
    member_meta = {row["member_id"]: row for row in population["members"]}
    expected_members = list(member_meta)

    dataset_root = ROOT / cfg["archive"]["dataset_root"]
    dataset_manifest_path = dataset_root / "manifest.json"
    if not dataset_manifest_path.is_file():
        raise RuntimeError("durable VILLAGE-001B dataset manifest is missing")
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    if dataset_manifest.get("stage") != "VILLAGE-001B-member-corpora":
        raise RuntimeError("VILLAGE-001B dataset has not been promoted")
    if dataset_manifest.get("member_contract_sha256") != population[
        "member_contract_sha256"
    ]:
        raise RuntimeError("dataset/member population contract mismatch")

    found = find_artifacts(args.artifacts)
    missing = sorted(set(expected_members) - set(found))
    extra = sorted(set(found) - set(expected_members))
    if missing or extra:
        raise RuntimeError(
            f"training artifacts mismatch missing={missing} extra={extra}"
        )

    evidence = args.evidence_out
    release = args.release_out
    if evidence.exists():
        shutil.rmtree(evidence)
    if release.exists():
        shutil.rmtree(release)
    evidence.mkdir(parents=True)
    release.mkdir(parents=True)

    rows = []
    assets = []
    cpu_counts: Counter[str] = Counter()
    total_seconds = []
    seq_rates = []
    sup_rates = []
    peak_rss = []
    adapter_bytes = []
    first_losses = []
    last_losses = []
    resumed_members = []

    for member_id in expected_members:
        src = found[member_id]
        result_path = src / "result.json"
        adapter_dir = src / "adapter"
        if not adapter_dir.is_dir():
            raise RuntimeError(f"{member_id}: adapter directory is missing")

        result = json.loads(result_path.read_text())
        meta = member_meta[member_id]
        checks = {
            "member_id": member_id,
            "lineage": meta["lineage"],
            "sibling": meta["sibling"],
            "specialization": meta["specialization"],
            "member_seed": meta["seed"],
            "member_contract_sha256": population["member_contract_sha256"],
            "base_model": cfg["base_model"]["id"],
            "base_revision": cfg["base_model"]["revision"],
        }
        for key, expected in checks.items():
            if result.get(key) != expected:
                raise RuntimeError(
                    f"{member_id}: {key} {result.get(key)!r} != {expected!r}"
                )
        if result.get("training_config") != cfg["sft_training"]:
            raise RuntimeError(f"{member_id}: training config drift")

        corpus_path = dataset_root / "member-corpora" / f"{member_id}.jsonl"
        if result.get("training_dataset_sha256") != sha256_file(corpus_path):
            raise RuntimeError(f"{member_id}: training corpus hash mismatch")
        example_ids = result.get("training_example_ids")
        if not isinstance(example_ids, list) or len(example_ids) != int(
            cfg["corpus"]["target_sft_examples_per_member"]
        ):
            raise RuntimeError(f"{member_id}: training example IDs incomplete")
        if len(example_ids) != len(set(example_ids)):
            raise RuntimeError(f"{member_id}: duplicate training example IDs")

        training = result.get("training") or {}
        expected_steps = (
            int(cfg["corpus"]["target_sft_examples_per_member"])
            * int(cfg["sft_training"]["epochs"])
        )
        if training.get("optimizer_steps") != expected_steps:
            raise RuntimeError(
                f"{member_id}: optimizer steps {training.get('optimizer_steps')} "
                f"!= {expected_steps}"
            )
        losses = training.get("losses")
        if not isinstance(losses, list) or len(losses) != expected_steps:
            raise RuntimeError(f"{member_id}: loss trace incomplete")
        if result.get("training_source_kind_counts") != {
            "ancestral_copy": int(cfg["corpus"]["ancestral_examples_per_lineage"]),
            "cloud_expansion": int(cfg["corpus"]["new_sft_examples_per_member"]),
        }:
            raise RuntimeError(f"{member_id}: source-kind training counts drift")

        evaluation = result.get("evaluation") or {}
        responses = evaluation.get("responses")
        if evaluation.get("hidden_member_prompt_used") is not False:
            raise RuntimeError(f"{member_id}: hidden member prompt flag is not false")
        if not isinstance(responses, list) or len(responses) != 24:
            raise RuntimeError(f"{member_id}: held-out response count != 24")
        scenario_ids = [row.get("scenario_id") for row in responses]
        if scenario_ids != [f"T{i:02d}" for i in range(1, 25)]:
            raise RuntimeError(f"{member_id}: held-out scenario sequence mismatch")

        probe_eval = result.get("member_probe_evaluation") or {}
        probes = probe_eval.get("responses")
        if probe_eval.get("held_out_from_member_training") is not True:
            raise RuntimeError(f"{member_id}: member probes are not marked held-out")
        if probe_eval.get("hidden_member_prompt_used") is not False:
            raise RuntimeError(f"{member_id}: hidden member prompt used for probes")
        if not isinstance(probes, list) or len(probes) != 8:
            raise RuntimeError(f"{member_id}: member probe response count != 8")
        if [row.get("scenario_id") for row in probes] != [
            f"P{i:02d}" for i in range(1, 9)
        ]:
            raise RuntimeError(f"{member_id}: member probe ID sequence mismatch")
        if len({row.get("prompt") for row in probes}) != 8:
            raise RuntimeError(f"{member_id}: duplicate member probe prompt")
        trained_ids = set(example_ids)
        for probe in probes:
            source_member = probe.get("source_member_id")
            source_example = probe.get("source_example_id")
            if (
                not isinstance(source_member, str)
                or source_member == member_id
                or not source_member.startswith(f"V-{meta['lineage']}-")
            ):
                raise RuntimeError(
                    f"{member_id}: invalid probe source member {source_member!r}"
                )
            if source_example in trained_ids:
                raise RuntimeError(
                    f"{member_id}: probe source {source_example} was in target training IDs"
                )

        actual_adapter_hash = directory_hash(adapter_dir)
        adapter = result.get("adapter") or {}
        if adapter.get("sha256") != actual_adapter_hash:
            raise RuntimeError(f"{member_id}: adapter directory hash mismatch")
        actual_adapter_bytes = sum(
            path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()
        )
        if adapter.get("bytes") != actual_adapter_bytes:
            raise RuntimeError(f"{member_id}: adapter byte count mismatch")

        result_dest = evidence / "results" / f"{member_id}.json"
        result_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, result_dest)

        archive_name = f"village-001-sft-{member_id}-smollm2-135m-lora.tar.gz"
        archive = release / archive_name
        deterministic_tar_gz(adapter_dir, archive)
        asset = {
            "member_id": member_id,
            "file": archive_name,
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
            "adapter_directory_sha256": actual_adapter_hash,
            "adapter_directory_bytes": actual_adapter_bytes,
        }
        assets.append(asset)

        cpu = result["runner"]["cpu_model"]
        cpu_counts[cpu] += 1
        total_seconds.append(float(training["seconds"]))
        seq_rates.append(float(training["sequence_tokens_per_second"]))
        sup_rates.append(float(training["supervised_tokens_per_second"]))
        peak_rss.append(int(result["peak_rss_kib"]))
        adapter_bytes.append(actual_adapter_bytes)
        first_losses.append(float(training["first_loss"]))
        last_losses.append(float(training["last_loss"]))
        if (result.get("recovery") or {}).get("resumed"):
            resumed_members.append(member_id)

        rows.append(
            {
                "member_id": member_id,
                "lineage": meta["lineage"],
                "sibling": meta["sibling"],
                "cpu_model": cpu,
                "training_seconds": training["seconds"],
                "sequence_tokens_per_second": training[
                    "sequence_tokens_per_second"
                ],
                "supervised_tokens_per_second": training[
                    "supervised_tokens_per_second"
                ],
                "first_loss": training["first_loss"],
                "last_loss": training["last_loss"],
                "peak_rss_kib": result["peak_rss_kib"],
                "adapter_directory_sha256": actual_adapter_hash,
                "adapter_archive_sha256": asset["sha256"],
                "adapter_archive_bytes": asset["bytes"],
                "resumed": (result.get("recovery") or {}).get("resumed", False),
            }
        )

    if len(rows) != 36 or len(assets) != 36:
        raise RuntimeError("population training evidence cardinality mismatch")

    release_manifest = {
        "schema_version": 1,
        "experiment": "VILLAGE-001C",
        "issue": 12,
        "release_tag": "village-001-sft-v1",
        "release_url": (
            "https://github.com/techrote/neurounner/releases/tag/"
            "village-001-sft-v1"
        ),
        "base_model": cfg["base_model"]["id"],
        "base_revision": cfg["base_model"]["revision"],
        "member_contract_sha256": population["member_contract_sha256"],
        "assets": assets,
    }
    write_json(evidence / "adapter-release.json", release_manifest)

    summary = {
        "schema_version": 1,
        "experiment": "VILLAGE-001C",
        "issue": 12,
        "members": 36,
        "successful_terminal_lanes": 36,
        "base_model": cfg["base_model"]["id"],
        "base_revision": cfg["base_model"]["revision"],
        "member_contract_sha256": population["member_contract_sha256"],
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "training_config": cfg["sft_training"],
        "cpu_models": dict(cpu_counts),
        "resumed_members": resumed_members,
        "metrics": {
            "training_seconds": {
                "min": min(total_seconds),
                "median": statistics.median(total_seconds),
                "max": max(total_seconds),
                "sum": sum(total_seconds),
            },
            "sequence_tokens_per_second": {
                "min": min(seq_rates),
                "median": statistics.median(seq_rates),
                "max": max(seq_rates),
            },
            "supervised_tokens_per_second": {
                "min": min(sup_rates),
                "median": statistics.median(sup_rates),
                "max": max(sup_rates),
            },
            "peak_rss_kib": {
                "min": min(peak_rss),
                "median": statistics.median(peak_rss),
                "max": max(peak_rss),
            },
            "adapter_directory_bytes": {
                "min": min(adapter_bytes),
                "median": statistics.median(adapter_bytes),
                "max": max(adapter_bytes),
                "sum": sum(adapter_bytes),
            },
            "first_loss": {
                "min": min(first_losses),
                "median": statistics.median(first_losses),
                "max": max(first_losses),
            },
            "last_loss": {
                "min": min(last_losses),
                "median": statistics.median(last_losses),
                "max": max(last_losses),
            },
        },
        "members_table": rows,
    }
    write_json(evidence / "summary.json", summary)

    run_id = os.environ.get("GITHUB_RUN_ID")
    provenance = {
        "schema_version": 1,
        "experiment": "VILLAGE-001C",
        "issue": 12,
        "workflow_run_id": int(run_id) if run_id else None,
        "workflow_run_url": (
            f"https://github.com/techrote/neurounner/actions/runs/{run_id}"
            if run_id
            else None
        ),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "config_sha256": sha256_file(args.config),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "member_contract_sha256": population["member_contract_sha256"],
    }
    write_json(evidence / "provenance.json", provenance)

    report_lines = [
        "# VILLAGE-001C — 36-member SFT training report",
        "",
        "All 36 canonical VILLAGE-001 members completed the same qualified "
        "SmolLM2-135M LoRA recipe on independent GitHub-hosted CPU lanes.",
        "",
        "No hidden member/persona description was supplied during held-out evaluation. "
        "Member differences arise from the member corpus, deterministic seed, and learned adapter.",
        "",
        "## Aggregate",
        "",
        f"- members: 36/36 terminal successful lanes",
        f"- base revision: `{cfg['base_model']['revision']}`",
        f"- optimizer steps/member: {int(cfg['corpus']['target_sft_examples_per_member']) * int(cfg['sft_training']['epochs'])}",
        f"- median training seconds: {statistics.median(total_seconds):.2f}",
        f"- median sequence tok/s: {statistics.median(seq_rates):.2f}",
        f"- median peak RSS KiB: {statistics.median(peak_rss):.0f}",
        f"- resumed members: {len(resumed_members)}",
        "",
        "## Members",
        "",
        "| Member | CPU | Seconds | Seq tok/s | Loss first→last | RSS KiB | Adapter archive |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    asset_by_member = {row["member_id"]: row for row in assets}
    for row in rows:
        asset = asset_by_member[row["member_id"]]
        report_lines.append(
            f"| {row['member_id']} | {row['cpu_model']} | "
            f"{float(row['training_seconds']):.2f} | "
            f"{float(row['sequence_tokens_per_second']):.2f} | "
            f"{float(row['first_loss']):.4f}→{float(row['last_loss']):.4f} | "
            f"{int(row['peak_rss_kib'])} | "
            f"`{asset['sha256'][:16]}…` |"
        )
    report_lines.extend(
        [
            "",
            "Raw held-out responses and full loss traces are retained in "
            "`results/*.json`. Adapter bytes are published as release assets; "
            "their archive and unpacked-directory hashes are recorded in "
            "`adapter-release.json`.",
            "",
        ]
    )
    (evidence / "REPORT.md").write_text("\n".join(report_lines))

    (evidence / "hashes.sha256").write_text(hash_inventory(evidence))

    print(
        json.dumps(
            {
                "experiment": "VILLAGE-001C",
                "members": 36,
                "release_assets": len(assets),
                "evidence_files": sum(
                    1 for path in evidence.rglob("*") if path.is_file()
                ),
                "release_bytes": sum(asset["bytes"] for asset in assets),
                "resumed_members": resumed_members,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
