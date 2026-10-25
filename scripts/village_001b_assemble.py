#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from village_001_gate import latest_summary, validate_summary
from village_001_manifest import build_manifest
from village_001_validate import normalize_text, validate_file

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


def find_member_artifacts(root: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("manifest.json"):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if obj.get("experiment") == "VILLAGE-001B":
            member = obj.get("member_id")
            if isinstance(member, str):
                found[member].append(path.parent)
    duplicates = {k: v for k, v in found.items() if len(v) != 1}
    if duplicates:
        raise RuntimeError(f"member artifact multiplicity error: {duplicates}")
    return {member: paths[0] for member, paths in found.items()}


def hash_inventory(out: Path) -> str:
    rows = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out).as_posix()
        if rel == "provenance/hashes.sha256":
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    population = build_manifest(cfg)
    expected_members = [x["member_id"] for x in population["members"]]
    member_meta = {x["member_id"]: x for x in population["members"]}
    found = find_member_artifacts(args.artifacts)

    missing = sorted(set(expected_members) - set(found))
    extra = sorted(set(found) - set(expected_members))
    if missing or extra:
        raise RuntimeError(f"member artifacts mismatch missing={missing} extra={extra}")

    provider_root = ROOT / cfg["nvidia"]["preflight"]["archive_root"]
    provider_run_id, provider_path, provider_summary = latest_summary(provider_root)
    validate_summary(cfg, provider_summary)

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total_rows = 0
    all_ids: set[str] = set()
    member_stats = {}
    lineage_expansion_inputs: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_hashes: dict[str, dict] = {}
    total_critic_records = 0
    total_adjudicator_records = 0
    all_expansion_pairs = []

    for member_id in expected_members:
        src = found[member_id]
        manifest = json.loads((src / "manifest.json").read_text())
        meta = member_meta[member_id]

        if manifest.get("member_contract_sha256") != population["member_contract_sha256"]:
            raise RuntimeError(f"{member_id}: member contract hash mismatch")
        if manifest.get("lineage") != meta["lineage"]:
            raise RuntimeError(f"{member_id}: lineage mismatch")
        if manifest.get("sibling") != meta["sibling"]:
            raise RuntimeError(f"{member_id}: sibling mismatch")
        if manifest.get("specialization") != meta["specialization"]:
            raise RuntimeError(f"{member_id}: specialization mismatch")

        examples_path = src / "examples.jsonl"
        summary = validate_file(
            examples_path,
            "corpus",
            member_id,
            int(cfg["corpus"]["target_sft_examples_per_member"]),
        )
        rows = [
            json.loads(line)
            for line in examples_path.read_text().splitlines()
            if line.strip()
        ]

        expected_ids = [
            f"{member_id}-E{i:03d}"
            for i in range(1, int(cfg["corpus"]["target_sft_examples_per_member"]) + 1)
        ]
        ids = [row["example_id"] for row in rows]
        if ids != expected_ids:
            raise RuntimeError(f"{member_id}: noncanonical example ID sequence")

        kinds = Counter(row["source_kind"] for row in rows)
        if kinds != {
            "ancestral_copy": int(cfg["corpus"]["ancestral_examples_per_lineage"]),
            "cloud_expansion": int(cfg["corpus"]["new_sft_examples_per_member"]),
        }:
            raise RuntimeError(f"{member_id}: source counts {kinds}")

        expansion = [row for row in rows if row["source_kind"] == "cloud_expansion"]
        expected_surface = cfg["corpus"]["expansion_surface_counts"][meta["sibling"]]
        surface_counts = Counter(row["surface"] for row in expansion)
        if any(
            surface_counts.get(surface, 0) != expected_surface.get(surface, 0)
            for surface in expected_surface
        ):
            raise RuntimeError(
                f"{member_id}: expansion surface counts {dict(surface_counts)} "
                f"!= {expected_surface}"
            )

        critic_path = src / "quality" / "critic.json"
        adjudicator_path = src / "quality" / "adjudicator.json"
        if not critic_path.is_file() or not adjudicator_path.is_file():
            raise RuntimeError(f"{member_id}: missing curation records")
        critic = json.loads(critic_path.read_text())
        adjudicator = json.loads(adjudicator_path.read_text())
        reviewed_ids = {
            row.get("example_id") for row in critic if isinstance(row, dict)
        }
        expansion_ids = {row["example_id"] for row in expansion}
        if not expansion_ids.issubset(reviewed_ids):
            raise RuntimeError(
                f"{member_id}: {len(expansion_ids - reviewed_ids)} expansion rows "
                "were never independently reviewed"
            )

        critic_by_id: dict[str, list[dict]] = defaultdict(list)
        adjudicator_by_key: dict[tuple[str, int], dict] = {}
        for record in critic:
            if not isinstance(record, dict):
                raise RuntimeError(f"{member_id}: malformed critic record")
            example_id = record.get("example_id")
            round_index = record.get("round")
            if example_id not in expansion_ids or not isinstance(round_index, int):
                raise RuntimeError(f"{member_id}: invalid critic identity/round")
            critic_by_id[example_id].append(record)

        for decision in adjudicator:
            if not isinstance(decision, dict):
                raise RuntimeError(f"{member_id}: malformed adjudicator record")
            example_id = decision.get("example_id")
            round_index = decision.get("round")
            key = (example_id, round_index)
            if (
                example_id not in expansion_ids
                or not isinstance(round_index, int)
                or key in adjudicator_by_key
            ):
                raise RuntimeError(f"{member_id}: invalid/duplicate adjudicator record")
            adjudicator_by_key[key] = decision

        terminal_curation = {}
        for example_id in sorted(expansion_ids):
            history = critic_by_id.get(example_id, [])
            if not history:
                raise RuntimeError(f"{member_id}: {example_id} has no critic history")
            latest_round = max(record["round"] for record in history)
            latest = [record for record in history if record["round"] == latest_round]
            if len(latest) != 1:
                raise RuntimeError(
                    f"{member_id}: {example_id} has ambiguous critic round {latest_round}"
                )
            final_review = latest[0]
            verdict = final_review.get("derived_verdict")
            if verdict == "pass":
                terminal_curation[example_id] = {
                    "round": latest_round,
                    "verdict": "critic_pass",
                }
            elif verdict == "borderline":
                decision = adjudicator_by_key.get((example_id, latest_round))
                if not decision or decision.get("decision") != "accept":
                    raise RuntimeError(
                        f"{member_id}: {example_id} borderline without terminal "
                        "adjudicator acceptance"
                    )
                terminal_curation[example_id] = {
                    "round": latest_round,
                    "verdict": "adjudicator_accept",
                }
            else:
                raise RuntimeError(
                    f"{member_id}: {example_id} terminal critic verdict {verdict!r}"
                )

        for row in expansion:
            lineage_expansion_inputs[meta["lineage"]][
                normalize_text(row["input"])
            ].append(member_id)
            all_expansion_pairs.append(
                (
                    normalize_text(row["input"]),
                    normalize_text(row["target"]),
                )
            )
            if not row["provenance"].get("provider_call_ids"):
                raise RuntimeError(
                    f"{member_id}: {row['example_id']} has no provider call lineage"
                )

        dest = out / "member-corpora" / f"{member_id}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(examples_path, dest)

        provider_src = src / "provider"
        if not provider_src.is_dir():
            raise RuntimeError(f"{member_id}: missing raw provider evidence")
        shutil.copytree(provider_src, out / "provider" / member_id)

        shutil.copytree(
            src / "quality",
            out / "quality" / "members" / member_id,
        )
        manifest_dest = out / "provenance" / "member-manifests" / f"{member_id}.json"
        manifest_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "manifest.json", manifest_dest)

        total_rows += len(rows)
        for example_id in ids:
            if example_id in all_ids:
                raise RuntimeError(f"global duplicate example ID {example_id}")
            all_ids.add(example_id)

        total_critic_records += len(critic)
        total_adjudicator_records += len(adjudicator)
        member_stats[member_id] = {
            "rows": len(rows),
            "ancestral_rows": kinds["ancestral_copy"],
            "expansion_rows": kinds["cloud_expansion"],
            "expansion_surface_counts": dict(surface_counts),
            "critic_records": len(critic),
            "adjudicator_records": len(adjudicator),
            "terminal_critic_pass": sum(
                1 for state in terminal_curation.values()
                if state["verdict"] == "critic_pass"
            ),
            "terminal_adjudicator_accept": sum(
                1 for state in terminal_curation.values()
                if state["verdict"] == "adjudicator_accept"
            ),
            "training_sha256": sha256_file(examples_path),
            "validator_summary": summary,
        }

        lineage = meta["lineage"]
        if lineage not in source_hashes:
            hashes = manifest["source_hashes"]
            source_hashes[lineage] = {
                **hashes,
                "packet_id": manifest["source_packet_id"],
                "source_babble_ids": manifest["source_babble_ids"],
            }
        elif source_hashes[lineage] != {
            **manifest["source_hashes"],
            "packet_id": manifest["source_packet_id"],
            "source_babble_ids": manifest["source_babble_ids"],
        }:
            raise RuntimeError(f"{member_id}: sibling ancestry hash disagreement")

    expected_total = int(cfg["population"]["expected_members"]) * int(
        cfg["corpus"]["target_sft_examples_per_member"]
    )
    if total_rows != expected_total:
        raise RuntimeError(f"global row count {total_rows} != {expected_total}")
    if len(all_ids) != expected_total:
        raise RuntimeError("global example IDs are not unique")

    sibling_overlap = {}
    for lineage, inputs in lineage_expansion_inputs.items():
        duplicates = {
            text: members for text, members in inputs.items() if len(set(members)) > 1
        }
        sibling_overlap[lineage] = {
            "duplicate_normalized_inputs_across_siblings": len(duplicates),
            "examples": [
                {"input": text, "members": sorted(set(members))}
                for text, members in list(sorted(duplicates.items()))[:25]
            ],
        }

    expansion_pair_duplicates = len(all_expansion_pairs) - len(
        set(all_expansion_pairs)
    )

    write_json(out / "ancestors" / "source-hashes.json", source_hashes)

    quality = {
        "schema_version": 1,
        "experiment": "VILLAGE-001B",
        "members": len(expected_members),
        "total_rows": total_rows,
        "ancestral_copies": int(cfg["corpus"]["ancestral_examples_per_lineage"])
        * len(expected_members),
        "cloud_expansion_rows": int(cfg["corpus"]["new_sft_examples_per_member"])
        * len(expected_members),
        "unique_example_ids": len(all_ids),
        "critic_review_records": total_critic_records,
        "adjudicator_records": total_adjudicator_records,
        "exact_expansion_input_target_duplicates": expansion_pair_duplicates,
        "sibling_input_overlap": sibling_overlap,
        "member_stats": member_stats,
    }
    write_json(out / "quality" / "validation.json", quality)

    source_run_id = os.environ.get("GITHUB_RUN_ID")
    source_sha = os.environ.get("GITHUB_SHA")
    provenance = {
        "schema_version": 1,
        "experiment": "VILLAGE-001B",
        "issue": 11,
        "workflow_run_id": int(source_run_id) if source_run_id else None,
        "workflow_run_url": (
            f"https://github.com/techrote/neurounner/actions/runs/{source_run_id}"
            if source_run_id
            else None
        ),
        "source_commit": source_sha,
        "member_contract_sha256": population["member_contract_sha256"],
        "provider_gate": {
            "run_id": provider_run_id,
            "summary_path": provider_path.relative_to(ROOT).as_posix(),
            "source_commit": provider_summary.get("github_sha"),
        },
        "source_datasets": ["datasets/babble-001", "datasets/babble-002"],
        "config_sha256": sha256_file(args.config),
        "member_schema_sha256": sha256_file(
            ROOT / cfg["schemas"]["member_corpus_row"]
        ),
        "provider": "nvidia",
        "models": {
            role: cfg["nvidia"]["models"][role]["id"]
            for role in ("compiler", "critic", "adjudicator")
        },
        "remote_unpinned_backend": True,
    }
    write_json(out / "provenance" / "source-artifacts.json", provenance)

    manifest = {
        "schema_version": 1,
        "dataset_id": "village-001",
        "stage": "VILLAGE-001B-member-corpora",
        "issue": 11,
        "member_contract_sha256": population["member_contract_sha256"],
        "counts": {
            "members": len(expected_members),
            "rows": total_rows,
            "rows_per_member": int(cfg["corpus"]["target_sft_examples_per_member"]),
            "ancestral_copies": quality["ancestral_copies"],
            "cloud_expansion_rows": quality["cloud_expansion_rows"],
        },
        "member_shards": [
            f"member-corpora/{member_id}.jsonl" for member_id in expected_members
        ],
        "quality_file": "quality/validation.json",
        "provenance_file": "provenance/source-artifacts.json",
        "ancestor_hashes_file": "ancestors/source-hashes.json",
    }
    write_json(out / "manifest.json", manifest)

    readme = f"""# VILLAGE-001 — durable campaign archive

Current promoted stage: **VILLAGE-001B member-specific SFT corpora**.

## Population

- 36 immutable members: G01–G12 × A/B/C
- {cfg['corpus']['target_sft_examples_per_member']} rows/member
- {quality['ancestral_copies']} lineage-preserving BABBLE-002 ancestral copies
- {quality['cloud_expansion_rows']} new member-specific cloud-compiled rows
- every new row independently reviewed by Nemotron Bolt before publication
- borderline rows adjudicated selectively by Nemotron Super
- raw compiler/critic/adjudicator requests and responses retained under provider/

## Authority

The canonical population contract is `experiments/village-001/config.json` plus
`scripts/village_001_manifest.py`. BABBLE-001 remains authority for raw Sprocket
babble; BABBLE-002 remains authority for the 1,200 ancestral positive/negative rows.

This directory is durable evidence. GitHub Actions artifacts are staging only.

See:

- `manifest.json`
- `quality/validation.json`
- `provenance/source-artifacts.json`
- `provenance/hashes.sha256`
- `ancestors/source-hashes.json`
"""
    (out / "README.md").write_text(readme)
    (out / "provenance" / "hashes.sha256").write_text(hash_inventory(out))

    print(
        json.dumps(
            {
                "dataset": "village-001",
                "stage": "VILLAGE-001B",
                "members": len(expected_members),
                "rows": total_rows,
                "critic_records": total_critic_records,
                "adjudicator_records": total_adjudicator_records,
                "expansion_pair_duplicates": expansion_pair_duplicates,
                "files": sum(1 for path in out.rglob("*") if path.is_file()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
