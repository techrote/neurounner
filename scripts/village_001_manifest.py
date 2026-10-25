#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def derive_seed(campaign_seed: int, lineage: str, sibling: str) -> int:
    raw = f"{campaign_seed}:{lineage}:{sibling}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")
    return value or 1


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build_manifest(config: dict) -> dict:
    population = config["population"]
    lineages = population["lineages"]
    siblings = population["siblings"]

    if len(lineages) != len(set(lineages)):
        raise ValueError("duplicate lineage IDs")
    if len(siblings) != len(set(siblings)):
        raise ValueError("duplicate sibling IDs")
    if set(siblings) != {"A", "B", "C"}:
        raise ValueError("VILLAGE-001 requires sibling codes A/B/C")

    expected = int(population["expected_members"])
    actual = len(lineages) * len(siblings)
    if actual != expected:
        raise ValueError(f"population cardinality mismatch: {actual} != {expected}")

    members = []
    for lineage in lineages:
        for sibling in sorted(siblings):
            sibling_cfg = siblings[sibling]
            member_id = population["id_template"].format(
                lineage=lineage,
                sibling=sibling,
            )
            members.append(
                {
                    "member_id": member_id,
                    "lineage": lineage,
                    "sibling": sibling,
                    "specialization": sibling_cfg["name"],
                    "surface_weights": sibling_cfg["surface_weights"],
                    "seed": derive_seed(
                        int(population["campaign_seed"]),
                        lineage,
                        sibling,
                    ),
                    "sources": {
                        "ancestral_training": f"datasets/babble-002/training/{lineage}.jsonl",
                        "packet": f"datasets/babble-001/compiled/packets/{lineage}.json",
                        "raw_babble": f"datasets/babble-001/conversations/{lineage}.json",
                    },
                    "targets": {
                        "member_corpus": f"datasets/village-001/member-corpora/{member_id}.jsonl",
                        "sprocket_feedback": f"datasets/village-001/sprocket-feedback/{member_id}.jsonl",
                        "preference_corpus": f"datasets/village-001/preference/{member_id}.jsonl",
                        "sft_result": f"results/village001/sft/{member_id}.json",
                        "preference_result": f"results/village001/preference/{member_id}.json",
                    },
                }
            )

    canonical_members = json.dumps(
        members,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return {
        "schema_version": 1,
        "experiment": "VILLAGE-001",
        "campaign_seed": population["campaign_seed"],
        "expected_members": expected,
        "member_count": len(members),
        "member_contract_sha256": hashlib.sha256(canonical_members).hexdigest(),
        "members": members,
    }


def validate_sources(manifest: dict) -> None:
    errors = []
    for member in manifest["members"]:
        for role, rel in member["sources"].items():
            path = ROOT / rel
            if not path.is_file():
                errors.append(f"{member['member_id']} missing {role}: {rel}")

        train_path = ROOT / member["sources"]["ancestral_training"]
        if train_path.is_file():
            rows = [
                json.loads(line)
                for line in train_path.read_text().splitlines()
                if line.strip()
            ]
            if len(rows) != 100:
                errors.append(
                    f"{member['member_id']} ancestral row count {len(rows)} != 100"
                )
            wrong = [
                row.get("example_id", "<missing>")
                for row in rows
                if row.get("source_guide_id") != member["lineage"]
            ]
            if wrong:
                errors.append(
                    f"{member['member_id']} has {len(wrong)} lineage-mismatched rows"
                )

    if errors:
        raise RuntimeError("\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/config.json",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    manifest = build_manifest(config)

    if args.check_sources:
        validate_sources(manifest)

    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")

    print(
        f"VILLAGE-001 manifest: {manifest['member_count']} members, "
        f"contract {manifest['member_contract_sha256']}"
    )


if __name__ == "__main__":
    main()
