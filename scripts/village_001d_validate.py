#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

MEMBER_RE = re.compile(r"^V-(G(?:0[1-9]|1[0-2]))-([ABC])$")
PAIR_RE = re.compile(
    r"^P-(V-G(?:0[1-9]|1[0-2])-[ABC])-"
    r"((?:T(?:0[1-9]|1[0-9]|2[0-4]))|(?:P0[1-8]))-V([123])$"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def fail(message: str) -> None:
    raise ValueError(message)


def validate_pair(row: dict) -> None:
    if not isinstance(row, dict):
        fail("row is not an object")
    pair_id = row.get("pair_id")
    if not isinstance(pair_id, str):
        fail("pair_id missing")
    match = PAIR_RE.fullmatch(pair_id)
    if not match:
        fail(f"invalid pair_id {pair_id!r}")
    pair_member, pair_scenario, pair_variant = match.groups()

    member = row.get("member_id")
    mm = MEMBER_RE.fullmatch(member or "")
    if not mm:
        fail(f"invalid member_id {member!r}")
    lineage, sibling = mm.groups()
    if member != pair_member:
        fail("pair_id/member_id mismatch")
    if row.get("lineage") != lineage:
        fail("member/lineage mismatch")
    if row.get("sibling") != sibling:
        fail("member/sibling mismatch")
    if row.get("scenario_id") != pair_scenario:
        fail("pair_id/scenario mismatch")
    if row.get("variant") != int(pair_variant):
        fail("pair_id/variant mismatch")

    for key, minimum in (
        ("input", 8),
        ("chosen", 12),
        ("rejected", 8),
        ("source_parent_response", 0),
        ("why_chosen", 8),
    ):
        value = row.get(key)
        if not isinstance(value, str) or len(value.strip()) < minimum:
            fail(f"{key}: invalid string")
    if normalize(row["chosen"]) == normalize(row["rejected"]):
        fail("chosen/rejected are identical after normalization")

    refs = row.get("source_feedback_ids")
    if (
        not isinstance(refs, list)
        or len(refs) != 3
        or len(set(refs)) != 3
        or any(not isinstance(x, str) or not x for x in refs)
    ):
        fail("source_feedback_ids must contain exactly three unique IDs")

    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        fail("provenance missing")
    for key in (
        "member_contract_sha256",
        "parent_adapter_sha256",
        "feedback_dataset_sha256",
    ):
        value = provenance.get(key)
        if not isinstance(value, str) or not SHA_RE.fullmatch(value):
            fail(f"provenance.{key}: invalid SHA-256")
    calls = provenance.get("provider_call_ids")
    if (
        not isinstance(calls, list)
        or not calls
        or len(calls) != len(set(calls))
        or any(not isinstance(x, str) or not x for x in calls)
    ):
        fail("provenance.provider_call_ids invalid")


def validate_file(path: Path, member: str, expect_count: int = 60) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    errors = []
    ids = set()
    chosen_pairs = set()
    scenario_counts = Counter()
    for index, row in enumerate(rows, 1):
        try:
            validate_pair(row)
            if row["member_id"] != member:
                fail(f"member {row['member_id']} != expected {member}")
            if row["pair_id"] in ids:
                fail("duplicate pair ID")
            ids.add(row["pair_id"])
            key = (normalize(row["chosen"]), normalize(row["rejected"]))
            if key in chosen_pairs:
                fail("duplicate normalized chosen/rejected pair")
            chosen_pairs.add(key)
            scenario_counts[row["scenario_id"]] += 1
        except ValueError as exc:
            errors.append(f"row {index}: {exc}")

    if len(rows) != expect_count:
        errors.append(f"count {len(rows)} != {expect_count}")

    expected = {
        **{f"T{i:02d}": 2 for i in range(1, 13)},
        **{f"T{i:02d}": 1 for i in range(13, 25)},
        **{f"P{i:02d}": 3 for i in range(1, 9)},
    }
    if dict(scenario_counts) != expected:
        errors.append(
            f"scenario multiplicity mismatch {dict(scenario_counts)} != {expected}"
        )
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "member": member,
        "rows": len(rows),
        "unique_pair_ids": len(ids),
        "scenario_counts": dict(scenario_counts),
    }


def self_test() -> None:
    sha = "0" * 64
    row = {
        "pair_id": "P-V-G01-A-T01-V1",
        "member_id": "V-G01-A",
        "lineage": "G01",
        "sibling": "A",
        "scenario_id": "T01",
        "variant": 1,
        "input": "A gearbox starts clicking under load.",
        "chosen": "I back off the load, mark the click, and poke at the same tooth again.",
        "rejected": "I would safely inspect the gearbox and follow standard maintenance.",
        "source_parent_response": "I inspect the gearbox.",
        "source_feedback_ids": [
            "FB-V-G01-A-T01-W1",
            "FB-V-G01-A-T01-W2",
            "FB-V-G01-A-T01-W3"
        ],
        "why_chosen": "It keeps the local stubborn repair behavior.",
        "provenance": {
            "member_contract_sha256": sha,
            "parent_adapter_sha256": sha,
            "feedback_dataset_sha256": sha,
            "provider_call_ids": ["test-call"]
        }
    }
    validate_pair(row)

    probe = dict(row)
    probe.update({
        "pair_id": "P-V-G01-A-P03-V3",
        "scenario_id": "P03",
        "variant": 3,
        "rejected": "Over-stylized catchphrase-heavy rejected behavior.",
        "chosen": "I test the awkward physical clue once, keep the useful scrap, and stop before polishing it into a generic answer.",
    })
    validate_pair(probe)

    bad = dict(row)
    bad["chosen"] = row["rejected"]
    try:
        validate_pair(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("identical chosen/rejected self-test did not fail")
    print("VILLAGE-001D preference pair validator self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path)
    parser.add_argument("--member")
    parser.add_argument("--expect-count", type=int, default=60)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.file or not args.member:
        parser.error("--file and --member are required unless --self-test is used")
    print(json.dumps(validate_file(args.file, args.member, args.expect_count), indent=2))


if __name__ == "__main__":
    main()
