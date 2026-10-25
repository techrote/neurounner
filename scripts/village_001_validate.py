#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMBER_RE = re.compile(r"^V-(G(?:0[1-9]|1[0-2]))-([ABC])$")
LINEAGE_RE = re.compile(r"^G(?:0[1-9]|1[0-2])$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SURFACES = {
    "physical_reaction",
    "embodied_decision",
    "local_memory",
    "conflicting_motives",
    "failure_and_recovery",
    "mundane_low_stakes",
    "conversation_social",
}
SPECIALIZATIONS = {
    "A": "failure_hardened",
    "B": "local_tinkerer",
    "C": "social_scavenger",
}


def fail(message: str) -> None:
    raise ValueError(message)


def require_str(row: dict, key: str, minimum: int = 1) -> str:
    value = row.get(key)
    if not isinstance(value, str) or len(value.strip()) < minimum:
        fail(f"{key}: expected string length >= {minimum}")
    return value.strip()


def require_list(row: dict, key: str, minimum: int = 0, maximum: int | None = None) -> list:
    value = row.get(key)
    if not isinstance(value, list):
        fail(f"{key}: expected list")
    if len(value) < minimum:
        fail(f"{key}: expected at least {minimum} items")
    if maximum is not None and len(value) > maximum:
        fail(f"{key}: expected at most {maximum} items")
    if len(value) != len({json.dumps(x, sort_keys=True) for x in value}):
        fail(f"{key}: duplicate items")
    return value


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def parse_member(row: dict) -> tuple[str, str, str]:
    member_id = require_str(row, "member_id")
    match = MEMBER_RE.fullmatch(member_id)
    if not match:
        fail(f"member_id: invalid {member_id!r}")
    lineage, sibling = match.groups()
    if row.get("lineage") != lineage:
        fail(f"lineage: {row.get('lineage')!r} does not match {member_id}")
    if row.get("sibling") != sibling:
        fail(f"sibling: {row.get('sibling')!r} does not match {member_id}")
    return member_id, lineage, sibling


def validate_corpus_row(row: dict) -> None:
    if not isinstance(row, dict):
        fail("row is not an object")
    member_id, lineage, sibling = parse_member(row)
    require_str(row, "example_id")
    if row.get("specialization") != SPECIALIZATIONS[sibling]:
        fail(f"specialization mismatch for {member_id}")
    if row.get("source_kind") not in {"ancestral_copy", "cloud_expansion"}:
        fail("source_kind: invalid")
    if row.get("surface") not in SURFACES:
        fail("surface: invalid")
    require_str(row, "input", 12)
    require_str(row, "target", 20)
    require_str(row, "generic_bad_target", 15)
    require_str(row, "why_bad", 12)
    if row.get("source_guide_id") != lineage:
        fail("source_guide_id must equal member lineage")
    require_str(row, "source_packet_id")
    babbles = require_list(row, "source_babble_ids", 1)
    if any(not isinstance(x, str) or not x.strip() for x in babbles):
        fail("source_babble_ids: all items must be non-empty strings")
    traits = require_list(row, "trait_refs", 1, 5)
    seeds = require_list(row, "seed_refs", 0, 3)
    if any(not isinstance(x, str) or not x.strip() for x in traits + seeds):
        fail("trait_refs/seed_refs: all items must be non-empty strings")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        fail("provenance: expected object")
    require_str(provenance, "source_dataset")
    contract = require_str(provenance, "member_contract_sha256")
    if not SHA_RE.fullmatch(contract):
        fail("provenance.member_contract_sha256: invalid SHA-256")


def validate_feedback_row(row: dict) -> None:
    if not isinstance(row, dict):
        fail("row is not an object")
    _, _, _ = parse_member(row)
    require_str(row, "feedback_id")
    parent = require_str(row, "parent_adapter_sha256")
    if not SHA_RE.fullmatch(parent):
        fail("parent_adapter_sha256: invalid SHA-256")
    require_str(row, "scenario_id")
    worker = row.get("sprocket_worker")
    if not isinstance(worker, int) or isinstance(worker, bool) or not 1 <= worker <= 3:
        fail("sprocket_worker: expected integer 1..3")
    request = row.get("request")
    if not isinstance(request, dict):
        fail("request: expected object")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        fail("request.messages: expected non-empty list")
    if not isinstance(row.get("raw_response"), str):
        fail("raw_response: expected string")
    if not isinstance(row.get("valid"), bool):
        fail("valid: expected boolean")
    if row["valid"]:
        parsed = row.get("parsed")
        if not isinstance(parsed, dict):
            fail("parsed: valid feedback requires object")
        for key in ("keep", "mutate"):
            values = require_list(parsed, key)
            if any(not isinstance(x, str) or not x.strip() for x in values):
                fail(f"parsed.{key}: expected non-empty strings")
        require_str(parsed, "failure_guess")
        if parsed.get("genericness") not in {"low", "medium", "high"}:
            fail("parsed.genericness: invalid")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        fail("provenance: expected object")
    require_str(provenance, "sprocket_revision")
    for key in ("sprocket_sha256", "member_contract_sha256"):
        value = require_str(provenance, key)
        if not SHA_RE.fullmatch(value):
            fail(f"provenance.{key}: invalid SHA-256")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def validate_file(path: Path, kind: str, expect_member: str | None, expect_count: int | None) -> dict:
    rows = load_jsonl(path)
    ids: set[str] = set()
    normalized_inputs: set[str] = set()
    source_counts: Counter[str] = Counter()
    errors = []

    for index, row in enumerate(rows, 1):
        try:
            if kind == "corpus":
                validate_corpus_row(row)
                row_id = row["example_id"]
                norm = normalize_text(row["input"])
                if norm in normalized_inputs:
                    fail("duplicate normalized input")
                normalized_inputs.add(norm)
                source_counts[row["source_kind"]] += 1
            else:
                validate_feedback_row(row)
                row_id = row["feedback_id"]
            if row_id in ids:
                fail(f"duplicate record ID {row_id}")
            ids.add(row_id)
            if expect_member and row["member_id"] != expect_member:
                fail(f"member_id {row['member_id']} != expected {expect_member}")
        except ValueError as exc:
            errors.append(f"row {index}: {exc}")

    if expect_count is not None and len(rows) != expect_count:
        errors.append(f"count: {len(rows)} != {expect_count}")

    if errors:
        raise ValueError("\n".join(errors))

    return {
        "path": str(path),
        "kind": kind,
        "rows": len(rows),
        "unique_ids": len(ids),
        "source_counts": dict(source_counts),
    }


def self_test() -> None:
    sha = "0" * 64
    corpus = {
        "example_id": "V-G01-A-E0001",
        "member_id": "V-G01-A",
        "lineage": "G01",
        "sibling": "A",
        "specialization": "failure_hardened",
        "source_kind": "cloud_expansion",
        "surface": "failure_and_recovery",
        "input": "A loose cog starts skipping under load.",
        "target": "I mark the skip, back the load off, and test the cog again before trusting it.",
        "generic_bad_target": "I would safely inspect the machine and follow normal maintenance procedures.",
        "why_bad": "It removes the lineage-specific stubborn local investigation.",
        "source_guide_id": "G01",
        "source_packet_id": "packet-G01",
        "source_babble_ids": ["G01-B01"],
        "trait_refs": ["E1"],
        "seed_refs": [],
        "provenance": {
            "source_dataset": "datasets/babble-001",
            "member_contract_sha256": sha,
        },
    }
    validate_corpus_row(corpus)

    feedback = {
        "feedback_id": "FB-V-G01-A-T01-W1",
        "member_id": "V-G01-A",
        "lineage": "G01",
        "sibling": "A",
        "parent_adapter_sha256": sha,
        "scenario_id": "T01",
        "sprocket_worker": 1,
        "request": {"messages": [{"role": "user", "content": "judge this"}]},
        "raw_response": "{}",
        "valid": True,
        "parsed": {
            "keep": ["keeps worrying the local fault"],
            "mutate": ["make the recovery less tidy"],
            "failure_guess": "loops on the same repair",
            "genericness": "low",
        },
        "provenance": {
            "sprocket_revision": "test",
            "sprocket_sha256": sha,
            "member_contract_sha256": sha,
        },
    }
    validate_feedback_row(feedback)

    bad = dict(corpus)
    bad["lineage"] = "G02"
    try:
        validate_corpus_row(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("lineage mismatch self-test did not fail")

    bad_feedback = dict(feedback)
    bad_feedback["valid"] = True
    bad_feedback["parsed"] = None
    try:
        validate_feedback_row(bad_feedback)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid parsed feedback self-test did not fail")

    print("VILLAGE-001 validator self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["corpus", "feedback"])
    parser.add_argument("--file", type=Path)
    parser.add_argument("--expect-member")
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.kind or not args.file:
        parser.error("--kind and --file are required unless --self-test is used")

    summary = validate_file(
        args.file,
        args.kind,
        args.expect_member,
        args.expect_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
