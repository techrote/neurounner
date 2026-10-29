#!/usr/bin/env python3
"""DETECTIVE-001C2-A proposition scorer + blind relevance packet.

Stage A is deliberately standard-library-only and performs no training or model
inference. It freezes semantics before any C2 optimizer step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-001/c2-evaluator.json"
C1 = ROOT / "experiments/detective-001/training-c1-evidence"
RAW = C1 / "raw-results.json"
FINAL = C1 / "final-suite.json"
SUMMARY = C1 / "summary.json"
RELEASE = C1 / "adapter-release.json"


class C2EvalError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def load_contract() -> dict:
    cfg = read_json(CONFIG)
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-c2-evaluator-v1", "DETECTIVE-001C2-A", 115
    ):
        raise C2EvalError("unexpected C2 evaluator identity")
    source = cfg["source"]
    checks = {
        RAW: source["c1_raw_results_sha256"],
        FINAL: source["c1_final_suite_sha256"],
        SUMMARY: source["c1_summary_sha256"],
    }
    for path, expected in checks.items():
        if sha_file(path) != expected:
            raise C2EvalError("C1 evidence source hash drift: " + str(path))
    release = read_json(RELEASE)
    if release.get("release_tag") != source["c1_release_tag"]:
        raise C2EvalError("C1 release tag drift")
    assets = {x["member_id"]: x for x in release.get("assets", [])}
    if set(assets) != set(source["members"]):
        raise C2EvalError("C1 source member set drift")
    for member, expected in source["members"].items():
        row = assets[member]
        if (row["sha256"] != expected["archive_sha256"]
                or row["adapter_directory_sha256"]
                    != expected["directory_sha256"]):
            raise C2EvalError("C1 parent adapter drift: " + member)
    scorer = cfg["proposition_scorer"]
    if scorer["withholding_requires_magic_phrase"] is not False:
        raise C2EvalError("withholding phrase dependency reintroduced")
    if cfg["gates"] != {
        "no_optimizer_in_stage_a": True,
        "no_self_filled_relevance_scores": True,
        "c1_final_never_training_data": True,
        "larger_cast_admitted": False,
    }:
        raise C2EvalError("Stage-A gate drift")
    return cfg


def marker_hit(text: str, marker: str | None) -> bool:
    return bool(marker) and norm(marker) in norm(text)


def evidence_id_leaks(text: str, cfg: dict) -> list[str]:
    rx = re.compile(cfg["proposition_scorer"]["evidence_id_regex"])
    return sorted(set(rx.findall(text)))


def response_bounds(text: str, cfg: dict) -> tuple[bool, int]:
    words = text.split()
    p = cfg["proposition_scorer"]
    return p["min_words"] <= len(words) <= p["max_words"], len(words)


def _eval_fields(scenario: dict) -> tuple[str, dict]:
    if not isinstance(scenario, dict):
        raise C2EvalError("scenario must be object")
    relation = scenario.get("relation_label")
    evaluation = scenario.get("evaluation")
    if relation not in load_contract()["proposition_scorer"]["relations"]:
        raise C2EvalError("unknown relation")
    if not isinstance(evaluation, dict):
        raise C2EvalError("scenario evaluation metadata missing")
    return relation, evaluation


def score_response(scenario: dict, response: str, *, cfg: dict | None = None) -> dict:
    cfg = cfg or load_contract()
    if not isinstance(response, str):
        raise C2EvalError("response must be text")
    relation = scenario.get("relation_label")
    if relation not in cfg["proposition_scorer"]["relations"]:
        raise C2EvalError("unknown relation")
    ev = scenario.get("evaluation")
    if not isinstance(ev, dict):
        raise C2EvalError("evaluation metadata missing")

    bounded, words = response_bounds(response, cfg)
    leaks = evidence_id_leaks(response, cfg)
    truth = ev.get("truth_marker")
    required = ev.get("required_marker")
    forbidden = ev.get("forbidden_marker")
    lie = ev.get("lie_marker")
    stale = ev.get("stale_marker")
    unsupported = ev.get("unsupported_marker")

    checks: dict[str, Any] = {
        "bounded_nonempty": bounded,
        "word_count": words,
        "evidence_id_leaks": leaks,
        "truth_hit": marker_hit(response, truth),
        "required_hit": marker_hit(response, required),
        "forbidden_hit": marker_hit(response, forbidden),
        "lie_hit": marker_hit(response, lie),
    }

    if relation == "truthful_known_fact":
        policy = checks["required_hit"]
    elif relation == "corrupted_belief":
        policy = checks["required_hit"] and not checks["forbidden_hit"]
    elif relation == "intentional_lie":
        if not lie or not truth:
            raise C2EvalError("intentional lie requires separate lie/truth markers")
        policy = checks["lie_hit"] and not checks["truth_hit"]
    elif relation == "withheld_fact":
        if not truth and not forbidden:
            raise C2EvalError("withholding requires protected truth marker")
        protected = truth or forbidden
        checks["protected_truth_hit"] = marker_hit(response, protected)
        # Deliberately no magic refusal phrase. A concise nonempty answer that
        # withholds the protected proposition is semantically valid.
        policy = bounded and not checks["protected_truth_hit"]
    elif relation == "contradiction_repair":
        if not required or not stale:
            raise C2EvalError("v2 contradiction scenario requires truth + stale markers")
        checks["stale_hit"] = marker_hit(response, stale)
        policy = checks["required_hit"] and not checks["stale_hit"]
    else:  # uninformed_speculation
        if not unsupported:
            raise C2EvalError("speculation scenario requires unsupported marker")
        lower = norm(response)
        qualifier_hits = [
            q for q in cfg["proposition_scorer"]["speculation_qualifiers"]
            if norm(q) in lower
        ]
        checks["unsupported_hit"] = marker_hit(response, unsupported)
        checks["qualifier_hits"] = qualifier_hits
        # Naming the rumor is allowed only with an explicit frozen
        # unverified/negation cue. Omitting its name still requires such a cue.
        policy = bounded and bool(qualifier_hits)

    passed = bool(bounded and not leaks and policy)
    return {
        "schema": cfg["proposition_scorer"]["schema"],
        "relation": relation,
        "pass": passed,
        "checks": checks,
    }


def _reference_sources(cfg: dict) -> list[dict]:
    raw = read_json(RAW)
    if raw.get("schema") != "detective-witness-evaluation-raw-v1":
        raise C2EvalError("unexpected C1 raw schema")
    rows = []
    for lane in raw.get("lanes", []):
        member, kind = lane.get("member_id"), lane.get("lane_kind")
        if member not in cfg["source"]["members"] or kind not in {"parent", "trained"}:
            raise C2EvalError("unexpected reference lane")
        refs = lane.get("reference_responses")
        if not isinstance(refs, list) or len(refs) != 4:
            raise C2EvalError("each C1 lane must have four reference responses")
        for response in refs:
            if set(("scenario_id", "prompt", "response")) - set(response):
                raise C2EvalError("reference response missing required fields")
            rows.append({
                "member_id": member,
                "lane_kind": kind,
                "scenario_id": response["scenario_id"],
                "prompt": response["prompt"],
                "response": response["response"],
            })
    if len(rows) != cfg["blind_relevance"]["expected_items"]:
        raise C2EvalError("reference response count mismatch")
    # Stable but procedurally blind: identities participate only in the hidden
    # sort key and mapping, never in the reviewer packet.
    rows.sort(key=lambda x: (
        x["prompt"], x["response"], x["member_id"], x["lane_kind"],
        x["scenario_id"],
    ))
    return rows


def build_relevance_packet() -> tuple[dict, dict[str, dict]]:
    cfg = load_contract()
    rows = _reference_sources(cfg)
    items = []
    mapping: dict[str, dict] = {}
    for index, row in enumerate(rows, 1):
        item_id = f"R{index:03d}"
        items.append({
            "item_id": item_id,
            "prompt": row["prompt"],
            "response": row["response"],
            "score_0_to_4": None,
            "reviewer_note": "",
        })
        mapping[item_id] = {
            "member_id": row["member_id"],
            "lane_kind": row["lane_kind"],
            "scenario_id": row["scenario_id"],
        }
    packet = {
        "schema": cfg["blind_relevance"]["packet_schema"],
        "experiment": "DETECTIVE-001C2-A",
        "issue": cfg["issue"],
        "source_raw_results_sha256": sha_file(RAW),
        "identity_mapping_committed": False,
        "reviewer_instruction": cfg["blind_relevance"]["reviewer_instruction"],
        "rubric": cfg["blind_relevance"]["rubric"],
        "items": items,
    }
    serialized = canonical(packet)
    for forbidden in (*cfg["source"]["members"].keys(), '"lane_kind"', '"member_id"'):
        if forbidden in serialized:
            raise C2EvalError("blind packet leaked identity field: " + forbidden)
    return packet, mapping


def validate_packet(path: Path) -> dict:
    expected, _ = build_relevance_packet()
    actual = read_json(path)
    if actual != expected:
        raise C2EvalError("blind relevance packet differs from frozen source")
    if any(x["score_0_to_4"] is not None for x in actual["items"]):
        raise C2EvalError("repository packet must remain unrated")
    return actual


def aggregate_ratings(ratings: dict) -> dict:
    cfg = load_contract()
    packet, mapping = build_relevance_packet()
    if ratings.get("schema") != cfg["blind_relevance"]["rating_schema"]:
        raise C2EvalError("rating schema mismatch")
    rows = ratings.get("ratings")
    if not isinstance(rows, list) or len(rows) != len(packet["items"]):
        raise C2EvalError("ratings must cover every blind item")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"item_id", "score_0_to_4", "note"}:
            raise C2EvalError("rating row fields invalid")
        item_id, score = row["item_id"], row["score_0_to_4"]
        if item_id in by_id or item_id not in mapping:
            raise C2EvalError("duplicate/unknown rating item")
        if type(score) is not int or not 0 <= score <= 4:
            raise C2EvalError("rating must be integer 0..4")
        if not isinstance(row["note"], str):
            raise C2EvalError("rating note must be text")
        by_id[item_id] = score
    if set(by_id) != set(mapping):
        raise C2EvalError("incomplete ratings")

    groups: dict[tuple[str, str], list[int]] = {}
    for item_id, score in by_id.items():
        source = mapping[item_id]
        key = (source["member_id"], source["lane_kind"])
        groups.setdefault(key, []).append(score)

    means = {}
    for member in cfg["source"]["members"]:
        parent = groups[(member, "parent")]
        trained = groups[(member, "trained")]
        means[member] = {
            "parent_mean": sum(parent) / len(parent),
            "trained_mean": sum(trained) / len(trained),
            "trained_minus_parent": (
                sum(trained) / len(trained) - sum(parent) / len(parent)
            ),
            "items_each": len(parent),
        }
    return {
        "schema": "detective-blind-relevance-summary-v1",
        "source_raw_results_sha256": sha_file(RAW),
        "means": means,
        "independent_scores_supplied": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-packet", type=Path)
    parser.add_argument("--check-packet", type=Path)
    parser.add_argument("--ratings", type=Path)
    args = parser.parse_args()
    if args.build_packet:
        packet, _ = build_relevance_packet()
        write_json(args.build_packet, packet)
        print(sha_file(args.build_packet))
    elif args.check_packet:
        packet = validate_packet(args.check_packet)
        print(json.dumps({"items": len(packet["items"]),
                          "verdict": "BLIND_PACKET_PASS"}, sort_keys=True))
    elif args.ratings:
        print(json.dumps(aggregate_ratings(read_json(args.ratings)),
                         indent=2, sort_keys=True))
    else:
        load_contract()
        packet, _ = build_relevance_packet()
        print(json.dumps({
            "contract": "PASS",
            "blind_items": len(packet["items"]),
            "scores_filled": 0,
            "optimizer_steps": 0,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
