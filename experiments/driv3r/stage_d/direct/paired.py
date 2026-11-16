"""Offline-only evidence validation and matched-policy diagnostic tooling.

No D-COMMON conformance, native receipts, paintings or model quality are claimed.
The held-out split is strictly excluded from behavioral diagnostic entry points.
"""
from __future__ import annotations

import json
import random
from typing import Any, Mapping

from . import offline as d
from . import trajectory

VERSION = "d1a-paired-audit/1"
POLICIES = frozenset(("astar", "contour_astar", "random_legal"))


class EvidenceError(ValueError):
    """A purported synthetic episode cannot be reproduced from its frozen fixture."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise EvidenceError(message)


def _canonical(value: Any) -> Any:
    """Accept original tuples and losslessly JSON-roundtripped event arrays."""
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def verify_episode(f: d.Fixture, record: dict[str, Any]) -> dict[str, Any]:
    """Verify a complete *synthetic* event chain against the frozen geometry.

    Recreate all offered candidates, each pre-decision observation, exact
    choice, swept cursor path, post-position, score and STOP. In particular,
    a purported 'native' flag or receipt must not pass this verifier.
    """
    _require(f.split == "development", "Behavioral validation is development-only")
    _require(type(record) is dict, "Episode must be an object")
    _require(record.get("experiment_version") == d.VERSION, "Wrong episode version")
    _require(record.get("fixture") == f.fixture_id and
             record.get("fixture_sha256") == f.sha256 and record.get("split") == f.split,
             "Fixture identity/source digest/split mismatch")
    _require(record.get("source_kind") == "offline_simulation_only" and
             record.get("native_qualified") is False and
             record.get("sequential_native_receipts") == 0,
             "Offline evidence claims native execution or receipts")
    policy = record.get("policy")
    seed = record.get("seed")
    _require(policy in POLICIES and type(seed) is int and 0 <= seed <= 4294967295,
             "Unsupported policy or unbounded seed")
    raw_lengths = record.get("macro_lengths")
    _require(type(raw_lengths) in (tuple, list) and
             bool(raw_lengths) and all(type(n) is int and n in (1, 2, 4, 8)
                                       for n in raw_lengths) and
             list(raw_lengths) == sorted(set(raw_lengths)) and 1 in raw_lengths,
             "Invalid sorted, unique, bounded macro-length menu")
    lengths = tuple(raw_lengths)
    events = record.get("events")
    _require(type(events) is list and len(events) <= f.budget, "Invalid event count")
    pos = f.start
    path = [pos]
    rng = random.Random(seed)
    stop_reason = None
    for index, event in enumerate(events):
        _require(type(event) is dict and event.get("seq") == index,
                 "Nonmonotonic or missing event sequence")
        _require(pos != f.target, "Episode continued after arrival")
        expected_options = d.candidates(f, pos, lengths)
        expected_observation = d.observation_stub(f, pos, path, expected_options, index)
        _require(_canonical(event.get("observation")) ==
                 _canonical(expected_observation),
                 "Observed legal candidates, context or hash differ from frozen fixture")
        admitted = {item.candidate_id: item for item in expected_options}
        candidate_id = event.get("choice_id")
        _require(type(candidate_id) is str and candidate_id in admitted,
                 "Unknown or forged candidate ID")
        chosen = admitted[candidate_id]
        expected_choice = d.select_candidate(f, pos, expected_options, policy, rng)
        _require(chosen == expected_choice,
                 "Choice not reproducible from declared policy and seed")
        _require(all(f.legal(p) for p in chosen.path),
                 "Unadmitted or blocked swept cell")
        if chosen.stop:
            stop_reason = "policy_stop"
        else:
            path.extend(chosen.path)
            pos = chosen.path[-1]
            if pos == f.target:
                stop_reason = "arrived"
        _require(_canonical(event.get("simulated_post_position")) == _canonical(pos),
                 "Post-action cursor mismatch")
        _require(event.get("native_receipt_ref") is None,
                 "Synthetic action has counterfeit native receipt")
        _require(event.get("simulated_path_sha256") == d.digest(path),
                 "Simulated trajectory hash mismatch")
        _require(stop_reason is None or index == len(events) - 1,
                 "Events continue after STOP or arrival")
    if stop_reason is None:
        _require(len(events) == f.budget, "Truncated episode with no termination")
        stop_reason = "budget_exhausted"
    expected_metrics = d.score_episode(f, path, len(events), stop_reason)
    _require(_canonical(record.get("path")) == _canonical(path),
             "Final path does not replay from accepted events")
    _require(_canonical(record.get("metrics")) == _canonical(expected_metrics),
             "Final geometry/STOP/violation metrics differ from replay")
    return {"validated": True, "synthetic_only": True, "fixture_sha256": f.sha256,
            "event_count": len(events), "path_sha256": d.digest(path),
            "metrics_sha256": d.digest(expected_metrics)}


def compare_matched(f: d.Fixture,
                    arms: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Audit common legal action menus through the first meaningful divergence.

    After an arm selects a different action it legitimately reaches a
    different World/cursor. Later candidate sets MUST NOT be called matched.
    This is a comparison design aid, not model inference or native testing.
    """
    _require(f.split == "development", "Cannot evaluate held-out behavior")
    _require(type(arms) is dict and 2 <= len(arms) <= 8,
             "Need 2-8 independently named arms")
    names = sorted(arms)
    for name in names:
        _require(type(name) is str and bool(name) and
                 type(arms[name]) is dict, "Invalid named arm")
        verify_episode(f, arms[name])
    lengths = {tuple(arms[name]["macro_lengths"]) for name in names}
    _require(len(lengths) == 1, "Different action menus invalidate candidate matching")
    starts = [arms[name]["events"] for name in names]
    limit = min(len(events) for events in starts)
    divergence = None
    reason = None
    for index in range(limit):
        before = [events[index]["observation"] for events in starts]
        identities = [(view["position"], view["candidate_set_sha256"])
                      for view in before]
        _require(len({json.dumps(_canonical(identity), sort_keys=True) for identity in identities}) == 1,
                 "Unequal legal candidates or positions before choice divergence")
        choices = [events[index]["choice_id"] for events in starts]
        if len(set(choices)) != 1:
            divergence = index
            reason = "different_admitted_choice"
            break
    if divergence is None and len({len(events) for events in starts}) != 1:
        divergence = limit
        reason = "different_episode_length"
    prefix = limit if divergence is None else divergence
    first_catalog = None if divergence is None or divergence >= limit else (
        starts[0][divergence]["observation"]["candidate_set_sha256"])
    return {
        "version": VERSION, "source_kind": "offline_simulation_only",
        "native_qualified": False, "fixture": f.fixture_id, "fixture_sha256": f.sha256,
        "arm_names": names, "first_choice_divergence_index": divergence,
        "divergence_reason": reason, "common_choice_prefix_decisions": prefix,
        "catalog_sha256_at_choice_divergence": first_catalog,
        "matched_candidate_sets_checked_through_divergence": True,
        "post_divergence_candidate_matching_claimed": False,
        "arm_metrics": {name: arms[name]["metrics"] for name in names},
        "arm_path_sha256": {name: d.digest(arms[name]["path"]) for name in names},
        "synthetic_effect_audit": trajectory.compare_simulated_effects(arms),
    }


def run_development_pair(f: d.Fixture, seed: int = 0,
                         macro_lengths: tuple[int, ...] = (1,)) -> dict[str, Any]:
    """Compare strong contour/geometric A* and random-legal, development only."""
    _require(f.split == "development", "Held-out episode generation forbidden")
    strong = "contour_astar" if f.task == "wall_offset" else "astar"
    arms = {
        "geometric": d.run_episode(f, strong, seed, macro_lengths),
        "random_legal": d.run_episode(f, "random_legal", seed, macro_lengths),
    }
    return compare_matched(f, arms)
