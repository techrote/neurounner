"""Pre-M1 A1/A2/A3 paired shadow-choice comparison; no native authority.

A frozen decision fork has ONE host-owned legal candidate set shared by all
arms. Following a consequential fork, arm rollouts are independent; they are
not falsely treated as paired native states. Metrics are supplied synthetic
illustrations, not estimated or verified by this module.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re

ARMS = ("A1", "A2", "A3")
SHA = re.compile(r"[a-f0-9]{64}\Z")
COMMIT = re.compile(r"[a-f0-9]{40}\Z")


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(value: object, name: str) -> None:
    _fail(isinstance(value, str) and SHA.fullmatch(value) is not None,
          f"{name} must be a lowercase SHA-256")


def _arms(value: object, name: str) -> None:
    _fail(isinstance(value, dict) and set(value) == set(ARMS),
          f"{name} must contain exactly A1, A2 and A3")


def _integer(value: object, name: str, minimum: int = 0) -> None:
    _fail(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")


def _hash_record(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def analyze_shadow_comparison(*, forks: list[dict], rollouts: dict[str, list[dict]],
                              scenario_sha256: str, source_revision: str,
                              config_sha256: dict[str, str], seed: int,
                              evidence_class: str = "synthetic") -> dict:
    """Audit frozen candidate equality, first effect divergence and follow-through.

    Each fork has a shared source-world hash, legal candidate IDs, hypothetical
    effect hashes, and one arm choice per identical source snapshot. Different
    candidate IDs may have identical *predicted* effect hashes; this distinction
    matters when judging the first consequential choice. Effect hashes are
    unverified predictions. No inferences about real native commits are made.

    After a consequential fork, the three rollouts describe independent branch
    outcomes, not three observations of an identical evolving World. Cell
    counts must be supplied by a later qualified host to become evidence.
    """
    _fail(evidence_class == "synthetic", "only synthetic pre-M1 comparisons are supported")
    _sha(scenario_sha256, "scenario_sha256")
    _fail(isinstance(source_revision, str) and COMMIT.fullmatch(source_revision) is not None,
          "source_revision must be a 40-char lowercase source commit")
    _integer(seed, "seed")
    _fail(seed <= 2**32 - 1, "seed exceeds u32")
    _arms(config_sha256, "config_sha256")
    for arm, digest in config_sha256.items():
        _sha(digest, f"config_sha256[{arm}]")
    _arms(rollouts, "rollouts")
    _fail(isinstance(forks, list) and 1 <= len(forks) <= 36, "1..36 frozen decision forks required")

    first_choice, first_effect = None, None
    seen_snapshots = set()
    candidate_id_hashes = []
    predicted_effect_hashes = []
    for number, fork in enumerate(forks):
        _fail(isinstance(fork, dict), f"fork {number} must be a record")
        snapshot = fork.get("snapshot_id")
        _fail(isinstance(snapshot, str) and 1 <= len(snapshot) <= 96 and
              snapshot not in seen_snapshots, "unique nonempty snapshot_id required")
        seen_snapshots.add(snapshot)
        _sha(fork.get("source_world_sha256"), f"fork {number} source_world_sha256")
        candidates = fork.get("legal_candidates")
        _fail(isinstance(candidates, list) and 1 <= len(candidates) <= 128 and
              all(isinstance(c, str) and 1 <= len(c) <= 96 for c in candidates) and
              len(set(candidates)) == len(candidates), f"fork {number} needs unique legal candidates")
        effects = fork.get("candidate_effect_sha256")
        _fail(isinstance(effects, dict) and set(effects) == set(candidates),
              f"fork {number} must declare the effect of every legal candidate")
        for candidate, digest in effects.items():
            _sha(digest, f"fork {number} predicted effect {candidate}")
        choices = fork.get("choices")
        _arms(choices, f"fork {number} choices")
        _fail(all(choice in candidates for choice in choices.values()),
              f"fork {number} contains a nonlegal choice")
        # A model-facing order shuffle is not a new host candidate set.
        # Predicted effects are separately keyed by stable candidate identity.
        candidate_id_hashes.append(_hash_record(sorted(candidates)))
        predicted_effect_hashes.append(_hash_record(effects))
        if first_choice is None and len(set(choices.values())) > 1:
            first_choice = {"decision_index": number, "snapshot_id": snapshot}
        if first_effect is None and len({effects[c] for c in choices.values()}) > 1:
            first_effect = {"decision_index": number, "snapshot_id": snapshot,
                            "classification": "different_predicted_effect_not_native_verified"}

    _fail(first_effect is not None or all(not row for row in rollouts.values()),
          "follow-through cannot be compared without a consequential decision fork")
    totals = {}
    for arm in ARMS:
        rows = rollouts[arm]
        _fail(isinstance(rows, list) and len(rows) <= 36,
              f"{arm} rollout must contain at most 36 bounded steps")
        total_committed = total_productive = total_progress = 0
        for ordinal, row in enumerate(rows):
            _fail(isinstance(row, dict) and row.get("decision_ordinal") == ordinal,
                  f"{arm} rollout decisions must use contiguous ordinal indexes")
            _fail(row.get("evidence_class") == "synthetic",
                  f"{arm} pre-M1 rollout must remain marked synthetic")
            committed, productive, progress = (row.get("committed_cells"),
                                               row.get("productive_cells"),
                                               row.get("net_objective_progress_cells"))
            _integer(committed, f"{arm} committed_cells")
            _integer(productive, f"{arm} productive_cells")
            _integer(progress, f"{arm} net_objective_progress_cells", minimum=-2**31)
            _fail(productive <= committed, f"{arm} productive cells exceed committed cells")
            total_committed += committed
            total_productive += productive
            total_progress += progress
        totals[arm] = {"decisions_observed": len(rows),
                       "reported_committed_cells": total_committed,
                       "reported_productive_cells": total_productive,
                       "reported_net_objective_progress_cells": total_progress}
    return {"protocol": "cybersand.driv3r.d1b-shadow-comparison/1",
            "evidence_class": "synthetic", "native_execution_verified": False,
            "scenario_sha256": scenario_sha256, "source_revision": source_revision,
            "config_sha256": dict(config_sha256), "seed": seed,
            "frozen_forks": len(forks),
            "candidate_id_set_sha256_by_fork": candidate_id_hashes,
            "predicted_effect_catalog_sha256_by_fork": predicted_effect_hashes,
            "first_choice_divergence": first_choice,
            "first_consequential_divergence": first_effect,
            "post_fork_reported_metrics": totals,
            "limitation": "candidate ID hashes are NOT legal-footprint hashes; predicted effects "
                          "and supplied rollout counts only, no native causal qualification"}
