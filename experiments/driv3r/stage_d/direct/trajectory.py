"""Read-only D-1A synthetic trajectory-effect comparison.

This helper compares *replayed offline* event chains. Callers must verify each
record against frozen source, legal candidates and its declared policy before
calling it. Cursor position and swept path are not native material changes.
"""
from __future__ import annotations

from typing import Any, Mapping

VERSION = "d1a-trajectory-effect-audit/1"


def compare_simulated_effects(arms: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare realised simulated cursor and accumulated path, never forecasts.

    Unlike the pre-choice catalog audit, effect comparisons do not require
    post-divergence legal candidates to be equal. An arm that has stopped keeps
    its last observed cursor/path for subsequent comparison steps. A mismatch
    between choices may be *non-consequential* if their effects are identical.
    """
    if not isinstance(arms, dict) or not 2 <= len(arms) <= 8:
        raise ValueError("Expected 2–8 verified named synthetic arms")
    names = sorted(arms)
    if any(not isinstance(n, str) or not n or not isinstance(arms[n], dict)
           or not isinstance(arms[n].get("events"), list)
           or not isinstance(arms[n].get("path"), list)
           or not arms[n]["path"] for n in names):
        raise ValueError("Malformed verified synthetic arm")

    def first_difference(snapshots: list[tuple[Any, ...]]) -> int | None:
        for index, state in enumerate(snapshots):
            if any(value != state[0] for value in state[1:]):
                return index
        return None

    max_steps = max(len(arms[name]["events"]) for name in names)
    cursors = []
    path_digests = []
    event_present = []
    for index in range(max_steps):
        cursors.append(tuple(
            tuple(arms[name]["events"][index]["simulated_post_position"])
            if index < len(arms[name]["events"]) else tuple(arms[name]["path"][-1])
            for name in names
        ))
        path_digests.append(tuple(
            arms[name]["events"][index]["simulated_path_sha256"]
            if index < len(arms[name]["events"]) else
            (arms[name]["events"][-1]["simulated_path_sha256"]
             if arms[name]["events"] else None)
            for name in names
        ))
        event_present.append(tuple(index < len(arms[name]["events"]) for name in names))

    first_cursor = first_difference(cursors)
    first_path = first_difference(path_digests)
    first_event_length = first_difference(event_present)
    first_geometry = min((x for x in (first_cursor, first_path) if x is not None), default=None)

    paths = [arms[name]["path"] for name in names]
    shared_cells = 0
    for cell_set in zip(*paths):
        if any(tuple(cell) != tuple(cell_set[0]) for cell in cell_set[1:]):
            break
        shared_cells += 1

    return {
        "version": VERSION,
        "evidence_kind": "synthetic_cursor_and_path_only",
        "native_qualified": False,
        "first_cursor_divergence_index": first_cursor,
        "first_swept_path_divergence_index": first_path,
        "first_geometric_divergence_index": first_geometry,
        "first_episode_length_divergence_index": first_event_length,
        "common_simulated_path_prefix_cells": shared_cells,
        "compared_event_index_count": max_steps,
        "first_consequential_native_divergence": "not_measured",
        "post_divergence_candidate_parity": "not_asserted",
    }
