#!/usr/bin/env python3
"""WORM-001 support: bounded, literal raw-action cycle/repetition diagnostics.

Scores exact unmodified model/controller choices, not W0 applied fallbacks.
An amusing repeated motif is a visual observation, NOT evidence of exploration.
"""
from __future__ import annotations


def dominant_repeated_motif(actions: list[str], max_period: int = 6) -> dict:
    """Return longest contiguous exact sequence repeated >=3 complete times.

    This is diagnostic only. A score near 1 may mean pathological action
    collapse. Equal-span ties prefer shorter cycles and earlier starts.
    No smoothing, token repair, host fallback or policy label inference.
    Runtime O(n^2 * max_period), bounded by the W0 512-choice budget.
    """
    if (not isinstance(actions, list) or len(actions) > 512
            or any(not isinstance(x, str) for x in actions)
            or not 1 <= max_period <= 6):
        raise ValueError("invalid bounded literal raw-action trace")
    n = len(actions)
    best = {"period": None, "motif": [], "start": None,
            "span": 0, "full_repeats": 0, "fraction": 0.0}
    for period in range(1, min(max_period, n // 3) + 1):
        for start in range(n - 3 * period + 1):
            end = start + period
            while end < n and actions[end] == actions[end - period]:
                end += 1
            repeats = (end - start) // period
            span = repeats * period
            if repeats < 3 or span <= best["span"]:
                continue
            best = {"period": period, "motif": actions[start:start+period],
                    "start": start, "span": span, "full_repeats": repeats,
                    "fraction": round(span / n, 6)}
    return best
