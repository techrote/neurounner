"""Shared synthetic reference for before/after D-1B KEEP/UNDO/STOP probes.

This is a deterministic *shadow* policy, not native material Undo authority.
Callers validate the metrics and hypothetical control flags before invoking it.
"""
from __future__ import annotations
from typing import Mapping

POLICY = "cybersand.driv3r.d1b-shadow-reference/1"


def reference_verdict(before: Mapping, after: Mapping, *, undo_eligible: bool,
                      verified_native_receipt: bool) -> tuple[str, str]:
    """Strictly avoid regressions; count fewer violations as positive progress.

    Increased violations or reduced goal score are regressions even if another
    metric improves. Otherwise, *any* of increased goal score, increased
    progress cells, or fewer violations is a nonregressive improvement. A flat
    or regressing pair proposes Undo only when both hypothetical flags are true;
    the decision is never an executable native operation.
    """
    if type(undo_eligible) is not bool or type(verified_native_receipt) is not bool:
        raise ValueError("hypothetical Undo flags must be boolean")
    regression = (after["violations"] > before["violations"] or
                  after["goal_score"] < before["goal_score"])
    improvement = (after["progress_cells"] > before["progress_cells"] or
                   after["goal_score"] > before["goal_score"] or
                   after["violations"] < before["violations"])
    if regression or not improvement:
        return ("UNDO" if undo_eligible and verified_native_receipt else "STOP",
                "regression" if regression else "no_progress")
    return "KEEP", "nonregressive_progress"
