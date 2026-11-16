"""Synthetic single-state D-1B guide/critic interaction, no native actions.

The host supplies an identical legal candidate catalogue to all arms. Model
inference, successful painting, real Undo, and permission claims are excluded.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re

from .shadow_policy import reference_verdict

PROTOCOL = "cybersand.driv3r.d1b-fork-probe/1"
HASH = re.compile(r"[0-9a-f]{64}\Z")
ARMS = ("geometry", "guidance", "critic", "combined")


def _digest(item: object) -> str:
    return sha256(json.dumps(item, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def compare_frozen_fork(*, world_sha256: str, guide_sha256: str,
                        legal_candidates: list[dict], gain: int = 2,
                        undo_eligible: bool = False,
                        hypothetical_receipt_available: bool = False) -> dict:
    """Compare matching legal options with/without scalar and shadow critique.

    A candidate contains ID, exact one-cell host target, integer geometry
    progress, u16 guidance value and hypothetical before/after goal/violations.
    Critic access is *after* the proposed choice and never changes that choice.
    Guidance is positive preference, not collision/permission authority.
    """
    if not all(type(s) is str and HASH.fullmatch(s) for s in (world_sha256, guide_sha256)):
        raise ValueError("source World and guidance SHA-256 required")
    if type(gain) is not int or not 0 <= gain <= 8:
        raise ValueError("gain must be bounded integer 0..8")
    if type(undo_eligible) is not bool or type(hypothetical_receipt_available) is not bool:
        raise ValueError("undo flags must be hypothetical booleans")
    if (type(legal_candidates) is not list or not 1 <= len(legal_candidates) <= 16):
        raise ValueError("bounded nonempty host-legal candidates required")
    ids = set()
    before_common = None
    for c in legal_candidates:
        if type(c) is not dict or set(c) != {"id", "to_xy", "geometry_progress",
                                                "guidance_u16", "before", "after"}:
            raise ValueError("invalid candidate shape")
        if (type(c["id"]) is not str or not 1 <= len(c["id"]) <= 64 or c["id"] in ids):
            raise ValueError("unique candidate IDs required")
        ids.add(c["id"])
        if (type(c["to_xy"]) is not list or len(c["to_xy"]) != 2 or
                any(type(v) is not int or not 0 <= v < 2**31 for v in c["to_xy"])):
            raise ValueError("invalid host-provided destination")
        if (type(c["geometry_progress"]) is not int or
                not -32768 <= c["geometry_progress"] <= 32767 or
                type(c["guidance_u16"]) is not int or not 0 <= c["guidance_u16"] <= 65535):
            raise ValueError("invalid bounded geometry or guide value")
        for label in ("before", "after"):
            m = c[label]
            if (type(m) is not dict or set(m) != {"goal_score", "violations", "progress_cells"} or
                    any(type(m[k]) is not int or not -32768 <= m[k] <= 32767
                        for k in ("goal_score", "progress_cells")) or
                    type(m["violations"]) is not int or not 0 <= m["violations"] <= 32767):
                raise ValueError("invalid hypothetical critic metrics")
        if before_common is None:
            before_common = c["before"]
        elif c["before"] != before_common:
            raise ValueError("candidate outcomes must share one frozen before-state")
    # Sort by immutable ID, independent of model-facing presentation order.
    ordered = sorted(legal_candidates, key=lambda c: c["id"])
    legal_catalogue = [{"id": c["id"], "to_xy": c["to_xy"]} for c in ordered]
    selection = {}
    for arm in ARMS:
        guided = arm in ("guidance", "combined")
        selected = max(ordered, key=lambda c: (c["geometry_progress"] * 65536 +
                                             (gain * c["guidance_u16"] if guided else 0),
                                             -ordered.index(c)))
        verdict = None
        if arm in ("critic", "combined"):
            verdict, reason = reference_verdict(
                selected["before"], selected["after"],
                undo_eligible=undo_eligible,
                verified_native_receipt=hypothetical_receipt_available)
            verdict = {"choice": verdict, "reason": reason, "executed": False}
        selection[arm] = {"candidate_id": selected["id"], "critic": verdict}
    return {"protocol": PROTOCOL, "evidence_class": "synthetic_single_fork_only",
            "declared_world_sha256": world_sha256, "declared_guide_sha256": guide_sha256,
            "declared_legal_id_target_sha256": _digest(legal_catalogue),
            "candidate_effects_sha256": _digest(ordered), "gain": gain,
            "selection": selection,
            "guide_changes_choice": selection["geometry"]["candidate_id"] != selection["guidance"]["candidate_id"],
            "native_material_writes": 0, "native_undo_executions": 0,
            "model_inference_performed": False, "native_receipts_verified": False,
            "interpretation": "host-declared hypothetical effects, not verified native outcomes or learned choices"}
