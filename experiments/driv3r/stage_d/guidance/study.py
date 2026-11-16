"""Synthetic matched observations and bounded critic fixtures (local /1 isolation).

No model inference, real native Undo, permission changes or native World mutation.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping
from .shadow_policy import reference_verdict


def stable_hash(obj: object) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def matched_arms(*, snapshot_id: str, candidate_ids: list[str], geometry: Mapping,
                 guidance: Mapping, critic: Mapping) -> dict[str, dict]:
    """All four views share the same frozen state/candidates and source hashes.

    These are local *shadow* study records, not accepted D-COMMON observations.
    Withheld information is explicitly null; no arm is allowed to smuggle
    the omitted data through a summary.
    """
    if (not isinstance(snapshot_id, str) or not snapshot_id or
            not isinstance(candidate_ids, list) or not candidate_ids or
            any(not isinstance(cid, str) or not cid for cid in candidate_ids) or
            len(set(candidate_ids)) != len(candidate_ids)):
        raise ValueError("snapshot and unique candidate IDs required")
    # Display order is not host candidate-set identity.
    candidate_hash = stable_hash(sorted(candidate_ids))
    inputs = {"geometry": deepcopy(dict(geometry)), "guidance": deepcopy(dict(guidance)),
              "critic": deepcopy(dict(critic))}
    if any(not isinstance(x, dict) for x in inputs.values()):
        raise ValueError("invalid view")
    flags = {
        "geometry_only": (True, False, False),
        "guidance_only": (False, True, False),
        "critic_only": (False, False, True),
        "combined": (True, True, True),
    }
    result = {}
    full_hashes = {name: stable_hash(value) for name, value in inputs.items()}
    for arm, use in flags.items():
        visible_hashes = {name: full_hashes[name]
                          for name, enabled in zip(("geometry", "guidance", "critic"), use)
                          if enabled}
        view = {
            "protocol": "cybersand.driv3r.d1b-shadow/1",
            "status": "synthetic_pre_m1_no_native_authority",
            "arm": arm, "snapshot_id": snapshot_id,
            "candidate_ids": list(candidate_ids), "candidate_set_sha256": candidate_hash,
            # Do not leak fingerprints of withheld signals into the model view.
            # The caller keeps all source digests in a separate research audit.
            "input_hashes": visible_hashes,
        }
        view.update({name: deepcopy(inputs[name]) if enabled else None
                    for name, enabled in zip(("geometry", "guidance", "critic"), use)})
        result[arm] = view
    return result


@dataclass(frozen=True)
class CriticSnapshot:
    """Host-computed values, supplied from an explicit original before/after pair."""
    snapshot_id: str
    goal_score: float
    violations: int
    progress_cells: int

    def __post_init__(self) -> None:
        from math import isfinite
        if (type(self.snapshot_id) is not str or not 1 <= len(self.snapshot_id) <= 96 or
                type(self.goal_score) not in (int, float) or not isfinite(self.goal_score)):
            raise ValueError("invalid shadow snapshot")
        if isinstance(self.violations, bool) or not isinstance(self.violations, int) or self.violations < 0:
            raise ValueError("invalid violation count")
        if isinstance(self.progress_cells, bool) or not isinstance(self.progress_cells, int) or self.progress_cells < 0:
            raise ValueError("invalid progress count")


def shadow_verdict(*, before: CriticSnapshot, after: CriticSnapshot,
                   undo_eligible: bool, verified_native_receipt: bool) -> dict:
    """Deterministic reference oracle for KEEP/UNDO/STOP, not a d1 response.

    Eligibility is only a hypothetical control supplied by host. UNDO never
    dispatches history; a later native owner must separately qualify rollback.
    """
    if type(before) is not CriticSnapshot or type(after) is not CriticSnapshot:
        raise ValueError("shadow verdict requires typed before/after snapshots")
    if type(undo_eligible) is not bool or type(verified_native_receipt) is not bool:
        raise ValueError("hypothetical Undo eligibility/receipt flags must be boolean")
    if before.snapshot_id == after.snapshot_id:
        raise ValueError("before and after must be different frozen snapshots")
    choice, cause = reference_verdict(
        {"goal_score": before.goal_score, "violations": before.violations,
         "progress_cells": before.progress_cells},
        {"goal_score": after.goal_score, "violations": after.violations,
         "progress_cells": after.progress_cells},
        undo_eligible=undo_eligible, verified_native_receipt=verified_native_receipt)
    return {
        "protocol": "cybersand.driv3r.d1b-shadow/0",
        "decision": choice, "reason": cause,
        "before": before.snapshot_id, "after": after.snapshot_id,
        "undo_eligible": undo_eligible, "verified_native_receipt": verified_native_receipt,
        "execution": "shadow_only_no_material_rollback",
        "native_material_changed_by_critic": False,
    }


def factorial_arms(*, snapshot_id: str, candidate_ids: list[str], geometry: Mapping,
                   guidance: Mapping, critic: Mapping) -> dict[str, dict]:
    """Formal-design A/B/C/D information availability, not executed A/B results.

    The geometric host baseline is shared by every arm. The independently
    isolated views from matched_arms are diagnostic information ablations.
    """
    isolated = matched_arms(snapshot_id=snapshot_id, candidate_ids=candidate_ids,
                            geometry=geometry, guidance=guidance, critic=critic)
    specifications = {
        "A_geometry": ("geometry_only", False, False),
        "B_geometry_guidance": ("guidance_only", True, False),
        "C_geometry_critic": ("critic_only", False, True),
        "D_geometry_guidance_critic": ("combined", True, True),
    }
    result = {}
    for arm, (origin, include_guidance, include_critic) in specifications.items():
        view = deepcopy(isolated[origin])
        view["arm"] = arm
        view["geometry"] = deepcopy(isolated["geometry_only"]["geometry"])
        view["guidance"] = deepcopy(isolated["combined"]["guidance"]) if include_guidance else None
        view["critic"] = deepcopy(isolated["combined"]["critic"]) if include_critic else None
        # Rebuild visibility after restoring the common host-geometry baseline;
        # never inherit hashes of excluded guidance/critique signals.
        view["input_hashes"] = {
            name: stable_hash(view[name]) for name in ("geometry", "guidance", "critic")
            if view[name] is not None
        }
        result[arm] = view
    return result
