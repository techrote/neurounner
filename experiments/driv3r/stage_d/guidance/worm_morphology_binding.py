"""Source-bound, offline morphology accounting for synthetic worm traces.

This joins the exact same inputs through the independently versioned worm_trace
validator and worm_morphology topology analyzer. The standalone morphology
function explicitly expects an already validated trace and cannot prove native
legality; ONLY this joining wrapper verifies the shared initial cursor, World
frame, cardinal movement, STOP, field bytes, source and scenario identities.

All results are synthetic intentions, never actual materials or model choices.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .worm_trace import audit_synthetic_worm_pair
from .worm_morphology import compare_validated_traces

PROTOCOL = "cybersand.driv3r.d1b-worm-morphology-bound/1"


def _digest(obj: object) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def bind_synthetic_worm_morphology(
        *, guidance_u16le: bytes, width: int, height: int,
        origin_xy: list[int], guidance_sha256: str,
        initial_xy: list[int], initial_world_sha256: str,
        scenario_sha256: str, source_revision: str, seed: int,
        free_trace: list[dict], soft_trace: list[dict],
        max_decisions: int = 36, high_threshold: int = 32768) -> dict:
    """Audit BOTH traces before examining morphology; no native/model authority.

    The explicit signature prevents unrecognized native flags or independent
    field/initial-world inputs from being smuggled into only one analyzer.
    Identical trace digests and pen-contact cardinality are mandatory joins.
    """
    movement = audit_synthetic_worm_pair(
        guidance_u16le=guidance_u16le, width=width, height=height,
        origin_xy=origin_xy, guidance_sha256=guidance_sha256,
        initial_xy=initial_xy, initial_world_sha256=initial_world_sha256,
        scenario_sha256=scenario_sha256, source_revision=source_revision,
        seed=seed, free_trace=free_trace, soft_trace=soft_trace,
        max_decisions=max_decisions, high_threshold=high_threshold)
    morphology = compare_validated_traces(free_trace, soft_trace)
    for arm in ("free", "soft"):
        route = movement["arms"][arm]
        ink = morphology["arms"][arm]
        if (route["trajectory_sha256"] != ink["validated_trace_sha256"] or
                route["pen_down_intent_moves"] != ink["intended_contact_moves"] or
                route["pen_down_intent_reference"]["unique_target_cells"] !=
                ink["distinct_intended_cells"]):
            raise ValueError("cross-analyzer trace or pen-intent custody mismatch")
    return {
        "protocol": PROTOCOL,
        "evidence_class": "source_bound_synthetic_ink_intent_only",
        "source_revision": movement["source_revision"],
        "initial_world_sha256": movement["initial_world_sha256"],
        "scenario_sha256": movement["scenario_sha256"],
        "guidance_sha256": movement["reference_guidance_sha256"],
        "seed": movement["seed"],
        "initial_xy": movement["initial_xy"][:],
        "frame": movement["frame"][:],
        "movement_audit_sha256": _digest(movement),
        "morphology_audit_sha256": _digest(morphology),
        "movement_audit": movement,
        "morphology_audit": morphology,
        "native_material_verified": False,
        "native_undo_executions": 0,
        "model_behavior_verified": False,
        "interpretation": "Bound synthetic legal-motion and ink-intent accounting; "
                          "no native readback, actual d1 steering or causal art result",
    }
