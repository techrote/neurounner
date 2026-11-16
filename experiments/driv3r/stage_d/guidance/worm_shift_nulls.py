"""Read-only, deterministic shifted-field controls for synthetic line-worm ink.

The input MUST be an independently generated bind_synthetic_worm_morphology
report; this module recomputes its synthetic motion/topology audits but cannot
verify host/native authenticity.
Toroidal translations preserve the field's value histogram, not its alignment
with authored ink destinations. These posthoc nulls are descriptive and cannot
establish causal model steering, visual input, or genuine material edits.
"""
from __future__ import annotations

from hashlib import sha256
from math import gcd
import json
import re
import struct

from .worm_morphology_binding import bind_synthetic_worm_morphology

PROTOCOL = "cybersand.driv3r.d1b-worm-shift-nulls/1"
BOUND = "cybersand.driv3r.d1b-worm-morphology-bound/1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(obj: object) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def _ink_destinations(trace: list[dict], digest: str, expected_count: int) -> list[tuple[int, int]]:
    _require(type(trace) is list and 1 <= len(trace) <= 36 and
             type(digest) is str and HEX64.fullmatch(digest) is not None and
             _digest(trace) == digest, "trajectory seal mismatch")
    # The bound report's original movement validator owns cardinal legality.
    # Recheck input identity and intended contacts here, not native legality.
    seen: set[tuple[int, int]] = set()
    for index, step in enumerate(trace, 1):
        _require(type(step) is dict and set(step) ==
                 {"decision_index", "action", "to_xy", "pen_down"} and
                 type(step["decision_index"]) is int and step["decision_index"] == index and
                 type(step["pen_down"]) is bool and
                 step["action"] in ("MOVE", "STOP"), "invalid sealed trajectory")
        if step["action"] == "STOP":
            _require(step["to_xy"] == [] and not step["pen_down"] and index == len(trace),
                     "invalid sealed STOP")
        else:
            xy = step["to_xy"]
            _require(type(xy) is list and len(xy) == 2 and
                     all(type(v) is int and 0 <= v < 2**31 for v in xy),
                     "invalid sealed ink coordinate")
            if step["pen_down"]:
                seen.add(tuple(xy))
    _require(len(seen) == expected_count, "ink count differs from bound report")
    return sorted(seen)


def _score(points: list[tuple[int, int]], raw: bytes, frame: list[int],
           shift_x: int, shift_y: int) -> dict:
    x0, y0, w, h = frame
    total = 0
    for x, y in points:
        _require(x0 <= x < x0 + w and y0 <= y < y0 + h,
                 "ink outside bound evaluation field")
        ix = (x - x0 - shift_x) % w
        iy = (y - y0 - shift_y) % h
        total += struct.unpack_from("<H", raw, 2 * (iy * w + ix))[0]
    return {"sum_u16": total, "distinct_ink_cells": len(points)}


def _difference(free: dict, soft: dict) -> dict | None:
    f, s = free["distinct_ink_cells"], soft["distinct_ink_cells"]
    if not f or not s:
        return None
    return {"numerator": soft["sum_u16"] * f - free["sum_u16"] * s,
            "denominator": f * s}


def audit_shifted_field_nulls(*, bound_report: dict, guidance_u16le: bytes,
                              free_trace: list[dict], soft_trace: list[dict],
                              max_null_shifts: int = 16) -> dict:
    """Posthoc score of exact same intended ink under deterministic field shifts.

    Reports exact-rational soft-minus-free mean preference on DISTINCT ink
    destinations. Duplicate contacts and pen-up cursor visits cannot inflate it.
    A high observed score relative to the shifted-field nulls is NOT a p-value,
    especially when the field was authored with knowledge of the trajectories.
    """
    _require(type(bound_report) is dict and bound_report.get("protocol") == BOUND and
             bound_report.get("evidence_class") == "source_bound_synthetic_ink_intent_only" and
             bound_report.get("native_material_verified") is False and
             bound_report.get("model_behavior_verified") is False and
             bound_report.get("native_undo_executions") == 0,
             "requires synthetic bound morphology report")
    _require(type(max_null_shifts) is int and 1 <= max_null_shifts <= 32,
             "invalid null-shift budget")
    frame = bound_report.get("frame")
    _require(type(frame) is list and len(frame) == 4 and
             all(type(v) is int for v in frame) and
             0 <= frame[0] < 2**31 and 0 <= frame[1] < 2**31 and
             1 <= frame[2] <= 4096 and 1 <= frame[3] <= 4096 and
             2 <= frame[2] * frame[3] <= 4096 and
             frame[0] + frame[2] <= 2**31 and frame[1] + frame[3] <= 2**31,
             "invalid bounded null-field frame")
    _require(type(guidance_u16le) is bytes and
             len(guidance_u16le) == frame[2] * frame[3] * 2 and
             type(bound_report.get("guidance_sha256")) is str and
             HEX64.fullmatch(bound_report["guidance_sha256"]) is not None and
             sha256(guidance_u16le).hexdigest() == bound_report["guidance_sha256"],
             "field byte provenance mismatch")
    movement, morphology = bound_report.get("movement_audit"), bound_report.get("morphology_audit")
    _require(type(movement) is dict and type(morphology) is dict and
             _digest(movement) == bound_report.get("movement_audit_sha256") and
             _digest(morphology) == bound_report.get("morphology_audit_sha256") and
             movement.get("source_revision") == bound_report.get("source_revision") and
             movement.get("scenario_sha256") == bound_report.get("scenario_sha256") and
             movement.get("initial_world_sha256") == bound_report.get("initial_world_sha256") and
             movement.get("reference_guidance_sha256") == bound_report.get("guidance_sha256") and
             movement.get("frame") == frame and
             movement.get("seed") == bound_report.get("seed") and
             movement.get("initial_xy") == bound_report.get("initial_xy") and
             movement.get("native_material_authority") is False and
             movement.get("model_inference_performed") is False and
             morphology.get("native_material_verified") is False and
             morphology.get("model_behavior_verified") is False,
             "bound subreport custody mismatch")
    _require(type(movement.get("arms")) is dict and
             type(morphology.get("arms")) is dict, "missing bound arms")

    # Hashes prove only that supplied dictionaries match their supplied seals.
    # Rebuild both audits from the actual bytes and traces, rather than trusting
    # an adversary who re-sealed fabricated "legal" motion or morphology.
    try:
        canonical = bind_synthetic_worm_morphology(
            guidance_u16le=guidance_u16le,
            width=frame[2], height=frame[3], origin_xy=frame[:2],
            guidance_sha256=bound_report["guidance_sha256"],
            initial_xy=bound_report["initial_xy"],
            initial_world_sha256=bound_report["initial_world_sha256"],
            scenario_sha256=bound_report["scenario_sha256"],
            source_revision=bound_report["source_revision"],
            seed=bound_report["seed"],
            free_trace=free_trace, soft_trace=soft_trace,
            max_decisions=movement["max_decisions"],
            high_threshold=movement["high_threshold_u16"],
        )
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError("canonical source-bound replay invalid: " + str(exc)) from exc
    _require(bound_report == canonical, "canonical source-bound replay mismatch")
    ink = {}
    for arm, trace in (("free", free_trace), ("soft", soft_trace)):
        route = movement["arms"][arm]
        topo = morphology["arms"][arm]
        _require(route["trajectory_sha256"] == topo["validated_trace_sha256"] and
                 route["pen_down_intent_reference"]["unique_target_cells"] ==
                 topo["distinct_intended_cells"], "prior trace/morphology join mismatch")
        ink[arm] = _ink_destinations(trace, route["trajectory_sha256"],
                                     topo["distinct_intended_cells"])
    w, h = frame[2], frame[3]
    n = w * h
    basis = sha256((bound_report["guidance_sha256"] + ":" +
                    bound_report["scenario_sha256"] + ":" +
                    str(bound_report["seed"])).encode()).digest()
    start = int.from_bytes(basis[:8], "little") % n
    stride = 1 + int.from_bytes(basis[8:16], "little") % (n - 1)
    while gcd(stride, n) != 1:
        stride = (stride % (n - 1)) + 1
    shifts = []
    cursor = start
    for _ in range(n):
        cursor = (cursor + stride) % n
        if cursor:
            shifts.append((cursor % w, cursor // w))
        if len(shifts) >= min(max_null_shifts, n - 1):
            break
    def measure(dx: int, dy: int) -> dict:
        free = _score(ink["free"], guidance_u16le, frame, dx, dy)
        soft = _score(ink["soft"], guidance_u16le, frame, dx, dy)
        return {"translation_xy": [dx, dy], "free": free, "soft": soft,
                "soft_minus_free_mean_unique_ink_u16": _difference(free, soft)}
    observed = measure(0, 0)
    nulls = [measure(x, y) for x, y in shifts]
    reference = observed["soft_minus_free_mean_unique_ink_u16"]
    ge = None
    if reference is not None:
        ge = sum((row["soft_minus_free_mean_unique_ink_u16"]["numerator"] *
                  reference["denominator"] >= reference["numerator"] *
                  row["soft_minus_free_mean_unique_ink_u16"]["denominator"])
                 for row in nulls)
    payload = {
        "protocol": PROTOCOL, "evidence_class": "synthetic_posthoc_field_translation_control_only",
        "bound_movement_sha256": bound_report["movement_audit_sha256"],
        "bound_morphology_sha256": bound_report["morphology_audit_sha256"],
        "source_revision": bound_report["source_revision"],
        "scenario_sha256": bound_report["scenario_sha256"],
        "initial_world_sha256": bound_report["initial_world_sha256"],
        "guidance_sha256": bound_report["guidance_sha256"],
        "free_trace_sha256": movement["arms"]["free"]["trajectory_sha256"],
        "soft_trace_sha256": movement["arms"]["soft"]["trajectory_sha256"],
        "frame": frame[:], "shift_design": "deterministic_permuted_nonzero_torus_translations_v1",
        "null_count": len(nulls), "observed": observed, "nulls": nulls,
        "null_scores_at_least_observed_count": ge,
        "null_comparison_defined": ge is not None,
        "native_material_verified": False, "model_behavior_verified": False,
        "interpretation": "Posthoc distinct intended ink preference under histogram-preserving field translations; not a held-out test, causal experiment, significance estimate or native/model result",
    }
    return {**payload, "audit_sha256": _digest(payload)}
