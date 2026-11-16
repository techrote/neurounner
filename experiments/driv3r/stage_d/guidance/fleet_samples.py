"""D-1B offline, four-stream/four-arm Commander input handoff (research only).

The four *sample streams* are NOT resident worm processes. M1 is an immutable
accepted schema pin, not a claim that these previews are full native M1 records.
No native material writes, model inference, permission or Undo authority.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re
import struct

M1_COMMIT = "8fa90de6be209ed0b7bd125039e6ec8b332fb1a6"
M1_SCHEMA_SHA256 = "ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d"
PROTOCOL = "cybersand.driv3r.d1b-fleet-input-samples/1"
ARMS = ("sham", "guidance", "critic", "combined")
VIEWS = ("core", "preference", "geometry", "all")
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z")
OPTIONALS = {
    "soft_attraction/v1": ("preference", "u16le"),
    "soft_avoidance/v1": ("preference", "u16le"),
    "signed_preference/v1": ("preference", "i16le"),
    "signed_distance/v1": ("geometry_metric", "i16le"),
    "flow_vector_xy/v1": ("geometry_metric", "vec2_i16le"),
}
GEOMETRY = frozenset({"clearance_cells", "cursor_xy", "target_xy", "selected_wall_id", "obstacle_count"})


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _digest(obj: object) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _uint(v: object, maximum: int) -> bool:
    return type(v) is int and 0 <= v <= maximum


def _coord(v: object) -> bool:
    return type(v) is list and len(v) == 2 and all(_uint(x, 2**31 - 1) for x in v)


def _channel(semantic: str, values: object, frame: list[int], revision: str, seed: int) -> dict:
    _check(semantic == "positive_scalar/v1" or semantic in OPTIONALS, "unregistered M1 semantic")
    klass, encoding = ("preference", "u16le") if semantic == "positive_scalar/v1" else OPTIONALS[semantic]
    n = frame[2] * frame[3]
    if encoding == "vec2_i16le":
        _check(type(values) is list and len(values) == n and
               all(type(p) is list and len(p) == 2 and
                   all(type(v) is int and -32768 <= v <= 32767 for v in p) for p in values),
               "invalid signed XY vector tile")
        flat = [v for pair in values for v in pair]
    else:
        lo, hi = (0, 65535) if encoding == "u16le" else (-32768, 32767)
        _check(type(values) is list and len(values) == n and
               all(type(v) is int and lo <= v <= hi for v in values), "invalid numeric tile")
        flat = values
    data = struct.pack("<" + ("H" if encoding == "u16le" else "h") * len(flat), *flat)
    return {"semantic_id": semantic, "signal_class": klass, "encoding": encoding,
            "frame": frame[:], "values": deepcopy(values), "sample_bytes": len(data),
            "sample_sha256": sha256(data).hexdigest(),
            "producer_revision": revision, "seed": seed}


def _catalog(items: object) -> tuple[list[dict], str]:
    _check(type(items) is list and 1 <= len(items) <= 128, "1..128 host candidates required")
    result, ids = [], set()
    for entry in items:
        _check(type(entry) is dict and set(entry) ==
               {"id", "destination_xy", "swept_xy", "native_limit_cells"},
               "host legal candidate shape required; no scoring or permissions in catalog")
        cid = entry["id"]
        _check(type(cid) is str and IDENT.fullmatch(cid) is not None and cid not in ids,
               "invalid/duplicate legal ID")
        ids.add(cid)
        swept = entry["swept_xy"]
        _check(_coord(entry["destination_xy"]) and type(swept) is list and
               all(_coord(p) for p in swept) and len({tuple(p) for p in swept}) == len(swept) and
               _uint(entry["native_limit_cells"], 4096) and
               len(swept) <= entry["native_limit_cells"] <= 4096,
               "invalid host-supplied candidate footprint")
        result.append({"id": cid, "destination_xy": entry["destination_xy"][:],
                       "swept_xy": sorted(deepcopy(swept)),
                       "native_limit_cells": entry["native_limit_cells"]})
    result.sort(key=lambda e: e["id"])
    return result, _digest(result)


def _geometry(data: object) -> dict:
    _check(type(data) is dict and set(data) <= GEOMETRY, "host geometry may not carry hidden scores")
    for key, value in data.items():
        if key in ("cursor_xy", "target_xy"):
            _check(_coord(value), "invalid host geometry coordinate")
        elif key == "selected_wall_id":
            _check(type(value) is str and IDENT.fullmatch(value) is not None, "invalid wall ID")
        else:
            _check(_uint(value, 2**31 - 1), "invalid host geometry metric")
    return deepcopy(data)


def _shadow(case: dict) -> dict:
    _check(type(case) is dict and set(case) ==
           {"before", "after", "undo_eligible", "hypothetical_receipt_available"},
           "only post-choice hypothetical critic case accepted")
    a, b = case["before"], case["after"]
    for v in (a, b):
        _check(type(v) is dict and set(v) == {"goal_score", "violations", "progress_cells"} and
               all(type(v[k]) is int and -32768 <= v[k] <= 32767
                   for k in ("goal_score", "progress_cells")) and
               _uint(v["violations"], 32767), "invalid synthetic critic metrics")
    _check(type(case["undo_eligible"]) is bool and
           type(case["hypothetical_receipt_available"]) is bool, "invalid shadow flags")
    regression = b["violations"] > a["violations"] or b["goal_score"] < a["goal_score"]
    progress = (b["progress_cells"] > a["progress_cells"] or
                b["goal_score"] > a["goal_score"] or b["violations"] < a["violations"])
    verdict = ("KEEP" if not regression and progress else
               "UNDO" if case["undo_eligible"] and case["hypothetical_receipt_available"] else "STOP")
    return {"before_after": deepcopy(case), "shadow_verdict": verdict,
            "shadow_only": True, "native_undo_executed": False,
            "native_receipt_verified": False}


def build_stream(source: dict) -> dict:
    """One seed; all arms use exactly the same declared synthetic host state.

    'critic' changes only the separate post-choice shadow analysis; its model
    input is byte-identical to the sham's, likewise combined versus guidance.
    """
    required = {"stream_id", "seed", "source_revision", "snapshot_id", "world_sha256",
                "material_sha256", "frame", "guidance_u16", "host_geometry",
                "legal_candidates", "optional_channels", "critic_cases"}
    _check(type(source) is dict and set(source) == required, "strict synthetic input shape required")
    for key in ("stream_id", "snapshot_id"):
        _check(type(source[key]) is str and IDENT.fullmatch(source[key]) is not None,
               "invalid stream/snapshot ID")
    _check(_uint(source["seed"], 2**32 - 1), "seed must be u32")
    _check(type(source["source_revision"]) is str and COMMIT.fullmatch(source["source_revision"]) is not None,
           "exact source commit required")
    for key in ("world_sha256", "material_sha256"):
        _check(type(source[key]) is str and SHA.fullmatch(source[key]) is not None,
               "World/material SHA-256 required")
    frame = source["frame"]
    _check(type(frame) is list and len(frame) == 4 and
           all(_uint(x, 2**31 - 1) for x in frame[:2]) and
           all(_uint(x, 4096) and x > 0 for x in frame[2:]) and
           frame[2] * frame[3] <= 4096 and
           frame[0] + frame[2] <= 2**31 and frame[1] + frame[3] <= 2**31,
           "invalid bounded u16 tile frame")
    legal, legal_hash = _catalog(source["legal_candidates"])
    geometry = _geometry(source["host_geometry"])
    options = source["optional_channels"]
    _check(type(options) is dict and set(options) <= set(OPTIONALS), "only M1-registered optionals")
    optional = [_channel(semantic, values, frame, source["source_revision"], source["seed"])
                for semantic, values in sorted(options.items())]
    _check(type(source["critic_cases"]) is list and 1 <= len(source["critic_cases"]) <= 8,
           "bounded critic before/after cases required")
    shadow_cases = [_shadow(c) for c in source["critic_cases"]]
    guide = _channel("positive_scalar/v1", source["guidance_u16"], frame,
                     source["source_revision"], source["seed"])
    _check(any(guide["values"]), "real guidance must differ from neutral sham")
    zeros = _channel("positive_scalar/v1", [0] * (frame[2] * frame[3]), frame,
                     source["source_revision"], source["seed"])
    # Keep factual geometry metrics identical in every arm, but neutralize
    # EVERY preference signal for sham/critic. Otherwise the optional "all"
    # view contains active signed attraction/avoidance in its alleged control:
    # a confounded guidance-vs-sham comparison even with a zero scalar core.
    # Neutralization retains each channel's registered M1 encoding, tile
    # dimensions, producer identity and seed while hashing the actual zero
    # sample bytes. The original active signal never enters a sham model view.
    neutral_optional = [
        _channel(d["semantic_id"], [0] * (frame[2] * frame[3]), frame,
                 source["source_revision"], source["seed"])
        if d["signal_class"] == "preference" else d
        for d in optional
    ]
    arms = {}
    for arm in ARMS:
        active = arm in ("guidance", "combined")
        core = guide if active else zeros
        channels = [core] + (optional if active else neutral_optional)
        variants = {}
        for variant in VIEWS:
            used = [d for d in channels if d is core or
                    variant == "all" or
                    (variant == "preference" and d["signal_class"] == "preference") or
                    (variant == "geometry" and d["signal_class"] == "geometry_metric")]
            view = {"protocol": PROTOCOL, "evidence_class": "synthetic_offline_not_native",
                    "snapshot_id": source["snapshot_id"], "world_sha256": source["world_sha256"],
                    "material_sha256": source["material_sha256"],
                    "source_revision": source["source_revision"],
                    "host_geometry": deepcopy(geometry), "legal_candidates": deepcopy(legal),
                    "legal_candidate_set_sha256": legal_hash, "guidance": deepcopy(used)}
            ignored = [{"semantic_id": d["semantic_id"], "sample_sha256": d["sample_sha256"]}
                       for d in channels if d not in used]
            variants[variant] = {"model_view": view, "view_sha256": _digest(view),
                                 "audit_only_ignored": ignored}
        arms[arm] = {"views": variants,
                     "post_choice_shadow_critic": deepcopy(shadow_cases)
                     if arm in ("critic", "combined") else None,
                     "native_material_authority": False,
                     "real_undo_available": False}
    _check(all(arms["sham"]["views"][v]["view_sha256"] ==
               arms["critic"]["views"][v]["view_sha256"] for v in VIEWS),
           "critic must not leak into pre-choice model view")
    _check(all(arms["guidance"]["views"][v]["view_sha256"] ==
               arms["combined"]["views"][v]["view_sha256"] for v in VIEWS),
           "combined must not leak critic into pre-choice model view")
    return {"stream_id": source["stream_id"], "seed": source["seed"],
            "snapshot_id": source["snapshot_id"],
            "host_source_sha256": _digest(source),
            "world_sha256": source["world_sha256"], "material_sha256": source["material_sha256"],
            "legal_candidate_set_sha256": legal_hash, "arms": arms,
            "source_truth": "declared_synthetic_unverified_not_native",
            "active_model_pid": None, "resident_worm_count": 0}


def build_four_streams(sources: list[dict]) -> dict:
    """Require four independent seed/world *sample* leases, not four active worms."""
    _check(type(sources) is list and len(sources) == 4, "exactly four seed sources required")
    rows = [build_stream(src) for src in sources]
    for key in ("stream_id", "seed", "snapshot_id"):
        _check(len({r[key] for r in rows}) == 4, f"duplicate {key} between sample streams")
    # Requiring only unique (World, material) *pairs* permitted aliasing the
    # same World under a different material digest (and vice versa). That is a
    # source-isolation confound when presented as four independent worm Worlds.
    # Both declared source identities must be distinct; original bytes remain
    # UNVERIFIED here and must be checked by the consuming host.
    _check(len({r["world_sha256"] for r in rows}) == 4,
           "duplicate World/material source alias: World")
    _check(len({r["material_sha256"] for r in rows}) == 4,
           "duplicate World/material source alias: material")
    # Equal software revision is required for a same-protocol fleet comparison;
    # different Worlds and seeds do not justify silently mixing host versions.
    _check(len({src["source_revision"] for src in sources}) == 1,
           "mixed source revisions in one matched fleet comparison")
    body = {"protocol": PROTOCOL, "accepted_m1_source": M1_COMMIT,
            "accepted_m1_schema_sha256": M1_SCHEMA_SHA256,
            "evidence_class": "synthetic_offline_inputs_only", "sample_stream_count": 4,
            "resident_worm_count": 0, "native_material_writes": 0,
            "real_undo_executions": 0, "model_inference_calls": 0,
            "streams": rows}
    body["content_sha256_without_self_digest"] = _digest(body)
    return body
