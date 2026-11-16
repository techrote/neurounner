"""Lane-only, synthetic D-1B -> owner-frozen D-COMMON v1 projection.

Does not import, edit or invoke the native backend. Returns real numeric artifacts
and a guidance record plus *audit-only* observation channel-handling fragments;
it does NOT issue full native observations, receipts or permission decisions.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re
import struct

from .information import PROTOCOL as SHADOW_PROTOCOL, _catalog, _hashable

M1_COMMIT = "8fa90de6be209ed0b7bd125039e6ec8b332fb1a6"
M1_SCHEMA_SHA256 = "ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d"
PROTOCOL = "cybersand.driv3r.d-common/1"
CORE = "positive_scalar/v1"
# Exact semantics from the accepted M1 contract's verify.py, lines 28-39.
PRESETS = {
    "soft_attraction/v1": ("preference", "u16le", "normalized_0_1", 1, 65535,
                            "no_extra_attraction", "greater_attraction", None),
    "soft_avoidance/v1": ("preference", "u16le", "normalized_0_1", 1, 65535,
                           "no_soft_avoidance", "greater_soft_avoidance", None),
    "signed_preference/v1": ("preference", "i16le", "normalized_signed", 1, 32767,
                             "neutral", "attraction", "repulsion"),
    "signed_distance/v1": ("geometry_metric", "i16le", "cells", 1, 1,
                           "on_reference_boundary", "outside_reference", "inside_reference"),
    "flow_vector_xy/v1": ("geometry_metric", "vec2_i16le", "world_unit_vector", 1, 32767,
                          "no_direction", "+x_right_+y_down", "-x_left_-y_up"),
}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z")
ARTIFACT_ROOT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")


def _must(valid: bool, message: str) -> None:
    if not valid:
        raise ValueError(message)


def _raw(descriptor: dict) -> bytes:
    encoding = descriptor["encoding"]
    _must(encoding in ("u16le", "i16le", "vec2_i16le"), "unregistered numeric encoding")
    values = descriptor["values"]
    if encoding == "vec2_i16le":
        _must(type(values) is list and all(type(v) is list and len(v) == 2 for v in values),
              "vector values must be explicit XY pairs")
        values = [v for pair in values for v in pair]
    _must(type(values) is list and all(type(v) is int and
          (0 <= v <= 65535 if encoding == "u16le" else -32768 <= v <= 32767)
          for v in values), "invalid numeric sample")
    rect = descriptor["frame"]
    _must(len(values) == rect[2] * rect[3] * (2 if encoding == "vec2_i16le" else 1),
          "numeric sample count does not match tile extent")
    raw = struct.pack("<" + ("H" if encoding == "u16le" else "h") * len(values), *values)
    _must(len(raw) == descriptor["sample_bytes"] and
          sha256(raw).hexdigest() == descriptor["sample_sha256"],
          "source numeric bytes/hash mismatch")
    return raw


def _meta(descriptor: dict) -> dict:
    return {key: val for key, val in descriptor.items() if key != "values"}


def project_m1_guidance(arms: dict, *, canvas_size: tuple[int, int], record_id: str,
                        episode_id: str, sequence: int, label: str, asset_root: str,
                        speckle: dict, exit_below: int, reenter_at_or_above: int,
                        reference_ids: dict[str, str] | None = None) -> dict:
    """Project already-frozen synthetic A1/A2/A3 views; never dispatch them.

    Input is the output of information.make_shadow_arms. Returned artifacts
    are numeric byte strings keyed by safe *relative* paths. A later central
    owner must validate full M1 journals with the pinned schema and evidence.
    """
    _must(type(arms) is dict and set(arms) == {"A1", "A2", "A3"}, "exact A1/A2/A3 arms required")
    _must(type(canvas_size) in (tuple, list) and len(canvas_size) == 2 and
          all(type(v) is int and 1 <= v <= 65536 for v in canvas_size), "invalid M1 canvas size")
    _must(all(type(v) is str and IDENTIFIER.fullmatch(v) for v in (record_id, episode_id)),
          "invalid M1 record/episode identity")
    _must(type(sequence) is int and 0 <= sequence <= 1000000 and
          type(label) is str and 1 <= len(label) <= 512, "invalid M1 record metadata")
    _must(type(asset_root) is str and ARTIFACT_ROOT.fullmatch(asset_root) and
          ".." not in asset_root and "//" not in asset_root and not asset_root.startswith("/"),
          "unsafe relative artifact root")
    _must(type(exit_below) is int and type(reenter_at_or_above) is int and
          0 <= exit_below < reenter_at_or_above <= 65535, "invalid M1 hysteresis")
    _must(type(speckle) is dict and set(speckle) ==
          {"seed", "radius_cells", "centre_density", "hardness", "gradient_length"} and
          type(speckle["seed"]) is int and 0 <= speckle["seed"] <= 2**32 - 1 and
          type(speckle["radius_cells"]) is int and 1 <= speckle["radius_cells"] <= 512 and
          all(type(speckle[k]) in (int, float) and 0 <= speckle[k] <= 1
              for k in ("centre_density", "hardness", "gradient_length")),
          "invalid or incomplete M1 speckle metadata")
    references = reference_ids or {}
    _must(type(references) is dict, "reference_ids must be a mapping")
    canonical = None
    descriptors = {}
    for arm in ("A1", "A2", "A3"):
        pair = arms[arm]
        view, audit = pair["model_view"], pair["audit_only"]
        _must(view["protocol"] == SHADOW_PROTOCOL and
              view["evidence_class"] == "synthetic_pre_m1" and
              audit["protocol"] == SHADOW_PROTOCOL and audit["arm"] == arm and
              audit["unregistered_channels_used"] == [] and
              audit["native_material_authority"] is False and
              audit["native_execution_verified"] is False,
              "native/mismatched source not admissible to synthetic M1 projection")
        _must(audit["legal_candidate_set_sha256"] == view["legal_candidate_set_sha256"],
              "audit legal candidate set hash mismatch")
        # Recompute from the exact host-supplied footprint catalog. A supplied
        # digest alone cannot establish that the catalog is internally coherent.
        _catalog_rows, computed_legal_sha = _catalog(view["legal_candidates"])
        _must(computed_legal_sha == view["legal_candidate_set_sha256"],
              "source legal candidate set hash mismatch")
        common = (view["snapshot_id"], view["world_sha256"], view["source_revision"],
                  view["legal_candidate_set_sha256"], json.dumps(view["host_geometry"], sort_keys=True),
                  json.dumps(view["legal_candidates"], sort_keys=True))
        _must(canonical is None or canonical == common, "arms have different frozen host states")
        canonical = common
        used, ignored = audit["used_channels"], audit["ignored_channels"]
        _must(set(_meta(d)["semantic_id"] for d in used) ==
              set(d["semantic_id"] for d in view["guidance"]), "model-visible channels disagree with audit")
        for item in view["guidance"]:
            _must(_meta(item) in used, "hidden channel metadata leaked/mismatched")
            # Validate EVERY consumed copy against its declared byte/hash
            # provenance, not just the last one retained by all_views below.
            # A1/A2 core tampering was otherwise masked by the A3 copy.
            _raw(item)
        _must(len(used) + len(ignored) <= 9, "too many channels")
        for item in used + ignored:
            semantic = item["semantic_id"]
            _must(semantic not in descriptors or descriptors[semantic]["meta"] == item,
                  "channel data/provenance differs by arm")
            descriptors.setdefault(semantic, {"meta": item})
        actual = [d["semantic_id"] for d in used]
        # Exact membership is finally checked after collecting every arm.
        _must(CORE in actual, "mandatory core omitted")
        _must(len(actual) == len(set(actual)), "duplicate used channel")
    _must(CORE in descriptors and len(descriptors) <= 9, "invalid mandatory core/optionals")
    _must(descriptors[CORE]["meta"]["encoding"] == "u16le", "core must be unsigned u16le")
    for arm in ("A1", "A2", "A3"):
        pair = arms[arm]
        all_meta = pair["audit_only"]["used_channels"] + pair["audit_only"]["ignored_channels"]
        _must(len(all_meta) == len(descriptors) and
              {d["semantic_id"] for d in all_meta} == set(descriptors),
              "missing, extra or duplicated channel audit")
        used = {d["semantic_id"] for d in pair["audit_only"]["used_channels"]}
        expected = {CORE}
        if arm == "A2":
            expected |= {k for k, d in descriptors.items() if k != CORE and d["meta"]["signal_class"] == "preference"}
        if arm == "A3":
            expected |= {k for k, d in descriptors.items() if k != CORE and d["meta"]["signal_class"] == "geometry"}
        _must(used == expected, "arm consumed wrong information set")
        _must(arms[arm]["audit_only"]["model_view_sha256"] ==
              _hashable(arms[arm]["model_view"]),
              "model-view provenance hash mismatch")
    artifacts = {}
    all_views = {d["semantic_id"]: d for a in arms.values() for d in a["model_view"]["guidance"]}
    optional = []
    for semantic, item in sorted(descriptors.items()):
        desc = all_views[semantic]
        meta = item["meta"]
        rect = meta["frame"]
        _must(type(rect) is list and len(rect) == 4 and
              all(type(v) is int for v in rect) and
              0 <= rect[0] <= 65535 and 0 <= rect[1] <= 65535 and
              1 <= rect[2] <= 65536 and 1 <= rect[3] <= 65536 and
              rect[0] + rect[2] <= canvas_size[0] and rect[1] + rect[3] <= canvas_size[1],
              "channel frame extends outside accepted World canvas")
        data = _raw(desc)
        filename = "core.u16le" if semantic == CORE else semantic.replace("/", "_") + "." + meta["encoding"]
        path = asset_root.rstrip("/") + "/" + filename
        artifacts[path] = data
        blob = {"path": path, "sha256": sha256(data).hexdigest(), "bytes": len(data),
                "format": meta["encoding"]}
        if semantic == CORE:
            core = (rect, blob, meta)
            continue
        _must(semantic in PRESETS, "unregistered M1 optional semantic")
        klass, encoding, units, numerator, denominator, zero, positive, negative = PRESETS[semantic]
        _must(meta["signal_class"] == ("geometry" if klass == "geometry_metric" else klass) and
              meta["encoding"] == encoding, "M1 optional signal type mismatch")
        reference = references.get(semantic)
        _must(semantic != "signed_distance/v1" or
              (type(reference) is str and IDENTIFIER.fullmatch(reference)),
              "signed distance requires named boundary reference")
        cid = semantic.replace("/", ":")
        optional.append({
            "channel_id": cid, "signal_class": klass, "semantic_id": semantic,
            "encoding": encoding, "units": units,
            "scale": {"numerator": numerator, "denominator": denominator},
            "zero_meaning": zero, "positive_meaning": positive, "negative_meaning": negative,
            "reference_id": reference, "canvas_size": list(canvas_size),
            "frame": rect[:], "values": blob,
            "producer": {"kind": "derived_geometry" if klass == "geometry_metric" else "procedural",
                         "producer_id": "d1b.source", "revision": meta["producer_revision"],
                         "source_sha256": meta["sample_sha256"],
                         "parameters_sha256": meta["config_sha256"], "seed": meta["seed"]},
            "native_material_authority": False,
        })
    guidance = {
        "protocol": PROTOCOL, "type": "guidance", "record_id": record_id,
        "episode_id": episode_id, "sequence": sequence, "label": label,
        "frame": core[0][:], "source_kind": "manual_scalar",
        "purpose": "soft_positive_preference", "values": core[1],
        "display": None, "encoding": "u16le_higher_is_better_zero_is_neutral",
        "speckle": dict(speckle),
        "hysteresis": {"exit_below": exit_below, "reenter_at_or_above": reenter_at_or_above},
        "native_material_authority": False, "authoring_seed": speckle["seed"],
        "canvas_size": list(canvas_size), "optional_channels": optional,
    }
    handlings = {}
    for arm in ("A1", "A2", "A3"):
        used = {d["semantic_id"] for d in arms[arm]["audit_only"]["used_channels"]}
        ignored = [c for c in optional if c["semantic_id"] not in used]
        handlings[arm] = {
            "unsupported_policy": "ignore_with_audit" if ignored else "reject_unsupported",
            "used_channel_ids": [c["channel_id"] for c in optional if c["semantic_id"] in used],
            "ignored_channels": [
                {"channel_id": c["channel_id"], "semantic_id": c["semantic_id"],
                 "values_sha256": c["values"]["sha256"],
                 "reason": "explicit_information_ablation"} for c in ignored],
        }
    return {"m1_source_commit": M1_COMMIT, "m1_schema_sha256": M1_SCHEMA_SHA256,
            "guidance": guidance, "artifacts": artifacts,
            "observation_channel_handling": handlings,
            "evidence_class": "synthetic_offline_not_native"}
