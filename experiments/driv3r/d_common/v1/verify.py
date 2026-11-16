#!/usr/bin/env python3
"""D-COMMON-001 schema + cross-record semantic and byte-custody checker.

This checker cannot independently establish that a Godot-native execution took
place. Real native qualification additionally requires authentic World receipts,
replay and a source-matched independent runtime gate. Synthetic examples must
not be mislabelled as native success.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent
SCHEMA = ROOT / "interface.schema.json"
EXAMPLES = ROOT / "examples.json"
PROTOCOL = "cybersand.driv3r.d-common/1"
MAX_GUIDANCE_TILE_CELLS = 16_777_216
MAX_LOCAL_CROP_CELLS = 4_194_304
MAX_ARTIFACT_BYTES = 134_217_728
MAX_OPTIONAL_GUIDANCE_BYTES = 134_217_728
# Built-in semantics are normative wire meanings, not model instructions.
# Unknown experimental IDs require a separately registered consumer, or explicit audited omission.
CHANNEL_PRESETS = {
    "soft_attraction/v1": ("preference","u16le","normalized_0_1",1,65535,
                           "no_extra_attraction","greater_attraction",None,False),
    "soft_avoidance/v1": ("preference","u16le","normalized_0_1",1,65535,
                          "no_soft_avoidance","greater_soft_avoidance",None,False),
    "signed_preference/v1": ("preference","i16le","normalized_signed",1,32767,
                             "neutral","attraction","repulsion",False),
    "signed_distance/v1": ("geometry_metric","i16le","cells",1,1,
                           "on_reference_boundary","outside_reference","inside_reference",True),
    "flow_vector_xy/v1": ("geometry_metric","vec2_i16le","world_unit_vector",1,32767,
                           "no_direction","+x_right_+y_down","-x_left_-y_up",False),
}
ENCODED_CELL_BYTES = {"u16le":2,"i16le":2,"vec2_i16le":4}



class ContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def records_from_examples() -> list[dict]:
    return _load(EXAMPLES)["fixtures"]


def _world_size(world: dict) -> tuple[int, int]:
    return world["width_cells"], world["height_cells"]


def _inside(point: list[int], world: dict) -> bool:
    return 0 <= point[0] < world["width_cells"] and 0 <= point[1] < world["height_cells"]


def _rect_fits(rect: list[int], size: tuple[int, int]) -> bool:
    x, y, w, h = rect
    return 0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= size[0] and y + h <= size[1]


def _in_rect(point: list[int], rect: list[int]) -> bool:
    x, y, w, h = rect
    return x <= point[0] < x + w and y <= point[1] < y + h


def _same_world_dimensions(before: dict, after: dict) -> bool:
    return _world_size(before) == _world_size(after) and before["id"] == after["id"]


def _artifacts(record: dict) -> list[dict]:
    name = record["type"]
    if name == "observation":
        return [record["local_material"]] + [view["artifact"] for view in record["feedback_views"]]
    if name == "guidance":
        return ([record["values"]] +
                ([record["display"]] if record["display"] is not None else []) +
                [channel["values"] for channel in record["optional_channels"]])
    if name == "action":
        return ([record["native_receipt_artifact"]] if record["native_receipt_artifact"] is not None else [])
    if name == "reversal":
        return ([record["native_undo_receipt_artifact"]]
                if record["native_undo_receipt_artifact"] is not None else [])
    if name == "provenance":
        return (([record["native"]] if record.get("native") is not None else []) +
                record.get("native_receipts", []) + record["artifacts"])
    return []



def _validate_guidance_channel(channel: dict, canvas: tuple[int, int]) -> None:
    _require(tuple(channel["canvas_size"]) == canvas,
             "Optional channel canvas size differs from positive scalar baseline")
    _require(_rect_fits(channel["frame"], canvas),
             "Optional channel footprint extends outside its canvas")
    _, _, w, h = channel["frame"]
    _require(w * h <= MAX_GUIDANCE_TILE_CELLS, "Optional channel tile too large")
    encoding = channel["encoding"]
    blob = channel["values"]
    _require(blob["format"] == encoding, "Channel encoding and artifact format mismatch")
    _require(blob["bytes"] == w * h * ENCODED_CELL_BYTES[encoding],
             "Optional channel byte count is inconsistent with encoding/extent")
    _require(channel["native_material_authority"] is False,
             "Guidance channel cannot grant native material authority")
    semantic = channel["semantic_id"]
    if semantic in CHANNEL_PRESETS:
        (signal_class, expected_encoding, units, num, den,
         zero, positive, negative, requires_reference) = CHANNEL_PRESETS[semantic]
        _require((channel["signal_class"], encoding, channel["units"],
                  channel["scale"]["numerator"], channel["scale"]["denominator"],
                  channel["zero_meaning"], channel["positive_meaning"],
                  channel["negative_meaning"]) ==
                 (signal_class, expected_encoding, units, num, den,
                  zero, positive, negative),
                 "Built-in channel semantics changed without an interface version bump: " + semantic)
        if requires_reference:
            _require(channel["reference_id"] is not None,
                     "Signed-distance channel requires a named reference boundary")
    else:
        _require(semantic.startswith("experimental."),
                 "Unknown built-in semantic ID; do not silently reinterpret")


def compare_initial_matched_actions(arms: dict[str, list[dict]]) -> dict:
    """Pure contract-level paired audit, NOT native execution or fitness.

    Compare the host-legal candidate-set digest and equivalent native material
    state/cursor BEFORE the first different selected ID. Experimental guide
    content, model prompts, and full candidate-presentation hashes may differ.
    Choice divergence is only a proxy; first *consequential* material/trajectory
    divergence requires separate native readbacks. After choices diverge, the
    host must recompute legal sets independently for the now-different Worlds.
    """
    _require(len(arms) >= 2 and all(type(v) is list and v for v in arms.values()),
             "At least two nonempty matched action traces required")
    names = sorted(arms)
    n = min(len(arms[name]) for name in names)
    first = None
    for i in range(n):
        rows = [arms[name][i] for name in names]
        _require(all(row["type"] == "action" and row["sequence"] >= 0 for row in rows),
                 "Matched traces require action records")
        def state_key(row: dict) -> tuple:
            world = row["world_precondition"]
            # Independent Worlds have different IDs/epochs; compare observed bytes
            # and cursor, not process-local identity or the guide representation.
            return (world["width_cells"],world["height_cells"],
                    world["material_sha256"],world["world_material_sha256"],
                    tuple(row["cursor_before"]))
        _require(all(state_key(row) == state_key(rows[0]) for row in rows[1:]),
                 "Native geometry/cursor diverged before first differing choice")
        _require(all(row["legal_candidate_set_sha256"] ==
                     rows[0]["legal_candidate_set_sha256"] for row in rows[1:]),
                 "Unequal legal recovery candidates on the same paired native state")
        if len({row["candidate_id"] for row in rows}) > 1:
            first = i
            break
    return {"first_choice_divergence_index":first,
            "common_prefix_actions":first if first is not None else n,
            "paired_actions_examined":(first + 1) if first is not None else n,
            "legal_candidate_catalog_equal_until_divergence":True,
            "requires_native_effects_to_identify_consequential_divergence":True}

def validate_bundle(records: list[dict], *, evidence_root: Path | None = None,
                    supported_semantics: set[str] | None = None) -> dict[str, Any]:
    """Validate five record types and cross-record semantics.

    Synthetic records may omit source/native artifacts and have no native
    execution claim. A record of actual committed material edits MUST be
    accompanied by native provenance and byte-checking against evidence_root.
    Byte-custody remains weaker than independent real-native qualification.
    """
    schema_bytes = SCHEMA.read_bytes()
    schema = json.loads(schema_bytes)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    supported_semantics = set(CHANNEL_PRESETS) if supported_semantics is None else set(supported_semantics)
    _require(type(records) is list and bool(records), "Expected a nonempty record list")
    indexed: dict[str, dict] = {}
    episodes: dict[str, list[int]] = {}
    evidence_classes: dict[str, set[str]] = {}
    receipts_by_episode: dict[str, set[tuple[str, str]]] = {}
    material_mutation_by_episode: dict[str, bool] = {}
    for row in records:
        validator.validate(row)
        _require(row["protocol"] == PROTOCOL, "Wrong protocol")
        rid, eid = row["record_id"], row["episode_id"]
        _require(rid not in indexed, "Duplicate record identity")
        indexed[rid] = row
        episodes.setdefault(eid, []).append(row["sequence"])
        if row["type"] == "provenance":
            evidence_classes.setdefault(eid, set()).add(row["evidence_class"])
        if row["type"] == "action" and row["execution_status"] in ("committed", "partial") and row["brush_state"] != "up":
            material_mutation_by_episode[eid] = True
        if row["type"] == "reversal" and row["status"] == "applied" and row["mechanism"] == "UNDO_MATERIAL_EDIT":
            material_mutation_by_episode[eid] = True
    for eid, seq in episodes.items():
        _require(seq == sorted(set(seq)), f"Nonmonotonic or duplicate sequence in {eid}")
    for row in records:
        name, eid = row["type"], row["episode_id"]

        def resolve(ref: str | None, expected: str | None = None) -> dict | None:
            if ref is None:
                return None
            _require(ref in indexed, f"Missing referenced record {ref}")
            target = indexed[ref]
            _require(target["episode_id"] == eid, "Cross-episode reference")
            if expected is not None:
                _require(target["type"] == expected, f"Wrong referenced record type for {ref}")
            _require(target["sequence"] < row["sequence"], "Reference points forward in journal")
            return target

        if name == "guidance":
            _require(_rect_fits(row["frame"], tuple(row["canvas_size"])), "Guidance frame out of canvas")
            _, _, w, h = row["frame"]
            _require(w * h <= MAX_GUIDANCE_TILE_CELLS, "Guidance tile exceeds bounded experimental size")
            _require(row["values"]["format"] == "u16le", "Guidance needs lossless u16le baseline")
            _require(row["values"]["bytes"] == 2 * w * h, "Guidance raster size mismatch")
            _require(row["hysteresis"]["exit_below"] < row["hysteresis"]["reenter_at_or_above"],
                     "Guidance exit threshold must precede reentry threshold")
            _require(row["native_material_authority"] is False, "Guidance must not grant material authority")
            if row["display"] is not None:
                _require(row["display"]["format"] == "png", "Speckle display should be PNG")
            channel_ids = set()
            total_channel_bytes = 0
            for channel in row["optional_channels"]:
                _require(channel["channel_id"] not in channel_ids,
                         "Duplicate optional channel ID")
                channel_ids.add(channel["channel_id"])
                _validate_guidance_channel(channel, tuple(row["canvas_size"]))
                total_channel_bytes += channel["values"]["bytes"]
            _require(total_channel_bytes <= MAX_OPTIONAL_GUIDANCE_BYTES,
                     "Optional guidance channels exceed bounded byte budget")
        elif name == "observation":
            guidance = resolve(row["guidance_ref"], "guidance")
            resolve(row["last_action_ref"], "action")
            world = row["world"]
            size = _world_size(world)
            _require(_rect_fits(row["rules"]["focus"], size), "Focus exceeds current World")
            _require(_rect_fits(row["local_crop"], size), "Native crop exceeds World")
            _require(_inside(row["cursor"], world), "Cursor outside current World")
            _require(_in_rect(row["cursor"], row["rules"]["focus"]), "Cursor outside host focus")
            if row["target"] is not None:
                _require(_inside(row["target"], world), "Target outside current World")
            handling = row["guidance_channel_handling"]
            used = handling["used_channel_ids"]
            ignored = handling["ignored_channels"]
            declared = {channel["channel_id"]: channel for channel in guidance["optional_channels"]} if guidance else {}
            used_set = set(used)
            ignored_ids = [item["channel_id"] for item in ignored]
            ignored_set = set(ignored_ids)
            _require(len(ignored_ids) == len(ignored_set) and
                     not used_set.intersection(ignored_set),
                     "Duplicate or both-consumed-and-ignored guidance channel")
            _require(used_set.union(ignored_set) == set(declared),
                     "Every declared optional channel needs an explicit usage decision")
            _require(handling["unsupported_policy"] == "ignore_with_audit" or not ignored,
                     "Reject-unsupported policy cannot ignore channels")
            for item in ignored:
                target = declared[item["channel_id"]]
                _require(item["semantic_id"] == target["semantic_id"] and
                         item["values_sha256"] == target["values"]["sha256"],
                         "Ignored channel lacks exact immutable semantic/data identity")
            for channel_id in used:
                target = declared[channel_id]
                _require(target["semantic_id"] in supported_semantics,
                         "Unregistered optional semantic ID claimed as consumed")
            for item in ignored:
                if item["reason"] == "unsupported_semantics":
                    _require(declared[item["channel_id"]]["semantic_id"] not in supported_semantics,
                             "Supported channel cannot be mislabelled unsupported")
            if guidance is not None:
                _require(tuple(guidance["canvas_size"]) == size,
                         "Guidance canvas dimensions differ from observed World")
            _, _, w, h = row["local_crop"]
            _require(w * h <= MAX_LOCAL_CROP_CELLS, "Material crop exceeds bounded observation budget")
            _require(row["local_material"]["format"] == "i32le", "Native crop must be signed i32le")
            _require(row["local_material"]["bytes"] == 4 * w * h, "Native crop byte extent mismatch")
            fx, fy, fw, fh = row["rules"]["focus"]
            x, y, cw, ch = row["local_crop"]
            _require(x >= fx and y >= fy and x + cw <= fx + fw and y + ch <= fy + fh,
                     "Native crop exceeds focus")
            _require(set(row["rules"]["allowed_materials"]).isdisjoint(
                row["rules"]["forbidden_materials"]), "Hard material sets overlap")
            for rect in row["rules"]["protected_rectangles"]:
                _require(_rect_fits(rect, size), "Protected region exceeds actual World")
            for view in row["feedback_views"]:
                _require(_world_size(view["source_world"]) == size,
                         "Image witness dimensions mismatch World")
                _require(view["artifact"]["format"] == "png", "Image witness not PNG")
        elif name == "action":
            source = resolve(row["observation_ref"], "observation")
            resolve(row["guidance_ref"], "guidance")
            world = row["world_precondition"]
            _require(source["world"] == world, "Action has stale World fence")
            _require(source["guidance_ref"] == row["guidance_ref"], "Action guidance mismatch")
            _require(source["cursor"] == row["cursor_before"], "Action not at observed cursor")
            focus = source["rules"]["focus"]
            for point in [row["cursor_before"], row["cursor_after"], *row["path"]]:
                _require(_inside(point, world) and _in_rect(point, focus),
                         "Action swept cursor path outside World/focus")
            if row["path"]:
                _require(row["path"][0] == row["cursor_before"] and
                         row["path"][-1] == row["cursor_after"], "Action path endpoints inconsistent")
            else:
                _require(row["cursor_before"] == row["cursor_after"],
                         "No path cannot move cursor")
            if row["mode"] in ("stop", "no_paint_plan"):
                _require(row["brush_state"] == "up" and row["operation_sha256"] is None,
                         "No-paint action carries material operation")
                _require(row["cursor_after"] == row["cursor_before"],
                         "No-paint plan or stop moved cursor")
            if row["mode"] == "add_material":
                _require(row["brush_state"] == "add", "Add action requires add brush")
            if row["mode"] == "erase_material":
                _require(row["brush_state"] == "erase", "Erase action requires erase brush")
            if row["brush_state"] == "up":
                _require(row["operation_sha256"] is None, "Brush-up path carries native edit")
            else:
                _require(row["operation_sha256"] is not None,
                         "Painting or erasure needs exact native operation digest")
            if row["execution_status"] in ("candidate", "selected"):
                _require(row["native_receipt_ref"] is None and
                         row["native_receipt_artifact"] is None and
                         row["world_postcondition"] is None,
                         "Unexecuted candidate claims native receipt/post-state")
            if row["execution_status"] in ("committed", "partial"):
                _require(row["world_postcondition"] is not None,
                         "Committed/partial action missing post-state")
                _require(_same_world_dimensions(world, row["world_postcondition"]),
                         "Committed action silently changed World shape/identity")
                if row["brush_state"] != "up":
                    _require(row["native_receipt_ref"] is not None and
                             row["native_receipt_artifact"] is not None,
                             "Committed/partial material action requires exact native receipt artifact")
                    _require(row["native_receipt_artifact"]["format"] == "json",
                             "Native receipt artifact must be JSON")
                    receipts_by_episode.setdefault(eid, set()).add(
                        (row["native_receipt_artifact"]["path"], row["native_receipt_artifact"]["sha256"]))
            if row["brush_state"] != "up" or row["mode"] == "cursor_move":
                _require(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                             for a, b in zip(row["path"], row["path"][1:])),
                         "Physical travel or paint path is not cardinal-contiguous")
        elif name == "reversal":
            previous_action = resolve(row["source_action_ref"], "action")
            resolve(row["history_anchor_ref"])
            if previous_action["world_postcondition"] is not None:
                _require(previous_action["world_postcondition"] == row["before_world"],
                         "Reversal is not fenced to preceding action post-state")
            _require(row["brush_lifted"] is True, "Reversal must lift brush")
            cursor_only = row["mechanism"] != "UNDO_MATERIAL_EDIT"
            _require(row["preserve_material_edits"] is cursor_only,
                     "Wrong reversal persistence semantics")
            _require(_inside(row["cursor_before"], row["before_world"]) and
                     _inside(row["cursor_after"], row["before_world"]),
                     "Reversal cursor out of current World")
            if row["status"] == "applied":
                _require(row["after_world"] is not None, "Applied reversal missing post-state")
                _require(_same_world_dimensions(row["before_world"], row["after_world"]),
                         "Reversal changed World dimensions or identity")
                if cursor_only:
                    _require(row["before_world"]["world_material_sha256"] ==
                             row["after_world"]["world_material_sha256"] and
                             row["before_world"]["material_sha256"] ==
                             row["after_world"]["material_sha256"] and
                             row["actual_material_cells_changed"] == 0 and
                             row["native_undo_receipt_ref"] is None and
                             row["native_undo_receipt_artifact"] is None,
                             "Cursor-only reversal modified material or invented Undo receipt")
                else:
                    _require(row["native_undo_receipt_ref"] is not None and
                             row["native_undo_receipt_artifact"] is not None and
                             row["native_undo_receipt_artifact"]["format"] == "json",
                             "Actual Undo requires hashed JSON native receipt")
                    _require(row["actual_material_cells_changed"] > 0 and
                             row["before_world"]["world_material_sha256"] !=
                             row["after_world"]["world_material_sha256"],
                             "Applied native Undo did not reverse actual material")
                    receipts_by_episode.setdefault(eid, set()).add(
                        (row["native_undo_receipt_artifact"]["path"],
                         row["native_undo_receipt_artifact"]["sha256"]))
            if row["status"] in ("proposed", "refused", "cancelled"):
                _require(row["actual_material_cells_changed"] == 0 and
                         row["native_undo_receipt_ref"] is None and
                         row["native_undo_receipt_artifact"] is None,
                         "Non-applied reversal claims an Undo/material mutation")
            if row["status"] in ("refused", "cancelled"):
                _require(row["cursor_after"] == row["cursor_before"],
                         "Rejected reversal unexpectedly moved cursor")
                if row["after_world"] is not None:
                    _require(row["after_world"] == row["before_world"],
                             "Rejected reversal unexpectedly changed World")
        elif name == "provenance":
            resolve(row["guidance_ref"], "guidance")
            for ref in row["episode_record_refs"]:
                resolve(ref)
            _require(len(row["episode_record_refs"]) == len(set(row["episode_record_refs"])),
                     "Repeated event in provenance")
            _require([indexed[ref]["sequence"] for ref in row["episode_record_refs"]] ==
                     sorted(indexed[ref]["sequence"] for ref in row["episode_record_refs"]),
                     "Provenance original events are out of order")
            if row["evidence_class"] == "native":
                _require(evidence_root is not None, "Native claim requires evidence byte custody")
                _require(row["interface_schema_sha256"] ==
                         hashlib.sha256(schema_bytes).hexdigest(),
                         "Native provenance schema SHA mismatch")
                if row["final_world"] is not None:
                    _require(row["native"]["format"] == "i32le" and
                             row["native"]["bytes"] == 4 *
                             row["final_world"]["width_cells"] *
                             row["final_world"]["height_cells"],
                             "Native full World byte extent does not match dimensions")
                    _require(row["native"]["sha256"] == row["final_world"]["world_material_sha256"],
                             "Native full World artifact hash differs from final World identity")
                included = {(blob["path"], blob["sha256"])
                            for blob in row["native_receipts"]}
                _require(receipts_by_episode.get(eid, set()).issubset(included),
                         "Native provenance omits a changed-material receipt artifact")
            else:
                _require(row["replay"] == "not_attempted",
                         "Synthetic result cannot claim native replay")
    for eid, has_material_mutation in material_mutation_by_episode.items():
        if has_material_mutation:
            _require(evidence_classes.get(eid) == {"native"},
                     "Actual material edits cannot be labelled synthetic or lack native provenance")
    if evidence_root is not None:
        base = evidence_root.resolve(strict=True)
        seen: dict[Path, str] = {}
        for row in records:
            for artifact in _artifacts(row):
                _require(artifact["bytes"] <= MAX_ARTIFACT_BYTES,
                         "Excessive artifact custody request")
                candidate = (base / artifact["path"]).resolve(strict=True)
                _require(candidate.is_relative_to(base) and candidate.is_file(),
                         "Evidence path escapes checked root")
                data = candidate.read_bytes()
                _require(len(data) == artifact["bytes"] and
                         hashlib.sha256(data).hexdigest() == artifact["sha256"],
                         "Evidence artifact custody failure: " + artifact["path"])
                if candidate in seen:
                    _require(seen[candidate] == artifact["sha256"],
                             "Same path assigned conflicting SHA identities")
                seen[candidate] = artifact["sha256"]
    return {"validated": len(records),
            "types": sorted({r["type"] for r in records}),
            "evidence_bytes_verified": evidence_root is not None,
            "schema_sha256": hashlib.sha256(schema_bytes).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(validate_bundle(records_from_examples()), sort_keys=True))
