"""Read-only retrospective of source-bound #611 native worm artifacts.

This D-1B researcher never invokes a model, Godot, native tools, or Undo.
It verifies archived host execution receipts and numeric scalar samples before
producing descriptive, NOT causal, cross-arm feedback measurements.

Usage: python3 -m experiments.driv3r.stage_d.guidance.native_pair_shadow FREE.zip SOFT.zip
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
from zipfile import ZipFile

from experiments.driv3r.backend import BackendRefusal, parse_choice_response

PROTOCOL = "cybersand.driv3r.d1b-native-pair-retrospective/1"
M1_COMMIT = "8fa90de6be209ed0b7bd125039e6ec8b332fb1a6"
SOURCE = "313b836b3c9ce2c60e4659e3763df140f84347a6"
MOVES = {"MOVE_N": (0, -1), "MOVE_S": (0, 1), "MOVE_E": (1, 0), "MOVE_W": (-1, 0)}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _read_json(z: ZipFile, name: str) -> dict:
    # No extraction and no unbounded external paths. Native archives have a
    # fixed, source-owned structure. Missing or corrupt members fail closed.
    info = z.getinfo(name)
    _require(info.file_size <= 2_000_000, "oversized native JSON member")
    return json.loads(z.read(info))


def _world_sha(observation: dict) -> str:
    return observation["world_material_sha256"]


def _guide_at(data: bytes, guide: dict, xy: list[int]) -> int:
    x, y, w, h = guide["frame"]
    if not (x <= xy[0] < x + w and y <= xy[1] < y + h):
        return 0  # Out of the explicitly authored overlay; not a hard mask.
    return struct.unpack_from("<H", data, 2 * ((xy[1] - y) * w + (xy[0] - x)))[0]


def _load_guide(z: ZipFile) -> tuple[dict, bytes]:
    g = _read_json(z, "execution/original-guide.json")
    _require(g["protocol"] == "cybersand.driv3r.d-common/1" and
             g["encoding"] == "u16le_higher_is_better_zero_is_neutral" and
             g["native_material_authority"] is False and
             g["type"] == "guidance" and g["purpose"] == "soft_positive_preference", "invalid guide semantics")
    x, y, w, h = g["frame"]
    _require(all(type(n) is int for n in (x, y, w, h)) and
             0 <= x < 512 and 0 <= y < 512 and 1 <= w <= 512 - x and 1 <= h <= 512 - y and
             w * h <= 4096, "invalid bounded guide frame")
    desc = g["values"]
    _require(desc["path"] == "guide/positive.u16le" and desc["format"] == "u16le" and
             desc["bytes"] == 2 * w * h, "invalid guide payload descriptor")
    data = z.read("execution/guide/positive.u16le")
    _require(len(data) == desc["bytes"] and sha256(data).hexdigest() == desc["sha256"],
             "guide byte custody mismatch")
    return g, data


def _events(z: ZipFile) -> list[tuple[dict, dict]]:
    request_names = sorted(n for n in z.namelist()
                           if n.startswith("execution/native/request-") and n.endswith(".json"))
    events = []
    for name in request_names:
        num = name.removeprefix("execution/native/request-").removesuffix(".json")
        req = _read_json(z, name)
        resp = _read_json(z, f"execution/native/response-{num}.json")
        _require(req["request_id"] == resp["request_id"] == int(num), "request/response mismatch")
        if req["kind"] in ("apply", "m3_cursor_move"):
            _require(resp["fatal"] == "" and resp["receipt"]["readback_matches"] is True,
                     "invalid native receipt")
            events.append((req, resp))
    return events


def _verify_world_readback(z: ZipFile, req: dict, resp: dict,
                           expected_before_sha: str, expected_after_sha: str,
                           changed_xy: list[int] | None) -> None:
    """Check *actual archived full-World bytes*, not just self-reported receipt flags.

    One model MOVE writes one 0->58 cell. A logical lifted-cursor step
    must preserve all 262144 material IDs. Neither is an Undo operation.
    """
    num = req["request_id"]
    base = f"execution/native/readback-{num:06d}"
    before_info = z.getinfo(base + "-before.world.i32le")
    after_info = z.getinfo(base + "-after.world.i32le")
    _require(before_info.file_size == after_info.file_size == 512 * 512 * 4,
             "full native World readback has wrong byte count")
    before = z.read(before_info)
    after = z.read(after_info)
    _require(sha256(before).hexdigest() == expected_before_sha and
             sha256(after).hexdigest() == expected_after_sha and
             resp["observation"]["world_material_path"] ==
             f"readback-{num:06d}-after.world.i32le", "full-World native source bytes mismatch")
    if changed_xy is None:
        _require(before == after, "lifted cursor changed real World material")
    else:
        x, y = changed_xy
        _require(all(type(v) is int and 0 <= v < 512 for v in (x, y)),
                 "native edit destination outside full World")
        offset = 4 * (y * 512 + x)
        _require(before[:offset] == after[:offset] and
                 before[offset+4:] == after[offset+4:] and
                 struct.unpack_from("<i", before, offset)[0] == 0 and
                 struct.unpack_from("<i", after, offset)[0] == 58,
                 "native material delta differs from sole expected 0-to-58 cell")


def _canonical_digest(value: object) -> str:
    """Exact original #611 JSON digest; independent of claimed archive digest."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode("utf-8")).hexdigest()


def _verify_original_model_choice(turn: dict) -> None:
    """Verify raw Q8 choice and every offered move; no model/native execution."""
    state = turn["model_state"]
    offered = turn["exact_candidate_ids_in_presented_order"]
    _require(type(offered) is list and len(offered) >= 2 and
             len(offered) == len(set(offered)) and
             all(type(x) is str and x for x in offered) and
             state["eligible_moves"] == offered,
             "archived presented/eligible candidate list mismatch")
    # The max-scent statistic is invalid if a higher-scent LEGAL move has
    # disappeared from the archived hints even if all remaining hints match.
    _require(set(state["candidate_hints"]) == {x for x in offered if x in MOVES},
             "missing or foreign legal movement hint")
    questions = turn["questions"]
    _require(questions["worm_move"]["type"] == "choice" and
             set(questions["worm_move"]["criteria"]) == set(offered),
             "model criteria do not match offered legal candidate IDs")
    _require(turn["presented_order_sha256"] == _canonical_digest(offered) and
             turn["presentation_sha256"] == _canonical_digest({
                 "state": state, "questions": questions, "ordered_candidates": offered}),
             "original model presentation SHA-256 mismatch")
    original = turn["original_response"]
    _require(original["status"] == "ok", "original model response not successful")
    try:
        parsed = parse_choice_response(original["raw_response"], "worm_move", offered)
    except (BackendRefusal, KeyError, TypeError, ValueError) as exc:
        raise ValueError("original Q8 response invalid") from exc
    _require(parsed == turn["parsed_choice"] and
             parsed["candidate_id"] == turn["chosen_candidate"],
             "archived chosen ID differs from original Q8 response")


def _verify_recovery_catalog(state: dict, offered: list[str], cursor: list[int],
                             history: list[list[int]], edits: int, recoveries: int) -> None:
    """Rebuild archived host recovery eligibility, without trusting model hints."""
    _require(history and history[-1] == cursor and
             state["recent_positions"] == history[-7:] and
             state["already_inked_cells_this_run"] == len(history) - 1,
             "archived cursor history/model context mismatch")
    can_backtrack = (edits < 36 and recoveries < 8 and len(history) > 2 and
                     sum(abs(a - b) for a, b in zip(cursor, history[-2])) == 1)
    can_relocate = (edits < 36 and recoveries < 8 and len(history) > 5 and
                    cursor != [12, 24])
    _require(("BACKTRACK" in offered) == can_backtrack and
             ("RELOCATE_START" in offered) == can_relocate,
             "model menu disagrees with original host recovery legality")


def _advance_replayed_history(history: list[list[int]], cursor: list[int],
                              choice: str, destination: list[int]) -> None:
    """Enforce the exact original native worm history transitions."""
    _require(history and history[-1] == cursor, "replay history lost current cursor")
    if choice in MOVES:
        delta = MOVES[choice]
        _require(destination == [cursor[0] + delta[0], cursor[1] + delta[1]],
                 "replayed paint did not follow chosen cardinal control")
        history.append(destination[:])
    elif choice == "BACKTRACK":
        _require(len(history) > 2 and destination == history[-2] and
                 sum(abs(a - b) for a, b in zip(cursor, history[-2])) == 1,
                 "backtrack did not return to legal previous host cursor")
        history.pop()
    elif choice == "RELOCATE_START":
        _require(len(history) > 5 and cursor != [12, 24] and destination == [12, 24],
                 "branch relocation did not return to original saved birth cursor")
        history.append(destination[:])
    else:
        raise ValueError("unsupported historical worm control")


def audit_arm(path: Path, expected_mode: str, expected_decisions: int = 36) -> dict:
    """Bind model-selected legal controls to actual archived native material changes."""
    _require(expected_mode in ("free", "soft") and type(expected_decisions) is int and
             1 <= expected_decisions <= 36, "invalid arm/count")
    with ZipFile(path) as z:
        _require(len(z.namelist()) == len(set(z.namelist())), "duplicate ZIP member names")
        summary = _read_json(z, "execution/execution/summary.json")
        _require(summary["schema"] == "driv3r-wormbrain-native-pilot/1" and
                 summary["status"] == "PASS" and summary["mode"] == expected_mode and
                 summary["source_sha"] == SOURCE and summary["source_fenced_m1_valid"] is True and
                 summary["seed"] == 570 and summary["independent_native_replay"] is True and
                 summary["model_decisions"] == summary["model_artifact_count"] == expected_decisions,
                 "mismatched native campaign provenance")
        guide, numeric = _load_guide(z)
        _require(guide["episode_id"] == f"driv3r-worm-{expected_mode}-seed570", "guide episode mismatch")
        events = _events(z)
        cursor_events = [(req, resp) for req, resp in events if req["kind"] == "m3_cursor_move"]
        _require(len(cursor_events) == expected_decisions, "missing native cursor receipts")
        initial_world = summary["initial_world_sha256"]
        original = z.read("execution/native/readback-000001-after.world.i32le")
        _require(len(original) == 512 * 512 * 4 and
                 sha256(original).hexdigest() == initial_world,
                 "initial full-World native material bytes mismatch")
        current_world = initial_world
        previous_cursor = [12, 24]
        replay_history = [previous_cursor[:]]
        event_offset = 0
        edited = 0
        recoveries = 0
        at_best = 0
        sampled = 0
        guide_sum = 0
        gain_sum = 0
        pairs = []
        observations = []
        for i in range(expected_decisions):
            t = _read_json(z, f"execution/model/turn-{i:02}.json")
            _verify_original_model_choice(t)
            state, choice = t["model_state"], t["chosen_candidate"]
            _verify_recovery_catalog(state, t["exact_candidate_ids_in_presented_order"],
                                     previous_cursor, replay_history, edited, recoveries)
            _require(t["index"] == i and t["mode"] == expected_mode and
                     state["position"] == previous_cursor and
                     state["native_world_material_sha256"] == current_world and
                     state["scent_field_sha256"] == guide["values"]["sha256"] and
                     t["host_legal_candidate_sha256"] == state.get("host_legal_candidate_sha256", t["host_legal_candidate_sha256"]) and
                     choice in t["exact_candidate_ids_in_presented_order"] and
                     choice in state["eligible_moves"] and len(state["eligible_moves"]) ==
                     len(set(state["eligible_moves"])), "model/source/host-catalog mismatch")
            hints = state["candidate_hints"]
            for key, hint in hints.items():
                _require(key in MOVES and key in state["eligible_moves"] and
                         hint["destination"] == [previous_cursor[0] + MOVES[key][0],
                                                 previous_cursor[1] + MOVES[key][1]] and
                         hint["soft_scent_u16"] == _guide_at(numeric, guide, hint["destination"]),
                         "scent hint not bound to original numeric field")
            if choice in MOVES:
                _require(choice in hints, "chosen move missing guide hint")
                _require(event_offset + 1 < len(events), "native apply/cursor event absent")
                request, response = events[event_offset]
                _require(request["kind"] == "apply" and request["action_id"] ==
                         f"WORM_{expected_mode}_{i}" and request["cell_budget"] == 1 and
                         len(request["operations"]) == 1, "missing matching native paint operation")
                op = request["operations"][0]
                dest = hints[choice]["destination"]
                _require(op == {"shape": "rectangle", "x0": dest[0], "x1": dest[0],
                                "y0": dest[1], "y1": dest[1], "material": 58},
                         "native paint not the chosen legal target")
                rec = response["receipt"]
                _require(rec["before_world_sha256"] == current_world and
                         rec["after_world_sha256"] == _world_sha(response["observation"]) and
                         rec["actual_changed_cells"] == 1 and
                         rec["native_receipts"][0]["readback"] == [[dest[0], dest[1], 0, 58]] and
                         rec["receipts_reconstruct_materials"] is True,
                         "real native edit lacks one-cell material readback")
                _verify_world_readback(z, request, response, current_world,
                                       rec["after_world_sha256"], dest)
                current_world = rec["after_world_sha256"]
                event_offset += 1
                edited += 1
                eligible_h = [h["soft_scent_u16"] for h in hints.values()]
                _require(len(eligible_h) > 0, "no host-legal move field")
                chosen = hints[choice]["soft_scent_u16"]
                at_best += int(chosen == max(eligible_h))
                sampled += 1
                guide_sum += chosen
                gain_sum += chosen - _guide_at(numeric, guide, previous_cursor)
            else:
                _require(choice in ("BACKTRACK", "RELOCATE_START"), "unsupported archived recovery")
                recoveries += 1
                dest = None
            request, response = events[event_offset]
            _require(request["kind"] == "m3_cursor_move" and
                     request["m3_payload"]["from"] == previous_cursor and
                     request["m3_payload"]["brush_lifted"] is True,
                     "cursor recovery was not material preserving")
            destination = request["m3_payload"]["to"]
            if dest is not None:
                _require(destination == dest, "native cursor is not chosen destination")
            rec = response["receipt"]
            _require(rec["actual_changed_cells"] == 0 and rec["before_world_sha256"] ==
                     rec["after_world_sha256"] == current_world and
                     _world_sha(response["observation"]) == current_world and
                     response["m3"]["cursor"] == destination and
                     response["observation"]["cumulative_changed_cells"] == edited,
                     "native cursor step unexpectedly mutated material")
            _verify_world_readback(z, request, response, current_world,
                                   current_world, None)
            _advance_replayed_history(replay_history, previous_cursor, choice, destination)
            observations.append({"index": i, "choice": choice,
                                 "from": previous_cursor[:], "to": destination[:],
                                 "world_after_sha256": current_world,
                                 "material_edited": choice in MOVES,
                                 "guide_u16_at_destination": _guide_at(numeric, guide, destination)})
            pairs.append((choice, current_world, destination))
            previous_cursor = destination[:]
            event_offset += 1
        _require(event_offset == len(events) and edited == summary["native_painted_cells"] and
                 recoveries == summary["brush_up_recoveries"] and
                 current_world == summary["final_world_sha256"], "native summary/turn divergence")
        return {"mode": expected_mode, "zip_sha256": sha256(Path(path).read_bytes()).hexdigest(),
                "source_sha": SOURCE, "m1_source_commit": M1_COMMIT,
                "initial_world_sha256": initial_world, "final_world_sha256": current_world,
                "initial_host_legal_catalog_sha256": summary["initial_host_legal_catalog_sha256"],
                "guide_sha256": guide["values"]["sha256"],
                "decisions": expected_decisions, "native_edits": edited,
                "cursor_only_recoveries": recoveries, "guide_sampled_move_count": sampled,
                "chosen_max_legal_guide_count": at_best, "sum_chosen_guide_u16": guide_sum,
                "sum_signed_guide_u16_change": gain_sum, "observations": observations,
                "native_undo_performed": False, "visual_model_input_verified": False,
                "assessment": "retrospective host-native receipts; one matched seed; not causal or artistic preference proof"}


def audit_pair(free_path: Path, soft_path: Path) -> dict:
    free = audit_arm(free_path, "free")
    soft = audit_arm(soft_path, "soft")
    _require(free["initial_world_sha256"] == soft["initial_world_sha256"] and
             free["initial_host_legal_catalog_sha256"] == soft["initial_host_legal_catalog_sha256"],
             "arms are not matched at initial native World and legal candidate catalog")
    first_choice = next((i for i, (a, b) in enumerate(zip(free["observations"], soft["observations"]))
                         if a["choice"] != b["choice"]), None)
    first_material = next((i for i, (a, b) in enumerate(zip(free["observations"], soft["observations"]))
                           if a["world_after_sha256"] != b["world_after_sha256"]), None)
    return {"protocol": PROTOCOL, "evidence_class": "archived_real_native_text_only_model_choices",
            "source_sha": SOURCE, "m1_source_commit": M1_COMMIT,
            "free": {k: v for k, v in free.items() if k != "observations"},
            "soft": {k: v for k, v in soft.items() if k != "observations"},
            "first_choice_divergence_zero_based": first_choice,
            "first_native_material_divergence_zero_based": first_material,
            "matched_initial_world": True, "matched_initial_legal_catalog": True,
            "guidance_not_hard_permission": True, "native_undo_executed": False,
            "new_model_or_native_execution": False, "visual_d1_feedback_verified": False,
            "interpretation": "read-only archived pilot, one seed; no model critic or native rollback"}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("Usage: native_pair_shadow.py FREE.zip SOFT.zip", file=sys.stderr)
        return 2
    print(json.dumps(audit_pair(Path(args[0]), Path(args[1])), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
