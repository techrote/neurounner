#!/usr/bin/env python3
"""DETECTIVE-001E model-free playable case adapter over frozen 001A/003A/003C.

The world ledger owns evidence. NPC speech is always a claim, never a grant.
Typed proposals run only in the fictional 003C interpreter; applying a receipt
is a separate explicit game decision. No inference, training, host scripting,
new cast admission, production admission, or alterations to frozen authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_003a_session as sessions  # noqa: E402
from scripts import detective_003c_scripter_bridge as scripting  # noqa: E402

SCHEMA = "detective-001e-offline-play-v1"
PARTICIPANTS = ("V-G02-A", "V-G09-C")
ZERO = "0" * 64
RESPONSES = {
    "V-G02-A": "Clock's nine minutes wrong. Or I'm nine minutes right. Ask the ledger, not me.",
    "V-G09-C": "I remember the scan. I remember the scan. Neither memory is a receipt.",
}


class PlayError(ValueError):
    """Invalid game-owned action or corrupted local state."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def authority() -> dict:
    """Return only exact source-pinned Bay Seventeen game authority."""
    _, case = scripting.load_contract()
    return case


def _seal(value: dict) -> dict:
    new = copy.deepcopy(value)
    new["sha256"] = digest({k: v for k, v in new.items() if k != "sha256"})
    return new


def replay(state: dict) -> dict:
    """Validate both original 003A session chain and local public inspection chain."""
    case = authority()
    if not isinstance(state, dict) or set(state) != {
        "schema", "case_sha256", "session", "inspections", "sha256"
    } or state["schema"] != SCHEMA:
        raise PlayError("game schema/field mismatch")
    if state["case_sha256"] != digest(case):
        raise PlayError("canonical case changed")
    if state["sha256"] != digest({k: v for k, v in state.items() if k != "sha256"}):
        raise PlayError("game state hash mismatch")
    try:
        sessions.replay_session(case, state["session"])
    except (ValueError, TypeError, KeyError) as exc:
        raise PlayError("session event chain invalid") from exc
    if state["session"]["participants"] != list(PARTICIPANTS):
        raise PlayError("participant identity changed")
    if not isinstance(state["inspections"], list):
        raise PlayError("inspection journal malformed")
    previous, inspected = ZERO, []
    for i, entry in enumerate(state["inspections"]):
        if not isinstance(entry, dict) or set(entry) != {
            "sequence", "asset_id", "prev_sha256", "sha256"
        }:
            raise PlayError("inspection event fields invalid")
        if entry["sequence"] != i + 1 or entry["prev_sha256"] != previous:
            raise PlayError("inspection replay order invalid")
        asset_id = entry["asset_id"]
        asset = next((x for x in case["assets"] if x["id"] == asset_id), None)
        if asset is None or not asset["public"] or asset_id in inspected:
            raise PlayError("private/duplicate inspection is forbidden")
        if entry["sha256"] != digest({k: v for k, v in entry.items() if k != "sha256"}):
            raise PlayError("inspection receipt hash invalid")
        inspected.append(asset_id)
        previous = entry["sha256"]
    return {"case_sha256": state["case_sha256"],
            "session_sha256": state["session"]["sha256"],
            "public_assets": inspected}


def start() -> dict:
    case = authority()
    state = _seal({
        "schema": SCHEMA, "case_sha256": digest(case),
        "session": sessions.start_session(case, list(PARTICIPANTS),
                                           "bay-seventeen-local-offline-v1"),
        "inspections": [],
    })
    replay(state)
    return state


def inspect(state: dict, asset_id: str) -> tuple[dict, str]:
    prior = replay(state)
    case = authority()
    asset = next((x for x in case["assets"] if x["id"] == asset_id), None)
    if asset is None or asset["public"] is not True:
        raise PlayError("only case-owned public evidence may be inspected")
    if asset_id in prior["public_assets"]:
        return copy.deepcopy(state), asset["excerpt"]
    new = copy.deepcopy(state)
    event = {"sequence": len(new["inspections"]) + 1,
             "asset_id": asset_id,
             "prev_sha256": new["inspections"][-1]["sha256"]
             if new["inspections"] else ZERO}
    new["inspections"].append({**event, "sha256": digest(event)})
    new = _seal(new)
    replay(new)
    return new, asset["excerpt"]


def view(state: dict, actor: str, question: str) -> dict:
    replay(state)
    return sessions.build_prompt(authority(), state["session"], actor,
                                 question, include_history=True)["prompt"]


def ask(state: dict, actor: str, question: str,
        untrusted_reply: str | None = None) -> tuple[dict, str]:
    replay(state)
    if actor not in PARTICIPANTS:
        raise PlayError("unknown witness")
    case = authority()
    receipt = sessions.build_prompt(case, state["session"], actor,
                                    question, include_history=True)
    reply = RESPONSES[actor] if untrusted_reply is None else untrusted_reply
    new = copy.deepcopy(state)
    # Explicitly no model-output claims, unlocks, or truth promotion.
    new["session"] = sessions.record_turn(case, state["session"], receipt,
                                          reply, annotations=[])
    new = _seal(new)
    replay(new)
    return new, reply


def present_public(state: dict, actor: str, asset_id: str) -> dict:
    existing = replay(state)["public_assets"]
    if asset_id not in existing:
        raise PlayError("public asset must be inspected before presentation")
    new = copy.deepcopy(state)
    new["session"] = sessions.present_evidence(
        authority(), state["session"], actor, asset_id)
    new = _seal(new)
    replay(new)
    return new


def propose(state: dict, raw_program: str,
            player_credential: str | None = None) -> tuple[dict, dict]:
    """Run a *fictional*, typed programme without changing the game session."""
    inspected = replay(state)["public_assets"]
    case = authority()
    credentials = {} if player_credential is None else {
        "camera-console": player_credential
    }
    request = scripting.build_scripter_request(
        case, state["session"],
        "Try only admitted fictional evidence operations.",
        inspected, ["camera-console"], credentials)
    result = scripting.execute_scripter_transaction(
        case, state["session"], request, raw_program)
    if replay(state)["session_sha256"] != request["session_sha256"]:
        raise PlayError("fictional programme mutated the game")
    return request, result


def apply(state: dict, request: dict, transaction: dict,
          actor: str, asset_id: str) -> dict:
    """Only explicit authoritative game application may present tool evidence."""
    replay(state)
    new_session = scripting.apply_transaction_evidence(
        authority(), state["session"], request, transaction, actor, asset_id)
    new = copy.deepcopy(state)
    new["session"] = new_session
    new = _seal(new)
    replay(new)
    return new


def status(state: dict) -> dict:
    inspected = replay(state)["public_assets"]
    case = authority()
    mandatory = [lead for lead in case["leads"] if lead["mandatory"]]
    available = {}
    for lead in mandatory:
        distinct = {(r["kind"], r["source"]) for r in lead["routes"]}
        public_routes = [
            r["source"] for r in lead["routes"]
            if r["kind"] == "asset"
            and any(a["id"] == r["source"] and a["public"]
                    for a in case["assets"])
        ]
        if len(distinct) < 2 or not public_routes:
            raise PlayError("mandatory lead lacks independent public route")
        available[lead["fact_id"]] = any(x in inspected for x in public_routes)
    return {
        "schema": "detective-001e-case-progress-v1",
        "mandatory_leads_recovered": sum(available.values()),
        "mandatory_leads_total": len(available),
        "solvable_without_dialogue": all(available.values()),
        "evidence_authority": "inspected_public_game_assets_only",
        "session": sessions.session_summary(case, state["session"]),
        "case_sha256": state["case_sha256"],
        "optimizer_steps": 0,
        "production_admission": False,
        "population_expansion_admitted": False,
    }


def smoke() -> dict:
    game = start()
    for actor, prompt in (
        ("V-G02-A", "Do you remember the clock?"),
        ("V-G09-C", "What happened near the rail?"),
    ):
        game, _ = ask(game, actor, prompt)
    for asset in ("dock-clock-ledger", "transfer-card"):
        game, _ = inspect(game, asset)
    result = status(game)
    if result["mandatory_leads_recovered"] != 2:
        raise PlayError("public route recovery failed")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline Bay Seventeen playable case; no AI model")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return
    game, pending = start(), None
    print("BAY SEVENTEEN / NO MODEL. Commands: help, status, inspect <asset>,")
    print("ask <actor> <question>, present <actor> <asset>,")
    print("script <credential|-> <JSON>, apply <actor> <asset>, quit")
    while True:
        try:
            line = input("bay17> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended")
            return
        if line in ("quit", "exit"):
            return
        try:
            fields = line.split(maxsplit=2)
            if not fields:
                continue
            command = fields[0]
            if command == "help":
                print("Witnesses: V-G02-A, V-G09-C; public: dock-clock-ledger, transfer-card.")
                print("Use only fictional detective-script-v1 operations. No host code execution.")
            elif command == "status" and len(fields) == 1:
                print(json.dumps(status(game), indent=2, sort_keys=True))
            elif command == "inspect" and len(fields) == 2:
                game, excerpt = inspect(game, fields[1])
                print("AUTHORITATIVE PUBLIC RECORD:", excerpt)
            elif command == "ask" and len(fields) == 3:
                game, raw = ask(game, fields[1], fields[2])
                print("UNTRUSTED SCRIPTED TESTIMONY:", raw)
            elif command == "present" and len(fields) == 3:
                actor, asset = fields[1], fields[2]
                game = present_public(game, actor, asset)
                print("Public record shown. Case truth unchanged.")
            elif command == "script" and len(fields) == 3:
                credential = None if fields[1] == "-" else fields[1]
                pending = propose(game, fields[2], credential)
                print("FICTIONAL SCRIPT RESULT:", json.dumps({
                    "status": pending[1]["status"],
                    "visible_asset_ids_after": pending[1]["visible_asset_ids_after"],
                    "rejection": pending[1]["rejection"],
                }, sort_keys=True))
            elif command == "apply" and len(fields) == 3:
                if pending is None:
                    raise PlayError("no pending typed scripter receipt")
                game = apply(game, *pending, fields[1], fields[2])
                pending = None
                print("Game-owned transaction applied. No automatic truth promotion.")
            else:
                raise PlayError("invalid command; type help")
        except (PlayError, sessions.SessionError, scripting.BridgeError,
                TypeError, KeyError, ValueError) as exc:
            print("REFUSED:", exc)


if __name__ == "__main__":
    main()
