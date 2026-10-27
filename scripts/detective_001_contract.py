"""DETECTIVE-001A: offline fictional case and typed scripter contract.

No model weights, filesystem commands, network clients, or arbitrary execution.
Models can propose programmes; the game owns the case, permissions and results.
This module is a deterministic research fixture, not a production game bridge.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


class ContractError(ValueError):
    """A case, observation or model programme violates its contract."""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("unserializable or non-finite JSON value") from exc


def _id(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", value) is None:
        raise ContractError("invalid ID")
    return value


def _text(value: Any, limit: int = 512) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ContractError("invalid bounded text")
    return value


def _exact(obj: Any, fields: set[str]) -> dict:
    if not isinstance(obj, dict) or set(obj) != fields:
        raise ContractError("invalid fields")
    return obj


def _unique(items: Any, key: str) -> dict[str, dict]:
    if not isinstance(items, list) or not items:
        raise ContractError("empty or invalid collection")
    result = {}
    for item in items:
        if not isinstance(item, dict):
            raise ContractError("invalid collection entry")
        ident = _id(item.get(key))
        if ident in result:
            raise ContractError("duplicate ID")
        result[ident] = item
    return result


def validate_case(case: Any) -> None:
    """Check source attribution and non-dialogue recovery of mandatory leads."""
    _exact(case, {"schema", "case_id", "facts", "npcs", "assets", "terminals", "leads"})
    if case["schema"] != "detective-case-v1":
        raise ContractError("unsupported case schema")
    _id(case["case_id"])
    if len(_json(case).encode("utf-8")) > 64_000:
        raise ContractError("case exceeds fixture bound")

    facts = _unique(case["facts"], "id")
    npcs = _unique(case["npcs"], "id")
    assets = _unique(case["assets"], "id")
    for fact in facts.values():
        _exact(fact, {"id", "text"})
        _text(fact["text"])
    for npc in npcs.values():
        _exact(npc, {"id", "kind", "backstory", "motivation", "fault", "known_fact_ids"})
        if npc["kind"] not in {"goblin", "robot"}:
            raise ContractError("unsupported actor kind")
        for field in ("backstory", "motivation", "fault"):
            _text(npc[field])
        known = npc["known_fact_ids"]
        if not isinstance(known, list) or len(known) != len(set(_id(i) for i in known)):
            raise ContractError("invalid known facts")
        if not set(known) <= facts.keys():
            raise ContractError("actor references unknown fact")
    for asset in assets.values():
        _exact(asset, {"id", "excerpt", "fact_ids", "public"})
        _text(asset["excerpt"], 1024)
        if type(asset["public"]) is not bool:
            raise ContractError("invalid asset access")
        known = asset["fact_ids"]
        if not isinstance(known, list) or len(known) != len(set(_id(i) for i in known)):
            raise ContractError("invalid asset facts")
        if not set(known) <= facts.keys():
            raise ContractError("asset references unknown fact")

    terminals = case["terminals"]
    if not isinstance(terminals, list):
        raise ContractError("invalid terminals")
    terminal_map = _unique(terminals, "id") if terminals else {}
    for terminal in terminal_map.values():
        _exact(terminal, {"id", "credential", "grants"})
        _text(terminal["credential"], 64)
        if not terminal["credential"]:
            raise ContractError("empty credential")
        grants = terminal["grants"]
        if not isinstance(grants, list) or len(grants) != len(set(_id(i) for i in grants)):
            raise ContractError("invalid grants")
        if not set(grants) <= assets.keys():
            raise ContractError("unknown granted asset")

    leads = case["leads"]
    if not isinstance(leads, list) or not leads:
        raise ContractError("missing lead list")
    seen_facts = set()
    for lead in leads:
        _exact(lead, {"fact_id", "mandatory", "routes"})
        fact_id = _id(lead["fact_id"])
        if fact_id not in facts or fact_id in seen_facts:
            raise ContractError("duplicate or unknown lead fact")
        seen_facts.add(fact_id)
        if type(lead["mandatory"]) is not bool or not isinstance(lead["routes"], list):
            raise ContractError("invalid lead")
        routes = lead["routes"]
        route_ids, source_ids = set(), set()
        public_evidence = False
        for route in routes:
            _exact(route, {"id", "kind", "source"})
            route_id, source_id = _id(route["id"]), _id(route["source"])
            if route_id in route_ids or source_id in source_ids:
                raise ContractError("route IDs and independent sources must be unique")
            route_ids.add(route_id)
            source_ids.add(source_id)
            if route["kind"] == "witness":
                if source_id not in npcs or fact_id not in npcs[source_id]["known_fact_ids"]:
                    raise ContractError("witness cannot disclose lead")
            elif route["kind"] == "asset":
                if source_id not in assets or fact_id not in assets[source_id]["fact_ids"]:
                    raise ContractError("evidence cannot disclose lead")
                public_evidence |= assets[source_id]["public"]
            else:
                raise ContractError("unsupported lead route")
        if lead["mandatory"] and (len(routes) < 2 or not public_evidence):
            raise ContractError("mandatory lead needs two independent routes and public evidence")


def witness_view(case: dict, npc_id: str, question: str, *, blind: bool = False) -> dict:
    """Whitelisted generation view; optional identity-blind qualification input.

    The orchestrator binds model/member identity in separate run metadata.
    The blind text contains no ID, backstory, motive, fault or species cue.
    """
    validate_case(case)
    _id(npc_id)
    _text(question, 512)
    if type(blind) is not bool:
        raise ContractError("invalid interview mode")
    npcs = {item["id"]: item for item in case["npcs"]}
    if npc_id not in npcs:
        raise ContractError("unknown witness")
    npc = npcs[npc_id]
    facts = {item["id"]: item["text"] for item in case["facts"]}
    view = {
        "schema": "detective-interview-v1",
        "case_id": case["case_id"],
        "admitted_memories": [{"id": i, "text": facts[i]} for i in npc["known_fact_ids"]],
        "player_question": question,
    }
    if not blind:
        view.update({
            "npc_id": npc_id,
            "kind": npc["kind"],
            "backstory": npc["backstory"],
            "motivation": npc["motivation"],
            "fault": npc["fault"],
        })
    return view


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(_: str) -> None:
    raise ContractError("non-finite JSON number")


def parse_program(program: str | dict) -> list[dict]:
    """Fail closed on entire proposal before any fictional step executes."""
    if isinstance(program, str):
        if len(program.encode("utf-8")) > 4_096:
            raise ContractError("programme exceeds byte bound")
        try:
            program = json.loads(program, object_pairs_hook=_no_duplicate_pairs, parse_constant=_bad_constant)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError("invalid programme JSON") from exc
    elif len(_json(program).encode("utf-8")) > 4_096:
        raise ContractError("programme exceeds byte bound")
    _exact(program, {"schema", "steps"})
    if program["schema"] != "detective-script-v1":
        raise ContractError("unsupported programme schema")
    steps = program["steps"]
    if not isinstance(steps, list) or len(steps) > 8:
        raise ContractError("too many or invalid steps")
    contracts = {
        "read_asset": {"op", "asset_id"},
        "correlate_assets": {"op", "left_id", "right_id"},
        "probe_terminal": {"op", "terminal_id", "credential"},
    }
    for step in steps:
        if not isinstance(step, dict) or step.get("op") not in contracts:
            raise ContractError("unrecognized fictional operation")
        _exact(step, contracts[step["op"]])
        for key, value in step.items():
            if key == "op":
                continue
            if key == "credential":
                _text(value, 64)
            else:
                _id(value)
    return steps


def execute_program(case: dict, program: str | dict, authorized_asset_ids: list[str]) -> dict:
    """Interpret fictional evidence ops with game-supplied access, no host effects.

    The returned receipt is the entire result: no input is mutated, and a model
    cannot promote dialogue or its own statements into canonical evidence.
    """
    validate_case(case)
    steps = parse_program(program)
    assets = {item["id"]: item for item in case["assets"]}
    terminals = {item["id"]: item for item in case["terminals"]}
    if not isinstance(authorized_asset_ids, list) or len(authorized_asset_ids) != len(set(_id(i) for i in authorized_asset_ids)):
        raise ContractError("invalid game authority permissions")
    if not set(authorized_asset_ids) <= assets.keys():
        raise ContractError("game authorized unknown asset")
    visible = set(authorized_asset_ids)
    events = []
    for index, step in enumerate(steps):
        op = step["op"]
        if op == "read_asset":
            ident = step["asset_id"]
            if ident not in visible:
                outcome = {"status": "ACCESS_DENIED"}
            else:
                asset = assets[ident]
                outcome = {"status": "OK", "excerpt": asset["excerpt"], "fact_ids": asset["fact_ids"][:]}
        elif op == "correlate_assets":
            left, right = step["left_id"], step["right_id"]
            if left not in visible or right not in visible:
                outcome = {"status": "ACCESS_DENIED"}
            else:
                common = sorted(set(assets[left]["fact_ids"]) & set(assets[right]["fact_ids"]))
                outcome = {"status": "OK", "shared_fact_ids": common}
        else:
            terminal = terminals.get(step["terminal_id"])
            if terminal is None or terminal["credential"] != step["credential"]:
                outcome = {"status": "DENIED"}
            else:
                visible.update(terminal["grants"])
                outcome = {"status": "GRANTED", "asset_ids": sorted(terminal["grants"])}
        events.append({"step": index, "op": op, "result": outcome})
    receipt = {
        "schema": "detective-receipt-v1",
        "case_id": case["case_id"],
        "events": events,
        "visible_asset_ids": sorted(visible),
    }
    receipt["sha256"] = hashlib.sha256(_json(receipt).encode("utf-8")).hexdigest()
    return receipt
