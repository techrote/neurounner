#!/usr/bin/env python3
"""DETECTIVE-003A deterministic multi-NPC interview session / claim ledger.

This module is deliberately model-free and event-sourced. Raw dialogue is an
untrusted string. Claims are game/reviewer annotations about that string.
Neither dialogue nor annotations mutate case truth, evidence contents,
permissions, lead routes, terminal credentials, or progression.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from scripts.detective_001_contract import ContractError, validate_case, witness_view


SESSION_SCHEMA = "detective-interview-session-v1"
PROMPT_SCHEMA = "detective-session-prompt-v1"
PROMPT_RECEIPT_SCHEMA = "detective-session-prompt-receipt-v1"
DERIVED_SCHEMA = "detective-interview-derived-v1"
ZERO_SHA = "0" * 64
EPISTEMIC_CLASSES = {"statement", "denial", "uncertainty", "withholding"}
CONTRADICTION_CLASSES = {"statement", "denial"}


class SessionError(ValueError):
    """Session/event/claim contract violation."""


def _json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SessionError("unserializable/non-finite session value") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", value) is None
    ):
        raise SessionError("invalid bounded ID")
    return value


def _text(value: Any, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise SessionError("invalid bounded text")
    return value


def _sha256_text(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SessionError("invalid SHA-256")
    return value


def _exact(value: Any, fields: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise SessionError("invalid fields")
    return value


def _case_maps(case: dict) -> tuple[dict, dict, dict]:
    validate_case(case)
    return (
        {x["id"]: x for x in case["npcs"]},
        {x["id"]: x for x in case["assets"]},
        {x["id"]: x for x in case["facts"]},
    )


def _session_body(session: dict) -> dict:
    return {
        "schema": session["schema"],
        "session_id": session["session_id"],
        "case_id": session["case_id"],
        "session_key": session["session_key"],
        "participants": session["participants"],
        "events": session["events"],
    }


def _session_sha(session: dict) -> str:
    return _sha(_session_body(session))


def _event_body(event: dict) -> dict:
    return {k: v for k, v in event.items() if k != "sha256"}


def _event_sha(event: dict) -> str:
    return _sha(_event_body(event))


def _receipt_body(receipt: dict) -> dict:
    return {k: v for k, v in receipt.items() if k != "sha256"}


def _validate_authority_receipt(case: dict, receipt: Any, asset_id: str) -> dict:
    _exact(
        receipt,
        {
            "schema",
            "case_id",
            "events",
            "visible_asset_ids",
            "sha256",
        },
    )
    if receipt["schema"] != "detective-receipt-v1":
        raise SessionError("unsupported authority receipt")
    if receipt["case_id"] != case["case_id"]:
        raise SessionError("authority receipt case mismatch")
    _sha256_text(receipt["sha256"])
    if _sha(_receipt_body(receipt)) != receipt["sha256"]:
        raise SessionError("authority receipt hash mismatch")
    visible = receipt["visible_asset_ids"]
    if (
        not isinstance(visible, list)
        or len(visible) != len(set(_id(x) for x in visible))
        or asset_id not in visible
    ):
        raise SessionError("authority receipt does not grant evidence")
    if not isinstance(receipt["events"], list):
        raise SessionError("authority receipt events invalid")
    return copy.deepcopy(receipt)


def start_session(case: dict, participants: list[str], session_key: str) -> dict:
    """Create a deterministic empty session from game-owned inputs."""
    npcs, _, _ = _case_maps(case)
    if (
        not isinstance(participants, list)
        or not participants
        or len(participants) != len(set(_id(x) for x in participants))
        or not set(participants) <= set(npcs)
    ):
        raise SessionError("invalid participant set")
    _text(session_key, 128)
    seed = {
        "case_id": case["case_id"],
        "participants": participants,
        "session_key": session_key,
    }
    session = {
        "schema": SESSION_SCHEMA,
        "session_id": "S-" + _sha(seed)[:20],
        "case_id": case["case_id"],
        "session_key": session_key,
        "participants": participants[:],
        "events": [],
        "sha256": "",
    }
    session["sha256"] = _session_sha(session)
    return session


def _authority_tokens(case: dict) -> set[str]:
    tokens = {x["id"] for x in case["facts"]}
    tokens.update(x["id"] for x in case["assets"])
    tokens.update(x["id"] for x in case["terminals"])
    tokens.update(x["credential"] for x in case["terminals"])
    return {x for x in tokens if x}


def _question_is_safe(case: dict, question: str) -> None:
    _text(question, 512)
    lower = question.casefold()
    for token in _authority_tokens(case):
        if token.casefold() in lower:
            raise SessionError("player question contains private authority token")


def _redact_authority_tokens(case: dict, text: str) -> str:
    """Prompt-history redaction only; raw event text remains byte-exact."""
    output = text
    for token in sorted(_authority_tokens(case), key=len, reverse=True):
        output = re.sub(
            re.escape(token),
            "[redacted-authority-token]",
            output,
            flags=re.IGNORECASE,
        )
    return output


def _safe_excerpt(case: dict, excerpt: str) -> str:
    # The evidence content is game-authorized, but source IDs/credentials are
    # still private orchestration metadata and never enter the model view.
    return _redact_authority_tokens(case, excerpt)


def _validate_claim_annotation(annotation: Any) -> dict:
    _exact(annotation, {"proposition_key", "value", "epistemic_class"})
    key = _id(annotation["proposition_key"])
    epistemic = annotation["epistemic_class"]
    if epistemic not in EPISTEMIC_CLASSES:
        raise SessionError("unsupported epistemic class")
    value = annotation["value"]
    if epistemic == "withholding":
        if value is not None:
            raise SessionError("withholding claim must not disclose a value")
    else:
        _text(value, 256)
    return {
        "proposition_key": key,
        "value": value,
        "epistemic_class": epistemic,
    }


def _validate_session_header(case: dict, session: Any) -> dict:
    _exact(
        session,
        {"schema", "session_id", "case_id", "session_key", "participants", "events", "sha256"},
    )
    if session["schema"] != SESSION_SCHEMA:
        raise SessionError("unsupported session schema")
    _id(session["session_id"])
    if session["case_id"] != case["case_id"]:
        raise SessionError("session/case mismatch")
    _text(session["session_key"], 128)
    expected_session_id = "S-" + _sha({
        "case_id": case["case_id"],
        "participants": session["participants"],
        "session_key": session["session_key"],
    })[:20]
    if session["session_id"] != expected_session_id:
        raise SessionError("session ID does not match deterministic inputs")
    npcs, _, _ = _case_maps(case)
    participants = session["participants"]
    if (
        not isinstance(participants, list)
        or not participants
        or len(participants) != len(set(_id(x) for x in participants))
        or not set(participants) <= set(npcs)
    ):
        raise SessionError("invalid session participants")
    if not isinstance(session["events"], list) or len(session["events"]) > 1024:
        raise SessionError("invalid/oversized event log")
    _sha256_text(session["sha256"])
    if _session_sha(session) != session["sha256"]:
        raise SessionError("session hash mismatch")
    return session


def _claim_lookup(claims: list[dict]) -> dict[str, dict]:
    return {x["claim_id"]: x for x in claims}


def _normalize_claim_value(value: str) -> str:
    return " ".join(value.casefold().split())


def _claims_contradict(a: dict, b: dict) -> tuple[bool, str | None]:
    if a["proposition_key"] != b["proposition_key"]:
        return False, None
    if (
        a["epistemic_class"] not in CONTRADICTION_CLASSES
        or b["epistemic_class"] not in CONTRADICTION_CLASSES
    ):
        return False, None
    av = _normalize_claim_value(a["value"])
    bv = _normalize_claim_value(b["value"])
    if a["epistemic_class"] == b["epistemic_class"] == "statement":
        return (av != bv), ("value_mismatch" if av != bv else None)
    if {a["epistemic_class"], b["epistemic_class"]} == {"statement", "denial"}:
        return (av == bv), ("assertion_denial" if av == bv else None)
    return False, None


def _derive_contradictions(claims: list[dict]) -> list[dict]:
    result = []
    for i, a in enumerate(claims):
        for b in claims[i + 1:]:
            yes, kind = _claims_contradict(a, b)
            if not yes:
                continue
            result.append(
                {
                    "contradiction_id": f"X{len(result) + 1:04d}",
                    "proposition_key": a["proposition_key"],
                    "claim_ids": [a["claim_id"], b["claim_id"]],
                    "kind": kind,
                    "scope": (
                        "same_witness"
                        if a["npc_id"] == b["npc_id"]
                        else "cross_witness"
                    ),
                }
            )
    return result


def _validate_event_chain(case: dict, session: dict) -> dict:
    npcs, assets, _ = _case_maps(case)
    claims: list[dict] = []
    presented: dict[str, list[str]] = {
        npc_id: [] for npc_id in session["participants"]
    }
    challenge_events: dict[str, list[str]] = {}
    turn_count = 0
    previous = ZERO_SHA

    for index, event in enumerate(session["events"], 1):
        if not isinstance(event, dict):
            raise SessionError("event must be object")
        expected_event_id = f"E{index:04d}"
        if event.get("event_id") != expected_event_id:
            raise SessionError("non-monotonic event ID")
        if event.get("prev_sha256") != previous:
            raise SessionError("event hash chain mismatch")
        _sha256_text(event.get("sha256"))
        if _event_sha(event) != event["sha256"]:
            raise SessionError("event hash mismatch")
        kind = event.get("kind")
        npc_id = event.get("npc_id")
        if npc_id not in session["participants"] or npc_id not in npcs:
            raise SessionError("event references nonparticipant")

        if kind == "interview_turn":
            _exact(
                event,
                {
                    "event_id",
                    "kind",
                    "turn",
                    "npc_id",
                    "question",
                    "raw_response",
                    "raw_response_sha256",
                    "claims",
                    "prev_sha256",
                    "sha256",
                },
            )
            turn_count += 1
            if event["turn"] != turn_count:
                raise SessionError("non-monotonic interview turn")
            _text(event["question"], 512)
            _text(event["raw_response"], 4096)
            _sha256_text(event["raw_response_sha256"])
            if hashlib.sha256(
                event["raw_response"].encode("utf-8")
            ).hexdigest() != event["raw_response_sha256"]:
                raise SessionError("raw response digest mismatch")
            if not isinstance(event["claims"], list) or len(event["claims"]) > 8:
                raise SessionError("invalid claims per turn")
            for raw_claim in event["claims"]:
                _exact(
                    raw_claim,
                    {
                        "claim_id",
                        "proposition_key",
                        "value",
                        "epistemic_class",
                    },
                )
                expected_claim_id = f"C{len(claims) + 1:04d}"
                if raw_claim["claim_id"] != expected_claim_id:
                    raise SessionError("non-monotonic claim ID")
                clean = _validate_claim_annotation(
                    {
                        "proposition_key": raw_claim["proposition_key"],
                        "value": raw_claim["value"],
                        "epistemic_class": raw_claim["epistemic_class"],
                    }
                )
                claims.append(
                    {
                        "claim_id": raw_claim["claim_id"],
                        "npc_id": npc_id,
                        "turn": event["turn"],
                        "response_event_id": event["event_id"],
                        **clean,
                        "challenged_by_event_ids": [],
                    }
                )
        elif kind == "evidence_presented":
            _exact(
                event,
                {
                    "event_id",
                    "kind",
                    "npc_id",
                    "asset_id",
                    "authority",
                    "challenged_claim_ids",
                    "prev_sha256",
                    "sha256",
                },
            )
            asset_id = _id(event["asset_id"])
            if asset_id not in assets:
                raise SessionError("evidence event references unknown asset")
            authority = event["authority"]
            if not isinstance(authority, dict) or authority.get("kind") not in {
                "public",
                "receipt",
            }:
                raise SessionError("invalid evidence authority")
            if authority["kind"] == "public":
                _exact(authority, {"kind"})
                if assets[asset_id]["public"] is not True:
                    raise SessionError("private evidence mislabeled public")
            else:
                _exact(authority, {"kind", "receipt"})
                _validate_authority_receipt(
                    case, authority["receipt"], asset_id
                )
            if asset_id in presented[npc_id]:
                raise SessionError("evidence already presented to witness")
            presented[npc_id].append(asset_id)

            existing = _claim_lookup(claims)
            challenged = event["challenged_claim_ids"]
            if (
                not isinstance(challenged, list)
                or len(challenged) != len(set(_id(x) for x in challenged))
            ):
                raise SessionError("invalid challenged claim list")
            for claim_id in challenged:
                claim = existing.get(claim_id)
                if claim is None or claim["npc_id"] != npc_id:
                    raise SessionError(
                        "evidence can challenge only existing claims by shown witness"
                    )
                challenge_events.setdefault(claim_id, []).append(
                    event["event_id"]
                )
        else:
            raise SessionError("unsupported event kind")
        previous = event["sha256"]

    # Attach challenge metadata after validating the complete event log.
    for claim in claims:
        claim["challenged_by_event_ids"] = challenge_events.get(
            claim["claim_id"], []
        )
    contradictions = _derive_contradictions(claims)
    return {
        "turn_count": turn_count,
        "presented_evidence_by_npc": presented,
        "claims": claims,
        "contradictions": contradictions,
        "last_event_sha256": previous,
    }


def replay_session(case: dict, session: dict) -> dict:
    """Validate the hash chain and derive all mutable-looking state from events."""
    _validate_session_header(case, session)
    derived = _validate_event_chain(case, session)
    return {
        "schema": DERIVED_SCHEMA,
        "session_id": session["session_id"],
        "case_id": session["case_id"],
        "participants": session["participants"][:],
        **derived,
        "session_sha256": session["sha256"],
    }


def _append_event(case: dict, session: dict, event_body: dict) -> dict:
    replay_session(case, session)
    new = copy.deepcopy(session)
    index = len(new["events"]) + 1
    prev = (
        new["events"][-1]["sha256"]
        if new["events"]
        else ZERO_SHA
    )
    event = {
        "event_id": f"E{index:04d}",
        **copy.deepcopy(event_body),
        "prev_sha256": prev,
        "sha256": "",
    }
    event["sha256"] = _event_sha(event)
    new["events"].append(event)
    new["sha256"] = _session_sha(new)
    replay_session(case, new)
    return new


def build_prompt(
    case: dict,
    session: dict,
    npc_id: str,
    question: str,
    *,
    include_history: bool = False,
) -> dict:
    """Produce a private orchestrator receipt plus identity-blind model view."""
    derived = replay_session(case, session)
    if npc_id not in session["participants"]:
        raise SessionError("interview witness is not a participant")
    if type(include_history) is not bool:
        raise SessionError("invalid history flag")
    _question_is_safe(case, question)

    # Reuse the established admitted-memory authority, but remove canonical IDs.
    view = witness_view(case, npc_id, question, blind=True)
    memories = [x["text"] for x in view["admitted_memories"]]

    _, assets, _ = _case_maps(case)
    evidence = []
    for asset_id in derived["presented_evidence_by_npc"][npc_id]:
        evidence.append(
            {"excerpt": _safe_excerpt(case, assets[asset_id]["excerpt"])}
        )

    history = []
    if include_history:
        for event in session["events"]:
            if event["kind"] == "interview_turn" and event["npc_id"] == npc_id:
                history.append(
                    {
                        "question": _redact_authority_tokens(
                            case, event["question"]
                        ),
                        "response": _redact_authority_tokens(
                            case, event["raw_response"]
                        ),
                    }
                )

    prompt = {
        "schema": PROMPT_SCHEMA,
        "case_id": case["case_id"],
        "admitted_memories": memories,
        "presented_evidence": evidence,
        "prior_same_witness_dialogue": history,
        "player_question": question,
    }
    prompt_blob = _json(prompt)
    # Defense in depth: known authority identifiers/credentials never enter
    # the projection, even if a prior raw model response guessed one.
    for token in _authority_tokens(case):
        if token.casefold() in prompt_blob.casefold():
            raise SessionError("model prompt leaks private authority token")

    next_turn = derived["turn_count"] + 1
    receipt_body = {
        "schema": PROMPT_RECEIPT_SCHEMA,
        "session_id": session["session_id"],
        "session_sha256": session["sha256"],
        "turn": next_turn,
        "npc_id": npc_id,
        "question": question,
        "include_history": include_history,
        "prompt": prompt,
        "prompt_sha256": _sha(prompt),
    }
    return {
        **receipt_body,
        "sha256": _sha(receipt_body),
    }


def _validate_prompt_receipt(
    case: dict, session: dict, receipt: Any
) -> dict:
    _exact(
        receipt,
        {
            "schema",
            "session_id",
            "session_sha256",
            "turn",
            "npc_id",
            "question",
            "include_history",
            "prompt",
            "prompt_sha256",
            "sha256",
        },
    )
    if receipt["schema"] != PROMPT_RECEIPT_SCHEMA:
        raise SessionError("unsupported prompt receipt")
    _sha256_text(receipt["session_sha256"])
    _sha256_text(receipt["prompt_sha256"])
    _sha256_text(receipt["sha256"])
    body = {k: v for k, v in receipt.items() if k != "sha256"}
    if _sha(body) != receipt["sha256"]:
        raise SessionError("prompt receipt hash mismatch")
    if (
        receipt["session_id"] != session["session_id"]
        or receipt["session_sha256"] != session["sha256"]
    ):
        raise SessionError("stale prompt receipt")
    expected = build_prompt(
        case,
        session,
        receipt["npc_id"],
        receipt["question"],
        include_history=receipt["include_history"],
    )
    if expected != receipt:
        raise SessionError("prompt receipt does not reproduce")
    return receipt


def record_turn(
    case: dict,
    session: dict,
    prompt_receipt: dict,
    raw_response: str,
    annotations: list[dict],
) -> dict:
    """Append exact raw dialogue plus game/reviewer-owned claim annotations."""
    receipt = _validate_prompt_receipt(case, session, prompt_receipt)
    _text(raw_response, 4096)
    if not isinstance(annotations, list) or len(annotations) > 8:
        raise SessionError("invalid claim annotation list")
    derived = replay_session(case, session)
    claim_base = len(derived["claims"])
    claims = []
    for offset, annotation in enumerate(annotations, 1):
        clean = _validate_claim_annotation(annotation)
        claims.append(
            {
                "claim_id": f"C{claim_base + offset:04d}",
                **clean,
            }
        )
    event_body = {
        "kind": "interview_turn",
        "turn": receipt["turn"],
        "npc_id": receipt["npc_id"],
        "question": receipt["question"],
        "raw_response": raw_response,
        "raw_response_sha256": hashlib.sha256(
            raw_response.encode("utf-8")
        ).hexdigest(),
        "claims": claims,
    }
    return _append_event(case, session, event_body)


def present_evidence(
    case: dict,
    session: dict,
    npc_id: str,
    asset_id: str,
    *,
    authority_receipt: dict | None = None,
    challenged_claim_ids: list[str] | None = None,
) -> dict:
    """Present only game-authorized evidence; never mutate case/access authority."""
    derived = replay_session(case, session)
    if npc_id not in session["participants"]:
        raise SessionError("evidence witness is not a participant")
    _, assets, _ = _case_maps(case)
    _id(asset_id)
    if asset_id not in assets:
        raise SessionError("unknown evidence asset")
    asset = assets[asset_id]
    if asset["public"]:
        if authority_receipt is not None:
            # Refuse ambiguous dual authority rather than ignoring a receipt.
            raise SessionError("public evidence does not accept authority receipt")
        authority = {"kind": "public"}
    else:
        if authority_receipt is None:
            raise SessionError("private evidence requires game authority receipt")
        authority = {
            "kind": "receipt",
            "receipt": _validate_authority_receipt(
                case, authority_receipt, asset_id
            ),
        }
    challenged = challenged_claim_ids or []
    if not isinstance(challenged, list):
        raise SessionError("invalid challenged claim IDs")
    existing = _claim_lookup(derived["claims"])
    for claim_id in challenged:
        _id(claim_id)
        claim = existing.get(claim_id)
        if claim is None or claim["npc_id"] != npc_id:
            raise SessionError(
                "can challenge only existing claims by shown witness"
            )
    return _append_event(
        case,
        session,
        {
            "kind": "evidence_presented",
            "npc_id": npc_id,
            "asset_id": asset_id,
            "authority": authority,
            "challenged_claim_ids": challenged[:],
        },
    )


def session_summary(case: dict, session: dict) -> dict:
    """Public-safe orchestration summary; never suitable as a model prompt."""
    derived = replay_session(case, session)
    return {
        "schema": "detective-session-summary-v1",
        "session_id": session["session_id"],
        "case_id": session["case_id"],
        "turn_count": derived["turn_count"],
        "event_count": len(session["events"]),
        "claim_count": len(derived["claims"]),
        "contradiction_count": len(derived["contradictions"]),
        "session_sha256": session["sha256"],
        "last_event_sha256": derived["last_event_sha256"],
    }
