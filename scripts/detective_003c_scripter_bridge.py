#!/usr/bin/env python3
"""DETECTIVE-003C authoritative fictional scripter transaction bridge.

No model inference, no host execution, no training. An untrusted raw scripter
proposal is parsed/executed only through DETECTIVE-001A. A successful execution
receipt still does not mutate the interview session; the game must separately
choose an evidence asset to present through DETECTIVE-003A.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import detective_003a_session as session_api  # noqa: E402
from scripts.detective_001_contract import (  # noqa: E402
    ContractError,
    execute_program,
    parse_program,
    validate_case,
)

CONFIG = ROOT / "experiments/detective-003/scripter-bridge.json"
CASE = ROOT / "experiments/detective-001/pilot-case.json"
CONTRACT_001A = ROOT / "scripts/detective_001_contract.py"
SESSION_003A = ROOT / "scripts/detective_003a_session.py"

REQUEST_SCHEMA = "detective-scripter-request-receipt-v1"
TRANSACTION_SCHEMA = "detective-scripter-transaction-v1"


class BridgeError(ValueError):
    """Scripter request/transaction violates the bridge contract."""


def canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BridgeError("unserializable/non-finite bridge value") from exc


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"))
    h.update(data)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", value) is None
    ):
        raise BridgeError("invalid bounded ID")
    return value


def _text(value: Any, limit: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise BridgeError("invalid bounded text")
    if not allow_empty and not value:
        raise BridgeError("empty bounded text")
    return value


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BridgeError("invalid SHA-256")
    return value


def _exact(value: Any, fields: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise BridgeError("invalid fields")
    return value


def _case_maps(case: dict) -> tuple[dict, dict, dict]:
    validate_case(case)
    return (
        {x["id"]: x for x in case["facts"]},
        {x["id"]: x for x in case["assets"]},
        {x["id"]: x for x in case["terminals"]},
    )


def load_contract() -> tuple[dict, dict]:
    cfg = read_json(CONFIG)
    case = read_json(CASE)
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-003c-scripter-bridge-v1",
        "DETECTIVE-003C",
        144,
    ):
        raise BridgeError("003C config identity drift")
    validate_case(case)
    authority = cfg["authority"]
    checks = (
        (
            CONTRACT_001A,
            authority["detective_001a"]["path"],
            authority["detective_001a"]["git_blob_sha1"],
            "b10254b7de26a8df676a26c89b00c6ccb8b9829e",
        ),
        (
            SESSION_003A,
            authority["detective_003a"]["path"],
            authority["detective_003a"]["git_blob_sha1"],
            "73801ea59ce0f6160acc7027bd6aeaa085950236",
        ),
        (
            CASE,
            authority["case"]["path"],
            authority["case"]["git_blob_sha1"],
            "bbe5e0b3991a8296033376147bf830b05d33d4bb",
        ),
    )
    for path, declared_path, declared_blob, expected_blob in checks:
        if declared_path != path.relative_to(ROOT).as_posix():
            raise BridgeError("authority path drift")
        if declared_blob != expected_blob or git_blob_sha1(path) != declared_blob:
            raise BridgeError("authority Git blob drift: " + declared_path)
    if case["case_id"] != authority["case"]["case_id"]:
        raise BridgeError("case ID drift")
    if cfg["view"]["schema"] != "detective-scripter-view-v1":
        raise BridgeError("view schema drift")
    if cfg["transaction"] != {
        "schema": TRANSACTION_SCHEMA,
        "allowed_status": ["proposal_rejected", "executed"],
        "auto_present_evidence": False,
        "auto_mutate_session": False,
    }:
        raise BridgeError("transaction authority widened")
    if cfg["decision"] != {
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "model_inference": False,
        "learner_final_accessed": False,
        "relevance_mapping_accessed": False,
        "body_selection_made": False,
        "population_expansion_admitted": False,
        "host_execution": False,
    }:
        raise BridgeError("003C decision boundary drift")
    return cfg, case


def _validate_unique_ids(values: Any, *, max_items: int) -> list[str]:
    if not isinstance(values, list) or len(values) > max_items:
        raise BridgeError("invalid bounded ID collection")
    clean = [_id(x) for x in values]
    if len(clean) != len(set(clean)):
        raise BridgeError("duplicate ID")
    return clean


def _safe_player_request(
    case: dict,
    request: str,
    visible_asset_ids: set[str],
    disclosed_terminal_ids: set[str],
) -> str:
    cfg = read_json(CONFIG)
    request = _text(request, cfg["bounds"]["player_request_chars"])
    lower = request.casefold()
    facts, assets, terminals = _case_maps(case)

    # Canonical truth IDs/text never enter the scripter prompt.
    for fact in facts.values():
        if fact["id"].casefold() in lower or fact["text"].casefold() in lower:
            raise BridgeError("player request contains canonical fact authority")

    # Exact case credentials must be supplied through the explicit game-owned
    # credential field, never smuggled through prose.
    for terminal in terminals.values():
        if terminal["credential"].casefold() in lower:
            raise BridgeError("player request contains case credential literal")

    # Asset and terminal IDs are only legal in prose when already disclosed.
    for asset_id in assets:
        if asset_id.casefold() in lower and asset_id not in visible_asset_ids:
            raise BridgeError("player request contains undisclosed asset ID")
    for terminal_id in terminals:
        if (
            terminal_id.casefold() in lower
            and terminal_id not in disclosed_terminal_ids
        ):
            raise BridgeError("player request contains undisclosed terminal ID")
    return request


def _validate_player_credentials(
    case: dict,
    disclosed_terminal_ids: set[str],
    credentials: Any,
    *,
    max_items: int,
) -> dict[str, str]:
    _, _, terminals = _case_maps(case)
    if not isinstance(credentials, dict) or len(credentials) > max_items:
        raise BridgeError("invalid player-known credential mapping")
    out: dict[str, str] = {}
    for terminal_id, credential in credentials.items():
        terminal_id = _id(terminal_id)
        if terminal_id not in terminals or terminal_id not in disclosed_terminal_ids:
            raise BridgeError("credential supplied for undisclosed terminal")
        credential = _text(credential, 64)
        out[terminal_id] = credential
    return out


def _request_view(
    case: dict,
    player_request: str,
    visible_asset_ids: list[str],
    disclosed_terminal_ids: list[str],
    credentials: dict[str, str],
    cfg: dict,
) -> dict:
    facts, assets, terminals = _case_maps(case)
    view = {
        "schema": cfg["view"]["schema"],
        "case_id": case["case_id"],
        "player_request": player_request,
        "visible_assets": [
            {
                "asset_id": asset_id,
                "excerpt": assets[asset_id]["excerpt"],
            }
            for asset_id in visible_asset_ids
        ],
        "disclosed_terminals": [
            {
                "terminal_id": terminal_id,
                "player_known_credential": credentials.get(terminal_id),
            }
            for terminal_id in disclosed_terminal_ids
        ],
        "allowed_operations": copy.deepcopy(cfg["view"]["allowed_operations"]),
    }

    blob = canonical(view).casefold()
    # No canonical fact authority is ever exposed.
    for fact in facts.values():
        if fact["id"].casefold() in blob or fact["text"].casefold() in blob:
            raise BridgeError("scripter view leaks canonical fact authority")

    # No hidden asset ID/excerpt.
    visible = set(visible_asset_ids)
    for asset_id, asset in assets.items():
        if asset_id in visible:
            continue
        if asset_id.casefold() in blob or asset["excerpt"].casefold() in blob:
            raise BridgeError("scripter view leaks hidden evidence")

    # No undisclosed terminal ID/grants.
    disclosed = set(disclosed_terminal_ids)
    for terminal_id, terminal in terminals.items():
        if terminal_id not in disclosed and terminal_id.casefold() in blob:
            raise BridgeError("scripter view leaks undisclosed terminal")
        for grant in terminal["grants"]:
            if grant.casefold() in blob and grant not in visible:
                raise BridgeError("scripter view leaks terminal grant")

    # Case credentials are visible only if exactly game-supplied.
    supplied_values = set(credentials.values())
    for terminal in terminals.values():
        credential = terminal["credential"]
        if credential.casefold() in blob and credential not in supplied_values:
            raise BridgeError("scripter view leaks undisclosed credential")

    # Interview identities/state never enter this distinct scripter lane.
    for npc in case["npcs"]:
        if npc["id"].casefold() in blob:
            raise BridgeError("scripter view leaks witness identity")
    for forbidden in ("claim_id", "contradiction", "epistemic_class"):
        if forbidden in blob:
            raise BridgeError("scripter view leaks interview metadata")
    return view


def build_scripter_request(
    case: dict,
    session: dict,
    player_request: str,
    visible_asset_ids: list[str],
    disclosed_terminal_ids: list[str],
    player_known_credentials: dict[str, str],
) -> dict:
    """Build private request receipt + nested model-facing scripter view."""
    cfg, authority_case = load_contract()
    if canonical(case) != canonical(authority_case):
        raise BridgeError("003C accepts only the frozen Bay Seventeen fixture")
    session_api.replay_session(case, session)

    _, assets, terminals = _case_maps(case)
    visible = _validate_unique_ids(
        visible_asset_ids,
        max_items=cfg["bounds"]["max_visible_assets"],
    )
    if any(x not in assets for x in visible):
        raise BridgeError("visible set references unknown asset")
    # Stage 003C request construction is intentionally conservative: private
    # evidence enters the session only through a transaction receipt, not as an
    # initial scripter-view asset.
    if any(assets[x]["public"] is not True for x in visible):
        raise BridgeError("initial scripter view may contain public assets only")

    disclosed = _validate_unique_ids(
        disclosed_terminal_ids,
        max_items=cfg["bounds"]["max_disclosed_terminals"],
    )
    if any(x not in terminals for x in disclosed):
        raise BridgeError("disclosed terminal set references unknown terminal")
    credentials = _validate_player_credentials(
        case,
        set(disclosed),
        player_known_credentials,
        max_items=cfg["bounds"]["max_player_credentials"],
    )
    request = _safe_player_request(
        case,
        player_request,
        set(visible),
        set(disclosed),
    )
    view = _request_view(
        case,
        request,
        visible,
        disclosed,
        credentials,
        cfg,
    )
    request_seed = {
        "session_id": session["session_id"],
        "session_sha256": session["sha256"],
        "player_request": request,
        "visible_asset_ids": visible,
        "disclosed_terminal_ids": disclosed,
        "player_known_credentials": credentials,
        "view_sha256": sha(view),
    }
    receipt_body = {
        "schema": REQUEST_SCHEMA,
        "request_id": "SR-" + sha(request_seed)[:20],
        "session_id": session["session_id"],
        "session_sha256": session["sha256"],
        "player_request": request,
        "visible_asset_ids": visible,
        "disclosed_terminal_ids": disclosed,
        "player_known_credentials": credentials,
        "view": view,
        "view_sha256": sha(view),
    }
    return {**receipt_body, "sha256": sha(receipt_body)}


def validate_scripter_request(
    case: dict,
    session: dict,
    receipt: Any,
) -> dict:
    _exact(
        receipt,
        {
            "schema",
            "request_id",
            "session_id",
            "session_sha256",
            "player_request",
            "visible_asset_ids",
            "disclosed_terminal_ids",
            "player_known_credentials",
            "view",
            "view_sha256",
            "sha256",
        },
    )
    if receipt["schema"] != REQUEST_SCHEMA:
        raise BridgeError("unsupported scripter request receipt")
    _id(receipt["request_id"])
    _sha256(receipt["session_sha256"])
    _sha256(receipt["view_sha256"])
    _sha256(receipt["sha256"])
    body = {k: v for k, v in receipt.items() if k != "sha256"}
    if sha(body) != receipt["sha256"]:
        raise BridgeError("scripter request receipt hash mismatch")
    if (
        receipt["session_id"] != session["session_id"]
        or receipt["session_sha256"] != session["sha256"]
    ):
        raise BridgeError("stale scripter request")
    expected = build_scripter_request(
        case,
        session,
        receipt["player_request"],
        receipt["visible_asset_ids"],
        receipt["disclosed_terminal_ids"],
        receipt["player_known_credentials"],
    )
    if expected != receipt:
        raise BridgeError("scripter request receipt does not reproduce")
    return receipt


def _transaction_body(transaction: dict) -> dict:
    return {k: v for k, v in transaction.items() if k != "sha256"}


def execute_scripter_transaction(
    case: dict,
    session: dict,
    request_receipt: dict,
    raw_proposal: str,
) -> dict:
    """Strictly parse + execute proposal through DETECTIVE-001A authority."""
    cfg, authority_case = load_contract()
    if canonical(case) != canonical(authority_case):
        raise BridgeError("003C accepts only the frozen Bay Seventeen fixture")
    request = validate_scripter_request(case, session, request_receipt)
    raw_proposal = _text(
        raw_proposal,
        cfg["bounds"]["raw_proposal_bytes"],
    )
    if len(raw_proposal.encode("utf-8")) > cfg["bounds"]["raw_proposal_bytes"]:
        raise BridgeError("raw proposal byte bound exceeded")
    raw_sha = hashlib.sha256(raw_proposal.encode("utf-8")).hexdigest()

    parsed_steps: list[dict] | None = None
    execution_receipt: dict | None = None
    rejection: dict | None = None
    visible_before = request["visible_asset_ids"][:]
    try:
        parsed_steps = copy.deepcopy(parse_program(raw_proposal))
        # Reconstruct a programme from parser output so the stored parsed form
        # is exactly the form executed.
        programme = {
            "schema": "detective-script-v1",
            "steps": copy.deepcopy(parsed_steps),
        }
        execution_receipt = execute_program(
            case,
            programme,
            visible_before,
        )
        visible_after = execution_receipt["visible_asset_ids"][:]
        status = "executed"
    except ContractError as exc:
        visible_after = visible_before[:]
        status = "proposal_rejected"
        rejection = {
            "error_type": "ContractError",
            "message": _text(str(exc), 256),
        }

    tx_seed = {
        "request_receipt_sha256": request["sha256"],
        "raw_proposal_sha256": raw_sha,
    }
    body = {
        "schema": TRANSACTION_SCHEMA,
        "transaction_id": "ST-" + sha(tx_seed)[:20],
        "request_id": request["request_id"],
        "request_receipt_sha256": request["sha256"],
        "session_id": session["session_id"],
        "session_sha256_before": session["sha256"],
        "raw_proposal": raw_proposal,
        "raw_proposal_sha256": raw_sha,
        "visible_asset_ids_before": visible_before,
        "status": status,
        "parsed_steps": parsed_steps,
        "execution_receipt": execution_receipt,
        "visible_asset_ids_after": visible_after,
        "rejection": rejection,
        "auto_presented_asset_ids": [],
        "session_sha256_after_execution": session["sha256"],
    }
    return {**body, "sha256": sha(body)}


def validate_scripter_transaction(
    case: dict,
    session: dict,
    request_receipt: dict,
    transaction: Any,
) -> dict:
    _exact(
        transaction,
        {
            "schema",
            "transaction_id",
            "request_id",
            "request_receipt_sha256",
            "session_id",
            "session_sha256_before",
            "raw_proposal",
            "raw_proposal_sha256",
            "visible_asset_ids_before",
            "status",
            "parsed_steps",
            "execution_receipt",
            "visible_asset_ids_after",
            "rejection",
            "auto_presented_asset_ids",
            "session_sha256_after_execution",
            "sha256",
        },
    )
    if transaction["schema"] != TRANSACTION_SCHEMA:
        raise BridgeError("unsupported scripter transaction")
    _id(transaction["transaction_id"])
    for key in (
        "request_receipt_sha256",
        "session_sha256_before",
        "raw_proposal_sha256",
        "session_sha256_after_execution",
        "sha256",
    ):
        _sha256(transaction[key])
    body = _transaction_body(transaction)
    if sha(body) != transaction["sha256"]:
        raise BridgeError("transaction hash mismatch")
    expected = execute_scripter_transaction(
        case,
        session,
        request_receipt,
        transaction["raw_proposal"],
    )
    if expected != transaction:
        raise BridgeError("transaction does not reproduce")
    return transaction


def apply_transaction_evidence(
    case: dict,
    session: dict,
    request_receipt: dict,
    transaction: dict,
    npc_id: str,
    asset_id: str,
    *,
    challenged_claim_ids: list[str] | None = None,
) -> dict:
    """Game explicitly presents one transaction-visible asset to a witness."""
    tx = validate_scripter_transaction(
        case,
        session,
        request_receipt,
        transaction,
    )
    if tx["status"] != "executed" or tx["execution_receipt"] is None:
        raise BridgeError("rejected transaction cannot present evidence")
    asset_id = _id(asset_id)
    if asset_id not in tx["visible_asset_ids_after"]:
        raise BridgeError("transaction does not make selected asset visible")
    _, assets, _ = _case_maps(case)
    if asset_id not in assets:
        raise BridgeError("transaction selected unknown asset")

    if assets[asset_id]["public"]:
        authority_receipt = None
    else:
        authority_receipt = tx["execution_receipt"]
    return session_api.present_evidence(
        case,
        session,
        npc_id,
        asset_id,
        authority_receipt=authority_receipt,
        challenged_claim_ids=challenged_claim_ids or [],
    )


def fixture_program(cfg: dict, credential: str) -> str:
    programme = copy.deepcopy(cfg["fixture"]["raw_program_template"])
    programme["steps"][0]["credential"] = credential
    return canonical(programme)


def run_fixture() -> dict:
    """Run deterministic denial/grant/rejection bridge fixture, no model."""
    cfg, case = load_contract()
    session = session_api.start_session(
        case,
        cfg["fixture"]["participants"],
        cfg["fixture"]["session_key"],
    )
    session_before_sha = session["sha256"]

    wrong_request = build_scripter_request(
        case,
        session,
        cfg["fixture"]["player_request"],
        cfg["fixture"]["visible_asset_ids"],
        cfg["fixture"]["disclosed_terminal_ids"],
        {
            "camera-console": cfg["fixture"]["wrong_credential"],
        },
    )
    wrong_tx = execute_scripter_transaction(
        case,
        session,
        wrong_request,
        fixture_program(cfg, cfg["fixture"]["wrong_credential"]),
    )
    if wrong_tx["status"] != "executed":
        raise BridgeError("wrong-credential fixture programme should parse/execute")
    if cfg["fixture"]["private_asset_id"] in wrong_tx["visible_asset_ids_after"]:
        raise BridgeError("wrong credential unexpectedly exposed private asset")
    try:
        apply_transaction_evidence(
            case,
            session,
            wrong_request,
            wrong_tx,
            cfg["fixture"]["present_private_asset_to"],
            cfg["fixture"]["private_asset_id"],
        )
    except BridgeError:
        wrong_apply_refused = True
    else:
        wrong_apply_refused = False
        raise BridgeError("wrong credential transaction presented private evidence")
    if session["sha256"] != session_before_sha:
        raise BridgeError("transaction execution mutated session")

    correct_request = build_scripter_request(
        case,
        session,
        cfg["fixture"]["player_request"],
        cfg["fixture"]["visible_asset_ids"],
        cfg["fixture"]["disclosed_terminal_ids"],
        {
            "camera-console": cfg["fixture"]["correct_credential"],
        },
    )
    correct_tx = execute_scripter_transaction(
        case,
        session,
        correct_request,
        fixture_program(cfg, cfg["fixture"]["correct_credential"]),
    )
    if (
        correct_tx["status"] != "executed"
        or cfg["fixture"]["private_asset_id"]
        not in correct_tx["visible_asset_ids_after"]
    ):
        raise BridgeError("correct credential did not expose private asset")
    session_after = apply_transaction_evidence(
        case,
        session,
        correct_request,
        correct_tx,
        cfg["fixture"]["present_private_asset_to"],
        cfg["fixture"]["private_asset_id"],
    )

    prompt = session_api.build_prompt(
        case,
        session_after,
        cfg["fixture"]["present_private_asset_to"],
        "What does the newly presented archive excerpt show?",
    )

    malformed_request = build_scripter_request(
        case,
        session,
        "Attempt only the permitted fictional evidence operations.",
        cfg["fixture"]["visible_asset_ids"],
        cfg["fixture"]["disclosed_terminal_ids"],
        {},
    )
    rejected_tx = execute_scripter_transaction(
        case,
        session,
        malformed_request,
        cfg["fixture"]["malformed_host_program"],
    )
    if rejected_tx["status"] != "proposal_rejected":
        raise BridgeError("host-operation fixture was not rejected")
    if rejected_tx["execution_receipt"] is not None:
        raise BridgeError("rejected proposal unexpectedly executed")

    return {
        "schema": "detective-003c-fixture-result-v1",
        "experiment": "DETECTIVE-003C",
        "issue": 144,
        "session_sha256_before": session_before_sha,
        "wrong_request": wrong_request,
        "wrong_transaction": wrong_tx,
        "wrong_apply_refused": wrong_apply_refused,
        "correct_request": correct_request,
        "correct_transaction": correct_tx,
        "session_after_private_evidence": session_after,
        "witness_prompt_after_private_evidence": prompt,
        "rejected_request": malformed_request,
        "rejected_transaction": rejected_tx,
        "case_sha256": hashlib.sha256(canonical(case).encode("utf-8")).hexdigest(),
        "decision": copy.deepcopy(cfg["decision"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not args.fixture:
        cfg, _ = load_contract()
        payload = {
            "contract": "DETECTIVE_003C_BRIDGE_CONTRACT_PASS",
            "issue": cfg["issue"],
            "model_inference": False,
            "optimizer_steps": 0,
            "population_expansion_admitted": False,
        }
    else:
        payload = run_fixture()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
