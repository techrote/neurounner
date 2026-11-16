"""D-1B offline feedback audit over *synthetic* before/after probes.

This cannot evaluate a real model, qualify native receipt custody or perform Undo.
The selected decision is an authored hypothetical, never an executable command.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import re
from .shadow_policy import reference_verdict

PROTOCOL = "cybersand.driv3r.d1b-feedback-audit/0"
SHA = re.compile(r"[0-9a-f]{64}\Z")
CHOICES = frozenset({"KEEP", "UNDO", "STOP"})
FIELDS = frozenset({"case_id", "decision_index", "scenario_sha256", "before", "after",
                    "hypothetical_undo_eligible", "hypothetical_receipt_available", "shadow_choice",
                    "case_sha256"})
METRICS = frozenset({"snapshot_id", "goal_score", "violations", "progress_cells"})


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _digest(obj: object) -> str:
    return sha256(json.dumps(obj, allow_nan=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def seal_case(case: dict) -> dict:
    """Seal an authored synthetic case; hash is *not* a native-state identity."""
    _require(type(case) is dict and set(case) == FIELDS - {"case_sha256"},
             "invalid synthetic case keys")
    result = json.loads(json.dumps(case, allow_nan=False))
    result["case_sha256"] = _digest(result)
    return result


def _snapshot(value: object) -> None:
    _require(type(value) is dict and set(value) == METRICS, "invalid metric snapshot")
    _require(type(value["snapshot_id"]) is str and 1 <= len(value["snapshot_id"]) <= 96,
             "invalid synthetic snapshot id")
    _require(type(value["goal_score"]) in (int, float) and
             math.isfinite(value["goal_score"]), "invalid goal score")
    for name in ("violations", "progress_cells"):
        _require(type(value[name]) is int and 0 <= value[name] <= 1000000,
                 "invalid metric count")


def audit_shadow_cases(cases: list[dict], *, max_forward_decisions: int = 36,
                       max_undo_proposals: int = 6) -> dict:
    """Compare hypothetical KEEP/UNDO/STOP with a deterministic reference.

    A STOP ends the hypothetical sequence. Repeated identical metric pairs after
    an UNDO are flagged as oscillation *risk*, never actual native undo/retry.
    No model probabilities or effects are inferred from authored choices.
    """
    _require(type(max_forward_decisions) is int and 1 <= max_forward_decisions <= 144,
             "invalid forward budget")
    _require(type(max_undo_proposals) is int and 0 <= max_undo_proposals <= 6,
             "invalid shadow undo budget")
    _require(type(cases) is list and 1 <= len(cases) <= max_forward_decisions,
             "invalid synthetic case collection")
    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    # An immutable frozen snapshot ID cannot name different objective metrics
    # elsewhere in the same scenario. Case-level hashes alone do not enforce
    # cross-case identity consistency; the input is synthetic, not native.
    snapshot_metrics: dict[str, tuple[int | float, int, int]] = {}
    reference_counts = {c: 0 for c in sorted(CHOICES)}
    confusion = {r: {c: 0 for c in sorted(CHOICES)} for r in sorted(CHOICES)}
    failures = {"false_keep": 0, "false_undo": 0, "false_stop": 0,
                "inadmissible_undo": 0, "oscillation_risk": 0}
    journal = []
    scenario = None
    previous_index = 0
    undo_count = 0
    terminated = False
    for case in cases:
        _require(not terminated, "synthetic STOP is terminal")
        _require(type(case) is dict and set(case) == FIELDS, "invalid synthetic case keys")
        _require(type(case["case_sha256"]) is str and SHA.fullmatch(case["case_sha256"]) is not None and
                 _digest({k: v for k, v in case.items() if k != "case_sha256"}) == case["case_sha256"],
                 "synthetic case hash mismatch")
        cid = case["case_id"]
        _require(type(cid) is str and 1 <= len(cid) <= 96 and cid not in seen_ids,
                 "duplicate or invalid case id")
        seen_ids.add(cid)
        index = case["decision_index"]
        _require(type(index) is int and previous_index < index <= max_forward_decisions,
                 "unordered or out-of-budget decision index")
        previous_index = index
        src = case["scenario_sha256"]
        _require(type(src) is str and SHA.fullmatch(src) is not None,
                 "invalid scenario source hash")
        _require(scenario is None or scenario == src, "unmatched synthetic scenario")
        scenario = src
        before, after = case["before"], case["after"]
        _snapshot(before)
        _snapshot(after)
        _require(before["snapshot_id"] != after["snapshot_id"],
                 "before/after synthetic snapshots must differ")
        for snapshot in (before, after):
            sid = snapshot["snapshot_id"]
            measures = (snapshot["goal_score"], snapshot["violations"],
                        snapshot["progress_cells"])
            if sid in snapshot_metrics:
                _require(snapshot_metrics[sid] == measures,
                         "conflicting synthetic snapshot identity")
            else:
                snapshot_metrics[sid] = measures
        eligible = case["hypothetical_undo_eligible"]
        receipt = case["hypothetical_receipt_available"]
        _require(type(eligible) is bool and type(receipt) is bool,
                 "hypothetical control flags must be booleans")
        choice = case["shadow_choice"]
        _require(type(choice) is str and choice in CHOICES, "invalid shadow choice")
        reference, _reference_reason = reference_verdict(
            before, after, undo_eligible=eligible,
            verified_native_receipt=receipt)
        reference_counts[reference] += 1
        confusion[reference][choice] += 1
        if choice == "KEEP" and reference != "KEEP":
            failures["false_keep"] += 1
        if choice == "STOP" and reference != "STOP":
            failures["false_stop"] += 1
        if choice == "UNDO":
            undo_count += 1
            _require(undo_count <= max_undo_proposals,
                     "shadow undo proposals exceed authorised research bound")
            if not (eligible and receipt):
                failures["inadmissible_undo"] += 1
            if reference != "UNDO":
                failures["false_undo"] += 1
            # Compare *metric* pairs, not volatile snapshot identifiers: a
            # retried edit can generate new IDs for identical objective states.
            pair_sig = _digest({"before": {k: v for k, v in before.items() if k != "snapshot_id"},
                                "after": {k: v for k, v in after.items() if k != "snapshot_id"}})
            if pair_sig in seen_pairs:
                failures["oscillation_risk"] += 1
            seen_pairs.add(pair_sig)
        journal.append({"case_id": cid, "decision_index": index,
                        "case_sha256": case["case_sha256"], "reference": reference,
                        "shadow_choice": choice, "agreement": reference == choice,
                        "undo_admissible_if_native_were_qualified": eligible and receipt})
        terminated = choice == "STOP"
    return {"protocol": PROTOCOL, "evidence_class": "synthetic_only",
            "scenario_sha256": scenario, "input_case_hashes": [c["case_sha256"] for c in cases],
            "audit_sha256": _digest(cases), "forward_decision_indices_seen": len(journal),
            "hypothetical_undo_proposals": undo_count, "reference_counts": reference_counts,
            "confusion": confusion, "errors": failures, "journal": journal,
            "native_material_writes": 0, "native_undo_executions": 0,
            "model_inference_performed": False, "actual_receipts_verified": False}
