"""Read-only, source-fenced native/Q8 soft-guidance rank calibration.

Run: python -m experiments.driv3r.stage_d.guidance.native_guidance_rank FREE.zip SOFT.zip
Audits the actual native receipts and original model responses before interpreting
per-decision scalar ranks. No model, Workbench, Undo, network, or source writes.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

from .native_pair_shadow import MOVES, SOURCE, audit_pair

PROTOCOL = "cybersand.driv3r.d1b-native-guidance-rank/1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def score_arm(turns: list[dict]) -> dict:
    """Rank only real painted MOVE selections, conditioning the null on MOVE.

    The equal-probability baseline is *among legal painting moves*, not STOP,
    BACKTRACK or relocation. It is descriptive, not a counterfactual model run.
    """
    _require(type(turns) is list and len(turns) > 0, "nonempty turn list required")
    selected_max = 0
    expected_max = 0.0
    ranks = []
    max_mass = []
    offered = []
    chosen_values = []
    for i, turn in enumerate(turns):
        _require(type(turn) is dict and turn.get("index") == i,
                 "turn identity/order mismatch")
        ids = turn["exact_candidate_ids_in_presented_order"]
        hints = turn["model_state"]["candidate_hints"]
        _require(type(ids) is list and len(ids) == len(set(ids)) and
                 set(hints) == set(ids) & set(MOVES), "incomplete legal move hints")
        _require(all(type(h["soft_scent_u16"]) is int and
                     0 <= h["soft_scent_u16"] <= 65535 for h in hints.values()),
                 "invalid scalar guidance")
        probabilities = turn["parsed_choice"]["probabilities"]
        _require(set(probabilities) == set(ids) and
                 all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1
                     for p in probabilities.values()) and
                 abs(math.fsum(probabilities.values()) - 1) < 1e-5,
                 "invalid original model-choice probabilities")
        choice = turn["chosen_candidate"]
        _require(choice in ids and choice == turn["parsed_choice"]["candidate_id"],
                 "chosen candidate unbound to original model response")
        if choice not in MOVES:
            continue
        _require(bool(hints), "no legal painting destination")
        values = {k: h["soft_scent_u16"] for k, h in hints.items()}
        chosen = values[choice]
        maximum = max(values.values())
        top = {k for k, v in values.items() if v == maximum}
        count = len(values)
        selected_max += int(choice in top)
        expected_max += len(top) / count
        ranks.append((sum(v < chosen for v in values.values()) +
                      0.5 * sum(v == chosen for v in values.values())) / count)
        mass_on_top = math.fsum(probabilities[k] for k in top)
        move_mass = math.fsum(probabilities[k] for k in values)
        _require(move_mass > 0, "zero model probability of every legal MOVE")
        max_mass.append(mass_on_top / move_mass)
        offered.append(math.fsum(values.values()) / count)
        chosen_values.append(chosen)
    _require(bool(ranks), "no actual painted MOVE decisions to rank")
    n = len(ranks)
    return {
        "painted_move_count": n,
        "chosen_highest_guide_count": selected_max,
        "uniform_legal_move_expected_highest_count": round(expected_max, 9),
        "highest_count_minus_uniform_expectation": round(selected_max - expected_max, 9),
        "mean_chosen_tie_aware_percentile": round(math.fsum(ranks) / n, 9),
        "mean_chosen_guide_u16": round(math.fsum(chosen_values) / n, 6),
        "mean_offered_legal_guide_u16": round(math.fsum(offered) / n, 6),
        "mean_model_probability_on_highest_guide_conditional_on_move":
            round(math.fsum(max_mass) / n, 9),
        "null_condition": "uniform among legal MOVE candidates, conditional on MOVE",
    }


def matching_prefix(free: list[dict], soft: list[dict]) -> dict:
    """Compare probabilities only before the *actual* states diverge."""
    prefix = []
    for i, (a, b) in enumerate(zip(free, soft)):
        sa, sb = a["model_state"], b["model_state"]
        same = (a["index"] == b["index"] == i and
                sa["position"] == sb["position"] and
                sa["recent_positions"] == sb["recent_positions"] and
                sa["native_world_material_sha256"] == sb["native_world_material_sha256"] and
                a["exact_candidate_ids_in_presented_order"] ==
                b["exact_candidate_ids_in_presented_order"] and
                a["host_legal_candidate_sha256"] == b["host_legal_candidate_sha256"])
        if not same:
            break
        ids = a["exact_candidate_ids_in_presented_order"]
        pa, pb = a["parsed_choice"]["probabilities"], b["parsed_choice"]["probabilities"]
        _require(set(pa) == set(pb) == set(ids), "matched-state probability catalog mismatch")
        prefix.append({"turn": i, "same_choice": a["chosen_candidate"] == b["chosen_candidate"],
                       "chosen_free": a["chosen_candidate"], "chosen_soft": b["chosen_candidate"],
                       "largest_absolute_option_probability_shift": round(
                           max(abs(pb[k] - pa[k]) for k in ids), 9)})
    return {"matched_initial_decision_state_count": len(prefix),
            "first_choice_divergence_in_matched_prefix": next(
                (x["turn"] for x in prefix if not x["same_choice"]), None),
            "matched_prefix": prefix}


def _load_turns(z: ZipFile, count: int) -> list[dict]:
    return [json.loads(z.read(f"execution/model/turn-{i:02}.json"))
            for i in range(count)]


def analyze_pair(free_path: Path, soft_path: Path) -> dict:
    # Native, numeric guide and original Q8 response custody is verified first.
    provenance = audit_pair(free_path, soft_path)
    _require(provenance["source_sha"] == SOURCE and
             provenance["matched_initial_world"] and provenance["matched_initial_legal_catalog"],
             "original source/native comparison cannot be attested")
    with ZipFile(free_path) as fz, ZipFile(soft_path) as sz:
        free = _load_turns(fz, provenance["free"]["decisions"])
        soft = _load_turns(sz, provenance["soft"]["decisions"])
    result = {
        "protocol": PROTOCOL, "evidence_class": "archived_original_Q8_native_readonly_calibration",
        "source_sha": SOURCE, "m1_source_commit": provenance["m1_source_commit"],
        "archive_sha256": {a: provenance[a]["zip_sha256"] for a in ("free", "soft")},
        "arms": {"free": score_arm(free), "soft": score_arm(soft)},
        "shared_state_comparison": matching_prefix(free, soft),
        "qualification": "underlying original archive and real native receipt audit passed",
        "interpretation_limits": [
            "One original seed; the soft and free worlds diverge after the fifth decision",
            "Uniform legal-MOVE null is descriptive, not a matched replay/control policy",
            "Model option probabilities are not calibrated causal effects of scent",
            "No multimodal d1, new native execution, actual material Undo or aesthetics measured",
        ],
    }
    # A concise deterministic byte-level result hash permits independent replay.
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["content_sha256_without_self_digest"] = sha256(encoded.encode()).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("Usage: native_guidance_rank.py FREE.zip SOFT.zip", file=sys.stderr)
        return 2
    print(json.dumps(analyze_pair(Path(args[0]), Path(args[1])), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
