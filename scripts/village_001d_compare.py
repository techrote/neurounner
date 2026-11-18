#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
from pathlib import Path

from babble_002_expand import extract_json, packet_for
from village_001_manifest import build_manifest
from village_001b_generate_member import call_api

ROOT = Path(__file__).resolve().parents[1]
SPECIALIZATION_GUIDANCE = {
    "A": (
        "failure-hardened: awkward failure/recovery, conflicting motives, physical "
        "reaction, imperfect settling"
    ),
    "B": (
        "local tinkerer: bounded local memory, mundane habits, maintenance choices, "
        "practical physical interaction"
    ),
    "C": (
        "social scavenger: bargaining, opportunism, social friction, embodied choices, "
        "conflicting motives"
    ),
}
SCORE_KEYS = (
    "task_relevance",
    "genericness",
    "lineage_fit",
    "style_dependence",
    "behavior_fit",
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def member_contract(campaign: dict, member_id: str) -> tuple[dict, dict]:
    population = build_manifest(campaign)
    for row in population["members"]:
        if row["member_id"] == member_id:
            return population, row
    raise ValueError(f"unknown member {member_id}")


def orientation(member_id: str, scenario_id: str) -> str:
    digest = hashlib.sha256(f"{member_id}:{scenario_id}".encode()).digest()
    return "parent-x" if digest[0] % 2 == 0 else "child-x"


def build_cases(member: dict, parent: list[dict], child: list[dict]) -> list[dict]:
    if len(parent) != 24 or len(child) != 24:
        raise RuntimeError("parent/child held-out response count must be exactly 24")
    parent_by_id = {row["scenario_id"]: row for row in parent}
    child_by_id = {row["scenario_id"]: row for row in child}
    expected = [f"T{i:02d}" for i in range(1, 25)]
    if sorted(parent_by_id) != expected or sorted(child_by_id) != expected:
        raise RuntimeError("parent/child held-out scenario sets are incomplete")

    cases = []
    for scenario_id in expected:
        p = parent_by_id[scenario_id]
        c = child_by_id[scenario_id]
        if p["prompt"] != c["prompt"]:
            raise RuntimeError(f"{scenario_id}: parent/child prompt drift")
        orient = orientation(member["member_id"], scenario_id)
        if orient == "parent-x":
            x_text, y_text = p["response"], c["response"]
        else:
            x_text, y_text = c["response"], p["response"]
        cases.append(
            {
                "scenario_id": scenario_id,
                "family": p["family"],
                "prompt": p["prompt"],
                "X": x_text,
                "Y": y_text,
                "_orientation": orient,
            }
        )
    return cases


def judge_prompt(member: dict, packet: dict, cases: list[dict]) -> str:
    public_cases = [
        {k: value for k, value in case.items() if not k.startswith("_")}
        for case in cases
    ]
    sibling_defs = "\n".join(
        f"- {key}: {value}" for key, value in SPECIALIZATION_GUIDANCE.items()
    )
    return f"""Blindly compare two responses per scenario.

Target ancestral lineage summary:
{packet['persona_summary']}

The three sibling ecologies are:
{sibling_defs}

The actual target sibling is NOT stated here. Infer it from behavior.

For every response label X and Y, return integer 0..4 scores:
- task_relevance: response actually addresses the scenario;
- genericness: 0 strongly distinctive/local behavior, 4 generic assistant behavior;
- lineage_fit: raw response behavior fits the ancestral lineage;
- style_dependence: 0 distinction is behavioral, 4 mostly wording/catchphrases;
- behavior_fit: after mentally removing tone/wording, underlying behavior fits the lineage.

Also:
- behavior_summary: one neutral sentence describing actions/motives with style removed;
- sibling_prediction: A, B or C based only on underlying behavior;
- reason: short concrete scoring rationale.

Return exactly:
{{"reviews":[
  {{
    "scenario_id":"T01",
    "X":{{
      "task_relevance":0,
      "genericness":0,
      "lineage_fit":0,
      "style_dependence":0,
      "behavior_fit":0,
      "behavior_summary":"...",
      "sibling_prediction":"A",
      "reason":"..."
    }},
    "Y":{{ "...":"same fields" }}
  }}
]}}

Do not guess which response is earlier/later training. Judge X and Y independently.
Return JSON only, one review per supplied scenario, no extras.

CASES:
{json.dumps(public_cases, ensure_ascii=False)}
"""


def validate_side(side: dict, scenario_id: str, label: str) -> dict:
    if not isinstance(side, dict):
        raise ValueError(f"{scenario_id}:{label}: side not object")
    out = {}
    for key in SCORE_KEYS:
        value = side.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 4
        ):
            raise ValueError(f"{scenario_id}:{label}:{key}")
        out[key] = value
    summary = side.get("behavior_summary")
    reason = side.get("reason")
    sibling = side.get("sibling_prediction")
    if not isinstance(summary, str) or len(summary.strip()) < 8:
        raise ValueError(f"{scenario_id}:{label}:behavior_summary")
    if not isinstance(reason, str) or len(reason.strip()) < 4:
        raise ValueError(f"{scenario_id}:{label}:reason")
    if sibling not in {"A", "B", "C"}:
        raise ValueError(f"{scenario_id}:{label}:sibling_prediction")
    out.update(
        {
            "behavior_summary": summary.strip(),
            "sibling_prediction": sibling,
            "reason": reason.strip(),
        }
    )
    return out


def parse_reviews_partial(
    raw: str,
    cases: list[dict],
    *,
    salvage_partial: bool = True,
) -> tuple[dict[str, dict], list[str]]:
    expected = [case["scenario_id"] for case in cases]
    expected_set = set(expected)
    problems: list[str] = []
    header = re.match(r'\s*(?:```(?:json)?\s*)?\{\s*"reviews"\s*:\s*\[', raw)
    try:
        if salvage_partial:
            # Bind to the first outer JSON value (optionally fenced).
            # extract_json searches for later JSON and is reserved below for
            # verifying already-terminal historical comparisons.
            prefix = re.match(r'\s*(?:```(?:json)?\s*)?', raw)
            obj, _ = json.JSONDecoder().raw_decode(raw, prefix.end())
        else:
            obj = extract_json(raw)
    except ValueError:
        obj = None
    reviews = obj.get("reviews") if isinstance(obj, dict) else None
    if not isinstance(reviews, list):
        if not salvage_partial or not header:
            return {}, ["reviews_not_list"]
        # Recover only an unambiguous prefix of independently complete array
        # elements. Stop at the first syntax error; never scan inside a broken
        # element, beyond the array closure, or for plausible-looking suffixes.
        decoder = json.JSONDecoder()
        reviews = []
        position = header.end()
        while position < len(raw):
            while position < len(raw) and raw[position].isspace():
                position += 1
            if position == len(raw) or raw[position] == "]":
                break
            try:
                row, end = decoder.raw_decode(raw, position)
            except json.JSONDecodeError:
                break
            separator = end
            while separator < len(raw) and raw[separator].isspace():
                separator += 1
            if separator == len(raw) or raw[separator] not in {",", "]"}:
                break
            reviews.append(row)
            if raw[separator] == "]":
                break
            position = separator + 1
        problems.append("review_json_complete_prefix_only")

    out: dict[str, dict] = {}
    for row in reviews:
        if not isinstance(row, dict):
            problems.append("review_not_object")
            continue
        scenario_id = row.get("scenario_id")
        if scenario_id not in expected_set or scenario_id in out:
            continue
        try:
            parsed = {
                "scenario_id": scenario_id,
                "X": validate_side(row.get("X"), scenario_id, "X"),
                "Y": validate_side(row.get("Y"), scenario_id, "Y"),
            }
        except ValueError as exc:
            problems.append(str(exc))
            continue
        out[scenario_id] = parsed

    missing = [scenario_id for scenario_id in expected if scenario_id not in out]
    if missing:
        problems.append("review_missing:" + ",".join(missing))
    return out, problems


def validate_reviews(raw: str, cases: list[dict]) -> list[dict]:
    expected = [case["scenario_id"] for case in cases]
    out, problems = parse_reviews_partial(raw, cases)
    missing = [scenario_id for scenario_id in expected if scenario_id not in out]
    if missing:
        raise ValueError(
            "review scenarios incomplete: "
            + ",".join(missing)
            + (f" problems={problems}" if problems else "")
        )
    return [out[scenario_id] for scenario_id in expected]


def score_summary(rows: list[dict], correct_sibling: str) -> dict:
    result = {
        key: statistics.mean(row[key] for row in rows)
        for key in SCORE_KEYS
    }
    result["sibling_accuracy"] = (
        sum(1 for row in rows if row["sibling_prediction"] == correct_sibling)
        / len(rows)
    )
    result["raw_vs_behavior_gap"] = (
        result["lineage_fit"] - result["behavior_fit"]
    )
    return result


def request_payload(member: dict, packet: dict, cases: list[dict], model_id: str,
                    *, repair: bool) -> dict:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": (
                "Score only the supplied missing blind-review scenarios. "
                "Return final JSON only, one review per scenario."
                if repair else
                "You are a blind behavioral evaluator. Separate behavior from "
                "wording and score both candidate responses independently. "
                "Return JSON only."
            )},
            {"role": "user", "content": judge_prompt(member, packet, cases)},
        ],
        "max_tokens": min(6144, max(2048, 768 * len(cases))) if repair else 6144,
        "temperature": 0.1 if repair else 0.2,
        "top_p": 0.9,
        "stream": False,
    }


def repair_batch_size(round_index: int) -> int:
    if type(round_index) is not int or round_index not in (1, 2, 3):
        raise ValueError("comparison repair round must be 1, 2 or 3")
    return (3, 2, 1)[round_index - 1]


def replay_provider_history(provider: Path, member: dict, packet: dict,
                            cases: list[dict], model_id: str, *,
                            salvage_partial: bool = True) -> tuple[dict, set, int]:
    """Replay source-bound original payloads without requesting another judgment."""
    prefix = re.escape(member["member_id"]) + r"-before-after-(0[1-4])-bolt"
    pattern = re.compile(prefix + r"(?:-targeted-repair([1-3])|"
                         r"-resume-([0-9]+)-targeted-repair([1-3])-batch([0-9]+))?$")
    records = []
    requested = set()
    generation = 0
    for path in sorted(provider.glob("*-request.json")):
        label = path.name.removesuffix("-request.json")
        match = pattern.fullmatch(label)
        if not match:
            raise RuntimeError(f"unrecognized comparison request identity: {label}")
        chunk_no = int(match[1])
        old_round = int(match[2] or 0)
        epoch = int(match[3] or 0)
        repair_round = int(match[4] or old_round)
        batch = int(match[5] or 0)
        generation = max(generation, epoch)
        chunk = cases[(chunk_no - 1) * 6:chunk_no * 6]
        repair = bool(old_round or epoch)
        record = json.loads(path.read_text())
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError(f"missing archived comparison payload: {label}")
        try:
            public = json.loads(payload["messages"][1]["content"].rsplit("CASES:\n", 1)[1])
            ids = [row["scenario_id"] for row in public]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid archived comparison cases: {label}") from exc
        selected = [case for case in chunk if case["scenario_id"] in ids]
        if not ids or ids != [case["scenario_id"] for case in selected]:
            raise RuntimeError(f"archived comparison scenario identity/order mismatch: {label}")
        if not repair and selected != chunk:
            raise RuntimeError(f"archived primary comparison chunk mismatch: {label}")
        expected = request_payload(member, packet, selected, model_id, repair=repair)
        role = "before-after-critic-targeted-repair" if repair else "before-after-critic"
        if payload != expected or record.get("provider") != "nvidia" or record.get("role") != role:
            raise RuntimeError(f"archived comparison request/source mismatch: {label}")
        requested.add(label)
        content_path = provider / f"{label}-content.txt"
        response_path = provider / f"{label}-provider.json"
        if not content_path.is_file() and not response_path.is_file():
            # A recorded failed transport request has no model judgment to replay.
            continue
        if not content_path.is_file() or not response_path.is_file():
            raise RuntimeError(f"incomplete archived comparison response custody: {label}")
        response = json.loads(response_path.read_text())
        try:
            content = (response["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"invalid archived comparison provider response: {label}") from exc
        if content_path.read_text().strip() != content:
            raise RuntimeError(f"archived comparison content/provider mismatch: {label}")
        records.append(((chunk_no, epoch, repair_round, batch), content, selected))
    reviews = {}
    for _, raw, selected in sorted(records, key=lambda row: row[0]):
        parsed, _ = parse_reviews_partial(raw, selected, salvage_partial=salvage_partial)
        for scenario_id, review in parsed.items():
            # First valid judgment is permanent; later repairs cannot cherry-pick.
            reviews.setdefault(scenario_id, review)
    return reviews, requested, generation


def collect_reviews(*, member: dict, packet: dict, cases: list[dict], model_id: str,
                    provider: Path, endpoint: str, key: str, attempts_limit: int,
                    offline: bool = False) -> list[dict]:
    reviews, requested, previous_generation = replay_provider_history(
        provider, member, packet, cases, model_id)
    recovered = len(reviews)
    calls = 0

    def checkpoint() -> None:
        write_json(provider.parent / "recovery.json", {
            "schema_version": 1,
            "member_id": member["member_id"],
            "recovered_reviews": recovered,
            "new_provider_requests": calls,
            "validated_scenario_ids": [c["scenario_id"] for c in cases if c["scenario_id"] in reviews],
            "missing_scenario_ids": [c["scenario_id"] for c in cases if c["scenario_id"] not in reviews],
            "reviews": [reviews[c["scenario_id"]] for c in cases if c["scenario_id"] in reviews],
        })

    def request(selected: list[dict], label: str, *, repair: bool) -> None:
        nonlocal calls
        if offline or not key:
            missing = [c["scenario_id"] for c in cases if c["scenario_id"] not in reviews]
            raise RuntimeError("comparison requires fresh provider reviews: " + ",".join(missing))
        if any(provider.glob(f"{label}-*")):
            raise RuntimeError(f"refusing to overwrite comparison provider history: {label}")
        calls += 1
        checkpoint()
        raw, _, _ = call_api(
            key=key, endpoint=endpoint,
            payload=request_payload(member, packet, selected, model_id, repair=repair),
            outdir=provider, label=label, attempts_limit=attempts_limit,
            role="before-after-critic-targeted-repair" if repair else "before-after-critic",
        )
        parsed, _ = parse_reviews_partial(raw, selected)
        for scenario_id, review in parsed.items():
            reviews.setdefault(scenario_id, review)
        checkpoint()

    checkpoint()
    for chunk_index in range(0, len(cases), 6):
        chunk = cases[chunk_index:chunk_index + 6]
        label = f"{member['member_id']}-before-after-{chunk_index // 6 + 1:02d}-bolt"
        missing = [c for c in chunk if c["scenario_id"] not in reviews]
        if missing == chunk and label not in requested:
            # No original request exists for this chunk. All cases are new.
            request(missing, label, repair=False)
        for repair_round in range(1, 4):
            missing = [c for c in chunk if c["scenario_id"] not in reviews]
            if not missing:
                break
            size = repair_batch_size(repair_round)
            for index in range(0, len(missing), size):
                selected = missing[index:index + size]
                repair_label = (f"{label}-resume-{previous_generation + 1:03d}"
                                f"-targeted-repair{repair_round}-batch{index // size + 1:02d}")
                request(selected, repair_label, repair=True)
        missing = [c["scenario_id"] for c in chunk if c["scenario_id"] not in reviews]
        if missing:
            raise ValueError("before/after reviews incomplete after targeted repair: " + ",".join(missing))
    return [reviews[case["scenario_id"]] for case in cases]


def build_evaluation(member: dict, cases: list[dict], all_reviews: list[dict], *,
                     contract_sha: str, parent_sha: str, child_sha: str,
                     model_id: str) -> dict:
    parent_rows = []
    child_rows = []
    mapped = []
    reviews_by_id = {row["scenario_id"]: row for row in all_reviews}
    for case in cases:
        review = reviews_by_id[case["scenario_id"]]
        if case["_orientation"] == "parent-x":
            parent_side, child_side = review["X"], review["Y"]
            parent_label, child_label = "X", "Y"
        else:
            parent_side, child_side = review["Y"], review["X"]
            parent_label, child_label = "Y", "X"
        parent_rows.append(parent_side)
        child_rows.append(child_side)
        mapped.append(
            {
                "scenario_id": case["scenario_id"],
                "family": case["family"],
                "orientation": case["_orientation"],
                "parent_label": parent_label,
                "child_label": child_label,
                "parent": parent_side,
                "child": child_side,
            }
        )

    before = score_summary(parent_rows, member["sibling"])
    after = score_summary(child_rows, member["sibling"])
    delta = {
        key: after[key] - before[key]
        for key in (
            *SCORE_KEYS,
            "sibling_accuracy",
            "raw_vs_behavior_gap",
        )
    }

    relevance_gate = delta["task_relevance"] >= -0.5
    style_only_regression = (
        delta["lineage_fit"] > 0
        and delta["behavior_fit"] < 0
        and delta["raw_vs_behavior_gap"] > 0.25
    )
    behavior_gate = not style_only_regression

    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "stage": "before-after-evaluation",
        "issue": 13,
        "member_id": member["member_id"],
        "lineage": member["lineage"],
        "sibling": member["sibling"],
        "member_contract_sha256": contract_sha,
        "parent_adapter_sha256": parent_sha,
        "preference_adapter_sha256": child_sha,
        "judge_model": model_id,
        "blind_orientation_rule": "sha256(member_id:scenario_id) first-byte parity",
        "scenario_count": 24,
        "before": before,
        "after": after,
        "delta": delta,
        "gates": {
            "task_relevance_delta_at_least_minus_0_5": relevance_gate,
            "no_raw_only_behavior_regression": behavior_gate,
            "style_only_regression_detected": style_only_regression,
            "qualified_for_population_tournament": (
                relevance_gate and behavior_gate
            ),
        },
        "reviews": mapped,
    }
    return result


def prepare_output(out: Path, resume: Path | None, identity: dict) -> None:
    if resume is not None:
        source, target = resume.resolve(), out.resolve()
        if source == target or source in target.parents or target in source.parents:
            raise RuntimeError("comparison resume and output must be disjoint directories")
        receipt = resume / "comparison-identity.json"
        if receipt.is_file() and json.loads(receipt.read_text()) != identity:
            raise RuntimeError("comparison resume identity mismatch")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("comparison output must be empty; use a separate --resume directory")
    if resume is not None:
        shutil.copytree(resume, out, dirs_exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
    write_json(out / "comparison-identity.json", identity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/preference-config.json",
    )
    parser.add_argument("--member", required=True)
    parser.add_argument("--preference-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--offline", action="store_true", help="Replay archived judgments only; refuse missing reviews")
    args = parser.parse_args()

    pref = json.loads(args.config.read_text())
    campaign = json.loads((ROOT / pref["campaign_config"]).read_text())
    population, member = member_contract(campaign, args.member)
    packet = packet_for(member["lineage"])

    parent_path = (
        ROOT / pref["sft_evidence_root"] / "results" / f"{member['member_id']}.json"
    )
    parent_result = json.loads(parent_path.read_text())
    child_result = json.loads(args.preference_result.read_text())

    if parent_result.get("member_contract_sha256") != population[
        "member_contract_sha256"
    ]:
        raise RuntimeError("parent member contract mismatch")
    if child_result.get("member_contract_sha256") != population[
        "member_contract_sha256"
    ]:
        raise RuntimeError("preference member contract mismatch")
    if child_result.get("member_id") != member["member_id"]:
        raise RuntimeError("preference result member mismatch")
    if child_result.get("parent_adapter_sha256") != parent_result["adapter"][
        "sha256"
    ]:
        raise RuntimeError("preference parent adapter hash mismatch")

    parent_responses = parent_result["evaluation"]["responses"]
    child_responses = child_result["evaluation"]["responses"]
    cases = build_cases(member, parent_responses, child_responses)

    model = campaign["nvidia"]["models"]["critic"]
    identity = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "stage": "before-after-evaluation",
        "member_id": member["member_id"],
        "member_contract_sha256": population["member_contract_sha256"],
        "parent_result_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "preference_result_sha256": hashlib.sha256(args.preference_result.read_bytes()).hexdigest(),
        "campaign_config_sha256": hashlib.sha256((ROOT / pref["campaign_config"]).read_bytes()).hexdigest(),
        "preference_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "judge_model": model["id"],
    }
    out = args.out
    prepare_output(out, args.resume, identity)
    # Existing terminal results are fixed evidence. Recompute them from original
    # source-bound transcripts, accepting either the historical whole-batch
    # parser or the new complete-object salvage policy, without rejudging.
    old_evaluation = out / "evaluation.json"
    if old_evaluation.is_file():
        archived = json.loads(old_evaluation.read_text())
        for salvage in (False, True):
            reviews, _, _ = replay_provider_history(
                out / "provider", member, packet, cases, model["id"],
                salvage_partial=salvage)
            if set(reviews) != {case["scenario_id"] for case in cases}:
                continue
            replayed = build_evaluation(
                member, cases, [reviews[c["scenario_id"]] for c in cases],
                contract_sha=population["member_contract_sha256"],
                parent_sha=parent_result["adapter"]["sha256"],
                child_sha=child_result["adapter"]["sha256"], model_id=model["id"],
            )
            if replayed == archived:
                write_json(out / "recovery.json", {
                    "schema_version": 1, "member_id": member["member_id"],
                    "terminal_result_reused": True, "recovered_reviews": 24,
                    "new_provider_requests": 0,
                    "replay_partial_object_salvage": salvage,
                })
                print(json.dumps({"member": member["member_id"],
                                  "resume": "complete_artifact_reused"}))
                return
        raise RuntimeError("archived evaluation differs from source-bound provider replay")
    all_reviews = collect_reviews(
        member=member, packet=packet, cases=cases, model_id=model["id"],
        provider=out / "provider", endpoint=campaign["nvidia"]["endpoint"],
        key=os.environ.get("NVIDIA_API_KEY", "").strip(),
        attempts_limit=int(pref["pair_generation"]["attempts"]), offline=args.offline,
    )
    result = build_evaluation(
        member, cases, all_reviews,
        contract_sha=population["member_contract_sha256"],
        parent_sha=parent_result["adapter"]["sha256"],
        child_sha=child_result["adapter"]["sha256"], model_id=model["id"],
    )
    write_json(out / "evaluation.json", result)
    print(json.dumps({"member": member["member_id"], "gates": result["gates"],
                      "delta": result["delta"]}, indent=2))


if __name__ == "__main__":
    main()
