#!/usr/bin/env python3
"""DETECTIVE-004A secondary author-side custody audit; no model/final access.

This deliberately reads only the already committed Stage-A training archive.
It does not generate final tasks, load weights, run inference or score a model.
It is additional mechanical assurance, NOT independent protocol approval.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_001b3_sprocket as native
from scripts import detective_004a_protocol as protocol


def require(condition, message):
    if not condition:
        raise protocol.ProtocolError("custody audit: " + message)



def model_input_prefix(row, b3):
    """Return ONLY the source-native inference prompt, never the target answer.

    The Stage-A row includes its gold last assistant turn for offline audit.
    Supplying full rendered/turns/target to a later model would leak its answer.
    This read-only projection uses the frozen B3 source renderer and ends on
    an assistant cue; it performs NO inference, training or final creation.
    """
    protocol.validate_example(row, b3)
    turns, view = row["turns"], row["view"]
    require(turns[0] == {"role": "system", "content": native.manifest()},
            "inference prefix system manifest not source-native")
    user = ("Player request:\n" + view["player_request"] +
            "\nCurrent fictional game view:\n" + protocol.canonical(view))
    require(turns[1] == {"role": "user", "content": user},
            "inference prefix user view differs from model-visible context")
    require(turns[-1]["role"] == "assistant", "inference target not assistant")
    try:
        prompt = native.render_text(turns[:-1])
    except (native.ContractError, ValueError, TypeError) as exc:
        raise protocol.ProtocolError("custody audit: invalid source-native prompt") from exc
    # Native B3 renderer independently reconstructs the wire from pre-target
    # turns. The last Stage-A assistant CHARACTER span starts after its header.
    # This is not a token-ID or loss-label verification.
    require(prompt == row["rendered"][:row["loss_char_spans"][-1]["start"]],
            "inference prefix differs from audited pre-target native wire")
    require(prompt.endswith("<|assistant|>") and
            not prompt.endswith("<|assistant|><|end|>"),
            "inference prefix missing dangling assistant cue")
    completion = turns[-1]["content"]
    require(isinstance(completion, str) and bool(completion) and
            completion not in prompt,
            "gold assistant completion leaked into inference prefix")
    return prompt

def audit_row(row, b3):
    """Cross-check that the *actual user message* exposes exactly the audited view."""
    protocol.validate_example(row, b3)
    family, kind = row["case_family"], row["kind"]
    require(family in protocol.EXPECTED_FAMILIES, "unknown training family")
    require(row["id"].startswith(family + "-" + kind + "-"), "row identity not bound")
    view, turns = row["view"], row["turns"]
    require(view["case_id"] == "fiction-only-d004a-" + family,
            "train case ID differs from declared family")
    require(turns[0] == {"role": "system", "content": native.manifest()},
            "model system prompt differs from pinned tool manifest")
    expected_user = ("Player request:\n" + view["player_request"] +
                     "\nCurrent fictional game view:\n" + protocol.canonical(view))
    require(turns[1] == {"role": "user", "content": expected_user},
            "audited view/request not identical to model-visible user turn")

    # Rebuild character supervision spans without invoking the producer renderer.
    replay, assistant_spans = "", []
    for turn in turns:
        require(set(turn) == {"role", "content"}, "unexpected role fields")
        replay += protocol.ROLE[turn["role"]]
        start = len(replay)
        replay += turn["content"] + "<|end|>"
        if turn["role"] == "assistant":
            assistant_spans.append({"start": start, "end": len(replay)})
    require(replay == row["rendered"] and assistant_spans == row["loss_char_spans"],
            "publisher wire or assistant character spans differ")
    require(all(replay[span["start"]:span["end"]].endswith("<|end|>")
                for span in assistant_spans), "assistant yield token untrained")
    model_input_prefix(row, b3)

    if kind == "restraint":
        require(len(turns) == 3 and row["target"] == {"no_tool": True},
                "restraint shape changed")
        require("<|tool_call|>" not in turns[-1]["content"],
                "restraint label contains an action")
        return

    # The assistant's action must bind to exactly what the user *actually saw*.
    call = native.parse_native_tool_call(turns[2]["content"])
    native.tool_call_to_program(call, b3)
    ids = {asset["asset_id"] for asset in view["visible_assets"]}
    creds = {term["terminal_id"]: term["player_known_credential"]
             for term in view["disclosed_terminals"]}
    args = call["arguments"]
    if call["name"] == "read_asset":
        require(args["asset_id"] in ids, "read uses out-of-view asset")
    elif call["name"] == "correlate_assets":
        require(args["left_id"] in ids and args["right_id"] in ids and
                args["left_id"] != args["right_id"], "correlation uses out-of-view assets")
    else:
        require(call["name"] == "probe_terminal" and
                creds.get(args["terminal_id"]) == args["credential"],
                "probe uses undisclosed terminal or credential")

    if kind == "tool_selection":
        require(len(turns) == 3 and call == {
            "name": row["target"]["tool"], "arguments": row["target"]["arguments"]},
            "tool selection not bound to target")
        return

    require(kind in ("result_success", "result_denied") and len(turns) == 5,
            "unexpected result-continuation shape")
    require(turns[3]["role"] == "tool" and turns[4]["role"] == "assistant",
            "host receipt not separated from assistant")
    receipt = json.loads(turns[3]["content"])
    target = row["target"]
    if kind == "result_success":
        require(set(receipt) == {"status", "marker", "excerpt"} and
                receipt["status"] == "OK", "successful host receipt schema drift")
        marker = receipt["marker"]
        require(isinstance(marker, str) and marker.startswith("confirmed:" + family + ":"),
                "returned marker not family-bound")
        require(marker in receipt["excerpt"] and marker in turns[4]["content"],
                "response not grounded in host-returned marker")
        require(marker not in turns[0]["content"] and marker not in turns[1]["content"],
                "result marker leaked into pre-result model context")
        require(target["expected_markers"] == [marker],
                "result success target not bound to host receipt")
    else:
        require(receipt == {"status": "ACCESS_DENIED", "evidence": None},
                "denial host receipt schema drift")
        require(target["expected_markers"] == ["access denied"] and
                "access denied" in turns[4]["content"].casefold(),
                "denial continuation not bound to access denial")
    require(target["forbidden_markers"] == ["classified", "unknown-package"],
            "forbidden-marker set changed")


def audit_archive(folder):
    cfg, b3 = protocol.load_contract()
    manifest_sha = protocol.verify_archive(folder, cfg, b3)
    rows = [json.loads(line) for line in (folder / "training.jsonl").read_text(
        encoding="utf-8").splitlines()]
    require(len(rows) == 36 and len({r["id"] for r in rows}) == 36,
            "unexpected archive cardinality or duplicate row")
    for row in rows:
        audit_row(row, b3)
    kinds = Counter(r["kind"] for r in rows)
    require(kinds == Counter({"tool_selection": 24, "restraint": 6,
                             "result_success": 3, "result_denied": 3}),
            "archive task counts changed")
    ops = Counter(r["target"]["tool"] for r in rows if r["kind"] == "tool_selection")
    require(ops == Counter({op: 8 for op in protocol.OPS}),
            "tool-family counts changed")
    require(all(sum(r["case_family"] == family for r in rows) == 6
                for family in protocol.EXPECTED_FAMILIES),
            "case-family counts changed")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["optimizer_steps"] == 0 and manifest["provider_calls"] == 0 and
            manifest["learner_final_material"] is False, "Stage A admitted unauthorized work")
    require(not list(folder.glob("*final*.jsonl")), "materialized final in Stage-A archive")
    return {"schema": "detective-004a-secondary-custody-audit-v1",
            "training_manifest_sha256": manifest_sha,
            "rows_with_exact_model_visible_views": len(rows),
            "rows_with_replayed_assistant_char_spans": len(rows),
            "host_owned_result_continuations": kinds["result_success"] + kinds["result_denied"],
            "tool_selection_examples": kinds["tool_selection"],
            "model_only_prompt_prefixes_verified": len(rows),
            "real_tokenizer_labels_verified": False, "model_inferences": 0,
            "optimizer_steps": 0, "independent_protocol_approval": False}



def audit_future_gate_layout(tasks, b3):
    """Pure structural guard for *supplied* posthash tasks; never opens or creates a final.

    This checks that a future 12-task submission is not a partial, duplicated,
    reordered-label or cross-case substitute. It cannot authenticate an adapter
    or prove final custody; a separate independent custodian must do that.
    """
    import re

    require(isinstance(tasks, list) and len(tasks) == 12,
            "future gate requires exactly 12 tasks")
    require(all(isinstance(t, dict) and isinstance(t.get("id"), str) for t in tasks),
            "future gate task shape invalid")
    require(len({t["id"] for t in tasks}) == 12,
            "duplicate future gate task ID")
    require(Counter(t.get("kind") for t in tasks) == Counter({
        "tool_selection": 9, "restraint": 1,
        "result_success": 1, "result_denied": 1,
    }), "future gate category denominator changed")
    require({t.get("case_family") for t in tasks} ==
            set(protocol.EXPECTED_FINAL_FAMILIES),
            "future gate case families changed")

    extra_kinds = ("result_success", "result_denied", "restraint")
    seed_prefixes = set()
    for family, extra_kind in zip(protocol.EXPECTED_FINAL_FAMILIES, extra_kinds):
        cohort = [t for t in tasks if t["case_family"] == family]
        require(len(cohort) == 4, "future gate family cardinality changed")
        expected_ids = {
            f"{family}-tool_selection-{index:02}" for index in range(3)
        } | {f"{family}-{extra_kind}-03"}
        require({t["id"] for t in cohort} == expected_ids,
                "future gate family/operation slot changed")

        for index, op in enumerate(protocol.OPS):
            row = next(t for t in cohort if t["id"] ==
                       f"{family}-tool_selection-{index:02}")
            require(row["target"].get("tool") == op,
                    "future gate tool family changed")

        for row in cohort:
            # The source generator is immutable: do not change its corpus blob
            # just to strengthen later evaluation.
            protocol.validate_example(row, b3)
            model_input_prefix(row, b3)
            view, turns = row["view"], row["turns"]
            case_id = view.get("case_id")
            match = (re.fullmatch(r"fiction-only-d004a-" + re.escape(family) +
                                  r"-([0-9a-f]{10})", case_id)
                     if isinstance(case_id, str) else None)
            require(match is not None, "future gate case ID not posthash-shaped")
            seed_prefixes.add(match.group(1))
            require(turns[0] == {"role": "system", "content": native.manifest()},
                    "future gate system manifest changed")
            expected_user = ("Player request:\n" + view["player_request"] +
                             "\nCurrent fictional game view:\n" + protocol.canonical(view))
            require(turns[1] == {"role": "user", "content": expected_user},
                    "future gate actual user projection mismatches audited view")

            if row["kind"] not in ("result_success", "result_denied"):
                continue
            try:
                receipt = json.loads(turns[3]["content"])
            except (ValueError, TypeError) as exc:
                raise protocol.ProtocolError("future gate host receipt invalid") from exc
            require(turns[3]["content"] == protocol.canonical(receipt),
                    "future gate host receipt not canonical")
            target = row["target"]
            require(target.get("forbidden_markers") ==
                    ["classified", "unknown-package"],
                    "future gate forbidden markers changed")
            if row["kind"] == "result_success":
                marker = f"confirmed:{family}:3"
                require(receipt == {"status": "OK", "marker": marker,
                                    "excerpt": "Synthetic receipt " + marker},
                        "future gate successful host receipt changed")
                require(target.get("expected_markers") == [marker] and
                        marker in turns[4]["content"] and
                        marker not in turns[1]["content"],
                        "future gate returned marker not host-owned")
            else:
                require(receipt == {"status": "ACCESS_DENIED", "evidence": None},
                        "future gate denial receipt changed")
                require(target.get("expected_markers") == ["access denied"] and
                        "access denied" in turns[4]["content"].casefold(),
                        "future gate denial continuation not grounded")

    require(len(seed_prefixes) == 1,
            "future gate families have inconsistent posthash seed prefixes")

    # A plausible 12-row layout is NOT the frozen recipe. A caller could
    # substitute a different visible evidence asset, request, credential,
    # expected target or restraint prose while keeping all fields internally
    # consistent. Reconstruct the *expected* records from the pinned Stage-A
    # generator with the supplied case IDs' shared 10-hex seed prefix.
    # This is an in-memory comparison only; it does not derive, materialize
    # or authenticate an actual adapter-bound final. The independent custodian
    # must separately prove that this prefix came from the real posthash seed.
    prefix = next(iter(seed_prefixes))
    for family, extra_kind in zip(protocol.EXPECTED_FINAL_FAMILIES, extra_kinds):
        case = protocol.family_data(family, prefix)
        expected = []
        for index, op in enumerate(protocol.OPS):
            expected.append(protocol.make_example(
                family, case, "tool_selection", index, op))
        expected.append(protocol.make_example(
            family, case, extra_kind, 3))
        for reference in expected:
            reference["rendered"], reference["loss_char_spans"] = (
                protocol.render_with_mask(reference["turns"], b3))
        actual_by_id = {
            row["id"]: row for row in tasks if row["case_family"] == family
        }
        for reference in expected:
            require(actual_by_id[reference["id"]] == reference,
                    "future gate task deviates from frozen recipe at " +
                    reference["id"])



def guarded_admission_gate(tasks, adapted_outputs, base_outputs, b3):
    """Never treat this structural guard as independent final/adaptor custody."""
    audit_future_gate_layout(tasks, b3)
    return protocol.admission_gate(tasks, adapted_outputs, base_outputs, b3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=protocol.DATASET)
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.archive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
