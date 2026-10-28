#!/usr/bin/env python3
"""DETECTIVE-004A secondary author-side custody audit; no model/final access.

This deliberately reads only the already committed Stage-A training archive.
It does not generate final tasks, load weights, run inference or score a model.
It is additional mechanical assurance, NOT independent protocol approval.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import detective_001b3_sprocket as native
from scripts import detective_004a_protocol as protocol


def require(condition, message):
    if not condition:
        raise protocol.ProtocolError("custody audit: " + message)


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
            "real_tokenizer_labels_verified": False, "model_inferences": 0,
            "optimizer_steps": 0, "independent_protocol_approval": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=protocol.DATASET)
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.archive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
