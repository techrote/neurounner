#!/usr/bin/env python3
"""Independent full-position Sprocket publisher-wire loss-mask replay.

Reads the BYTE-EXACT original PR #161 physical receipt from the PR #162 Git
custody bundle; downloads only pinned tokenizer/config/card files. Never loads
a model, performs an optimizer step, uses a held-out fixture or admits #72.
This is a new source/serialization recheck, not a new physical-host result.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr001-sprocket-wire-fullmask"
HOST_SOURCE = ROOT / "scripts/ncr_001_postlease_sprocket_wire.py"
HOST_HEAD = "aadbfe823c9778066acac0a0869e0c559a533f05"
WIRE_HEAD = "3592841802d85c442ca2521479b5d8795bee7233"
WIRE_BLOB = "4e558ae530a1b79050e22bee587b3caf056e4563"
WIRE_PATH = "scripts/ncr_001e_sprocket_publisher_wire.py"
CARD_SHA = "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"
EXPECTED_BRANCH = "review/ncr001-sprocket-wire-fullmask-20261028"
ROLE_IDS = {"system": 2, "user": 3, "assistant": 4, "tool": 7, "memory": 10}
LABEL_COUNTS = {"assistant": True, "system": False, "user": False,
                "tool": False, "memory": False}
REPORT = {
    "schema_version": 1, "experiment": "NCR-001E", "lane": "sprocket_instruct",
    "status": "STARTED", "physical_model_executed_this_run": False,
    "independent_scientific_review": False, "full_72_conformance_pass": False,
    "train_authorized": False, "optimizer_steps": 0,
    "comparative_training_performed": False, "heldouts_opened": False,
    "model_generated_tool_memory_semantics_qualified": False,
    "historical_model_output": "INCOMPLETE_SIX_TOKEN_TOOL_CALL_PREFIX",
    "retained_original_failed_source_run": 40718490313,
    "original_physical_run": 40718706396,
}

def demand(ok, reason):
    if not ok:
        raise RuntimeError("NCR001_FULLMASK_REFUSED: " + reason)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "receipt.json").write_text(json.dumps(REPORT, sort_keys=True, indent=2)
                                      + "\n", encoding="utf-8")

def extract_synthetic_turns(source: Path) -> list[dict[str, str]]:
    """Extract only the literal diagnostic fixture from unchanged host source."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "synthetic_turns"]
    demand(len(matches) == 1, "synthetic fixture source missing/duplicated")
    returns = [n for n in ast.walk(matches[0]) if isinstance(n, ast.Return)]
    demand(len(returns) == 1, "synthetic fixture must have a single literal return")
    turns = ast.literal_eval(returns[0].value)
    demand(isinstance(turns, list) and len(turns) == 12,
           "original 12-turn synthetic diagnostic changed")
    demand(all(isinstance(t, dict) and set(t) == {"role", "content"}
               and t["role"] in ROLE_IDS and type(t["content"]) is str
               and t["content"] for t in turns),
           "unexpected source diagnostic turn")
    return turns

def independently_rebuild(turns: list[dict[str, str]], tokenizer):
    """Per-turn body supervision, without calling publisher label builder.

    Header is always context, body and END are assistant-only supervised.
    Tool-result and host-memory bodies must not receive any assistant loss.
    """
    ids: list[int] = []
    labels: list[bool] = []
    for turn in turns:
        role = turn["role"]
        body = tokenizer.encode(turn["content"], add_special_tokens=False)
        demand(isinstance(body, list) and bool(body)
               and all(type(t) is int and 0 <= t < 32000 for t in body),
               "invalid body tokenizer IDs")
        trained = LABEL_COUNTS[role]
        ids.extend([ROLE_IDS[role], *body, 5])
        labels.extend([False, *([trained] * len(body)), trained])
    ids.append(ROLE_IDS["assistant"])
    labels.append(False)
    return ids, labels

def check_full_mask(ids, observed, independent_ids, expected):
    demand(len(ids) == len(observed) == len(independent_ids) == len(expected),
           "full mask/source length drift")
    demand(all(type(x) is int for x in ids + independent_ids)
           and all(type(x) is bool for x in observed + expected),
           "mask or token type drift")
    demand(ids == independent_ids, "independent role/body token reconstruction mismatch")
    demand(observed == expected, "assistant/host body label drift at one or more positions")
    return {"units": len(ids), "assistant_units": sum(expected),
            "host_units": len(expected) - sum(expected),
            "all_positions_checked": True}

def original_receipt():
    sys.path.insert(0, str(ROOT / "scripts"))
    import ncr_001_postlease_wire_evidence_audit as custody
    summary = custody.audit()
    demand(summary.get("retained_original_failures") is True
           and summary.get("full_72_admitted") is False
           and summary.get("training_authorized") is False,
           "original archive audit not fail-closed")
    archives = custody.original_zip_bytes()
    name = "artifact-12828893309.zip"
    members = custody.raw_zip_members(archives[name], custody.ORIGINAL_ZIPS[name])
    return summary, json.loads(members["receipt.json"])

def checked_protocol_and_source(receipt):
    protocol_raw = (ROOT / "experiments/ncr-001/protocol.json").read_bytes()
    pins_raw = (ROOT / "experiments/ncr-001/source-pins.json").read_bytes()
    protocol, pins = json.loads(protocol_raw), json.loads(pins_raw)
    require_keys = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
    demand(protocol["train_authorized"] is False
           and set(protocol["models"]) == set(pins["models"]) == require_keys,
           "four-lane frozen protocol changed")
    demand(pins["models"]["sprocket_instruct"]["hf_weight_file"] == "model.safetensors"
           and pins["models"]["sprocket_chat"]["hf_weight_file"] == "chat/model.safetensors"
           and pins["models"]["sprocket_instruct"]["weight_declared_lfs_sha256_not_downloaded"]
           != pins["models"]["sprocket_chat"]["weight_declared_lfs_sha256_not_downloaded"],
           "separate Sprocket checkpoint identity lost")
    demand(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip() == os.environ["NCR_EXPECTED_HEAD"]
           and os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
           "not an exact PR-head source-only run")
    demand(subprocess.check_output(
        ["git", "-C", str(ROOT / "wire-authority"), "rev-parse", "HEAD"],
        text=True).strip() == WIRE_HEAD, "reviewed publisher tree changed")
    demand(subprocess.check_output(
        ["git", "-C", str(ROOT / "wire-authority"), "hash-object", WIRE_PATH],
        text=True).strip() == WIRE_BLOB, "reviewed publisher renderer changed")
    recorded = receipt["source_sha256"]
    demand(sha(HOST_SOURCE.read_bytes()) ==
           recorded["scripts/ncr_001_postlease_sprocket_wire.py"],
           "original physical source script changed")
    return pins["models"]["sprocket_instruct"], {
        "protocol_git_blob": subprocess.check_output(
            ["git", "hash-object", "experiments/ncr-001/protocol.json"],
            cwd=ROOT, text=True).strip(),
        "source_pins_git_blob": subprocess.check_output(
            ["git", "hash-object", "experiments/ncr-001/source-pins.json"],
            cwd=ROOT, text=True).strip(),
        "physical_source_sha256": sha(HOST_SOURCE.read_bytes()),
        "publisher_renderer_git_blob": WIRE_BLOB,
    }

def verified_tokenizer(pin):
    """Download only root Instruct small source files; verify BEFORE loading."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    demand(pin["repo_id"] == "kandivault/sprocket-500m"
           and pin["revision"] == "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
           "root Instruct source revision drift")
    expected = {"README.md": CARD_SHA,
                "config.json": pin["config_sha256"],
                "tokenizer.json": pin["tokenizer_sha256"],
                "tokenizer_config.json": pin["tokenizer_config_sha256"],
                "special_tokens_map.json": pin["special_tokens_map_sha256"]}
    files = {}
    for name, target in expected.items():
        path = Path(hf_hub_download(pin["repo_id"], name, revision=pin["revision"]))
        files[name] = {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
        demand(files[name]["sha256"] == target, "pinned root source mismatch: " + name)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(pin["repo_id"],
        revision=pin["revision"], local_files_only=True, trust_remote_code=False)
    for marker, expected_id in (("<|end|>", 5), ("<|tool_call|>", 6),
                                ("<|tool_result|>", 7), ("<|memory_read|>", 10),
                                ("<|memory_write|>", 11)):
        demand(tokenizer.encode(marker, add_special_tokens=False) == [expected_id],
               "special marker tokenization changed: " + marker)
    tokenizer.pad_token_id = 5  # Same explicit model-host transport setting.
    return tokenizer, files

def run():
    archive, original = original_receipt()
    pin, sources = checked_protocol_and_source(original)
    REPORT["original_archive"] = archive
    REPORT["source"] = sources
    save()
    turns = extract_synthetic_turns(HOST_SOURCE)
    sys.path.insert(0, str(ROOT / "wire-authority" / "scripts"))
    from ncr_001e_sprocket_publisher_wire import render_ids_mask, render_text
    tokenizer, files = verified_tokenizer(pin)
    REPORT["tokenizer_source"] = files
    save()
    rendered_ids, rendered_labels = render_ids_mask(
        turns, tokenizer, variant="instruct")
    independent_ids, independent_labels = independently_rebuild(turns, tokenizer)
    checked = check_full_mask(rendered_ids, rendered_labels,
                              independent_ids, independent_labels)
    demand(len(rendered_ids) == 123 and sum(rendered_labels) == 45,
           "original 123-unit, 45-assistant-label diagnostic changed")
    orig = original["wire"]
    text = render_text(turns, variant="instruct")
    short, _ = render_ids_mask(turns[:3], tokenizer, variant="instruct")
    offset = len(rendered_ids) - len(short)
    attention = ([0] * offset + [1] * len(short)
                 + [1] * len(rendered_ids))
    observed = {
        "prompt_sha256": sha(text.encode("utf-8")),
        "ids_sha256": sha(struct.pack("<" + "q" * len(rendered_ids), *rendered_ids)),
        "assistant_mask_sha256": sha(bytes(int(x) for x in independent_labels)),
        "attention_mask_sha256": sha(struct.pack(
            "<" + "q" * len(attention), *attention)),
    }
    for field, value in observed.items():
        demand(value == orig[field], "archived real-host witness disagrees: " + field)
    demand(offset == orig["synthetic_pad_positions"] == 97
           and len(turns) == 12 and orig["source_end_id5_count"] == 12,
           "original source end markers/padding drift")
    REPORT["full_position_mask"] = checked
    REPORT["original_host_receipt_sha256"] = (
        "44f59fdf8eaa65ff6c4721b7f9c0c40986297442cf93382dce9201c83f33dc6e"
    )
    REPORT["replayed_hashes"] = observed
    REPORT["runner"] = {
        "head_sha": os.environ["NCR_EXPECTED_HEAD"],
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "name": os.environ.get("RUNNER_NAME"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "physical_model_executed": False,
    }
    REPORT["status"] = "PASS_FULL_POSITION_SOURCE_MASK_REPLAY__72_REVIEW_BLOCKED"
    save()
    return REPORT

def main():
    save()
    try:
        result = run()
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        REPORT["status"] = "FAIL_SOURCE_MASK_REPLAY__NO_ADMISSION"
        REPORT["failure"] = {"kind": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        save()
        raise

if __name__ == "__main__":
    main()
