#!/usr/bin/env python3
"""Source-only causal-shift/left-pad alignment for original Sprocket Instruct wire.

Replays the original pinned tokenizer and archived physical receipt; never loads
model weights, trains an adapter, opens held-outs, or claims #72 acceptance.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_sprocket_wire_fullmask as fullmask
import ncr_001_sprocket_wire_fullmask_custody as custody

OUT = ROOT / "runtime/ncr001-sprocket-wire-shifted-loss"
PARENT_HEAD = "8ca28b8aa8c43f97795d1149a7563bee9444763b"
SOURCE_HEAD = "aadbfe823c9778066acac0a0869e0c559a533f05"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
STATUS = "PASS_CAUSAL_SHIFTED_MASK_SOURCE_ONLY__FINAL_72_BLOCKED"


def refuse(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("NCR001_SHIFTED_WIRE_REFUSED: " + reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def int64_sha(rows: list[list[int]]) -> str:
    flattened = [value for row in rows for value in row]
    return sha(struct.pack("<" + "q" * len(flattened), *flattened))


def construct(long_ids: list[int], long_mask: list[bool],
              short_ids: list[int], short_mask: list[bool],
              pad_id: int = 5) -> dict:
    """Build causal loss labels independently of publisher mask code.

    Labels at position j are predicted by logits at j-1. A supervised label
    must have both the target and its preceding context position attended.
    The host-only shorter prefix must not acquire supervision from padding.
    """
    refuse(type(pad_id) is int and pad_id == 5, "pinned END/pad ID changed")
    refuse(all(type(xs) is list for xs in
               (long_ids, long_mask, short_ids, short_mask)),
           "expected explicit typed lists")
    refuse(len(long_ids) == len(long_mask) and
           len(short_ids) == len(short_mask) and
           5 < len(short_ids) < len(long_ids) <= 2048,
           "source sequence/mask lengths invalid")
    refuse(all(type(x) is int and 0 <= x < 32000
               for x in long_ids + short_ids)
           and all(type(x) is bool for x in long_mask + short_mask),
           "source token or supervision types invalid")
    refuse(long_ids[0] == 2 and long_ids[-1] == 4 and
           not long_mask[0] and not long_mask[-1],
           "source role headers or pending assistant cue invalid")
    refuse(not any(short_mask), "host-only short prefix became supervised")
    refuse(long_ids.count(5) == 12 and short_ids.count(5) == 3,
           "original real END=5 markers absent or spoofed")
    refuse(sum(long_mask) > 0 and long_ids[0:len(short_ids)-1] ==
           short_ids[0:len(short_ids)-1],
           "source short prefix or assistant loss absent")
    # The short prefix ends in an assistant cue. It must match long prefix
    # up to (but not including) that cue: its next long unit is a new role.
    refuse(short_ids[-1] == 4 and not short_mask[-1],
           "short host prefix has no unsupervised assistant cue")
    offset = len(long_ids) - len(short_ids)
    input_ids = [[pad_id] * offset + short_ids, list(long_ids)]
    attention = [[0] * offset + [1] * len(short_ids),
                 [1] * len(long_ids)]
    labels = [[-100] * offset +
              [token if label else -100
               for token, label in zip(short_ids, short_mask)],
              [token if label else -100
               for token, label in zip(long_ids, long_mask)]]
    supervised = []
    for row in range(2):
        for j in range(len(long_ids)):
            if labels[row][j] == -100:
                continue
            refuse(j > 0 and attention[row][j] == 1 and
                   attention[row][j-1] == 1,
                   "causal target sees missing/padded predecessor")
            refuse(input_ids[row][j] == labels[row][j],
                   "loss target not actual next token")
            supervised.append((row, j))
    refuse(len(supervised) == sum(long_mask)
           and all(row == 1 for row, _ in supervised),
           "supervision leaked into short host/pad row")
    refuse(any(token == 5 and attend == 1
               for row, mask in zip(input_ids, attention)
               for token, attend in zip(row, mask)),
           "real END=5 hidden with padding")
    return {"input_ids": input_ids, "attention_mask": attention,
            "labels": labels, "offset": offset,
            "supervised_positions": supervised,
            "shifted_targets": [row[1:] for row in labels]}


def validate(expected: dict, observed: dict) -> None:
    for key in ("input_ids", "attention_mask", "labels", "shifted_targets"):
        refuse(observed.get(key) == expected[key],
               "padded causal transport mismatch: " + key)
    refuse(observed.get("offset") == expected["offset"],
           "synthetic left-pad offset drift")


def exact_source() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()
    refuse(head == os.environ.get("NCR_EXPECTED_HEAD")
           and os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
           "not executing exact pull-request head")
    blobs = {}
    for name in ("experiments/ncr-001/protocol.json",
                 "experiments/ncr-001/source-pins.json"):
        blobs[name] = subprocess.check_output(
            ["git", "hash-object", name], cwd=ROOT, text=True).strip()
    refuse(blobs["experiments/ncr-001/protocol.json"] == PROTOCOL_BLOB
           and blobs["experiments/ncr-001/source-pins.json"] == PINS_BLOB,
           "protocol or four-source pins changed")
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    pins = json.loads((ROOT / "experiments/ncr-001/source-pins.json").read_text())
    keys = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
    refuse(protocol["train_authorized"] is False
           and set(protocol["models"]) == set(pins["models"]) == keys,
           "scientific admission or model identities changed")
    refuse(pins["models"]["sprocket_chat"]["hf_weight_file"] ==
           "chat/model.safetensors"
           and pins["models"]["sprocket_instruct"]["hf_weight_file"] ==
           "model.safetensors"
           and pins["models"]["sprocket_instruct"]
           ["weight_declared_lfs_sha256_not_downloaded"] !=
           pins["models"]["sprocket_chat"]
           ["weight_declared_lfs_sha256_not_downloaded"],
           "distinct frozen Sprocket variants collapsed")
    return {"head": head, "blobs": blobs,
            "pin": pins["models"]["sprocket_instruct"]}


def run() -> dict:
    source = exact_source()
    archived = custody.audit()
    refuse(archived["source_head"] ==
           "40efcb5748894c6c7847c18b1ad442a455bd0aa3"
           and archived["train_authorized"] is False
           and archived["new_model_execution"] is False,
           "original fullmask custody no longer authoritative")
    _, original = fullmask.original_receipt()
    refuse(original["lane"] == "sprocket_instruct"
           and original["original_frozen_model_executed"] is True
           and original["train_authorized"] is False
           and original["optimizer_steps"] == 0,
           "original physical frozen-host witness invalid")
    source_hash = sha(fullmask.HOST_SOURCE.read_bytes())
    refuse(source_hash == original["source_sha256"]
           ["scripts/ncr_001_postlease_sprocket_wire.py"],
           "original 12-turn physical diagnostic source changed")
    turns = fullmask.extract_synthetic_turns(fullmask.HOST_SOURCE)
    tokenizer, verified_files = fullmask.verified_tokenizer(source["pin"])
    long_ids, long_mask = fullmask.independently_rebuild(turns, tokenizer)
    short_ids, short_mask = fullmask.independently_rebuild(turns[:3], tokenizer)
    fullmask.check_full_mask(long_ids, long_mask, long_ids, long_mask)
    refuse(len(long_ids) == 123 and sum(long_mask) == 45
           and len(short_ids) == 26, "original source tokenizer units changed")
    observed_source = original["wire"]
    refuse(sha(struct.pack("<" + "q" * len(long_ids), *long_ids)) ==
           observed_source["ids_sha256"]
           and sha(bytes(map(int, long_mask))) ==
           observed_source["assistant_mask_sha256"],
           "original frozen-model host token/label receipt differs")
    expected = construct(long_ids, long_mask, short_ids, short_mask)
    refuse(expected["offset"] == 97 and
           int64_sha(expected["attention_mask"]) ==
           observed_source["attention_mask_sha256"],
           "original physical padded attention receipt differs")
    refuse(len(expected["supervised_positions"]) == 45,
           "causal shift lost original assistant targets")
    naive = {**expected, "attention_mask":
             [[int(token != 5) for token in row]
              for row in expected["input_ids"]]}
    refused_naive = False
    try:
        validate(expected, naive)
    except ValueError:
        refused_naive = True
    refuse(refused_naive, "naive end-equals-padding attention sentinel failed")
    return {
        "status": STATUS, "experiment": "NCR-001E",
        "lane": "sprocket_instruct", "parent_pr": 182,
        "parent_head": PARENT_HEAD, "source": source["blobs"],
        "source_head": source["head"], "original_physical_run": 40718706396,
        "original_source_fullmask_run": 40782101613,
        "original_custody_run": 40783012701,
        "original_physical_source_sha256": source_hash,
        "tokenizer_files": verified_files, "long_units": len(long_ids),
        "short_units": len(short_ids), "synthetic_left_pad_units": 97,
        "real_end_id": 5, "real_end_units_long": long_ids.count(5),
        "assistant_supervised_units": 45,
        "padded_attention_sha256": int64_sha(expected["attention_mask"]),
        "causal_shifted_labels_sha256": int64_sha(expected["shifted_targets"]),
        "unshifted_labels_sha256": int64_sha(expected["labels"]),
        "naive_pad_equals_end_refused": refused_naive,
        "runner": {"name": os.environ.get("RUNNER_NAME"),
                   "run_id": os.environ.get("GITHUB_RUN_ID"),
                   "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                   "python": platform.python_version()},
        "new_model_execution": False, "model_weights_downloaded": False,
        "optimizer_steps": 0, "comparative_training": False,
        "heldouts_opened": False, "full_72_conformance_pass": False,
        "independent_scientific_review": False, "train_authorized": False,
        "generated_tool_memory_behaviour_qualified": False,
        "historical_incomplete_tool_call_preserved": True,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        result = run()
    except Exception as exc:
        result = {"status": "FAIL_CAUSAL_SHIFTED_MASK__NO_ADMISSION",
                  "error": type(exc).__name__, "detail": str(exc),
                  "traceback": traceback.format_exc(),
                  "train_authorized": False, "full_72_conformance_pass": False}
        (OUT / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (OUT / "receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
