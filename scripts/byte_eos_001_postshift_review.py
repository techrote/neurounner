#!/usr/bin/env python3
"""Independent review of the MERGED #163 corrected void.1 future EOS labels.

This is a distinct, source-pinned successor to the immutable original #126
review ZIP. No model weights, optimizer, training or held-out fixture access.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CORRECTED_COMMIT = "bb60d700fe86cf530773f1f639a246cc24bc0943"
CORRECTED_BLOB = "4c00c78b6776ade6901a99eee9a42e7ca61c698b"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
SOURCE_PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
ORIGINAL_REVIEW_HEAD = "54eda2e177d4dbaebcff84cc09a08add1900226c"
TOKENIZER_SHA256 = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
VOID_REVISION = "9c2be5087772867d45961abd11f6b35110617563"
PREREG_COMMIT = "a1b41c358d215974ea9ef2748baa13c3872b0fbb"
EOS, MAX_UNITS = 258, 1024


def require(condition, why):
    if not condition:
        raise RuntimeError("POSTSHIFT_REVIEW_REFUSED: " + why)


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


class ByteTokenizer:
    """Offline synthetic byte codec only; never a real-runtime substitute."""
    vocab_size = 259
    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=True):
        return list(text.encode("utf-8", errors="strict"))


def oracle(prompt, completion, *, max_units=MAX_UNITS):
    """Independently derive exactly the shifted-supervised byte-token labels."""
    if type(prompt) is not str or type(completion) is not str:
        raise TypeError("Unicode text required")
    if type(max_units) is not int or not 1 <= max_units <= MAX_UNITS:
        raise ValueError("unapproved max_units")
    p = prompt.encode("utf-8", errors="strict")
    c = completion.encode("utf-8", errors="strict")
    if len(p) == 0:
        raise ValueError("causal decoder must have a prompt position")
    ids = list(p + c) + [EOS]
    if len(ids) > max_units:
        raise RuntimeError("terminal EOS would exceed cap; no truncation")
    labels = [-100] * len(p) + list(c) + [EOS]
    shifted = labels[1:]
    assert sum(i != -100 for i in shifted) == len(c) + 1
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "sequence_tokenizer_units": len(ids),
        "supervised_tokenizer_units": len(c) + 1,
        "causal_shift_supervised_units": len(c) + 1,
        "prompt_tokenizer_units": len(p),
        "utf8_source_bytes": len(p) + len(c),
        "source_characters": len(prompt + completion),
        "terminal_eos_token_id": EOS,
        "terminal_eos_supervised": True,
        "no_truncation": True,
        "serializer_version": "BYTE-EOS-001-v1",
        "historical_train002b_corrected": False,
    }


def source_integrity(root):
    root = Path(root)
    require(git(root, "rev-parse", "HEAD") == CORRECTED_COMMIT,
            "corrected main checkout is not the reviewed commit")
    require(git(root, "hash-object", "scripts/byte_eos_001_void_labels.py")
            == CORRECTED_BLOB, "corrected #163 serializer Git blob changed")
    require(git(root, "hash-object", "experiments/ncr-001/protocol.json")
            == PROTOCOL_BLOB, "four-model protocol changed")
    require(git(root, "hash-object", "experiments/ncr-001/source-pins.json")
            == SOURCE_PINS_BLOB, "source pins changed")
    protocol = json.loads((root / "experiments/ncr-001/protocol.json").read_text())
    pins = json.loads((root / "experiments/ncr-001/source-pins.json").read_text())
    require(protocol["train_authorized"] is False
            and set(protocol["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "four-model/no-training admission policy changed")
    m = protocol["models"]["void1"]
    p = pins["models"]["void1"]
    require(m["repo_id"] == p["repo_id"] == "dotlabs/void.1"
            and m["revision"] == p["revision"] == VOID_REVISION
            and m["tokenizer_reference"]["ids"] ==
            {"pad": 256, "bos": 257, "eos": 258}
            and m["tokenizer_reference"]["train002b_source_sha256"] ==
            p["tokenization_byte_py_sha256"] == TOKENIZER_SHA256,
            "real void.1 source identity changed")
    prereg = ROOT / "prereg"
    require(git(prereg, "rev-parse", "HEAD") == PREREG_COMMIT,
            "original prereg checkout missing")
    frozen = json.loads((prereg / "experiments/ncr-001/protocol.json").read_text())
    require(frozen["train_authorized"] is False
            and frozen["training"]["per_model_sequence_unit_limits"]["void1"] == MAX_UNITS,
            "locked 1024-unit budget changed")
    return p


def load_candidate(root):
    path = Path(root) / "scripts/byte_eos_001_void_labels.py"
    spec = importlib.util.spec_from_file_location("reviewed_corrected_eos_labels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    require(mod.VOID_EOS == EOS and mod.VOID_NCR_MAX_UNITS == MAX_UNITS,
            "corrected code constants changed")
    return mod


def positive_cases():
    # Public synthetic strings only; no protected or preregistered held-outs.
    return [
        ("ascii", "Task: ", "inspect bearing"),
        ("unicode", "πΩ→🔧 ", "café 日本語"),
        ("literal_eos_content", "User wrote: ", "<eos>"),
        ("empty_completion", "x", ""),
        ("one_byte_prefix", "x", "π"),
        ("emoji_boundary", "🙂" * 7, "🔧" * 5),
        ("max_1024", "x", "y" * 1022),
    ]


def verify(candidate, tokenizer):
    cases = []
    for name, prompt, completion in positive_cases():
        expected = oracle(prompt, completion)
        actual = candidate.encode_supervised_byte_completion(tokenizer, prompt, completion)
        require(actual == expected, name + " differs from shifted independent oracle")
        require(actual["labels"][-1] == actual["input_ids"][-1] == EOS,
                name + " terminal true EOS not supervised")
        require(EOS not in actual["input_ids"][:-1],
                name + " ordinary UTF-8 contains hidden EOS")
        cases.append({"name": name, "sequence_units": len(actual["input_ids"]),
                      "shifted_supervised": actual["causal_shift_supervised_units"],
                      "oracle_match": True})
    # This is the previously missed #126 defect: an unprefixed first label is
    # not predicted by the causal shift, even when the unshifted mask says yes.
    for completion in ("", "A", "π", "<eos>"):
        try:
            candidate.encode_supervised_byte_completion(tokenizer, "", completion)
        except ValueError as e:
            require("at least one prompt byte" in str(e), "wrong empty-prompt refusal")
        else:
            raise RuntimeError("POSTSHIFT_REVIEW_REFUSED: old #126 empty-prompt false PASS")
    for prompt, completion in (("x", "y" * 1023), ("", "x" * 1023)):
        try:
            candidate.encode_supervised_byte_completion(tokenizer, prompt, completion)
        except (RuntimeError, ValueError):
            pass
        else:
            raise RuntimeError("POSTSHIFT_REVIEW_REFUSED: cap/empty-prompt false acceptance")
    for bad in (True, 0, 1025):
        try:
            candidate.encode_supervised_byte_completion(
                tokenizer, "x", "A", max_units=bad)
        except ValueError:
            pass
        else:
            raise RuntimeError("POSTSHIFT_REVIEW_REFUSED: invalid cap accepted")
    try:
        candidate.encode_supervised_byte_completion(tokenizer, "ok", "\ud800")
    except UnicodeEncodeError:
        pass
    else:
        raise RuntimeError("POSTSHIFT_REVIEW_REFUSED: malformed Unicode replaced")
    hist = candidate.compare_historical_literal_eos(
        tokenizer, "Task: ", '{"action":"inspect"}')
    require(hist["legacy_suffix_ids"] == [60, 101, 111, 115, 62]
            and hist["corrected_suffix_ids"] == [EOS]
            and hist["units_saved"] == 4
            and hist["legacy_had_actual_eos"] is False
            and hist["corrected_has_actual_eos"] is True
            and hist["TRAIN002b_retroactive_modification"] is False,
            "historical literal EOS identity changed")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-tokenizer", action="store_true")
    args = parser.parse_args()
    corrected = ROOT / "corrected"
    pins = source_integrity(corrected)
    candidate = load_candidate(corrected)
    if args.real_tokenizer:
        require(os.getenv("HF_HUB_OFFLINE") ==
                os.getenv("TRANSFORMERS_OFFLINE") == "1",
                "real tokenizer must execute offline after pinned source audit")
        import transformers
        from transformers import AutoTokenizer
        require(transformers.__version__ == "4.57.6",
                "reviewed tokenizer runtime version drift")
        tokenizer = AutoTokenizer.from_pretrained(
            pins["repo_id"], revision=pins["revision"], trust_remote_code=True,
            local_files_only=True)
        require(tokenizer.__class__.__name__ == "ByteTokenizer"
                and tokenizer.vocab_size == 259
                and tokenizer.pad_token_id == 256
                and tokenizer.bos_token_id == 257
                and tokenizer.eos_token_id == EOS,
                "real tokenizer mismatch")
        runtime = "pinned-real-ByteTokenizer-transformers-4.57.6"
    else:
        tokenizer = ByteTokenizer()
        runtime = "synthetic-byte-codec-source-only"
    cases = verify(candidate, tokenizer)
    result = {
        "schema": "byte-eos-001-postshift-review-v1",
        "issue": 92,
        "review_pr": 126,
        "historical_original_126_head": ORIGINAL_REVIEW_HEAD,
        "historical_review_had_empty_prompt_false_acceptance": True,
        "reviewed_corrected_source_commit": CORRECTED_COMMIT,
        "reviewed_corrected_serializer_git_blob": CORRECTED_BLOB,
        "void_tokenizer_source_sha256": TOKENIZER_SHA256,
        "runtime": runtime,
        "status": ("PASS_CORRECTED_POSTSHIFT_REAL_TOKENIZER_SOURCE_ONLY"
                   if args.real_tokenizer else "PASS_CORRECTED_POSTSHIFT_SYNTHETIC_SOURCE_ONLY"),
        "cases": cases,
        "empty_prompt_refused_count": 4,
        "historical_generation_effect": "UNMEASURED",
        "new_model_execution": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "future_training_integration_accepted": False,
        "full_72_conformance_pass": False,
        "runner": {
            "python": platform.python_version(),
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "runner_name": os.getenv("RUNNER_NAME"),
        },
    }
    out = ROOT / "runtime/ncr-001"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "real" if args.real_tokenizer else "synthetic"
    (out / ("byte-eos-postshift-" + suffix + ".json")).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
