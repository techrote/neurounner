#!/usr/bin/env python3
"""Pure standard-library custody guards for NCR-001E void.1 fresh-process evidence.

This module does not load a model, execute remote tokenizer code, or allow training.
All checks run *before* adapter/reference deserialization in the new interpreter.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PIN = "9c2be5087772867d45961abd11f6b35110617563"
REPO = "dotlabs/void.1"
ORIGINAL_HOOK = "5fa3dc8c0de153098d1800f7b0c809819620aef4"
WEIGHT = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
BYTE_SOURCE = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
LANES = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
LENGTH = 256
PREFIX = 240
EOS_ID = 258


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(protocol: dict, pins: dict) -> dict:
    require(protocol.get("train_authorized") is False, "NCR training authorization must remain false")
    require(set(protocol.get("models", {})) == LANES, "Four frozen models must remain separate")
    lane = protocol["models"]["void1"]
    pin = pins["models"]["void1"]
    require(lane["repo_id"] == pin["repo_id"] == REPO, "Wrong void.1 repo")
    require(lane["revision"] == pin["revision"] == PIN, "Wrong void.1 revision")
    require(lane["weight_file"] == pin["hf_weight_file"] == "model.safetensors",
            "Wrong void.1 checkpoint file")
    require(pin["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT,
            "Wrong frozen Qwen3 checkpoint SHA")
    require(pin["weight_bytes"] == 360622744, "Wrong Qwen3 checkpoint size")
    require(pin["tokenization_byte_py_sha256"] == BYTE_SOURCE,
            "Wrong custom ByteTokenizer source SHA")
    require(lane["taps_after_block_zero_based"] == [7, 15, 23],
            "Pinned source depth taps changed")
    require(lane["expected_config"]["max_position_embeddings"] == 2048,
            "Pinned model context changed")
    require(protocol["conformance"]["full_vs_cached_prefill_atol"] == 0.0005
            and protocol["conformance"]["full_vs_cached_prefill_rtol"] == 0.0005,
            "Published causal/cache tolerances may not be relaxed")
    return pin


def validate_tokenizer_units(tokenizer, ids, original_text: str) -> None:
    require(tokenizer.vocab_size == 259 and
            (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
            == (256, 257, EOS_ID), "Invalid special-token ID semantics")
    require(tokenizer.encode("<eos>", add_special_tokens=True) == [60, 101, 111, 115, 62],
            "EOS text bytes wrongly equated with source EOS token ID")
    require(ids.ndim == 2 and ids.shape == (1, LENGTH), "Wrong byte-unit input shape")
    require(all(0 <= int(n) < 256 for n in ids.flatten()),
            "Source-only synthetic plain text contained a control token")
    require(len(original_text.encode("utf-8")) >= LENGTH and
            any(ord(c) > 127 for c in original_text),
            "Missing independently UTF-8-encoded multibyte source prompt")


def validate_manifest(manifest: dict, *, current_pid: int, protocol_sha: str,
                      pins_sha: str, token_ids_sha: str) -> None:
    expected = {
        "schema": 1, "model_key": "void1", "repo": REPO,
        "revision": PIN, "checkpoint_file": "model.safetensors",
        "checkpoint_sha256": WEIGHT,
        "byte_tokenizer_source_sha256": BYTE_SOURCE,
        "original_hook_commit": ORIGINAL_HOOK,
        "protocol_sha256": protocol_sha, "source_pins_sha256": pins_sha,
        "input_token_ids_sha256": token_ids_sha,
        "model_training_performed": False, "optimizer_steps": 0,
        "train_authorized": False, "heldout_evaluation": False,
        "full_four_model_conformance_pass": False,
        "historical_byte_eos_impact_measured": False,
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, "Cold-process source/permission drift: " + key)
    producer = manifest.get("producer_pid")
    require(type(producer) is int and producer > 0 and producer != current_pid,
            "Source and destination must be distinct live Python interpreters")
    for key in ("adapter_sha256", "adapter_metadata_sha256", "reference_sha256"):
        value = manifest.get(key)
        require(isinstance(value, str) and len(value) == 64 and
                all(c in "0123456789abcdef" for c in value),
                "Manifest lacks a valid sealed SHA256: " + key)
