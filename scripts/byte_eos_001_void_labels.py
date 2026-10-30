"""BYTE-EOS-001: correct, fail-closed future void.1 byte-token supervised labels.

NO model loading, optimizer, training workflow or modification of TRAIN-002b.
Designed for the reviewed pinned void.1 ByteTokenizer semantics only.
"""
from __future__ import annotations

from typing import Any

VOID_PAD = 256
VOID_BOS = 257
VOID_EOS = 258
VOID_VOCAB = 259
VOID_NCR_MAX_UNITS = 1024


def check_tokenizer(tokenizer: Any) -> None:
    """Reject model-family/codec substitutions and auto-EOS double-appending."""
    if tokenizer.__class__.__name__ != "ByteTokenizer":
        raise ValueError("Only verified pinned dotlabs/void.1 ByteTokenizer is supported")
    expected = (
        tokenizer.vocab_size == VOID_VOCAB
        and tokenizer.pad_token_id == VOID_PAD
        and tokenizer.bos_token_id == VOID_BOS
        and tokenizer.eos_token_id == VOID_EOS
    )
    if not expected:
        raise ValueError("Unexpected void.1 byte tokenizer vocabulary/special IDs")
    if tokenizer.encode("x", add_special_tokens=True) != [120]:
        raise ValueError("Tokenizer has unexpected automatic BOS/EOS insertion")
    if tokenizer.encode("<eos>", add_special_tokens=True) != [60,101,111,115,62]:
        raise ValueError("Tokenizer literal EOS string behaviour changed from reviewed source")


def encode_supervised_byte_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    max_units: int = VOID_NCR_MAX_UNITS,
) -> dict:
    """Build assistant-target labels with REAL terminal ID 258.

    Output uses plain Python lists deliberately; caller may create torch tensors
    in a separately reviewed model training implementation *after* NCR admission.

    Prompt and completion are UTF-8 text, NOT a list of special token strings.
    Strict UTF-8 source encoding refuses lone-surrogate replacement corruption.
    The model's tokenizer is required to be byte-exact for these texts.
    """
    check_tokenizer(tokenizer)
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise TypeError("Prompt and completion must be Python Unicode text")
    if type(max_units) is not int or not 1 <= max_units <= VOID_NCR_MAX_UNITS:
        raise ValueError("Source-bound void.1 max_units must be 1..1024")
    prompt_bytes = prompt.encode("utf-8", errors="strict")
    completion_bytes = completion.encode("utf-8", errors="strict")
    joined = prompt + completion
    joined_bytes = joined.encode("utf-8", errors="strict")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    joined_ids = tokenizer.encode(joined, add_special_tokens=True)
    if prompt_ids != list(prompt_bytes) or joined_ids != list(joined_bytes):
        raise RuntimeError(
            "Tokenizer byte-to-ID mapping changed or inserted implicit special IDs"
        )
    if joined_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Assistant target does not extend the prompt prefix")
    # Special EOS is an integer token ID, NOT a literal text suffix: TRAIN-002b
    # erroneously supervised the five byte IDs for '<eos>' at its tail.
    input_ids = joined_ids + [VOID_EOS]
    if len(input_ids) > max_units:
        raise RuntimeError(
            f"Prepared example needs {len(input_ids)} byte units (including EOS) "
            f"but source-bound cap is {max_units}; no truncation allowed"
        )
    labels = ([-100] * len(prompt_ids)
              + joined_ids[len(prompt_ids):]
              + [VOID_EOS])
    if len(labels) != len(input_ids) or labels[-1] != VOID_EOS:
        raise AssertionError("Real EOS terminal supervision must be preserved")
    if any(v >= 256 for v in joined_ids):
        raise AssertionError("Printable target unexpectedly contains control token")
    if all(v == -100 for v in labels):
        raise AssertionError("No supervised assistant target")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "sequence_tokenizer_units": len(input_ids),
        "supervised_tokenizer_units": len(completion_bytes) + 1,
        "prompt_tokenizer_units": len(prompt_bytes),
        "utf8_source_bytes": len(joined_bytes),
        "source_characters": len(joined),
        "terminal_eos_token_id": VOID_EOS,
        "terminal_eos_supervised": labels[-1] == VOID_EOS,
        "no_truncation": True,
        "serializer_version": "BYTE-EOS-001-v1",
        "historical_train002b_corrected": False,
    }


def compare_historical_literal_eos(
    tokenizer: Any, prompt: str, completion: str
) -> dict:
    """Pure identity diagnostic only — does not rewrite TRAIN-002b evidence."""
    check_tokenizer(tokenizer)
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise TypeError("Text fields required")
    joined = prompt + completion
    old_ids = tokenizer.encode(joined + tokenizer.eos_token,
                               add_special_tokens=True)
    correct = tokenizer.encode(joined, add_special_tokens=True) + [VOID_EOS]
    if old_ids[-5:] != [60,101,111,115,62]:
        raise AssertionError("Historical literal-EOS byte target not reproduced")
    if correct[-1] != VOID_EOS:
        raise AssertionError("Corrected tail not EOS")
    return {
        "legacy_suffix_ids": old_ids[-5:],
        "corrected_suffix_ids": correct[-1:],
        "legacy_total_units": len(old_ids),
        "corrected_total_units": len(correct),
        "units_saved": len(old_ids) - len(correct),
        "legacy_had_actual_eos": VOID_EOS in old_ids,
        "corrected_has_actual_eos": VOID_EOS in correct,
        "TRAIN002b_retroactive_modification": False,
    }
