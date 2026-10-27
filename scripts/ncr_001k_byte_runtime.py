#!/usr/bin/env python3
"""NCR-001K runtime qualification of *reviewed exact* void.1 ByteTokenizer.

This script runs only AFTER a separate source-only inventory. It requires
HF_HUB_OFFLINE=1 and local_files_only=True; never loads an LM or trains.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import traceback

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import transformers

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
OUTPUT = ROOT / "runtime/ncr-001/void-byte-runtime.json"
REPO = "dotlabs/void.1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
CLASS_METHODS = {
    "__init__", "vocab_size", "get_vocab", "_tokenize",
    "_convert_token_to_id", "_convert_id_to_token", "convert_tokens_to_string",
    "build_inputs_with_special_tokens",
    "create_token_type_ids_from_sequences", "save_vocabulary"
}
RESULT = {
    "experiment": "NCR-001K",
    "stage": "reviewed_pinned_byte_tokenizer_runtime_semantics",
    "status": "RUNNING",
    "model_training_performed": False,
    "model_weights_loaded": False,
    "optimizer_steps": 0,
    "training_authorized": False,
    "tokenizer_runtime_executed": False,
    "byte_semantics_qualified": False,
    "remaining_full_void1_conformance": True,
    "tests": {},
}


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def persist() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(RESULT, indent=2, sort_keys=True,
                                ensure_ascii=False) + "\n", encoding="utf-8")


def check(label: str, truth: bool, values: dict | None = None) -> None:
    RESULT["tests"][label] = {"passed": bool(truth), **(values or {})}
    persist()
    if not truth:
        raise AssertionError(f"void.1 tokenizer failure: {label} {values}")


def verify_reviewed_shape(tree: ast.Module) -> dict:
    expected_imports = {("import", "json"), ("import", "os"),
                        ("from", "transformers:PreTrainedTokenizer")}
    imports = set()
    top = []
    for n in tree.body:
        top.append(type(n).__name__)
        if isinstance(n, ast.Import):
            imports.update(("import", alias.name) for alias in n.names)
        elif isinstance(n, ast.ImportFrom):
            imports.add(("from", f"{n.module}:{','.join(a.name for a in n.names)}"))
        elif isinstance(n, (ast.Assign, ast.ClassDef)):
            pass
        else:
            raise RuntimeError(f"Unreviewed top-level tokenizer statement {type(n).__name__}")
    if imports != expected_imports:
        raise RuntimeError(f"Unreviewed tokenizer imports {sorted(imports)}")
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "ByteTokenizer":
        raise RuntimeError("Unexpected remote tokenizer class shape")
    cls = classes[0]
    if len(cls.bases) != 1 or not isinstance(cls.bases[0], ast.Name) or cls.bases[0].id != "PreTrainedTokenizer":
        raise RuntimeError("Unreviewed remote tokenizer inheritance")
    methods = {x.name for x in cls.body if isinstance(x, ast.FunctionDef)}
    if methods != CLASS_METHODS:
        raise RuntimeError(f"Unreviewed tokenizer methods {methods ^ CLASS_METHODS}")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)) and n not in tree.body:
            raise RuntimeError("Nested/dynamic import in pinned tokenizer")
    return {
        "imports": sorted(list(imports)),
        "class": cls.name,
        "methods": sorted(list(methods)),
        "top_level_statement_types": top,
    }


def main() -> int:
    # Transformers/HF should have been imported with these process-level
    # offline settings set in the workflow before Python launched.
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Runtime tokenizer review must run under network-offline environment")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert protocol["train_authorized"] is False
    model = protocol["models"]["void1"]
    pinned = pins["models"]["void1"]
    assert model["repo_id"] == REPO == pinned["repo_id"]
    assert model["revision"] == REV == pinned["revision"]
    assert model["tokenizer_reference"]["train002b_source_sha256"] == pinned["tokenization_byte_py_sha256"]
    RESULT["source"] = {
        "repo_id": REPO, "revision": REV,
        "protocol_sha256": h(PROTOCOL.read_bytes()),
        "source_pins_sha256": h(PINS.read_bytes()),
        "tokenizer_source_sha256": pinned["tokenization_byte_py_sha256"],
        "torch_loaded": "torch" in sys.modules,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "network_offline": True,
    }

    files = (
        ("tokenization_byte.py", "tokenization_byte_py_sha256"),
        ("byte_vocab.json", "byte_vocab_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("special_tokens_map.json", "special_tokens_map_sha256"),
        ("config.json", "config_sha256"),
    )
    RESULT["source_files"] = {}
    for filename, pin_field in files:
        path = Path(hf_hub_download(
            REPO, filename, revision=REV, local_files_only=True
        ))
        raw = path.read_bytes()
        actual = h(raw)
        expected = pinned[pin_field]
        check(f"source_sha_{filename}", actual == expected,
              {"sha256": actual, "bytes": len(raw)})
        RESULT["source_files"][filename] = {"sha256": actual, "bytes": len(raw)}
        if filename == "tokenization_byte.py":
            RESULT["reviewed_shape"] = verify_reviewed_shape(ast.parse(
                raw.decode("utf-8"), filename=filename
            ))
    vocab = json.loads(Path(hf_hub_download(
        REPO, "byte_vocab.json", revision=REV, local_files_only=True
    )).read_text(encoding="utf-8"))
    check("vocabulary_complete_259_entries",
          len(vocab) == 259 and
          all(vocab.get(f"<0x{i:02X}>") == i for i in range(256))
          and vocab.get("<pad>") == 256
          and vocab.get("<bos>") == 257
          and vocab.get("<eos>") == 258)

    # The remote Python file is executed only now: pinned-byte SHA + the
    # manually-reviewed imports/classes/methods have already passed.
    tokenizer = AutoTokenizer.from_pretrained(
        REPO, revision=REV, trust_remote_code=True,
        use_fast=False, local_files_only=True,
    )
    src_path = inspect.getsourcefile(tokenizer.__class__)
    if not src_path:
        raise RuntimeError("Cannot identify the exact dynamically loaded tokenizer source")
    actual_src = h(Path(src_path).read_bytes())
    check("actual_executed_python_matches_pin",
          actual_src == pinned["tokenization_byte_py_sha256"],
          {"executed_sha256": actual_src, "module_file": Path(src_path).name})
    RESULT["tokenizer_runtime_executed"] = True
    RESULT["tokenizer_class"] = tokenizer.__class__.__name__
    check("class_and_token_ids", tokenizer.__class__.__name__ == "ByteTokenizer"
          and tokenizer.vocab_size == 259
          and tokenizer.pad_token_id == 256
          and tokenizer.bos_token_id == 257
          and tokenizer.eos_token_id == 258
          and tokenizer.padding_side == "left",
          {"pad": tokenizer.pad_token_id, "bos": tokenizer.bos_token_id,
           "eos": tokenizer.eos_token_id, "padding_side": tokenizer.padding_side})

    samples = {
        "ascii": "bolt-17",
        "punctuation": "x=4;y=-2",
        "accent_nfc": "café",
        "accent_combining": "cafe\u0301",
        "greek": "πΩ",
        "emoji": "🔧",
        "japanese": "日本語",
        "newlines": "first\nsecond\tthird",
        "apostrophe": "don't",
        "special_literal_in_text": "some literal <eos> text",
    }
    RESULTS = {}
    for name, sample in samples.items():
        expected = list(sample.encode("utf-8"))
        ids = tokenizer.encode(sample, add_special_tokens=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        check(f"exact_utf8_encode_{name}", ids == expected,
              {"bytes": len(expected), "tokenizer_units": len(ids),
               "characters": len(sample)})
        check(f"exact_utf8_decode_{name}", decoded == sample,
              {"tokenizer_units": len(ids)})
        RESULTS[name] = {
            "input_characters": len(sample),
            "utf8_bytes": len(expected),
            "tokenizer_units": len(ids),
            "roundtrip": decoded == sample,
        }
    RESULT["valid_unicode_roundtrips"] = RESULTS

    # A concrete regression for the old TRAIN-002b literal-EOS convention.
    eos_literal = tokenizer.encode(tokenizer.eos_token, add_special_tokens=True)
    with_special_text = tokenizer.encode("done" + tokenizer.eos_token,
                                         add_special_tokens=True)
    explicit = tokenizer.encode("done", add_special_tokens=True) + [258]
    RESULT["eos_experiment"] = {
        "literal_string_encode_ids": eos_literal,
        "literal_string_is_actual_eos_id": eos_literal == [258],
        "concatenated_text_last_id": with_special_text[-1],
        "concatenated_text_has_eos_id": 258 in with_special_text,
        "explicit_appended_id_last": explicit[-1],
        "explicit_appended_id_contains_eos": 258 in explicit,
        "first_train002b_semantic_warning":
            "TRAIN-002b concatenated literal tokenizer.eos_token to plain text; "
            "this output records if it produced actual ID 258 or only regular bytes."
    }
    check("literal_eos_text_is_bytes_not_stop_token",
          eos_literal == list(b"<eos>") and
          with_special_text == list(b"done<eos>") and
          258 not in with_special_text,
          {"literal_ids": eos_literal,
           "legacy_train002b_concat_contains_real_eos": 258 in with_special_text})
    check("actual_eos_must_be_appended_as_integer_id",
          explicit[-1] == 258 and explicit[:-1] == list(b"done"),
          {"explicit_ids": explicit, "eos_id": 258})
    check("special_id_conversion",
          tokenizer.convert_tokens_to_ids(tokenizer.eos_token) == 258
          and tokenizer.convert_ids_to_tokens(258) == tokenizer.eos_token)
    check("explicit_appended_eos_decode_skip",
          tokenizer.decode(explicit, skip_special_tokens=True,
                           clean_up_tokenization_spaces=False) == "done")
    check("no_automatic_bos_eos",
          tokenizer.encode("hello", add_special_tokens=True) == list(b"hello"))
    check("invalid_utf8_replacement",
          tokenizer.decode([255], skip_special_tokens=True,
                           clean_up_tokenization_spaces=False) == "\ufffd")
    check("unicode_normalization_is_not_applied",
          tokenizer.encode("café") != tokenizer.encode("cafe\u0301"))
    # Multi-byte sequences must be assembled before decoding. Streaming
    # individual byte tokens separately can produce replacement characters.
    pile = list("🔧".encode("utf-8"))
    RESULT["streaming_caveat"] = {
        "valid_merged_decode": tokenizer.decode(pile, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False),
        "incorrect_independent_decode": "".join(tokenizer.decode([v], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False) for v in pile),
    }

    batch = tokenizer(["a", "longer"], padding=True, add_special_tokens=True)
    left_ids = batch["input_ids"]
    left_mask = batch["attention_mask"]
    check("left_padding_uses_pad_256",
          len(left_ids[0]) == len(left_ids[1]) and
          left_ids[0][0] == 256 and
          left_mask[0][0] == 0 and left_mask[0][-1] == 1,
          {"first_ids": left_ids[0], "first_mask": left_mask[0]})
    tokenizer.padding_side = "right"
    right = tokenizer(["a", "longer"], padding=True, add_special_tokens=True)
    check("right_padding_uses_pad_256",
          right["input_ids"][0][-1] == 256 and
          right["attention_mask"][0][-1] == 0 and
          right["attention_mask"][0][0] == 1)

    RESULT["status"] = "PASS_REVIEWED_VOID_BYTE_TOKENIZER_RUNTIME_SEMANTICS__MODEL_CONFORMANCE_BLOCKED"
    RESULT["byte_semantics_qualified"] = True
    RESULT["remaining_full_void1_conformance"] = True
    RESULT["model_training_performed"] = False
    RESULT["optimizer_steps"] = 0
    RESULT["limitations"] = [
        "The exact reviewed Python source was executed only on an isolated CI runner offline",
        "No void.1 model weights, real inference, training or held-out scoring",
        "No BOS/EOS auto insertion; downstream training/generation must use actual ID 258 explicitly",
        "Only valid UTF-8 text roundtrips; malformed byte sequences decode with replacement",
        "Tokenizer must not be ranked in BPE tokens/sec; use UTF-8 bytes, chars, units and EOS/budget",
        "Source semantics pass is not cold-cache/long-context model-side #72 conformance",
    ]
    persist()
    print(json.dumps({
        "status": RESULT["status"],
        "eos_experiment": RESULT["eos_experiment"],
        "special_tokens": {"pad":256,"bos":257,"eos":258},
        "unicode_results": RESULT["valid_unicode_roundtrips"],
        "streaming_caveat": RESULT["streaming_caveat"],
        "training_performed": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        RESULT["status"] = "FAILED_REVIEWED_BYTE_TOKENIZER_RUNTIME_SEMANTICS"
        RESULT["error"] = f"{type(exc).__name__}: {exc}"
        RESULT["traceback"] = traceback.format_exc()[-5500:]
        persist()
        raise
