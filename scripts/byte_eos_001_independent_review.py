#!/usr/bin/env python3
"""Independent BYTE-EOS-001 future serializer acceptance review.

No model weights, optimizer, backward pass, heldout evaluation or modification
of historical TRAIN-002b. The candidate serializer is treated as code under
review and compared against an independent UTF-8 byte oracle plus the exact
reviewed dotlabs/void.1 ByteTokenizer runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr-001/byte-eos-001-independent-review.json"
REVIEWED_PR94_HEAD = "b1c558f7d46f5d7c76b4e6336a37143886d16c08"
CANDIDATE_BLOB = "724a5040d2fc9eb06ef0b17143d44daa36a4aa89"
VOID_REPO = "dotlabs/void.1"
VOID_REV = "9c2be5087772867d45961abd11f6b35110617563"
TOKENIZER_SOURCE_SHA = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
PAD, BOS, EOS, VOCAB = 256, 257, 258, 259
MAX_UNITS = 1024
PREREG_COMMIT = "a1b41c358d215974ea9ef2748baa13c3872b0fbb"


def require(ok, reason):
    if not ok:
        raise RuntimeError("BYTE-EOS-001 independent review REFUSED: " + reason)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path: str) -> str:
    return subprocess.check_output(["git", "hash-object", path], text=True).strip()


def oracle(prompt: str, completion: str, max_units: int = MAX_UNITS) -> dict:
    """Independent byte-level source oracle; never invokes candidate code."""
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise TypeError("Unicode strings required")
    if type(max_units) is not int or not 1 <= max_units <= MAX_UNITS:
        raise ValueError("max_units outside fixed 1..1024")
    prompt_bytes = prompt.encode("utf-8", errors="strict")
    completion_bytes = completion.encode("utf-8", errors="strict")
    joined = prompt_bytes + completion_bytes
    ids = list(joined) + [EOS]
    if len(ids) > max_units:
        raise RuntimeError("oracle refuses truncation of terminal EOS")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt_bytes) + list(completion_bytes) + [EOS],
        "sequence_tokenizer_units": len(ids),
        "supervised_tokenizer_units": len(completion_bytes) + 1,
        "prompt_tokenizer_units": len(prompt_bytes),
        "utf8_source_bytes": len(joined),
        "source_characters": len(prompt + completion),
        "terminal_eos_token_id": EOS,
        "terminal_eos_supervised": True,
        "no_truncation": True,
        "serializer_version": "BYTE-EOS-001-v1",
        "historical_train002b_corrected": False,
    }


def source_cases():
    """Fixed new reviewer cases; none are sealed evaluation fixtures."""
    return [
        ("ascii", "Task: ", "inspect bearing"),
        ("unicode", "πΩ→🔧 ", "café 日本語"),
        ("literal_eos_content", "User wrote: ", "<eos>"),
        ("empty_completion", "Prompt", ""),
        ("empty_prompt", "", "assistant bytes"),
        ("emoji_boundary", "🙂" * 7, "🔧" * 5),
        ("max_1024", "", "x" * 1023),
    ]


def compare_candidate(candidate, tokenizer, name, prompt, completion,
                      *, max_units=MAX_UNITS) -> dict:
    expected = oracle(prompt, completion, max_units)
    actual = candidate(tokenizer, prompt, completion, max_units=max_units)
    require(actual == expected, name + " candidate differs from independent byte oracle")
    require(actual["input_ids"][-1] == actual["labels"][-1] == EOS,
            name + " terminal EOS 258 is not supervised")
    require(EOS not in actual["input_ids"][:-1],
            name + " printable source unexpectedly contains EOS control ID")
    require(all((v == -100 or 0 <= v < 256) for v in actual["labels"][:-1]),
            name + " assistant byte labels contain hidden control IDs")
    return {
        "name": name,
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "completion_utf8_bytes": len(completion.encode("utf-8")),
        "sequence_units": len(actual["input_ids"]),
        "supervised_units": actual["supervised_tokenizer_units"],
        "terminal_id": actual["input_ids"][-1],
        "exact_oracle_match": True,
    }


def guard_report(report):
    require(report["training_authorized"] is False
            and report["optimizer_steps"] == 0
            and report["model_weights_loaded"] is False
            and report["historical_train002b_modified"] is False
            and report["historical_impact_measured"] is False
            and report["future_serializer_integrated_into_training"] is False,
            "false training/history/impact promotion")


def main():
    report = {
        "schema_version": 1,
        "experiment": "BYTE-EOS-001",
        "stage": "independent_future_void_serializer_acceptance",
        "status": "RUNNING",
        "reviewed_pr94_head": REVIEWED_PR94_HEAD,
        "candidate_blob": CANDIDATE_BLOB,
        "training_authorized": False,
        "optimizer_steps": 0,
        "model_weights_loaded": False,
        "historical_train002b_modified": False,
        "historical_impact_measured": False,
        "future_serializer_integrated_into_training": False,
        "cases": [],
        "runner": {
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "python": platform.python_version(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def save():
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")

    save()
    try:
        require(git_blob("scripts/byte_eos_001_void_labels.py") == CANDIDATE_BLOB,
                "reviewed candidate serializer code changed")
        protocol_path = ROOT / "experiments/ncr-001/protocol.json"
        pins_path = ROOT / "experiments/ncr-001/source-pins.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        pins = json.loads(pins_path.read_text(encoding="utf-8"))
        require(protocol["train_authorized"] is False,
                "NCR training unexpectedly authorized")
        require(set(protocol["models"]) ==
                {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
                "four-model protocol changed")
        p = pins["models"]["void1"]
        m = protocol["models"]["void1"]
        require(p["repo_id"] == m["repo_id"] == VOID_REPO
                and p["revision"] == m["revision"] == VOID_REV
                and p["tokenization_byte_py_sha256"] ==
                p["train002b_tokenizer_source_sha256"] ==
                m["tokenizer_reference"]["train002b_source_sha256"] ==
                TOKENIZER_SOURCE_SHA,
                "exact void ByteTokenizer source identity changed")
        require(m["tokenizer_reference"]["ids"] ==
                {"pad": PAD, "bos": BOS, "eos": EOS},
                "void source special token IDs changed")

        prereg_root = ROOT / "prereg"
        require(prereg_root.is_dir()
                and subprocess.check_output(
                    ["git", "-C", str(prereg_root), "rev-parse", "HEAD"],
                    text=True).strip() == PREREG_COMMIT,
                "exact four-model preregistration checkout missing or changed")
        authoritative_path = prereg_root / "experiments/ncr-001/protocol.json"
        authoritative = json.loads(authoritative_path.read_text(encoding="utf-8"))
        require(authoritative["train_authorized"] is False
                and authoritative["training"]["per_model_sequence_unit_limits"]["void1"]
                == MAX_UNITS
                and authoritative["models"]["void1"]["revision"] == VOID_REV
                and authoritative["models"]["void1"]["tokenizer_reference"][
                    "train002b_source_sha256"] == TOKENIZER_SOURCE_SHA,
                "exact preregistered void unit/source policy changed")
        report["source"] = {
            "repo_id": VOID_REPO,
            "revision": VOID_REV,
            "tokenizer_source_sha256": TOKENIZER_SOURCE_SHA,
            "protocol_sha256": sha256(protocol_path),
            "source_pins_sha256": sha256(pins_path),
            "preregistration_commit": PREREG_COMMIT,
            "preregistration_protocol_sha256": sha256(authoritative_path),
            "ids": {"pad": PAD, "bos": BOS, "eos": EOS, "vocab": VOCAB},
            "max_units": MAX_UNITS,
        }
        save()

        # Workflow has already cached and source-audited this exact remote Python.
        require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1",
                "real tokenizer runtime review must execute offline")
        import transformers
        from transformers import AutoTokenizer
        report["runner"]["transformers"] = transformers.__version__
        require(transformers.__version__ == "4.57.6",
                "reviewed tokenizer runtime library changed")
        tok = AutoTokenizer.from_pretrained(
            VOID_REPO, revision=VOID_REV, trust_remote_code=True,
            local_files_only=True)
        require(tok.__class__.__name__ == "ByteTokenizer"
                and tok.vocab_size == VOCAB
                and tok.pad_token_id == PAD and tok.bos_token_id == BOS
                and tok.eos_token_id == EOS,
                "actual reviewed ByteTokenizer special IDs/class changed")
        require(tok.encode("<eos>", add_special_tokens=True) ==
                [60, 101, 111, 115, 62]
                and tok.encode("x", add_special_tokens=True) == [120],
                "actual pinned literal EOS/automatic-special semantics changed")

        sys.path.insert(0, str(ROOT / "scripts"))
        import byte_eos_001_void_labels as reviewed
        require(reviewed.VOID_EOS == EOS and reviewed.VOID_NCR_MAX_UNITS == MAX_UNITS,
                "candidate constants changed")

        for name, prompt, completion in source_cases():
            row = compare_candidate(
                reviewed.encode_supervised_byte_completion,
                tok, name, prompt, completion)
            report["cases"].append(row)
            save()

        # Max boundary must preserve EOS, one extra byte must be refused.
        exact = reviewed.encode_supervised_byte_completion(
            tok, "", "x" * 1023, max_units=1024)
        require(len(exact["input_ids"]) == 1024 and exact["input_ids"][-1] == EOS,
                "1024 boundary does not preserve real EOS")
        try:
            reviewed.encode_supervised_byte_completion(
                tok, "", "x" * 1024, max_units=1024)
        except RuntimeError:
            report["overflow_1025_refused"] = True
        else:
            raise RuntimeError("candidate silently truncates 1025-unit source")

        # Strict Unicode must refuse lone surrogate before token replacement.
        try:
            reviewed.encode_supervised_byte_completion(tok, "ok", "\ud800")
        except UnicodeEncodeError:
            report["malformed_unicode_refused"] = True
        else:
            raise RuntimeError("candidate silently replaced malformed Unicode")

        hist = reviewed.compare_historical_literal_eos(
            tok, "Task: ", '{"action":"inspect"}')
        require(hist["legacy_suffix_ids"] == [60, 101, 111, 115, 62]
                and hist["corrected_suffix_ids"] == [EOS]
                and hist["units_saved"] == 4
                and hist["legacy_had_actual_eos"] is False
                and hist["corrected_has_actual_eos"] is True
                and hist["TRAIN002b_retroactive_modification"] is False,
                "historical mismatch comparison was rewritten")
        report["historical_identity"] = hist

        # Bind existing original qualification without treating it as this review.
        provenance = json.loads(
            (ROOT / "experiments/ncr-001/byte-eos-001/provenance.json")
            .read_text(encoding="utf-8"))
        receipt_path = ROOT / "experiments/ncr-001/byte-eos-001/receipt.json"
        require(sha256(receipt_path) ==
                provenance["original_raw_report_sha256"] ==
                "6c611bdc0f021f6d9e2fa39d91b7e141d937844f2b849f9ae437151254572922",
                "original BYTE-EOS qualification receipt changed")
        require(provenance["workflow_run_id"] == 40466048138
                and provenance["actual_tokenizer_runtime_test_pass"] is True
                and provenance["exact_runtime_source_sha256"] == TOKENIZER_SOURCE_SHA
                and provenance["boundaries"]["TRAIN002b_original_source_modified"] is False
                and provenance["boundaries"]["TRAIN002b_old_evidence_modified"] is False
                and provenance["boundaries"]["model_weights_loaded"] is False
                and provenance["boundaries"]["optimizer_steps"] == 0
                and provenance["boundaries"]["NCR_training_authorized"] is False,
                "original qualification boundaries changed")
        report["original_qualification"] = {
            "run_id": 40466048138,
            "receipt_sha256": provenance["original_raw_report_sha256"],
            "artifact_id": provenance["original_artifact_id"],
            "artifact_zip_sha256": provenance["original_artifact_zip_sha256"],
        }

        guard_report(report)
        report["status"] = (
            "PASS_INDEPENDENT_FUTURE_EOS_SERIALIZER__"
            "VOID_TRAINING_INTEGRATION_STILL_BLOCKED"
        )
        report["decision"] = {
            "future_serializer_source_accepted": True,
            "integration_into_void_training_accepted": False,
            "historical_train002b_reinterpreted": False,
            "historical_generation_effect_size": "UNMEASURED",
            "requires_void72_and_explicit_training_authorization": True,
        }
    except BaseException:
        report["status"] = "FAIL_INDEPENDENT_FUTURE_EOS_SERIALIZER"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save()

    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
