#!/usr/bin/env python3
"""DETECTIVE-004A frozen-tokenizer label preflight; NEVER trains or infers.

Stage-A CI exercises this module ONLY with an artificial tokenizer. Actual
frozen-tokenizer qualification requires a separate explicit local invocation
after independent Stage-A approval. No remote downloads, model weights,
posthash finals, relevance maps or optimizer code are reachable here.

Publisher authority: kandivault-source/Sprocket-500M at
a003767e70d766bc14b40c89792543ee8641c8bc,
src/train/sft_data.py Git blob 90bddf8976b9d6410fe7c5c1f084708e1cebab73.
The pinned ChatTemplate encodes each TURN CONTENT separately, inserts role
headers and closing end tokens as single IDs, masks host turns/assistant
headers, and keeps assistant content plus its closing end tokens trainable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_001b3_sprocket as native
from scripts import detective_004a_protocol as protocol
from scripts import detective_004a_custody_audit as custody

LOCAL_TOKENIZER_FILES = (
    ("tokenizer.json", "tokenizer_sha256"),
    ("tokenizer_config.json", "tokenizer_config_sha256"),
    ("special_tokens_map.json", "special_tokens_map_sha256"),
)
MAX_STAGE_B_TOKENS = 512

# Stage A froze a bounded optimizer budget but not enough semantics to make
# a short LoRA run reproducible. Do NOT silently inherit library defaults.
# These are FIELD REQUIREMENTS only; this module does not choose values.
REQUIRED_OPTIMIZER_PREREG_FIELDS = (
    "training_seed",
    "batch_size",
    "gradient_accumulation_steps",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "weight_decay",
    "lr_scheduler",
    "warmup_steps",
)


def require(ok, why):
    if not ok:
        raise protocol.ProtocolError("token preflight: " + why)


def content_ids(tok, text):
    # Pin the publisher's tokenizers.Tokenizer.encode(..., add_special_tokens=False)
    # semantics, not an assumed whole-wire character-to-token span mapping.
    encoded = tok.encode(text, add_special_tokens=False)
    result = encoded.ids if hasattr(encoded, "ids") else encoded
    require(isinstance(result, list) and
            all(type(i) is int and i >= 0 for i in result),
            "invalid tokenizer ID sequence")
    return result


def check_special_ids(tok, expected):
    require(set(expected) == set(native.CONTROL),
            "special-token list not pinned to B3 native controls")
    for marker, wanted in expected.items():
        require(tok.token_to_id(marker) == wanted and
                content_ids(tok, marker) == [wanted],
                "pinned special-token identity mismatch: " + marker)


def source_role_ids(turns, tok, special):
    """Exact publisher ChatTemplate.render semantics, without packing.

    Returns token IDs, loss mask and the source-native prefix for predicting
    the FINAL assistant completion. The first assistant tool call, if any,
    remains included as trusted historical context in result-use examples.
    """
    require(isinstance(turns, list) and len(turns) in (2, 3, 4, 5),
            "unexpected source-native turn count")
    ids, mask, regions = [], [], []
    for turn in turns:
        require(type(turn) is dict and set(turn) == {"role", "content"},
                "turn fields changed")
        role, content = turn["role"], turn["content"]
        require(role in native.ROLE and isinstance(content, str),
                "invalid source-native role/content")
        body = content_ids(tok, content)
        header = special[native.ROLE[role]]
        trained = role == "assistant"
        begin = len(ids)
        ids += [header] + body + [special["<|end|>"]]
        mask += [False] + [trained] * len(body) + [trained]
        regions.append({"role": role, "start": begin, "header": begin,
                        "content_start": begin + 1, "end": len(ids),
                        "body_tokens": len(body)})
        require(mask[begin] is False and
                mask[-1] is trained and
                all(x is trained for x in mask[begin + 1:len(ids)]),
                "source-native trained role mask changed")
    return ids, mask, regions


def inspect_row(row, b3, tok, include_arrays=False):
    # The committed row must match the frozen source/actual user projection.
    custody.audit_row(row, b3)
    special = b3["special_token_ids"]
    ids, mask, regions = source_role_ids(row["turns"], tok, special)
    require(len(ids) <= MAX_STAGE_B_TOKENS,
            row["id"] + " exceeds 512 tokens (no truncation permitted)")
    require(bool(ids) and len(ids) == len(mask) and any(mask),
            "empty/misaligned source-native labels")
    require(regions[-1]["role"] == "assistant" and mask[-1] is True and
            ids[-1] == special["<|end|>"],
            "last assistant yield is missing from supervised target")
    for region in regions:
        if region["role"] == "assistant":
            require(ids[region["header"]] == special["<|assistant|>"] and
                    mask[region["header"]] is False,
                    "assistant header incorrectly supervised")
        if region["role"] == "tool":
            require(ids[region["header"]] == special["<|tool_result|>"] and
                    not any(mask[region["start"]:region["end"]]),
                    "host tool result was supervised")
    if row["kind"] in ("tool_selection", "result_success", "result_denied"):
        call_region = next(x for x in regions if x["role"] == "assistant")
        require(ids[call_region["content_start"]] == special["<|tool_call|>"] and
                mask[call_region["content_start"]] is True,
                "model-emitted native tool marker not trained")

    # Compare true token-ID boundaries for a future prompt; character-span
    # equality alone cannot establish equivalence after BPE tokenization.
    prior_turns = row["turns"][:-1]
    prior_ids, prior_mask, _ = source_role_ids(prior_turns, tok, special)
    prompt_ids = prior_ids + [special["<|assistant|>"]]
    require(prior_turns[-1]["role"] != "assistant" and prior_mask[-1] is False,
            "model-only prompt must end after a masked host turn")
    require(prompt_ids[-1] == special["<|assistant|>"] and
            len(prompt_ids) < len(ids) and
            prompt_ids == ids[:len(prompt_ids)],
            "native prompt differs from the trained pre-target prefix")
    # B3's actual inference runner encodes the entire publisher wire text
    # rather than each content separately; both MUST produce identical IDs.
    wire_prompt = custody.model_input_prefix(row, b3)
    require(content_ids(tok, wire_prompt) == prompt_ids,
            "whole-wire B3 inference IDs differ from publisher turnwise IDs")

    # Publisher's shifted autoregressive targets use IDs at position i+1.
    x, y = ids[:-1], [ids[i + 1] if mask[i + 1] else -1
                       for i in range(len(ids) - 1)]
    require(all(target == -1 or target == ids[i + 1]
                for i, target in enumerate(y)),
            "causal shift or ignored token mismatch")
    require(sum(label != -1 for label in y) == sum(mask[1:]),
            "trained label count changed after causal shift")
    canonical = protocol.canonical
    h = lambda value: hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()
    summary = {
        "id": row["id"], "kind": row["kind"], "num_tokens": len(ids),
        "trained_tokens": sum(mask), "shifted_train_labels": sum(y_i != -1 for y_i in y),
        "assistant_turns": sum(x["role"] == "assistant" for x in regions),
        "prompt_tokens": len(prompt_ids), "token_ids_sha256": h(ids),
        "train_mask_sha256": h(mask), "shifted_labels_sha256": h(y),
        "prompt_ids_sha256": h(prompt_ids),
        "prompt_utf8_sha256": hashlib.sha256(wire_prompt.encode("utf-8")).hexdigest(),
    }
    if include_arrays:
        # Real-tokenizer receipts must preserve the original token arrays, not
        # only irreversible digests. This never loads weights or final tasks.
        summary["raw_arrays"] = {
            "full_input_ids": ids, "trained_token_mask": mask,
            "causal_input_ids": x, "causal_target_labels": y,
            "model_only_prompt_ids": prompt_ids,
            "model_only_prompt_utf8": wire_prompt,
        }
    return summary


def inspect_archive_with_tokenizer(tok, include_arrays=False):
    """Read only committed Stage-A training; no future final tasks or models."""
    cfg, b3 = protocol.load_contract()
    manifest_sha = protocol.verify_archive(protocol.DATASET, cfg, b3)
    check_special_ids(tok, b3["special_token_ids"])
    rows = [json.loads(line) for line in
            (protocol.DATASET / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(rows) == 36 and len({r["id"] for r in rows}) == 36,
            "committed training cardinality/uniqueness changed")
    ledger = [inspect_row(row, b3, tok, include_arrays=include_arrays) for row in rows]
    return {
        "schema": "detective-004a-token-supervision-preflight-v1",
        "training_manifest_sha256": manifest_sha,
        "rows_checked": len(ledger),
        "max_tokens_seen": max(item["num_tokens"] for item in ledger),
        "max_allowed_tokens": MAX_STAGE_B_TOKENS,
        "assistant_only_mask_and_yield_structurally_checked": True,
        "real_frozen_tokenizer_identity_verified": False,
        "independent_stage_a_review_verified": False,
        "optimizer_steps": 0, "model_inferences": 0,
        "stage_b_training_admitted": False, "population_expansion_admitted": False,
        "ledger": ledger,
    }



def audit_optimizer_preregistration(cfg=None):
    """Report a SOURCE-BOUND Stage-B contract gap, never guessed defaults.

    The optional cfg argument supports synthetic adversarial tests only. A
    caller-supplied mutated cfg cannot transform a frozen protocol gap into
    a verified execution contract. No optimizer or adapter is reached here.
    """
    frozen, _ = protocol.load_contract()
    if cfg is None:
        cfg = frozen
    try:
        source_exact = type(cfg) is dict and protocol.canonical(cfg) == protocol.canonical(frozen)
    except (TypeError, ValueError):
        source_exact = False  # malformed untrusted claim cannot spoof a source pin
    future = cfg.get('future_training', {}) if type(cfg) is dict else {}
    require(type(future) is dict, 'malformed future-training claim')
    missing = [name for name in REQUIRED_OPTIMIZER_PREREG_FIELDS
               if name not in future]
    invalid = []
    def finite_numeric(value):
        return type(value) in (int, float) and math.isfinite(value)

    # The existing 36 distinct examples / 36 optimizer steps are expressly
    # SINGLE-example steps. These checks validate possible future claims,
    # not selection of missing seed/AdamW/scheduler values.
    checks = (
        ('training_seed', lambda v: type(v) is int and v >= 0),
        ('batch_size', lambda v: type(v) is int and v == 1),
        ('gradient_accumulation_steps', lambda v: type(v) is int and v == 1),
        ('adam_beta1', lambda v: finite_numeric(v) and 0 <= v < 1),
        ('adam_beta2', lambda v: finite_numeric(v) and 0 <= v < 1),
        ('adam_epsilon', lambda v: finite_numeric(v) and v > 0),
        ('weight_decay', lambda v: finite_numeric(v) and v >= 0),
        ('lr_scheduler', lambda v: type(v) is str and bool(v.strip())),
        ('warmup_steps', lambda v: type(v) is int and 0 <= v <= 36),
    )
    for name, predicate in checks:
        if name in future and not predicate(future[name]):
            invalid.append(name)
    structurally_valid = not missing and not invalid
    return {
        "schema": "detective-004a-optimizer-preregistration-gap-v1",
        "experiment": cfg.get("experiment") if type(cfg) is dict else None,
        "frozen_config_exact_match": source_exact,
        "untrusted_config_override": not source_exact,
        "frozen_optimizer": frozen["future_training"].get("optimizer"),
        "frozen_learning_rate": frozen["future_training"].get("learning_rate"),
        "frozen_optimizer_steps": frozen["future_training"].get("optimizer_steps"),
        "frozen_epochs": frozen["future_training"].get("epochs"),
        "frozen_gradient_clip": frozen["future_training"].get("gradient_clip"),
        "frozen_target_modules": frozen["future_training"].get("target_modules"),
        "frozen_rank": frozen["future_training"].get("rank"),
        "frozen_alpha": frozen["future_training"].get("alpha"),
        "frozen_dropout": frozen["future_training"].get("dropout"),
        "missing_preregistered_fields": missing,
        "invalid_claimed_fields": invalid,
        "claimed_field_set_structurally_valid": structurally_valid,
        "optimizer_execution_contract_complete": structurally_valid and source_exact,
        "values_independently_reviewed": False,
        "real_frozen_tokenizer_identity_verified": False,
        "optimizer_execution_verified": False,
        "adapter_ancestry_verified": False,
        "stage_b_training_admitted": False,
        "production_admission": False,
        "population_expansion_admitted": False,
    }

def planned_single_example_steps(tok):
    """Source-bound future 36-step plan; no optimizer and no packed examples.

    This is a proposed per-example INPUT ledger, not proof that any optimizer
    consumed the data. Actual token identity and provenance remain separate
    gates; Stage-A tests use only a synthetic tokenizer.
    """
    report = inspect_archive_with_tokenizer(tok)
    _, b3 = protocol.load_contract()
    manifest = json.loads((protocol.DATASET / "manifest.json").read_text(encoding="utf-8"))
    expected_ids = manifest["example_ids"]
    ledger = report["ledger"]
    require(len(expected_ids) == 36 and
            len(set(expected_ids)) == 36 and
            [entry["id"] for entry in ledger] == expected_ids,
            "planned step order does not match committed archive")
    steps = []
    for index, entry in enumerate(ledger, 1):
        # A single untruncated rendered row, not publisher SFT's packed
        # ctx+1-block concatenation or a batch of multiple examples.
        require(entry["num_tokens"] <= MAX_STAGE_B_TOKENS and
                entry["trained_tokens"] == entry["shifted_train_labels"] and
                entry["trained_tokens"] > 0,
                "planned step has invalid supervision/length")
        steps.append({
            "step_1based": index,
            "row_id": entry["id"],
            "num_tokens": entry["num_tokens"],
            "trained_tokens": entry["trained_tokens"],
            "token_ids_sha256": entry["token_ids_sha256"],
            "train_mask_sha256": entry["train_mask_sha256"],
            "shifted_labels_sha256": entry["shifted_labels_sha256"],
            "prompt_ids_sha256": entry["prompt_ids_sha256"],
        })
    optimizer_gap = audit_optimizer_preregistration()
    return {
        "schema": "detective-004a-single-example-step-plan-v1",
        "training_manifest_sha256": report["training_manifest_sha256"],
        "steps": steps,
        "no_packing_or_truncation": True,
        "real_frozen_tokenizer_identity_verified":
            report["real_frozen_tokenizer_identity_verified"],
        "optimizer_preregistration_complete":
            optimizer_gap["optimizer_execution_contract_complete"],
        "missing_optimizer_preregistered_fields":
            optimizer_gap["missing_preregistered_fields"],
        "optimizer_execution_verified": False,
        "training_admitted": False,
    }


def audit_claimed_training_steps(tok, claimed_steps):
    """Reject claimed 36-step duplication, reordering, packing or label drift.

    Exactly matching hashes are still only SELF-CONSISTENT CLAIMS: they do not
    attest an optimizer, learned adapter, actual batch inputs, or review gate.
    """
    plan = planned_single_example_steps(tok)
    expected = plan["steps"]
    require(type(claimed_steps) is list and len(claimed_steps) == 36,
            "exactly 36 single-example step claims required")
    require(all(type(step) is dict for step in claimed_steps),
            "malformed step claim")
    require([step.get("row_id") for step in claimed_steps] ==
            [step["row_id"] for step in expected] and
            [step.get("step_1based") for step in claimed_steps] ==
            list(range(1, 37)),
            "duplicate, missing or reordered step")
    require(all(step == exp for step, exp in zip(claimed_steps, expected)),
            "per-step token IDs, mask, label, prompt hash or length changed")
    digest = hashlib.sha256(
        protocol.canonical(claimed_steps).encode("utf-8")).hexdigest()
    return {
        "schema": "detective-004a-claimed-step-audit-v1",
        "training_manifest_sha256": plan["training_manifest_sha256"],
        "claimed_step_ledger_sha256": digest,
        "claimed_step_count": len(claimed_steps),
        "structurally_exact_single_example_steps": True,
        "optimizer_preregistration_complete":
            plan["optimizer_preregistration_complete"],
        "missing_optimizer_preregistered_fields":
            plan["missing_optimizer_preregistered_fields"],
        "real_frozen_tokenizer_identity_verified":
            plan["real_frozen_tokenizer_identity_verified"],
        "actual_optimizer_batches_independently_attested": False,
        "adapter_ancestry_verified": False,
        "stage_b_training_admitted": False,
        "production_admission": False,
        "population_expansion_admitted": False,
    }


def inspect_local_frozen_tokenizer(folder):
    """Opt-in post-review read-only validation; NO download or model import."""
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(),
            "explicit local tokenizer directory required")
    _, b3 = protocol.load_contract()
    checks = {}
    for filename, config_key in LOCAL_TOKENIZER_FILES:
        path = folder / filename
        require(path.is_file() and not path.is_symlink(),
                "missing/symlinked frozen local tokenizer file: " + filename)
        actual = protocol.file_sha(path)
        expected = b3["model"][config_key]
        require(actual == expected, "frozen tokenizer SHA drift: " + filename)
        checks[filename] = actual
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise protocol.ProtocolError(
            "token preflight: local tokenizers package required; no downloads attempted") from exc
    tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    report = inspect_archive_with_tokenizer(tok, include_arrays=True)
    report["real_frozen_tokenizer_identity_verified"] = True
    report["local_tokenizer_file_sha256"] = checks
    # This token-level readback is one PRE-OPTIMIZER gate, NOT Stage-A review,
    # optimizer authority, adapter ancestry or posthash final custody.
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-dir", required=True,
                    help="already-local exact pinned root-Instruct tokenizer files; never fetched")
    ap.add_argument("--output", help="optional path for read-only qualification receipt")
    args = ap.parse_args()
    report = inspect_local_frozen_tokenizer(args.tokenizer_dir)
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        out = Path(args.output)
        require(not out.exists(), "refuse overwrite of existing evidence receipt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
