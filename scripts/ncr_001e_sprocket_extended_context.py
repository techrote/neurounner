#!/usr/bin/env python3
"""NCR-001E: fresh, source-bound Sprocket context-limit and control-ID transport prepass.

No optimizer, held-out fixtures, invented tool/memory grammar, or training admission.
A source-preserving extension of the independently pinned #97 two-process runner.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import traceback

import torch
import transformers
from transformers import AutoTokenizer

import ncr_001e_sprocket_cross_process as source
from ncr_001e_sprocket_format import serialize_dialogue, validate_pinned_tokenizer

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (512, 1024, 2048)
TAIL = 8
CONTROL_IDS = {"<|tool_call|>": 6, "<|tool_result|>": 7,
               "<|memory_read|>": 10, "<|memory_write|>": 11}


def assert_fixed_context(lengths: tuple[int, ...], context_limit: int) -> None:
    """Fixed synthetic test lengths; never silently trim a failed context claim."""
    source.require(lengths == LENGTHS, "Predeclared synthetic context lengths changed")
    source.require(context_limit == 2048, "Pinned Sprocket max context is not 2048")
    source.require(all(0 < TAIL < n <= context_limit for n in lengths),
                   "Out-of-context or empty cache-tail witness")


def assert_control_inventory(inventory: dict) -> None:
    controls = inventory["controls"]
    for marker, identity in CONTROL_IDS.items():
        record = controls[marker]
        source.require(record["dedicated"] is True and
                       record["token_ids"] == [identity] and
                       record["vocab_lookup"] == identity,
                       "Exact source control ID mismatch: " + marker)
    source.require(inventory["tool_and_memory_semantics_qualified"] is False,
                   "Control-token recognition is NOT tool/memory grammar qualification")


def probe(variant: str) -> None:
    source.OUT = (ROOT / "runtime/ncr-001" /
                  ("sprocket-extended-context-" + variant + ".json"))
    source.REPORT = {
        "experiment": "NCR-001E",
        "variant": variant,
        "status": "RUNNING",
        "actual_model_training": False, "optimizer_steps": 0,
        "heldout_evaluation": False,
        "train_authorized": False,
        "tool_and_memory_behavior_qualified": False,
        "full_four_model_conformance_pass": False,
        "control_probe_is_valid_tool_memory_dialogue": False,
        "original_short_256_process_results_reused_as_new_model_run": False,
        "checks": {},
        "runner": {
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_job": os.environ.get("GITHUB_JOB"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    source.save_receipt(source.OUT)
    try:
        source.require(os.environ.get("HF_HUB_OFFLINE") == "1" and
                       os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                       "Actual model execution must remain strictly offline")
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        torch.manual_seed(20261027)
        spec, pin, protocol, _, wrapper, _ = source.verify_sources(variant)
        assert_fixed_context(LENGTHS,
                             protocol["models"][spec["key"]]["expected_config"]["max_position_embeddings"])
        source.require(wrapper.nca.trainable_parameters() == 658904,
                       "Original adapter geometry changed")
        opts = {"revision": source.SOURCE_REV, "local_files_only": True}
        if spec["subfolder"]:
            opts["subfolder"] = spec["subfolder"]
        tokenizer = AutoTokenizer.from_pretrained(source.SOURCE_REPO, **opts)
        inventory = validate_pinned_tokenizer(tokenizer, variant=variant)
        assert_control_inventory(inventory)
        framed = serialize_dialogue(
            [{"role": "user", "content": source.PROMPT * 10}], variant=variant)
        tokens = tokenizer(framed, return_tensors="pt",
                           add_special_tokens=False)["input_ids"]
        source.require(tokens.size(1) >= LENGTHS[-1],
                       "Synthetic ordinary-role input does not reach pinned context limit")
        tokens = tokens[:, :LENGTHS[-1]].contiguous()
        source.require(tokens[0, 0].item() == 3 and
                       int(tokens.max().item()) < 32000,
                       "Synthetic source-token stream contains invalid IDs")
        source.REPORT["tokens_sha256"] = hashlib.sha256(
            tokens.numpy().tobytes()).hexdigest()
        source.REPORT["context_lengths"] = list(LENGTHS)
        source.REPORT["cache_tail_units"] = TAIL
        source.REPORT["source_checkpoint_sha256"] = pin[
            "weight_declared_lfs_sha256_not_downloaded"]
        source.REPORT["control_tokens"] = dict(CONTROL_IDS)
        source.REPORT["source_hook_commit"] = spec["implementation_commit"]
        source.save_receipt(source.OUT)

        atol = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
        rtol = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])
        with torch.inference_mode():
            # Full-context zero residual must preserve precisely the original model.
            for n in (512, 2048):
                ids = tokens[:, :n]
                disabled = wrapper(input_ids=ids, enabled=False,
                                   use_cache=False).logits
                zero = wrapper(input_ids=ids, use_cache=False).logits
                source.compare(f"{n}_frozen_zero_adapter_exact", disabled, zero,
                               atol=0, rtol=0, exact=True)
                del zero, disabled

            # An untrained scalar readout is an intervention, NOT learning.
            with torch.no_grad():
                wrapper.nca.readout.weight[0, 0] = 0.03
            for n in LENGTHS:
                ids = tokens[:, :n]
                full = wrapper(input_ids=ids, use_cache=False).logits
                disabled = wrapper(input_ids=ids, enabled=False,
                                   use_cache=False).logits
                effect = float((full[:, -TAIL:] - disabled[:, -TAIL:]).abs().max())
                source.require(effect > 0 and torch.isfinite(full).all().item(),
                               "No effect or nonfinite actual-model logits")
                source.REPORT["checks"][f"{n}_untrained_effect"] = {
                    "max_abs": effect, "trained": False, "pass": True}
                prefill = wrapper(input_ids=ids[:, :-TAIL], use_cache=True)
                past = prefill.past_key_values
                pieces = []
                for index in range(n - TAIL, n):
                    one = wrapper(input_ids=ids[:, index:index+1],
                                  past_key_values=past, use_cache=True)
                    pieces.append(one.logits[:, -1:, :])
                    past = one.past_key_values
                    source.require(len(wrapper.last_hook_trace) == 1,
                                   "Incremental decode did not isolate single-token depth state")
                cached = torch.cat(pieces, dim=1)
                source.compare(f"{n}_full_vs_prefill_plus_{TAIL}_incremental",
                               full[:, -TAIL:], cached, atol=atol, rtol=rtol)
                del full, disabled, prefill, cached, past, pieces
                source.save_receipt(source.OUT)

            # True IDs test byte transport/caching ONLY; no fabricated argument grammar.
            raw = tokens[:, :64].clone()
            raw[0, -4:] = torch.tensor(list(CONTROL_IDS.values()))
            control_full = wrapper(input_ids=raw, use_cache=False).logits[:, -4:]
            control_prefill = wrapper(input_ids=raw[:, :-4], use_cache=True)
            past = control_prefill.past_key_values
            control_pieces = []
            for index in range(60, 64):
                one = wrapper(input_ids=raw[:, index:index+1],
                              past_key_values=past, use_cache=True)
                control_pieces.append(one.logits[:, -1:, :])
                past = one.past_key_values
            source.compare("raw_control_ids_transport_only_cache",
                           control_full, torch.cat(control_pieces, dim=1),
                           atol=atol, rtol=rtol)
            altered = tokens[:, :512].clone()
            original = wrapper(input_ids=altered, use_cache=False).logits
            altered[0, -1] = (altered[0, -1] + 7) % 32000
            changed = wrapper(input_ids=altered, use_cache=False).logits
            source.compare("512_strict_future_suffix", original[:, :-1],
                           changed[:, :-1], atol=0, rtol=0, exact=True)
            a = wrapper(input_ids=tokens[:, :32], use_cache=False).logits
            wrapper(input_ids=tokens[:, 32:64], use_cache=False)
            a_again = wrapper(input_ids=tokens[:, :32], use_cache=False).logits
            source.compare("cross_request_A_B_A_exact", a, a_again,
                           atol=0, rtol=0, exact=True)

        wrapper.verify_frozen_base()
        source.require(all(p.grad is None for p in wrapper.backbone.parameters()),
                       "Actual original checkpoint unexpectedly accumulated gradients")
        source.REPORT["status"] = "PASS_SOURCE_BOUND_EXTENDED_CONTEXT_PREPASS"
        source.REPORT["limits"] = [
            "Actual pinned Chat/Instruct frozen models, synthetic ordinary role input only",
            "Raw tool/memory IDs tested as tokenizer/model transport, not conversation or tool grammar",
            "Even the 2048-unit pass does not establish all-context long-conversation acceptance",
            "No independent #72 decision or model-specific training authorization",
            "No gradient training, optimizer, held-out evaluation or NCA-vs-MLP experiment",
        ]
    except BaseException as exc:
        source.REPORT["status"] = "FAIL_SOURCE_BOUND_EXTENDED_CONTEXT_PREPASS"
        source.REPORT["exception_type"] = type(exc).__name__
        source.REPORT["failure_traceback"] = traceback.format_exc()
        raise
    finally:
        source.save_receipt(source.OUT)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("chat", "instruct"), required=True)
    args = parser.parse_args()
    probe(args.variant)
    print(json.dumps({"variant": args.variant, "status": source.REPORT["status"],
                      "checks": source.REPORT["checks"]}, sort_keys=True))
