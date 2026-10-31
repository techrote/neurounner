#!/usr/bin/env python3
"""Real frozen Sprocket *Instruct* publisher-wire/cache probe; zero training.

This is not an assessment of whether the model *generates* a tool or memory
action or understands a tool result. All actions below are synthetic, injected
at source-authorized positions; only serialization and causal cache transport
are qualified. Original pinned checkpoint/adapter hook independently loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import torch
import transformers
from transformers import AutoTokenizer

import ncr_001e_sprocket_cross_process as source
from ncr_001e_sprocket_publisher_wire import (
    IDS, UPSTREAM_COMMIT, SFT_BLOB, HARVEST_BLOB,
    render_ids_mask, render_text,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr-001/sprocket-instruct-publisher-wire.json"
source.OUT = OUT
source.REPORT = {
    "experiment": "NCR-001E",
    "variant": "instruct",
    "status": "RUNNING",
    "source": "publisher SFT renderer; fixed GitHub commit (not a build attestation)",
    "publisher_source_commit": UPSTREAM_COMMIT,
    "publisher_sft_git_blob": SFT_BLOB,
    "publisher_harvest_git_blob": HARVEST_BLOB,
    "actual_model_training": False,
    "optimizer_steps": 0,
    "train_authorized": False,
    "heldout_evaluation": False,
    "tool_and_memory_semantics_qualified": False,
    "source_structural_tool_result_memory_wire_test": True,
    "model_generated_tool_result_memory_semantics_test": False,
    "final_four_model_conformance_pass": False,
    "checks": {},
    "runner": {
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_job": os.environ.get("GITHUB_JOB"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    },
}

TURNS = [
    {"role": "system", "content": "Only report meter readings from explicit results."},
    {"role": "memory", "content": "The user prefers degrees Celsius."},
    {"role": "user", "content": "Read meter channel A."},
    {"role": "assistant",
     "content": '<|tool_call|>{"name":"read_meter","arguments":{"channel":"A"}}'},
    {"role": "tool", "content": '{"reading":42,"unit":"C"}'},
    {"role": "assistant", "content": "The returned meter value was 42 C."},
    {"role": "user", "content": "Please remember the unit preference."},
    {"role": "assistant", "content": "<|memory_write|>Unit preference: Celsius."},
    {"role": "user", "content": "Look up that preference."},
    {"role": "assistant", "content": "<|memory_read|>temperature unit preference"},
    {"role": "memory", "content": "Unit preference: Celsius."},
    {"role": "user", "content": "Now report the reading again."},
]
BOUNDARY_IDS = {
    "tool_call": IDS["<|tool_call|>"],
    "tool_result": IDS["<|tool_result|>"],
    "memory_write": IDS["<|memory_write|>"],
}


def assert_wire_inventory(tokenizer) -> tuple[list[int], list[bool]]:
    ids, mask = render_ids_mask(TURNS, tokenizer, variant="instruct")
    source.require(len(ids) == len(mask) and len(ids) <= 512,
                   "Source wire prompt exceeds bounded synthetic length")
    source.require(ids[0] == IDS["<|system|>"] and
                   ids[-1] == IDS["<|assistant|>"] and not mask[-1],
                   "Host system/header/dangling assistant cue mismatch")
    for marker, number in BOUNDARY_IDS.items():
        source.require(ids.count(number) == 1,
                       marker + " must have exactly one actual dedicated ID")
    # A host-injected memory read and assistant-emitted memory request share
    # token 10, but carry different SFT ownership; this is the central gate.
    memory_positions = [i for i, token in enumerate(ids)
                        if token == IDS["<|memory_read|>"]]
    source.require(len(memory_positions) == 3 and
                   [mask[i] for i in memory_positions] == [False, True, False],
                   "Memory host/model role ambiguity detected")
    source.require(mask[ids.index(IDS["<|tool_call|>"])] and
                   not mask[ids.index(IDS["<|tool_result|>"])] and
                   mask[ids.index(IDS["<|memory_write|>"])],
                   "Model/host SFT control ownership mismatch")
    source.REPORT["wire"] = {
        "prompt_source": "new synthetic source-authorized role/tool/memory sequence",
        "tokenizer_units": len(ids),
        "tokens_sha256_int64": hashlib.sha256(
            torch.tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest(),
        "serialized_text_sha256": hashlib.sha256(
            render_text(TURNS, variant="instruct").encode("utf-8")).hexdigest(),
        "trained_mask_true_units": sum(mask),
        "role_boundaries": [{"index": i, "id": token, "trained": mask[i]}
                            for i, token in enumerate(ids) if token in IDS.values()],
        "not_heldout": True,
    }
    source.save_receipt(OUT)
    return ids, mask


def main() -> None:
    source.save_receipt(OUT)
    try:
        source.require(os.environ.get("HF_HUB_OFFLINE") == "1" and
                       os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                       "Pinned checkpoint execution must be offline")
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        torch.manual_seed(20261027)
        spec, pin, protocol, _, wrapper, _ = source.verify_sources("instruct")
        source.require(spec["weight"] == "model.safetensors" and
                       spec["key"] == "sprocket_instruct",
                       "Chat checkpoint would invalidate Instruct wire test")
        tok = AutoTokenizer.from_pretrained(
            source.SOURCE_REPO, revision=source.SOURCE_REV, local_files_only=True)
        ids, mask = assert_wire_inventory(tok)
        tensor = torch.tensor([ids], dtype=torch.long)
        source.REPORT["checkpoint_sha256"] = pin[
            "weight_declared_lfs_sha256_not_downloaded"]
        source.REPORT["original_hook_commit"] = spec["implementation_commit"]
        atol = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
        rtol = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])
        with torch.inference_mode():
            disabled = wrapper(input_ids=tensor, enabled=False,
                               use_cache=False).logits
            zero = wrapper(input_ids=tensor, use_cache=False).logits
            source.compare("exact_frozen_source_vs_zero_NCA", disabled, zero,
                           atol=0, rtol=0, exact=True)
            with torch.no_grad():
                wrapper.nca.readout.weight[0, 0] = 0.03
            active = wrapper(input_ids=tensor, use_cache=False).logits
            change = float((active - disabled).abs().max())
            source.require(change > 0 and torch.isfinite(active).all().item(),
                           "Frozen activated adapter is not finite/nonzero")
            source.REPORT["checks"]["untrained_manual_effect"] = {
                "max_abs": change, "trained": False, "pass": True}
            # Compare genuine sequential one-token KV cache across three
            # distinct publisher wire boundaries. Each prefill stops *before*
            # the special token; full forward supplies the reference.
            for label, marker in BOUNDARY_IDS.items():
                start = ids.index(marker)
                tail = min(5, len(ids) - start)
                source.require(start > 0 and tail >= 2,
                               "Missing cache boundary for " + label)
                past = wrapper(input_ids=tensor[:, :start],
                               use_cache=True).past_key_values
                parts = []
                for at in range(start, start + tail):
                    one = wrapper(input_ids=tensor[:, at:at+1],
                                  past_key_values=past, use_cache=True)
                    parts.append(one.logits[:, -1:, :])
                    past = one.past_key_values
                    source.require(len(wrapper.last_hook_trace) == 1,
                                   "Cache hook did not isolate one token")
                cached = torch.cat(parts, dim=1)
                source.compare("source_wire_" + label + "_prefill_incremental",
                               active[:, start:start+tail, :], cached,
                               atol=atol, rtol=rtol)
            a = wrapper(input_ids=tensor[:, :16], use_cache=False).logits
            wrapper(input_ids=tensor[:, 16:32], use_cache=False)
            b = wrapper(input_ids=tensor[:, :16], use_cache=False).logits
            source.compare("source_wire_independent_request_reset", a, b,
                           atol=0, rtol=0, exact=True)
            corrupted = tensor.clone()
            corrupted[0, -1] = (corrupted[0, -1] + 13) % 32000
            alt = wrapper(input_ids=corrupted, use_cache=False).logits
            source.compare("source_wire_strict_future_suffix",
                           active[:, :-1], alt[:, :-1], atol=0, rtol=0, exact=True)
        wrapper.verify_frozen_base()
        source.require(all(p.grad is None for p in wrapper.backbone.parameters()),
                       "Original source backbone accumulated gradients")
        source.REPORT["status"] = "PASS_SOURCE_PUBLISHER_WIRE_CACHE_ONLY"
        source.REPORT["limits"] = [
            "Publisher GitHub SFT wire source pin does not cryptographically attest its historical checkpoint build",
            "Synthetic host-supplied tool result and memory roles; not model-emitted tool execution or useful memory",
            "Only Instruct root original checkpoint; Chat excluded from tool/memory behavioural claims",
            "No final independent NCR-001E admission, training or heldout evaluation",
        ]
    except BaseException as exc:
        source.REPORT["status"] = "FAIL_SOURCE_PUBLISHER_WIRE_CACHE"
        source.REPORT["error_type"] = type(exc).__name__
        source.REPORT["traceback"] = traceback.format_exc()
        raise
    finally:
        source.save_receipt(OUT)


if __name__ == "__main__":
    main()
    print(json.dumps({"status": source.REPORT["status"],
                      "checks": source.REPORT["checks"],
                      "wire": source.REPORT.get("wire", {})}, sort_keys=True))
