#!/usr/bin/env python3
"""Pinned Sprocket-500M Instruct read-only NCA hook conformance spike.

Loads the exact upstream HF checkpoint, tests frozen no-op and cache/causality,
and computes a gradient without ANY optimizer step. Do not train or publish
modified weights. This pin uses root Instruct model.safetensors, not chat/model.safetensors.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import torch
import transformers
import huggingface_hub
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from ncr_001_sprocket_instruct import (
    FrozenSprocketInstructDepthNCA, MODEL_ID, MODEL_REVISION, POST_BLOCK_TAPS,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
OUT = ROOT / "runtime/ncr-001/sprocket-instruct-preflight.json"
STATE = {"experiment": "NCR-001J",
         "classification": "real_frozen_HF_hook_and_cache_conformance_prepass",
         "status": "running",
         "model_training_performed": False,
         "optimizer_steps": 0,
         "comparative_evaluation_performed": False}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def persist() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    STATE["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUT.write_text(json.dumps(STATE, indent=2, sort_keys=True) + "\n")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def within(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict:
    error = (a - b).abs()
    return {
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "close": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def need_close(title: str, stats: dict) -> None:
    STATE.setdefault("checks", {})[title] = stats
    if not stats["close"]:
        raise AssertionError(f"{title} out of tolerance: {stats}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261026)
    protocol = json.loads(PROTOCOL.read_text())
    pin = json.loads(PINS.read_text())["models"]["sprocket_instruct"]
    assert protocol["train_authorized"] is False
    assert pin["repo_id"] == MODEL_ID
    assert pin["revision"] == MODEL_REVISION
    assert protocol["models"]["sprocket_instruct"]["taps_after_block_zero_based"] == list(POST_BLOCK_TAPS)
    assert pin["hf_weight_file"] == "model.safetensors"
    assert pin["weight_present"] is True
    STATE["protocol_sha256"] = sha_file(PROTOCOL)
    STATE["source_pins_sha256"] = sha_file(PINS)
    STATE["model"] = MODEL_ID
    STATE["model_variant"] = "instruct"
    STATE["weight_path"] = "model.safetensors"
    STATE["revision"] = MODEL_REVISION
    STATE["runner"] = {
        "cpu": cpu_model(), "threads": torch.get_num_threads(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
    }
    STATE["source_files"] = {}
    persist()

    # Exact upstream source check. Only Sprocket Instruct weights are downloaded here.
    for filename, hash_key in (
        ("config.json", "config_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("model.safetensors", "weight_declared_lfs_sha256_not_downloaded"),
    ):
        path = Path(hf_hub_download(MODEL_ID, filename, revision=MODEL_REVISION))
        actual = sha_file(path)
        expected = pin[hash_key]
        rec = {"filename": filename, "sha256": actual,
               "expected_sha256": expected, "bytes": path.stat().st_size}
        STATE["source_files"][filename] = rec
        if actual != expected:
            raise AssertionError(f"Pinned source checksum mismatch: {filename}")
        print("PINNED SOURCE OK:", filename, actual, flush=True)
    persist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float32, attn_implementation="eager",
    )
    base.eval()
    adapter = FrozenSprocketInstructDepthNCA(base, verified_weight_sha256=pin["weight_declared_lfs_sha256_not_downloaded"])
    adapter.eval()
    STATE["base_parameters"] = adapter.frozen_parameters()
    STATE["nca_parameters"] = adapter.nca.trainable_parameters()
    if STATE["nca_parameters"] != 658904:
        raise AssertionError("NCA unexpected parameter count")
    assert all(not p.requires_grad for p in adapter.backbone.parameters())
    STATE["base_frozen"] = True
    persist()

    atol = float(protocol["conformance"]["frozen_base_noop_atol"])
    rtol = float(protocol["conformance"]["frozen_base_noop_rtol"])
    cache_atol = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
    cache_rtol = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])

    with torch.inference_mode():
        for text in (
            "Small gears move.",
            "An old workshop pump rattles only after it warms up.",
            "One broken bolt and two spare washers."
        ):
            batch = tokenizer(text, return_tensors="pt")
            no_adapter = adapter(**batch, enabled=False, use_cache=False).logits
            with_adapter = adapter(**batch, use_cache=False).logits
            need_close(f"zero_init_logits_length_{batch['input_ids'].shape[1]}",
                       within(no_adapter, with_adapter, atol, rtol))
            if len(adapter.last_hook_trace) != 1:
                raise AssertionError("Incomplete single-forward hook trace")

        batch = tokenizer(
            ["A hinge starts rattling in the workshop.",
             "The spring latch suddenly slips."],
            padding=True, return_tensors="pt"
        )
        base_batch = adapter(**batch, enabled=False, use_cache=False).logits
        nca_batch = adapter(**batch, use_cache=False).logits
        need_close("zero_init_logits_batch2_padded",
                   within(base_batch, nca_batch, atol, rtol))

        prompt = tokenizer("A damaged guide rail", return_tensors="pt")["input_ids"]
        generate_opts = {
            "max_new_tokens": 3,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        original = adapter.generate(prompt, enabled=False, **generate_opts)
        generated = adapter.generate(prompt, enabled=True, **generate_opts)
        if not torch.equal(original, generated):
            raise AssertionError("Zero-init generate path differs from frozen base")
        STATE["checks"]["zero_init_greedy_generation"] = {
            "equal": True,
            "nca_decoder_forward_calls": len(adapter.last_hook_trace),
            "generated_tokens": 3,
        }
        if len(adapter.last_hook_trace) != 3:
            raise AssertionError("Generation hook reset count mismatch")

    # Readout-only gradient smoke, with absolutely zero optimizer steps.
    adapter.train()
    assert not adapter.backbone.training
    adapter.zero_grad(set_to_none=True)
    encoded = tokenizer("The spring latch makes a sharp click.", return_tensors="pt")
    ids = encoded["input_ids"]
    loss = adapter(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    readout_grad = adapter.nca.readout.weight.grad
    if readout_grad is None or not bool(torch.isfinite(readout_grad).all()):
        raise AssertionError("No valid readout gradients")
    magnitude = float(readout_grad.abs().sum())
    if magnitude <= 0:
        raise AssertionError("Zero readout gradient magnitude")
    if any(p.grad is not None for p in adapter.backbone.parameters()):
        raise AssertionError("Unexpected frozen backbone gradient")
    STATE["checks"]["frozen_base_readout_gradient"] = {
        "loss": float(loss.detach()),
        "readout_grad_l1": magnitude,
        "base_grads_none": True,
        "optimizer_steps": 0,
    }
    adapter.zero_grad(set_to_none=True)
    adapter.eval()

    # Nonzero, nontrained readout tests the actual dynamical path rather than
    # the trivial zero-output adapter. This is a diagnostic weight perturbation,
    # NOT a trained model or part of the controlled NCA vs MLP study.
    with torch.no_grad():
        adapter.nca.readout.weight[0, 0] = 0.03
    with torch.inference_mode():
        input_ids = tokenizer(
            "A mechanical arm drifts left during braking.",
            return_tensors="pt",
        )["input_ids"]
        full = adapter(input_ids=input_ids, use_cache=False).logits
        unmodified = adapter(input_ids=input_ids, use_cache=False,
                             enabled=False).logits
        difference = float((full - unmodified).abs().max())
        if difference <= 0:
            raise AssertionError("NCA nonzero readout has no effect on logits")
        STATE["checks"]["nonzero_untrained_readout_changes_logits"] = {
            "max_abs": difference,
            "training": False,
        }

        prefix_logits = []
        past = None
        for i in range(input_ids.shape[1]):
            result = adapter(input_ids=input_ids[:, i:i+1],
                             past_key_values=past, use_cache=True)
            prefix_logits.append(result.logits[:, -1:])
            past = result.past_key_values
            if len(adapter.last_hook_trace) != 1:
                raise AssertionError("Incremental decoder did not reset exactly once")
        cached = torch.cat(prefix_logits, dim=1)
        need_close("nonzero_nca_full_vs_incremental_cache",
                   within(full, cached, cache_atol, cache_rtol))

        altered = input_ids.clone()
        altered[:, -1] = (altered[:, -1] + 10) % base.config.vocab_size
        before = adapter(input_ids=input_ids, use_cache=False).logits
        after = adapter(input_ids=altered, use_cache=False).logits
        need_close("nonzero_nca_future_suffix_invariance",
                   within(before[:, :-1], after[:, :-1],
                          cache_atol, cache_rtol))

    STATE["status"] = "PASS_PRELIMINARY_SOURCE_BOUND_SMOLLM2_HOOK_CONFORMANCE"
    STATE["qualification_limitations"] = [
        "One model only (Sprocket Instruct); four-model NCR conformance not complete",
        "No optimizer step on the full HF model; test was readout-gradient only",
        "Nonzero readout was manually perturbed, not learned",
        "Additional independent #72 tests for longer contexts, request boundaries and cold reload needed",
        "No MLP-vs-NCA held-out task comparison or performance conclusion",
    ]
    persist()
    print(json.dumps({
        "status": STATE["status"],
        "checks": STATE["checks"],
        "nca_parameters": STATE["nca_parameters"],
        "peak_rss_kib": STATE["peak_rss_kib"],
        "model_training_performed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        STATE["status"] = "FAILED_SOURCE_BOUND_PREPASS"
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        STATE["traceback"] = traceback.format_exc()[-5000:]
        persist()
        raise
