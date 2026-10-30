#!/usr/bin/env python3
"""NCR-001K: source-bound void.1 Qwen3 NCA prepass with NO custom code execution.

Read and hash ByteTokenizer source without importing it; token inputs are
deliberately raw valid vocabulary IDs for architecture/causality testing only.
There is no semantic/text evaluation and no optimizer update.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import traceback
from pathlib import Path

import torch
import transformers
import huggingface_hub
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from ncr_001_void1 import (
    FrozenVoid1DepthNCA, MODEL_ID, MODEL_REVISION,
    POST_BLOCK_TAPS, PINNED_SAFETENSORS_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr-001/void1-preflight.json"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
EVIDENCE = {
    "experiment": "NCR-001K",
    "status": "running",
    "classification": "byte_tokenizer_source_checked_qwen3_architecture_prepass",
    "model_training_performed": False,
    "optimizer_steps": 0,
    "comparative_evaluation_performed": False,
    "custom_tokenizer_code_executed": False,
    "tokenizer_semantic_qualification": False,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def persist() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUT.write_text(json.dumps(EVIDENCE, indent=2, sort_keys=True) + "\n")


def measured(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict:
    diffs = (a - b).abs()
    return {
        "max_abs": float(diffs.max()),
        "mean_abs": float(diffs.mean()),
        "atol": atol,
        "rtol": rtol,
        "close": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def assert_close(key: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> None:
    result = measured(a, b, atol, rtol)
    EVIDENCE.setdefault("checks", {})[key] = result
    if not result["close"]:
        raise AssertionError(f"{key}: failed cached/logit agreement: {result}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261026)
    protocol = json.loads(PROTOCOL.read_text())
    pin = json.loads(PINS.read_text())["models"]["void1"]
    settings = protocol["models"]["void1"]
    assert protocol["train_authorized"] is False
    assert pin["repo_id"] == MODEL_ID
    assert pin["revision"] == MODEL_REVISION
    assert pin["hf_weight_file"] == "model.safetensors"
    assert pin["weight_present"] is True
    assert pin["weight_declared_lfs_sha256_not_downloaded"] == PINNED_SAFETENSORS_SHA256
    assert settings["expected_config"]["model_type"] == "qwen3"
    assert settings["taps_after_block_zero_based"] == list(POST_BLOCK_TAPS)
    EVIDENCE.update({
        "source_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "base_variant": "byte_model",
        "protocol_sha256": digest(PROTOCOL),
        "source_pins_sha256": digest(PINS),
        "runner": {
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        },
        "source_files": {},
    })
    persist()

    # No trust_remote_code, no importlib of downloaded tokenizer source.
    checks = {
        "config.json": "config_sha256",
        "model.safetensors": "weight_declared_lfs_sha256_not_downloaded",
        "tokenization_byte.py": "tokenization_byte_py_sha256",
        "byte_vocab.json": "byte_vocab_sha256",
        "tokenizer_config.json": "tokenizer_config_sha256",
        "special_tokens_map.json": "special_tokens_map_sha256",
    }
    for filename, key in checks.items():
        source = Path(hf_hub_download(MODEL_ID, filename, revision=MODEL_REVISION))
        actual = digest(source)
        expected = pin[key]
        if actual != expected:
            raise AssertionError(f"Pinned void.1 source mismatch: {filename}")
        EVIDENCE["source_files"][filename] = {
            "bytes": source.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
        }
        print("PINNED VOID SOURCE OK", filename, actual, flush=True)
    assert EVIDENCE["source_files"]["tokenization_byte.py"]["sha256"] == (
        settings["tokenizer_reference"]["train002b_source_sha256"]
    )
    EVIDENCE["byte_tokenizer_source_matches_prior_train002b"] = True
    EVIDENCE["custom_tokenizer_code_executed"] = False
    persist()

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False,
    )
    assert base.config.vocab_size == 259
    base.eval()
    adapter = FrozenVoid1DepthNCA(
        base, verified_weight_sha256=PINNED_SAFETENSORS_SHA256,
    ).eval()
    EVIDENCE["base_frozen_parameters"] = adapter.frozen_parameters()
    EVIDENCE["nca_trainable_parameters"] = adapter.nca.trainable_parameters()
    assert EVIDENCE["nca_trainable_parameters"] == 329944
    assert all(not p.requires_grad for p in adapter.backbone.parameters())
    EVIDENCE["checks"] = {}

    atol = float(protocol["conformance"]["frozen_base_noop_atol"])
    rtol = float(protocol["conformance"]["frozen_base_noop_rtol"])
    cache_atol = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
    cache_rtol = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])
    # The vocabulary uses 0..255 for byte values and dedicated special IDs.
    # These are chosen as generic input IDs, not interpreted as decoded text.
    samples = [
        torch.tensor([[257, 65, 66, 67]], dtype=torch.long),
        torch.tensor([[257, 72, 101, 108, 108, 111]], dtype=torch.long),
        torch.tensor([[257, 105, 110, 100, 117, 99, 116, 33]], dtype=torch.long),
    ]
    with torch.inference_mode():
        for sample in samples:
            original = adapter(input_ids=sample, use_cache=False, enabled=False).logits
            hooked = adapter(input_ids=sample, use_cache=False).logits
            assert_close(f"zero_init_len_{sample.shape[1]}",
                         original, hooked, atol, rtol)
            assert len(adapter.last_hook_trace) == 1
        batch = torch.tensor([[257, 65, 66, 67], [257, 67, 66, 65]],
                             dtype=torch.long)
        original = adapter(input_ids=batch, use_cache=False, enabled=False).logits
        hooked = adapter(input_ids=batch, use_cache=False).logits
        assert_close("zero_init_batch2", original, hooked, atol, rtol)
        # Greedy generation is a structural smoke; the byte tokenizer is not used.
        original_gen = adapter.generate(
            input_ids=samples[0], enabled=False, do_sample=False,
            max_new_tokens=3, min_new_tokens=3, pad_token_id=256, eos_token_id=258,
        )
        hooked_gen = adapter.generate(
            input_ids=samples[0], enabled=True, do_sample=False,
            max_new_tokens=3, min_new_tokens=3, pad_token_id=256, eos_token_id=258,
        )
        if not torch.equal(original_gen, hooked_gen):
            raise AssertionError("Zero-init greedy generation differs")
        EVIDENCE["checks"]["zero_init_greedy_generation"] = {
            "equal": True, "forward_calls": len(adapter.last_hook_trace),
            "generated_tokenizer_units": 3,
        }

    # Diagnostic gradient without optimizer step or model training.
    adapter.train()
    adapter.zero_grad(set_to_none=True)
    loss = adapter(input_ids=samples[1], labels=samples[1],
                   use_cache=False).loss
    loss.backward()
    grad = adapter.nca.readout.weight.grad
    if grad is None or not bool(torch.isfinite(grad).all()) or float(grad.abs().sum()) <= 0:
        raise AssertionError("No finite nonzero readout gradient")
    assert all(p.grad is None for p in adapter.backbone.parameters())
    EVIDENCE["checks"]["frozen_base_readout_gradient"] = {
        "loss": float(loss.detach()),
        "readout_grad_l1": float(grad.abs().sum()),
        "base_grads_none": True,
        "optimizer_steps": 0,
    }
    adapter.zero_grad(set_to_none=True)
    adapter.eval()

    # One manually perturbed nontrained weight to exercise causal field effects.
    with torch.no_grad():
        adapter.nca.readout.weight[0, 0] = 0.03
    with torch.inference_mode():
        ids = samples[-1]
        full = adapter(input_ids=ids, use_cache=False).logits
        original = adapter(input_ids=ids, enabled=False, use_cache=False).logits
        magnitude = float((full - original).abs().max())
        if magnitude <= 0:
            raise AssertionError("Nonzero cellular path had no effect")
        EVIDENCE["checks"]["nonzero_untrained_readout_changes_logits"] = {
            "max_abs": magnitude,
            "trained": False,
        }
        past = None
        cached_parts = []
        for i in range(ids.shape[1]):
            out = adapter(input_ids=ids[:, i:i+1],
                          past_key_values=past, use_cache=True)
            past = out.past_key_values
            cached_parts.append(out.logits[:, -1:])
            assert len(adapter.last_hook_trace) == 1
        cached = torch.cat(cached_parts, dim=1)
        assert_close("nonzero_nca_full_vs_incremental_cache",
                     full, cached, cache_atol, cache_rtol)
        altered = ids.clone()
        altered[:, -1] = 42
        before = adapter(input_ids=ids, use_cache=False).logits[:, :-1]
        after = adapter(input_ids=altered, use_cache=False).logits[:, :-1]
        assert_close("nonzero_nca_future_suffix_invariance",
                     before, after, cache_atol, cache_rtol)

    EVIDENCE["status"] = "PASS_PRELIMINARY_VOID1_QWEN3_HOOK_CONFORMANCE"
    EVIDENCE["limitations"] = [
        "Qwen3 model and exact custom byte tokenizer source SHA checked, but custom Python code was NOT executed",
        "No byte tokenizer roundtrip, chat template, EOS/Unicode or natural-language evaluation",
        "Only short valid raw vocabulary ID sequences; no actual tokenizer-generated inputs",
        "No optimizer step on model, no learned adapter and no quality comparison",
        "Independent NCR-001E causal/cache and tokenizer-specific evaluation still blocked",
    ]
    persist()
    print(json.dumps({
        "status": EVIDENCE["status"],
        "source_model": MODEL_ID,
        "checks": EVIDENCE["checks"],
        "peak_rss_kib": EVIDENCE["peak_rss_kib"],
        "custom_tokenizer_code_executed": False,
        "model_training_performed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        EVIDENCE["status"] = "FAILED_QWEN3_SOURCE_OR_CONFORMANCE_PREPASS"
        EVIDENCE["error"] = f"{type(exc).__name__}: {exc}"
        EVIDENCE["traceback"] = traceback.format_exc()[-5000:]
        persist()
        raise
