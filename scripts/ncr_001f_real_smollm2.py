#!/usr/bin/env python3
"""NCR-001F actual 135M HF matched MLP hook prepass; zero optimizer steps.

Uses original exact pinned SmolLM2 source with frozen full weights. Does NOT
compare trained NCA versus MLP, sample the sealed holdout or admit #72.
Preserves a partial machine-readable source/run receipt on every failure.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import traceback

import torch
import transformers
import huggingface_hub
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from ncr_001f_frozen_hooks import FrozenDepthGatedMLP, PINNED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr-001f/real-smollm2-mlp.json"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"

RECEIPT = {
    "experiment": "NCR-001F",
    "scope": "fresh_exact_SmolLM2_frozen_MLP_postblock_hooks_no_train_prepass",
    "status": "STARTED",
    "model_training_performed": False,
    "nca_training_performed": False,
    "comparative_training_performed": False,
    "optimizer_steps": 0,
    "train_authorized": False,
    "independent_ncr_001e_conformance_pass": False,
    "heldout_data_opened": False,
    "four_model_actual_host_mlp_qualified": False,
    "checks": {},
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_frozen_base(base) -> str:
    """Full in-memory tensor-byte check before and after all adapter forwards."""
    h = hashlib.sha256()
    for name, p in base.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(repr((tuple(p.shape), str(p.dtype))).encode("ascii"))
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def persist() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    OUT.write_text(json.dumps(RECEIPT, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")


def require_close(name, a, b, atol, rtol):
    diff = (a - b).abs()
    rec = {
        "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
        "atol": atol, "rtol": rtol,
        "close": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }
    RECEIPT["checks"][name] = rec
    persist()
    if not rec["close"]:
        raise AssertionError(f"{name}: out of frozen protocol tolerance {rec}")


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261026)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    spec = PINNED["smollm2"]
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    pin = pins["models"]["smollm2"]
    if protocol["train_authorized"] is not False:
        raise RuntimeError("NCR training authorization changed; stop all work")
    if set(protocol["models"]) != set(PINNED):
        raise RuntimeError("Four frozen model lanes mismatch")
    if (pin["repo_id"] != spec["model_id"]
            or pin["revision"] != spec["revision"]
            or pin["hf_weight_file"] != spec["weight_path"]
            or pin["weight_declared_lfs_sha256_not_downloaded"] != spec["weight_sha256"]
            or protocol["models"]["smollm2"]["taps_after_block_zero_based"] != list(spec["taps"])):
        raise RuntimeError("Source/model protocol and previously proven complete checkpoint disagree")
    RECEIPT["source_identity"] = dict(spec)
    RECEIPT["source_identity"]["taps"] = list(spec["taps"])
    RECEIPT["source_sha256"] = {
        p: sha_file(ROOT / p) for p in (
            "scripts/ncr_001f_controls.py",
            "scripts/ncr_001f_frozen_hooks.py",
            "scripts/ncr_001f_real_smollm2.py",
            "experiments/ncr-001/protocol.json",
            "experiments/ncr-001/source-pins.json",
        )
    }
    RECEIPT["runner"] = {
        "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", "local"),
        "github_sha": os.getenv("GITHUB_SHA", "local"),
        "github_job": os.getenv("GITHUB_JOB", "local"),
        "runner_os": os.getenv("RUNNER_OS", platform.system()),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    RECEIPT["source_files"] = {}
    persist()

    # Hash ACTUAL downloaded safetensors bytes, not merely an HF LFS declaration.
    for name, expected_key in (
        ("config.json", "config_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("model.safetensors", "weight_declared_lfs_sha256_not_downloaded"),
    ):
        f = Path(hf_hub_download(spec["model_id"], name, revision=spec["revision"]))
        got = sha_file(f)
        expected = pin[expected_key]
        RECEIPT["source_files"][name] = {
            "sha256": got, "expected_sha256": expected, "bytes": f.stat().st_size,
        }
        persist()
        if got != expected:
            raise RuntimeError(f"Actual source bytes drifted: {name}")
        print("Verified complete pinned source bytes:", name, got, flush=True)

    # No tokenizer/fixture derived training or hidden held-out input is used here.
    base = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=spec["revision"],
        torch_dtype=torch.float32, attn_implementation="eager",
    )
    base.eval()
    adapter = FrozenDepthGatedMLP(
        base, lane="smollm2", verified_weight_sha256=spec["weight_sha256"],
        source_revision=spec["revision"],
    )
    RECEIPT["base_frozen_parameters"] = sum(p.numel() for p in base.parameters())
    RECEIPT["gated_mlp_trainable_parameters"] = adapter.control.trainable_parameters()
    RECEIPT["nca_trainable_parameters_reference"] = 514 * spec["width"] + 984
    if RECEIPT["gated_mlp_trainable_parameters"] != 296576:
        raise RuntimeError("Matched control count drifted")
    if not all(not p.requires_grad for p in base.parameters()):
        raise RuntimeError("Base unexpectedly trainable")
    original_digest = fingerprint_frozen_base(base)
    RECEIPT["frozen_base_in_memory_sha256_before"] = original_digest
    persist()

    noop_atol = protocol["conformance"]["frozen_base_noop_atol"]
    noop_rtol = protocol["conformance"]["frozen_base_noop_rtol"]
    cache_atol = protocol["conformance"]["full_vs_cached_prefill_atol"]
    cache_rtol = protocol["conformance"]["full_vs_cached_prefill_rtol"]
    vocab = base.config.vocab_size
    with torch.inference_mode():
        for length in (16, 64, 128):
            ids = (torch.arange(length).reshape(1, -1) + 42) % vocab
            raw = adapter(input_ids=ids, enabled=False, use_cache=False).logits
            got = adapter(input_ids=ids, use_cache=False).logits
            require_close(f"zero_init_{length}_tokens", raw, got, noop_atol, noop_rtol)
            if adapter.last_hook_trace != [{"tap_count": 3, "lane": "smollm2"}]:
                raise RuntimeError("Missing ordered three-tap actual decoder visit")

        for side in ("left", "right"):
            # Use valid ordinary source token IDs; no sealed evaluation prompts.
            ids = torch.tensor([[23] * 20, [24] * 20])
            mask = torch.ones_like(ids)
            if side == "left":
                ids[1, :8] = 0
                mask[1, :8] = 0
            else:
                ids[1, -8:] = 0
                mask[1, -8:] = 0
            kwargs = dict(input_ids=ids, attention_mask=mask, use_cache=False)
            raw = adapter(enabled=False, **kwargs).logits
            got = adapter(**kwargs).logits
            require_close(f"zero_init_masked_batch2_{side}", raw, got, noop_atol, noop_rtol)

        prompt = torch.tensor([[101, 102, 103, 104]])
        options = dict(max_new_tokens=3, do_sample=False, pad_token_id=base.config.eos_token_id)
        raw = adapter.generate(prompt, enabled=False, **options)
        got = adapter.generate(prompt, **options)
        if not torch.equal(raw, got):
            raise RuntimeError("Initial greedy generation diverged from frozen model")
        RECEIPT["checks"]["zero_init_greedy_generate"] = {
            "equal": True,
            "generated": int(got.shape[1] - prompt.shape[1]),
            "decoder_forwards": len(adapter.last_hook_trace),
        }
        persist()

    # One source-bound gradient computation, NO optimizer construction or step.
    adapter.train()
    loss_ids = torch.tensor([[13, 14, 15, 16, 17, 18, 19, 20]])
    adapter.zero_grad(set_to_none=True)
    loss = adapter(input_ids=loss_ids, labels=loss_ids, use_cache=False).loss
    loss.backward()
    readout_grad = adapter.control.down.weight.grad
    if readout_grad is None or not torch.isfinite(readout_grad).all():
        raise RuntimeError("No finite real gated readout gradient")
    grad_l1 = float(readout_grad.abs().sum())
    if grad_l1 <= 0 or any(p.grad is not None for p in base.parameters()):
        raise RuntimeError("Missing gated gradient or unexpectedly trainable base")
    RECEIPT["checks"]["readout_gradient_only_no_optimizer"] = {
        "loss": float(loss.detach()), "grad_l1": grad_l1,
        "frozen_base_grads_none": True, "optimizer_steps": 0,
    }
    adapter.zero_grad(set_to_none=True)
    adapter.eval()
    persist()

    # Manually activate ONE untrained scalar to exercise nontrivial cached path.
    with torch.no_grad():
        adapter.control.down.weight[0, 0] = 0.05
    with torch.inference_mode():
        ids = (torch.arange(48).reshape(1, -1) + 50) % vocab
        full = adapter(input_ids=ids, use_cache=False).logits
        frozen = adapter(input_ids=ids, enabled=False, use_cache=False).logits
        max_effect = float((full - frozen).abs().max())
        if max_effect <= 0:
            raise RuntimeError("Manual untrained MLP readout did not affect output")
        RECEIPT["checks"]["nontrained_readout_effect"] = {"max_abs": max_effect}
        pieces = []
        past = None
        for i in range(ids.shape[1]):
            result = adapter(input_ids=ids[:, i:i + 1],
                             past_key_values=past, use_cache=True)
            pieces.append(result.logits[:, -1:])
            past = result.past_key_values
            if len(adapter.last_hook_trace) != 1:
                raise RuntimeError("Actual KV cached decoder failed per-token hook reset")
        cached = torch.cat(pieces, dim=1)
        require_close("nontrained_48_token_full_vs_real_incremental_KV",
                      full, cached, cache_atol, cache_rtol)
        alternate = ids.clone()
        alternate[:, -1] = (alternate[:, -1] + 7) % vocab
        before = adapter(input_ids=ids, use_cache=False).logits
        after = adapter(input_ids=alternate, use_cache=False).logits
        require_close("future_suffix_causal_invariance", before[:, :-1],
                      after[:, :-1], cache_atol, cache_rtol)
        adapter(input_ids=ids[:, :8], use_cache=False)
        reset = adapter(input_ids=ids, use_cache=False).logits
        require_close("request_A_B_A_reset", full, reset, noop_atol, noop_rtol)

        with tempfile.TemporaryDirectory() as td:
            adapter.save_adapter(td)
            copy = FrozenDepthGatedMLP(
                base, lane="smollm2", verified_weight_sha256=spec["weight_sha256"],
                source_revision=spec["revision"],
            )
            copy.load_adapter(td)
            cold = copy(input_ids=ids, use_cache=False).logits
            require_close("new_adapter_same_base_cold_reload", full, cold,
                          noop_atol, noop_rtol)
            meta = Path(td) / "metadata.json"
            obj = json.loads(meta.read_text())
            obj["weight_path"] = "chat/model.safetensors"
            meta.write_text(json.dumps(obj))
            try:
                copy.load_adapter(td)
            except RuntimeError:
                RECEIPT["checks"]["wrong_checkpoint_variant_refusal"] = True
            else:
                raise AssertionError("Wrong variant metadata accepted")

    final_digest = fingerprint_frozen_base(base)
    RECEIPT["frozen_base_in_memory_sha256_after"] = final_digest
    if final_digest != original_digest:
        raise RuntimeError("Actual frozen model parameter bytes changed")
    RECEIPT["checks"]["full_base_tensor_bytes_unchanged"] = True
    RECEIPT["status"] = "PASS_SOURCE_BOUND_REAL_SMOLLM2_MLP_HOOK_PREPASS__TRAINING_BLOCKED"
    RECEIPT["limits"] = [
        "Only SmolLM2 135M has an actual independent pinned MLP model invocation",
        "Sprocket Chat/Instruct and void.1 MLP hooks have only synthetic layouts",
        "Only same-process new adapter reload on the same original model",
        "No tool or memory serialization claimed",
        "Full four-model #72 acceptance not established",
        "No optimizer created, no learned NCA or MLP, no held-out quality result",
    ]
    persist()
    print(json.dumps(RECEIPT, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        RECEIPT["status"] = "FAILED_SOURCE_BOUND_REAL_SMOLLM2_MLP_HOOK_PREPASS"
        RECEIPT["error"] = f"{type(exc).__name__}: {exc}"
        RECEIPT["traceback"] = traceback.format_exc()[-6000:]
        persist()
        raise
