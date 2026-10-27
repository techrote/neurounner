#!/usr/bin/env python3
"""NCR-001F: actual frozen Sprocket Chat/Instruct matched-MLP hook prepass.

Only source-documented ordinary user/assistant framing is exercised. The role
formatter is checked out at an exact, independently produced #95 commit.
No tool/memory grammar, held-out fixture, optimizer or training is involved.
A partial receipt is written on *every* failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import traceback

import torch
import transformers
import huggingface_hub
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from ncr_001f_frozen_hooks import FrozenDepthGatedMLP, PINNED

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
FORMAT_AUTHORITY = ROOT / "format-authority"
FORMAT_COMMIT = "2271218d625e1462b2cf218445da975efe17fe32"
CARD_SHA256 = "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"
SOURCE_FILES = {
    "chat": (
        ("chat/config.json", "chat_config_sha256"),
        ("chat/tokenizer.json", "chat_tokenizer_sha256"),
        ("chat/tokenizer_config.json", "chat_tokenizer_config_sha256"),
    ),
    "instruct": (
        ("config.json", "config_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("special_tokens_map.json", "special_tokens_map_sha256"),
    ),
}
REPORT = {
    "experiment": "NCR-001F",
    "scope": "source_bound_frozen_Sprocket_matched_MLP_real_model_no_training",
    "status": "STARTED",
    "training_performed": False,
    "optimizer_steps": 0,
    "comparative_training_performed": False,
    "train_authorized": False,
    "heldout_data_opened": False,
    "independent_ncr_001e_conformance_pass": False,
    "four_model_actual_host_mlp_qualified": False,
    "tool_and_memory_behavior_qualified": False,
    "separate_process_cold_reload_proved": False,
    "checks": {},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_source_contract(variant: str, protocol: dict, pins: dict) -> dict:
    """No model load permitted if identity, dependency or protocol drifts."""
    if variant not in ("chat", "instruct"):
        raise ValueError("Unknown Sprocket variant")
    if protocol.get("train_authorized") is not False:
        raise RuntimeError("Training admission is not closed")
    if set(protocol["models"]) != set(PINNED):
        raise RuntimeError("Frozen four-model protocol lane identities changed")
    lane = "sprocket_" + variant
    spec = PINNED[lane]
    model = protocol["models"][lane]
    pin = pins["models"][lane]
    sibling = pins["models"]["sprocket_instruct" if variant == "chat" else "sprocket_chat"]
    required = (
        spec["model_id"] == "kandivault/sprocket-500m",
        spec["revision"] == "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
        model["repo_id"] == pin["repo_id"] == spec["model_id"],
        model["revision"] == pin["revision"] == spec["revision"],
        model["variant"] == pin["variant"] == variant,
        model["weight_file"] == pin["hf_weight_file"] == spec["weight_path"],
        pin["weight_declared_lfs_sha256_not_downloaded"] == spec["weight_sha256"],
        sibling["weight_declared_lfs_sha256_not_downloaded"] != spec["weight_sha256"],
        pin["weight_bytes"] == 1002207984,
        model["taps_after_block_zero_based"] == list(spec["taps"]),
        pin["tap_indices_post_block_zero_based"] == list(spec["taps"]),
        model["expected_config"]["hidden_size"] == spec["width"],
        model["expected_config"]["num_hidden_layers"] == spec["layers"],
        pin["architecture"]["hidden"] == spec["width"],
        pin["architecture"]["layers"] == spec["layers"],
        spec["model_type"] == "llama",
    )
    if not all(required):
        raise RuntimeError("Wrong Sprocket Chat/Instruct source, weight, architecture or tap geometry")
    return spec


def fingerprint_backbone(base) -> str:
    h = hashlib.sha256()
    for name, param in base.named_parameters():
        h.update(name.encode())
        h.update(repr((tuple(param.shape), str(param.dtype))).encode())
        h.update(param.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def save(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out.write_text(json.dumps(REPORT, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare(out: Path, name: str, expected, observed, *, atol: float, rtol: float, exact=False):
    if expected.shape != observed.shape:
        raise RuntimeError(f"{name}: shapes disagree")
    delta = (expected - observed).abs()
    result = {
        "shape": list(expected.shape), "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()), "atol": atol, "rtol": rtol,
        "close": bool(torch.allclose(expected, observed, atol=atol, rtol=rtol)),
        "exact_required": exact,
    }
    REPORT["checks"][name] = result
    save(out)
    if not result["close"] or (exact and result["max_abs"] != 0.0):
        raise AssertionError(f"{name}: numeric/causal check failed: {result}")


def main(variant: str, out: Path) -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Actual-host execution must be offline after source prefetch")
    protocol = json.loads(PROTOCOL.read_text())
    pins = json.loads(PINS.read_text())
    spec = validate_source_contract(variant, protocol, pins)
    lane = "sprocket_" + variant
    pin = pins["models"][lane]
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.manual_seed(20261027)
    REPORT["variant"] = variant
    REPORT["source_identity"] = {**spec, "taps": list(spec["taps"])}
    REPORT["runner"] = {
        "run_id": os.getenv("GITHUB_RUN_ID", "local"),
        "attempt": os.getenv("GITHUB_RUN_ATTEMPT", "local"),
        "github_sha": os.getenv("GITHUB_SHA", "local"),
        "job": os.getenv("GITHUB_JOB", "local"),
        "runner_os": os.getenv("RUNNER_OS", platform.system()),
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_threads": torch.get_num_threads(), "torch": torch.__version__,
        "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
        "cpu_model": next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                           if s.startswith("model name")), "unknown")
                     if Path("/proc/cpuinfo").exists() else "unknown",
    }
    REPORT["source_sha256"] = {
        path: digest(ROOT / path) for path in (
            "experiments/ncr-001/protocol.json",
            "experiments/ncr-001/source-pins.json",
            "scripts/ncr_001f_controls.py",
            "scripts/ncr_001f_frozen_hooks.py",
            "scripts/ncr_001f_real_sprocket.py",
        )
    }
    save(out)

    if subprocess.check_output(["git", "-C", str(FORMAT_AUTHORITY), "rev-parse", "HEAD"],
                               text=True).strip() != FORMAT_COMMIT:
        raise RuntimeError("Independent source formatter checkout changed")
    sys.path.insert(0, str(FORMAT_AUTHORITY / "scripts"))
    from ncr_001e_sprocket_format import (
        SOURCE_REPO, SOURCE_REV, serialize_dialogue,
        validate_pinned_card, validate_pinned_tokenizer,
    )
    if (SOURCE_REPO, SOURCE_REV) != (spec["model_id"], spec["revision"]):
        raise RuntimeError("Reviewer's source format revision disagrees with model pin")
    REPORT["formatter_source"] = {
        "reviewed_pr_head": FORMAT_COMMIT,
        "script_sha256": digest(FORMAT_AUTHORITY / "scripts/ncr_001e_sprocket_format.py"),
        "supported_scope": "ordinary_user_assistant_roles_only",
        "tool_memory_grammar_supported": False,
    }
    save(out)

    files = [("README.md", None), *SOURCE_FILES[variant],
             (spec["weight_path"], "weight_declared_lfs_sha256_not_downloaded")]
    REPORT["verified_source_files"] = {}
    for name, pin_key in files:
        path = Path(hf_hub_download(spec["model_id"], name, revision=spec["revision"],
                                    local_files_only=True))
        got = digest(path)
        expected = CARD_SHA256 if name == "README.md" else pin[pin_key]
        REPORT["verified_source_files"][name] = {
            "sha256": got, "pinned_sha256": expected, "bytes": path.stat().st_size,
        }
        save(out)
        if got != expected:
            raise RuntimeError(f"Source file byte hash changed: {name}")
        if name == spec["weight_path"] and path.stat().st_size != pin["weight_bytes"]:
            raise RuntimeError("Actual complete model weight size changed")

    readme = Path(hf_hub_download(spec["model_id"], "README.md",
                                  revision=spec["revision"], local_files_only=True))
    validate_pinned_card(readme.read_text(encoding="utf-8"))
    tok_opts = {"revision": spec["revision"], "local_files_only": True, "trust_remote_code": False}
    model_opts = {**tok_opts, "torch_dtype": torch.float32, "attn_implementation": "eager"}
    if variant == "chat":
        tok_opts["subfolder"] = "chat"
        model_opts["subfolder"] = "chat"
    tok = AutoTokenizer.from_pretrained(spec["model_id"], **tok_opts)
    inventory = validate_pinned_tokenizer(tok, variant=variant)
    REPORT["role_sources"] = {
        "user": inventory["controls"]["<|user|>"]["token_ids"],
        "turn_end": inventory["controls"]["<|end|>"]["token_ids"],
        "assistant": inventory["controls"]["<|assistant|>"]["token_ids"],
        "tool_and_memory_behavior_qualified": False,
    }
    save(out)
    if tok.pad_token_id is None:
        tok.pad_token_id = inventory["controls"]["<|end|>"]["token_ids"][0]

    base = AutoModelForCausalLM.from_pretrained(spec["model_id"], **model_opts).eval()
    adapter = FrozenDepthGatedMLP(
        base, lane=lane, source_revision=spec["revision"],
        verified_weight_sha256=spec["weight_sha256"],
    ).eval()
    if any(p.requires_grad for p in base.parameters()):
        raise RuntimeError("Actual frozen model parameters became trainable")
    REPORT["frozen_base_parameters"] = sum(p.numel() for p in base.parameters())
    REPORT["gated_mlp_parameters"] = adapter.control.trainable_parameters()
    REPORT["reference_nca_parameters"] = 514 * spec["width"] + 984
    if REPORT["gated_mlp_parameters"] != 658432:
        raise RuntimeError("Parameter match unexpectedly changed")
    before_sha = fingerprint_backbone(base)
    REPORT["in_memory_base_before_sha256"] = before_sha
    save(out)

    def user_tokens(content: str):
        serialized = serialize_dialogue([{"role": "user", "content": content}],
                                        variant=variant)
        return tok(serialized, add_special_tokens=False, return_tensors="pt")["input_ids"]

    text = ("Inspect a warm bearing and record safe reversible steps. "
            "Measure vibration, inspect the guide rail and check the drive. ") * 24
    ids = user_tokens(text)
    if ids.shape[1] < 128:
        raise RuntimeError("Source role serialization yielded fewer than 128 units")
    REPORT["long_source_role_tokens"] = int(ids.shape[1])
    noop_atol = protocol["conformance"]["frozen_base_noop_atol"]
    noop_rtol = protocol["conformance"]["frozen_base_noop_rtol"]
    cache_atol = protocol["conformance"]["full_vs_cached_prefill_atol"]
    cache_rtol = protocol["conformance"]["full_vs_cached_prefill_rtol"]

    with torch.inference_mode():
        for length in (16, 64, 128):
            inputs = ids[:, :length]
            raw = adapter(input_ids=inputs, enabled=False, use_cache=False).logits
            mlp = adapter(input_ids=inputs, use_cache=False).logits
            compare(out, f"zero_mlp_{length}_source_role", raw, mlp,
                    atol=noop_atol, rtol=noop_rtol, exact=True)
            if adapter.last_hook_trace != [{"tap_count": 3, "lane": lane}]:
                raise RuntimeError("Actual Sprocket decoder did not visit all three MLP taps")
        for side in ("left", "right"):
            tok.padding_side = side
            batch = tok([
                serialize_dialogue([{"role": "user", "content": "A bearing rattles."}], variant=variant),
                serialize_dialogue([{"role": "user",
                                     "content": "Inspect the gearbox before the motor heats."}], variant=variant),
            ], padding=True, add_special_tokens=False, return_tensors="pt")
            raw = adapter(**batch, enabled=False, use_cache=False).logits
            mlp = adapter(**batch, use_cache=False).logits
            compare(out, f"zero_mlp_{side}_masked_batch2", raw, mlp,
                    atol=noop_atol, rtol=noop_rtol, exact=True)
        tok.padding_side = "right"
        prompt = user_tokens("Check the drive coupling.")
        gen = {"max_new_tokens": 4, "min_new_tokens": 4, "do_sample": False,
               "use_cache": True, "pad_token_id": tok.pad_token_id}
        if tok.eos_token_id is not None:
            gen["eos_token_id"] = tok.eos_token_id
        original = adapter.generate(prompt, enabled=False, **gen)
        zero = adapter.generate(prompt, **gen)
        if not torch.equal(original, zero):
            raise RuntimeError("Zero-initialized Sprocket MLP diverged on greedy generation")
        REPORT["checks"]["zero_mlp_greedy_generation"] = {
            "exact": True, "generated_units": int(zero.shape[1] - prompt.shape[1]),
        }
        save(out)

    # One backward only; no optimizer construction, update, training corpus or holdout.
    adapter.train()
    adapter.zero_grad(set_to_none=True)
    inputs = ids[:, :16]
    loss = adapter(input_ids=inputs, labels=inputs, use_cache=False).loss
    loss.backward()
    grad = adapter.control.down.weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0:
        raise RuntimeError("Missing actual-model MLP readout gradient")
    if any(p.grad is not None for p in base.parameters()):
        raise RuntimeError("Original frozen checkpoint accumulated gradients")
    REPORT["checks"]["readout_gradient_no_optimizer"] = {
        "finite_loss": bool(torch.isfinite(loss.detach())),
        "readout_grad_l1": float(grad.abs().sum()),
        "base_gradients_none": True, "optimizer_steps": 0,
    }
    if not REPORT["checks"]["readout_gradient_no_optimizer"]["finite_loss"]:
        raise RuntimeError("Nonfinite actual-model gradient loss")
    adapter.zero_grad(set_to_none=True)
    adapter.eval()
    save(out)

    with torch.no_grad():
        adapter.control.down.weight[0, 0] = 0.05
    with torch.inference_mode():
        x = ids[:, :48]
        full = adapter(input_ids=x, use_cache=False).logits
        frozen = adapter(input_ids=x, enabled=False, use_cache=False).logits
        effect = float((full - frozen).abs().max())
        if not effect > 0:
            raise RuntimeError("Manually activated, untrained MLP had no effect on logits")
        REPORT["checks"]["manual_untrained_readout_effect"] = {
            "max_abs": effect, "learned": False,
        }
        save(out)
        cache = None
        pieces = []
        for index in range(x.shape[1]):
            step = adapter(input_ids=x[:, index:index + 1],
                           past_key_values=cache, use_cache=True)
            pieces.append(step.logits[:, -1:])
            cache = step.past_key_values
            if adapter.last_hook_trace != [{"tap_count": 3, "lane": lane}]:
                raise RuntimeError("Real incremental KV-cache forward did not reset MLP tap guards")
        cached = torch.cat(pieces, dim=1)
        compare(out, "active_48_full_vs_incremental_KV", full, cached,
                atol=cache_atol, rtol=cache_rtol)
        alternate = x.clone()
        alternate[:, -1] = (alternate[:, -1] + 7) % base.config.vocab_size
        changed = adapter(input_ids=alternate, use_cache=False).logits
        compare(out, "causal_future_suffix_exact", full[:, :-1], changed[:, :-1],
                atol=cache_atol, rtol=cache_rtol, exact=True)
        adapter(input_ids=ids[:, 8:24], use_cache=False)
        reset = adapter(input_ids=x, use_cache=False).logits
        compare(out, "request_A_B_A_exact", full, reset,
                atol=noop_atol, rtol=noop_rtol, exact=True)

        with tempfile.TemporaryDirectory() as td:
            adapter.save_adapter(td)
            fresh = FrozenDepthGatedMLP(
                base, lane=lane, source_revision=spec["revision"],
                verified_weight_sha256=spec["weight_sha256"],
            ).eval()
            fresh.load_adapter(td)
            cold = fresh(input_ids=x, use_cache=False).logits
            compare(out, "new_adapter_same_frozen_process_exact", full, cold,
                    atol=noop_atol, rtol=noop_rtol, exact=True)
            metadata_path = Path(td) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            for key, wrong in (
                ("lane", "sprocket_instruct" if variant == "chat" else "sprocket_chat"),
                ("weight_path", "model.safetensors" if variant == "chat" else "chat/model.safetensors"),
                ("verified_weight_sha256", "0" * 64),
                ("model_revision", "UNPINNED"),
            ):
                modified = dict(metadata, **{key: wrong})
                metadata_path.write_text(json.dumps(modified))
                try:
                    fresh.load_adapter(td)
                except RuntimeError:
                    REPORT["checks"]["refuse_corrupt_metadata_" + key] = True
                else:
                    raise AssertionError(f"Cross-variant or unpinned adapter accepted: {key}")
            metadata_path.write_text(json.dumps(metadata))
    after_sha = fingerprint_backbone(base)
    REPORT["in_memory_base_after_sha256"] = after_sha
    if before_sha != after_sha:
        raise RuntimeError("Full original frozen Sprocket backbone parameter bytes mutated")
    REPORT["checks"]["frozen_original_tensor_bytes_unchanged"] = True
    REPORT["status"] = f"PASS_SOURCE_BOUND_SPROCKET_{variant.upper()}_MLP_PREPASS__NO_TRAINING_ADMITTED"
    REPORT["limitations"] = [
        "Only ordinary user/assistant source roles; tool/result/memory grammar unqualified",
        "48 true one-unit incremental cache steps, not 2048-token full-context qualification",
        "Adapter new instance against same frozen model/process, not separate-process restart",
        "MLP conformance prepass, NOT matched trained MLP-vs-NCA comparison",
        "No held-out fixtures opened, no optimizer, no NCR-001E final acceptance or training",
    ]
    save(out)
    print(json.dumps({"status": REPORT["status"], "checks": REPORT["checks"],
                      "peak_rss_kib": REPORT["peak_rss_kib"]}, indent=2), flush=True)


if __name__ == "__main__":
    v = "unknown"
    destination = ROOT / "runtime/ncr-001f/sprocket-unknown-mlp.json"
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--variant", required=True, choices=["chat", "instruct"])
        parsed = parser.parse_args()
        v = parsed.variant
        destination = ROOT / f"runtime/ncr-001f/sprocket-{v}-mlp.json"
        main(v, destination)
    except Exception as exc:
        REPORT["status"] = "FAILED_SOURCE_BOUND_SPROCKET_MLP_PREPASS__TRAINING_BLOCKED"
        REPORT["variant"] = v
        REPORT["error"] = f"{type(exc).__name__}: {exc}"
        REPORT["traceback"] = traceback.format_exc()[-6500:]
        save(destination)
        raise
