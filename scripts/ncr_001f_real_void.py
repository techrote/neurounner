#!/usr/bin/env python3
"""NCR-001F: reviewed UTF-8 ByteTokenizer + original frozen void.1 MLP hooks.

Source-bound, zero-optimizer, no-heldout prepass. Do not interpret this as
NCR-001E admission or a trained NCA/MLP comparison. Persist failures too.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
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
OUT = ROOT / "runtime/ncr-001f/void1-real-mlp.json"
REVIEW = ROOT / "byte-authority"
REVIEW_SHA = "1fdc2e7e5c7b7b77923dbc4b47dcafa132aef47a"
SOURCE_FILES = {
    "model.safetensors": "weight_declared_lfs_sha256_not_downloaded",
    "config.json": "config_sha256",
    "tokenization_byte.py": "tokenization_byte_py_sha256",
    "byte_vocab.json": "byte_vocab_sha256",
    "tokenizer_config.json": "tokenizer_config_sha256",
    "special_tokens_map.json": "special_tokens_map_sha256",
}
RECORD = {
    "experiment": "NCR-001F",
    "scope": "actual_frozen_void1_MLP_reviewed_byte_tokenizer_no_training",
    "status": "STARTED",
    "model_training_performed": False,
    "comparative_training_performed": False,
    "optimizer_steps": 0,
    "heldout_data_opened": False,
    "train_authorized": False,
    "final_ncr_001e_admission": False,
    "four_model_mlp_qualification": False,
    "new_interpreter_cold_reload": False,
    "eos_quality_measured": False,
    "checks": {},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def save() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    RECORD["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUT.write_text(json.dumps(RECORD, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def source_contract(protocol: dict, sources: dict) -> dict:
    """Refuse drift before importing any untrusted tokenizer Python."""
    spec = PINNED["void1"]
    pin = sources["models"]["void1"]
    lane = protocol["models"]["void1"]
    require(protocol.get("train_authorized") is False, "Training permission unexpectedly changed")
    require(set(protocol["models"]) == set(PINNED), "Frozen four-lane protocol changed")
    for ok in (
        spec["model_id"] == "dotlabs/void.1",
        spec["revision"] == "9c2be5087772867d45961abd11f6b35110617563",
        spec["model_type"] == "qwen3",
        spec["weight_path"] == "model.safetensors",
        spec["width"] == 640 and spec["layers"] == 24,
        tuple(spec["taps"]) == (7, 15, 23),
        pin["repo_id"] == lane["repo_id"] == spec["model_id"],
        pin["revision"] == lane["revision"] == spec["revision"],
        pin["variant"] == lane["variant"] == "base",
        pin["hf_weight_file"] == lane["weight_file"] == spec["weight_path"],
        pin["weight_declared_lfs_sha256_not_downloaded"] == spec["weight_sha256"],
        pin["weight_bytes"] == 360622744,
        pin["architecture"]["layers"] == spec["layers"],
        pin["architecture"]["hidden"] == spec["width"],
        pin["tap_indices_post_block_zero_based"] == lane["taps_after_block_zero_based"] == list(spec["taps"]),
        lane["tokenizer_reference"]["train002b_source_sha256"] == pin["tokenization_byte_py_sha256"],
        pin["tokenization_byte_py_sha256"] == "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03",
    ):
        require(ok, "Pinned void.1 source/tokenizer/architecture mismatch")
    return pin


def compare(name: str, left: torch.Tensor, right: torch.Tensor,
            *, atol: float, rtol: float, exact: bool = False) -> None:
    require(left.shape == right.shape, name + ": tensor shape mismatch")
    delta = (left.float() - right.float()).abs()
    result = {
        "shape": list(left.shape),
        "max_abs": float(delta.max()),
        "atol": float(atol), "rtol": float(rtol),
        "close": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        "exact_required": exact,
        "exact": bool(torch.equal(left, right)),
    }
    RECORD["checks"][name] = result
    save()
    require(result["close"] and (not exact or result["exact"]),
            name + ": causal/cache/no-op check failed")


def base_fingerprint(model) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(repr((tuple(p.shape), str(p.dtype))).encode("utf-8"))
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Reviewed tokenizer and actual frozen base require offline execution")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261027)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sources = json.loads(PINS.read_text(encoding="utf-8"))
    pin = source_contract(protocol, sources)
    spec = PINNED["void1"]
    require(REVIEW.is_dir(), "Missing independent original tokenizer review checkout")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(REVIEW), "rev-parse", "HEAD"], text=True
    ).strip()
    require(actual_commit == REVIEW_SHA, "Unpinned tokenizer source reviewer")
    RECORD["source"] = {
        "model_repo": spec["model_id"], "revision": spec["revision"],
        "model_weight_sha256": spec["weight_sha256"],
        "byte_reviewer_commit": actual_commit,
        "byte_review_script_sha256": sha(REVIEW / "scripts/ncr_001k_byte_runtime.py"),
        "protocol_sha256": sha(PROTOCOL),
        "source_pins_sha256": sha(PINS),
        "hook_sha256": sha(ROOT / "scripts/ncr_001f_frozen_hooks.py"),
        "runner_sha256": sha(ROOT / "scripts/ncr_001f_real_void.py"),
    }
    RECORD["runner"] = {
        "run_id": os.getenv("GITHUB_RUN_ID", "local"),
        "attempt": os.getenv("GITHUB_RUN_ATTEMPT", "local"),
        "github_sha": os.getenv("GITHUB_SHA", "local"),
        "job": os.getenv("GITHUB_JOB", "local"),
        "runner_name": os.getenv("RUNNER_NAME", "local"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_threads": torch.get_num_threads(),
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown")
                     if Path("/proc/cpuinfo").exists() else "unknown",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
    }
    save()
    RECORD["source_files"] = {}
    for filename, field in SOURCE_FILES.items():
        path = Path(hf_hub_download(spec["model_id"], filename=filename,
                                    revision=spec["revision"], local_files_only=True))
        digest = sha(path)
        RECORD["source_files"][filename] = {
            "sha256": digest, "expected": pin[field], "bytes": path.stat().st_size
        }
        save()
        require(digest == pin[field], "Changed pinned source bytes: " + filename)
    require(RECORD["source_files"]["model.safetensors"]["bytes"] == pin["weight_bytes"],
            "Incomplete full original checkpoint")

    # Inspect the exact reviewed source shape before any trust_remote_code import.
    sys.path.insert(0, str(REVIEW / "scripts"))
    from ncr_001k_byte_runtime import verify_reviewed_shape
    tokenizer_file = Path(hf_hub_download(
        spec["model_id"], filename="tokenization_byte.py",
        revision=spec["revision"], local_files_only=True,
    ))
    RECORD["tokenizer_reviewed_shape"] = verify_reviewed_shape(
        ast.parse(tokenizer_file.read_text(encoding="utf-8"), filename="tokenization_byte.py")
    )
    save()
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"],
        trust_remote_code=True, use_fast=False, local_files_only=True,
    )
    runtime_file = inspect.getsourcefile(tokenizer.__class__)
    require(bool(runtime_file) and sha(Path(runtime_file)) == pin["tokenization_byte_py_sha256"],
            "Executed tokenizer source did not match reviewed bytes")
    require(tokenizer.vocab_size == 259 and tokenizer.pad_token_id == 256
            and tokenizer.bos_token_id == 257 and tokenizer.eos_token_id == 258,
            "Invalid pinned PAD/BOS/EOS IDs")
    require(tokenizer.encode("<eos>", add_special_tokens=True) == list(b"<eos>"),
            "Literal EOS text control mismatch")
    RECORD["tokenizer"] = {
        "byte_source_sha256": pin["tokenization_byte_py_sha256"],
        "actual_source_reviewed": True,
        "pad_id": 256, "bos_id": 257, "eos_id": 258,
        "literal_eos_ids": [60, 101, 111, 115, 62],
        "automatic_eos_insertion": False,
    }
    save()

    base = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=spec["revision"], torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False,
        local_files_only=True,
    ).eval()
    adapter = FrozenDepthGatedMLP(
        base, lane="void1", verified_weight_sha256=spec["weight_sha256"],
        source_revision=spec["revision"],
    ).eval()
    adapter.verify_frozen_base()
    require(all(not p.requires_grad for p in base.parameters()),
            "Original frozen base became trainable")
    RECORD["frozen_base_parameters"] = sum(p.numel() for p in base.parameters())
    RECORD["mlp_parameters"] = adapter.control.trainable_parameters()
    RECORD["nca_parameters_reference"] = 514 * spec["width"] + 984
    require(RECORD["mlp_parameters"] == 329472, "Matched control parameter count drift")
    before_sha = base_fingerprint(base)
    RECORD["loaded_original_base_sha256_before"] = before_sha
    save()

    text = ("🔧 café 日本語 πΩ — The workshop hinge moves along a guide rail. "
            "Measure the force and inspect the latch without altering the source. ") * 9
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"]
    require(ids.shape[1] >= 128, "Synthetic multilingual byte prompt too short")
    require(torch.equal(ids[0], torch.tensor(list(text.encode("utf-8")))),
            "Reviewed source UTF-8 encoding changed")
    RECORD["input_lengths"] = {
        "characters": len(text), "utf8_bytes": len(text.encode("utf-8")),
        "tokenizer_units": ids.shape[1],
    }
    noop_a = float(protocol["conformance"]["frozen_base_noop_atol"])
    noop_r = float(protocol["conformance"]["frozen_base_noop_rtol"])
    cache_a = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
    cache_r = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])

    with torch.inference_mode():
        for length in (16, 64, 128):
            raw = adapter(input_ids=ids[:, :length], enabled=False, use_cache=False).logits
            got = adapter(input_ids=ids[:, :length], use_cache=False).logits
            compare(f"zero_mlp_UTF8_{length}", raw, got,
                    atol=noop_a, rtol=noop_r, exact=True)
            require(adapter.last_hook_trace == [{"tap_count": 3, "lane": "void1"}],
                    "Pinned Qwen3 post-block taps not all visited")
        for side in ("left", "right"):
            tokenizer.padding_side = side
            batch = tokenizer(["🔧 bolt", "日本語: a mechanical latch."],
                              padding=True, return_tensors="pt")
            raw = adapter(**batch, enabled=False, use_cache=False).logits
            got = adapter(**batch, use_cache=False).logits
            compare("zero_mlp_UTF8_masked_" + side, raw, got,
                    atol=noop_a, rtol=noop_r, exact=True)
        tokenizer.padding_side = "left"
        generation = {
            "max_new_tokens": 4, "min_new_tokens": 4, "do_sample": False,
            "eos_token_id": 258, "pad_token_id": 256, "use_cache": True,
        }
        prompt = ids[:, :24]
        raw_gen = adapter.generate(prompt, enabled=False, **generation)
        mlp_gen = adapter.generate(prompt, **generation)
        require(torch.equal(raw_gen, mlp_gen), "Zero-init frozen ByteTokenizer generation drifted")
        RECORD["checks"]["zero_mlp_greedy_byte_units"] = {
            "exact": True, "generated_units": int(raw_gen.shape[1] - prompt.shape[1]),
            "eos_id": 258, "eos_reached": bool((mlp_gen[:, 24:] == 258).any()),
            "eos_quality_measured": False,
        }
        save()

    # A finite readout-gradient diagnostic is not an optimization or training run.
    adapter.train()
    adapter.zero_grad(set_to_none=True)
    loss_ids = ids[:, :16]
    loss = adapter(input_ids=loss_ids, labels=loss_ids, use_cache=False).loss
    loss.backward()
    grad = adapter.control.down.weight.grad
    require(grad is not None and bool(torch.isfinite(grad).all())
            and float(grad.abs().sum()) > 0 and bool(torch.isfinite(loss.detach())),
            "Frozen Qwen3 MLP readout gradient is absent or nonfinite")
    require(all(p.grad is None for p in base.parameters()),
            "Original Qwen3 base acquired gradients")
    RECORD["checks"]["readout_gradient_only_zero_steps"] = {
        "finite_loss": True, "l1": float(grad.abs().sum()),
        "base_gradients_none": True, "optimizer_steps": 0,
    }
    adapter.zero_grad(set_to_none=True)
    adapter.eval()
    save()

    with torch.no_grad():
        adapter.control.down.weight[0, 0] = 0.05
    with torch.inference_mode():
        x = ids[:, :48]
        full = adapter(input_ids=x, use_cache=False).logits
        raw = adapter(input_ids=x, enabled=False, use_cache=False).logits
        effect = float((full - raw).abs().max())
        require(effect > 0, "Nontrained manually activated MLP had no effect")
        RECORD["checks"]["manual_untrained_effect"] = {"max_abs": effect, "trained": False}
        save()
        cache, pieces = None, []
        for i in range(48):
            response = adapter(input_ids=x[:, i:i+1], past_key_values=cache, use_cache=True)
            pieces.append(response.logits[:, -1:])
            cache = response.past_key_values
            require(adapter.last_hook_trace == [{"tap_count": 3, "lane": "void1"}],
                    "Cached Qwen3 decoder hooks did not reset per byte")
        cached = torch.cat(pieces, dim=1)
        compare("active_48_UTF8_full_vs_true_incremental_KV", full, cached,
                atol=cache_a, rtol=cache_r)
        changed = x.clone()
        changed[:, -1] = (changed[:, -1] + 7) % 256
        changed_logits = adapter(input_ids=changed, use_cache=False).logits
        compare("strict_causal_future_suffix", full[:, :-1], changed_logits[:, :-1],
                atol=0, rtol=0, exact=True)
        adapter(input_ids=ids[:, 24:60], use_cache=False)
        again = adapter(input_ids=x, use_cache=False).logits
        compare("request_A_B_A_no_cross_request_state", full, again,
                atol=0, rtol=0, exact=True)

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_adapter(directory)
            copy = FrozenDepthGatedMLP(
                base, lane="void1", verified_weight_sha256=spec["weight_sha256"],
                source_revision=spec["revision"],
            ).eval()
            copy.load_adapter(directory)
            cold = copy(input_ids=x, use_cache=False).logits
            compare("new_adapter_same_original_model_exact", full, cold,
                    atol=0, rtol=0, exact=True)
            metadata_file = Path(directory) / "metadata.json"
            metadata = json.loads(metadata_file.read_text())
            for key, wrong in (
                ("lane", "sprocket_chat"),
                ("model_revision", "UNPINNED"),
                ("weight_path", "chat/model.safetensors"),
                ("verified_weight_sha256", "0" * 64),
                ("model_type", "llama"),
            ):
                metadata_file.write_text(json.dumps(dict(metadata, **{key: wrong})))
                try:
                    copy.load_adapter(directory)
                except RuntimeError:
                    RECORD["checks"]["reject_wrong_metadata_" + key] = True
                else:
                    raise AssertionError("Wrong frozen adapter metadata was accepted: " + key)
            metadata_file.write_text(json.dumps(metadata))
    after_sha = base_fingerprint(base)
    RECORD["loaded_original_base_sha256_after"] = after_sha
    require(before_sha == after_sha, "Frozen full original Qwen3 tensor bytes changed")
    RECORD["checks"]["frozen_base_all_bytes_unchanged"] = True
    RECORD["status"] = "PASS_SOURCE_BOUND_VOID1_UTF8_MLP_REAL_MODEL_PREPASS__TRAINING_BLOCKED"
    RECORD["limits"] = [
        "Real frozen Qwen3 and hash-reviewed executed ByteTokenizer; ordinary synthetic text only",
        "Manually activated nontrained MLP and gradient without optimizer, zero steps",
        "48 byte-units full vs genuine incremental cache; not 2048-context admission",
        "Adapter restoration on same original base/process only, no fresh interpreter",
        "Only ByteTokenizer EOS control parity, no EOS rate, historical correction impact or quality measured",
        "NCA and MLP scientific comparison and final NCR-001E admission still blocked",
    ]
    save()
    print(json.dumps({"status": RECORD["status"], "checks": RECORD["checks"],
                      "source": RECORD["source"], "runner": RECORD["runner"]},
                     indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        RECORD["status"] = "FAIL_SOURCE_BOUND_VOID1_MLP__TRAINING_BLOCKED"
        RECORD["error"] = type(exc).__name__ + ": " + str(exc)
        RECORD["traceback"] = traceback.format_exc()[-6500:]
        save()
        raise
