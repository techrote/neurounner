#!/usr/bin/env python3
"""NCR-001E: actual pinned Sprocket adapter replay across separate Python processes.

Source-only, synthetic prompt, no optimizer, no held-out fixture, no tool grammar.
The producer and consumer independently load the original frozen checkpoint.
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
import traceback

import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ncr_001e_sprocket_format import (
    SOURCE_REPO, SOURCE_REV, serialize_dialogue, validate_pinned_card,
    validate_pinned_tokenizer,
)
from ncr_001e_sprocket_fresh import SOURCE_SPECS, filehash, require

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
CARD_SHA = "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"
PROMPT = (
    "On the workbench, a compact motor turns an eccentric wheel. "
    "Record each bearing temperature and verify which fastener is loose. "
    "Do not replace a component until the power source is isolated. "
) * 28
PREFIX = 240
LENGTH = 256
CACHE_TAIL = LENGTH - PREFIX
REPORT = {
    "experiment": "NCR-001E",
    "status": "RUNNING",
    "stage": None,
    "actual_model_training": False,
    "optimizer_steps": 0,
    "heldout_evaluation": False,
    "tool_and_memory_behavior_qualified": False,
    "full_four_model_conformance_pass": False,
    "train_authorized": False,
    "checks": {},
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_receipt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    path.write_text(json.dumps(REPORT, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def guard_manifest(manifest: dict, *, variant: str, pin: dict,
                   implementation_commit: str, producer_pid: int) -> None:
    """Reject altered provenance before any adapter deserialization or replay."""
    expected = {
        "schema": 1,
        "variant": variant,
        "repo": SOURCE_REPO,
        "revision": SOURCE_REV,
        "checkpoint_file": pin["hf_weight_file"],
        "checkpoint_sha256": pin["weight_declared_lfs_sha256_not_downloaded"],
        "implementation_commit": implementation_commit,
        "protocol_sha256": filehash(PROTOCOL),
        "pins_sha256": filehash(PINS),
        "producer_pid": producer_pid,
        "training": False,
        "optimizer_steps": 0,
        "heldout_evaluation": False,
        "tool_and_memory_behavior_qualified": False,
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, "Cold-process manifest identity mismatch: " + key)
    require(manifest.get("producer_pid") != os.getpid(),
            "This is NOT a fresh Python interpreter: producer PID is current PID")
    for key in ("adapter_sha256", "metadata_sha256", "reference_sha256", "token_ids_sha256"):
        require(isinstance(manifest.get(key), str) and len(manifest[key]) == 64
                and all(c in "0123456789abcdef" for c in manifest[key]),
                "Cold-process manifest missing valid SHA256: " + key)


def compare(key: str, a: torch.Tensor, b: torch.Tensor, *,
            atol: float, rtol: float, exact: bool = False) -> None:
    require(a.shape == b.shape, key + ": shape mismatch")
    diff = (a.float() - b.float()).abs()
    maximum = float(diff.max())
    close = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    REPORT["checks"][key] = {
        "max_abs": maximum,
        "atol": atol,
        "rtol": rtol,
        "pass": close and (not exact or torch.equal(a, b)),
        "exact_required": exact,
    }
    save_receipt(OUT)
    require(REPORT["checks"][key]["pass"], key + ": cold/cache logits mismatch")


def verify_sources(variant: str) -> tuple[dict, dict, dict, type, object, object]:
    spec = SOURCE_SPECS[variant]
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    require(protocol["train_authorized"] is False, "Training is not admitted")
    require(set(protocol["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "Four-lane protocol changed")
    pin = pins["models"][spec["key"]]
    lane = protocol["models"][spec["key"]]
    require(pin["repo_id"] == lane["repo_id"] == SOURCE_REPO
            and pin["revision"] == lane["revision"] == SOURCE_REV
            and pin["variant"] == lane["variant"] == variant,
            "Source/model/variant pin mismatch")
    require(pin["hf_weight_file"] == lane["weight_file"] == spec["weight"],
            "Checkpoint path/variant swapped")
    other = pins["models"]["sprocket_chat" if variant == "instruct" else "sprocket_instruct"]
    require(other["weight_declared_lfs_sha256_not_downloaded"] !=
            pin["weight_declared_lfs_sha256_not_downloaded"],
            "Instruct and Chat must have distinct actual weights")
    impl = ROOT / "implementation"
    commit = subprocess.check_output(
        ["git", "-C", str(impl), "rev-parse", "HEAD"], text=True
    ).strip()
    require(commit == spec["implementation_commit"], "Original hook source not exact")
    sys.path.insert(0, str(impl / "scripts"))
    from importlib import import_module
    source_module = import_module(spec["module"])
    wrapper_cls = getattr(source_module, spec["class"])
    require(source_module.MODEL_ID == SOURCE_REPO
            and source_module.MODEL_REVISION == SOURCE_REV
            and list(source_module.POST_BLOCK_TAPS) == [7, 16, 25],
            "Original independent hook module source mismatch")

    requested = [
        ("README.md", None),
        (spec["config"], spec["config_pin"]),
        (spec["tokenizer_json"], spec["tokenizer_pin"]),
        (spec["tokenizer_config"], spec["tokenizer_config_pin"]),
        (spec["weight"], "weight_declared_lfs_sha256_not_downloaded"),
    ]
    if variant == "instruct":
        requested.append(("special_tokens_map.json", "special_tokens_map_sha256"))
    REPORT["source_files"] = {}
    for filename, pin_key in requested:
        path = Path(hf_hub_download(SOURCE_REPO, filename=filename,
                                    revision=SOURCE_REV, local_files_only=True))
        real = filehash(path)
        if pin_key:
            require(real == pin[pin_key], "Actual upstream source SHA differs: " + filename)
        if filename == "README.md":
            require(real == CARD_SHA, "Pinned model card modified")
            validate_pinned_card(path.read_text(encoding="utf-8"))
        if filename == spec["weight"]:
            require(path.stat().st_size == pin["weight_bytes"],
                    "Full source checkpoint length changed")
        REPORT["source_files"][filename] = {
            "sha256": real, "bytes": path.stat().st_size,
        }
        save_receipt(OUT)

    opts = {"revision": SOURCE_REV, "local_files_only": True}
    if spec["subfolder"]:
        opts["subfolder"] = spec["subfolder"]
    tok = AutoTokenizer.from_pretrained(SOURCE_REPO, **opts)
    inventory = validate_pinned_tokenizer(tok, variant=variant)
    REPORT["role_ids"] = {
        key: inventory["controls"][key]["token_ids"]
        for key in ("<|user|>", "<|end|>", "<|assistant|>")
    }
    require(REPORT["role_ids"] == {
        "<|user|>": [3], "<|end|>": [5], "<|assistant|>": [4]
    }, "Dedicated exact source role control IDs changed")
    encoded = serialize_dialogue([{"role": "user", "content": PROMPT}], variant=variant)
    ids = tok(encoded, return_tensors="pt", add_special_tokens=False)["input_ids"]
    require(ids.shape[-1] >= LENGTH, "Synthetic source-framed prompt too short")
    ids = ids[:, :LENGTH].contiguous()
    require(int(ids[0, 0]) == 3, "Missing source-defined user boundary")
    REPORT["token_ids_sha256"] = digest(ids.numpy().tobytes())
    REPORT["prompt_tokenizer_units"] = LENGTH
    REPORT["context_limit"] = lane["expected_config"]["max_position_embeddings"]

    model_opts = {
        "revision": SOURCE_REV, "torch_dtype": torch.float32,
        "attn_implementation": "eager",
        "local_files_only": True, "trust_remote_code": False,
    }
    if spec["subfolder"]:
        model_opts["subfolder"] = spec["subfolder"]
    base = AutoModelForCausalLM.from_pretrained(SOURCE_REPO, **model_opts).eval()
    wrapper = wrapper_cls(
        base, verified_weight_sha256=pin["weight_declared_lfs_sha256_not_downloaded"]
    ).eval()
    wrapper.verify_frozen_base()
    require(all(not p.requires_grad for p in base.parameters()),
            "Original Sprocket backbone is not fully frozen")
    require(wrapper.nca.trainable_parameters() == 658904,
            "Frozen NCA source changed parameter count")
    REPORT["base_parameters"] = wrapper.frozen_parameters()
    REPORT["adapter_parameters"] = wrapper.nca.trainable_parameters()
    return spec, pin, protocol, wrapper_cls, wrapper, ids


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("chat", "instruct"))
    parser.add_argument("--stage", required=True, choices=("producer", "consumer"))
    opts = parser.parse_args()
    variant, stage = opts.variant, opts.stage
    folder = ROOT / "runtime/ncr-001" / ("sprocket-cross-process-" + variant)
    OUT = folder / (stage + ".json")
    REPORT.update({
        "variant": variant, "stage": stage,
        "process_pid": os.getpid(), "process_parent_pid": os.getppid(),
        "runner": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_job": os.environ.get("GITHUB_JOB"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
        },
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    })
    save_receipt(OUT)
    require(os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Both independent hosts must execute strictly offline")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261027)
    spec, pin, protocol, wrapper_cls, wrapper, ids = verify_sources(variant)
    catol = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
    crtol = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])
    adapter_path = folder / "adapter"
    manifest_path = folder / "manifest.json"
    reference_path = folder / "reference.safetensors"

    if stage == "producer":
        require(not adapter_path.exists() and not manifest_path.exists()
                and not reference_path.exists(),
                "Existing cold-process evidence cannot be overwritten")
        with torch.no_grad():
            wrapper.nca.readout.weight[0, 0] = 0.03
        with torch.inference_mode():
            full = wrapper(input_ids=ids, use_cache=False).logits
            bare = wrapper(input_ids=ids, enabled=False, use_cache=False).logits
            effect = float((full - bare).abs().max())
            require(effect > 0, "Manually activated NCA did not affect logits")
            REPORT["checks"]["manual_untrained_nonzero_effect"] = {
                "max_abs": effect, "trained": False,
            }
            pref = wrapper(input_ids=ids[:, :PREFIX], use_cache=True)
            past = pref.past_key_values
            pieces = []
            for index in range(PREFIX, LENGTH):
                response = wrapper(
                    input_ids=ids[:, index:index + 1],
                    past_key_values=past, use_cache=True,
                )
                pieces.append(response.logits[:, -1:, :])
                past = response.past_key_values
                require(len(wrapper.last_hook_trace) == 1,
                        "Cache step did not produce a single independent depth field")
            cached = torch.cat(pieces, dim=1)
            compare("256_unit_240_prefill_16_incremental_tail",
                    full[:, PREFIX:], cached, atol=catol, rtol=crtol)
            altered = ids.clone()
            altered[:, -1] = (altered[:, -1] + 7) % 32000
            other_full = wrapper(input_ids=altered, use_cache=False).logits
            compare("future_suffix_strict_255_units",
                    full[:, :-1], other_full[:, :-1],
                    atol=0, rtol=0, exact=True)
            first = wrapper(input_ids=ids[:, :32], use_cache=False).logits
            middle = wrapper(input_ids=ids[:, 32:64], use_cache=False).logits
            again = wrapper(input_ids=ids[:, :32], use_cache=False).logits
            compare("request_A_B_A_producer", first, again,
                    atol=0, rtol=0, exact=True)
            tensors = {
                "active_tail": full[:, PREFIX:].cpu().contiguous(),
                "disabled_tail": bare[:, PREFIX:].cpu().contiguous(),
                "request_a": first.cpu().contiguous(),
                "request_b": middle.cpu().contiguous(),
            }
        folder.mkdir(parents=True, exist_ok=True)
        metadata = wrapper.save_adapter(adapter_path)
        require(metadata["model_variant"] == variant
                and metadata["checkpoint_sha256"] ==
                pin["weight_declared_lfs_sha256_not_downloaded"],
                "Saved adapter metadata lost frozen-model identity")
        save_file(tensors, str(reference_path))
        require(all(p.grad is None for p in wrapper.backbone.parameters()),
                "Original frozen model unexpectedly accumulated gradient")
        manifest = {
            "schema": 1, "variant": variant, "repo": SOURCE_REPO,
            "revision": SOURCE_REV,
            "checkpoint_file": spec["weight"],
            "checkpoint_sha256": pin["weight_declared_lfs_sha256_not_downloaded"],
            "implementation_commit": spec["implementation_commit"],
            "protocol_sha256": filehash(PROTOCOL),
            "pins_sha256": filehash(PINS),
            "producer_pid": os.getpid(),
            "adapter_sha256": filehash(adapter_path / "adapter.pt"),
            "metadata_sha256": filehash(adapter_path / "metadata.json"),
            "reference_sha256": filehash(reference_path),
            "token_ids_sha256": REPORT["token_ids_sha256"],
            "training": False, "optimizer_steps": 0,
            "heldout_evaluation": False,
            "tool_and_memory_behavior_qualified": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        REPORT["manifest_sha256"] = filehash(manifest_path)
        REPORT["producer_pid"] = os.getpid()
        REPORT["status"] = "PASS_SOURCE_BOUND_PROCESS_PRODUCER"
    else:
        require(manifest_path.is_file() and reference_path.is_file(),
                "No prior producer evidence; a cold restart cannot be claimed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        guard_manifest(manifest, variant=variant, pin=pin,
                       implementation_commit=spec["implementation_commit"],
                       producer_pid=int(manifest.get("producer_pid", -1)))
        require(manifest["token_ids_sha256"] == REPORT["token_ids_sha256"],
                "Source-framed prompt token IDs differ across processes")
        for filename, sha in [
            (adapter_path / "adapter.pt", manifest["adapter_sha256"]),
            (adapter_path / "metadata.json", manifest["metadata_sha256"]),
            (reference_path, manifest["reference_sha256"]),
        ]:
            require(filehash(filename) == sha,
                    "Cold-process handoff bytes changed: " + str(filename))
        wrapper.load_adapter(adapter_path)
        reference = load_file(str(reference_path), device="cpu")
        with torch.inference_mode():
            full = wrapper(input_ids=ids, use_cache=False).logits
            bare = wrapper(input_ids=ids, enabled=False, use_cache=False).logits
            first = wrapper(input_ids=ids[:, :32], use_cache=False).logits
            wrapper(input_ids=ids[:, 32:64], use_cache=False)
            again = wrapper(input_ids=ids[:, :32], use_cache=False).logits
            compare("restart_active_tail_exact",
                    full[:, PREFIX:], reference["active_tail"],
                    atol=0, rtol=0, exact=True)
            compare("restart_disabled_tail_exact",
                    bare[:, PREFIX:], reference["disabled_tail"],
                    atol=0, rtol=0, exact=True)
            compare("restart_request_A_exact",
                    first, reference["request_a"],
                    atol=0, rtol=0, exact=True)
            compare("restart_after_B_no_leak_exact",
                    again, reference["request_a"],
                    atol=0, rtol=0, exact=True)
            require(float((full - bare).abs().max()) > 0,
                    "Reloaded NCA became inactive after process restart")
        require(all(p.grad is None for p in wrapper.backbone.parameters()),
                "Replay unexpectedly populated frozen backbone gradients")
        REPORT["producer_pid"] = manifest["producer_pid"]
        REPORT["source_manifest_sha256"] = filehash(manifest_path)
        REPORT["source_reference_sha256"] = manifest["reference_sha256"]
        REPORT["source_adapter_sha256"] = manifest["adapter_sha256"]
        REPORT["status"] = "PASS_FRESH_INTERPRETER_SPROCKET_ADAPTER_REPLAY"
    REPORT["limits"] = [
        "Separate Python interpreter/PID and newly loaded frozen original checkpoint, same hosted runner/cache",
        "256 source-role token prefix; 240-prefill +16 genuine cached continuation, not full 2048-context",
        "Manually activated untrained adapter only; no optimization or held-out probing",
        "No source-approved Instruct tool-result/memory grammar or Chat tool ability claim",
        "No full independent four-model NCR-001E acceptance or training authorization",
    ]
    save_receipt(OUT)
    print(json.dumps({
        "status": REPORT["status"], "variant": variant,
        "stage": stage, "checks": REPORT["checks"],
        "runner": REPORT["runner"], "optimizer_steps": 0,
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        fallback_variant = sys.argv[sys.argv.index("--variant") + 1] \
            if "--variant" in sys.argv and sys.argv.index("--variant") + 1 < len(sys.argv) \
            else "unknown"
        fallback_stage = sys.argv[sys.argv.index("--stage") + 1] \
            if "--stage" in sys.argv and sys.argv.index("--stage") + 1 < len(sys.argv) \
            else "unknown"
        output = ROOT / "runtime/ncr-001" / ("sprocket-cross-process-" + fallback_variant) / (fallback_stage + ".json")
        REPORT["status"] = "FAILED_SOURCE_BOUND_CROSS_PROCESS_SPROCKET"
        REPORT["error"] = type(exc).__name__ + ": " + str(exc)
        REPORT["traceback"] = traceback.format_exc()[-7000:]
        save_receipt(output)
        raise
