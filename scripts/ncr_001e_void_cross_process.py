#!/usr/bin/env python3
"""Fresh original frozen void.1 Qwen3/ByteTokenizer NCA adapter in two OS interpreters.

NCR-001E source/runtime prepass only: no optimizer, held-out access or training.
The producer and consumer independently load the pinned 90M frozen backbone and
reviewed dynamic byte tokenizer. They exchange a hash-bound adapter and logits.
"""
from __future__ import annotations

import argparse
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
import traceback

import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ncr_001e_void_process_guard import (
    REPO, PIN, ORIGINAL_HOOK, WEIGHT, BYTE_SOURCE, LENGTH, PREFIX,
    require, sha256_file, validate_protocol, validate_tokenizer_units,
    validate_manifest,
)
from ncr_001k_byte_runtime import verify_reviewed_shape

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "implementation"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
OUTPUT_ROOT = ROOT / "runtime/ncr-001/void1-cross-process"
SOURCE_TEXT = (
    "🔧 café 日本語 πΩ — The workshop hinge shifts across the guide rail. "
    "A mechanic compares small forces without changing the baseline. "
) * 8
REPORT: dict = {}
OUT: Path


def save() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUT.write_text(json.dumps(REPORT, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                   encoding="utf-8")


def compare(label: str, a: torch.Tensor, b: torch.Tensor, *,
            atol: float, rtol: float, exact: bool = False) -> None:
    require(a.shape == b.shape, "Source replay logits shape mismatch: " + label)
    maximum = float((a.float() - b.float()).abs().max())
    close = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    passed = close and (not exact or torch.equal(a, b))
    REPORT["checks"][label] = {
        "pass": passed, "max_abs": maximum, "atol": atol, "rtol": rtol,
        "exact_required": exact,
    }
    save()
    require(passed, "Source replay failed: " + label)


def verified_model_and_bytes():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    pin = validate_protocol(protocol, pins)
    require(IMPL.is_dir(), "Original frozen Qwen3 source checkout missing")
    commit = subprocess.check_output(
        ["git", "-C", str(IMPL), "rev-parse", "HEAD"], text=True
    ).strip()
    require(commit == ORIGINAL_HOOK, "Actual original frozen adapter source SHA changed")
    sys.path.insert(0, str(IMPL / "scripts"))
    from ncr_001_void1 import (
        FrozenVoid1DepthNCA, MODEL_ID, MODEL_REVISION, POST_BLOCK_TAPS
    )
    require(MODEL_ID == REPO and MODEL_REVISION == PIN and
            list(POST_BLOCK_TAPS) == [7, 15, 23], "Original model/hook implementation mismatch")
    REPORT["source"] = {
        "repo": REPO, "revision": PIN, "original_hook_commit": commit,
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_pins_sha256": sha256_file(PINS),
        "byte_tokenizer_source_sha256": BYTE_SOURCE,
    }
    REPORT["source_files"] = {}
    for filename, field in (
        ("model.safetensors", "weight_declared_lfs_sha256_not_downloaded"),
        ("config.json", "config_sha256"),
        ("tokenization_byte.py", "tokenization_byte_py_sha256"),
        ("byte_vocab.json", "byte_vocab_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("special_tokens_map.json", "special_tokens_map_sha256"),
    ):
        path = Path(hf_hub_download(REPO, filename=filename, revision=PIN,
                                    local_files_only=True))
        actual = sha256_file(path)
        require(actual == pin[field], "Pinned actual model/tokenizer source bytes drifted: " + filename)
        if filename == "model.safetensors":
            require(path.stat().st_size == 360622744 and actual == WEIGHT,
                    "Full actual frozen Qwen3 weight identity mismatch")
        REPORT["source_files"][filename] = {
            "sha256": actual, "bytes": path.stat().st_size,
        }
        save()
    source_file = Path(hf_hub_download(
        REPO, filename="tokenization_byte.py", revision=PIN, local_files_only=True
    ))
    REPORT["reviewed_tokenizer_shape"] = verify_reviewed_shape(
        ast.parse(source_file.read_text(encoding="utf-8"), filename="tokenization_byte.py")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        REPO, revision=PIN, trust_remote_code=True, use_fast=False,
        local_files_only=True
    )
    executed_path = inspect.getsourcefile(tokenizer.__class__)
    require(bool(executed_path) and
            sha256_file(Path(executed_path)) == BYTE_SOURCE,
            "Executed remote ByteTokenizer differs from reviewed pinned source")
    raw_ids = tokenizer(SOURCE_TEXT, return_tensors="pt",
                        add_special_tokens=True)["input_ids"]
    require(raw_ids.shape[-1] >= LENGTH, "Insufficient reviewed mixed UTF-8 positions")
    ids = raw_ids[:, :LENGTH].contiguous()
    validate_tokenizer_units(tokenizer, ids, SOURCE_TEXT)
    REPORT["input"] = {
        "utf8_bytes": len(SOURCE_TEXT.encode("utf-8")),
        "python_characters": len(SOURCE_TEXT),
        "original_tokenizer_units": int(raw_ids.shape[1]),
        "selected_units": LENGTH,
        "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "pad_id": 256, "bos_id": 257, "eos_id": 258,
        "literal_eos_text_is_five_ordinary_bytes": True,
    }
    save()

    base = AutoModelForCausalLM.from_pretrained(
        REPO, revision=PIN, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False,
        local_files_only=True
    ).eval()
    wrapper = FrozenVoid1DepthNCA(
        base, verified_weight_sha256=WEIGHT
    ).eval()
    wrapper.verify_frozen_base()
    require(wrapper.nca.trainable_parameters() == 329944,
            "Original frozen void1 NCA parameter geometry changed")
    require(all(not p.requires_grad for p in base.parameters()),
            "Frozen Qwen3 base unexpectedly trainable")
    REPORT["frozen_base_parameters"] = wrapper.frozen_parameters()
    REPORT["nca_parameters"] = wrapper.nca.trainable_parameters()
    save()
    return protocol, pin, wrapper, ids


def main() -> None:
    global OUT, REPORT
    cli = argparse.ArgumentParser()
    cli.add_argument("--stage", required=True, choices=("producer", "consumer"))
    stage = cli.parse_args().stage
    OUT = OUTPUT_ROOT / (stage + ".json")
    REPORT = {
        "experiment": "NCR-001E", "model_key": "void1", "stage": stage,
        "status": "RUNNING", "checks": {},
        "model_training_performed": False, "optimizer_steps": 0,
        "train_authorized": False, "heldout_evaluation": False,
        "full_four_model_conformance_pass": False,
        "historical_byte_eos_impact_measured": False,
        "runner": {
            "pid": os.getpid(), "ppid": os.getppid(),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "job": os.environ.get("GITHUB_JOB"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    save()
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Producer/consumer must both run offline with pinned source")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261027)
    protocol, pin, wrapper, ids = verified_model_and_bytes()
    atol = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
    rtol = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])
    adapter = OUTPUT_ROOT / "adapter"
    manifest_path = OUTPUT_ROOT / "manifest.json"
    reference_path = OUTPUT_ROOT / "reference.safetensors"

    if stage == "producer":
        require(not any(path.exists() for path in (adapter, manifest_path, reference_path)),
                "Producer refused to overwrite pre-existing handoff evidence")
        with torch.inference_mode():
            no_adapter = wrapper(input_ids=ids, enabled=False, use_cache=False).logits
            zero_adapter = wrapper(input_ids=ids, use_cache=False).logits
            compare("original_zero_init_noop_exact", no_adapter, zero_adapter,
                    atol=0, rtol=0, exact=True)
        with torch.no_grad():
            wrapper.nca.readout.weight[0, 0] = 0.03
        with torch.inference_mode():
            full = wrapper(input_ids=ids, use_cache=False).logits
            bare = wrapper(input_ids=ids, enabled=False, use_cache=False).logits
            effect = float((full - bare).abs().max())
            require(effect > 0, "Untrained adapter intervention produced no change")
            REPORT["checks"]["manual_untrained_readout_effect"] = {
                "max_abs": effect, "trained": False, "pass": True,
            }
            prefill = wrapper(input_ids=ids[:, :PREFIX], use_cache=True)
            past = prefill.past_key_values
            pieces = []
            for idx in range(PREFIX, LENGTH):
                out = wrapper(input_ids=ids[:, idx:idx+1],
                              past_key_values=past, use_cache=True)
                pieces.append(out.logits[:, -1:, :])
                past = out.past_key_values
                require(len(wrapper.last_hook_trace) == 1,
                        "Incremental cached byte crossed persistent token boundary")
            cached = torch.cat(pieces, dim=1)
            compare("256_UTF8_240_prefill_16_real_incremental",
                    full[:, PREFIX:], cached, atol=atol, rtol=rtol)
            changed = ids.clone()
            changed[:, -1] = (changed[:, -1] + 1) % 256
            later = wrapper(input_ids=changed, use_cache=False).logits
            compare("strict_future_suffix", full[:, :-1], later[:, :-1],
                    atol=0, rtol=0, exact=True)
            a = wrapper(input_ids=ids[:, :36], use_cache=False).logits
            b = wrapper(input_ids=ids[:, 24:60], use_cache=False).logits
            again = wrapper(input_ids=ids[:, :36], use_cache=False).logits
            compare("request_A_B_A", a, again, atol=0, rtol=0, exact=True)
            references = {
                "active_tail": full[:, PREFIX:].cpu().contiguous(),
                "disabled_tail": bare[:, PREFIX:].cpu().contiguous(),
                "request_a": a.cpu().contiguous(),
                "request_b": b.cpu().contiguous(),
            }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        wrapper.save_adapter(adapter)
        save_file(references, str(reference_path))
        manifest = {
            "schema": 1, "model_key": "void1", "repo": REPO, "revision": PIN,
            "checkpoint_file": "model.safetensors", "checkpoint_sha256": WEIGHT,
            "byte_tokenizer_source_sha256": BYTE_SOURCE,
            "original_hook_commit": ORIGINAL_HOOK,
            "protocol_sha256": sha256_file(PROTOCOL),
            "source_pins_sha256": sha256_file(PINS),
            "input_token_ids_sha256": REPORT["input"]["input_ids_sha256"],
            "producer_pid": os.getpid(),
            "adapter_sha256": sha256_file(adapter / "adapter.pt"),
            "adapter_metadata_sha256": sha256_file(adapter / "metadata.json"),
            "reference_sha256": sha256_file(reference_path),
            "model_training_performed": False, "optimizer_steps": 0,
            "train_authorized": False, "heldout_evaluation": False,
            "full_four_model_conformance_pass": False,
            "historical_byte_eos_impact_measured": False,
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8")
        validate_manifest(manifest, current_pid=-1,
                          protocol_sha=sha256_file(PROTOCOL),
                          pins_sha=sha256_file(PINS),
                          token_ids_sha=REPORT["input"]["input_ids_sha256"])
        REPORT["manifest_sha256"] = sha256_file(manifest_path)
        REPORT["status"] = "PASS_PINNED_VOID1_UTF8_PROCESS_PRODUCER"
    else:
        require(manifest_path.is_file() and reference_path.is_file() and
                (adapter / "adapter.pt").is_file() and
                (adapter / "metadata.json").is_file(),
                "Missing original producer handoff; no cold-restart inference")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, current_pid=os.getpid(),
                          protocol_sha=sha256_file(PROTOCOL),
                          pins_sha=sha256_file(PINS),
                          token_ids_sha=REPORT["input"]["input_ids_sha256"])
        for filename, checksum in (
            (adapter / "adapter.pt", manifest["adapter_sha256"]),
            (adapter / "metadata.json", manifest["adapter_metadata_sha256"]),
            (reference_path, manifest["reference_sha256"]),
        ):
            require(sha256_file(filename) == checksum,
                    "Fresh-process reference/adaptor source bytes were altered: " + str(filename))
        wrapper.load_adapter(adapter)
        references = load_file(str(reference_path), device="cpu")
        require(set(references) == {"active_tail", "disabled_tail", "request_a", "request_b"},
                "Source reference logits missing or extra")
        with torch.inference_mode():
            full = wrapper(input_ids=ids, use_cache=False).logits
            bare = wrapper(input_ids=ids, enabled=False, use_cache=False).logits
            a = wrapper(input_ids=ids[:, :36], use_cache=False).logits
            b = wrapper(input_ids=ids[:, 24:60], use_cache=False).logits
            again = wrapper(input_ids=ids[:, :36], use_cache=False).logits
            compare("fresh_process_active_tail_exact", full[:, PREFIX:],
                    references["active_tail"], atol=0, rtol=0, exact=True)
            compare("fresh_process_disabled_tail_exact", bare[:, PREFIX:],
                    references["disabled_tail"], atol=0, rtol=0, exact=True)
            compare("fresh_process_request_A_exact", a,
                    references["request_a"], atol=0, rtol=0, exact=True)
            compare("fresh_process_request_B_exact", b,
                    references["request_b"], atol=0, rtol=0, exact=True)
            compare("fresh_process_A_B_A_exact", again,
                    references["request_a"], atol=0, rtol=0, exact=True)
            require(float((full - bare).abs().max()) > 0,
                    "Freshly loaded adapter inactive")
            prefill = wrapper(input_ids=ids[:, :PREFIX], use_cache=True)
            past = prefill.past_key_values
            pieces = []
            for idx in range(PREFIX, LENGTH):
                out = wrapper(input_ids=ids[:, idx:idx+1],
                              past_key_values=past, use_cache=True)
                pieces.append(out.logits[:, -1:, :])
                past = out.past_key_values
                require(len(wrapper.last_hook_trace) == 1,
                        "Fresh-process per-token cellular field leaked")
            cached = torch.cat(pieces, dim=1)
            compare("fresh_process_full_vs_real_cached_tail",
                    full[:, PREFIX:], cached, atol=atol, rtol=rtol)
        REPORT["producer_pid"] = manifest["producer_pid"]
        REPORT["manifest_sha256"] = sha256_file(manifest_path)
        REPORT["reference_sha256"] = manifest["reference_sha256"]
        REPORT["adapter_sha256"] = manifest["adapter_sha256"]
        REPORT["status"] = "PASS_NEW_INTERPRETER_ORIGINAL_FROZEN_VOID1_ADAPTER_REPLAY"
    wrapper.verify_frozen_base()
    require(all(param.grad is None for param in wrapper.backbone.parameters()),
            "Frozen source backbone accumulated gradients")
    REPORT["limitations"] = [
        "Separate producer/consumer Python processes on ONE hosted runner and cached pinned source",
        "Original frozen Qwen3/ByteTokenizer, 256 UTF-8 units and 240+16 real cached byte steps",
        "Only a manually activated untrained NCA; no optimizer or sealed held-out evaluation",
        "No historical BYTE-EOS-001 impact or EOS generation improvement measurement",
        "No full four-model NCR-001E independent acceptance or training authorization",
    ]
    save()
    print(json.dumps({
        "status": REPORT["status"], "stage": stage,
        "checks": REPORT["checks"], "optimizer_steps": 0,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        stage = ("consumer" if "--stage" in sys.argv and
                 sys.argv.index("--stage") + 1 < len(sys.argv) and
                 sys.argv[sys.argv.index("--stage") + 1] == "consumer" else "producer")
        OUT = OUTPUT_ROOT / (stage + ".json")
        if not REPORT:
            REPORT = {"experiment": "NCR-001E", "stage": stage,
                      "optimizer_steps": 0, "train_authorized": False,
                      "model_training_performed": False}
        REPORT["status"] = "FAIL_SOURCE_BOUND_VOID1_TWO_PROCESS"
        REPORT["exception_type"] = type(exc).__name__
        REPORT["error"] = str(exc)
        REPORT["traceback"] = traceback.format_exc()[-8500:]
        save()
        raise
