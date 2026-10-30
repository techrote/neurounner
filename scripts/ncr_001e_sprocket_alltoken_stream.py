#!/usr/bin/env python3
"""NCR-001E: single-stream full 2048 vs EVERY genuine one-token KV-cache forward.

Two independently pinned frozen Sprocket lanes are run separately by Actions.
No comparative adapter learning, no optimizer, no sealed evaluation fixtures,
no tool/memory semantic admission and no changes to historic PR #202 receipts.

This strengthens #202's 2048 masked final-four probe with a *different* B=1
all-position cache witness: all 2,048 genuine positions are compared against
their own single 2,048-position full-forward logits; predeclared uncached
prefix checkpoints and long-context altered-future negative also run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback

import torch
import transformers
import ncr_001_postlease_sprocket as source

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BASE_SHA = "b085d0cec41473b33d844acbf644b57304526779"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
SOURCE_POSTLEASE_BLOB = "66e6b4d97eea5e917339b8e7bfb5d1c191db4a2d"
FORMATTER_HEAD = "2271218d625e1462b2cf218445da975efe17fe32"
FOUR_LANES = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
CONTEXT = 2048
TOL = 0.0005
PREFIXES = (1, 2, 16, 64, 128, 256, 512, 1024, 1536, 2048)
REPORT_PATH_ROOT = ROOT / "runtime/ncr001-sprocket-alltoken2048"
REPORT: dict = {}


def refuse(ok: bool, detail: str) -> None:
    if not ok:
        raise RuntimeError("NCR001_ALLTOKEN_STREAM_REFUSED: " + detail)


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii")
                        + b"\x00" + raw).hexdigest()


def fixed_contract(variant: str, protocol: dict, pins: dict) -> dict:
    refuse(variant in ("chat", "instruct"), "wrong frozen Sprocket variant")
    refuse(protocol.get("train_authorized") is False and
           set(protocol.get("models", {})) ==
           set(pins.get("models", {})) == FOUR_LANES,
           "the four-model protocol or training gate changed")
    require_tol = protocol["conformance"]
    refuse(require_tol["full_vs_cached_prefill_atol"] ==
           require_tol["full_vs_cached_prefill_rtol"] == TOL,
           "the original numeric tolerances changed")
    return source.verify_contract(variant, protocol, pins)


def check_position(value: int, count: int = CONTEXT) -> None:
    refuse(type(value) is int and 0 <= value < count,
           "invalid next genuine one-token cache position")


def compare_logits(label: str, left: torch.Tensor, right: torch.Tensor,
                   *, exact: bool = False, atol: float = TOL,
                   rtol: float = TOL) -> float:
    refuse(tuple(left.shape) == tuple(right.shape) and
           left.numel() > 0, label + " shape or empty logits")
    refuse(atol == rtol == (0 if exact else TOL),
           "preregistered cache/exact equality tolerance drift")
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    refuse(finite, label + " non-finite logits")
    max_abs = float((left - right).abs().max())
    passed = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    if exact:
        passed = passed and max_abs == 0
    refuse(passed, label + f" numeric inequality {max_abs}")
    return max_abs


def save(variant: str) -> None:
    dest = REPORT_PATH_ROOT / variant / "report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_process_ru_maxrss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss
    dest.write_text(json.dumps(REPORT, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def actual_tokens(tokenizer, formatter, variant: str) -> torch.Tensor:
    content = ("Inspect the bearing, verify the rail and check the drive. "
               "Record reversible observations before a shutdown. ") * 180
    tokens = source.ids_for(tokenizer, formatter, variant, content)
    refuse(tuple(tokens.shape[:1]) == (1,) and
           tokens.shape[1] >= CONTEXT and tokens.dtype == torch.long,
           "source-formatted ordinary-role sequence too short")
    x = tokens[:, :CONTEXT].contiguous()
    refuse(int(x.min()) >= 0 and int(x.max()) < 32000,
           "source-formatted token ID out of pinned Sprocket range")
    return x


def incremental_all_tokens(wrapper, x: torch.Tensor,
                           full: torch.Tensor, variant: str) -> dict:
    """Observe every one of 2,048 real cached positions, not just a tail."""
    refuse(tuple(x.shape) == (1, CONTEXT) and
           tuple(full.shape) == (1, CONTEXT, 32000),
           "all-token B=1 source and full-forward shape changed")
    record = {
        "all_real_token_positions": CONTEXT,
        "tokens_compared": 0,
        "numeric_atol": TOL,
        "numeric_rtol": TOL,
        "cached_step_size": 1,
        "all_steps_past_key_values_returned": True,
        "all_steps_depth_field_per_current_token": True,
        "max_abs": 0.0,
        "max_abs_position": None,
        "uncached_prefix_checkpoint_positions": list(PREFIXES),
        "uncached_prefix_max_abs": {},
        "bucket_max_abs": {},
    }
    past = None
    since = time.perf_counter()
    for index in range(CONTEXT):
        check_position(index)
        kwargs = {
            "input_ids": x[:, index:index + 1],
            "attention_mask": torch.ones((1, index + 1), dtype=torch.long),
            "position_ids": torch.tensor([[index]], dtype=torch.long),
            "use_cache": True,
        }
        if past is not None:
            kwargs["past_key_values"] = past
        forward = wrapper(**kwargs)
        require_past = forward.past_key_values
        refuse(require_past is not None,
               f"no authentic returned KV cache at position {index}")
        past = require_past
        trace = getattr(wrapper, "last_hook_trace", None)
        refuse(type(trace) is list and len(trace) == 1 and
               trace[0]["state_shape"][:2] == [1, 1],
               f"depth-state token/request ownership at position {index}")
        current = forward.logits
        refuse(tuple(current.shape) == (1, 1, 32000),
               f"the decoder returned more than the current token at {index}")
        max_abs = compare_logits("alltoken_" + str(index),
                                 full[:, index:index + 1], current)
        if max_abs > record["max_abs"]:
            record["max_abs"] = max_abs
            record["max_abs_position"] = index
        bucket = index // 512
        record["bucket_max_abs"][str(bucket)] = max(
            max_abs, record["bucket_max_abs"].get(str(bucket), 0.0))
        record["tokens_compared"] += 1
        length = index + 1
        if length in PREFIXES:
            prefix = wrapper(input_ids=x[:, :length],
                             attention_mask=torch.ones((1, length),
                                                       dtype=torch.long),
                             position_ids=torch.arange(length).unsqueeze(0),
                             use_cache=False).logits[:, -1:]
            record["uncached_prefix_max_abs"][str(length)] = compare_logits(
                "uncached_prefix_" + str(length),
                full[:, index:index + 1], prefix)
        if length % 128 == 0:
            REPORT["all_token_check"] = record
            REPORT["last_verified_position"] = index
            save(variant)
            print("NCR001_ALL_TOKEN_PROGRESS", variant, length,
                  "max_abs", record["max_abs"], flush=True)
    record["whole_stream_elapsed_s"] = time.perf_counter() - since
    refuse(record["tokens_compared"] == CONTEXT and
           set(record["uncached_prefix_max_abs"]) ==
               {str(x) for x in PREFIXES} and
           len(record["bucket_max_abs"]) == 4,
           "not all tokens, prefix checkpoints or context buckets were checked")
    return record


def future_suffix_sentinel(wrapper, x: torch.Tensor,
                           full: torch.Tensor) -> float:
    """Same-length future edit must not alter earlier 2047 positions."""
    altered = x.clone()
    altered[0, -1] = (altered[0, -1] + 7) % 32000
    refuse(int(altered[0, -1]) != int(x[0, -1]),
           "future suffix change sentinel was not actually applied")
    later = wrapper(input_ids=altered,
                    attention_mask=torch.ones_like(altered),
                    position_ids=torch.arange(CONTEXT).unsqueeze(0),
                    use_cache=False).logits
    return compare_logits("changed_future_2048_earlier_2047_exact",
                          full[:, :-1], later[:, :-1], exact=True,
                          atol=0, rtol=0)


def run(variant: str) -> None:
    refuse(os.environ.get("HF_HUB_OFFLINE") == "1" and
           os.environ.get("TRANSFORMERS_OFFLINE") == "1" and
           os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
           "source-locked physical test requires exact-head offline PR runner")
    expected = os.environ.get("NCR_EXPECTED_HEAD")
    actual = subprocess.check_output(["git", "-C", str(ROOT),
                                      "rev-parse", "HEAD"], text=True).strip()
    refuse(expected and actual == expected, "source/head mismatch")
    refuse(gitblob((ROOT / "experiments/ncr-001/protocol.json").read_bytes())
           == PROTOCOL_BLOB and
           gitblob((ROOT / "experiments/ncr-001/source-pins.json").read_bytes())
           == PINS_BLOB and
           gitblob((ROOT / "scripts/ncr_001_postlease_sprocket.py").read_bytes())
           == SOURCE_POSTLEASE_BLOB,
           "original source, protocol or model pins modified")
    torch.set_num_threads(4)
    torch.manual_seed(20261030)
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    pins = json.loads((ROOT / "experiments/ncr-001/source-pins.json").read_text())
    spec = fixed_contract(variant, protocol, pins)

    REPORT.update({
        "schema": "ncr001-sprocket-alltoken-stream-2048-v1",
        "experiment": "NCR-001E",
        "lane": "sprocket_" + variant,
        "status": "STARTED",
        "frozen_source_base_commit": SOURCE_BASE_SHA,
        "source_formatter_commit": FORMATTER_HEAD,
        "original_source_postlease_git_blob": SOURCE_POSTLEASE_BLOB,
        "original_model_revision": source.MODEL_REVISION,
        "source_protocol_git_blob": PROTOCOL_BLOB,
        "source_pins_git_blob": PINS_BLOB,
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "actual_compute_budget_qualified": False,
        "optimizer_steps": 0,
        "comparative_training": False,
        "heldouts_opened": False,
        "generated_tool_memory_semantics_qualified": False,
        "original_frozen_model_executed": False,
        "all_prefixes_run_without_cache": False,
        "all_token_streaming_cache_observation_only": True,
        "batch_size": 1,
        "context_units": CONTEXT,
        "source_identity": spec,
        "checks": {},
        "runner": {
            "head_sha": actual,
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "job": os.getenv("GITHUB_JOB"),
            "pid": os.getpid(),
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cpu_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
    })
    save(variant)
    formatter = source.load_formatter()
    model, tok, nca, mlp, sources, serializer = source.load_actual(
        variant, spec, pins["models"]["sprocket_" + variant], formatter)
    REPORT["frozen_source_files"] = sources
    REPORT["original_frozen_model_executed"] = True
    REPORT["frozen_backbone_before"] = source.fingerprint_backbone(model)
    refuse(not any(p.requires_grad for p in model.parameters()),
           "original backbone trainable")
    save(variant)
    x = actual_tokens(tok, serializer, variant)
    REPORT["ordinary_role_tokens_sha256"] = source.tensor_hash(x)
    with torch.inference_mode():
        no_op = x[:, :16]
        raw = model(input_ids=no_op, use_cache=False).logits
        zero = nca(input_ids=no_op, use_cache=False).logits
        disabled = nca(input_ids=no_op, enabled=False, use_cache=False).logits
        REPORT["checks"]["native_zero_logit_parity"] = compare_logits(
            "native_zero_logit_parity", raw, zero, exact=True, atol=0, rtol=0)
        REPORT["checks"]["native_disabled_logit_parity"] = compare_logits(
            "native_disabled_logit_parity", raw, disabled,
            exact=True, atol=0, rtol=0)
        with torch.no_grad():
            nca.nca.readout.weight[0, 0] = 0.03
        probe = nca(input_ids=no_op, use_cache=False).logits
        refuse(float((probe - raw).abs().max()) > 0,
               "manually untrained adapter did not affect logits")
        REPORT["checks"]["manual_untrained_effect_max_abs"] = float(
            (probe - raw).abs().max())
        REPORT["checks"]["manual_intervention_learned"] = False
        full = nca(input_ids=x,
                   attention_mask=torch.ones_like(x),
                   position_ids=torch.arange(CONTEXT).unsqueeze(0),
                   use_cache=False).logits
        refuse(tuple(full.shape) == (1, CONTEXT, 32000) and
               bool(torch.isfinite(full).all()),
               "full 2048 frozen NCA source logits missing or nonfinite")
        REPORT["checks"]["full_reference_shape"] = list(full.shape)
        REPORT["checks"]["full_reference_finite"] = True
        save(variant)
        REPORT["all_token_check"] = incremental_all_tokens(
            nca, x, full, variant)
        REPORT["checks"]["altered_future_suffix_max_abs"] = (
            future_suffix_sentinel(nca, x, full))
    REPORT["frozen_backbone_after"] = source.fingerprint_backbone(model)
    refuse(REPORT["frozen_backbone_after"] ==
           REPORT["frozen_backbone_before"] and
           all(p.grad is None for p in model.parameters()),
           "the original frozen backbone changed or acquired gradients")
    REPORT["status"] = "PASS_2048_EVERY_TOKEN_TRUE_KV_STREAM__FINAL_72_BLOCKED"
    REPORT["scope"] = [
        "B=1 source-formatted ordinary role only, 2048 genuine token positions",
        "Each streamed cached one-token logit compared to same-sequence full score",
        "Only ten uncached prefix checkpoints, not every uncached prefix",
        "Previous independent B=2 left/right masks and same-runner cold replay are separately in #202/#203",
        "One deterministic manual untrained NCA readout; no optimizer or held-outs",
        "No generated tool/memory competence, compute matching, four-model #72 PASS or training",
    ]
    save(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("chat", "instruct"))
    args = parser.parse_args()
    try:
        run(args.variant)
    except BaseException as exc:
        REPORT["status"] = "FAIL_2048_EVERY_TOKEN_TRUE_KV_STREAM__FINAL_72_BLOCKED"
        REPORT["failure_type"] = type(exc).__name__
        REPORT["failure_traceback"] = traceback.format_exc()
        raise
    finally:
        save(args.variant)
        print(json.dumps({
            "status": REPORT.get("status"),
            "lane": args.variant,
            "last_verified_position": REPORT.get("last_verified_position"),
            "train_authorized": False,
            "optimizer_steps": 0,
            "run_id": os.getenv("GITHUB_RUN_ID"),
        }, sort_keys=True), flush=True)
