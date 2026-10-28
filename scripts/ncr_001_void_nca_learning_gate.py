#!/usr/bin/env python3
"""NCR-001E void.1 NCA-only diagnostic learning conformance; NOT comparative training.

Exactly one in-memory NCA SGD diagnostic step on an independently fixed synthetic
ByteTokenizer prompt/completion. No corpus, heldout, MLP update, model checkpoint,
NCA/MLP quality comparison, training authorization or final #72 admission.
Original source/serializer/weight identities are enforced by the post-lease loader.
All partial failures are written as original-host receipts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback

import torch

import ncr_001_void_eos_postlease as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr001-void-nca-learning"
SCRIPT = ROOT / "scripts/ncr_001_void_nca_learning_gate.py"
SOURCE_HEAD_PARENT = "1738ac2b6bc6105d57d6a5b4c35f4beaad94ff86"
EXPECTED_GRADIENTS = ("readout.weight", "observe.weight",
                      "perception.weight", "rule.weight",
                      "input_norm.weight")
PROMPT = "Q: π?\nA: "
COMPLETION = "🔧 café 日本語"
LEARNING_RATE = 0.005
REPORT: dict = {}


def require(condition: bool, why: str) -> None:
    if not condition:
        raise RuntimeError("void1 NCA-only diagnostic REFUSED: " + why)


def write_receipt() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    REPORT["elapsed_seconds"] = round(time.monotonic() - REPORT["_started"], 5)
    row = {k: v for k, v in REPORT.items() if k != "_started"}
    (OUT / "receipt.json").write_text(
        json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n", encoding="utf-8")


def finite_loss(loss: torch.Tensor) -> float:
    require(isinstance(loss, torch.Tensor) and loss.ndim == 0
            and loss.requires_grad
            and bool(torch.isfinite(loss.detach()).item())
            and bool((loss.detach() > 0).item()),
            "native supervised loss must be scalar, finite, positive and differentiable")
    return float(loss.detach().item())


def check_gradients(backbone, nca, *, warmed: bool) -> dict:
    require(all(not p.requires_grad and p.grad is None
                for p in backbone.parameters()),
            "source backbone gradient/trainable parameter observed")
    parameters = dict(nca.named_parameters())
    require(set(EXPECTED_GRADIENTS).issubset(parameters),
            "required cellular gradient parameter missing")
    magnitudes = {}
    for name, p in parameters.items():
        grad = p.grad
        if grad is None:
            magnitudes[name] = None
            continue
        require(bool(torch.isfinite(grad).all().item()),
                "nonfinite cellular gradient " + name)
        magnitudes[name] = float(grad.detach().abs().sum().item())
    require(magnitudes["readout.weight"] is not None
            and magnitudes["readout.weight"] > 0,
            "zero/absent readout gradient")
    for name in EXPECTED_GRADIENTS[1:]:
        magnitude = magnitudes[name]
        if warmed:
            require(magnitude is not None and magnitude > 0,
                    "absent post-warmup cellular gradient " + name)
        else:
            require(magnitude is None or magnitude == 0,
                    "upstream cellular gradient bypassed zero readout " + name)
    return magnitudes


def make_sample(serialize, tokenizer) -> tuple:
    """Strict source-verified UTF-8 and *integer* 258 terminal EOS."""
    encoded = serialize(tokenizer, PROMPT, COMPLETION)
    witness = original.validate_causal_eos(encoded, PROMPT, COMPLETION)
    require(witness["real_eos_id"] == 258
            and witness["supervised_after_shift"] == len(COMPLETION.encode("utf-8")) + 1
            and witness["total_units"] <= 96,
            "one synthetic causal-EOS sample is outside fixed budget")
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long)
    mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long)
    require(torch.equal(mask, torch.ones_like(mask))
            and labels[0, -1].item() == 258
            and labels[0, :witness["prompt_bytes"]].eq(-100).all().item(),
            "EOS supervision or assistant-only mask changed")
    return ids, labels, mask, {
        **witness,
        "sample_kind": "synthetic_public_diagnostic_only",
        "heldout": False, "supervised_eos_id": 258,
        "ids_sha256": original.tensor_sha(ids),
        "labels_sha256": original.tensor_sha(labels),
        "mask_sha256": original.tensor_sha(mask),
        "source_text_utf8_bytes": len((PROMPT + COMPLETION).encode("utf-8")),
    }


def one_step_diagnostic(backbone, wrapper, tokenizer, serialize) -> None:
    ids, labels, mask, sample = make_sample(serialize, tokenizer)
    REPORT["synthetic_sample"] = sample
    require(not any(p.requires_grad for p in backbone.parameters()),
            "backbone must already be frozen")
    wrapper.train()
    require(not backbone.training, "frozen Qwen3 erroneously entered train mode")
    initial_frozen_sha = original.frozen_sha(backbone)
    REPORT["backbone_before_sha256"] = initial_frozen_sha
    write_receipt()
    with torch.no_grad():
        raw = wrapper(input_ids=ids, attention_mask=mask,
                      use_cache=False, enabled=False)
        zero = wrapper(input_ids=ids, attention_mask=mask,
                       use_cache=False, enabled=True)
    require(torch.equal(raw.logits, zero.logits),
            "zero-adapter source parity failed")
    REPORT["initial_source_logits_sha256"] = original.tensor_sha(raw.logits)
    REPORT["initial_zero_readout_parity_exact"] = True

    # A single diagnostic SGD step is a #72 gradient/freeze probe, not a
    # comparative optimizer schedule or an admitted NCR research train run.
    optimizer = torch.optim.SGD(wrapper.nca.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    first = wrapper(input_ids=ids, labels=labels, attention_mask=mask,
                    use_cache=False).loss
    initial_loss = finite_loss(first)
    first.backward()
    before_gradients = check_gradients(backbone, wrapper.nca, warmed=False)
    readout_before = original.tensor_sha(wrapper.nca.readout.weight)
    optimizer.step()
    readout_after = original.tensor_sha(wrapper.nca.readout.weight)
    require(readout_before != readout_after,
            "the only diagnostic SGD step did not change cellular readout")
    REPORT["diagnostic_optimizer_steps"] = 1
    REPORT["warmup"] = {
        "native_causal_eos_loss": initial_loss,
        "gradient_l1": before_gradients,
        "readout_before_sha256": readout_before,
        "readout_after_sha256": readout_after,
        "optimizer": "NCA-only SGD", "learning_rate": LEARNING_RATE,
    }
    write_receipt()
    del first
    optimizer.zero_grad(set_to_none=True)

    tap_fields = []
    original_step = wrapper.nca.step

    def trace_step(hidden, state, *, tap_index, **kwargs):
        modified, next_state = original_step(
            hidden, state, tap_index=tap_index, **kwargs)
        require(bool(torch.isfinite(next_state.field.detach()).all().item()),
                "nonfinite persistent-depth field")
        tap_fields.append({
            "tap_index": tap_index, "next_tap": next_state.next_tap,
            "shape": list(next_state.field.shape),
            "field_sha256": original.tensor_sha(next_state.field),
        })
        return modified, next_state

    wrapper.nca.step = trace_step
    try:
        second = wrapper(input_ids=ids, labels=labels,
                         attention_mask=mask, use_cache=False).loss
    finally:
        wrapper.nca.step = original_step
    second_loss = finite_loss(second)
    second.backward()
    after_gradients = check_gradients(backbone, wrapper.nca, warmed=True)
    require([x["tap_index"] for x in tap_fields] == [0, 1, 2]
            and [x["next_tap"] for x in tap_fields] == [1, 2, 3]
            and all(x["shape"] == [1, ids.shape[1], 4, 8, 8] for x in tap_fields)
            and len({x["field_sha256"] for x in tap_fields}) == 3,
            "real Qwen3 hook failed ordered three-field persistence")
    REPORT["post_warmup"] = {
        "native_causal_eos_loss": second_loss,
        "gradient_l1": after_gradients,
        "ordered_field_trace": tap_fields,
    }
    write_receipt()
    del second
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        after_disabled = wrapper(input_ids=ids, attention_mask=mask,
                                 use_cache=False, enabled=False).logits
    require(torch.equal(raw.logits, after_disabled),
            "original disabled Qwen3 logits drifted after NCA-only step")
    final_frozen_sha = original.frozen_sha(backbone)
    REPORT["backbone_after_sha256"] = final_frozen_sha
    require(initial_frozen_sha == final_frozen_sha
            and all(p.grad is None for p in backbone.parameters()),
            "frozen model changed or accumulated gradients")
    require(not backbone.model._forward_hooks
            and not backbone.model._forward_pre_hooks,
            "Qwen3 hook leak after diagnostic backward")
    REPORT["checks"] = {
        "native_shifted_byte_eos_loss": True,
        "initial_zero_adapter_parity": True,
        "readout_gradient_first_backward": True,
        "post_step_upstream_gradients": True,
        "three_distinct_depth_states": True,
        "backbone_unchanged_after_one_step": True,
        "disabled_source_logits_unchanged": True,
        "no_cross_model_comparison": True,
    }


def main() -> None:
    REPORT.update({
        "_started": time.monotonic(),
        "status": "STARTED",
        "experiment": "NCR-001E",
        "lane": "void1",
        "source_parent": SOURCE_HEAD_PARENT,
        "train_authorized": False,
        "comparative_training_performed": False,
        "model_training_experiment": False,
        "heldout_opened": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "diagnostic_optimizer_steps": 0,
        "runner": {
            "expected_head": os.environ.get("NCR_EXPECTED_HEAD"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "name": os.environ.get("RUNNER_NAME"),
            "platform": platform.platform(), "python": platform.python_version(),
            "pid": os.getpid(), "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "source_sha256": {"learning_runner": original.file_sha(SCRIPT)},
    })
    write_receipt()
    try:
        require(os.getenv("GITHUB_EVENT_NAME") == "pull_request",
                "actual host must run from source-matched draft PR")
        require(os.getenv("HF_HUB_OFFLINE") == "1"
                and os.getenv("TRANSFORMERS_OFFLINE") == "1",
                "actual host must run with frozen offline model source")
        require(os.getenv("NCR_EXPECTED_HEAD") == original.git_head(ROOT),
                "source head mismatch")
        # Deliberately reuse the already qualified post-lease loader. It
        # validates protocol/pins, full weights, actual custom tokenizer code,
        # original owner lease, serializer commit and exact HF runtime.
        original.REPORT.clear()
        original.REPORT.update({
            "_started": time.monotonic(), "phase": "diagnostic",
            "status": "LOADING", "checks": {},
        })
        torch_mod, _, backbone, tokenizer, nca, _mlp, serialize = original.run()
        require(torch_mod is torch
                and nca.nca.trainable_parameters() == 329944,
                "post-lease actual NCA source or parameter identity changed")
        REPORT["source"] = original.REPORT["source"]
        REPORT["packages"] = original.REPORT["packages"]
        REPORT["actual_source_files"] = original.REPORT["actual_source_files"]
        REPORT["original_frozen_model_loaded"] = True
        write_receipt()
        one_step_diagnostic(backbone, nca, tokenizer, serialize)
        REPORT["status"] = "PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED"
    except BaseException:
        REPORT["status"] = "FAIL_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_BLOCKED"
        REPORT["traceback"] = traceback.format_exc()
        raise
    finally:
        write_receipt()
        print(json.dumps({
            "lane": "void1", "status": REPORT["status"],
            "diagnostic_optimizer_steps": REPORT["diagnostic_optimizer_steps"],
            "train_authorized": False, "full_72_conformance_pass": False,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
