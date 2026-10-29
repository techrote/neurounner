#!/usr/bin/env python3
"""NCR-001E: one-step *diagnostic* NCA learning gates on each pinned Sprocket host.

This is not the preregistered NCA/MLP comparison or an evaluation run:
one synthetic assistant-only teacher-forcing sample, one NCA-only SGD
conformance update, and a second backward to observe post-warmup gradients.
Never open train/held-out corpora, persist an adapted checkpoint, or change
the four-model protocol's train_authorized=false admission state.
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
import traceback

import torch
import transformers

from ncr_001_postlease_sprocket import (
    ROOT, digest, fingerprint_backbone, load_actual, load_formatter,
    require, verify_contract,
)

OUT = ROOT / "runtime/ncr001-sprocket-learning-gate"
UPSTREAM = ("observe.weight", "perception.weight", "rule.weight")
READOUT = "readout.weight"
REPORT: dict = {}


def save(variant: str) -> None:
    path = OUT / variant / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    path.write_text(json.dumps(REPORT, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def supervised_sample(tokenizer, serialize, variant: str):
    """One explicitly non-heldout diagnostic utterance; preserve EOS as data."""
    prefix = serialize(
        [{"role": "user", "content": "Diagnostic: report the test relay state."}],
        variant=variant,
    )
    completion = "The test relay is idle.<|end|>"
    prompt = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target = tokenizer(completion, add_special_tokens=False)["input_ids"]
    require(bool(prompt) and len(target) > 1 and target[-1] == 5,
            "The synthetic assistant answer lacks the genuine end token")
    require(5 not in target[:-1], "Unexpected end marker within answer")
    require(0 < len(prompt) + len(target) <= 96,
            "Diagnostic input exceeds bounded non-heldout budget")
    ids = torch.tensor([prompt + target], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prompt) + target], dtype=torch.long)
    attention = torch.ones_like(ids)
    require(int((labels != -100).sum()) == len(target)
            and labels[0, -1].item() == 5,
            "The assistant-only EOS supervision is not exact")
    return ids, labels, attention, {
        "prefix_units": len(prompt), "supervised_units": len(target),
        "sequence_units": ids.shape[1],
        "assistant_eos_id": int(target[-1]),
        "ids_sha256": tensor_sha(ids),
        "labels_sha256": tensor_sha(labels),
        "attention_mask_sha256": tensor_sha(attention),
        "synthetic_fixture": True, "heldout": False,
    }


def finite_supervised_loss(loss) -> float:
    require(isinstance(loss, torch.Tensor) and loss.ndim == 0,
            "Model did not produce a scalar supervised loss")
    require(bool(torch.isfinite(loss.detach()).item()) and
            bool(loss.detach().item() > 0) and loss.requires_grad,
            "Supervised loss must be finite, positive and differentiable")
    return float(loss.detach().item())


def verify_gradient_gate(base, adapter, *, warmed: bool) -> dict:
    """Reject absent, nonfinite or identically zero required gradients."""
    require(all(p.grad is None and not p.requires_grad
                for p in base.parameters()),
            "Frozen source backbone obtained gradients or trainable parameters")
    params = dict(adapter.named_parameters())
    for name in (READOUT, *UPSTREAM):
        require(name in params, "Missing preregistered NCA parameter " + name)
    output = {}
    for name, p in params.items():
        if p.grad is None:
            output[name] = None
            continue
        require(bool(torch.isfinite(p.grad).all().item()),
                "Nonfinite NCA gradient: " + name)
        output[name] = float(p.grad.detach().abs().sum().item())
    require(output[READOUT] is not None and output[READOUT] > 0,
            "No readout learning signal")
    for name in UPSTREAM:
        magnitude = output[name]
        if warmed:
            require(magnitude is not None and magnitude > 0,
                    "Post-warmup NCA gradient absent/zero: " + name)
        else:
            require(magnitude is None or magnitude == 0,
                    "Zero-readout warmup unexpectedly touched upstream: " + name)
    return output


def learning_probe(variant: str, model, tokenizer, wrapper, serialize):
    ids, labels, attention, fixture = supervised_sample(
        tokenizer, serialize, variant)
    REPORT["synthetic_sample"] = fixture
    save(variant)
    model.eval()
    wrapper.train()
    require(not any(p.requires_grad for p in model.parameters()),
            "Pinned backbone must be frozen before diagnostics")
    before_hash = fingerprint_backbone(model)
    REPORT["backbone_before_sha256"] = before_hash
    with torch.no_grad():
        baseline = wrapper(input_ids=ids, attention_mask=attention,
                           use_cache=False, enabled=False).logits.detach().clone()
        zero = wrapper(input_ids=ids, attention_mask=attention,
                       use_cache=False, enabled=True).logits.detach()
    require(torch.equal(baseline, zero),
            "Original zero-readout source parity failed")
    REPORT["initial_zero_readout_source_parity_exact"] = True
    save(variant)

    # This optimizer is *not* part of the training/evaluation protocol.
    # One local conformance step only; no checkpoint, matched MLP, or dataset.
    optim = torch.optim.SGD(wrapper.nca.parameters(), lr=0.05)
    optim.zero_grad(set_to_none=True)
    first = wrapper(input_ids=ids, attention_mask=attention,
                    labels=labels, use_cache=False, enabled=True).loss
    first_loss = finite_supervised_loss(first)
    first.backward()
    stage0 = verify_gradient_gate(model, wrapper.nca, warmed=False)
    readout_before = tensor_sha(wrapper.nca.readout.weight)
    optim.step()
    readout_after = tensor_sha(wrapper.nca.readout.weight)
    require(readout_after != readout_before,
            "NCA-only diagnostic optimizer update did not modify readout")
    REPORT["diagnostic_optimizer_steps"] = 1
    REPORT["warmup"] = {
        "loss": first_loss, "gradient_abs_sums": stage0,
        "readout_before_sha256": readout_before,
        "readout_after_sha256": readout_after,
    }
    save(variant)
    del first
    optim.zero_grad(set_to_none=True)

    # The three field snapshots are from one actual transformer forward,
    # not a synthetic standalone NCA step.
    field_trace = []
    original_step = wrapper.nca.step

    def traced_step(hidden, state, *, tap_index, **kwargs):
        result, next_state = original_step(
            hidden, state, tap_index=tap_index, **kwargs)
        field = next_state.field
        require(bool(torch.isfinite(field.detach()).all().item()),
                "Nonfinite persistent-depth field")
        field_trace.append({
            "tap_index": tap_index, "next_tap": next_state.next_tap,
            "shape": list(field.shape), "field_sha256": tensor_sha(field),
        })
        return result, next_state

    wrapper.nca.step = traced_step
    try:
        second = wrapper(input_ids=ids, attention_mask=attention,
                         labels=labels, use_cache=False, enabled=True).loss
    finally:
        wrapper.nca.step = original_step
    second_loss = finite_supervised_loss(second)
    second.backward()
    stage1 = verify_gradient_gate(model, wrapper.nca, warmed=True)
    require([x["tap_index"] for x in field_trace] == [0, 1, 2]
            and [x["next_tap"] for x in field_trace] == [1, 2, 3],
            "Real frozen host did not traverse ordered persistent-depth taps")
    require(len({x["field_sha256"] for x in field_trace}) == 3,
            "Depth field failed to evolve across real transformer taps")
    REPORT["post_warmup"] = {
        "loss": second_loss, "gradient_abs_sums": stage1,
        "ordered_field_trace": field_trace,
    }
    save(variant)
    del second
    optim.zero_grad(set_to_none=True)
    with torch.no_grad():
        after_disabled = wrapper(input_ids=ids, attention_mask=attention,
                                 use_cache=False, enabled=False).logits
    require(torch.equal(baseline, after_disabled),
            "Post-diagnostic frozen baseline logits changed")
    after_hash = fingerprint_backbone(model)
    REPORT["backbone_after_sha256"] = after_hash
    require(before_hash == after_hash,
            "Pinned original frozen model parameters changed after SGD")
    require(all(p.grad is None for p in model.parameters()),
            "Pinned frozen model accumulated gradients")
    REPORT["checks"] = {
        "assistant_only_finite_loss": True,
        "readout_first_step_gradient": True,
        "post_warmup_upstream_gradients": True,
        "one_diagnostic_update_changes_only_nca": True,
        "source_baseline_unchanged": True,
        "three_distinct_ordered_depth_states": True,
    }


def run(variant: str) -> None:
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json")
                          .read_text(encoding="utf-8"))
    pins = json.loads((ROOT / "experiments/ncr-001/source-pins.json")
                      .read_text(encoding="utf-8"))
    spec = verify_contract(variant, protocol, pins)
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(os.environ.get("NCR_EXPECTED_HEAD") == head
            and len(head) == 40, "Actual PR head mismatch")
    require(os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
            "Only draft PR source-matched execution is permitted")
    require(os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Original frozen source must be verified and offline")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.manual_seed(20261028)
    REPORT.update({
        "experiment": "NCR-001", "lane": "sprocket_" + variant,
        "status": "STARTED",
        "train_authorized": False, "comparative_training_performed": False,
        "preregistered_evaluation_performed": False, "heldout_opened": False,
        "model_training_experiment": False,
        "diagnostic_optimizer_steps": 0,
        "full_72_conformance_pass": False,
        "tool_memory_semantics_qualified": False,
        "independent_review_complete": False,
        "source_identity": spec,
        "runner": {
            "head_sha": head, "run_id": os.environ.get("GITHUB_RUN_ID"),
            "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "host": platform.node(), "pid": os.getpid(),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
        },
        "source_sha256": {
            name: digest(ROOT / name) for name in (
                "scripts/ncr_001_sprocket_learning_gate.py",
                "scripts/ncr_001_postlease_sprocket.py",
                "scripts/ncr_001_core.py",
                "scripts/ncr_001_sprocket_chat.py",
                "scripts/ncr_001_sprocket_instruct.py",
                "scripts/ncr_001_backbone_lease.py",
                "experiments/ncr-001/protocol.json",
                "experiments/ncr-001/source-pins.json",
            )
        },
    })
    save(variant)
    formatter = load_formatter()
    model, tok, wrapper, _, files, serialize = load_actual(
        variant, spec, pins["models"]["sprocket_" + variant], formatter)
    REPORT["verified_original_source_files"] = files
    REPORT["original_frozen_model_executed"] = True
    REPORT["formatter"] = {
        "commit": "2271218d625e1462b2cf218445da975efe17fe32",
        "sha256": digest(ROOT / "format-authority/scripts/ncr_001e_sprocket_format.py"),
        "scope": "ordinary user/assistant only; tool/memory semantics not inferred",
    }
    save(variant)
    learning_probe(variant, model, tok, wrapper, serialize)
    REPORT["status"] = "PASS_DIAGNOSTIC_NCA_LEARNING_GATES__FINAL_72_BLOCKED"
    save(variant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("chat", "instruct"))
    variant = parser.parse_args().variant
    try:
        run(variant)
    except BaseException as exc:
        REPORT["status"] = "FAIL_DIAGNOSTIC_NCA_LEARNING_GATES"
        REPORT["exception_type"] = type(exc).__name__
        REPORT["failure_traceback"] = traceback.format_exc()
        raise
    finally:
        save(variant)
        print(json.dumps({
            "lane": "sprocket_" + variant,
            "status": REPORT.get("status"),
            "diagnostic_optimizer_steps": REPORT.get("diagnostic_optimizer_steps", 0),
            "train_authorized": False, "full_72_conformance_pass": False,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
