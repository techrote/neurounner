#!/usr/bin/env python3
"""NCR-001 Sprocket source-matched explicit attention-mask generation witness.

Both pinned frozen original Chat and root Instruct models; no training, no
held-out prompts, no inference about tool/memory semantics. Complement to the
immutable post-lease #155/#156 physical receipts, not their replacement.
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

from ncr_001_postlease_sprocket import (
    ROOT, digest, fingerprint_backbone, load_actual, load_formatter,
    require, verify_contract,
)

OUTPUT = ROOT / "runtime/ncr001-sprocket-explicit-mask"
REPORT = {}


def save(variant: str) -> None:
    path = OUTPUT / variant / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    path.write_text(json.dumps(REPORT, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def check(variant: str, key: str, value) -> None:
    REPORT.setdefault("checks", {})[key] = value
    save(variant)


def mask_authority(tokenizer, serialize, variant: str):
    """Prove content <|end|> (ID 5) is NOT a padding position.

    Pad and end share the numeric ID in the pinned conformance harness.
    Token equality cannot recover which positions were synthetic padding.
    The tokenizer-created attention mask and unpadded source ID sequences can.
    """
    require(tokenizer.pad_token_id == 5, "Pinned Sprocket pad/end marker contract changed")
    tokenizer.padding_side = "left"
    turns = [
        [{"role": "user", "content": "Check the drive."}],
        [
            {"role": "user", "content": "Inspect the instrument rail."},
            {"role": "assistant", "content": "Record the bearing temperature."},
            {"role": "user", "content": "Now verify the gauge and stop."},
        ],
    ]
    formatted = [serialize(t, variant=variant) for t in turns]
    originals = [
        tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"]
        for s in formatted
    ]
    batch = tokenizer(formatted, add_special_tokens=False, padding=True,
                      return_attention_mask=True, return_tensors="pt")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    require(ids.ndim == mask.ndim == 2 and ids.shape == mask.shape,
            "Original source mask and ID tensor shape mismatch")
    require(ids.shape[0] == 2 and ids.shape[1] > originals[0].shape[1],
            "No actual left-padding witness")
    require(set(torch.unique(mask).tolist()) == {0, 1},
            "Mask lacks both observed and padded positions")
    checks = []
    for i, original in enumerate(originals):
        length = original.shape[1]
        pad = ids.shape[1] - length
        require(pad >= 0, "Tokenizer length regression")
        require(torch.equal(ids[i, pad:], original[0]),
                "Source-formatted IDs changed during left padding")
        require(torch.equal(mask[i, pad:], torch.ones(length, dtype=mask.dtype)),
                "Real source tokens are masked (including <|end|>)")
        require(torch.equal(mask[i, :pad], torch.zeros(pad, dtype=mask.dtype)),
                "Synthetic left padding is not fully masked")
        require(bool((original == 5).any()), "Documented real <|end|> marker not present")
        require(bool((ids[i, pad:][mask[i, pad:] == 1] == 5).any()),
                "True end-marker ID was hidden as padding")
        checks.append({"unpad_length": int(length), "synthetic_pad": int(pad),
                       "real_end_marker_id_5_visible": True})
    wrong = (ids != tokenizer.pad_token_id).to(dtype=mask.dtype)
    require(not torch.equal(wrong, mask),
            "Negative control ineffective: naive ID-derived mask unexpectedly correct")
    return originals, ids, mask, checks


def generated(wrapper, ids, mask, *, enabled: bool, cache: bool):
    return wrapper.generate(
        input_ids=ids, attention_mask=mask, enabled=enabled,
        do_sample=False, max_new_tokens=3, min_new_tokens=3,
        use_cache=cache, pad_token_id=5,
    )


def run(variant: str) -> None:
    protocol_path = ROOT / "experiments/ncr-001/protocol.json"
    pin_path = ROOT / "experiments/ncr-001/source-pins.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pins = json.loads(pin_path.read_text(encoding="utf-8"))
    spec = verify_contract(variant, protocol, pins)
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     text=True).strip()
    require(os.getenv("NCR_EXPECTED_HEAD") == actual and len(actual) == 40,
            "Exact PR head/source lease mismatch")
    require(os.getenv("GITHUB_EVENT_NAME") == "pull_request",
            "Only source-matched draft PR execution is authorized")
    require(os.getenv("HF_HUB_OFFLINE") == "1" and
            os.getenv("TRANSFORMERS_OFFLINE") == "1",
            "Original pinned frozen host must execute offline")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.manual_seed(20261028)
    REPORT.update({
        "experiment": "NCR-001", "lane": "sprocket_" + variant,
        "status": "STARTED", "checks": {},
        "train_authorized": False, "optimizer_steps": 0,
        "training_performed": False, "comparative_training_performed": False,
        "heldout_opened": False, "tool_memory_semantics_qualified": False,
        "full_72_conformance_pass": False,
        "runner": {
            "head_sha": actual, "run_id": os.getenv("GITHUB_RUN_ID"),
            "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "pid": os.getpid(), "host": platform.node(),
            "runner_name": os.getenv("RUNNER_NAME"),
            "runner_os": os.getenv("RUNNER_OS"),
            "runner_arch": os.getenv("RUNNER_ARCH"),
            "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
        },
        "source_sha256": {
            name: digest(ROOT / name) for name in (
                "scripts/ncr_001_sprocket_explicit_mask.py",
                "scripts/ncr_001_postlease_sprocket.py",
                "scripts/ncr_001_backbone_lease.py",
                "experiments/ncr-001/protocol.json",
                "experiments/ncr-001/source-pins.json",
            )
        },
        "source_identity": spec,
    })
    save(variant)
    formatter = load_formatter()
    model, tokenizer, nca, mlp, hashes, serialize = load_actual(
        variant, spec, pins["models"]["sprocket_" + variant], formatter)
    REPORT["source_files"] = hashes
    REPORT["original_frozen_model_executed"] = True
    before = fingerprint_backbone(model)
    REPORT["in_memory_frozen_before_sha256"] = before
    save(variant)
    originals, ids, mask, witnessed = mask_authority(tokenizer, serialize, variant)
    check(variant, "tokenizer_source_mask", {
        "pass": True, "batch_shape": list(ids.shape),
        "per_row": witnessed, "pad_equals_end_marker": True,
        "naive_id_mask_refused": True,
        "input_id_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "attention_mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
    })

    with torch.inference_mode():
        for kind, adapter in (("nca", nca), ("mlp", mlp)):
            disabled = {}
            for mode in ("disabled", "enabled_zero", "enabled_manual"):
                if mode == "enabled_manual":
                    with torch.no_grad():
                        if kind == "nca":
                            adapter.nca.readout.weight[0, 0] = 0.03
                        else:
                            adapter.control.down.weight[0, 0] = 0.05
                enabled = mode != "disabled"
                outputs = {}
                for cache in (False, True):
                    batched = generated(adapter, ids, mask, enabled=enabled, cache=cache)
                    new = batched[:, -3:]
                    solo_tokens = []
                    for original in originals:
                        single_mask = torch.ones_like(original)
                        solo = generated(adapter, original, single_mask,
                                         enabled=enabled, cache=cache)
                        solo_tokens.append(solo[0, -3:])
                    singles = torch.stack(solo_tokens)
                    equal = bool(torch.equal(new, singles))
                    check(variant, kind + "_" + mode + "_batch_vs_singles_" +
                          ("cached" if cache else "uncached"),
                          {"pass": equal, "explicit_mask": True,
                           "new_tokens": new.tolist(), "single_tokens": singles.tolist()})
                    require(equal, kind + " " + mode + ": left-padded generation diverged")
                    outputs[cache] = new
                cache_equal = bool(torch.equal(outputs[False], outputs[True]))
                check(variant, kind + "_" + mode + "_cache_on_off",
                      {"pass": cache_equal, "exact_tokens": cache_equal})
                require(cache_equal, kind + " " + mode + ": cache on/off generation diverged")
                if mode == "disabled":
                    disabled = outputs
                if mode == "enabled_zero":
                    equal = all(torch.equal(outputs[c], disabled[c]) for c in outputs)
                    check(variant, kind + "_zero_readout_vs_disabled",
                          {"pass": equal, "exact_tokens": equal})
                    require(equal, kind + ": zero-readout reference generation diverged")
    after = fingerprint_backbone(model)
    REPORT["in_memory_frozen_after_sha256"] = after
    require(after == before, "Frozen original backbone bytes modified")
    require(all(p.grad is None for p in model.parameters()),
            "Unexpected frozen backbone gradient")
    REPORT["status"] = "PASS_EXPLICIT_SOURCE_MASK_GENERATION_WITNESS"
    save(variant)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("chat", "instruct"), required=True)
    v = p.parse_args().variant
    try:
        run(v)
    except BaseException as e:
        REPORT["status"] = "FAIL_EXPLICIT_SOURCE_MASK_GENERATION_WITNESS"
        REPORT["exception_type"] = type(e).__name__
        REPORT["traceback"] = traceback.format_exc()
        raise
    finally:
        save(v)
        print(json.dumps({"lane": "sprocket_" + v,
                          "status": REPORT.get("status"),
                          "source_head": REPORT.get("runner", {}).get("head_sha"),
                          "train_authorized": False, "optimizer_steps": 0},
                         sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
