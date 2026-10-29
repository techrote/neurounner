#!/usr/bin/env python3
"""NCR-001E: source-pinned 2048 masked-batch KV-cache + cold-process NCA probe.

Separate Sprocket Chat and root Instruct frozen checkpoints only. Two Python
interpreters re-load one exact original frozen model on the same Actions host.
Synthetic ordinary-role inputs; no tool grammar, optimizer, heldouts or training.
Every partial failure retains source, run and model identity in a JSON receipt.
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
import ncr_001_postlease_sprocket as source

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80"
FORMAT_SHA = "2271218d625e1462b2cf218445da975efe17fe32"
VARIANTS = ("chat", "instruct")
LONG = 2048
SHORT = 1536
TAIL = 4
PAD = 5  # Real <|end|> token; only attention_mask distinguishes artificial pads.
TOL = 0.0005
OUTROOT = ROOT / "runtime/ncr001-masked2048-cold"
REPORT: dict = {}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError("NCR001_MASKED2048_REFUSED: " + message)


def sha_file(path: Path) -> str:
    return source.digest(path)


def save(variant: str, phase: str) -> None:
    dest = OUTROOT / variant / (phase + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_process_ru_maxrss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss
    dest.write_text(json.dumps(REPORT, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def geometry(long_ids: torch.Tensor, short_ids: torch.Tensor,
             side: str, *, pad: int = PAD) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both rows end in four REAL tokens, with artificial pads only in prefix.

    Right-pad probe intentionally models a masked gap followed by real tokens:
    this is a strict synthetic attention-mask/cache isolation sentinel, NOT a
    source-documented publisher conversation or standard autoregressive layout.
    """
    require(side in ("left", "right"), "unknown mask orientation")
    require(type(pad) is int and pad == PAD, "synthetic pad token ID drift")
    require(long_ids.ndim == short_ids.ndim == 1, "expected token vectors")
    length, smaller = int(long_ids.numel()), int(short_ids.numel())
    require(length in (64, LONG) and 4 < smaller < length,
            "unregistered context length or missing unequal batch")
    require(length - smaller >= 8 and smaller >= TAIL + 8,
            "mask negative requires real and padded tokens")
    require(long_ids.dtype == short_ids.dtype == torch.long,
            "tokenizer IDs must be integer long")
    require(bool(((long_ids >= 0) & (long_ids < 32000)).all()) and
            bool(((short_ids >= 0) & (short_ids < 32000)).all()),
            "invalid Sprocket tokenizer ID")

    width = length - TAIL
    prefixes, tails, masks = [], [], []
    for seq in (long_ids, short_ids):
        real_prefix, real_tail = seq[:-TAIL], seq[-TAIL:]
        artificial = width - int(real_prefix.numel())
        require(artificial >= 0, "padding width negative")
        pads = torch.full((artificial,), pad, dtype=torch.long)
        zero = torch.zeros((artificial,), dtype=torch.long)
        ones = torch.ones((int(real_prefix.numel()),), dtype=torch.long)
        if side == "left":
            prefixes.append(torch.cat((pads, real_prefix)))
            masks.append(torch.cat((zero, ones)))
        else:
            prefixes.append(torch.cat((real_prefix, pads)))
            masks.append(torch.cat((ones, zero)))
        tails.append(real_tail)
    ids = torch.stack([torch.cat((p, t)) for p, t in zip(prefixes, tails)])
    mask = torch.stack([torch.cat((m, torch.ones(TAIL, dtype=torch.long)))
                        for m in masks])
    positions = torch.clamp(mask.long().cumsum(-1) - 1, min=0)
    require(tuple(ids.shape) == tuple(mask.shape) == tuple(positions.shape)
            == (2, length), "batched geometry drift")
    require((mask[:, -TAIL:] == 1).all().item()
            and int((mask[1] == 0).sum()) == length - smaller
            and int((mask[0] == 0).sum()) == 0,
            "true continuation or attention-mask sentinel lost")
    return ids, mask, positions


def score(key: str, expected: torch.Tensor, actual: torch.Tensor,
          *, atol: float = TOL, rtol: float = TOL, exact: bool = False) -> None:
    require(expected.shape == actual.shape, key + ": mismatched shape")
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    deviation = float((expected - actual).abs().max())
    valid = finite and bool(torch.allclose(expected, actual, atol=atol, rtol=rtol))
    valid = valid and (not exact or deviation == 0)
    REPORT.setdefault("checks", {})[key] = {
        "max_abs": deviation, "atol": atol, "rtol": rtol,
        "exact": exact, "finite": finite, "pass": valid,
        "shape": list(actual.shape),
    }
    require(valid, key + ": causal cache/no-op/cold replay failure")


def masked_cache(wrapper, layout, side: str) -> torch.Tensor:
    x, mask, positions = layout
    with torch.inference_mode():
        full = wrapper(input_ids=x, attention_mask=mask,
                       position_ids=positions, use_cache=False).logits[:, -TAIL:].clone()
        prefix = wrapper(input_ids=x[:, :-TAIL],
                         attention_mask=mask[:, :-TAIL],
                         position_ids=positions[:, :-TAIL], use_cache=True)
        past, pieces = prefix.past_key_values, []
        require(past is not None, "prefill did not return genuine KV cache")
        for index in range(x.shape[1] - TAIL, x.shape[1]):
            next_step = wrapper(input_ids=x[:, index:index + 1],
                                attention_mask=mask[:, :index + 1],
                                position_ids=positions[:, index:index + 1],
                                past_key_values=past, use_cache=True)
            require(next_step.past_key_values is not None,
                    "incremental step lost genuine KV cache")
            past = next_step.past_key_values
            pieces.append(next_step.logits[:, -1:].clone())
            trace = getattr(wrapper, "last_hook_trace", None)
            require(isinstance(trace, list) and len(trace) == 1 and
                    trace[0]["state_shape"][:2] == [2, 1],
                    "cached depth field reused another token or batch")
        incremental = torch.cat(pieces, dim=1)
        score(side + "_2048_full_vs_four_real_cached",
              full, incremental)
        return full.detach().cpu().contiguous()


def synthetic_sequences(tokenizer, serialize, variant: str):
    long_text = ("Inspect the bearing, verify the rail and check the drive. "
                 "Record reversible observations before a shutdown. ") * 180
    short_text = ("Record the motor load, check the gearing and isolate power. "
                  "Keep unrelated requests separate and note the result. ") * 130
    long_ids = source.ids_for(tokenizer, serialize, variant, long_text).flatten()
    short_ids = source.ids_for(tokenizer, serialize, variant, short_text).flatten()
    require(long_ids.numel() >= LONG and short_ids.numel() >= SHORT,
            "exact source-format synthetic lengths unavailable")
    return long_ids[:LONG].contiguous(), short_ids[:SHORT].contiguous()


def run(variant: str, phase: str) -> None:
    require(variant in VARIANTS and phase in ("producer", "consumer"),
            "wrong Sprocket lane or process phase")
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "frozen host must be offline after verified full source prefetch")
    require(os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
            "execution reserved for source-bound draft PR")
    actual_head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(actual_head == os.environ.get("NCR_EXPECTED_HEAD"),
            "exact executed PR head differs from submitted head")
    torch.set_num_threads(4)
    torch.manual_seed(20261029)
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    pins = json.loads((ROOT / "experiments/ncr-001/source-pins.json").read_text())
    spec = source.verify_contract(variant, protocol, pins)
    require(protocol["conformance"]["full_vs_cached_prefill_atol"] ==
            protocol["conformance"]["full_vs_cached_prefill_rtol"] == TOL,
            "preregistered cache tolerance changed")
    formatter = source.load_formatter()
    REPORT.update({
        "experiment": "NCR-001E", "variant": "sprocket_" + variant,
        "phase": phase, "status": "STARTED", "checks": {},
        "source_branch_base": BASE_SHA, "formatter_source": FORMAT_SHA,
        "train_authorized": False, "full_72_conformance_pass": False,
        "optimizer_steps": 0, "heldouts_opened": False,
        "generated_tool_memory_semantics_qualified": False,
        "independent_scientific_admission": False,
        "original_frozen_model_executed": False,
        "scope": "synthetic_2048_masked_batch_true_KV_and_same_host_separate_process",
        "right_pad_tail_is_synthetic_gap_not_publisher_wire": True,
        "runner": {
            "head_sha": actual_head, "pid": os.getpid(),
            "run_id": os.getenv("GITHUB_RUN_ID"), "host": platform.node(),
            "python": platform.python_version(), "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "source_sha256": {
            p: sha_file(ROOT / p) for p in (
                "scripts/ncr_001e_sprocket_masked_cold_2048.py",
                "scripts/ncr_001_postlease_sprocket.py",
                "scripts/ncr_001_sprocket_chat.py",
                "scripts/ncr_001_sprocket_instruct.py",
                "scripts/ncr_001_backbone_lease.py",
                "experiments/ncr-001/protocol.json",
                "experiments/ncr-001/source-pins.json",
            )
        }
    })
    save(variant, phase)

    model, tokenizer, nca, mlp, verified_files, serialize = source.load_actual(
        variant, spec, pins["models"]["sprocket_" + variant], formatter)
    REPORT["verified_frozen_source_files"] = verified_files
    REPORT["original_frozen_model_executed"] = True
    require(not any(p.requires_grad for p in model.parameters()),
            "actual frozen backbone unexpectedly trainable")
    before = source.fingerprint_backbone(model)
    REPORT["frozen_before_sha256"] = before
    long_ids, short_ids = synthetic_sequences(tokenizer, serialize, variant)
    REPORT["synthetic_input_sha256"] = {
        "long": source.tensor_hash(long_ids),
        "short": source.tensor_hash(short_ids),
    }
    parent = OUTROOT / variant
    if phase == "consumer":
        previous = json.loads((parent / "producer.json").read_text())
        require(previous["status"] == "PASS_MASKED2048_PRODUCER" and
                previous["variant"] == REPORT["variant"] and
                previous["runner"]["head_sha"] == actual_head and
                previous["runner"]["run_id"] == REPORT["runner"]["run_id"] and
                previous["runner"]["pid"] != os.getpid() and
                previous["verified_frozen_source_files"] == verified_files and
                previous["synthetic_input_sha256"] == REPORT["synthetic_input_sha256"],
                "cold process reused a PID or mismatched frozen source/input")
        folder = parent / "adapter"
        for name, target in previous["adapter_sha256"].items():
            require(sha_file(folder / name) == target,
                    "original serialized adapter bytes changed")
        nca.load_adapter(folder)
        require(sha_file(parent / "reference.pt") ==
                previous["reference_file_sha256"],
                "producer reference tensors changed before cold restart")
        refs = torch.load(parent / "reference.pt", map_location="cpu",
                          weights_only=True)
    else:
        with torch.no_grad():
            nca.nca.readout.weight[0, 0] = 0.03
        # Zero/disabled parity is already witnessed separately in #155, but
        # this branch insists on a nontrivial, explicitly untrained field.
        refs = {}

    with torch.inference_mode():
        for side in ("left", "right"):
            layout = geometry(long_ids, short_ids, side)
            if phase == "producer":
                tiny = geometry(long_ids[:64], short_ids[:40], side)
                x, mask, positions = tiny
                frozen = nca(input_ids=x, attention_mask=mask,
                             position_ids=positions,
                             use_cache=False, enabled=False).logits
                active = nca(input_ids=x, attention_mask=mask,
                             position_ids=positions, use_cache=False).logits
                require(float((active - frozen).abs().max()) > 0,
                        "manually activated untrained field has no effect")
                # Deliberately corrupted mask must change a real position.
                corrupt = mask.clone()
                corrupt[1, corrupt[1] == 0] = 1
                bad_positions = torch.clamp(corrupt.cumsum(-1) - 1, min=0)
                invalid = nca(input_ids=x, attention_mask=corrupt,
                              position_ids=bad_positions,
                              use_cache=False).logits
                require(float((invalid[1, -TAIL:] -
                               active[1, -TAIL:]).abs().max()) > 0,
                        "invalid unmasked-pad negative sentinel undetected")
                REPORT["checks"][side + "_mask_corruption_sentinel"] = True
            true_tail = masked_cache(nca, layout, side)
            if phase == "producer":
                refs[side] = true_tail
                REPORT["checks"][side + "_raw_tail_sha256"] = source.tensor_hash(true_tail)
            else:
                require(side in refs, "missing original raw per-side cold reference")
                score(side + "_cold_new_python_same_model_2048",
                      refs[side], true_tail, atol=0, rtol=0, exact=True)
            save(variant, phase)
    if phase == "producer":
        folder = parent / "adapter"
        nca.save_adapter(folder)
        REPORT["adapter_sha256"] = {
            name: sha_file(folder / name) for name in ("metadata.json", "adapter.pt")
        }
        torch.save(refs, parent / "reference.pt")
        REPORT["reference_file_sha256"] = sha_file(parent / "reference.pt")
    else:
        REPORT["producer_pid"] = previous["runner"]["pid"]
        REPORT["reference_file_sha256"] = previous["reference_file_sha256"]
    after = source.fingerprint_backbone(model)
    REPORT["frozen_after_sha256"] = after
    require(before == after and
            all(p.grad is None for p in model.parameters()),
            "frozen backbone bytes/gradients changed")
    REPORT["status"] = ("PASS_MASKED2048_PRODUCER" if phase == "producer"
                        else "PASS_MASKED2048_SEPARATE_PYTHON_COLD_CONSUMER")
    REPORT["limits"] = [
        "Same hosted runner/cache, distinct interpreters; not independent physical host",
        "Synthetic ordinary-role IDs; right-pad gaps are not source publisher grammar",
        "Untrained manual NCA readout only; no optimizer, learning or held-outs",
        "No generated tool/memory competence and no four-model scientific admission",
        "Chat and root-Instruct evidence must remain separate variant-lane receipts",
    ]
    save(variant, phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--phase", choices=("producer", "consumer"), required=True)
    args = parser.parse_args()
    try:
        run(args.variant, args.phase)
    except BaseException as error:
        REPORT["status"] = "FAIL_MASKED2048_" + args.phase.upper()
        REPORT["exception_type"] = type(error).__name__
        REPORT["traceback"] = traceback.format_exc()
        raise
    finally:
        save(args.variant, args.phase)
        print(json.dumps({
            "variant": args.variant, "phase": args.phase,
            "status": REPORT.get("status"), "optimizer_steps": 0,
            "train_authorized": False, "run": os.getenv("GITHUB_RUN_ID"),
        }, sort_keys=True), flush=True)
