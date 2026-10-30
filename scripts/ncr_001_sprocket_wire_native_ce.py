#!/usr/bin/env python3
"""NCR-001E: exact frozen ROOT Sprocket Instruct masked causal-CE witness.

Actual pinned model on a public synthetic diagnostic only. Source / model
read-only; no backward, optimizer, held-outs, training or behavioral admission.
The source-only shifted-label witness is from immutable #184/#185; this script
does not rewrite its original evidence or claim a Chat-equivalent experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import traceback

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_sprocket_wire_fullmask as fullmask
import ncr_001_sprocket_wire_shifted_loss as shifted
from ncr_001_postlease_sprocket import (
    digest, fingerprint_backbone, load_actual, load_formatter, verify_contract,
)

OUT = ROOT / "runtime/ncr001-root-instruct-native-ce"
ORIGINAL_PHYSICAL_HEAD = "aadbfe823c9778066acac0a0869e0c559a533f05"
ORIGINAL_SHIFTED_HEAD = "adfc3cb3ba2b2b6b54b0b10e3aa1ae19d4ff10ba"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
EXPECTED_ROOT_WEIGHT = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
EXPECTED_CHAT_WEIGHT = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
STATUS = "PASS_FROZEN_ROOT_INSTRUCT_SYNTHETIC_MASKED_CE_ONLY__72_BLOCKED"


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("NCR001_NATIVE_CE_REFUSED: " + reason)


def blob(path: str) -> str:
    return subprocess.check_output(["git", "hash-object", path],
                                   cwd=ROOT, text=True).strip()


def record(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (OUT / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def independent_shifted_ce(logits: torch.Tensor,
                           labels: torch.Tensor) -> tuple[float, int]:
    """Second consumer: compute CE over ONLY actually supervised next tokens.

    Never infer padding or end-of-turn from token ID (both can be 5).
    Labels at position j are predicted by logits at j-1.
    """
    refuse(logits.ndim == 3 and labels.ndim == 2
           and logits.shape[:2] == labels.shape
           and logits.shape[0] == 2, "masked logits/label shape changed")
    targets = labels[:, 1:].contiguous()
    mask = targets != -100
    count = int(mask.sum().item())
    refuse(count == 45 and not bool(mask[0].any().item()),
           "unexpected host/pad supervision or original 45 targets drifted")
    next_logits = logits[:, :-1, :].float()
    chosen = next_logits[mask]
    gold = targets[mask]
    refuse(bool((gold >= 0).all().item()) and
           bool((gold < logits.shape[-1]).all().item()),
           "invalid supervised token ID")
    per_target = torch.logsumexp(chosen, dim=-1) - chosen.gather(
        1, gold.unsqueeze(1)).squeeze(1)
    refuse(bool(torch.isfinite(per_target).all().item()),
           "nonfinite independent target CE")
    return float(per_target.mean().item()), count


def run(report: dict) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()
    refuse(os.getenv("GITHUB_EVENT_NAME") == "pull_request" and
           os.getenv("NCR_EXPECTED_HEAD") == head,
           "must run at exact unmerged PR source head")
    refuse(blob("experiments/ncr-001/protocol.json") == PROTOCOL_BLOB
           and blob("experiments/ncr-001/source-pins.json") == PINS_BLOB,
           "preregistered source/protocol Git blobs changed")
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    pins = json.loads((ROOT / "experiments/ncr-001/source-pins.json").read_text())
    keys = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
    refuse(protocol["train_authorized"] is False and
           set(protocol["models"]) == set(pins["models"]) == keys,
           "four independent frozen models or training gate changed")
    spec = verify_contract("instruct", protocol, pins)
    refuse(spec["weight_sha256"] == EXPECTED_ROOT_WEIGHT and
           pins["models"]["sprocket_chat"]
               ["weight_declared_lfs_sha256_not_downloaded"] == EXPECTED_CHAT_WEIGHT
           and pins["models"]["sprocket_instruct"]["hf_weight_file"]
               == "model.safetensors"
           and pins["models"]["sprocket_chat"]["hf_weight_file"]
               == "chat/model.safetensors",
           "root Instruct source identity conflated with Chat")
    _, archived_original = fullmask.original_receipt()
    refuse(archived_original["runner"]["head_sha"] == ORIGINAL_PHYSICAL_HEAD
           and archived_original["train_authorized"] is False
           and archived_original["full_72_conformance_pass"] is False,
           "original physical host receipt no longer authentic")
    source_hash = hashlib.sha256(fullmask.HOST_SOURCE.read_bytes()).hexdigest()
    refuse(source_hash ==
           archived_original["source_sha256"]
               ["scripts/ncr_001_postlease_sprocket_wire.py"],
           "original synthetic source has drifted")
    turns = fullmask.extract_synthetic_turns(fullmask.HOST_SOURCE)
    report["original_physical_source_sha256"] = source_hash
    report["original_shifted_source_head"] = ORIGINAL_SHIFTED_HEAD
    report["original_physical_head"] = ORIGINAL_PHYSICAL_HEAD
    report["original_physical_run"] = 40718706396
    report["source"] = {"protocol_git_blob": PROTOCOL_BLOB,
                        "source_pins_git_blob": PINS_BLOB,
                        "root_weight_sha256": EXPECTED_ROOT_WEIGHT,
                        "excluded_chat_weight_sha256": EXPECTED_CHAT_WEIGHT}
    record(report)

    # Only download exact source weights and metadata; verify SHA-256 before
    # entering offline model construction. load_actual() rehashes every file.
    files = ["README.md", "config.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "model.safetensors"]
    for name in files:
        hf_hub_download(spec["model_id"], name, revision=spec["revision"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model, tokenizer, nca, mlp, verified_files, _ = load_actual(
        "instruct", spec, pins["models"]["sprocket_instruct"], load_formatter()
    )
    report["original_frozen_model_executed_this_run"] = True
    report["verified_source_files"] = verified_files
    record(report)
    torch.set_num_threads(4)
    start_base = fingerprint_backbone(model)
    report["frozen_backbone_before_sha256"] = start_base
    long_ids, long_mask = fullmask.independently_rebuild(turns, tokenizer)
    short_ids, short_mask = fullmask.independently_rebuild(turns[:3], tokenizer)
    refuse(len(long_ids) == 123 and sum(long_mask) == 45
           and len(short_ids) == 26, "original publisher-wire token counts drifted")
    orig = archived_original["wire"]
    tok_hash = hashlib.sha256(
        struct.pack("<" + "q" * len(long_ids), *long_ids)
    ).hexdigest()
    mask_hash = hashlib.sha256(bytes(map(int, long_mask))).hexdigest()
    refuse(tok_hash == orig["ids_sha256"]
           and mask_hash == orig["assistant_mask_sha256"],
           "real tokenizer/label vector differs from original frozen host")
    matrix = shifted.construct(long_ids, long_mask, short_ids, short_mask)
    refuse(shifted.int64_sha(matrix["attention_mask"]) ==
           orig["attention_mask_sha256"]
           and shifted.int64_sha(matrix["shifted_targets"]) ==
           "f8453abe8ee1780ecd9f536a90f6eb1cc114af7e704f52110889165ffecc85a4",
           "original causal-shifted/pad-source hashes differ")
    input_ids = torch.tensor(matrix["input_ids"], dtype=torch.long)
    attention = torch.tensor(matrix["attention_mask"], dtype=torch.long)
    labels = torch.tensor(matrix["labels"], dtype=torch.long)
    refuse(int((labels[:, 1:] != -100).sum().item()) == 45
           and bool((labels[0] == -100).all().item())
           and int((attention[0] == 0).sum().item()) == 97
           and bool(((input_ids == 5) & (attention == 1)).any().item()),
           "real END/pad split or causal targets invalid")
    report["transport"] = {
        "batch_shape": list(input_ids.shape), "synthetic_left_pad_units": 97,
        "supervised_after_causal_shift": 45,
        "original_attention_sha256": orig["attention_mask_sha256"],
        "original_shifted_targets_sha256": shifted.int64_sha(matrix["shifted_targets"]),
    }
    record(report)
    with torch.inference_mode():
        native = model(input_ids=input_ids, attention_mask=attention,
                       labels=labels, use_cache=False)
        refuse(native.loss is not None and bool(torch.isfinite(native.loss).item()),
               "native frozen masked causal loss nonfinite")
        manually, count = independent_shifted_ce(native.logits, labels)
        expected = float(native.loss.item())
        # Different CE reductions/order may incur float32 summation drift;
        # never relax any #72 causal/cache tolerance (these compare losses only).
        closeness = abs(manually - expected)
        refuse(math.isfinite(manually) and closeness <= 1e-5,
               "native vs independent causal CE disagrees")
        report["native_ce"] = {"native": expected,
                               "manual_logsumexp": manually,
                               "abs_difference": closeness,
                               "supervised_target_count": count,
                               "finite": True}
        record(report)
        for name, wrapper in (("nca", nca), ("mlp", mlp)):
            zero = wrapper(input_ids=input_ids, attention_mask=attention,
                           labels=labels, use_cache=False)
            disabled = wrapper(input_ids=input_ids, attention_mask=attention,
                               labels=labels, use_cache=False, enabled=False)
            zero_delta = float((zero.logits - native.logits).abs().max().item())
            disabled_delta = float((disabled.logits - native.logits).abs().max().item())
            zero_loss = float(zero.loss.item())
            disabled_loss = float(disabled.loss.item())
            refuse(zero_delta == disabled_delta == 0.0
                   and zero_loss == disabled_loss == expected,
                   name + ": zero/disabled frozen masked CE no-op parity failed")
            report["controls"][name] = {
                "adapter_untrained": True, "zero_logits_exact": True,
                "disabled_logits_exact": True, "native_ce_exact": True,
                "loss": zero_loss, "trainable_parameters": sum(
                    p.numel() for p in wrapper.adapter_parameters()
                ),
            }
            record(report)
    end_base = fingerprint_backbone(model)
    refuse(start_base == end_base and
           all(p.grad is None for p in model.parameters()),
           "frozen original backbone unexpectedly changed")
    report["frozen_backbone_after_sha256"] = end_base
    report["status"] = STATUS
    record(report)
    return report


def main() -> None:
    report = {
        "status": "STARTED_NO_ADMISSION", "experiment": "NCR-001E",
        "lane": "sprocket_instruct",
        "original_frozen_model_executed_this_run": False,
        "new_model_training": False, "optimizer_steps": 0,
        "comparative_training": False, "heldouts_opened": False,
        "full_72_conformance_pass": False, "train_authorized": False,
        "independent_scientific_review": False,
        "model_generated_tool_memory_semantics_qualified": False,
        "original_incomplete_tool_call_preserved": True,
        "controls": {},
        "runner": {"name": os.getenv("RUNNER_NAME"),
                   "run_id": os.getenv("GITHUB_RUN_ID"),
                   "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
                   "host": platform.node(), "pid": os.getpid(),
                   "os": os.getenv("RUNNER_OS"),
                   "python": platform.python_version(),
                   "torch": torch.__version__,
                   "transformers": transformers.__version__},
    }
    record(report)
    try:
        result = run(report)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        report["status"] = "FAIL_FROZEN_ROOT_NATIVE_CE__NO_ADMISSION"
        report["failure"] = {"kind": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        record(report)
        raise


if __name__ == "__main__":
    main()
