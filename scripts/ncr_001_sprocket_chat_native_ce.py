#!/usr/bin/env python3
"""NCR-001E: frozen Sprocket CHAT ordinary-role native causal CE (NO TRAIN).

A source/weight-pinned *Chat* checkpoint only. Two entirely synthetic,
assistant-only supervised, ordinary-role examples; no Instruct publisher
tool/memory grammar, no heldouts, no backward, no optimizer, no admission.
The model's ordinary-role serializer belongs to immutable #95. The NCA/MLP
wrappers and frozen checkpoint loader belong to original #155/#156.
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
import transformers
from huggingface_hub import hf_hub_download

from ncr_001_postlease_sprocket import (
    ROOT, SOURCE_FILES, digest, fingerprint_backbone, load_actual,
    load_formatter, verify_contract,
)

OUT = ROOT / "runtime/ncr001-chat-native-ce"
PARENT_HEAD = "5794964f2ef8f9640669b77d7da74cfcec4d9ddb"
FORMAT_HEAD = "2271218d625e1462b2cf218445da975efe17fe32"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
CHAT_SHA = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
INSTRUCT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
STATUS = "PASS_FROZEN_CHAT_ORDINARY_SYNTHETIC_ASSISTANT_CE_ONLY__72_BLOCKED"

# Not preregistered evaluation data and no similarity to source SFT samples
# asserted. The Chat publisher documents ONLY alternating user/assistant roles.
SYNTHETIC = (
    (
        ({"role": "user", "content": "Inspect the fictional relay reading."},),
        "The fictional relay reads 42 C.",
    ),
    (
        (
            {"role": "user", "content": "Inspect the fictional motor."},
            {"role": "assistant", "content": "I will use only the supplied fictional readings."},
            {"role": "user", "content": "Report the synthetic temperature in C."},
        ),
        "The fictional motor reads 23 C.",
    ),
)


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("NCR001_CHAT_NATIVE_CE_REFUSED: " + reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def int64_sha(rows: list[list[int]]) -> str:
    flat = [v for row in rows for v in row]
    return sha(struct.pack("<" + "q" * len(flat), *flat))


def record(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (OUT / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def construct(tokenizer, serialize, examples=SYNTHETIC) -> dict:
    """Independent assistant-only labels; no ID-5-equality padding masks.

    All text/context is synthetic. Prompt ends in a source-verified assistant
    header, then the *only supervised* span is that assistant completion
    followed by one genuine ID-5 turn terminator. Previous conversation turns
    remain context even if they contain earlier assistant text.
    """
    refuse(isinstance(examples, (list, tuple)) and len(examples) == 2,
           "must preserve two synthetic ordinary-role examples")
    refuse(tokenizer.pad_token_id == 5, "pinned source shared END/pad ID differs")
    pack = []
    for messages, completion in examples:
        refuse(isinstance(messages, (tuple, list)) and bool(messages)
               and isinstance(completion, str) and bool(completion)
               and "<|" not in completion and "\x00" not in completion,
               "invalid completion or control-token injection")
        prefix = serialize(messages, variant="chat")
        refuse(isinstance(prefix, str)
               and prefix.endswith("<|assistant|>")
               and "<|tool_" not in prefix
               and "<|memory_" not in prefix,
               "unqualified Chat tool/memory/source role grammar")
        pre = tokenizer.encode(prefix, add_special_tokens=False)
        body = tokenizer.encode(completion, add_special_tokens=False)
        refuse(all(type(t) is int and 0 <= t < 32000 for t in pre + body)
               and len(pre) > 4 and len(body) > 3
               and pre[0] == 3 and pre[-1] == 4 and 5 in pre
               and all(t not in (2, 3, 4, 5, 6, 7, 10, 11) for t in body),
               "wrong original Chat role tokens or illegal assistant content")
        complete = pre + body + [5]  # Explicit real END=5, not EOS text or pad.
        labels = [-100] * len(pre) + body + [5]
        refuse(len(complete) == len(labels)
               and labels[-1] == 5 and labels[len(pre)-1] == -100,
               "assistant-only causal label ownership changed")
        pack.append({
            "text_prefix": prefix, "completion": completion,
            "tokens": complete, "labels": labels,
            "prompt_units": len(pre), "supervised_units": len(body) + 1,
        })
    width = max(len(x["tokens"]) for x in pack)
    refuse(width <= 2048 and len(pack[0]["tokens"]) != len(pack[1]["tokens"]),
           "synthetic Chat batch lacks an actual left-pad boundary")
    ids, attention, labels = [], [], []
    for item in pack:
        offset = width - len(item["tokens"])
        ids.append([5] * offset + item["tokens"])
        attention.append([0] * offset + [1] * len(item["tokens"]))
        labels.append([-100] * offset + item["labels"])
    for row in range(2):
        for j, target in enumerate(labels[row]):
            if target == -100:
                continue
            refuse(j > 0 and attention[row][j-1] == attention[row][j] == 1
                   and ids[row][j] == target,
                   "causal target from padded/unattended predecessor")
    refuse(any(t == 5 and m == 1
               for xs, mask in zip(ids, attention)
               for t, m in zip(xs, mask)),
           "genuine role END hidden by synthetic pad")
    naive = [[int(t != 5) for t in row] for row in ids]
    refuse(naive != attention,
           "naive ID-5 pad/END conflation negative sentinel ineffective")
    refuse(sum(x != -100 for row in labels for x in row) ==
           sum(item["supervised_units"] for item in pack),
           "supervision count differs from synthetic assistant completions")
    return {
        "input_ids": ids, "attention_mask": attention, "labels": labels,
        "supervised_count": sum(x["supervised_units"] for x in pack),
        "items": [
            {"prompt_units": x["prompt_units"],
             "supervised_units": x["supervised_units"],
             "prefix_sha256": sha(x["text_prefix"].encode("utf-8")),
             "completion_sha256": sha(x["completion"].encode("utf-8"))}
            for x in pack
        ],
    }


def independent_ce(logits: torch.Tensor,
                   labels: torch.Tensor,
                   expected_count: int) -> tuple[float, int]:
    refuse(logits.ndim == 3 and labels.ndim == 2
           and logits.shape[:2] == labels.shape and logits.shape[0] == 2,
           "native logits and assistant-mask tensor shapes differ")
    targets = labels[:, 1:]
    active = targets != -100
    count = int(active.sum().item())
    refuse(type(expected_count) is int and expected_count > 8
           and count == expected_count
           and all(bool(active[i].any().item()) for i in range(2)),
           "missing/extra assistant-only targets or host-label leakage")
    predicted = logits[:, :-1, :].float()[active]
    gold = targets[active]
    refuse(bool((gold >= 0).all().item()) and
           bool((gold < logits.shape[-1]).all().item()),
           "wrong model vocabulary or causal target")
    terms = torch.logsumexp(predicted, dim=-1) - predicted.gather(
        1, gold[:, None]).squeeze(1)
    refuse(bool(torch.isfinite(terms).all().item()),
           "nonfinite independent synthetic CE")
    return float(terms.mean().item()), count


def original_source(report: dict):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()
    refuse(os.getenv("GITHUB_EVENT_NAME") == "pull_request"
           and os.getenv("NCR_EXPECTED_HEAD") == head,
           "not executing exact source PR head")
    for path, expected in (
        ("experiments/ncr-001/protocol.json", PROTOCOL_BLOB),
        ("experiments/ncr-001/source-pins.json", PINS_BLOB),
    ):
        actual = subprocess.check_output(["git", "hash-object", path],
                                         cwd=ROOT, text=True).strip()
        refuse(actual == expected, "frozen protocol/source pin changed: "+path)
    protocol = json.loads(
        (ROOT / "experiments/ncr-001/protocol.json").read_text("utf-8"))
    pins = json.loads(
        (ROOT / "experiments/ncr-001/source-pins.json").read_text("utf-8"))
    keys = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
    refuse(protocol["train_authorized"] is False
           and set(protocol["models"]) == set(pins["models"]) == keys,
           "four-lane scientific or train authorization changed")
    spec = verify_contract("chat", protocol, pins)
    refuse(spec["weight_path"] == "chat/model.safetensors"
           and spec["weight_sha256"] == CHAT_SHA
           and pins["models"]["sprocket_instruct"]["hf_weight_file"]
               == "model.safetensors"
           and pins["models"]["sprocket_instruct"]
               ["weight_declared_lfs_sha256_not_downloaded"] == INSTRUCT_SHA,
           "wrong Chat / root Instruct source identity")
    fixture_sha = sha(json.dumps(SYNTHETIC, sort_keys=True, separators=(",", ":")).encode())
    report.update({"source_head": head, "original_postlease_custody_parent": PARENT_HEAD,
                   "original_formatter_source_head": FORMAT_HEAD,
                   "synthetic_only_fixture_sha256": fixture_sha,
                   "sources": {
                       "protocol_git_blob": PROTOCOL_BLOB,
                       "source_pins_git_blob": PINS_BLOB,
                       "chat_checkpoint_sha256": CHAT_SHA,
                       "excluded_root_instruct_checkpoint_sha256": INSTRUCT_SHA,
                   }})
    record(report)
    return spec, pins, protocol


def run(report: dict):
    spec, pins, protocol = original_source(report)
    # Fetch EXACT Chat subfolder files and verify each full file SHA before
    # loader is given offline mode. Existing original #155 load_actual() then
    # independently rehashes every downloaded weight/config/tokenizer/card.
    for name in ("README.md", *SOURCE_FILES["chat"], "chat/model.safetensors"):
        hf_hub_download(spec["model_id"], name, revision=spec["revision"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(4)
    serialize = load_formatter()
    model, tokenizer, nca, mlp, verified, _ = load_actual(
        "chat", spec, pins["models"]["sprocket_chat"], serialize)
    report["frozen_original_chat_executed"] = True
    report["full_verified_source_files"] = verified
    record(report)
    before = fingerprint_backbone(model)
    report["backbone_before_sha256"] = before
    batch = construct(tokenizer, serialize[0])
    ids = torch.tensor(batch["input_ids"], dtype=torch.long)
    attn = torch.tensor(batch["attention_mask"], dtype=torch.long)
    labels = torch.tensor(batch["labels"], dtype=torch.long)
    report["synthetic_transport"] = {
        "shape": list(ids.shape),
        "supervised_targets": batch["supervised_count"],
        "case_sources": batch["items"],
        "input_ids_sha256": int64_sha(batch["input_ids"]),
        "attention_mask_sha256": int64_sha(batch["attention_mask"]),
        "labels_sha256": int64_sha(batch["labels"]),
        "left_padding": [int((attn[i] == 0).sum().item()) for i in range(2)],
        "attended_true_end_id5": bool(((ids == 5) & (attn == 1)).any().item()),
    }
    record(report)
    with torch.inference_mode():
        native = model(input_ids=ids, attention_mask=attn,
                       labels=labels, use_cache=False)
        refuse(native.loss is not None and bool(torch.isfinite(native.loss)),
               "nonfinite original Chat synthetic CE")
        native_loss = float(native.loss.item())
        manual, count = independent_ce(native.logits, labels, batch["supervised_count"])
        difference = abs(native_loss - manual)
        refuse(count == batch["supervised_count"] and math.isfinite(manual)
               and difference <= 1e-5,
               "frozen Chat native/manual shifted CE mismatch")
        report["ce"] = {
            "native": native_loss, "independent": manual, "abs_difference": difference,
            "targets": count, "finite": True, "tolerance_ce_only": 1e-5,
        }
        record(report)
        for kind, wrapper, expected in (
            ("nca", nca, 658904), ("mlp", mlp, 658432),
        ):
            zero = wrapper(input_ids=ids, attention_mask=attn,
                           labels=labels, use_cache=False)
            disabled = wrapper(input_ids=ids, attention_mask=attn,
                               labels=labels, use_cache=False, enabled=False)
            no_op = float((zero.logits - native.logits).abs().max().item())
            off = float((disabled.logits - native.logits).abs().max().item())
            refuse(no_op == off == 0.0 and
                   float(zero.loss.item()) == float(disabled.loss.item()) == native_loss,
                   kind + ": same-frozen-Chat zero/disabled CE parity drift")
            count_params = sum(p.numel() for p in wrapper.adapter_parameters())
            refuse(count_params == expected, "adapter parameter count drift")
            report["controls"][kind] = {
                "zero_logits_exact": True, "disabled_logits_exact": True,
                "zero_and_disabled_native_loss_exact": True,
                "trainable_adapter_parameters": count_params,
                "optimizer_steps": 0, "adapter_untrained": True,
            }
            record(report)
    after = fingerprint_backbone(model)
    refuse(after == before and all(p.grad is None for p in model.parameters()),
           "frozen source checkpoint changed/received a gradient")
    report["backbone_after_sha256"] = after
    report["status"] = STATUS
    record(report)
    return report


def main():
    report = {
        "schema_version": 1, "experiment": "NCR-001E",
        "status": "STARTED_ORIGINAL_CHAT_SOURCE_ONLY_NO_ADMISSION",
        "lane": "sprocket_chat",
        "frozen_original_chat_executed": False,
        "train_authorized": False, "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "tool_memory_behaviour_qualified": False,
        "historical_root_instruct_incomplete_tool_call_retained": True,
        "heldouts_opened": False, "optimizer_steps": 0,
        "comparative_nca_mlp_training": False, "trained_checkpoint_created": False,
        "controls": {},
        "runner": {
            "run": os.getenv("GITHUB_RUN_ID"),
            "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "name": os.getenv("RUNNER_NAME"),
            "host": platform.node(), "pid": os.getpid(),
            "python": platform.python_version(), "os": os.getenv("RUNNER_OS"),
            "torch": torch.__version__, "transformers": transformers.__version__,
        },
    }
    record(report)
    try:
        result = run(report)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        report["status"] = "FAIL_ORIGINAL_CHAT_SYNTHETIC_CE__NO_ADMISSION"
        report["failure"] = {"kind": type(exc).__name__,
                             "message": str(exc), "traceback": traceback.format_exc()}
        record(report)
        raise


if __name__ == "__main__":
    main()
