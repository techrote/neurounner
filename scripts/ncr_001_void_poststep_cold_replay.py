#!/usr/bin/env python3
"""One public synthetic NCA-only step, then genuine separate-process cold replay.

This is a #72 diagnostic, NOT comparative NCA/MLP training or model admission.
No heldouts, learned quality claims, fixture edits, or checkpoint publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr001-void-poststep-cold"
ADAPTER = OUT / "nca-after-one-public-synthetic-step"
PRODUCER = OUT / "producer.json"
EXPECTED_BLOBS = {
    "experiments/ncr-001/protocol.json": "14ce5fbaa38f8806c8c8738f19c01b91385eebaf",
    "experiments/ncr-001/source-pins.json": "59fff1aed872b5563aa6898171983973e0cdb02c",
    "scripts/ncr_001_core.py": "0427ad0b0f1fc68199369d5094952017bc70f503",
    "scripts/ncr_001_void1.py": "7e6abcbf1b3738094e2b460089babed150c72a9b",
    "scripts/ncr_001_void_nca_learning_gate.py": "672fede7cb488bb33315b067bba5292678435644",
    "scripts/ncr_001_void_eos_postlease.py": "45d170472f3b37c6e13088212deab60bf2b223d7",
}
EXPECTED_MODEL = "dotlabs/void.1"
EXPECTED_REV = "9c2be5087772867d45961abd11f6b35110617563"
EXPECTED_WEIGHTS = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
STATUS = "POSTSTEP_COLD_DIAGNOSTIC_ONLY__SCIENTIFIC_72_BLOCKED"


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError("NCR poststep cold replay REFUSED: " + reason)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def source_gate(*, expected_head: str | None = None) -> dict:
    """Pin committed original code and protocol; refuse an opened train gate."""
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if expected_head is None:
        expected_head = os.environ.get("NCR_EXPECTED_HEAD")
    require(bool(expected_head) and head == expected_head, "nonliteral checkout head")
    for name, expected in EXPECTED_BLOBS.items():
        actual = subprocess.check_output(
            ["git", "-C", str(ROOT), "hash-object", str(ROOT / name)], text=True
        ).strip()
        require(actual == expected, "immutable source Git-blob drift " + name)
    policy = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    require(policy["train_authorized"] is False and
            set(policy["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "four-model protocol training/identity drift")
    v = policy["models"]["void1"]
    require(v["repo_id"] == EXPECTED_MODEL and v["revision"] == EXPECTED_REV and
            v["weight_file"] == "model.safetensors" and
            v["tokenizer_reference"]["ids"]["eos"] == 258,
            "void.1 exact checkpoint/byte-EOS pin drift")
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "actual physical invocation must load pinned offline source")
    return {"head": head, "original_source_git_blobs": EXPECTED_BLOBS.copy()}


def validate_producer(row: dict) -> None:
    """Independent consumer refuses accidental training or sample substitutions."""
    require(row["status"] == STATUS and row["phase"] == "producer" and
            row["diagnostic_optimizer_steps"] == 1 and
            type(row["diagnostic_optimizer_steps"]) is int and
            row["train_authorized"] is False and
            row["comparative_training_performed"] is False and
            row["heldouts_opened"] is False and
            row["full_72_conformance_pass"] is False and
            row["independent_scientific_review"] is False,
            "original public-sample/administrative bounds changed")
    require(row["model_repo"] == EXPECTED_MODEL and
            row["revision"] == EXPECTED_REV and
            row["weights_sha256"] == EXPECTED_WEIGHTS and
            row["sample"]["real_eos_id"] == 258 and
            row["sample"]["supervised_after_shift"] == 21 and
            row["sample"]["total_units"] == 31 and
            row["source"]["head"] and
            row["threads_effective_after_load"] == 4 and
            type(row["threads_effective_after_load"]) is int and
            row["producer_pid"] > 0 and
            row["active_vs_disabled_max_abs"] > 0 and
            math.isfinite(row["active_vs_disabled_max_abs"]) and
            all(isinstance(row[k], str) and len(row[k]) == 64 and
                all(c in "0123456789abcdef" for c in row[k])
                for k in ("active_logits_sha256", "frozen_backbone_sha256",
                          "adapter_sha256", "adapter_metadata_sha256")),
            "original public sample/physical source or cold reference invalid")


def write(row: dict, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n", encoding="utf-8")


def load_original():
    import torch
    import ncr_001_void_eos_postlease as original
    original.REPORT.clear()
    original.REPORT.update({
        "_started": time.monotonic(), "phase": "poststep-cold",
        "status": "SOURCE_LOADER", "checks": {},
        "runner": {"pid": os.getpid(), "run_id": os.environ.get("GITHUB_RUN_ID")},
    })
    package, _, backbone, tokenizer, wrapper, _unused_mlp, serialize = original.run()
    require(package is torch and torch.get_num_threads() == 4,
            "original loader effective CPU threading not observed")
    require(original.frozen_sha(backbone) and
            not any(p.requires_grad for p in backbone.parameters()),
            "original frozen Qwen3 base absent or trainable")
    return torch, original, backbone, tokenizer, wrapper, serialize


def execute(phase: str) -> None:
    row = {
        "schema": "ncr001-void-poststep-cold-v1", "status": "STARTED",
        "phase": phase, "train_authorized": False,
        "comparative_training_performed": False, "heldouts_opened": False,
        "full_72_conformance_pass": False, "independent_scientific_review": False,
        "model_repo": EXPECTED_MODEL, "revision": EXPECTED_REV,
        "weights_sha256": EXPECTED_WEIGHTS,
        "producer_pid": os.getpid() if phase == "producer" else None,
        "diagnostic_optimizer_steps": 0,
        "runner": {"name": os.environ.get("RUNNER_NAME"),
                   "run_id": os.environ.get("GITHUB_RUN_ID"),
                   "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                   "python": platform.python_version(),
                   "platform": platform.platform()},
    }
    write(row, phase + ".json")
    try:
        row["source"] = source_gate()
        torch, original, backbone, tok, wrapper, serialize = load_original()
        import ncr_001_void_nca_learning_gate as learning
        row["threads_effective_after_load"] = torch.get_num_threads()
        row["runner"]["torch"] = torch.__version__
        row["frozen_backbone_sha256"] = original.frozen_sha(backbone)
        ids, labels, mask, sample = learning.make_sample(serialize, tok)
        row["sample"] = sample
        write(row, phase + ".json")
        if phase == "producer":
            # Reuse the exact immutable reviewed #172 NCA-only gradient diagnostic:
            # ONE synthetic SGD step, second backward only, NO MLP optimizer.
            learning.REPORT.clear()
            learning.REPORT.update({"_started": time.monotonic(),
                                    "status": "NEW_DIAGNOSTIC_NOT_ORIGINAL_172"})
            learning.one_step_diagnostic(backbone, wrapper, tok, serialize)
            require(learning.REPORT["diagnostic_optimizer_steps"] == 1 and
                    learning.REPORT["checks"]["backbone_unchanged_after_one_step"],
                    "one-step original-source diagnostic did not complete")
            row["diagnostic_optimizer_steps"] = 1
            row["learning_diagnostic_receipt_sha256"] = digest(
                learning.OUT / "receipt.json")
            wrapper.eval()
            with torch.no_grad():
                active = wrapper(input_ids=ids, attention_mask=mask,
                                 use_cache=False).logits
                disabled = wrapper(input_ids=ids, attention_mask=mask,
                                   use_cache=False, enabled=False).logits
            difference = float((active - disabled).abs().max())
            require(difference > 0 and bool(torch.isfinite(active).all()),
                    "single-step adapter produced no finite nonzero effect")
            row["active_vs_disabled_max_abs"] = difference
            row["active_logits_sha256"] = original.tensor_sha(active)
            wrapper.save_adapter(ADAPTER)
            row["adapter_sha256"] = digest(ADAPTER / "adapter.pt")
            row["adapter_metadata_sha256"] = digest(ADAPTER / "metadata.json")
            require(original.frozen_sha(backbone) ==
                    row["frozen_backbone_sha256"],
                    "backbone changed during one-step diagnostic")
            row["status"] = STATUS
            validate_producer(row)
        else:
            previous = json.loads(PRODUCER.read_text(encoding="utf-8"))
            validate_producer(previous)
            require(previous["source"] == row["source"] and
                    previous["frozen_backbone_sha256"] ==
                    row["frozen_backbone_sha256"] and
                    previous["producer_pid"] != os.getpid() and
                    previous["runner"]["run_id"] == row["runner"]["run_id"],
                    "not same pinned source/runner in a new Python process")
            require(digest(ADAPTER / "adapter.pt") == previous["adapter_sha256"] and
                    digest(ADAPTER / "metadata.json") ==
                    previous["adapter_metadata_sha256"],
                    "saved one-step adapter bytes altered between interpreters")
            wrapper.load_adapter(ADAPTER)
            wrapper.eval()
            with torch.no_grad():
                active = wrapper(input_ids=ids, attention_mask=mask,
                                 use_cache=False).logits
                a = wrapper(input_ids=ids, attention_mask=mask,
                            use_cache=False).logits
                wrapper(input_ids=ids[:, :-3], use_cache=False)
                b = wrapper(input_ids=ids, attention_mask=mask,
                            use_cache=False).logits
                prefix = wrapper(input_ids=ids[:, :-4],
                                 attention_mask=mask[:, :-4], use_cache=True)
                past = prefix.past_key_values
                tail = []
                for pos in range(ids.shape[1] - 4, ids.shape[1]):
                    one = wrapper(
                        input_ids=ids[:, pos:pos+1],
                        attention_mask=mask[:, :pos+1],
                        past_key_values=past, use_cache=True)
                    tail.append(one.logits[:, -1:])
                    past = one.past_key_values
                cached = torch.cat(tail, dim=1)
            observed = original.tensor_sha(active)
            max_abs = float((active[:, -4:] - cached).abs().max())
            require(observed == previous["active_logits_sha256"] and
                    torch.equal(a, b) and
                    bool(torch.isfinite(cached).all()) and
                    bool(torch.allclose(active[:, -4:], cached, atol=5e-4, rtol=5e-4)),
                    "post-step true-cold logits, reset, or incremental KV mismatch")
            require(original.frozen_sha(backbone) ==
                    previous["frozen_backbone_sha256"],
                    "cold-reloaded frozen backbone identity changed")
            row.update({
                "producer_pid": previous["producer_pid"],
                "cold_process_pid": os.getpid(),
                "active_logits_sha256": observed,
                "per_request_aba_exact": True,
                "post_step_last_four_true_kv_max_abs": max_abs,
                "incremental_tokens": 4,
                "context_units": 31,
                "original_producer_receipt_sha256": digest(PRODUCER),
                "post_step_cold_reload_exact": True,
            })
            row["status"] = STATUS
    except BaseException:
        row["status"] = "FAIL__72_BLOCKED"
        row["traceback"] = traceback.format_exc()
        raise
    finally:
        write(row, phase + ".json")
        print(json.dumps({"phase": phase, "status": row["status"],
                          "steps": row["diagnostic_optimizer_steps"],
                          "train_authorized": False,
                          "full_72_conformance_pass": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("producer", "consumer"), required=True)
    execute(parser.parse_args().phase)
