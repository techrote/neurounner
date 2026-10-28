#!/usr/bin/env python3
"""NCR-001: original void.1 frozen host + corrected causal EOS + post-lease NCA/MLP.

Strictly source-bound, synthetic-only, forward-only; no optimizer, backward,
held-out evaluation, protocol change, behavioral admission or comparison training.
Producer/consumer execute in separate Python interpreters. Partial failure
receipts survive, including source and runner identity.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr001-void-postlease"
AUTHORITY = ROOT / "byte-eos-authority"
AUTHORITY_SHA = "9fbdee880697903aaf5b91cc29442c9e53479e50"
MODEL_REPO = "dotlabs/void.1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHT = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
BYTE_SOURCE = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
LANES = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
CASES = [
    ("empty_completion_one_byte_prefix", "x", ""),
    ("multilingual_utf8", "Q: π?\nA: ", "🔧 café 日本語"),
    ("literal_eos_bytes", "Q:", "<eos>"),
]
REPORT = {}


def require(ok, why):
    if not ok:
        raise RuntimeError("NCR-001 void post-lease REFUSED: " + why)


def git_head(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def frozen_sha(model):
    """Exact tensor-value fingerprint; expensive deliberately, once per phase."""
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(str(tuple(parameter.shape)).encode("ascii"))
        h.update(str(parameter.dtype).encode("ascii"))
        h.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def check_contract(protocol, pins):
    """Pure fail-closed source gate. No model or tokenizer import required."""
    require(protocol.get("train_authorized") is False, "training gate open")
    require(set(protocol.get("models", {})) == LANES
            and set(pins.get("models", {})) == LANES, "four-lane set changed")
    v, p = protocol["models"]["void1"], pins["models"]["void1"]
    require(v["repo_id"] == p["repo_id"] == MODEL_REPO
            and v["revision"] == p["revision"] == REV
            and v["weight_file"] == p["hf_weight_file"] == "model.safetensors"
            and p["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT
            and p["weight_bytes"] == 360622744
            and v["tokenizer_reference"]["train002b_source_sha256"]
                == p["tokenization_byte_py_sha256"] == BYTE_SOURCE
            and v["tokenizer_reference"]["ids"]["eos"] == 258
            and v["expected_config"]["model_type"] == "qwen3"
            and v["expected_config"]["num_hidden_layers"] == 24
            and v["expected_config"]["hidden_size"] == 640
            and tuple(v["taps_after_block_zero_based"]) == (7, 15, 23),
            "original Qwen3/ByteTokenizer pins changed")
    return p


def validate_causal_eos(encoded, prompt, completion):
    """Independently reject real-EOS loss dropping after HF's shift."""
    ids, labels, mask = (encoded[k] for k in
                         ("input_ids", "labels", "attention_mask"))
    prefix = list(prompt.encode("utf-8", errors="strict"))
    targets = list(completion.encode("utf-8", errors="strict")) + [258]
    require(len(prefix) > 0 and len(ids) == len(labels) == len(mask),
            "missing causal context or tensor shape")
    require(len(ids) <= 1024 and mask == [1] * len(ids),
            "source tokenizer unit cap or explicit mask mismatch")
    require(ids == prefix + targets and ids.count(258) == 1,
            "literal EOS text used instead of integer 258")
    require(labels == [-100] * len(prefix) + targets,
            "prompt leak or incorrect terminal EOS supervision")
    require(labels[1:] == [-100] * (len(prefix) - 1) + targets,
            "shifted labels lost a supervised target")
    require(encoded["causal_shift_supervised_units"] == len(targets),
            "falsified shifted target count")
    return {"prompt_bytes": len(prefix), "supervised_after_shift": len(targets),
            "real_eos_id": 258, "total_units": len(ids)}


def save(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    REPORT["elapsed_seconds"] = round(time.monotonic() - REPORT["_started"], 5)
    data = {k: v for k, v in REPORT.items() if k != "_started"}
    (OUT / (phase + ".json")).write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8"
    )


def compare_exact(left, right, label):
    import torch
    require(torch.equal(left, right), label + ": zero/disabled parity changed")
    REPORT["checks"][label] = {"exact": True, "shape": list(left.shape),
                                "sha256": tensor_sha(left)}


def run():
    import torch
    import torch.nn.functional as F
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ncr_001_void1 import FrozenVoid1DepthNCA
    from ncr_001f_frozen_hooks import FrozenDepthGatedMLP, PINNED

    require(os.getenv("HF_HUB_OFFLINE") == "1"
            and os.getenv("TRANSFORMERS_OFFLINE") == "1",
            "actual-host execution must be offline")
    require(git_head(ROOT) == os.environ.get("NCR_EXPECTED_HEAD"),
            "wrong checked-out source head / synthetic merge")
    require(git_head(AUTHORITY) == AUTHORITY_SHA,
            "corrected EOS source is not immutable #163 head")
    for path, sha in (
        ("experiments/ncr-001/protocol.json", "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"),
        ("experiments/ncr-001/source-pins.json", "59fff1aed872b5563aa6898171983973e0cdb02c"),
    ):
        got = subprocess.check_output(
            ["git", "hash-object", str(ROOT / path)], text=True
        ).strip()
        require(got == sha, "frozen protocol/pins Git blob drift: " + path)
    protocol_path = ROOT / "experiments/ncr-001/protocol.json"
    pins_path = ROOT / "experiments/ncr-001/source-pins.json"
    protocol = json.loads(protocol_path.read_text())
    pins = json.loads(pins_path.read_text())
    p = check_contract(protocol, pins)
    spec = PINNED["void1"]
    require(spec["weight_sha256"] == WEIGHT and spec["revision"] == REV
            and spec["model_id"] == MODEL_REPO
            and spec["model_type"] == "qwen3"
            and tuple(spec["taps"]) == (7, 15, 23),
            "post-lease NCA/MLP identity drift")
    sys.path.insert(0, str(AUTHORITY / "scripts"))
    from byte_eos_001_void_labels import encode_supervised_byte_completion
    helper_file = AUTHORITY / "scripts/byte_eos_001_void_labels.py"
    REPORT["source"] = {
        "main_commit": git_head(ROOT), "serializer_commit": git_head(AUTHORITY),
        "protocol_sha256": file_sha(protocol_path),
        "pins_sha256": file_sha(pins_path),
        "corrected_serializer_sha256": file_sha(helper_file),
        "postlease_nca_source_sha256": file_sha(ROOT / "scripts/ncr_001_void1.py"),
        "postlease_mlp_source_sha256": file_sha(ROOT / "scripts/ncr_001f_frozen_hooks.py"),
    }
    REPORT["packages"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    require(transformers.__version__ == "4.57.6",
            "pinned original Qwen3 runtime version drift")
    source_files = {
        "model.safetensors": ("weight_declared_lfs_sha256_not_downloaded", 360622744),
        "config.json": ("config_sha256", None),
        "tokenization_byte.py": ("tokenization_byte_py_sha256", None),
        "byte_vocab.json": ("byte_vocab_sha256", None),
        "tokenizer_config.json": ("tokenizer_config_sha256", None),
        "special_tokens_map.json": ("special_tokens_map_sha256", None),
    }
    REPORT["actual_source_files"] = {}
    for filename, (key, size) in source_files.items():
        path = Path(hf_hub_download(MODEL_REPO, filename=filename,
                                    revision=REV, local_files_only=True))
        digest = file_sha(path)
        require(digest == p[key] and (size is None or path.stat().st_size == size),
                "full original HF source mismatch: " + filename)
        REPORT["actual_source_files"][filename] = {
            "sha256": digest, "bytes": path.stat().st_size}
        save(REPORT["phase"])

    tok = AutoTokenizer.from_pretrained(
        MODEL_REPO, revision=REV, use_fast=False, trust_remote_code=True,
        local_files_only=True)
    executed = inspect.getsourcefile(tok.__class__)
    require(bool(executed) and file_sha(executed) == BYTE_SOURCE,
            "executed remote ByteTokenizer differs from independently reviewed bytes")
    require(tok.eos_token_id == 258 and tok.pad_token_id == 256,
            "ByteTokenizer special-ID mapping mismatch")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261028)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, revision=REV, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False,
        local_files_only=True).eval()
    base.requires_grad_(False)
    require(base.config.model_type == "qwen3"
            and base.config.vocab_size == 259
            and len(base.model.layers) == 24, "original Qwen3 not loaded")
    nca = FrozenVoid1DepthNCA(base, verified_weight_sha256=WEIGHT).eval()
    mlp = FrozenDepthGatedMLP(
        base, lane="void1", verified_weight_sha256=WEIGHT,
        source_revision=REV).eval()
    require(nca._backbone_lock is mlp._backbone_lock,
            "NCA/MLP share no decoder-root ownership lease")
    require(nca.nca.trainable_parameters() == 329944
            and mlp.control.trainable_parameters() == 329472,
            "source-matched adapter size changed")
    REPORT["original_model_loaded"] = True
    REPORT["parameters"] = {"nca": nca.nca.trainable_parameters(),
                            "mlp": mlp.control.trainable_parameters()}
    before = frozen_sha(base)
    REPORT["frozen_backbone_sha256_before"] = before
    save(REPORT["phase"])
    return torch, F, base, tok, nca, mlp, encode_supervised_byte_completion


def producer(torch, F, base, tok, nca, mlp, serialize):
    with torch.inference_mode():
        for name, prompt, completion in CASES:
            row = serialize(tok, prompt, completion)
            witness = validate_causal_eos(row, prompt, completion)
            ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            mask = torch.tensor([row["attention_mask"]], dtype=torch.long)
            labels = torch.tensor([row["labels"]], dtype=torch.long)
            raw = base(input_ids=ids, attention_mask=mask,
                       labels=labels, use_cache=False)
            require(raw.loss is not None and bool(torch.isfinite(raw.loss)),
                    "original Qwen3 finite loss missing: " + name)
            manual = F.cross_entropy(
                raw.logits[:, :-1, :].contiguous().view(-1, 259),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100)
            require(abs(float(raw.loss) - float(manual)) <= 1e-5,
                    "native Qwen3 causal shift differs from independent CE")
            for kind, wrapper in (("nca", nca), ("mlp", mlp)):
                zero = wrapper(input_ids=ids, attention_mask=mask,
                               labels=labels, use_cache=False)
                off = wrapper(input_ids=ids, attention_mask=mask,
                              labels=labels, use_cache=False, enabled=False)
                compare_exact(raw.logits, zero.logits, name + "_" + kind + "_zero")
                compare_exact(raw.logits, off.logits, name + "_" + kind + "_disabled")
                require(float(zero.loss) == float(raw.loss),
                        "zero-adapter native EOS causal loss drift")
            REPORT["cases"].append({
                "name": name, **witness, "ids": row["input_ids"],
                "labels": row["labels"], "native_frozen_loss": float(raw.loss),
                "manual_shifted_ce": float(manual), "logits_sha256": tensor_sha(raw.logits),
            })
            save("producer")

        probe = ("Q: π 🔧 inspect the archive, check byte order. A: " * 6)
        bytes_ids = tok.encode(probe, add_special_tokens=True)
        require(len(bytes_ids) >= 128 and all(0 <= i < 256 for i in bytes_ids),
                "source probe insufficient or inserted control ID")
        x = torch.tensor([bytes_ids[:128]], dtype=torch.long)
        mask = torch.ones_like(x)
        for owner, foreign in ((nca, mlp), (mlp, nca)):
            with owner._instrument():
                try:
                    foreign(input_ids=x[:, :8], enabled=False, use_cache=False)
                except RuntimeError as exc:
                    require("Shared frozen" in str(exc),
                            "unexpected cross-lease refusal: " + str(exc))
                else:
                    raise RuntimeError("disabled baseline contaminated by foreign hooks")
        REPORT["checks"]["cross_adapter_lease"] = "disabled_reference_refused"
        with torch.no_grad():
            nca.nca.readout.weight[0, 0] = 0.03
            mlp.control.down.weight[0, 0] = 0.05
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            original = wrapper(input_ids=x[:, :64], attention_mask=mask[:, :64],
                               use_cache=False, enabled=False).logits
            active = wrapper(input_ids=x[:, :64], attention_mask=mask[:, :64],
                             use_cache=False).logits
            effect = float((original - active).abs().max())
            require(effect > 0 and bool(torch.isfinite(active).all()),
                    "manually perturbed untrained " + kind + " inactive/nonfinite")
            REPORT["checks"][kind + "_manual_untrained_effect"] = {
                "max_abs": effect, "trained": False}
            for n in (32, 128):
                full = wrapper(input_ids=x[:, :n],
                               attention_mask=mask[:, :n], use_cache=False).logits[:, -3:]
                pre = wrapper(input_ids=x[:, :n-3],
                              attention_mask=mask[:, :n-3], use_cache=True)
                past = pre.past_key_values
                parts = []
                for pos in range(n-3, n):
                    one = wrapper(input_ids=x[:, pos:pos+1],
                                  attention_mask=mask[:, :pos+1],
                                  past_key_values=past, use_cache=True)
                    parts.append(one.logits[:, -1:])
                    past = one.past_key_values
                cached = torch.cat(parts, dim=1)
                delta = float((full - cached).abs().max())
                require(torch.isfinite(cached).all() and
                        torch.allclose(full, cached, atol=0.0005, rtol=0.0005),
                        kind + " Qwen3 true incremental-cache failure: " + str(delta))
                REPORT["checks"][kind + "_cache_" + str(n)] = {
                    "max_abs": delta, "tolerance_atol": 0.0005, "tolerance_rtol": 0.0005}
                save("producer")
            a = wrapper(input_ids=x[:, :24], use_cache=False).logits
            wrapper(input_ids=x[:, 24:48], use_cache=False)
            b = wrapper(input_ids=x[:, :24], use_cache=False).logits
            compare_exact(a, b, kind + "_A_B_A_isolation")
            suffix = x[:, :32].clone()
            first = wrapper(input_ids=suffix, use_cache=False).logits
            suffix[0, -1] = (int(suffix[0, -1]) + 3) % 256
            second = wrapper(input_ids=suffix, use_cache=False).logits
            compare_exact(first[:, :-1], second[:, :-1],
                          kind + "_causal_future_suffix")
            REPORT["checks"][kind + "_cold_logits_sha256"] = tensor_sha(active)
            folder = OUT / kind
            wrapper.save_adapter(folder)
            REPORT["checks"][kind + "_state_sha256"] = file_sha(folder / "adapter.pt")
            REPORT["checks"][kind + "_metadata_sha256"] = file_sha(folder / "metadata.json")
            save("producer")
        require(not base.model._forward_hooks and
                not base.model._forward_pre_hooks,
                "original Qwen3 leaked decoder hooks")
    REPORT["status"] = "PASS_ORIGINAL_VOID1_POSTLEASE_EOS_NCA_MLP_FORWARD_ONLY__72_BLOCKED"


def consumer(torch, base, tok, nca, mlp):
    old = json.loads((OUT / "producer.json").read_text())
    require(old["status"] ==
            "PASS_ORIGINAL_VOID1_POSTLEASE_EOS_NCA_MLP_FORWARD_ONLY__72_BLOCKED",
            "producer incomplete")
    require(old["source"] == REPORT["source"] and
            old["actual_source_files"] == REPORT["actual_source_files"],
            "consumer source mismatch")
    require(old["runner"]["run_id"] == REPORT["runner"]["run_id"]
            and old["runner"]["pid"] != REPORT["runner"]["pid"],
            "not a different Python interpreter on same original run")
    require(old["frozen_backbone_sha256_before"] ==
            REPORT["frozen_backbone_sha256_before"], "frozen checkpoint mismatch")
    with torch.inference_mode():
        probe = ("Q: π 🔧 inspect the archive, check byte order. A: " * 6)
        x = torch.tensor([tok.encode(probe, add_special_tokens=True)[:64]],
                         dtype=torch.long)
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            folder = OUT / kind
            for name, key in (("adapter.pt", kind + "_state_sha256"),
                              ("metadata.json", kind + "_metadata_sha256")):
                require(file_sha(folder / name) == old["checks"][key],
                        "cold-load adapter artifact mutated: " + name)
            wrapper.load_adapter(folder)
            active = wrapper(input_ids=x, use_cache=False).logits
            require(tensor_sha(active) == old["checks"][kind + "_cold_logits_sha256"],
                    "new interpreter cold NCA/MLP logits differ: " + kind)
            REPORT["checks"][kind + "_separate_process_cold_reload"] = {
                "exact": True, "logits_sha256": tensor_sha(active)}
            save("consumer")
    REPORT["status"] = "PASS_ORIGINAL_VOID1_POSTLEASE_COLD_RELOAD_ONLY__72_BLOCKED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["producer", "consumer"], required=True)
    phase = ap.parse_args().phase
    REPORT.update({
        "_started": time.monotonic(), "schema": "ncr001-void-eos-postlease-v1",
        "phase": phase, "status": "RUNNING", "variant": "void1",
        "model": MODEL_REPO, "revision": REV, "training_authorized": False,
        "optimizer_steps": 0, "backward_executed": False,
        "comparative_training_performed": False, "heldout_opened": False,
        "full_72_conformance_pass": False, "independent_scientific_review": False,
        "original_model_loaded": False, "cases": [], "checks": {},
        "runner": {
            "pid": os.getpid(), "platform": platform.platform(),
            "python": platform.python_version(),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "runner_os": os.getenv("RUNNER_OS"),
            "expected_head": os.getenv("NCR_EXPECTED_HEAD"),
            "event_merge_sha": os.getenv("GITHUB_SHA"),
        },
    })
    save(phase)
    try:
        torch, F, model, tok, nca, mlp, serialize = run()
        if phase == "producer":
            producer(torch, F, model, tok, nca, mlp, serialize)
        else:
            consumer(torch, model, tok, nca, mlp)
        after = frozen_sha(model)
        REPORT["frozen_backbone_sha256_after"] = after
        require(after == REPORT["frozen_backbone_sha256_before"],
                "actual frozen Qwen3 backbone changed")
        require(all(p.grad is None for p in model.parameters()),
                "frozen original backbone accumulated gradients")
    except BaseException:
        REPORT["status"] = "FAIL_ORIGINAL_VOID1_POSTLEASE_EOS_QUALIFICATION"
        REPORT["traceback"] = traceback.format_exc()
        raise
    finally:
        save(phase)
    print(json.dumps({"status": REPORT["status"], "phase": phase,
                      "checks": list(REPORT["checks"]),
                      "train_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
