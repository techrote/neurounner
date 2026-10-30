#!/usr/bin/env python3
"""BYTE-EOS-001 real frozen Qwen3 causal-loss, forward only; NO training."""
from __future__ import annotations
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
OUT = ROOT / "runtime/ncr-001/byte-eos-frozen-qwen-forward.json"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
REPO = "dotlabs/void.1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHT_SHA = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER_SHA = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
LANES = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
CASES = (
    ("empty_completion_after_one_byte_context", "x", ""),
    ("mixed_utf8", "Q: π?\nA: ", "🔧 café 日本語"),
    ("literal_eos_text_is_plain_bytes", "Q:", "<eos>"),
)

def require(ok, why):
    if not ok:
        raise RuntimeError("BYTE-EOS original frozen host REFUSED: " + why)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def validate_shifted_record(encoded, prompt, completion):
    """Independent pre-forward witness for causal loss masks and real EOS."""
    ids, labels, mask = (encoded[k] for k in
                         ("input_ids", "labels", "attention_mask"))
    pbytes = list(prompt.encode("utf-8", errors="strict"))
    suffix = list(completion.encode("utf-8", errors="strict")) + [258]
    n = len(pbytes)
    require(n > 0 and len(ids) == len(labels) == len(mask), "shape or prefix")
    require(len(ids) <= 1024 and all(v == 1 for v in mask), "mask/cap")
    require(ids[:n] == pbytes and ids[n:] == suffix and ids.count(258) == 1,
            "real EOS suffix/UTF-8 identity")
    require(labels[:n] == [-100] * n and labels[n:] == suffix,
            "completion-only mask and true EOS")
    require(labels[1:] == [-100] * (n-1) + suffix,
            "post-shift causal supervision drift")
    require(encoded["causal_shift_supervised_units"] == len(suffix) and
            sum(x != -100 for x in labels[1:]) == len(suffix),
            "falsely counted shifted target")
    return {"prompt_units": n, "causal_supervised_units": len(suffix),
            "terminal_eos_id": 258, "all_real_tokens_attended": True}

def backbone_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update((name + str(tuple(value.shape)) + str(value.dtype)).encode())
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    start = time.monotonic()
    report = {
        "schema": "byte-eos-001-frozen-qwen-forward-v1", "status": "RUNNING",
        "variant": "void1", "repo": REPO, "revision": REV,
        "source_head": os.getenv("EXPECTED_HEAD"), "event_merge_sha": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"), "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "runner_name": os.getenv("RUNNER_NAME"), "runner_os": os.getenv("RUNNER_OS"),
        "python": platform.python_version(), "platform": platform.platform(),
        "model_executed": False, "byte_tokenizer_executed": False,
        "optimizer_steps": 0, "backward_executed": False,
        "train_authorized": False, "comparative_training_performed": False,
        "heldout_opened": False, "historical_train002b_effect_measured": False,
        "full_72_conformance_pass": False, "independent_scientific_review": False,
        "cases": [],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["wall_seconds"] = round(time.monotonic()-start, 5)
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True,
                                  ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    save()
    try:
        import torch
        import torch.nn.functional as F
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import hf_hub_download
        require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1",
                "actual host execution not offline")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        require(head == os.getenv("EXPECTED_HEAD"), "synthetic PR merge head/source mismatch")
        report["source_head"] = head
        report["torch"] = torch.__version__
        report["transformers"] = transformers.__version__
        protocol = json.loads(PROTOCOL.read_text())
        pins = json.loads(PINS.read_text())
        require(protocol["train_authorized"] is False and
                set(protocol["models"]) == set(pins["models"]) == LANES,
                "four-lane training gate changed")
        v, p = protocol["models"]["void1"], pins["models"]["void1"]
        require(v["repo_id"] == p["repo_id"] == REPO and
                v["revision"] == p["revision"] == REV and
                v["weight_file"] == p["hf_weight_file"] == "model.safetensors" and
                p["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT_SHA and
                p["weight_bytes"] == 360622744 and
                p["tokenization_byte_py_sha256"] ==
                    v["tokenizer_reference"]["train002b_source_sha256"] == TOKENIZER_SHA and
                v["expected_config"]["vocab_size"] == 259 and
                v["expected_config"]["num_hidden_layers"] == 24 and
                v["expected_config"]["hidden_size"] == 640,
                "frozen Qwen3 source/architecture pin")
        report["source_sha256"] = {
            "protocol": sha_file(PROTOCOL), "source_pins": sha_file(PINS),
            "future_serializer": sha_file(ROOT/"scripts/byte_eos_001_void_labels.py"),
        }
        for filename, key in (
            ("model.safetensors", "weight_declared_lfs_sha256_not_downloaded"),
            ("config.json", "config_sha256"),
            ("tokenization_byte.py", "tokenization_byte_py_sha256"),
            ("byte_vocab.json", "byte_vocab_sha256"),
            ("tokenizer_config.json", "tokenizer_config_sha256"),
            ("special_tokens_map.json", "special_tokens_map_sha256"),
        ):
            file = Path(hf_hub_download(REPO, filename=filename, revision=REV,
                                        local_files_only=True))
            digest = sha_file(file)
            require(digest == p[key], "downloaded exact source drift: " + filename)
            if filename == "model.safetensors":
                require(file.stat().st_size == 360622744, "incomplete weight bytes")
            report.setdefault("source_files", {})[filename] = {
                "sha256": digest, "bytes": file.stat().st_size}
            save()
        tokenizer = AutoTokenizer.from_pretrained(
            REPO, revision=REV, trust_remote_code=True, use_fast=False,
            local_files_only=True)
        executed_source = inspect.getsourcefile(tokenizer.__class__)
        require(bool(executed_source) and sha_file(executed_source) == TOKENIZER_SHA,
                "runtime tokenizer Python differs from reviewed original")
        sys.path.insert(0, str(ROOT/"scripts"))
        from byte_eos_001_void_labels import encode_supervised_byte_completion
        report["byte_tokenizer_executed"] = True
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261028)
        model = AutoModelForCausalLM.from_pretrained(
            REPO, revision=REV, torch_dtype=torch.float32,
            attn_implementation="eager", trust_remote_code=False,
            local_files_only=True).eval()
        model.requires_grad_(False)
        require(model.config.model_type == "qwen3" and
                model.config.vocab_size == 259 and
                len(model.model.layers) == 24 and
                all(not x.requires_grad for x in model.parameters()),
                "actual original frozen Qwen3 not loaded")
        report["model_executed"] = True
        report["backbone_sha_before"] = backbone_sha(model)
        save()
        with torch.inference_mode():
            for case, prompt, completion in CASES:
                encoded = encode_supervised_byte_completion(tokenizer, prompt, completion)
                witness = validate_shifted_record(encoded, prompt, completion)
                ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
                mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long)
                labels = torch.tensor([encoded["labels"]], dtype=torch.long)
                output = model(input_ids=ids, attention_mask=mask,
                               labels=labels, use_cache=False)
                require(output.loss is not None and
                        bool(torch.isfinite(output.loss).all()) and
                        output.logits.shape == (1, len(encoded["input_ids"]), 259) and
                        bool(torch.isfinite(output.logits).all()),
                        case + ": original-host nonfinite/invalid logits or loss")
                manual = F.cross_entropy(
                    output.logits[:, :-1, :].contiguous().view(-1, 259),
                    labels[:, 1:].contiguous().view(-1),
                    ignore_index=-100, reduction="mean")
                difference = abs(float(output.loss) - float(manual))
                require(bool(torch.isfinite(manual)) and difference <= 1e-5,
                        case + ": actual Qwen3 causal loss differs from shifted CE")
                report["cases"].append({
                    "name": case, "synthetic_prompt": prompt,
                    "synthetic_completion": completion, **witness,
                    "sequence_units": len(encoded["input_ids"]),
                    "source_utf8_bytes": encoded["utf8_source_bytes"],
                    "source_characters": encoded["source_characters"],
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": encoded["labels"],
                    "original_qwen3_loss": float(output.loss),
                    "manual_shifted_ce": float(manual),
                    "difference": difference,
                    "logits_sha256": hashlib.sha256(
                        output.logits.contiguous().cpu().numpy().tobytes()).hexdigest(),
                })
                save()
        report["backbone_sha_after"] = backbone_sha(model)
        require(report["backbone_sha_before"] == report["backbone_sha_after"],
                "original frozen Qwen3 weights changed")
        require(all(param.grad is None for param in model.parameters()),
                "frozen Qwen3 gradient unexpectedly populated")
        require(len(report["cases"]) == 3, "incomplete source-fixed cases")
        report["status"] = "PASS_FROZEN_QWEN3_CAUSAL_EOS_FORWARD_ONLY__72_BLOCKED"
        report["limitations"] = [
            "No NCA or matched MLP hooks tested by this forward-only witness",
            "No optimizer/backward, post-update hash or post-warmup gradients",
            "No behavioral EOS generation or historical TRAIN-002b effect measured",
            "Synthetic cases only, no held-out source opened",
            "Not independent scientific review or final four-model #72 PASS",
        ]
    except BaseException:
        report["status"] = "FAIL_FROZEN_QWEN3_CAUSAL_EOS_FORWARD"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save()
    print(json.dumps({"status":report["status"],
                      "case_losses":[(r["name"],r["original_qwen3_loss"]) for r in report["cases"]],
                      "train_authorized":False},sort_keys=True))

if __name__ == "__main__":
    main()
