#!/usr/bin/env python3
"""NCR-001E: fresh void.1 real-model byte-tokenized cache/cold-adapter review.

The original PR #85 wrapper is checked out at a pinned revision in implementation.
Only reviewed tokenizer bytes are dynamically loaded, in an offline process.
Zero optimizer steps. Does not qualify full four-model NCR-001E.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import traceback

import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "implementation"
IMPLEMENTATION_COMMIT = "5fa3dc8c0de153098d1800f7b0c809819620aef4"
PIN = "9c2be5087772867d45961abd11f6b35110617563"
REPO = "dotlabs/void.1"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
OUTPUT = ROOT / "runtime/ncr-001/void-real-text-fresh.json"

RECORD = {
    "experiment": "NCR-001E",
    "stage": "fresh_pinned_void1_reviewed_UTF8_text_long_cache_cold_adapter",
    "status": "RUNNING",
    "model_training_performed": False,
    "optimizer_steps": 0,
    "comparative_training_admitted": False,
    "full_four_model_conformance_pass": False,
    "actual_byte_tokenizer_executed_after_review": False,
    "checks": {},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(x)
    return h.hexdigest()


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def save() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RECORD["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUTPUT.write_text(json.dumps(RECORD, indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n", encoding="utf-8")


def compare(key: str, a: torch.Tensor, b: torch.Tensor,
            atol: float, rtol: float) -> None:
    require(a.shape == b.shape, f"{key}: logits shapes differ")
    delta = (a - b).abs()
    d = {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "atol": atol, "rtol": rtol,
        "within": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }
    RECORD["checks"][key] = d
    save()
    require(d["within"], f"{key}: {d}")


def main() -> None:
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Independent reviewer must use an offline pinned source")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261027)
    p = json.loads(PROTOCOL.read_text())
    sources = json.loads(PINS.read_text())
    m = p["models"]["void1"]
    source = sources["models"]["void1"]
    require(p["train_authorized"] is False, "Training admission gate unexpectedly open")
    require(source["revision"] == m["revision"] == PIN, "Wrong void pinned revision")
    require(source["repo_id"] == m["repo_id"] == REPO, "Wrong void source repo")
    require(m["taps_after_block_zero_based"] == [7, 15, 23], "Void taps changed")
    require(IMPLEMENTATION.is_dir(), "Exact PR #85 implementation checkout missing")
    commit = subprocess.check_output(
        ["git", "-C", str(IMPLEMENTATION), "rev-parse", "HEAD"],
        text=True
    ).strip()
    require(commit == IMPLEMENTATION_COMMIT, "Unexpected Qwen3 NCA implementation revision")
    sys.path.insert(0, str(IMPLEMENTATION / "scripts"))
    from ncr_001_void1 import (  # type: ignore
        FrozenVoid1DepthNCA, MODEL_ID, MODEL_REVISION, POST_BLOCK_TAPS
    )
    require(MODEL_ID == REPO and MODEL_REVISION == PIN
            and list(POST_BLOCK_TAPS) == [7, 15, 23],
            "Qwen3 adapter identity from source PR differs")

    RECORD["source"] = {
        "repo": REPO,
        "revision": PIN,
        "implementation_sha": IMPLEMENTATION_COMMIT,
        "protocol_sha256": sha(PROTOCOL),
        "source_pins_sha256": sha(PINS),
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
    }
    RECORD["source_files"] = {}
    file_specs = (
        ("model.safetensors", "weight_declared_lfs_sha256_not_downloaded"),
        ("config.json", "config_sha256"),
        ("tokenization_byte.py", "tokenization_byte_py_sha256"),
        ("byte_vocab.json", "byte_vocab_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("special_tokens_map.json", "special_tokens_map_sha256"),
    )
    for path, field in file_specs:
        source_path = Path(hf_hub_download(REPO, filename=path, revision=PIN,
                                           local_files_only=True))
        actual = sha(source_path)
        require(actual == source[field], f"Actual pinned byte mismatch {path}: {actual}")
        RECORD["source_files"][path] = {
            "sha256": actual, "bytes": source_path.stat().st_size
        }
        save()
    require(RECORD["source_files"]["tokenization_byte.py"]["sha256"]
            == m["tokenizer_reference"]["train002b_source_sha256"],
            "ByteTokenizer code differs from TRAIN-002b")

    # Prior source-only review #90 documents the exact AST surface and imports.
    # Recheck its strict allowlist before importing the reviewed Python module.
    sys.path.insert(0, str(ROOT / "scripts"))
    from ncr_001k_byte_runtime import verify_reviewed_shape  # type: ignore
    source_file = Path(hf_hub_download(
        REPO, filename="tokenization_byte.py", revision=PIN, local_files_only=True
    ))
    RECORD["reviewed_tokenizer_shape"] = verify_reviewed_shape(
        ast.parse(source_file.read_text(), filename="tokenization_byte.py")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        REPO, revision=PIN, trust_remote_code=True,
        use_fast=False, local_files_only=True,
    )
    runtime_source = inspect.getsourcefile(tokenizer.__class__)
    require(bool(runtime_source), "Actual executed tokenizer source path missing")
    require(sha(Path(runtime_source)) == source["tokenization_byte_py_sha256"],
            "Executed ByteTokenizer code differs from reviewed pinned SHA")
    require(tokenizer.vocab_size == 259
            and tokenizer.pad_token_id == 256
            and tokenizer.bos_token_id == 257
            and tokenizer.eos_token_id == 258,
            "Actual ByteTokenizer special ID semantics changed")
    require(tokenizer.encode("<eos>", add_special_tokens=True)
            == list(b"<eos>"), "Literal EOS text unexpectedly became token ID")
    RECORD["actual_byte_tokenizer_executed_after_review"] = True
    RECORD["tokenizer_identity"] = {
        "source_sha256": source["tokenization_byte_py_sha256"],
        "class": tokenizer.__class__.__name__,
        "pad_id": 256, "bos_id": 257, "eos_id": 258,
        "auto_insert_eos": False,
        "literal_text_eos_ids": [60, 101, 111, 115, 62],
    }
    save()

    base = AutoModelForCausalLM.from_pretrained(
        REPO, revision=PIN, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False,
        local_files_only=True,
    ).eval()
    wrapper = FrozenVoid1DepthNCA(
        base, verified_weight_sha256=source["weight_declared_lfs_sha256_not_downloaded"]
    ).eval()
    wrapper.verify_frozen_base()
    require(wrapper.nca.trainable_parameters() == 329944, "NCA count drift")
    require(all(not x.requires_grad for x in base.parameters()), "Model base unfrozen")
    RECORD["frozen_base_parameters"] = wrapper.frozen_parameters()
    RECORD["nca_parameters"] = wrapper.nca.trainable_parameters()

    text = ("🔧 café 日本語 πΩ — The workshop hinge shifts across the guide rail. "
            "A mechanic compares small forces without changing the baseline. ") * 8
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    require(ids.shape[1] >= 128, "Tokenizer did not produce 128 byte positions")
    RECORD["text_byte_units"] = {
        "utf8_bytes": len(text.encode("utf-8")),
        "python_characters": len(text),
        "tokenizer_units": int(ids.shape[1]),
        "contains_multibyte_UTF8": True,
    }
    atol = p["conformance"]["frozen_base_noop_atol"]
    rtol = p["conformance"]["frozen_base_noop_rtol"]
    cache_atol = p["conformance"]["full_vs_cached_prefill_atol"]
    cache_rtol = p["conformance"]["full_vs_cached_prefill_rtol"]

    with torch.inference_mode():
        for count in (16, 64, 128):
            inp = ids[:, :count]
            a = wrapper(input_ids=inp, enabled=False, use_cache=False).logits
            b = wrapper(input_ids=inp, enabled=True, use_cache=False).logits
            compare(f"zero_init_UTF8_{count}", a, b, atol=atol, rtol=rtol)

        for direction in ("left", "right"):
            tokenizer.padding_side = direction
            batch = tokenizer(["🔧 bolt", "日本語: the spring latch slides."],
                              padding=True, return_tensors="pt")
            a = wrapper(**batch, enabled=False, use_cache=False).logits
            b = wrapper(**batch, enabled=True, use_cache=False).logits
            compare(f"zero_init_{direction}_padding_UTF8", a,b,atol=atol,rtol=rtol)
        tokenizer.padding_side = "left"
        prefix = ids[:, :24]
        kwargs = {
            "max_new_tokens": 4, "min_new_tokens": 4,
            "do_sample": False,
            "eos_token_id": 258, "pad_token_id": 256,
            "use_cache": True,
        }
        baseline = wrapper.generate(prefix, enabled=False, **kwargs)
        patched = wrapper.generate(prefix, enabled=True, **kwargs)
        require(torch.equal(baseline, patched), "Zero-init generated byte token IDs drift")
        RECORD["checks"]["zero_init_greedy_generated_ID_parity"] = {
            "equal":True,
            "generated_tokenizer_units":int(patched.shape[-1]-prefix.shape[-1]),
            "eos_id_used":258,
            "eos_reached":bool((patched[0,prefix.shape[-1]:] == 258).any())
        }
        with torch.no_grad():
            wrapper.nca.readout.weight[0,0] = 0.03
        short = ids[:, :48]
        full = wrapper(input_ids=short, use_cache=False).logits
        original = wrapper(input_ids=short, enabled=False, use_cache=False).logits
        effect = float((full-original).abs().max())
        require(effect>0,"Manual nontrained NCA path did not affect Qwen logits")
        RECORD["checks"]["manual_nontrained_readout_effect"] = {
            "max_abs":effect, "trained":False
        }
        past=None
        pieces=[]
        for i in range(short.shape[-1]):
            out=wrapper(input_ids=short[:,i:i+1],past_key_values=past,use_cache=True)
            pieces.append(out.logits[:,-1:])
            past=out.past_key_values
            require(len(wrapper.last_hook_trace)==1,
                    "State not independently reset at each cached byte unit")
        cached=torch.cat(pieces,dim=1)
        compare("nonzero_48_UTF8_full_vs_KV_cache",full,cached,
                atol=cache_atol,rtol=cache_rtol)
        changed=short.clone()
        changed[:,-1]=(changed[:,-1]+1)%256
        before=wrapper(input_ids=short,use_cache=False).logits[:,:-1]
        after=wrapper(input_ids=changed,use_cache=False).logits[:,:-1]
        compare("UTF8_future_suffix_earlier_logits",before,after,
                atol=cache_atol,rtol=cache_rtol)
        require(RECORD["checks"]["UTF8_future_suffix_earlier_logits"]["max_abs"]==0,
                "Later byte changed earlier predictions")
        a = ids[:,:36]
        b = ids[:,24:60]
        before=wrapper(input_ids=a,use_cache=False).logits
        wrapper(input_ids=b,use_cache=False)
        after=wrapper(input_ids=a,use_cache=False).logits
        require(torch.equal(before,after),"A-B-A byte prompt leaked NCA field")
        RECORD["checks"]["request_A_B_A_UTF8_isolation"]={"exact":True}
    save()

    with tempfile.TemporaryDirectory() as td:
        wrapper.save_adapter(td)
        cold=FrozenVoid1DepthNCA(
            base,verified_weight_sha256=source["weight_declared_lfs_sha256_not_downloaded"]
        ).eval()
        cold.load_adapter(td)
        with torch.inference_mode():
            before=wrapper(input_ids=ids[:,:48],use_cache=False).logits
            after=cold(input_ids=ids[:,:48],use_cache=False).logits
        compare("cold_adapter_new_instance_same_frozen_base",
                before,after,atol=0,rtol=0)
        meta_path=Path(td)/"metadata.json"
        meta=json.loads(meta_path.read_text())
        meta["model_revision"]="unverified"
        meta_path.write_text(json.dumps(meta))
        rejected=False
        try:
            cold.load_adapter(td)
        except RuntimeError:
            rejected=True
        require(rejected,"Wrong revision metadata accepted")
        RECORD["checks"]["cold_wrong_checkpoint_metadata_rejected"]={"rejected":True}

    require(all(x.grad is None for x in base.parameters()),
            "Base gradients unexpectedly present")
    RECORD["status"]="PASS_REVIEWED_VOID_UTF8_REAL_MODEL_CACHE_COLD_ADAPTER_PREPASS"
    RECORD["limitations"]=[
        "Fresh reviewer code and exact pinned Qwen3 implementation; same base in cold adapter test, not a new OS process",
        "Tokenizer reviewed/source-hashed, but no claim about language quality or actual EOS propensity",
        "Only zero-init or manually perturbed untrained readout; no optimizer updates",
        "Four-model independent #72 conformance still requires Sprocket and more replay coverage",
        "No held-out evaluation, MLP control quality, or comparative training admission"
    ]
    RECORD["optimizer_steps"]=0
    save()
    print(json.dumps({
        "status":RECORD["status"],
        "checks":RECORD["checks"],
        "text_units":RECORD["text_byte_units"],
        "peak_rss_kib":RECORD["peak_rss_kib"],
        "training":False,
    },indent=2,ensure_ascii=False))


if __name__=="__main__":
    try:
        main()
    except Exception as ex:
        RECORD["status"]="FAILED_REVIEWED_VOID_UTF8_REAL_MODEL_CACHE"
        RECORD["error"]=f"{type(ex).__name__}: {ex}"
        RECORD["traceback"]=traceback.format_exc()[-6500:]
        save()
        raise
