#!/usr/bin/env python3
"""NCR-001E: fresh reviewer-owned SmolLM2 read-only cold-reload/cache invocation.

Loads PR #83's exact implementation commit via a second checkout in
implementation/; no edit to that model adapter or the original base.
This is NOT full independent four-model #72 conformance or model training.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "implementation"
REQUIRED_IMPLEMENTATION_COMMIT = "ec0d4a80e15f203a06c26ee9c23858f78976370b"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
OUT = ROOT / "runtime/ncr-001/smol-independent-fresh.json"

RESULT = {
    "schema_version": 1,
    "experiment": "NCR-001E",
    "stage": "fresh_smolllm2_source_bound_longer_masked_cache_and_cold_adapter_review",
    "status": "RUNNING",
    "training_performed": False,
    "optimizer_steps": 0,
    "model_quality_test_performed": False,
    "comparative_training_admitted": False,
    "full_four_model_conformance_pass": False,
    "checks": {},
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for b in iter(lambda: src.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def persist() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    OUT.write_text(json.dumps(RESULT, indent=2, sort_keys=True)+"\n")


def validate_pair(key: str, a: torch.Tensor, b: torch.Tensor, *,
                  atol: float, rtol: float) -> None:
    require(a.shape == b.shape, f"{key}: logits shapes differ")
    diff = (a-b).abs()
    metrics = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "atol": atol, "rtol": rtol,
        "within": bool(torch.allclose(a,b,atol=atol,rtol=rtol)),
    }
    RESULT["checks"][key] = metrics
    if not metrics["within"]:
        raise AssertionError(f"{key}: {metrics}")


def require(flag: bool, why: str) -> None:
    if not flag:
        raise RuntimeError(why)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261027)
    p=json.loads(PROTOCOL.read_text())
    pin=json.loads(PINS.read_text())["models"]["smollm2"]
    require(p["train_authorized"] is False,
            "Training admission unexpectedly open in source protocol")
    require(IMPL.exists(), "Pinned implementation checkout not available")
    sha = subprocess.check_output(
        ["git","-C",str(IMPL),"rev-parse","HEAD"],text=True
    ).strip()
    require(sha==REQUIRED_IMPLEMENTATION_COMMIT,
            f"Unexpected hooked-model source: {sha}")

    # Do NOT use model code from an ambient pip installation. Bind the exact PR.
    sys.path.insert(0,str(IMPL/"scripts"))
    from ncr_001_smollm2 import (  # type: ignore
        FrozenSmolLM2DepthNCA, MODEL_ID, MODEL_REVISION, POST_BLOCK_TAPS,
    )
    require(MODEL_ID == pin["repo_id"]
            and MODEL_REVISION == pin["revision"]
            and pin["hf_weight_file"] == "model.safetensors",
            "SmolLM2 source identity mismatch")
    require(list(POST_BLOCK_TAPS)==p["models"]["smollm2"]["taps_after_block_zero_based"],
            "Tap geometry changed from preregistration")

    RESULT.update({
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "integration_commit": sha,
        "source_protocol_sha256": file_sha(PROTOCOL),
        "source_pins_sha256": file_sha(PINS),
        "runner": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
            "python":sys.version.split()[0],
        }
    })
    persist()
    weight_path=Path(hf_hub_download(MODEL_ID,filename="model.safetensors",revision=MODEL_REVISION))
    weight_hash=file_sha(weight_path)
    require(weight_hash==pin["weight_declared_lfs_sha256_not_downloaded"],
            "Actual downloaded source safetensors SHA does not match pinned source")
    RESULT["actual_full_model_safetensors"]={
        "sha256":weight_hash,"bytes":weight_path.stat().st_size
    }
    persist()

    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,revision=MODEL_REVISION,
        torch_dtype=torch.float32,attn_implementation="eager",
    )
    base.eval()
    adapter=FrozenSmolLM2DepthNCA(base).eval()
    adapter.verify_frozen_base()
    RESULT["base_params"]=adapter.frozen_parameters()
    RESULT["adapter_params"]=adapter.nca.trainable_parameters()
    require(RESULT["adapter_params"]==297048,"NCA param count mismatch")
    require(all(not w.requires_grad for w in base.parameters()),"Unfrozen base")
    atol=p["conformance"]["frozen_base_noop_atol"]
    rtol=p["conformance"]["frozen_base_noop_rtol"]
    c_atol=p["conformance"]["full_vs_cached_prefill_atol"]
    c_rtol=p["conformance"]["full_vs_cached_prefill_rtol"]

    prose=("A small workshop hinge slips only after the spring warms. "
           "A technician checks the guide rail, then the fasteners, "
           "and writes down the observations before changing any parts. ") * 16
    all_ids=tokenizer(prose,return_tensors="pt")["input_ids"]
    require(all_ids.shape[1]>=128,"Unexpectedly short tokenizer source text")
    with torch.inference_mode():
        for length in (16,64,128):
            ids=all_ids[:,:length]
            before=adapter(input_ids=ids,use_cache=False,enabled=False).logits
            after=adapter(input_ids=ids,use_cache=False,enabled=True).logits
            validate_pair(f"zero_init_longer_{length}",before,after,atol=atol,rtol=rtol)
            require(len(adapter.last_hook_trace)==1,
                    f"Missing forward reset on {length}-unit request")

        for side in ("left","right"):
            tokenizer.padding_side=side
            batch=tokenizer([
                "The small fastener rattles.",
                "A spring-loaded workshop drawer shifts when the guide rail warms.",
            ],padding=True,return_tensors="pt")
            before=adapter(**batch,use_cache=False,enabled=False).logits
            after=adapter(**batch,use_cache=False,enabled=True).logits
            validate_pair(f"zero_init_{side}_padding_batch2",before,after,
                          atol=atol,rtol=rtol)
        tokenizer.padding_side="right"

        short=all_ids[:,:48]
        before=adapter.generate(
            input_ids=short[:,:10], enabled=False,
            max_new_tokens=5,do_sample=False,
            pad_token_id=tokenizer.eos_token_id,use_cache=True)
        after=adapter.generate(
            input_ids=short[:,:10], enabled=True,
            max_new_tokens=5,do_sample=False,
            pad_token_id=tokenizer.eos_token_id,use_cache=True)
        require(torch.equal(before,after),"Zero-init generation changed tokens")
        RESULT["checks"]["zero_init_generate_5_tokens"]={
            "identical": True,
            "instrumented_decoder_forwards":len(adapter.last_hook_trace)
        }

        # Carefully labelled diagnostic activation, *not* an optimizer update.
        with torch.no_grad():
            adapter.nca.readout.weight[0,0] = 0.03
        original=adapter(input_ids=short,use_cache=False,enabled=False).logits
        full=adapter(input_ids=short,use_cache=False).logits
        effect=float((full-original).abs().max())
        require(effect>0,"NCA path has no nonzero effect")
        RESULT["checks"]["manual_untrained_effect_max_abs"]=effect
        past=None
        logits=[]
        for idx in range(short.shape[1]):
            out=adapter(input_ids=short[:,idx:idx+1],
                        past_key_values=past,use_cache=True)
            past=out.past_key_values
            logits.append(out.logits[:,-1:])
            require(len(adapter.last_hook_trace)==1,
                    "Incremental decoder failed to reset field state exactly once")
        cached=torch.cat(logits,dim=1)
        validate_pair("nonzero_48_full_vs_incremental_KV",
                      full,cached,atol=c_atol,rtol=c_rtol)
        altered=short.clone()
        altered[:,-1]=(altered[:,-1]+11)%base.config.vocab_size
        old=adapter(input_ids=short,use_cache=False).logits[:,:-1]
        new=adapter(input_ids=altered,use_cache=False).logits[:,:-1]
        validate_pair("nonzero_future_suffix_isolation",old,new,
                      atol=c_atol,rtol=c_rtol)
        require(RESULT["checks"]["nonzero_future_suffix_isolation"]["max_abs"]==0,
                "Suffix changed earlier logits")

        a=short[:,:24]
        b=short[:,12:36]
        first=adapter(input_ids=a,use_cache=False).logits.clone()
        adapter(input_ids=b,use_cache=False)
        again=adapter(input_ids=a,use_cache=False).logits
        require(torch.equal(first,again),
                "Cellular field leaked from unrelated request into same prompt")
        RESULT["checks"]["request_A_B_A_isolation"]={"exact":True}
    persist()

    # Independent cold construction of adapter state, reusing the exact frozen
    # backbone within a single process to avoid doubling 135M model memory.
    with tempfile.TemporaryDirectory() as td:
        meta=adapter.save_adapter(td)
        require(meta["model_revision"]==MODEL_REVISION,
                "Saved adapter changed model revision")
        cold=FrozenSmolLM2DepthNCA(base).eval()
        cold.load_adapter(td)
        with torch.inference_mode():
            current=adapter(input_ids=short,use_cache=False).logits
            restored=cold(input_ids=short,use_cache=False).logits
        validate_pair("cold_adapter_reinstantiation_same_frozen_base",
                      current,restored,atol=0,rtol=0)
        path=Path(td)/"metadata.json"
        original_metadata=path.read_text()
        changed=json.loads(original_metadata)
        changed["model_revision"]="bad-revision"
        path.write_text(json.dumps(changed))
        rejected=False
        try:
            cold.load_adapter(td)
        except RuntimeError:
            rejected=True
        require(rejected,"Cold load failed to reject wrong model revision")
        RESULT["checks"]["cold_checkpoint_variant_refusal"]={"rejected":True}
    require(all(w.grad is None for w in base.parameters()),
            "Source backbone unexpectedly has gradients")
    RESULT["status"]="PASS_REVIEWER_FRESH_SMOLLM2_FROZEN_CACHE_COLD_ADAPTER_PREPASS"
    RESULT["independent_full_four_model_conformance_pass"]=False
    RESULT["limitations"]=[
        "Actual independent source-bound SmolLM2 rerun, but only one of four model variants",
        "Adapter cold reload uses a newly constructed adapter on the same original frozen backbone, not an entirely fresh OS process",
        "Nonzero readout was manually set, not trained; no optimizer or held-out model scoring",
        "No Sprocket Chat/Instruct tool/template or void ByteTokenizer qualification",
        "Further full #72 independent longer-run/restart review is still required",
    ]
    persist()
    print(json.dumps({
        "status":RESULT["status"],
        "checks":RESULT["checks"],
        "actual_weight_sha256":weight_hash,
        "peak_rss_kib":RESULT["peak_rss_kib"],
        "training_performed":False,
    },indent=2),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        RESULT["status"]="FAILED_REVIEWER_FRESH_SMOLLM2_SOURCE_CACHE_COLD_PREPASS"
        RESULT["error"]=f"{type(exc).__name__}: {exc}"
        RESULT["traceback"]=traceback.format_exc()[-6000:]
        persist()
        raise
