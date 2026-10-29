#!/usr/bin/env python3
"""NCR-001E: fresh separate source-bound Sprocket Chat/Instruct actual-host review.

Runs a frozen model against source-documented ordinary role prompts.
No optimizer steps, no held-out fixtures and no tool/memory evaluation.
The original implementation is checked out at a separately pinned PR head.
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
import tempfile
import traceback

import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

from ncr_001e_sprocket_format import (
    SOURCE_REPO,SOURCE_REV,serialize_dialogue,
    validate_pinned_card,validate_pinned_tokenizer
)

ROOT=Path(__file__).resolve().parents[1]
IMPL=ROOT/"implementation"
PINS=ROOT/"experiments/ncr-001/source-pins.json"
PROTOCOL=ROOT/"experiments/ncr-001/protocol.json"

SOURCE_SPECS={
    "chat":{
        "key":"sprocket_chat",
        "implementation_commit":"83a56835a7476f17a17980d28dd1a7c14bb6f2f6",
        "class":"FrozenSprocketChatDepthNCA",
        "module":"ncr_001_sprocket_chat",
        "weight":"chat/model.safetensors",
        "subfolder":"chat",
        "config":"chat/config.json",
        "config_pin":"chat_config_sha256",
        "tokenizer_json":"chat/tokenizer.json",
        "tokenizer_pin":"chat_tokenizer_sha256",
        "tokenizer_config":"chat/tokenizer_config.json",
        "tokenizer_config_pin":"chat_tokenizer_config_sha256",
    },
    "instruct":{
        "key":"sprocket_instruct",
        "implementation_commit":"b95c010f9b2ba3cbd9f81317981316bb9c8acd19",
        "class":"FrozenSprocketInstructDepthNCA",
        "module":"ncr_001_sprocket_instruct",
        "weight":"model.safetensors",
        "subfolder":None,
        "config":"config.json",
        "config_pin":"config_sha256",
        "tokenizer_json":"tokenizer.json",
        "tokenizer_pin":"tokenizer_sha256",
        "tokenizer_config":"tokenizer_config.json",
        "tokenizer_config_pin":"tokenizer_config_sha256",
    },
}

REPORT={
    "experiment":"NCR-001E",
    "stage":"fresh_pinned_Sprocket_ordinary_role_and_frozen_model_conformance",
    "status":"RUNNING",
    "actual_model_training":False,
    "optimizer_steps":0,
    "comparative_quality_study":False,
    "full_four_model_conformance_pass":False,
    "tool_and_memory_behavior_qualified":False,
    "checks":{},
}


def filehash(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, why: str)->None:
    if not ok:
        raise RuntimeError(why)


def persist(out: Path)->None:
    out.parent.mkdir(parents=True,exist_ok=True)
    REPORT["peak_rss_kib"]=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(REPORT,indent=2,sort_keys=True)+"\n")


def compare(k: str, a: torch.Tensor,b: torch.Tensor, *,atol:float,rtol:float,
            out:Path,exact:bool=False)->None:
    require(a.shape==b.shape,f"{k}: shape drift")
    difference=(a-b).abs()
    obs={
        "max_abs":float(difference.max()),
        "mean_abs":float(difference.mean()),
        "atol":atol,"rtol":rtol,
        "close":bool(torch.allclose(a,b,atol=atol,rtol=rtol)),
        "exact_required":exact,
    }
    REPORT["checks"][k]=obs
    persist(out)
    require(obs["close"] and (not exact or obs["max_abs"]==0),
            f"{k}: actual source-bound logits parity failed: {obs}")


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=sorted(SOURCE_SPECS),required=True)
    args=parser.parse_args()
    variant=args.variant
    spec=SOURCE_SPECS[variant]
    OUT=ROOT/f"runtime/ncr-001/sprocket-{variant}-fresh.json"
    require(os.environ.get("HF_HUB_OFFLINE")=="1" and
            os.environ.get("TRANSFORMERS_OFFLINE")=="1",
            "NCR independent frozen source review must run fully offline")
    torch.set_num_threads(min(4,os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261027)
    p=json.loads(PROTOCOL.read_text())
    allpins=json.loads(PINS.read_text())
    pin=allpins["models"][spec["key"]]
    model=p["models"][spec["key"]]
    require(p["train_authorized"] is False,"NCR training unexpectedly permitted")
    require(pin["repo_id"]==SOURCE_REPO==model["repo_id"] and
            pin["revision"]==SOURCE_REV==model["revision"],
            "Sprocket pinned repo/model/revision mismatch")
    require(model["variant"]==variant and
            pin["variant"]==variant and
            pin["hf_weight_file"]==spec["weight"]==model["weight_file"],
            "Sprocket variant/checkpoint substitution refused")
    sibling=allpins["models"]["sprocket_instruct" if variant=="chat" else "sprocket_chat"]
    require(sibling["weight_declared_lfs_sha256_not_downloaded"]
            !=pin["weight_declared_lfs_sha256_not_downloaded"],
            "Sprocket Chat/Instruct checkpoint collision")
    require(IMPL.is_dir(),"Independent source implementation checkout absent")
    actual_impl=subprocess.check_output(
        ["git","-C",str(IMPL),"rev-parse","HEAD"],text=True).strip()
    require(actual_impl==spec["implementation_commit"],
            "Original model-specific hook implementation revision mismatch")
    sys.path.insert(0,str(IMPL/"scripts"))
    # Independent exact-source import by the reviewed module name, rather
    # than constructing a substitute hook implementation in this reviewer.
    from importlib import import_module
    module=import_module(spec["module"])
    wrapper_class=getattr(module,spec["class"])
    require(module.MODEL_ID==SOURCE_REPO and module.MODEL_REVISION==SOURCE_REV
            and list(module.POST_BLOCK_TAPS)==[7,16,25],
            "Original implementation's model/tap identities differ")
    REPORT.update({
        "variant":variant,
        "model_repository":SOURCE_REPO,
        "model_revision":SOURCE_REV,
        "original_implementation_sha":actual_impl,
        "source_protocol_sha256":filehash(PROTOCOL),
        "source_pins_sha256":filehash(PINS),
        "torch":torch.__version__,
        "transformers":transformers.__version__,
        "runner":{"cpu_threads":torch.get_num_threads(),
                  "platform":platform.platform(),
                  "python":platform.python_version()},
        "source_files":{}
    })
    persist(OUT)

    # All metadata and the entire 1 GB model weight file are *already*
    # prefetched and rehashed online by the workflow before offline execution.
    files=[
        ("README.md",None),
        (spec["config"],spec["config_pin"]),
        (spec["tokenizer_json"],spec["tokenizer_pin"]),
        (spec["tokenizer_config"],spec["tokenizer_config_pin"]),
        (spec["weight"],"weight_declared_lfs_sha256_not_downloaded"),
    ]
    if variant=="instruct":
        files.append(("special_tokens_map.json","special_tokens_map_sha256"))
    for name,pin_key in files:
        file=Path(hf_hub_download(SOURCE_REPO,name,revision=SOURCE_REV,
                                  local_files_only=True))
        observed=filehash(file)
        if pin_key:
            require(observed==pin[pin_key],f"Downloaded Sprocket {variant} SHA mismatch: {name}")
        REPORT["source_files"][name]={
            "sha256":observed,"bytes":file.stat().st_size,
            "pinned":pin_key is not None
        }
        persist(OUT)
    require(REPORT["source_files"][spec["weight"]]["bytes"]==pin["weight_bytes"],
            "Full Sprocket checkpoint file length mismatch")
    model_card=Path(hf_hub_download(SOURCE_REPO,"README.md",revision=SOURCE_REV,
                                    local_files_only=True)).read_text()
    validate_pinned_card(model_card)
    subfolder=spec["subfolder"]
    tokenizer_opts={"revision":SOURCE_REV,"local_files_only":True}
    if subfolder:
        tokenizer_opts["subfolder"]=subfolder
    tok=AutoTokenizer.from_pretrained(SOURCE_REPO,**tokenizer_opts)
    token_inventory=validate_pinned_tokenizer(tok,variant=variant)
    REPORT["format_source"]={
        "model_card_sha256":REPORT["source_files"]["README.md"]["sha256"],
        "variant":variant,
        "role_boundary_ids":{k:token_inventory["controls"][k]["token_ids"]
                             for k in ("<|user|>","<|end|>","<|assistant|>")},
        "control_token_inventory":token_inventory["controls"],
        "tool_and_memory_behavior_qualified":False,
        "source_verified_dialogue":True,
        "no_invented_chat_template":True,
    }
    if tok.pad_token_id is None:
        # Padding is masked, and the model card documents <|end|> turn end.
        # Never silently conflate this value with tool/memory control tokens.
        tok.pad_token_id=token_inventory["controls"]["<|end|>"]["token_ids"][0]
    tok.padding_side="right"
    persist(OUT)

    base_opts={
        "revision":SOURCE_REV,
        "torch_dtype":torch.float32,
        "attn_implementation":"eager",
        "local_files_only":True,
        "trust_remote_code":False,
    }
    if subfolder:
        base_opts["subfolder"]=subfolder
    base=AutoModelForCausalLM.from_pretrained(SOURCE_REPO,**base_opts).eval()
    wrapper=wrapper_class(
        base,verified_weight_sha256=pin["weight_declared_lfs_sha256_not_downloaded"]
    ).eval()
    wrapper.verify_frozen_base()
    require(all(not v.requires_grad for v in base.parameters()),"Frozen source backbone unexpectedly trainable")
    require(wrapper.nca.trainable_parameters()==658904,
            "Sprocket NCA trainable count changed")
    REPORT["frozen_base_parameters"]=wrapper.frozen_parameters()
    REPORT["nca_parameters"]=wrapper.nca.trainable_parameters()
    atol=float(p["conformance"]["frozen_base_noop_atol"])
    rtol=float(p["conformance"]["frozen_base_noop_rtol"])
    catol=float(p["conformance"]["full_vs_cached_prefill_atol"])
    crtol=float(p["conformance"]["full_vs_cached_prefill_rtol"])

    def tokenized(content: str):
        prompt=serialize_dialogue([{"role":"user","content":content}],
                                  variant=variant)
        return tok(prompt,return_tensors="pt",add_special_tokens=False)["input_ids"]

    long_text=("A workshop conveyor jams when the guide rail warms. "
               "Observe the hinge, measure vibration, then record reversible steps. ")*22
    ids=tokenized(long_text)
    require(ids.shape[-1]>=128,"Documented role-prefix tokenizer yielded unexpectedly few units")
    REPORT["format_source"]["long_prompt_tokenizer_units"]=int(ids.shape[-1])

    with torch.inference_mode():
        for length in (16,64,128):
            x=ids[:,:length]
            original=wrapper(input_ids=x,enabled=False,use_cache=False).logits
            injected=wrapper(input_ids=x,enabled=True,use_cache=False).logits
            compare(f"zero_init_{length}_source_role_prefix",
                    original,injected,atol=atol,rtol=rtol,out=OUT,exact=True)
        for side in ("left","right"):
            tok.padding_side=side
            prompts=[
                serialize_dialogue([{"role":"user","content":"A gear rattles."}],variant=variant),
                serialize_dialogue([{"role":"user","content":"The temperature rises. Inspect the loose fastener before replacing parts."}],variant=variant),
            ]
            batch=tok(prompts,padding=True,return_tensors="pt",
                      add_special_tokens=False)
            original=wrapper(**batch,enabled=False,use_cache=False).logits
            injected=wrapper(**batch,enabled=True,use_cache=False).logits
            compare(f"zero_init_{side}_padded_source_roles",original,injected,
                    atol=atol,rtol=rtol,out=OUT,exact=True)
        tok.padding_side="right"

        prompt=tokenized("A seized bearing needs inspection.")
        end_id=token_inventory["controls"]["<|end|>"]["token_ids"][0]
        gen_kw={
            "do_sample":False,"max_new_tokens":4,"min_new_tokens":4,
            "pad_token_id":tok.pad_token_id,
            "eos_token_id":tok.eos_token_id if tok.eos_token_id is not None else end_id,
            "use_cache":True,
        }
        before=wrapper.generate(prompt,enabled=False,**gen_kw)
        after=wrapper.generate(prompt,enabled=True,**gen_kw)
        require(torch.equal(before,after),"Source role-based greedy generation changed under zero NCA")
        REPORT["checks"]["zero_init_greedy_generation"]={
            "identical":True,
            "units_generated":int(after.shape[-1]-prompt.shape[-1]),
            "actual_eos_id":gen_kw["eos_token_id"],
            "source_turn_end_id":end_id,
            "model_behavior_scored":False,
        }
        # Activate one fixed untrained readout scalar solely for conformance.
        with torch.no_grad():
            wrapper.nca.readout.weight[0,0]=0.03
        x=ids[:,:48]
        full=wrapper(input_ids=x,use_cache=False).logits
        baseline=wrapper(input_ids=x,enabled=False,use_cache=False).logits
        effect=float((full-baseline).abs().max())
        require(effect>0,"Manually activated NCA caused no source logit change")
        REPORT["checks"]["manual_nontrained_readout_effect"]={
            "max_abs":effect,"trained":False,
        }
        past=None
        chunks=[]
        for index in range(x.shape[-1]):
            response=wrapper(input_ids=x[:,index:index+1],
                             past_key_values=past,use_cache=True)
            chunks.append(response.logits[:,-1:])
            past=response.past_key_values
            require(len(wrapper.last_hook_trace)==1,
                    "Incremental decoder NCA depth state did not reset")
        incremental=torch.cat(chunks,dim=1)
        compare("nonzero_role_48_full_vs_incremental_cache",
                full,incremental,atol=catol,rtol=crtol,out=OUT)
        altered=x.clone()
        altered[:,-1]=(altered[:,-1]+7)%base.config.vocab_size
        orig=wrapper(input_ids=x,use_cache=False).logits[:,:-1]
        changed=wrapper(input_ids=altered,use_cache=False).logits[:,:-1]
        compare("nonzero_source_role_future_suffix",orig,changed,
                atol=catol,rtol=crtol,out=OUT,exact=True)
        a=ids[:,:32]
        b=ids[:,16:44]
        first=wrapper(input_ids=a,use_cache=False).logits.clone()
        wrapper(input_ids=b,use_cache=False)
        again=wrapper(input_ids=a,use_cache=False).logits
        require(torch.equal(first,again),"Cross-request NCA field leaked")
        REPORT["checks"]["request_A_B_A_no_field_leak"]={"exact":True}
    persist(OUT)

    with tempfile.TemporaryDirectory() as td:
        meta=wrapper.save_adapter(td)
        require(meta["model_variant"]==variant and
                meta["checkpoint_sha256"]==pin["weight_declared_lfs_sha256_not_downloaded"],
                "Adapter lost Sprocket model/checkpoint identity")
        new=wrapper_class(base,verified_weight_sha256=pin["weight_declared_lfs_sha256_not_downloaded"]).eval()
        new.load_adapter(td)
        with torch.inference_mode():
            old=wrapper(input_ids=ids[:,:48],use_cache=False).logits
            restored=new(input_ids=ids[:,:48],use_cache=False).logits
        compare("cold_adapter_new_instance_same_frozen_base",
                old,restored,atol=0,rtol=0,out=OUT,exact=True)
        file=Path(td)/"metadata.json"
        original=json.loads(file.read_text())
        for patch_key,patch_value in (
            ("model_variant","instruct" if variant=="chat" else "chat"),
            ("checkpoint_sha256","0"*64),
            ("model_revision","WRONG_PIN"),
        ):
            mutated=dict(original)
            mutated[patch_key]=patch_value
            file.write_text(json.dumps(mutated))
            rejected=False
            try:
                new.load_adapter(td)
            except RuntimeError:
                rejected=True
            require(rejected,f"Variant/checkpoint/revision source refusal failed: {patch_key}")
            REPORT["checks"][f"cold_{patch_key}_source_refusal"]={"rejected":True}
        file.write_text(json.dumps(original))
    require(all(v.grad is None for v in base.parameters()),
            "No optimizer was run, original base gradients must remain absent")
    REPORT["status"]="PASS_FRESH_SPROCKET_"+variant.upper()+"_SOURCE_ROLES_CACHE_COLD_PREPASS"
    REPORT["source_bound_preliminary_only"]=True
    REPORT["limitations"]=[
        "Dedicated role/control-token ID inventory and documented ordinary dialogue format only",
        "No source-authorized tool-call argument grammar, memory-write/read or tool response semantic conformance",
        "128-unit zero-init and 48-unit manually activated cached replay, not full 2048-context qualification",
        "Cold adapter = newly constructed adapter against the same frozen base/process, not fresh worker/process",
        "No optimizer steps, learned policy, held-out probes or matched MLP quality results",
        "Does not certify four-model NCR-001E final conformance or admit any training",
    ]
    persist(OUT)
    print(json.dumps({
        "status":REPORT["status"],
        "variant":variant,
        "checks":REPORT["checks"],
        "format_source":REPORT["format_source"],
        "peak_rss_kib":REPORT["peak_rss_kib"],
        "training":False,
    },indent=2),flush=True)
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        v=(sys.argv[sys.argv.index("--variant")+1]
           if "--variant" in sys.argv and sys.argv.index("--variant")+1<len(sys.argv)
           else "unknown")
        target=ROOT/f"runtime/ncr-001/sprocket-{v}-fresh.json"
        REPORT["status"]="FAILED_FRESH_SPROCKET_SOURCE_OR_MODEL_CONFORMANCE"
        REPORT["error"]=f"{type(exc).__name__}: {exc}"
        REPORT["traceback"]=traceback.format_exc()[-6500:]
        persist(target)
        raise
