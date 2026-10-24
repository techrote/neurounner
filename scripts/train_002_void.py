#!/usr/bin/env python3
from __future__ import annotations
import json, os, platform, random, resource, sys, time
from pathlib import Path
import psutil, torch, transformers, peft
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
BASE="dotlabs/void.1"
DATA=ROOT/"experiments"/"train-001"/"kernel_compiler_dataset.json"
OUT=ROOT/"results"
SEED=20261024
KEYS=["trigger","state","update","action","failure"]

def cpu_model():
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"): return line.split(":",1)[1].strip()
    return "unknown"

def prompt(system,item):
    return f"""Task: {system}
Input behaviour idea:
{item['input']}
Output JSON:
"""

def target(item):
    return json.dumps(item["output"],ensure_ascii=False,separators=(",",":"))

def encode(tok,system,item,maxlen=256):
    p=prompt(system,item)
    pi=tok(p,add_special_tokens=True)["input_ids"]
    fi=tok(p+target(item)+(tok.eos_token or ""),add_special_tokens=True)["input_ids"]
    common=0
    for a,b in zip(pi,fi):
        if a!=b: break
        common+=1
    if len(fi)>maxlen: raise RuntimeError(f"too long {len(fi)}")
    labels=[-100]*common+fi[common:]
    return {
      "input_ids":torch.tensor([fi]),
      "attention_mask":torch.ones((1,len(fi)),dtype=torch.long),
      "labels":torch.tensor([labels]),
      "tokens":len(fi)
    }

def parse_json(text):
    a=text.find("{"); b=text.rfind("}")
    if a<0 or b<=a:return None
    try:
      o=json.loads(text[a:b+1]); return o if isinstance(o,dict) else None
    except: return None

def score(text):
    o=parse_json(text)
    return {"json":o is not None,"exact":bool(o is not None and set(o)==set(KEYS) and len(o)==5),
            "keys":sum(k in o for k in KEYS) if o else 0}

@torch.inference_mode()
def evaluate(model,tok,system,items):
    model.eval(); rows=[]
    for item in items:
      p=prompt(system,item); ids=tok(p,return_tensors="pt")
      t=time.monotonic()
      out=model.generate(**ids,max_new_tokens=120,do_sample=False,pad_token_id=tok.eos_token_id)
      elapsed=time.monotonic()-t
      gen=out[0,ids["input_ids"].shape[-1]:]
      text=tok.decode(gen,skip_special_tokens=True).strip()
      rows.append({"input":item["input"],"expected":item["output"],"output":text,"score":score(text),
                   "tokens":int(gen.numel()),"elapsed_seconds":elapsed})
    return {"rows":rows,"json_rate":sum(r["score"]["json"] for r in rows)/len(rows),
            "exact_schema_rate":sum(r["score"]["exact"] for r in rows)/len(rows),
            "mean_expected_key_count":sum(r["score"]["keys"] for r in rows)/len(rows)}

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    torch.set_num_threads(min(4,os.cpu_count() or 4))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    random.seed(SEED); torch.manual_seed(SEED)
    data=json.loads(DATA.read_text()); system=data["system"]
    rev=HfApi().model_info(BASE).sha
    t0=time.monotonic()
    tok=AutoTokenizer.from_pretrained(BASE,revision=rev,trust_remote_code=True)
    if tok.pad_token_id is None:
      tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(BASE,revision=rev,torch_dtype=torch.float32)
    base.config.use_cache=False
    targets=[]
    for n,m in base.named_modules():
      s=n.rsplit(".",1)[-1]
      if s in {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"} and isinstance(m,torch.nn.Linear):
        targets.append(s)
    targets=sorted(set(targets))
    if not targets: raise RuntimeError("no LoRA targets found")
    model=get_peft_model(base,LoraConfig(task_type="CAUSAL_LM",r=8,lora_alpha=16,lora_dropout=.05,bias="none",target_modules=targets))
    load_seconds=time.monotonic()-t0
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=8e-4,betas=(.9,.95),weight_decay=.01)
    before=evaluate(model,tok,system,data["eval"])
    encoded=[encode(tok,system,x) for x in data["train"]]
    losses=[]; train_tokens=0; cursor=0
    model.train(); t1=time.monotonic()
    for step in range(48):
      ex=encoded[cursor%len(encoded)];cursor+=1
      opt.zero_grad(set_to_none=True)
      o=model(input_ids=ex["input_ids"],attention_mask=ex["attention_mask"],labels=ex["labels"])
      o.loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step()
      losses.append(float(o.loss.detach()));train_tokens+=ex["tokens"]
    train_seconds=time.monotonic()-t1
    after=evaluate(model,tok,system,data["eval"])
    adapter=ROOT/"train002"/"void-adapter";adapter.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(adapter,safe_serialization=True)
    bytes_=sum(p.stat().st_size for p in adapter.rglob("*") if p.is_file())
    result={
      "experiment":"TRAIN-002","base_model":BASE,"base_revision":rev,
      "runner":{"cpu_model":cpu_model(),"cpu_count":os.cpu_count(),"platform":platform.platform()},
      "packages":{"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__},
      "load_seconds":load_seconds,"total_parameters":total,"trainable_parameters":trainable,
      "trainable_fraction":trainable/total,"lora_targets":targets,"steps":48,
      "train_tokens":train_tokens,"train_seconds":train_seconds,
      "train_tokens_per_second":train_tokens/train_seconds,"first_loss":losses[0],"last_loss":losses[-1],
      "mean_loss":sum(losses)/len(losses),"peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
      "adapter_bytes":bytes_,"evaluation_before":before,"evaluation_after":after
    }
    OUT.mkdir(exist_ok=True)
    (OUT/"train002-void.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
