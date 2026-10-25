#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, re, resource, sys, time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
import psutil, torch, transformers, peft
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/"datasets"/"train-003"
BASE_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
BASE_REVISION="12fd25f77366fa6b3b4b768ec3050bf629380bac"
SEED=20261025
KEYS=("decision","trigger","state","update","action","failure")
CORE=("trigger","state","update","action","failure")
LABELS=("VALID","REPAIR","INTERESTING_BROKEN","REJECT")
EVALUATOR_VERSION=2
NONLOCAL=("global map","whole world","every place","omniscient","future knowledge","all enemies","all resources")
UPDATE=("store","set","decay","increase","raise","increment","decrement","reset","clear","choose","accumulate","advance")
ACTION=("move","turn","steer","jump","bias","suppress","apply","perform","wait","resume","curve","reduce","increase","widen","face","stop")
STATE_ACTION=(" move "," turn "," steer "," jump "," bias "," suppress "," apply "," perform "," wait "," resume "," curve ")
STOP={"the","and","for","from","into","while","with","when","then","this","that","only","local","bounded","scalar","unit","vector","above","below","after","before","toward","away","high","low","same","once","each","zero"}

def sha_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def sha_file(p:Path)->str: return sha_bytes(p.read_bytes())
def write_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"): return line.split(":",1)[1].strip()
    except Exception: pass
    return platform.processor() or "unknown"
def packages():
    import huggingface_hub
    return {"python":sys.version.split()[0],"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,"huggingface_hub":huggingface_hub.__version__,"psutil":psutil.__version__}
def configure():
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    torch.set_num_threads(min(4,os.cpu_count() or 4))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    random.seed(SEED); torch.manual_seed(SEED)

def manifest(): return json.loads((DATASET/"manifest.json").read_text())
def jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def train_rows():
    m=manifest(); rows=[]
    for rel in m["training_shards"]: rows+=jsonl(DATASET/rel)
    if len(rows)!=m["train_examples"]: raise RuntimeError("training count mismatch")
    return rows
def heldout_rows():
    m=manifest(); rows=jsonl(DATASET/m["heldout_shard"])
    if len(rows)!=m["heldout_examples"]: raise RuntimeError("heldout count mismatch")
    return rows
def dataset_id():
    m=manifest(); shards=m["training_shards"]+[m["heldout_shard"]]
    return {"manifest_sha256":sha_file(DATASET/"manifest.json"),"source_sha256":sha_file(DATASET/m["source_spec"]),"prompt_sha256":sha_file(DATASET/m["system_prompt"]),"shard_sha256":{x:sha_file(DATASET/x) for x in shards},"train_examples":m["train_examples"],"heldout_examples":m["heldout_examples"]}

def base():
    tok=AutoTokenizer.from_pretrained(BASE_MODEL,revision=BASE_REVISION)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(BASE_MODEL,revision=BASE_REVISION,torch_dtype=torch.float32)
    model.config.use_cache=False
    return model,tok
def targets(model):
    wanted=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    have={n.rsplit(".",1)[-1] for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)}
    out=[x for x in wanted if x in have]
    if not out: raise RuntimeError("no LoRA targets")
    return out
def new_lora():
    model,tok=base(); ts=targets(model)
    cfg=LoraConfig(task_type="CAUSAL_LM",r=8,lora_alpha=16,lora_dropout=.05,bias="none",target_modules=ts)
    return get_peft_model(model,cfg),tok,ts
def load_lora(checkpoint:Path):
    meta=json.loads((checkpoint/"metadata.json").read_text())
    if meta.get("base_model")!=BASE_MODEL or meta.get("base_revision")!=BASE_REVISION: raise RuntimeError("TRAIN-001 base identity mismatch")
    model,tok=base()
    return PeftModel.from_pretrained(model,checkpoint/"adapter",is_trainable=False),tok,meta

def compact(o): return json.dumps(o,separators=(",",":"),ensure_ascii=False)
def encode(tok,system,row,maxlen=384):
    prompt=[{"role":"system","content":system},{"role":"user","content":row["input"]}]
    full=prompt+[{"role":"assistant","content":compact(row["output"])}]
    p=tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
    f=tok.apply_chat_template(full,tokenize=True,add_generation_prompt=False)
    common=0
    for a,b in zip(p,f):
        if a!=b: break
        common+=1
    if len(f)>maxlen: raise RuntimeError(f"example too long: {len(f)}")
    labels=[-100]*common+f[common:]
    return {"input_ids":torch.tensor([f]),"attention_mask":torch.ones((1,len(f)),dtype=torch.long),"labels":torch.tensor([labels]),"tokens":len(f)}

def extract(text):
    a=text.find("{"); b=text.rfind("}")
    if a<0 or b<=a: return None
    try: x=json.loads(text[a:b+1]); return x if isinstance(x,dict) else None
    except json.JSONDecodeError: return None
def terms(text):
    raw=re.findall(r"[a-z][a-z0-9_]*",text.lower()); out=set()
    for t in raw:
        if t in STOP or len(t)<3: continue
        for s in ("ing","ed","es","s"):
            if t.endswith(s) and len(t)-len(s)>=4: t=t[:-len(s)]; break
        out.add(t)
    return out
def cover(actual,expected):
    want=terms(expected)
    return 1.0 if not want else len(want&terms(actual))/len(want)
def has(text,items): return any(x in f" {text.lower()} " for x in items)

def score(text,row):
    p=extract(text); expected=row["output"]; decision=expected["decision"]
    if p is None: return {"json":False,"exact_schema":False,"core_schema":False,"decision_correct":False,"mechanical_score":0.0,"overall_score":0.0,"components":{}}
    exact=tuple(p.keys())==KEYS; core=all(k in p for k in CORE); dc=str(p.get("decision","")).strip()==decision
    if decision=="REJECT":
        signal=" ".join(str(p.get(k,"")) for k in ("decision","trigger","failure")).lower()
        components={"recognizes_nonlocal":float(dc or any(x in signal for x in ("reject","nonlocal","non-local","omniscient","unavailable"))),"no_action":float(str(p.get("action","")).strip().lower() in ("","none","no action","reject")),"no_state_update":float(str(p.get("state","")).strip().lower() in ("","none","no state","reject") and str(p.get("update","")).strip().lower() in ("","none","no update","reject"))}
    else:
        vals={k:str(p.get(k,"")) for k in CORE}
        components={k:cover(vals[k],str(expected[k])) for k in CORE}
        components["separation"]=mean((float(not has(vals["state"],STATE_ACTION)),float(has(vals["update"],UPDATE)),float(has(vals["action"],ACTION))))
        components["local_only"]=float(not any(x in " ".join(vals.values()).lower() for x in NONLOCAL))
        components["interesting_preservation"]=1.0 if decision!="INTERESTING_BROKEN" else cover(vals["action"]+" "+vals["failure"],str(expected["action"])+" "+str(expected["failure"]))
    mech=mean(components.values())
    overall=.20*float(dc)+.10*float(exact)+.70*mech
    return {"json":True,"exact_schema":exact,"core_schema":core,"decision_correct":dc,"mechanical_score":mech,"overall_score":overall,"components":components,"parsed":p}

@torch.inference_mode()
def evaluate(model,tok,system,rows):
    model.eval(); out=[]
    for i,row in enumerate(rows):
        msgs=[{"role":"system","content":system},{"role":"user","content":row["input"]}]
        ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors="pt")
        t=time.monotonic(); gen=model.generate(ids,max_new_tokens=180,do_sample=False,pad_token_id=tok.eos_token_id)
        new=gen[0,ids.shape[-1]:]; text=tok.decode(new,skip_special_tokens=True).strip()
        out.append({"id":row["id"],"family":row["family"],"expected_decision":row["output"]["decision"],"input":row["input"],"expected":row["output"],"output":text,"generated_tokens":int(new.numel()),"elapsed_seconds":time.monotonic()-t,"score":score(text,row)})
        if (i+1)%16==0: print(f"evaluated {i+1}/{len(rows)}",flush=True)
    def metrics(xs):
        return {"count":len(xs),"json_rate":mean(float(x["score"]["json"]) for x in xs),"exact_schema_rate":mean(float(x["score"]["exact_schema"]) for x in xs),"core_schema_rate":mean(float(x["score"]["core_schema"]) for x in xs),"decision_accuracy":mean(float(x["score"]["decision_correct"]) for x in xs),"mechanical_score":mean(float(x["score"]["mechanical_score"]) for x in xs),"overall_score":mean(float(x["score"]["overall_score"]) for x in xs)}
    result=metrics(out); groups=defaultdict(list)
    for x in out: groups[x["expected_decision"]].append(x)
    result["by_decision"]={k:metrics(groups[k]) for k in LABELS}; result["rows"]=out
    return result

def inventory(path):
    files=[]
    for p in sorted(path.rglob("*")):
        if p.is_file(): files.append({"path":p.relative_to(path).as_posix(),"bytes":p.stat().st_size,"sha256":sha_file(p)})
    return {"files":files,"bytes":sum(x["bytes"] for x in files)}

def common(identity):
    return {"schema_version":1,"experiment":"TRAIN-003","identity":identity,"status":"running","source_commit":os.environ.get("TRAIN003_SOURCE_SHA") or os.environ.get("GITHUB_SHA"),"base_model":BASE_MODEL,"base_revision":BASE_REVISION,"evaluator_version":EVALUATOR_VERSION,"dataset":dataset_id(),"runner":{"cpu_model":cpu_model(),"logical_cpu_count":os.cpu_count()},"packages":packages()}

def run_eval(a):
    configure(); ev=common(a.identity); write_json(a.evidence_out,ev)
    model,tok,meta=load_lora(a.adapter_checkpoint)
    ev["checkpoint_metadata"]=meta; ev["checkpoint_adapter_inventory"]=inventory(a.adapter_checkpoint/"adapter")
    system=(DATASET/"prompts/system.txt").read_text().strip(); ev["evaluation"]=evaluate(model,tok,system,heldout_rows()); ev["status"]="success"; write_json(a.evidence_out,ev)
    print(json.dumps({k:v for k,v in ev["evaluation"].items() if k!="rows"},indent=2)); return 0

def run_train(a):
    configure(); ev=common("TRAIN-003-semantic-LoRA"); write_json(a.evidence_out,ev)
    model,tok,ts=new_lora(); ev["lora"]={"r":8,"alpha":16,"dropout":.05,"targets":ts}
    train=train_rows(); random.Random(SEED).shuffle(train); system=(DATASET/"prompts/system.txt").read_text().strip()
    enc=[encode(tok,system,x) for x in train]
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=8e-4,betas=(.9,.95),weight_decay=.01)
    losses=[]; tokens=0; started=time.monotonic(); model.train()
    for i in range(a.steps):
        ex=enc[i%len(enc)]; opt.zero_grad(set_to_none=True)
        loss=model(input_ids=ex["input_ids"],attention_mask=ex["attention_mask"],labels=ex["labels"]).loss; loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step()
        losses.append(float(loss.detach())); tokens+=ex["tokens"]
        if (i+1)%96==0: print(f"trained {i+1}/{a.steps} loss={losses[-1]:.4f}",flush=True)
    seconds=time.monotonic()-started; a.adapter_out.mkdir(parents=True,exist_ok=True); model.save_pretrained(a.adapter_out,safe_serialization=True)
    ev["training"]={"steps":a.steps,"examples_available":len(train),"sequence_tokens":tokens,"train_seconds":seconds,"sequence_tokens_per_second":tokens/seconds,"first_loss":losses[0],"last_loss":losses[-1],"mean_loss":mean(losses),"peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    ev["adapter"]=inventory(a.adapter_out); ev["evaluation"]=evaluate(model,tok,system,heldout_rows()); ev["status"]="success"; write_json(a.evidence_out,ev)
    print(json.dumps({"training":ev["training"],"evaluation":{k:v for k,v in ev["evaluation"].items() if k!="rows"}},indent=2)); return 0

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    e=sub.add_parser("evaluate"); e.add_argument("--adapter-checkpoint",type=Path,required=True); e.add_argument("--identity",required=True); e.add_argument("--evidence-out",type=Path,required=True)
    t=sub.add_parser("train"); t.add_argument("--steps",type=int,default=768); t.add_argument("--adapter-out",type=Path,required=True); t.add_argument("--evidence-out",type=Path,required=True)
    a=ap.parse_args()
    if a.cmd=="train" and not 1<=a.steps<=1536: raise SystemExit("--steps must be 1..1536")
    return run_eval(a) if a.cmd=="evaluate" else run_train(a)
if __name__=="__main__": raise SystemExit(main())
