#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]

def cpu_model():
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":",1)[1].strip()
    return platform.processor() or "unknown"

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def directory_hash(path:Path)->str:
    h=hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        rel=p.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4,"big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()

def choose_lora_targets(model):
    desired=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    present=set()
    for name,module in model.named_modules():
        suffix=name.rsplit(".",1)[-1]
        if suffix in desired and isinstance(module,torch.nn.Linear):
            present.add(suffix)
    targets=[x for x in desired if x in present]
    if not targets:
        raise RuntimeError("no LoRA target modules found")
    return targets

def load_rows(gid:str):
    path=ROOT/"datasets"/"babble-002"/"training"/f"{gid}.jsonl"
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows)!=100:
        raise RuntimeError(f"{gid}: expected 100 rows, got {len(rows)}")
    if len({x["example_id"] for x in rows})!=100:
        raise RuntimeError(f"{gid}: duplicate IDs")
    if any(x["source_guide_id"]!=gid for x in rows):
        raise RuntimeError(f"{gid}: lineage mismatch")
    return path,rows

def encode_example(tok,system,row,max_length):
    prompt=[
        {"role":"system","content":system},
        {"role":"user","content":row["input"]},
    ]
    full=prompt+[{"role":"assistant","content":row["target"]}]
    prompt_ids=tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
    full_ids=tok.apply_chat_template(full,tokenize=True,add_generation_prompt=False)
    common=0
    for a,b in zip(prompt_ids,full_ids):
        if a!=b:
            break
        common+=1
    if len(full_ids)>max_length:
        raise RuntimeError(f"{row['example_id']}: length {len(full_ids)} > {max_length}")
    labels=[-100]*common+full_ids[common:]
    return {
        "input_ids":torch.tensor([full_ids],dtype=torch.long),
        "attention_mask":torch.ones((1,len(full_ids)),dtype=torch.long),
        "labels":torch.tensor([labels],dtype=torch.long),
        "sequence_tokens":len(full_ids),
        "supervised_tokens":len(full_ids)-common,
    }

def schedule(n,epochs,seed):
    out=[]
    for epoch in range(epochs):
        order=list(range(n))
        random.Random(seed+epoch*100003).shuffle(order)
        out.extend(order)
    return out

@torch.inference_mode()
def evaluate(model,tok,system,scenarios,max_new_tokens):
    old=getattr(model.config,"use_cache",None)
    model.config.use_cache=True
    model.eval()
    rows=[]
    for item in scenarios:
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":item["prompt"]},
        ]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt")
        t0=time.monotonic()
        out=model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        elapsed=time.monotonic()-t0
        gen=out[0,ids.shape[-1]:]
        text=tok.decode(gen,skip_special_tokens=True).strip()
        rows.append({
            "scenario_id":item["id"],
            "family":item["family"],
            "prompt":item["prompt"],
            "response":text,
            "generated_tokens":int(gen.numel()),
            "elapsed_seconds":elapsed,
            "tokens_per_second":int(gen.numel())/elapsed if elapsed else None,
        })
    model.config.use_cache=old
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--persona",required=True)
    ap.add_argument("--base-only",action="store_true")
    args=ap.parse_args()

    cfg=json.loads(args.config.read_text())
    gid=args.persona
    if not args.base_only and gid not in cfg["personas"]:
        raise SystemExit(f"unknown persona {gid}")

    base_model=cfg["base_model"]
    base_revision=cfg["base_revision"]
    train_cfg=cfg["training"]
    eval_cfg=cfg["eval"]
    seed=int(train_cfg["seed"])

    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    torch.set_num_threads(min(4,os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(seed)
    torch.manual_seed(seed)

    t_load=time.monotonic()
    tok=AutoTokenizer.from_pretrained(base_model,revision=base_revision)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(base_model,revision=base_revision,torch_dtype=torch.float32)
    base.config.use_cache=False

    train_path=None
    train_rows=[]
    losses=[]
    seq_tokens=0
    sup_tokens=0
    train_seconds=0.0
    adapter_info=None
    targets=[]
    trainable=0
    total=sum(p.numel() for p in base.parameters())

    if args.base_only:
        model=base
    else:
        targets=choose_lora_targets(base)
        model=get_peft_model(base,LoraConfig(
            task_type="CAUSAL_LM",
            r=int(train_cfg["lora_r"]),
            lora_alpha=int(train_cfg["lora_alpha"]),
            lora_dropout=float(train_cfg["lora_dropout"]),
            bias="none",
            target_modules=targets,
        ))
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_path,train_rows=load_rows(gid)
        encoded=[encode_example(tok,eval_cfg["system"],x,int(train_cfg["max_length"])) for x in train_rows]
        order=schedule(len(encoded),int(train_cfg["epochs"]),seed)
        opt=torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(train_cfg["learning_rate"]),
            betas=(0.9,0.95),
            weight_decay=0.01,
        )
        model.train()
        t0=time.monotonic()
        for pos in order:
            ex=encoded[pos]
            opt.zero_grad(set_to_none=True)
            out=model(
                input_ids=ex["input_ids"],
                attention_mask=ex["attention_mask"],
                labels=ex["labels"],
            )
            loss=out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step()
            losses.append(float(loss.detach()))
            seq_tokens+=ex["sequence_tokens"]
            sup_tokens+=ex["supervised_tokens"]
        train_seconds=time.monotonic()-t0

        adapter_dir=ROOT/"persona001"/gid/"adapter"
        adapter_dir.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(adapter_dir,safe_serialization=True)
        adapter_info={
            "path":str(adapter_dir.relative_to(ROOT)),
            "bytes":sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file()),
            "sha256":directory_hash(adapter_dir),
        }

    load_seconds=time.monotonic()-t_load
    responses=evaluate(model,tok,eval_cfg["system"],eval_cfg["scenarios"],int(eval_cfg["max_new_tokens"]))

    result={
        "experiment":"PERSONA-001",
        "issue":3,
        "persona":"BASE" if args.base_only else gid,
        "base_only":bool(args.base_only),
        "base_model":base_model,
        "base_revision":base_revision,
        "training_config":train_cfg if not args.base_only else None,
        "training_dataset":str(train_path.relative_to(ROOT)) if train_path else None,
        "training_dataset_sha256":sha256_file(train_path) if train_path else None,
        "training_example_ids":[x["example_id"] for x in train_rows],
        "runner":{
            "cpu_model":cpu_model(),
            "cpu_count":os.cpu_count(),
            "platform":platform.platform(),
        },
        "packages":{
            "python":sys.version.split()[0],
            "torch":torch.__version__,
            "transformers":transformers.__version__,
            "peft":peft.__version__,
        },
        "load_seconds":load_seconds,
        "total_parameters":total,
        "trainable_parameters":trainable,
        "trainable_fraction":trainable/total if total else 0,
        "lora_targets":targets,
        "training":{
            "epochs":int(train_cfg["epochs"]) if not args.base_only else 0,
            "optimizer_steps":len(losses),
            "sequence_tokens":seq_tokens,
            "supervised_tokens":sup_tokens,
            "seconds":train_seconds,
            "sequence_tokens_per_second":seq_tokens/train_seconds if train_seconds else None,
            "supervised_tokens_per_second":sup_tokens/train_seconds if train_seconds else None,
            "first_loss":losses[0] if losses else None,
            "last_loss":losses[-1] if losses else None,
            "mean_loss":sum(losses)/len(losses) if losses else None,
            "losses":losses,
        },
        "peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "adapter":adapter_info,
        "evaluation":{
            "system":eval_cfg["system"],
            "max_new_tokens":int(eval_cfg["max_new_tokens"]),
            "responses":responses,
        },
    }

    outdir=ROOT/"results"/"persona001"
    outdir.mkdir(parents=True,exist_ok=True)
    label="BASE" if args.base_only else gid
    (outdir/f"{label}.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "persona":label,
        "cpu":result["runner"]["cpu_model"],
        "trainable_parameters":trainable,
        "training_seconds":train_seconds,
        "sequence_tokens_per_second":result["training"]["sequence_tokens_per_second"],
        "loss":[result["training"]["first_loss"],result["training"]["last_loss"]],
        "adapter":adapter_info,
        "eval_responses":len(responses),
    },indent=2))

if __name__=="__main__":
    main()
