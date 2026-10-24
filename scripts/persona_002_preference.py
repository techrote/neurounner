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
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
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

def load_rows(path:Path,gid:str):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows)!=100:
        raise RuntimeError(f"{gid}: expected 100 rows, got {len(rows)}")
    if any(x.get("source_guide_id")!=gid for x in rows):
        raise RuntimeError(f"{gid}: lineage mismatch")
    if len({x["example_id"] for x in rows})!=100:
        raise RuntimeError(f"{gid}: duplicate example IDs")
    return rows

def encode_response(tok,system,user,response,max_length):
    prompt=[
        {"role":"system","content":system},
        {"role":"user","content":user},
    ]
    full=prompt+[{"role":"assistant","content":response}]
    prompt_ids=tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
    full_ids=tok.apply_chat_template(full,tokenize=True,add_generation_prompt=False)
    common=0
    for a,b in zip(prompt_ids,full_ids):
        if a!=b:
            break
        common+=1
    if len(full_ids)>max_length:
        raise RuntimeError(f"sequence length {len(full_ids)} exceeds {max_length}")
    labels=[-100]*common+full_ids[common:]
    return {
        "input_ids":torch.tensor(full_ids,dtype=torch.long),
        "labels":torch.tensor(labels,dtype=torch.long),
        "sequence_tokens":len(full_ids),
        "supervised_tokens":len(full_ids)-common,
    }

def pair_batch(chosen,rejected,pad_id):
    ids=pad_sequence([chosen["input_ids"],rejected["input_ids"]],batch_first=True,padding_value=pad_id)
    labels=pad_sequence([chosen["labels"],rejected["labels"]],batch_first=True,padding_value=-100)
    attn=(ids!=pad_id).long()
    return ids,labels,attn

def assistant_mean_logps(model,input_ids,labels,attention_mask):
    out=model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
    logits=out.logits[:,:-1,:]
    targets=input_ids[:,1:]
    mask=labels[:,1:]!=-100
    logp=torch.log_softmax(logits,dim=-1)
    token_logp=logp.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    token_logp=token_logp*mask
    counts=mask.sum(dim=1).clamp_min(1)
    return token_logp.sum(dim=1)/counts

def schedule(n,epochs,seed):
    order=[]
    for epoch in range(epochs):
        idx=list(range(n))
        random.Random(seed+epoch*100003).shuffle(idx)
        order.extend(idx)
    return order

@torch.inference_mode()
def preference_eval(model,pairs,pad_id):
    model.eval()
    margins=[]
    chosen_logps=[]
    rejected_logps=[]
    for chosen,rejected in pairs:
        ids,labels,attn=pair_batch(chosen,rejected,pad_id)
        vals=assistant_mean_logps(model,ids,labels,attn)
        c=float(vals[0]);r=float(vals[1])
        chosen_logps.append(c);rejected_logps.append(r);margins.append(c-r)
    return {
        "mean_chosen_logp":sum(chosen_logps)/len(chosen_logps),
        "mean_rejected_logp":sum(rejected_logps)/len(rejected_logps),
        "mean_margin_nats_per_token":sum(margins)/len(margins),
        "preference_accuracy":sum(1 for x in margins if x>0)/len(margins),
        "margin_min":min(margins),
        "margin_max":max(margins),
        "margins":margins,
    }

@torch.inference_mode()
def run_heldout(model,tok,eval_cfg):
    old=getattr(model.config,"use_cache",None)
    model.config.use_cache=True
    model.eval()
    rows=[]
    for item in eval_cfg["scenarios"]:
        messages=[
            {"role":"system","content":eval_cfg["system"]},
            {"role":"user","content":item["prompt"]},
        ]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt")
        t0=time.monotonic()
        out=model.generate(
            ids,
            max_new_tokens=int(eval_cfg["max_new_tokens"]),
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
    ap.add_argument("--parent-adapter",type=Path,required=True)
    args=ap.parse_args()

    cfg=json.loads(args.config.read_text())
    gid=args.persona
    if gid not in cfg["personas"]:
        raise SystemExit(f"unknown persona {gid}")
    train_cfg=cfg["training"]
    tournament=json.loads((ROOT/cfg["tournament_config"]).read_text())
    eval_cfg=tournament["eval"]
    data_path=ROOT/cfg["dataset_dir"]/f"{gid}.jsonl"
    rows=load_rows(data_path,gid)

    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    torch.set_num_threads(min(4,os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed=int(train_cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    t_load=time.monotonic()
    tok=AutoTokenizer.from_pretrained(cfg["base_model"],revision=cfg["base_revision"])
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        revision=cfg["base_revision"],
        torch_dtype=torch.float32,
    )
    base.config.use_cache=False
    model=PeftModel.from_pretrained(base,args.parent_adapter,is_trainable=True)
    model.config.use_cache=False
    load_seconds=time.monotonic()-t_load

    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    max_length=int(train_cfg["max_length"])
    pairs=[]
    seq_tokens=0
    sup_tokens=0
    for row in rows:
        chosen=encode_response(tok,eval_cfg["system"],row["input"],row["target"],max_length)
        rejected=encode_response(tok,eval_cfg["system"],row["input"],row["generic_bad_target"],max_length)
        pairs.append((chosen,rejected))
        seq_tokens+=chosen["sequence_tokens"]+rejected["sequence_tokens"]
        sup_tokens+=chosen["supervised_tokens"]+rejected["supervised_tokens"]

    before=preference_eval(model,pairs,tok.pad_token_id)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        betas=(0.9,0.95),
        weight_decay=0.01,
    )
    beta=float(train_cfg["beta"])
    target_margin=float(train_cfg["target_margin_nats"])
    anchor_weight=float(train_cfg["chosen_nll_weight"])
    clip=float(train_cfg["gradient_clip"])
    order=schedule(len(pairs),int(train_cfg["epochs"]),seed)

    losses=[]
    pref_losses=[]
    anchor_losses=[]
    train_margins=[]
    model.train()
    t0=time.monotonic()
    processed_sequence_tokens=0
    processed_supervised_tokens=0
    for pos in order:
        chosen,rejected=pairs[pos]
        ids,labels,attn=pair_batch(chosen,rejected,tok.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        vals=assistant_mean_logps(model,ids,labels,attn)
        chosen_lp,rejected_lp=vals[0],vals[1]
        margin=chosen_lp-rejected_lp
        pref_loss=-torch.nn.functional.logsigmoid(beta*(margin-target_margin))
        anchor_loss=-chosen_lp
        loss=pref_loss+anchor_weight*anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        pref_losses.append(float(pref_loss.detach()))
        anchor_losses.append(float(anchor_loss.detach()))
        train_margins.append(float(margin.detach()))
        processed_sequence_tokens+=chosen["sequence_tokens"]+rejected["sequence_tokens"]
        processed_supervised_tokens+=chosen["supervised_tokens"]+rejected["supervised_tokens"]
    train_seconds=time.monotonic()-t0

    after=preference_eval(model,pairs,tok.pad_token_id)
    heldout=run_heldout(model,tok,eval_cfg)

    adapter_dir=ROOT/"persona002"/gid/"adapter"
    adapter_dir.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(adapter_dir,safe_serialization=True)
    adapter_bytes=sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file())

    result={
        "experiment":"PERSONA-002",
        "issue":6,
        "persona":gid,
        "parent_release":cfg["parent_release"],
        "parent_asset":cfg["parent_assets"][gid],
        "base_model":cfg["base_model"],
        "base_revision":cfg["base_revision"],
        "training_dataset":str(data_path.relative_to(ROOT)),
        "training_dataset_sha256":sha256_file(data_path),
        "training_config":train_cfg,
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
        "trainable_fraction":trainable/total,
        "preference_before":before,
        "training":{
            "epochs":int(train_cfg["epochs"]),
            "optimizer_steps":len(order),
            "pair_count":len(pairs),
            "sequence_tokens":processed_sequence_tokens,
            "supervised_tokens":processed_supervised_tokens,
            "seconds":train_seconds,
            "sequence_tokens_per_second":processed_sequence_tokens/train_seconds,
            "supervised_tokens_per_second":processed_supervised_tokens/train_seconds,
            "first_loss":losses[0],
            "last_loss":losses[-1],
            "mean_loss":sum(losses)/len(losses),
            "mean_preference_loss":sum(pref_losses)/len(pref_losses),
            "mean_anchor_nll":sum(anchor_losses)/len(anchor_losses),
            "mean_train_margin":sum(train_margins)/len(train_margins),
            "losses":losses,
        },
        "preference_after":after,
        "peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "adapter":{
            "path":str(adapter_dir.relative_to(ROOT)),
            "bytes":adapter_bytes,
            "sha256":directory_hash(adapter_dir),
        },
        "evaluation":{
            "system":eval_cfg["system"],
            "max_new_tokens":int(eval_cfg["max_new_tokens"]),
            "responses":heldout,
        },
    }

    outdir=ROOT/"results"/"persona002"
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/f"{gid}.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "persona":gid,
        "cpu":result["runner"]["cpu_model"],
        "steps":len(order),
        "seq_tok_s":result["training"]["sequence_tokens_per_second"],
        "loss":[losses[0],losses[-1]],
        "preference_margin_before":before["mean_margin_nats_per_token"],
        "preference_margin_after":after["mean_margin_nats_per_token"],
        "preference_accuracy_before":before["preference_accuracy"],
        "preference_accuracy_after":after["preference_accuracy"],
        "peak_rss_kib":result["peak_rss_kib"],
        "adapter":result["adapter"],
        "heldout_responses":len(heldout),
    },indent=2))

if __name__=="__main__":
    main()
