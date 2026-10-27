#!/usr/bin/env python3
"""DETECTIVE-001C3: one 28-step pairwise continuation from exact G02 C2."""
from __future__ import annotations

import argparse
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
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.detective_001c2_train import (  # noqa: E402
    sha_file,directory_hash,encode_response,pair_batch,
    assistant_mean_logps,preference_eval,cpu_model,write_json
)

CONFIG=ROOT/"experiments/detective-001/training-c3.json"


class C3TrainError(ValueError):
    pass


def load_rows(dataset_root:Path,cfg:dict)->tuple[dict,list[dict]]:
    manifest=json.loads((dataset_root/"manifest.json").read_text())
    if (manifest.get("schema")!="detective-witness-c3-pair-dataset-v1"
            or manifest.get("experiment")!="DETECTIVE-001C3"
            or manifest.get("final_suite_materialized") is not False
            or manifest.get("c1_final_used_as_training") is not False
            or manifest.get("c2_final_used_as_training") is not False):
        raise C3TrainError("C3 pair dataset custody mismatch")
    shard=dataset_root/manifest["training_shard"]
    if sha_file(shard)!=manifest["training_shard_sha256"]:
        raise C3TrainError("C3 pair shard hash mismatch")
    rows=[json.loads(x) for x in shard.read_text().splitlines() if x.strip()]
    if len(rows)!=cfg["dataset"]["pairs"] or len({x["pair_id"] for x in rows})!=28:
        raise C3TrainError("C3 pair count/ID mismatch")
    if any(x["member_id"]!="V-G02-A" for x in rows):
        raise C3TrainError("C3 contains unauthorized member")
    return manifest,rows


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--parent-adapter",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    cfg=json.loads(CONFIG.read_text())
    if (cfg.get("experiment")!="DETECTIVE-001C3"
            or cfg["trainable_parent"]["member_id"]!="V-G02-A"
            or cfg["frozen_control"]["optimizer_steps_authorized"]!=0):
        raise C3TrainError("C3 training authority drift")
    train=cfg["training"]
    if train["optimizer_steps"]!=28 or train["epochs"]!=1:
        raise C3TrainError("C3 optimizer budget drift")
    if train["paid_api_calls"]!=0 or train["gpu_required"]:
        raise C3TrainError("C3 unapproved compute/provider budget")

    manifest,rows=load_rows(args.dataset_root,cfg)
    parent_hash=directory_hash(args.parent_adapter)
    if parent_hash!=cfg["trainable_parent"]["directory_sha256"]:
        raise C3TrainError("C2 G02 parent adapter hash mismatch")

    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    torch.set_num_threads(min(train["max_threads"],os.cpu_count() or 1))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    seed=31802003
    random.seed(seed); torch.manual_seed(seed)

    weight=Path(hf_hub_download(
        cfg["base_model"]["id"],"model.safetensors",
        revision=cfg["base_model"]["revision"]))
    if sha_file(weight)!=cfg["base_model"]["weights_sha256"]:
        raise C3TrainError("base model SHA mismatch")

    started_load=time.monotonic()
    tok=AutoTokenizer.from_pretrained(
        cfg["base_model"]["id"],revision=cfg["base_model"]["revision"],
        trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["id"],revision=cfg["base_model"]["revision"],
        torch_dtype=torch.float32,trust_remote_code=False)
    base.config.use_cache=False
    model=PeftModel.from_pretrained(
        base,args.parent_adapter,is_trainable=True)
    model.config.use_cache=False
    load_seconds=time.monotonic()-started_load

    trainable_names=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise C3TrainError("no trainable LoRA parameters")
    offenders=[n for n in trainable_names
               if "lora_" not in n.casefold()
               and "modules_to_save" not in n.casefold()]
    if offenders:
        raise C3TrainError("non-LoRA parameter trainable: "+",".join(offenders[:8]))
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())

    pairs=[]
    for row in rows:
        c=encode_response(tok,cfg["system"],row["input"],row["chosen"],
                          train["max_length"])
        r=encode_response(tok,cfg["system"],row["input"],row["rejected"],
                          train["max_length"])
        pairs.append((c,r))
    before=preference_eval(model,pairs,tok.pad_token_id)

    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train["learning_rate"],betas=(0.9,0.95),weight_decay=0.01)
    order=list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    if len(order)!=train["optimizer_steps"]:
        raise C3TrainError("C3 schedule mismatch")

    losses=[]; pref_losses=[]; anchor_losses=[]; margins=[]
    seq_tokens=sup_tokens=0
    model.train()
    started=time.monotonic()
    for idx in order:
        chosen,rejected=pairs[idx]
        ids,labels,attention=pair_batch(chosen,rejected,tok.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        values=assistant_mean_logps(model,ids,labels,attention)
        chosen_lp,rejected_lp=values[0],values[1]
        margin=chosen_lp-rejected_lp
        pref=-torch.nn.functional.logsigmoid(
            train["beta"]*(margin-train["target_margin_nats"]))
        anchor=-chosen_lp
        loss=pref+train["chosen_nll_weight"]*anchor
        if not torch.isfinite(loss):
            raise C3TrainError("nonfinite C3 pairwise loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            train["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach()))
        pref_losses.append(float(pref.detach()))
        anchor_losses.append(float(anchor.detach()))
        margins.append(float(margin.detach()))
        seq_tokens+=chosen["sequence_tokens"]+rejected["sequence_tokens"]
        sup_tokens+=chosen["supervised_tokens"]+rejected["supervised_tokens"]
    seconds=time.monotonic()-started
    if len(losses)!=28:
        raise C3TrainError("incomplete C3 training")

    after=preference_eval(model,pairs,tok.pad_token_id)
    args.out.mkdir(parents=True,exist_ok=True)
    adapter=args.out/"adapter"
    model.save_pretrained(adapter,safe_serialization=True)
    trained_hash=directory_hash(adapter)

    result={
        "schema":"detective-witness-c3-preference-result-v1",
        "experiment":"DETECTIVE-001C3","issue":118,
        "member_id":"V-G02-A",
        "base_model":cfg["base_model"]["id"],
        "base_revision":cfg["base_model"]["revision"],
        "base_weight_sha256":cfg["base_model"]["weights_sha256"],
        "parent_release_tag":cfg["trainable_parent"]["release_tag"],
        "parent_archive_sha256":cfg["trainable_parent"]["archive_sha256"],
        "parent_directory_sha256":parent_hash,
        "pair_dataset_manifest_sha256":sha_file(args.dataset_root/"manifest.json"),
        "training_shard_sha256":manifest["training_shard_sha256"],
        "pair_ids":[x["pair_id"] for x in rows],
        "preference_before":before,
        "training":{
            "optimizer_steps":len(losses),"epochs":1,
            "learning_rate":train["learning_rate"],
            "beta":train["beta"],
            "target_margin_nats":train["target_margin_nats"],
            "chosen_nll_weight":train["chosen_nll_weight"],
            "max_length":train["max_length"],
            "gradient_clip":train["gradient_clip"],
            "seconds":seconds,
            "sequence_tokens":seq_tokens,
            "supervised_tokens":sup_tokens,
            "sequence_tokens_per_second":seq_tokens/seconds if seconds else None,
            "supervised_tokens_per_second":sup_tokens/seconds if seconds else None,
            "first_loss":losses[0],"last_loss":losses[-1],
            "mean_loss":sum(losses)/len(losses),
            "mean_preference_loss":sum(pref_losses)/len(pref_losses),
            "mean_anchor_nll":sum(anchor_losses)/len(anchor_losses),
            "mean_training_margin":sum(margins)/len(margins),
            "losses":losses
        },
        "preference_after":after,
        "model":{
            "total_parameters":total,"trainable_parameters":trainable,
            "trainable_fraction":trainable/total,
            "only_lora_parameters_trainable":True
        },
        "adapter":{
            "path":"adapter","directory_sha256":trained_hash,
            "bytes":sum(p.stat().st_size for p in adapter.rglob("*") if p.is_file())
        },
        "frozen_control":{
            "member_id":"V-G09-C",
            "optimizer_steps":0,
            "directory_sha256":cfg["frozen_control"]["directory_sha256"]
        },
        "runtime":{
            "cpu_model":cpu_model(),"cpu_count":os.cpu_count(),
            "platform":platform.platform(),"threads":torch.get_num_threads(),
            "python":platform.python_version(),"torch":torch.__version__,
            "transformers":transformers.__version__,"peft":peft.__version__,
            "peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "load_seconds":load_seconds
        }
    }
    write_json(args.out/"result.json",result)
    print(json.dumps({
        "steps":28,
        "preference_margin":[before["mean_margin_nats_per_token"],
                             after["mean_margin_nats_per_token"]],
        "preference_accuracy":[before["preference_accuracy"],
                               after["preference_accuracy"]],
        "loss":[losses[0],losses[-1]],
        "parent_sha256":parent_hash,
        "trained_sha256":trained_hash,
        "seconds":seconds
    },indent=2))


if __name__=="__main__":
    main()
