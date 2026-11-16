#!/usr/bin/env python3
"""SPROCKET WORM W6 E1 — real paired-trained six-logit host-seeded W0 paintings.

This is a *future zero-optimizer* evaluator; execute only after original W6 step24
is verified. Original unmodified greedy/EOS5 probes remain in the W6 trainer.
The host STOP deferral is explicit and IDENTICAL in learned/flat ablation arms.
Neither scoring nor masking grants native permissions to the model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
TRAINING=HERE.parents[1]/"training"
sys.path[:0]=[str(TRAINING),str(HERE)]
from common import profile,sha256_file,verify_checkpoint,write_json
from lane_adapters import adapter_for
from numerics import configure_cpu
import train as shared
import worm_v1 as contract
import worm_v4 as objective
from worm_score_v5_eval import action_tokens,host_draw,sample_scores,spatial

# Source fenced to the newly launched one-shot hosted optimizer; no guessed success.
TRAIN_SOURCE="fbeece84ff2609e564bb43b7f40ecde1d80ef187"
BUDGET=24
TEMPERATURE=2.25
STOP_DEFER_BEFORE=20
WORLD_SEEDS=(211,223)

def policy_legal(observation):
    """Host-visible bounded STOP deferral; NEVER add/replace W0-legal actions."""
    actual=observation["legal"]
    if not actual or len(actual)!=len(set(actual)) or set(actual)-set(contract.ACTIONS):
        raise ValueError("bad original W0 host legal mask")
    if observation["tick"]<STOP_DEFER_BEFORE and len(actual)>1:
        subset=[a for a in actual if a!="STOP"]
    else:
        subset=list(actual)
    if not subset or set(subset)-set(actual):
        raise ValueError("host policy accidentally enabled an illegal action")
    return subset

def rollout(w0,model,tokenizer,adapter,family,seed,mode,token_ids):
    import torch
    is_model=(mode=="trained")
    actions=[]
    receipts=[]
    if is_model: model.eval()
    for tick in range(BUDGET):
        probe=w0.run(seed,"replay",family,steps=tick+1,raw_actions=actions)
        if len(probe["events"])!=tick+1:
            raise RuntimeError("W0 ended before the requested decision")
        obs=probe["events"][tick]["observation"]
        permitted=policy_legal(obs)
        if is_model:
            text=objective.prompt(obs)
            prefix=adapter.encode_prompt(tokenizer,text,objective.SYSTEM)
            ids=torch.tensor([prefix],dtype=torch.long)
            with torch.no_grad():
                output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            logits=[float(output.logits[0,-1,t].item()) for t in token_ids]
        else:
            text=None
            prefix=None
            logits=[0.0]*6
        draw=host_draw(family,seed,tick,TEMPERATURE)
        selected,score=sample_scores(logits,permitted,TEMPERATURE,draw)
        paired_uniform,_=sample_scores([0.0]*6,permitted,TEMPERATURE,draw)
        actions.append(selected)
        applied=w0.run(seed,"replay",family,steps=tick+1,raw_actions=actions)
        event=applied["events"][-1]
        if event["observation"]!=obs or event["raw_choice"]!=selected or not event["raw_legal"] or event["applied"]!=selected:
            raise RuntimeError("actual W0 host application rejected or drifted")
        receipts.append({
            "tick":tick,"observation_sha256":hashlib.sha256(
                json.dumps(obs,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "original_W0_legal":obs["legal"],"host_policy_legal":permitted,
            "host_stop_deferred":obs["tick"]<STOP_DEFER_BEFORE and len(obs["legal"])>1,
            "host_selected":selected,"actual_W0_applied":event["applied"],
            "prefix_ids":prefix,"prompt":text,
            "token_ids":token_ids if is_model else None,
            "same_observation_flat":paired_uniform,
            "differs_from_flat":selected!=paired_uniform,
            "model_forward":is_model,"conditioned_six_action_scores":is_model,
            "raw_greedy_EOS_completion":False,
            **score
        })
        if selected=="STOP":
            break
    record=w0.run(seed,"replay",family,steps=BUDGET,raw_actions=actions)
    if len(record["events"])!=len(receipts):
        raise RuntimeError("final replay diverged from source actions")
    record["mode"]="sprocket-w6-e1-"+mode
    record["engine"]="pinned-Python-W0-host-legal-sampled-not-native"
    record["score_receipts"]=receipts
    return record,receipts

def main(args):
    if args.training_source!=TRAIN_SOURCE or args.checkpoint.name!="checkpoint-000024":
        raise ValueError("wrong W6 source/checkpoint lease")
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    numerical,runtime=configure_cpu(1)
    selected=profile("sprocket",require_supported=True)
    w0=contract.load_w0(args.w0)
    checkpoint=verify_checkpoint(args.checkpoint)
    cfg=checkpoint["config"]
    if (cfg.get("schema")!="cybersand.worm.sprocket-family-counterfactual-v6" or
        cfg.get("source_sha")!=TRAIN_SOURCE or cfg.get("model")!=selected or
        cfg.get("worm_common_git_commit")!=objective.W0_COMMIT or
        cfg.get("worm_common_sha256")!=sha256_file(args.w0) or
        cfg.get("worm_support_git_commit")!=objective.SUPPORT_COMMIT or
        cfg.get("synthetic_not_native") is not True or
        cfg.get("final_opened") is not False or cfg.get("numerics")!=numerical or
        cfg.get("rank")!=8 or cfg.get("threads")!=1):
        raise ValueError("unqualified original W6 source/model/recipe/W0 checkpoint")
    from peft import PeftModel
    base,tokenizer=shared.load_base(args.model,selected)
    adapter=adapter_for(selected)
    token_ids,token_mapping=action_tokens(tokenizer)
    tuned=PeftModel.from_pretrained(base,args.checkpoint/"adapter",
                                  is_trainable=False,local_files_only=True)
    rows=[]
    for family in contract.FAMILIES:
        for seed in WORLD_SEEDS:
            for mode in ("random","wall","fidget"):
                record=w0.run(seed,mode,family,BUDGET)
                w0.write_result(record,args.output/"png-and-json")
                rows.append({"family":family,"world_seed":seed,"mode":mode,
                             "model_forward":False,"steps":record["steps_executed"],
                             "metrics":record["metrics"],"spatial":spatial(record)})
            for mode in ("flat","trained"):
                record,receipts=rollout(w0,tuned,tokenizer,adapter,family,seed,mode,token_ids)
                w0.write_result(record,args.output/"png-and-json")
                rows.append({"family":family,"world_seed":seed,"mode":mode,
                             "model_forward":mode=="trained","steps":record["steps_executed"],
                             "metrics":record["metrics"],"spatial":spatial(record),
                             "early_stop":record["steps_executed"]<BUDGET,
                             "raw_unmasked_argmax":{a:sum(r["raw_unmasked_top_action"]==a for r in receipts)
                                                    for a in contract.ACTIONS},
                             "selected_histogram":{a:sum(r["host_selected"]==a for r in receipts)
                                                    for a in contract.ACTIONS},
                             "model_diff_from_same_observation_flat":
                                sum(r["differs_from_flat"] for r in receipts),
                             "decision_count":len(receipts)})
                write_json(args.output/"partial-progress.json",
                           {"status":"partial","rows":rows,"optimizer_steps_this_run":0})
    write_json(args.output/"comparison.json",{
        "schema":"cybersand.worm.sprocket-w6-e1-host-sampled-first-gallery",
        "train_source":TRAIN_SOURCE,"trained_update_checkpoint":24,
        "original_train_run":args.train_run,
        "checkpoint_inventory_sha256":sha256_file(args.checkpoint/"inventory.json"),
        "model":selected["model_id"],"revision":selected["revision"],
        "W0_common_commit":objective.W0_COMMIT,
        "host_policy":"score six actual digit class logits; W0 legal mask; explicit STOP deferral before tick20; SHA256 common draw",
        "host_flat_ablation":"same seeds, tick draws, W0 masks and early-STOP rule; no model forward",
        "selected_temperature":TEMPERATURE,"seed_groups":list(WORLD_SEEDS),
        "families":list(contract.FAMILIES),"steps_per_trajectory":BUDGET,
        "same_observation_flat_counterfactual":True,"complete":True,
        "source_faithful_W0_not_native":True,"sealed_final_opened":False,
        "optimizer_steps_this_run":0,"rows":rows,"wall_seconds":time.monotonic()-started,
        "cpu_seconds":time.process_time(),
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "token_mapping":token_mapping,"numerical":numerical,"runtime":runtime})
    print(json.dumps({"complete":"actual W6 trained versus host-flat PNG comparison",
                      "model_trajectories":sum(r["mode"]=="trained" for r in rows),
                      "flat_trajectories":sum(r["mode"]=="flat" for r in rows),
                      "control_trajectories":sum(r["mode"] in ("random","wall","fidget") for r in rows)}))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--training-source",required=True)
    p.add_argument("--train-run",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    try:
        main(args)
    except BaseException as exc:
        if args.output.exists():
            write_json(args.output/"failure.json",{"type":type(exc).__name__,
                "message":str(exc),"status":"FAILED_OR_PARTIAL","not_success":True})
        raise
