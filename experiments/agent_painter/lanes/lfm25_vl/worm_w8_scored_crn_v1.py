#!/usr/bin/env python3
"""W8 immutable checkpoint-only six-code score sampler with common-random-number controls.

This is explicitly a HOST-MASKED stochastic policy, NOT the W8 raw greedy EOS7
model output. The original W8 raw model refused at all 9 live start states.
Compare live model-score bias with flat legal sampling using the same host
world/tick random variate; never attribute host sampling alone to learned art.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import worm_train_w7_v1 as w7
import worm_w8_checkpoint_gallery_v1 as w8
from worm_w7_sensitivity_eval_v1 import score_next_token
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint,load_base

VERSION="lfm-worm-w8-score-CRN-sampler-v1"
SEEDS=(3101,3109,3119)
STEPS=16
TEMP=2.5
MODEL_FRACTION=0.70
DEADLINE=1600
STOP=False

def stop(_sig,_frame):
    global STOP
    STOP=True

def common_u(seed,tick):
    """Same RNG for learned and flat arm, independent of family and temperature."""
    digest=hashlib.sha256(f"WORM-W8-CRN-v1|world={seed}|tick={tick}".encode()).digest()
    return (int.from_bytes(digest[:8],"big")+0.5)/2**64

def scored_policy(scores,legal,mode,seed,tick):
    allowed=[a for a in w7.ACTIONS if a in legal and (a!="STOP" or len(legal)==1)]
    if not allowed:
        raise ValueError("W0 host supplied empty legal movement set")
    uniform={a:1/len(allowed) for a in allowed}
    if mode=="learned":
        shifted=[scores["candidate_logits"][a]/TEMP for a in allowed]
        peak=max(shifted)
        weight=[math.exp(x-peak) for x in shifted]
        total=sum(weight)
        conditioned={a:w/total for a,w in zip(allowed,weight)}
        probs={a:MODEL_FRACTION*conditioned[a]+(1-MODEL_FRACTION)*uniform[a] for a in allowed}
    elif mode=="flat":
        conditioned=uniform
        probs=uniform
    else:
        raise ValueError("unsupported score sampler")
    u=common_u(seed,tick)
    threshold=0
    chosen=allowed[-1]
    for a in allowed:
        threshold+=probs[a]
        if u < threshold:
            chosen=a
            break
    if chosen not in legal:
        raise ValueError("host sampled an illegal action")
    return chosen,{"legal":list(legal),"sampled_legal":allowed,"host_u":u,
      "conditioned_model_probs":conditioned,"sample_probs":probs,
      "choice":chosen,"sampler":mode,"temperature":TEMP,"model_fraction":MODEL_FRACTION,
      "STOP_deferred_unless_forced":True}

def main():
    parser=argparse.ArgumentParser()
    for key in ("model","worm_root","support_root","producer_root","output"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("immutable W8 scored output lease exists")
    args.output.mkdir(parents=True)
    start=time.monotonic()
    write_json(args.output/"lease.json",{
      "experiment":VERSION,"evaluator_source":os.environ.get("GITHUB_SHA"),
      "producer_source":w8.PRODUCER_SHA,"producer_run":w8.PRODUCER_RUN,
      "producer_artifact":w8.PRODUCER_ARTIFACT,
      "producer_zip_sha256":w8.PRODUCER_ZIP_SHA256,
      "model_temperature":TEMP,"model_fraction":MODEL_FRACTION,
      "common_rng_independent_of_family":True,
      "worlds":list(SEEDS),"max_steps":STEPS,
      "new_optimizer_updates":0,"native_execution":False,"sealed_FINAL_opened":False})
    worm=w7.worm_source(args.worm_root)
    support=w7.support_source(args.support_root,worm)
    selected=profile("lfm25_vl",require_supported=True)
    configure_cpu(1)
    import torch
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    receipt=w8.verify_original(args.producer_root,args.model,selected)
    base,processor=load_base(args.model,selected)
    adapter_root=args.producer_root/"checkpoint-000180"/"adapter"
    model=PeftModel.from_pretrained(base,adapter_root,is_trainable=True,local_files_only=True)
    adapter=adapter_for(selected)
    adapter.set_training_mode(model)
    contract=adapter.validate_trainable(model,selected)
    if fingerprint(model,False)!=receipt["frozen_base_sha256"]:
        raise ValueError("original W8 frozen full parent fingerprint mismatch")
    stored=load_file(str(adapter_root/"adapter_model.safetensors"))
    loaded=get_peft_model_state_dict(model)
    if stored.keys()!=loaded.keys() or any(not torch.equal(stored[k],loaded[k].detach().cpu()) for k in stored):
        raise ValueError("original W8 adapter checkpoint readback mismatch")
    model.eval()
    write_json(args.output/"model-readback.json",{"contract":contract,
      "original_optimizer_updates":180,"new_optimizer_updates":0,
      "exact_adapter_tensors":len(stored),"original_W8_raw_greedy_quality":"9/9 first-state RETRACE illegal, stopped"})
    gallery=args.output/"gallery"
    gallery.mkdir()
    results=[]
    observations=[]
    for seed in SEEDS:
        for family in w7.FAMILIES:
            for mode in ("learned","flat"):
                if STOP or time.monotonic()-start>DEADLINE-150:
                    write_json(args.output/"partial.json",{
                      "reason":"deadline-or-Stop","complete_trajectories":len(results)})
                    raise RuntimeError("finite W8 scored evaluator interrupted; partial receipts retained")
                trace=[]
                def choose(obs):
                    tick=len(trace)
                    prompt=w7.make_prompt(obs)
                    scores=score_next_token(model,processor,selected,prompt) if mode=="learned" else None
                    action,decision=scored_policy(scores,obs["legal"],mode,seed,tick)
                    trace.append({**decision,"tick":tick,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                      "scores":scores,"host_selected_action":action,
                      "W8_raw_greedy_not_overwritten":True})
                    return action
                record=support.rollout(seed,family,STEPS,choose)
                if len(trace)!=len(record["events"]):
                    raise ValueError("W0 actual applied-action trace mismatch")
                stem=f"{family}-{mode}-{seed}"
                png=gallery/(stem+".png")
                png.write_bytes(worm.png_bytes(record))
                write_json(gallery/(stem+".json"),record)
                write_json(gallery/(stem+"-scored-actions.json"),trace)
                entry={"family":family,"seed":seed,"policy":mode,
                       "metrics":support.statistics(record),"steps":len(trace),
                       "png_sha256":sha256_file(png)}
                results.append(entry)
                observations.append({"family":family,"seed":seed,"policy":mode,"trace":trace})
                write_json(args.output/"rolling-evidence.json",{"results":results,
                  "model_or_flat_scores":observations,"new_optimizer_updates":0})
    summary={
      "status":"complete","model_rollouts":sum(x["policy"]=="learned" for x in results),
      "flat_rollouts":sum(x["policy"]=="flat" for x in results),
      "model_scored_decisions":sum(len(x["trace"]) for x in observations if x["policy"]=="learned"),
      "distinct_model_pngs":len({x["png_sha256"] for x in results if x["policy"]=="learned"}),
      "model_families":list(w7.FAMILIES),"world_seeds":list(SEEDS),
      "learned_vs_flat_at_same_world_tick":"CRN same world/tick; later observations may diverge",
      "results":results,
      "original_raw_W8_greedy_refusal":"9/9 illegal RETRACE at starting W0 state",
      "new_optimizer_updates":0,"original_optimizer_updates":180,
      "wall_seconds":time.monotonic()-start,"cpu_seconds":time.process_time(),
      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      "native_execution":False,"sealed_FINAL_opened":False}
    write_json(args.output/"summary.json",summary)
    print(canonical({"event":"W8-scored-CRN-complete","rollouts":len(results),
                     "model_scored_decisions":summary["model_scored_decisions"]}),flush=True)

if __name__=="__main__":
    try:
        signal.signal(signal.SIGINT,stop)
        signal.signal(signal.SIGTERM,stop)
        main()
    except Exception as error:
        if "--output" in sys.argv:
            root=Path(sys.argv[sys.argv.index("--output")+1])
            root.mkdir(parents=True,exist_ok=True)
            write_json(root/"failure.json",{"experiment":VERSION,
              "error_type":type(error).__name__,"error":str(error)})
        raise
