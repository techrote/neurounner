#!/usr/bin/env python3
"""Zero-optimizer W8 checkpoint evaluation: raw decisions and real W0 PNGs only."""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import worm_train_w7_v1 as w7
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint, load_base, verify_checkpoint

VERSION="lfm-worm-w8-step180-checkpoint-gallery-v1"
PRODUCER_SHA="61730799eddfbd43eac5d7945da11d6f953b47e6"
PRODUCER_RUN=43731856620
PRODUCER_ARTIFACT=14169951142
PRODUCER_ZIP_SHA256="b9d19fa758572660e837e59a13f50843639f6ea9c2350e0ff936fc41f549add9"
TRAIN_PAIRS_SHA256="8d4c4ee60af6ab05776e150cc2b43fbf667ff364fd6f45d507b65d96e3844919"
SEEDS=(3101,3109,3119)
STEPS=16
DEADLINE=1500
STOP=False

def stop(_sig,_frame):
    global STOP
    STOP=True

def verify_original(root,model_root,selected):
    receipt=verify_checkpoint(root/"checkpoint-000180")
    cfg=receipt["config"]
    if (receipt["global_step"]!=180 or receipt["sample_cursor"]!=180
            or len(receipt["history"])!=180
            or [h["global_step"] for h in receipt["history"]]!=list(range(1,181))
            or any(h["gradient_norm"]<=0 or h["loss"]<0 for h in receipt["history"])):
        raise ValueError("W8 original AdamW receipts invalid")
    if (cfg["source_sha"]!=PRODUCER_SHA
            or cfg["objective"]!="lfm-worm-w8-paired-legal-contrast-fresh-v1"
            or cfg["model"]!=selected or cfg["system"]!=w7.SYSTEM
            or cfg["final_targets_opened"] or cfg["native_execution"]):
        raise ValueError("W8 model/recipe/FINAL source mismatch")
    m=cfg["corpus_manifest"]
    if (m["pair_bytes_sha256"]!=TRAIN_PAIRS_SHA256
            or m["W0_source"]!=w7.WORM_SHA
            or m["support_source"]!=w7.SUPPORT_SHA
            or m["original_W0_observations"]!=45
            or m["synthetic_not_host_executed"]!=45
            or m["TRAIN_DEV_prompt_overlap"]!=0
            or sha256_file(root/"optimizer-pairs.jsonl")!=TRAIN_PAIRS_SHA256):
        raise ValueError("W8 source-fenced TRAIN mismatch")
    if cfg["model_files"]!=json.loads((model_root/"download_receipt.json").read_text())["files"]:
        raise ValueError("full original official-model inventory differs")
    if cfg["recipe"]["action_codes"]!=w7.ACTION_CODES:
        raise ValueError("wrong original EOS7 action codes")
    s=json.loads((root/"summary.json").read_text())
    if s["status"]!="complete" or s["actual_optimizer_updates"]!=180 or s["source_sha"]!=PRODUCER_SHA:
        raise ValueError("original W8 180-step completion claim invalid")
    return receipt

def main():
    p=argparse.ArgumentParser()
    for name in ("model","worm_root","support_root","producer_root","output"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError("immutable gallery output lease already exists")
    args.output.mkdir(parents=True)
    t0=time.monotonic()
    write_json(args.output/"lease.json",{
      "experiment":VERSION,"evaluator_source":os.environ.get("GITHUB_SHA"),
      "producer_source":PRODUCER_SHA,"producer_run":PRODUCER_RUN,
      "producer_artifact":PRODUCER_ARTIFACT,"producer_zip_sha256":PRODUCER_ZIP_SHA256,
      "world_seeds":list(SEEDS),"steps_per_rollout":STEPS,
      "new_optimizer_updates":0,"sealed_FINAL_opened":False,"native_execution":False})
    worm=w7.worm_source(args.worm_root)
    support=w7.support_source(args.support_root,worm)
    selected=profile("lfm25_vl",require_supported=True)
    configure_cpu(1)
    import torch
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    receipt=verify_original(args.producer_root,args.model,selected)
    base,processor=load_base(args.model,selected)
    checkpoint=args.producer_root/"checkpoint-000180"/"adapter"
    model=PeftModel.from_pretrained(base,checkpoint,is_trainable=True,local_files_only=True)
    adapter=adapter_for(selected)
    adapter.set_training_mode(model)
    contract=adapter.validate_trainable(model,selected)
    if fingerprint(model,False)!=receipt["frozen_base_sha256"]:
        raise ValueError("frozen complete multimodal parent differs")
    stored=load_file(str(checkpoint/"adapter_model.safetensors"))
    loaded=get_peft_model_state_dict(model)
    if stored.keys()!=loaded.keys() or any(not torch.equal(stored[k],loaded[k].detach().cpu()) for k in stored):
        raise ValueError("immutable original LoRA readback differs")
    model.eval()
    write_json(args.output/"checkpoint-readback.json",{
      "producer_source":PRODUCER_SHA,"checkpoint":"checkpoint-000180",
      "adapter_tensors_verified":len(stored),"contract":contract,"new_optimizer_updates":0})
    gallery=args.output/"gallery"
    gallery.mkdir()
    records=[]
    raw_events=[]
    def save_record(stem,r,kind,family,seed):
        png=gallery/(stem+".png")
        png.write_bytes(worm.png_bytes(r))
        write_json(gallery/(stem+".json"),r)
        item={"policy":kind,"family":family,"seed":seed,
              "metrics":support.statistics(r),"png_sha256":sha256_file(png)}
        records.append(item)
        return item
    for seed in SEEDS:
        for family in w7.FAMILIES:
            if STOP or time.monotonic()-t0 > DEADLINE-130:
                write_json(args.output/"partial-reason.json",{
                  "reason":"STOP-or-deadline","completed_model_rollouts":len([r for r in records if r["policy"]=="trained-W8-raw"])})
                break
            record,raw=w7.rollout(model,processor,selected,support,family,seed,max_steps=STEPS)
            if len(record["events"])!=len(raw):
                raise ValueError("W0 raw-model callback count mismatch")
            stem=f"{family}-model-{seed}"
            save_record(stem,record,"trained-W8-raw",family,seed)
            write_json(gallery/(stem+"-raw-first-passes.json"),raw)
            raw_events.extend({"family":family,"seed":seed,"tick":i,
              "raw_text":x["raw_text"],"raw_action":x["raw_action"],
              "token_ids":x["token_ids"],"eos_terminated":x["eos_terminated"],
              "raw_legal":x["raw_legal"],"valid_first_pass":x["valid_first_pass"]}
              for i,x in enumerate(raw))
            for mode in ("random","wall"):
                control=worm.run(seed,mode=mode,family=family,steps=STEPS)
                save_record(f"{family}-{mode}-{seed}",control,mode,family,seed)
            oracle=support.rollout(seed,family,STEPS,support.oracle)
            save_record(f"{family}-oracle-{seed}",oracle,"oracle",family,seed)
            write_json(args.output/"rolling-receipt.json",{
              "records":records,"raw_model_decisions":raw_events,"new_optimizer_updates":0})
        if STOP or time.monotonic()-t0 > DEADLINE-130:
            break
    learned=[r for r in records if r["policy"]=="trained-W8-raw"]
    summary={
      "status":"complete" if len(learned)==9 else "partial",
      "model_rollouts":len(learned),"expected_model_rollouts":9,
      "raw_model_decisions":len(raw_events),
      "raw_text_distribution":dict(collections.Counter(r["raw_text"] for r in raw_events)),
      "raw_legal":sum(r["raw_legal"] for r in raw_events),
      "EOS7_terminated":sum(r["eos_terminated"] for r in raw_events),
      "distinct_model_pngs":len({r["png_sha256"] for r in learned}),
      "records":records,"original_optimizer_updates":180,"new_optimizer_updates":0,
      "wall_seconds":time.monotonic()-t0,"cpu_seconds":time.process_time(),
      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      "native_execution":False,"sealed_FINAL_opened":False}
    write_json(args.output/"summary.json",summary)
    if summary["status"]!="complete":
        raise ValueError("W8 model PNG gallery partial: originals retained")
    print(canonical({"event":"W8-gallery-PASS","model_rollouts":9,
                     "raw_actions":len(raw_events),"distinct_model_pngs":summary["distinct_model_pngs"]}),flush=True)

if __name__=="__main__":
    try:
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        main()
    except Exception as e:
        if "--output" in sys.argv:
            out=Path(sys.argv[sys.argv.index("--output")+1])
            out.mkdir(parents=True,exist_ok=True)
            write_json(out/"failure.json",{"experiment":VERSION,"error":str(e),"type":type(e).__name__})
        raise
