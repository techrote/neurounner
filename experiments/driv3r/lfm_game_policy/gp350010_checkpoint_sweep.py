#!/usr/bin/env python3
"""GP350-011: independent frozen checkpoint sweep of genuine GP350-010 LoRAs.
Only synthetic W0 game-policy research. No optimizer, game, provider or native writes.
Stages 24/48 receive fresh full-model forwards; immutable GP006/72 source
receipts are compared as REUSED original evidence, never represented as new inference.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import resource
import signal
import time
import lfm350 as c
import gp350_selective_branch_train as p

SCHEMA="driv3r.lfm350.gp350011-checkpoint-sweep/1"
SOURCE_RUN=43854890883
SOURCE_ARTIFACT=14221088357
SOURCE_SHA="ad5781667e2b4fdd29521a21bcd13adda5d5511c"
ADAPTER_HASHES={
    24:"047b2bacb61f6f04fb359e4d9839f6c198a17ba7db6dfb451d88e48846c66430",
    48:"159892e47cf76cef24274cf8d76881fb56a2b20052ae82eae228b7c036669e7b",
    72:"a829b47fdf424b66fa8ddedf59323d5e10406e0afd8d881a423194ea790dc82a",
}
CANCELLED=False
def on_stop(_signum,_frame):
    global CANCELLED
    CANCELLED=True
    p.CANCELLED=True

def original(source):
    src=source/"original-experiment"
    s=json.loads((src/"summary.json").read_text())
    c.check(s["status"]=="complete" and s["source_sha"]==SOURCE_SHA,
            "original completed GP350010 source mismatch")
    c.check(s["updates"]==s["backwards"]==72 and s["original_pngs"]==60,
            "original GP350010 updates/PNG proof missing")
    c.check(s["frozen_full_parent_unchanged"] and s["adapter_updated"],
            "original frozen model/checkpoint proof absent")
    c.check(s["parent_adapter_sha256"]==
            "436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c",
            "GP006 original parent mismatch")
    for step,h in ADAPTER_HASHES.items():
        f=src/f"adapter-step-{step}/adapter_model.safetensors"
        c.check(f.exists() and c.sha(f)==h,"GP010 adapter mismatch "+str(step))
    for row in s["baseline"]+s["trained"]:
        c.check(c.sha(src/row["png"])==row["png_sha256"],
                "GP010 original model PNG drift")
    return src,s

def make_sample(w0,source):
    observed=p.observed(w0,p.DEV_SEEDS)
    sample=p.stratified(observed)
    before=json.loads((source/"probe-initial.json").read_text())
    after=json.loads((source/"probe-final.json").read_text())
    c.check(len(sample)==len(before["rows"])==len(after["rows"])==72,
            "original heldout count drift")
    for row,a,b in zip(sample,before["rows"],after["rows"]):
        obs=row["observation"]
        key=(row["seed"],row["size"],obs["family"],row["ray2"],row["target"])
        for original_probe in (a,b):
            other=(original_probe["seed"],original_probe["size"],original_probe["family"],
                   original_probe["marker"],original_probe["target"])
            c.check(key==other,"heldout source-row or teacher recipe drift")
    return sample,before,after

def pressure_counts(out,episodes):
    results={}
    for fam in c.FAMILIES:
        rows=[r for r in episodes if r["family"]==fam]
        actions=collections.Counter()
        pressure=collections.Counter()
        bursts=[]
        for row in rows:
            rec=json.loads((out/(row["png"][:-4]+".json")).read_text())
            run=0
            for score in rec["scores"]:
                a=score["applied"]
                actions[a]+=1
                if score["ray2"] in ("#","+"):
                    pressure["encounters"]+=1
                    if a in ("LEFT","RIGHT"):pressure["anticipatory_turns"]+=1
                    if a=="BRANCH":pressure["branch_on_pressure"]+=1
                if a=="BRANCH":run+=1
                elif run:
                    bursts.append(run)
                    run=0
            if run:bursts.append(run)
        results[fam]={
            "episodes":len(rows),"mean_unique_cells":sum(r["unique_cells"] for r in rows)/len(rows),
            "actions":dict(actions),"pressure":dict(pressure),
            "branch_burst_max":max(bursts,default=0),
            "branch_burst_mean":sum(bursts)/len(bursts) if bursts else 0,
            "raw_illegal":sum(r["raw_illegal"] for r in rows),
            "host_illegal":sum(r["host_illegal"] for r in rows)}
    return results

def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi,snapshot_download
    started=time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(3510011)
    c.check(not args.out.exists(),"immutable GP350011 output exists")
    args.out.mkdir(parents=True)
    src,ss=original(args.source)
    w0=c.load_w0(args.w0)
    sample,before,after=make_sample(w0,src)
    c.save_json(args.out/"lease.json",{
        "schema":SCHEMA,"run":os.getenv("GITHUB_RUN_ID"),
        "source_sha":os.getenv("GITHUB_SHA"),
        "GP350010_source_run":SOURCE_RUN,
        "GP350010_source_artifact":SOURCE_ARTIFACT,
        "GP350010_source_sha":SOURCE_SHA,
        "adapter_sha256":ADAPTER_HASHES,
        "new_full_model_forwards_only":["gp350010-step24","gp350010-step48"],
        "original_GP006_and_GP010step72_receipts_reused_not_rescored":True,
        "native":False,"draft_hold":True})
    c.check(HfApi().model_info(c.MODEL,revision=c.REVISION).sha==c.REVISION,
            "upstream pinned model drift")
    snapshot_download(c.MODEL,revision=c.REVISION,local_dir=args.cache,max_workers=2,
        allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
            "generation_config.json","tokenizer*","special_tokens_map.json",
            "added_tokens.json","vocab.*","merges.txt","README.md","LICENSE*"])
    hashes={str(f.relative_to(args.cache)):c.sha(f) for f in sorted(args.cache.rglob("*"))
            if f.is_file() and ".cache" not in f.parts}
    prior_hashes=json.loads((src/"model-download.json").read_text())["sha256"]
    c.check(all(hashes.get(k)==v for k,v in prior_hashes.items()),
            "parent pinned full-model byte provenance mismatch")
    c.save_json(args.out/"model-sha256.json",hashes)
    tok=AutoTokenizer.from_pretrained(args.cache,local_files_only=True,trust_remote_code=False)
    ids=c.id_map(tok)
    base=AutoModelForCausalLM.from_pretrained(args.cache,local_files_only=True,
                                            trust_remote_code=False,dtype=torch.float32)
    c.check(base.config.model_type=="lfm2" and base.config.num_hidden_layers==16,
            "wrong parent architecture")
    base.requires_grad_(False)
    base.eval()
    model=PeftModel.from_pretrained(base,src/"adapter-step-24",is_trainable=False)
    model.load_adapter(src/"adapter-step-48",adapter_name="gp350010_step48",is_trainable=False)
    c.check(not any(param.requires_grad for param in model.parameters()),
            "checkpoint selection must be frozen inference")
    fingerprint=c.base_fingerprint(model)
    outputs={}
    for label,adapter in (("gp350010-step24","default"),
                          ("gp350010-step48","gp350010_step48")):
        c.check(not CANCELLED and time.monotonic()-started<args.deadline-500,
                "bounded checkpoint selection elapsed")
        model.set_adapter(adapter)
        model.eval()
        t0=time.monotonic()
        result=p.probe(model,tok,ids,sample)
        c.save_json(args.out/f"{label}-probe.json",result)
        episodes=[]
        for size in p.SIZES:
            for seed in p.EVAL_SEEDS:
                for family in c.FAMILIES:
                    c.check(not CANCELLED and time.monotonic()-started<args.deadline-300,
                            "bounded checkpoint selection stopped")
                    episodes.append(p.episode(w0,model,tok,ids,args.out,label,
                                              size,seed,family,started))
        data={
            "teacher_matches":result["teacher_matches"],
            "raw_illegal_probe":result["raw_illegal"],
            "marker_target_matches":result["marker_target_matches"],
            "episodes":episodes,"family":pressure_counts(args.out,episodes),
            "wall_seconds":time.monotonic()-t0,
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        c.save_json(args.out/f"{label}-stage.json",data)
        outputs[label]=data
        print(c.canonical({"GP350011_STAGE":label,
                           "heldout_teacher_matches":data["teacher_matches"],
                           "families":data["family"]}),flush=True)
    c.check(fingerprint==c.base_fingerprint(model),"frozen parent changed")
    c.check(len(outputs)==2 and sum(len(r["episodes"]) for r in outputs.values())==24,
            "incomplete 24/48 step source-matched rollout")
    c.check(all(r["host_illegal"]==0 for q in outputs.values()
                for r in q["episodes"]),"host legality regression")
    original_rows={
        "gp350006":{"probe":{k:v for k,v in before.items() if k!="rows"},
                    "episodes":ss["baseline"]},
        "gp350010-step72":{"probe":{k:v for k,v in after.items() if k!="rows"},
                           "episodes":ss["trained"]}}
    c.save_json(args.out/"summary.json",{
        "schema":SCHEMA,"status":"complete","source_sha":os.getenv("GITHUB_SHA"),
        "GP350010_exact_source_sha":SOURCE_SHA,
        "GP350010_original_artifact":SOURCE_ARTIFACT,
        "model":c.MODEL,"revision":c.REVISION,"native":False,"d1_matched":"NOT RUN",
        "fresh_checkpoint_stages":outputs,
        "reused_immutable_original_reference":original_rows,
        "fresh_model_decisions":24*64,"fresh_dev_probes":144,
        "new_optimizer_updates":0,"original_pngs_new":24,
        "wall_seconds":time.monotonic()-started,
        "cpu_seconds":time.process_time(),
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

def self_test(w0,source):
    src,_=original(source)
    sample,before,after=make_sample(c.load_w0(w0),src)
    c.check(before["teacher_matches"]==41 and after["teacher_matches"]==37,
            "source GP350010 teacher-score record drift")
    c.check(len(sample)==72 and all(r["target"] in r["observation"]["legal"]
                                    for r in sample),"illegal synthetic labels")
    print(c.canonical({"GP350011_SELFTEST":"PASS","observed_dev_rows":len(sample),
                       "source_original_probe":[41,37],"source_adapter_sha256":ADAPTER_HASHES}))

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--w0",required=True,type=Path)
    pa.add_argument("--source",required=True,type=Path)
    pa.add_argument("--cache",required=True,type=Path)
    pa.add_argument("--out",required=True,type=Path)
    pa.add_argument("--deadline",type=int,default=2200)
    pa.add_argument("--self-test",action="store_true")
    args=pa.parse_args()
    c.check(args.deadline==2200,"deadline recipe drift")
    if args.self_test:
        self_test(args.w0,args.source)
    else:
        try:evaluate(args)
        except BaseException as exc:
            args.out.mkdir(parents=True,exist_ok=True)
            f=args.out/"failure.json"
            if not f.exists():
                c.save_json(f,{"error":type(exc).__name__,"message":str(exc),
                    "source_sha":os.getenv("GITHUB_SHA"),"not_complete":True})
            raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,on_stop)
    signal.signal(signal.SIGINT,on_stop)
    main()
