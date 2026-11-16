#!/usr/bin/env python3
"""GP350-009: stock/GP350006/GP350008-step32/step64 checkpoint selection.

Read-only, original pinned W0 host. Genuine FULL pinned LFM2.5-350M forwards,
host-legal 64-turn trajectories, held-out real W0 local-observation probes.
Checkpoint-only: NO optimizer; no game/native/default-provider changes.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

import lfm350 as c
import gp350_obstacle_exposure_eval as previous
import gp350_observed_replay_train as replay

SCHEMA = "driv3r.lfm350.gp350009-checkpoint-selection/1"
GP006_RUN = 43806503356
GP008_RUN = 43820731360
GP006_SHA = "436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c"
GP008_32_SHA = "6ac60a0b9ef38c8a8ce63437dde70a5d3e27cfc195090f47c65c00eb6ea1e171"
GP008_64_SHA = "340979ffae0a0dce48be57bbd42d908f5b0c0c04399653381c1818d14dafb497"
STAGES = ("stock", "gp350006", "gp350008-step32", "gp350008-step64")
SIZES = (32, 48)
SEEDS = (9503, 9521)
STEPS = 64
LIMIT = 2400

def selftest(path):
    w0 = c.load_w0(path)
    previous.self_test(path)
    probe = replay.stratified(replay.observed(w0, replay.DEV_SEEDS))
    c.check(len(probe) == 72, "heldout W0 probe inventory mismatch")
    c.check(all(row["target"] in row["observation"]["legal"] for row in probe),
            "illegal heldout teacher target")
    # Stage receipts must never collide; save_json is write-once by design.
    partials=["partial-summary-"+name+".json" for name in STAGES]
    c.check(len(partials)==len(set(partials)), "stage partial receipts collide")
    print(c.canonical({"GP350009_SELFTEST": "PASS", "heldout_rows":len(probe),
        "original_W0":True, "checkpoint_only":True, "source_fenced":True}), flush=True)

def summarise(rows):
    n=len(rows)
    return {"episodes":n,"decisions":sum(x["decisions"] for x in rows),
        "mean_unique_cells":sum(x["unique_cells"] for x in rows)/n,
        "branches":sum(x["branches"] for x in rows),
        "retraces":sum(x["retraces"] for x in rows),
        "raw_illegal":sum(x.get("raw_illegal",0) for x in rows),
        "host_illegal":sum(x["host_illegal"] for x in rows),
        "choice_counts":dict(sum((collections.Counter(x.get("choice_counts",{}))
                            for x in rows),collections.Counter())),
        "ray2_counts":dict(sum((collections.Counter(x.get("forward_two_cell_counts",{}))
                          for x in rows),collections.Counter()))}

def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, snapshot_download
    torch.set_num_threads(2)
    started = time.monotonic()
    out = args.out
    c.check(not out.exists(),"immutable GP350009 output lease already exists")
    out.mkdir(parents=True)
    c.save_json(out/"lease.json",{"schema":SCHEMA,"source_sha":os.getenv("GITHUB_SHA"),
      "run_id":os.getenv("GITHUB_RUN_ID"),"attempt":os.getenv("GITHUB_RUN_ATTEMPT"),
      "optimizer_steps":0,"model":c.MODEL,"revision":c.REVISION,
      "parent_runs":{"GP350006":GP006_RUN,"GP350008":GP008_RUN},
      "adapter_hashes":{"006":GP006_SHA,"008_32":GP008_32_SHA,"008_64":GP008_64_SHA},
      "native":False,"vision":False,"paid_GPU":False,"draft_hold":True})
    for d,sha in ((args.adapter006,GP006_SHA),
                  (args.adapter32,GP008_32_SHA),
                  (args.adapter64,GP008_64_SHA)):
        c.check(d.is_dir() and c.sha(d/"adapter_model.safetensors")==sha,
                "immutable adapter download/hash mismatch: "+str(d))
    w0=c.load_w0(args.w0)
    selftest(args.w0)
    samples=replay.stratified(replay.observed(w0,replay.DEV_SEEDS))
    c.save_json(out/"heldout-inventory.json",{
        "heldout_worlds":list(replay.DEV_SEEDS),"n":len(samples),
        "from_actual_W0_observations":True,
        "target_is_synthetic_not_native":True,
        "marker_counts":dict(collections.Counter(r["ray2"] for r in samples)),
        "state_sha256":c.sha_bytes(("\n".join(c.canonical(r) for r in samples)+"\n").encode())})
    info=HfApi().model_info(c.MODEL,revision=c.REVISION)
    c.check(info.sha==c.REVISION,"upstream full model pin drift")
    snapshot_download(c.MODEL,revision=c.REVISION,local_dir=args.cache,max_workers=2,
      allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
       "generation_config.json","tokenizer*","special_tokens_map.json",
       "added_tokens.json","vocab.*","merges.txt","README.md","LICENSE*"])
    manifest={str(p.relative_to(args.cache)):c.sha(p)
       for p in sorted(args.cache.rglob("*")) if p.is_file() and ".cache" not in p.parts}
    c.check(any(k.endswith(".safetensors") for k in manifest),"full parent weights missing")
    c.save_json(out/"model-download.json",{"revision":info.sha,"sha256":manifest})
    tok=AutoTokenizer.from_pretrained(args.cache,local_files_only=True,trust_remote_code=False)
    ids=c.id_map(tok)
    base=AutoModelForCausalLM.from_pretrained(args.cache,local_files_only=True,
      trust_remote_code=False,dtype=torch.float32)
    c.check(base.config.model_type=="lfm2" and base.config.num_hidden_layers==16,
            "not pinned FULL LFM350")
    base.requires_grad_(False)
    base.eval()
    c.save_json(out/"preflight.json",{"full_parent":c.MODEL,
      "revision":c.REVISION,"token_ids":ids,"torch":torch.__version__,
      "frozen_base":True,"optimizers":0,"same_original_W0_worlds":True,
      "direct_matched_d1":"NOT EXECUTED"})
    model=None
    summaries={}
    for stage in STAGES:
        if stage=="stock":
            active=base
        else:
            if model is None:
                model=PeftModel.from_pretrained(base,args.adapter006,
                    adapter_name="gp350006",is_trainable=False)
                c.check(not any(p.requires_grad for p in model.parameters()),
                        "loaded LoRA unexpectedly trainable")
            if stage=="gp350008-step32":
                model.load_adapter(args.adapter32,adapter_name="gp350008-step32",
                    is_trainable=False)
            elif stage=="gp350008-step64":
                model.load_adapter(args.adapter64,adapter_name="gp350008-step64",
                    is_trainable=False)
            model.set_adapter(stage)
            active=model
        active.eval()
        probe=replay.probe(active,tok,ids,samples)
        c.save_json(out/(stage+"-heldout-probe.json"),probe)
        episodes=[]
        for size in SIZES:
            for seed in SEEDS:
                for family in c.FAMILIES:
                    episodes.append(previous.receipt(w0,active,tok,ids,size,seed,
                        family,stage,out,started,args.deadline))
                    print(c.canonical({"GP350009_EPISODE":episodes[-1]}),flush=True)
        summaries[stage]={"probe":{"n":probe["n"],
             "teacher_matches":probe["teacher_matches"],
             "raw_illegal":probe["raw_illegal"],
             "marker_target_matches":probe["marker_target_matches"]},
             "episodes":episodes,"stats":summarise(episodes)}
        # Keep every stage receipt immutable: save_json deliberately rejects overwrites.
        # The previous shared partial-summary.json failed after stage two.
        c.save_json(out/("partial-summary-"+stage+".json"),{
            "stage":stage,"stages_completed":list(summaries),
            "source_sha":os.getenv("GITHUB_SHA"),
            "no_success_claim_until_full_evaluation":True,
            "stats":summaries[stage]["stats"],
            "probe":summaries[stage]["probe"]})
    controls=[]
    for size in SIZES:
        for seed in SEEDS:
            for family in c.FAMILIES:
                for mode in ("random","wall","fidget"):
                    controls.append(previous.control(w0,size,seed,family,mode,out))
    images=[x for s in summaries.values() for x in s["episodes"]]+controls
    c.check(len(images)==84,"expected exactly 48 model + 36 control original PNGs")
    for x in images:
        c.check(c.sha(out/x["png"])==x["png_sha256"] and x["host_illegal"]==0,
                "PNG SHA mismatch or host-illegal action")
    stats={mode:summarise([r for r in controls if r["stage"]=="control-"+mode])
           for mode in ("random","wall","fidget")}
    summary={"schema":SCHEMA,"status":"complete",
      "source_sha":os.getenv("GITHUB_SHA"),"model":c.MODEL,
      "model_revision":c.REVISION,"stages":summaries,"controls":controls,
      "control_stats":stats,"original_pngs":len(images),
      "model_score_forwards":sum(s["stats"]["decisions"]+s["probe"]["n"]
                                 for s in summaries.values()),
      "adapter_hashes":{"GP350006":GP006_SHA,
         "GP350008_step32":GP008_32_SHA,"GP350008_step64":GP008_64_SHA},
      "optimizer_updates":0,"native":False,"vision_conditioned":False,
      "d1_matched":"NOT RUN","source_W0_only":True,
      "wall_seconds":time.monotonic()-started,
      "cpu_seconds":time.process_time(),
      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    c.save_json(out/"summary.json",summary)
    print(c.canonical({"GP350009_COMPLETE": {
      "model_score_forwards":summary["model_score_forwards"],
      "original_pngs":len(images),
      "probes":{k:v["probe"] for k,v in summaries.items()},
      "trajectory_stats":{k:v["stats"] for k,v in summaries.items()},
      "wall_seconds":summary["wall_seconds"]}}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--adapter006",type=Path)
    p.add_argument("--adapter32",type=Path)
    p.add_argument("--adapter64",type=Path)
    p.add_argument("--deadline",type=int,default=LIMIT)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    c.check(a.deadline==LIMIT,"immutable deadline changed")
    if a.self_test:
        selftest(a.w0)
    else:
        c.check(all(x is not None for x in (a.adapter006,a.adapter32,a.adapter64)),
                "missing adapter custody paths")
        try: run(a)
        except BaseException as exc:
            a.out.mkdir(parents=True,exist_ok=True)
            c.save_json(a.out/"failure.json",{"kind":type(exc).__name__,
             "message":str(exc),"source_sha":os.getenv("GITHUB_SHA"),
             "success":False})
            raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,previous.sig)
    signal.signal(signal.SIGINT,previous.sig)
    main()
