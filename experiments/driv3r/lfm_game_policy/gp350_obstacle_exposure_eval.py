#!/usr/bin/env python3
"""GP350-007: genuine full-LFM350 checkpoint-only, obstacle-exposure policy audit.

Research-only CyberSand-like W0 host. Never native Godot, vision, game writes,
or optimizer; inherits exact immutable GP350-006 rank8 LoRA checkpoint.
Smaller W0 maps expose boundary/obstacle pressure previously absent in all
GP350-006's original 24-turn straight-line galleries.
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
import gp350_paired_margin_train as previous

SCHEMA = "driv3r.lfm350.gp350007-obstacle-exposure/1"
SOURCE_RUN = 43806503356
SOURCE_ARTIFACT = 14199280586
ADAPTER_SHA = "436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c"
SIZES = (32, 48)
SEEDS = (9503, 9521)
STEPS = 64
LIMIT = 1800
STOP = False

def sig(_s, _f):
    global STOP
    STOP = True

def self_test(w0_path):
    w0 = c.load_w0(w0_path)
    assert SIZES == (32,48) and STEPS == 64
    for size in SIZES:
        r = w0.run(SEEDS[0], mode="wall", family="greebles", size=size, steps=STEPS)
        assert len(r["events"]) and r["metrics"]["invalid_raw_choices"] == 0
        assert all(e["applied"] in e["observation"]["legal"] for e in r["events"])
        a = [e["applied"] for e in r["events"]]
        replay = w0.run(SEEDS[0], mode="replay", family="greebles",
                        size=size, steps=STEPS, raw_actions=a)
        assert w0.png_bytes(r) == w0.png_bytes(replay)
        assert all(previous.ray2(e["observation"]) in ".#+" for e in r["events"])
    print(c.canonical({"GP350007_SELFTEST":"PASS","sizes":SIZES,
                       "W0_original_host":True,"native":False}), flush=True)

def receipt(w0, model, tok, ids, size, seed, family, stage, out, started, deadline):
    import torch
    decisions = []
    def choose(obs):
        c.check(not STOP and time.monotonic()-started < deadline-120,
                "bounded checkpoint-only inference deadline")
        with torch.inference_mode():
            z = previous.rich_score(model, tok, ids, obs)
        raw = c.ACTIONS[int(z.argmax().item())]
        chosen = max(obs["legal"], key=lambda a:float(z[c.ACTIONS.index(a)]))
        decisions.append({"observation":obs,"raw":raw,"raw_legal":raw in obs["legal"],
                          "applied":chosen,"six_logits":[float(v) for v in z],
                          "forward_two_cell":previous.ray2(obs)})
        return chosen
    record = w0.run(seed, mode="callback", family=family, size=size,
                    steps=STEPS, selector=choose)
    c.check(len(record["events"])==len(decisions), "model/W0 event count mismatch")
    for d,e in zip(decisions,record["events"]):
        c.check(d["applied"] in d["observation"]["legal"] and
                d["applied"]==e["applied"]==e["raw_choice"] and e["raw_legal"],
                "host-only legal selection violated")
        c.check(d["raw"] == c.ACTIONS[max(range(6),key=lambda i:d["six_logits"][i])],
                "retained six-logit argmax mismatch")
        c.check(d["applied"]==max(d["observation"]["legal"],
                key=lambda a:d["six_logits"][c.ACTIONS.index(a)]),
                "retained legal mask mismatch")
    replay = w0.run(seed,mode="replay",family=family,size=size,
                    steps=STEPS,raw_actions=[d["applied"] for d in decisions])
    original = w0.png_bytes(record)
    c.check(w0.png_bytes(replay)==original,"original W0 replay PNG mismatch")
    stem = f"{stage}-s{size}-{seed}-{family}"
    (out / (stem+".png")).write_bytes(original)
    c.save_json(out/(stem+".json"),{"decisions":decisions,
                "record":record,"model":c.MODEL,"revision":c.REVISION})
    m = record["metrics"]
    return {"stage":stage,"size":size,"seed":seed,"family":family,
            "decisions":len(decisions),"unique_cells":m["unique_trail_cells"],
            "branches":m["branches"],"retraces":m["retraces"],
            "stop":m["terminated_by_stop"],
            "raw_illegal":sum(not d["raw_legal"] for d in decisions),
            "host_illegal":m["invalid_raw_choices"],
            "forward_two_cell_counts":dict(collections.Counter(
                    d["forward_two_cell"] for d in decisions)),
            "obstacle_observations":sum("#" in "".join(d["observation"]["local_5x5"])
                  for d in decisions),"choice_counts":dict(collections.Counter(
                    d["applied"] for d in decisions)),
            "png":stem+".png","png_sha256":c.sha_bytes(original)}

def control(w0, size, seed, family, mode, out):
    record=w0.run(seed,mode=mode,family=family,size=size,steps=STEPS)
    c.check(record["metrics"]["invalid_raw_choices"]==0,"invalid W0 baseline")
    stem=f"control-{mode}-s{size}-{seed}-{family}"
    data=w0.png_bytes(record)
    (out/(stem+".png")).write_bytes(data)
    c.save_json(out/(stem+".json"), record)
    return {"stage":"control-"+mode,"size":size,"seed":seed,"family":family,
            "decisions":len(record["events"]),
            "unique_cells":record["metrics"]["unique_trail_cells"],
            "branches":record["metrics"]["branches"],
            "retraces":record["metrics"]["retraces"],
            "stop":record["metrics"]["terminated_by_stop"],
            "host_illegal":record["metrics"]["invalid_raw_choices"],
            "png":stem+".png","png_sha256":c.sha_bytes(data)}

def run(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi,snapshot_download
    torch.set_num_threads(2)
    started=time.monotonic()
    out=args.out
    c.check(not out.exists(),"immutable GP350007 output lease reused")
    out.mkdir(parents=True)
    c.save_json(out/"lease.json",{"schema":SCHEMA,"source_sha":os.getenv("GITHUB_SHA"),
       "run_id":os.getenv("GITHUB_RUN_ID"),"attempt":os.getenv("GITHUB_RUN_ATTEMPT"),
       "model":c.MODEL,"revision":c.REVISION,"adapter_sha256":ADAPTER_SHA,
       "parent_artifact":SOURCE_ARTIFACT,"parent_run":SOURCE_RUN,
       "optimizers":0,"native":False,"draft_hold":True})
    w0=c.load_w0(args.w0)
    self_test(args.w0)
    c.check(c.sha(args.adapter/"adapter_model.safetensors")==ADAPTER_SHA,
            "GP350006 adapter identity/custody drift")
    info=HfApi().model_info(c.MODEL,revision=c.REVISION)
    c.check(info.sha==c.REVISION,"parent LFM revision drift")
    snapshot_download(c.MODEL,revision=c.REVISION,local_dir=args.cache,max_workers=2,
       allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
        "generation_config.json","tokenizer*","special_tokens_map.json",
        "added_tokens.json","vocab.*","merges.txt","README.md","LICENSE*"])
    inventory={str(p.relative_to(args.cache)):c.sha(p) for p in sorted(args.cache.rglob("*"))
               if p.is_file() and ".cache" not in p.parts}
    c.check(any(x.endswith(".safetensors") for x in inventory),"full weights missing")
    c.save_json(out/"model-download.json",{"model":c.MODEL,"revision":info.sha,
                "sha256":inventory})
    tok=AutoTokenizer.from_pretrained(args.cache,local_files_only=True,
                                      trust_remote_code=False)
    ids=c.id_map(tok)
    model=AutoModelForCausalLM.from_pretrained(args.cache,local_files_only=True,
                             trust_remote_code=False,dtype=torch.float32)
    c.check(model.config.model_type=="lfm2" and model.config.num_hidden_layers==16,
            "not full pinned LFM350")
    model.requires_grad_(False)
    model.eval()
    c.save_json(out/"preflight.json",{"token_ids":ids,"full_parent":c.MODEL,
        "adapter_sha256":ADAPTER_SHA,"optimizer_steps":0,
        "frozen_model":not any(p.requires_grad for p in model.parameters()),
        "comparison":"same source W0 worlds and derived-ray prompt"})
    stock=[]
    for size in SIZES:
        for seed in SEEDS:
            for family in c.FAMILIES:
                stock.append(receipt(w0,model,tok,ids,size,seed,family,
                                     "stock",out,started,args.deadline))
                print(c.canonical({"GP350007_STOCK":stock[-1]}),flush=True)
    # GP350-006 checkpoint-only continuation: NOT a new trained optimizer and
    # NOT an unmodified stock model. Official pinned base stays frozen.
    trained_model=PeftModel.from_pretrained(model,args.adapter,is_trainable=False)
    c.check(not any(p.requires_grad for p in trained_model.parameters()),
            "checkpoint-only model unexpectedly trainable")
    trained_model.eval()
    trained=[]
    for size in SIZES:
        for seed in SEEDS:
            for family in c.FAMILIES:
                trained.append(receipt(w0,trained_model,tok,ids,size,seed,family,
                                       "gp350006-step96",out,started,args.deadline))
                print(c.canonical({"GP350007_TRAINED":trained[-1]}),flush=True)
    controls=[]
    for size in SIZES:
        for seed in SEEDS:
            for family in c.FAMILIES:
                for mode in ("random","wall","fidget"):
                    controls.append(control(w0,size,seed,family,mode,out))
    images=stock+trained+controls
    c.check(len(images)==60 and all(c.sha(out/x["png"])==x["png_sha256"]
                                  for x in images),"original W0 PNG integrity failure")
    def compact(items):
        return {"episodes":len(items),"decisions":sum(x["decisions"] for x in items),
          "mean_unique_cells":sum(x["unique_cells"] for x in items)/len(items),
          "total_branches":sum(x["branches"] for x in items),
          "total_retraces":sum(x["retraces"] for x in items),
          "early_stops":sum(bool(x["stop"]) for x in items),
          "total_host_illegal":sum(x["host_illegal"] for x in items),
          "total_raw_illegal":sum(x.get("raw_illegal",0) for x in items),
          "obstacle_observations":sum(x.get("obstacle_observations",0) for x in items),
          "ray2_counts":dict(sum((collections.Counter(x.get("forward_two_cell_counts",{}))
                                 for x in items),collections.Counter()))}
    c.save_json(out/"summary.json",{"schema":SCHEMA,"status":"complete",
       "source_sha":os.getenv("GITHUB_SHA"),"parent_model":c.MODEL,
       "model_revision":c.REVISION,"adapter_sha256":ADAPTER_SHA,
       "parent_artifact":SOURCE_ARTIFACT,"steps_per_episode":STEPS,
       "sizes":SIZES,"world_seeds":SEEDS,
       "stock":stock,"trained":trained,"controls":controls,
       "stats":{"stock":compact(stock),"trained":compact(trained),
                "random":compact([x for x in controls if x["stage"]=="control-random"]),
                "wall":compact([x for x in controls if x["stage"]=="control-wall"]),
                "fidget":compact([x for x in controls if x["stage"]=="control-fidget"])},
       "original_pngs":len(images),"optimizer_steps":0,
       "native_paint":False,"vision_conditioned":False,
       "host":"original synthetic W0 only","d1_matched":"NOT RUN",
       "wall_seconds":time.monotonic()-started,
       "process_cpu_seconds":time.process_time(),
       "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--w0",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--adapter",type=Path)
    parser.add_argument("--deadline",type=int,default=LIMIT)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    c.check(args.deadline==LIMIT,"bounded deadline altered")
    if args.self_test:
        self_test(args.w0)
    else:
        c.check(args.adapter is not None and args.adapter.is_dir(),"immutable adapter absent")
        try: run(args)
        except BaseException as exc:
            args.out.mkdir(parents=True,exist_ok=True)
            (args.out/"failure.json").write_text(c.canonical({
                "kind":type(exc).__name__,"message":str(exc),
                "source_sha":os.getenv("GITHUB_SHA"),"success":False})+"\n")
            raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,sig)
    signal.signal(signal.SIGINT,sig)
    main()
