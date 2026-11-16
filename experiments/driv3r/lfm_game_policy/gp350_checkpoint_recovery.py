#!/usr/bin/env python3
"""GP350-009-R: complete a time-bounded four-checkpoint W0 comparison without
discarding the immutable, successfully scored episodes of failed GP350-009.

The original immutable Actions #43838034564 artifact supplies 36 complete
model episodes, 4 heldout probes, and a partial final stage. This independent
recovery byte-replays every retained W0 decision/PNG, compares the 3 overlapping
step64 episodes to fresh full-model inference, scores the remaining step64
episodes, and creates fresh procedural controls. It NEVER pretends to have
recomputed stock/GP006/step32 model logits in this new job.

Experimental model policy only. No native, Godot, gameplay, providers, training,
external API, or paid GPU writes. Failed originals remain immutable.
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
import gp350_checkpoint_selection_eval as original
import gp350_obstacle_exposure_eval as w0_eval
import gp350_observed_replay_train as replay

SCHEMA="driv3r.lfm350.gp350009-recovery/1"
PARENT_RUN=43838034564
PARENT_ARTIFACT=14213783262
PARENT_SHA="d8f017e81e553c9d92d6995d470e8a50e191238e"
PARENT_ARTIFACT_DIGEST="dce8cbbc7c5e0145f64974c4a3969d141fb50c6a807a1d14a1b7f8b55a64b805"
STAGES=original.STAGES
FINAL=STAGES[-1]
STEPS=original.STEPS
LIMIT=1800

def name(stage,size,seed,family):
    return f"{stage}-s{size}-{seed}-{family}"

def fail_if(condition,message):
    c.check(condition,message)

def check_probe(probe):
    fail_if(probe["n"]==72 and len(probe["rows"])==72,"bad old heldout probe length")
    rows=probe["rows"]
    fail_if(all(r["raw"]==c.ACTIONS[max(range(6),key=lambda i:r["six_logits"][i])]
        and r["match"]==(r["legal"]==r["target"])
        for r in rows),"retained probe logits/decisions invalid")
    # Raw legality depends on each W0 observation's action menu; it is
    # independently checked using the source W0 observations, not ACTIONS.
    fail_if(probe["teacher_matches"]==sum(r["match"] for r in rows),
            "heldout teacher score mismatch")
    fail_if(probe["marker_target_matches"]==
            {m:sum(r["match"] for r in rows if r["marker"]==m) for m in ".#+"},
            "marker heldout score mismatch")

def recover_episode(source,w0,stage,size,seed,family,out):
    stem=name(stage,size,seed,family)
    p=source/(stem+".json")
    png=source/(stem+".png")
    fail_if(p.is_file() and png.is_file(),"missing prior immutable episode "+stem)
    entry=json.loads(p.read_text())
    decisions=entry["decisions"]
    record=entry["record"]
    fail_if(entry["model"]==c.MODEL and entry["revision"]==c.REVISION,
            "prior full model identity mismatch")
    fail_if(len(decisions)==STEPS==len(record["events"]),
            "prior W0 decision/record mismatch "+stem)
    for d,e in zip(decisions,record["events"]):
        obs=d["observation"]
        z=d["six_logits"]
        fail_if(len(z)==6 and d["raw"]==c.ACTIONS[max(range(6),
              key=lambda i:z[i])],"old model score invalid")
        legal=max(obs["legal"],key=lambda a:z[c.ACTIONS.index(a)])
        fail_if(d["applied"]==legal==e["applied"]==e["raw_choice"]
            and e["raw_legal"] and d["raw_legal"]==(d["raw"] in obs["legal"]),
            "old host authority violated "+stem)
    actual=w0.png_bytes(record)
    rerun=w0.run(seed,mode="replay",family=family,size=size,steps=STEPS,
                 raw_actions=[d["applied"] for d in decisions])
    fail_if(actual==png.read_bytes()==w0.png_bytes(rerun),
            "old W0 byte replay mismatch "+stem)
    m=record["metrics"]
    row={"stage":stage,"size":size,"seed":seed,"family":family,
      "decisions":len(decisions),"unique_cells":m["unique_trail_cells"],
      "branches":m["branches"],"retraces":m["retraces"],
      "stop":m["terminated_by_stop"],
      "raw_illegal":sum(not d["raw_legal"] for d in decisions),
      "host_illegal":m["invalid_raw_choices"],
      "forward_two_cell_counts":dict(collections.Counter(d["forward_two_cell"] for d in decisions)),
      "obstacle_observations":sum("#" in "".join(d["observation"]["local_5x5"])
                                     for d in decisions),
      "choice_counts":dict(collections.Counter(d["applied"] for d in decisions)),
      "png":stem+".png","png_sha256":c.sha_bytes(actual)}
    (out/(stem+".png")).write_bytes(actual)
    c.save_json(out/(stem+".json"),entry)
    return row

def selftest(path):
    w0=c.load_w0(path)
    original.selftest(path)
    fail_if(STEPS==64 and STAGES==("stock","gp350006",
           "gp350008-step32","gp350008-step64"),"original protocol drift")
    fail_if(len(tuple((s,seed,fam) for s in original.SIZES
            for seed in original.SEEDS for fam in c.FAMILIES))==12,
            "original episode index drift")
    # Independent tiny host byte-replay test, without importing ML libraries.
    r=w0.run(9503,mode="wall",family="greebles",size=32,steps=STEPS)
    again=w0.run(9503,mode="replay",family="greebles",size=32,
                 steps=STEPS,raw_actions=[e["applied"] for e in r["events"]])
    fail_if(w0.png_bytes(r)==w0.png_bytes(again),"W0 byte replay failed")
    print(c.canonical({"GP350009_RECOVERY_SELFTEST":"PASS",
        "original_W0":True,"source_artifact":PARENT_ARTIFACT,
        "native":False,"optimizers":0}),flush=True)

def run(a):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi,snapshot_download
    torch.set_num_threads(2)
    started=time.monotonic()
    fail_if(not a.out.exists(),"output lease already exists")
    a.out.mkdir(parents=True)
    c.save_json(a.out/"lease.json",{"schema":SCHEMA,"source_sha":os.getenv("GITHUB_SHA"),
      "run_id":os.getenv("GITHUB_RUN_ID"),"attempt":os.getenv("GITHUB_RUN_ATTEMPT"),
      "old_run":PARENT_RUN,"old_artifact":PARENT_ARTIFACT,
      "old_artifact_zip_sha256":PARENT_ARTIFACT_DIGEST,
      "old_source_sha":PARENT_SHA,"model":c.MODEL,"revision":c.REVISION,
      "full_model_new_inference_stages":[FINAL],
      "reused_original_scored_stages":list(STAGES[:3]),
      "native":False,"vision":False,"optimizer_steps":0,"draft_hold":True})
    prior=json.loads((a.old/"lease.json").read_text())
    fail_if(str(prior["run_id"])==str(PARENT_RUN)
        and prior["source_sha"]==PARENT_SHA
        and prior["model"]==c.MODEL and prior["revision"]==c.REVISION,
        "prior artifact lineage/model mismatch")
    fail_if(c.sha(a.adapter/"adapter_model.safetensors")==original.GP008_64_SHA,
        "step64 immutable adapter hash mismatch")
    w0=c.load_w0(a.w0)
    selftest(a.w0)
    summaries={}
    for stage in STAGES[:3]:
        src_partial=json.loads((a.old/("partial-summary-"+stage+".json")).read_text())
        fail_if(src_partial["source_sha"]==PARENT_SHA and src_partial["stage"]==stage,
            "old stage partial source mismatch")
        probe=json.loads((a.old/(stage+"-heldout-probe.json")).read_text())
        check_probe(probe)
        rows=[recover_episode(a.old,w0,stage,size,seed,family,a.out)
              for size in original.SIZES for seed in original.SEEDS
              for family in c.FAMILIES]
        stats=original.summarise(rows)
        fail_if(stats==src_partial["stats"],"old stage metric/receipt mismatch "+stage)
        info={k:probe[k] for k in ("n","teacher_matches",
                                  "raw_illegal","marker_target_matches")}
        fail_if(info==src_partial["probe"],"old stage probe/partial mismatch "+stage)
        summaries[stage]={"probe":info,"episodes":rows,"stats":stats,
                          "provenance":"original run 43838034564; byte-replayed in recovery"}
        c.save_json(a.out/(stage+"-heldout-probe.json"),probe)
        print(c.canonical({"GP350009_RECOVERED_STAGE":{"stage":stage,
            "episodes":len(rows),"decisions":stats["decisions"],
            "original_png_sha_verified":len(rows)}}),flush=True)
    # Recover original final-stage probe and all overlapping original episodes.
    oldprobe=json.loads((a.old/(FINAL+"-heldout-probe.json")).read_text())
    check_probe(oldprobe)
    overlaps={}
    for size in original.SIZES:
        for seed in original.SEEDS:
            for family in c.FAMILIES:
                stem=name(FINAL,size,seed,family)
                if (a.old/(stem+".json")).is_file():
                    oldrow=recover_episode(a.old,w0,FINAL,size,seed,family,a.out)
                    overlaps[stem]=oldrow
    fail_if(len(overlaps)>=3,"prior failed run lacks expected three overlapping episodes")
    # Remove recovered overlap outputs; regenerate and compare independently.
    for stem in overlaps:
        (a.out/(stem+".png")).unlink()
        (a.out/(stem+".json")).unlink()

    info=HfApi().model_info(c.MODEL,revision=c.REVISION)
    fail_if(info.sha==c.REVISION,"pinned full LFM revision drift")
    snapshot_download(c.MODEL,revision=c.REVISION,local_dir=a.cache,max_workers=2,
      allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
       "generation_config.json","tokenizer*","special_tokens_map.json",
       "added_tokens.json","vocab.*","merges.txt","README.md","LICENSE*"])
    inventory={str(p.relative_to(a.cache)):c.sha(p)
       for p in sorted(a.cache.rglob("*")) if p.is_file() and ".cache" not in p.parts}
    fail_if(any(x.endswith(".safetensors") for x in inventory),"full weights missing")
    c.save_json(a.out/"model-download.json",{"revision":info.sha,"sha256":inventory})
    tok=AutoTokenizer.from_pretrained(a.cache,local_files_only=True,trust_remote_code=False)
    ids=c.id_map(tok)
    base=AutoModelForCausalLM.from_pretrained(a.cache,local_files_only=True,
      trust_remote_code=False,dtype=torch.float32)
    fail_if(base.config.model_type=="lfm2" and base.config.num_hidden_layers==16,
        "not full pinned LFM350")
    base.requires_grad_(False)
    base.eval()
    model=PeftModel.from_pretrained(base,a.adapter,is_trainable=False)
    fail_if(not any(p.requires_grad for p in model.parameters()),
        "inference LoRA unexpectedly trainable")
    model.eval()
    samples=replay.stratified(replay.observed(w0,replay.DEV_SEEDS))
    probe=replay.probe(model,tok,ids,samples)
    check_probe(probe)
    # Same W0 probe inventory and adapter; compare actual choices, not fragile
    # bit-exact floating logits across hosted CPU instances.
    prior_rows=oldprobe["rows"]
    fail_if(all((r["seed"],r["size"],r["family"],r["marker"],r["target"],r["legal"])
            ==(s["seed"],s["size"],s["family"],s["marker"],s["target"],s["legal"])
            for r,s in zip(probe["rows"],prior_rows)),
            "new stage64 model/probe choices differ from original artifact")
    c.save_json(a.out/(FINAL+"-heldout-probe.json"),probe)
    rows=[]
    for size in original.SIZES:
        for seed in original.SEEDS:
            for family in c.FAMILIES:
                row=w0_eval.receipt(w0,model,tok,ids,size,seed,family,
                                    FINAL,a.out,started,a.deadline)
                stem=name(FINAL,size,seed,family)
                if stem in overlaps:
                    fail_if(row==overlaps[stem],
                        "fresh model rollout diverges from immutable overlapping old "+stem)
                rows.append(row)
                print(c.canonical({"GP350009_RECOVERY_REAL_MODEL_EPISODE":row}),flush=True)
    summaries[FINAL]={"probe":{k:probe[k] for k in ("n","teacher_matches",
                  "raw_illegal","marker_target_matches")},"episodes":rows,
                  "stats":original.summarise(rows),
                  "provenance":"fresh full pinned LFM350+original step64 LoRA CPU forwards"}
    controls=[w0_eval.control(w0,size,seed,family,mode,a.out)
         for size in original.SIZES for seed in original.SEEDS
         for family in c.FAMILIES for mode in ("random","wall","fidget")]
    images=[r for stage in STAGES for r in summaries[stage]["episodes"]]+controls
    fail_if(len(images)==84 and all(r["host_illegal"]==0
            and c.sha(a.out/r["png"])==r["png_sha256"] for r in images),
            "not 84 legal original PNGs")
    controls_by_mode={mode:original.summarise([r for r in controls
                 if r["stage"]=="control-"+mode]) for mode in ("random","wall","fidget")}
    result={"schema":SCHEMA,"status":"complete","source_sha":os.getenv("GITHUB_SHA"),
       "source_original_run":PARENT_RUN,"source_original_sha":PARENT_SHA,
       "source_original_artifact":PARENT_ARTIFACT,
       "source_original_zip_sha256":PARENT_ARTIFACT_DIGEST,
       "model":c.MODEL,"revision":c.REVISION,
       "reused_stage_score_forwards":3*(12*STEPS+72),
       "fresh_step64_score_forwards":12*STEPS+72,
       "original_scored_total":4*(12*STEPS+72),
       "stage64_overlapping_pngs_exact":len(overlaps),
       "stages":summaries,"controls":controls,"control_stats":controls_by_mode,
       "original_pngs":84,"adapter_hashes":{"GP350006":original.GP006_SHA,
       "GP350008_step32":original.GP008_32_SHA,
       "GP350008_step64":original.GP008_64_SHA},
       "optimizer_updates":0,"native":False,"vision_conditioned":False,
       "d1_matched":"NOT RUN","source_W0_only":True,
       "original_old_stages_reused_not_rescored":True,
       "wall_seconds":time.monotonic()-started,
       "cpu_seconds":time.process_time(),
       "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    c.save_json(a.out/"summary.json",result)
    print(c.canonical({"GP350009_RECOVERY_COMPLETE":{"stages":
       {k:{"probe":v["probe"],"stats":v["stats"]} for k,v in summaries.items()},
       "pngs":84,"overlaps":len(overlaps),
       "fresh_step64_forwards":result["fresh_step64_score_forwards"],
       "wall_seconds":result["wall_seconds"]}}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--old",type=Path)
    p.add_argument("--adapter",type=Path)
    p.add_argument("--cache",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--deadline",type=int,default=LIMIT)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    fail_if(a.deadline==LIMIT,"deadline recipe drift")
    if a.self_test:
        selftest(a.w0)
    else:
        fail_if(all(x is not None for x in (a.old,a.adapter,a.cache,a.out)),
            "missing immutable source/adapter/out paths")
        try:run(a)
        except BaseException as exc:
            a.out.mkdir(parents=True,exist_ok=True)
            c.save_json(a.out/"failure.json",{"kind":type(exc).__name__,
               "message":str(exc),"source_sha":os.getenv("GITHUB_SHA"),
               "old_artifact":PARENT_ARTIFACT,"no_success_claim":True})
            raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,w0_eval.sig)
    signal.signal(signal.SIGINT,w0_eval.sig)
    main()
