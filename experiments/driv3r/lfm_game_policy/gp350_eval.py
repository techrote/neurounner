#!/usr/bin/env python3
"""GP350-002: genuine checkpoint-only, matched-world family counterfactuals.

Full pinned LiquidAI/LFM2.5-350M for every forward. No optimizer, no native.
Each family sees identical initial seed/obstacles; all 12-turn post-divergence
observations are freshly scored on-policy. STOP always legal and available.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import sys

import lfm350 as core

SCHEMA="driv3r.lfm350.game-policy-counterfactual/1"
PRODUCER_RUN=43789213200
PRODUCER_HEAD="f2d2d6a4b70f97de1d633ae43f37cd5b307d4978"
ADAPTER_SHA="53fc6646b586712a794707ecf649167ea94f3af8be82963380886dc63ac9d72d"
SEEDS=(9503,9521,9539)
FAMILIES=core.FAMILIES
TURNS=12
ACTIONS=core.ACTIONS

def check(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score_to_probs(values):
    top=max(values); w=[math.exp(z-top) for z in values]
    total=sum(w)
    return [v/total for v in w]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--producer",type=Path,required=True)
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--model-cache",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    check(not args.out.exists(),"immutable W2 output exists")
    args.out.mkdir(parents=True)
    started=time.monotonic()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi,snapshot_download
    torch.set_num_threads(2)
    w0=core.load_w0(args.w0)
    original=json.loads((args.producer/"summary.json").read_text())
    download=json.loads((args.producer/"download.json").read_text())
    check(original["source_sha"]==PRODUCER_HEAD and original["status"]=="completed" and
          original["updates"]==24 and original["backward_passes"]==48,"producer identity wrong")
    check(download["revision"]==core.REVISION and original["model_revision"]==core.REVISION,
          "producer model mismatch")
    cp=args.producer/"adapter-step-24"/"adapter_model.safetensors"
    check(sha(cp)==ADAPTER_SHA,"producer adapter checkpoint modified")
    for n,h in original["adapter_files_sha256"].items():
        check(sha(args.producer/n)==h,"producer adapter file tampered: "+n)
    info=HfApi().model_info(core.MODEL,revision=core.REVISION)
    check(info.sha==core.REVISION,"upstream model changed")
    snapshot_download(core.MODEL,revision=core.REVISION,local_dir=args.model_cache,max_workers=2,
        allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
                        "generation_config.json","tokenizer*","special_tokens_map.json",
                        "added_tokens.json","vocab.*","merges.txt","README.md","LICENSE*"])
    for name,hashval in download["sha256"].items():
        check(sha(args.model_cache/name)==hashval,"model identity file mismatch: "+name)
    tokenizer=AutoTokenizer.from_pretrained(args.model_cache,trust_remote_code=False,local_files_only=True)
    ids=core.id_map(tokenizer)
    base=AutoModelForCausalLM.from_pretrained(args.model_cache,trust_remote_code=False,
                                              local_files_only=True,dtype=torch.float32)
    check(base.config.model_type=="lfm2","unexpected full model")
    # Pure inference: eval() alone does not clear PyTorch parameters.requires_grad.
    base.requires_grad_(False)
    base.eval()
    output=[]
    first_by_world={}
    for stage in ("stock","trained"):
        if stage=="trained":
            model=PeftModel.from_pretrained(base,args.producer/"adapter-step-24",is_trainable=False)
        else:
            model=base
        check(not any(x.requires_grad for x in model.parameters()),"inference accidentally trainable")
        model.eval()
        for seed in SEEDS:
            for family in FAMILIES:
                choices=[]
                def choose(obs):
                    with torch.inference_mode():
                        values=core.score(model,tokenizer,ids,obs)
                    arr=[float(x) for x in values]
                    check(all(math.isfinite(v) for v in arr),"nonfinite score")
                    raw=ACTIONS[max(range(len(ACTIONS)),key=lambda i:arr[i])]
                    applied=max(obs["legal"],key=lambda a:arr[ACTIONS.index(a)])
                    choices.append({"observation":obs,"raw_argmax":raw,"raw_legal":raw in obs["legal"],
                                    "selected":applied,"six_scores":arr,
                                    "six_probabilities":score_to_probs(arr)})
                    return applied
                record=w0.run(seed,mode="callback",family=family,steps=TURNS,selector=choose)
                check(len(choices)==len(record["events"]),"model/W0 forward mismatch")
                check(all(c["selected"]==e["applied"] and e["raw_choice"]==c["selected"]
                          for c,e in zip(choices,record["events"])),"host legal action mismatch")
                replay=w0.run(seed,mode="replay",family=family,steps=TURNS,
                              raw_actions=[c["selected"] for c in choices])
                png=w0.png_bytes(record)
                check(w0.png_bytes(replay)==png,"new original PNG not replayable")
                stem=f"{stage}-{seed}-{family}"
                (args.out/(stem+".png")).write_bytes(png)
                core.save_json(args.out/(stem+".json"),{"record":record,"choices":choices})
                row={"stage":stage,"seed":seed,"family":family,
                     "steps":len(choices),"first_raw":choices[0]["raw_argmax"],
                     "first_host":choices[0]["selected"],"first_probabilities":choices[0]["six_probabilities"],
                     "all_unmasked_illegal":sum(not c["raw_legal"] for c in choices),
                     "trajectory_actions":[c["selected"] for c in choices],
                     "unique_trail_cells":record["metrics"]["unique_trail_cells"],
                     "branches":record["metrics"]["branches"],
                     "retraces":record["metrics"]["retraces"],
                     "stopped":record["metrics"]["terminated_by_stop"],
                     "png":stem+".png","png_sha256":core.sha_bytes(png)}
                output.append(row)
                check(row["steps"]<=TURNS,"overflow")
                if (stage,seed) not in first_by_world:
                    first_by_world[(stage,seed)]=choices[0]["observation"]
                else:
                    a=choices[0]["observation"]
                    b=first_by_world[(stage,seed)]
                    check({k:v for k,v in a.items() if k!="family"}==
                          {k:v for k,v in b.items() if k!="family"},
                          "family counterfactual changed initial geometry/history/legal menu")
        # Remove adapter before stock reuse not necessary: stock precedes trained.
    result=[]
    for stage in ("stock","trained"):
        for seed in SEEDS:
            group=[r for r in output if r["stage"]==stage and r["seed"]==seed]
            check(len(group)==3,"missing original same-world family triple")
            firsts=[r["first_host"] for r in group]
            hashes=[r["png_sha256"] for r in group]
            pairs=[]
            for i in range(3):
                for j in range(i+1,3):
                    a,b=group[i],group[j]
                    prefix=0
                    for x,y in zip(a["trajectory_actions"],b["trajectory_actions"]):
                        if x!=y:break
                        prefix+=1
                    tv=.5*sum(abs(x-y) for x,y in
                               zip(a["first_probabilities"],b["first_probabilities"]))
                    pairs.append({"families":[a["family"],b["family"]],
                                  "matching_action_prefix":prefix,
                                  "first_decision_diverges":prefix==0,
                                  "first_tvd":tv,"png_distinct":a["png_sha256"]!=b["png_sha256"]})
            result.append({"stage":stage,"seed":seed,"distinct_first_actions":len(set(firsts)),
                           "distinct_pngs":len(set(hashes)),"pairs":pairs})
    report={"schema":SCHEMA,"source_sha":os.getenv("GITHUB_SHA"),
            "producer_run_id":PRODUCER_RUN,"producer_source":PRODUCER_HEAD,
            "model":core.MODEL,"revision":core.REVISION,"adapter_sha256":ADAPTER_SHA,
            "new_optimizer_updates":0,"native_PAINT":0,
            "full_model_genuine_scored_forwards":sum(x["steps"] for x in output),
            "model_episodes":len(output),"worlds":list(SEEDS),
            "first_state_pairs_truly_identical_except_family":True,
            "all_true_on_policy_post_divergence":True,"original_pngs":len(output),
            "original_episodes":output,"counterfactual_groups":result,
            "wall_seconds":time.monotonic()-started,
            "cpu_seconds":time.process_time(),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "synthetic_W0_not_native":True}
    core.save_json(args.out/"summary.json",report)
    print(core.canonical({"event":"GP350-002-COMPLETE","episodes":len(output),
                          "forwards":report["full_model_genuine_scored_forwards"],
                          "distinct_first_actions":[{"stage":r["stage"],"seed":r["seed"],
                          "count":r["distinct_first_actions"]} for r in result]}),flush=True)

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        if "--out" in sys.argv:
            path=Path(sys.argv[sys.argv.index("--out")+1]);path.mkdir(parents=True,exist_ok=True)
            (path/"failure.json").write_text(json.dumps(
                {"type":type(exc).__name__,"message":str(exc),"no_success_claim":True})+"\n")
        raise
