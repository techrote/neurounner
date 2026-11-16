#!/usr/bin/env python3
"""W9: learned geometric residual for *pinned, frozen* root-Instruct SPROCKET.

Distinct from W6-W8 LoRA: 360 genuine optimizer steps update ONLY a small
auxiliary W0-observation MLP; original root is not fine-tuned. At inference the
root's actual six action-token scores are fused with the head, then the pinned
Python W0 host applies legal/STOP enforcement. No native CyberSand claims.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, resource, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "training"), str(HERE)]
import worm_v4 as objective
from common import profile, sha256_file, write_json
from lane_adapters import adapter_for
from numerics import configure_cpu

ACTIONS = tuple(objective.ACTIONS)
GRID = ".#+@"
FAMILIES = tuple(objective.FAMILIES)
FEATURE_VERSION = "sprocket-w9-geom-residual-v1"
TRAIN_SEEDS = (149, 151, 157)
EVAL_SEEDS = (1601, 1613)
FUSION_SEED = 1601
UPDATES_PER_STAGE = 120
FUSION_WEIGHT = 2.0
TEMP = 2.25

def features(obs):
    """Only current W0 observation fields: no future positions or teacher label."""
    if obs["family"] not in FAMILIES or obs["heading"] not in range(4):
        raise ValueError("wrong family/heading")
    grid = obs["local_5x5"]
    if len(grid) != 5 or any(len(row) != 5 or set(row) - set(GRID) for row in grid) or grid[2][2] != "@":
        raise ValueError("invalid 5x5")
    if set(obs["legal"]) - set(ACTIONS) or "STOP" not in obs["legal"]:
        raise ValueError("invalid legal")
    v = [float(c == symbol) for row in grid for c in row for symbol in GRID]
    v += [float(obs["heading"] == i) for i in range(4)]
    v += [float(obs["family"] == f) for f in FAMILIES]
    tick = obs["tick"]
    v += [math.sin(2*math.pi*tick/p) for p in (6, 9, 12, 13, 15, 17, 32)]
    v += [math.cos(2*math.pi*tick/p) for p in (6, 9, 12, 13, 15, 17, 32)]
    v += [min(tick/128, 1.0), obs["x"]/96, obs["y"]/96]
    v += [float(a in obs["legal"]) for a in ACTIONS]
    recent = obs["last_actions"]
    v += [sum(a == move for a in recent)/4 for move in ACTIONS]
    v += [float(bool(recent) and recent[-1] == move) for move in ACTIONS]
    if len(v) != 142:
        raise AssertionError("unexpected W9 feature dimensionality: "+str(len(v)))
    return v

def model_class(torch):
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(142,64), torch.nn.Tanh(),
                torch.nn.Linear(64,64), torch.nn.Tanh(),
                torch.nn.Linear(64,6))
        def forward(self, x):
            return self.layers(x)
    return Head

def as_tensors(torch, rows):
    x = torch.tensor([features(r["observation"]) for r in rows],dtype=torch.float32)
    y = torch.tensor([ACTIONS.index(r["action"]) for r in rows],dtype=torch.long)
    return x,y

def balanced_indices(rows, n, rng):
    categories = {}
    for i,r in enumerate(rows):
        categories.setdefault((r["family"],r["action"]), []).append(i)
    keys = sorted(categories)
    return [rng.choice(categories[keys[i % len(keys)]]) for i in range(n)]

def accuracy(torch,head,rows):
    with torch.no_grad():
        x,y = as_tensors(torch,rows)
        scores = head(x)
        guess = scores.argmax(dim=1)
        legal = torch.tensor([[a in r["legal"] for a in ACTIONS] for r in rows],dtype=torch.bool)
        masked = scores.masked_fill(~legal,-1e9).argmax(dim=1)
        return {"rows":len(rows),
                "raw_top_accuracy":round((guess==y).float().mean().item(),6),
                "host_masked_teacher_accuracy":round((masked==y).float().mean().item(),6),
                "raw_illegal_top":int((~legal.gather(1,guess[:,None])).sum().item()),
                "predictions":dict(Counter(ACTIONS[i] for i in guess.tolist()))}

def pair_metrics(torch,head,pairs):
    with torch.no_grad():
        a,b = zip(*pairs[:min(36,len(pairs))])
        sa = head(as_tensors(torch,a)[0])
        sb = head(as_tensors(torch,b)[0])
        ia = as_tensors(torch,a)[1]
        ib = as_tensors(torch,b)[1]
        switched = (sa.argmax(1)!=sb.argmax(1))
        both = (sa.argmax(1)==ia)&(sb.argmax(1)==ib)
        contrast = ((sa.gather(1,ia[:,None])-sa.gather(1,ib[:,None]))
                    -(sb.gather(1,ia[:,None])-sb.gather(1,ib[:,None])))
        return {"pairs":len(a),"unmasked_switches":int(switched.sum()),
                "both_teacher_argmax":int(both.sum()),
                "mean_paired_delta":round(float(contrast.mean()),6)}

def save_stage(torch,head,opt,out,name,meta):
    path = out / (name+".pt")
    torch.save({"version":FEATURE_VERSION, "stage":name,
                "head":head.state_dict(),"optimizer":opt.state_dict(),
                "torch_rng_state":torch.random.get_rng_state(),
                "metadata":meta},path)
    return {"file":path.name,"bytes":path.stat().st_size,"sha256":sha256_file(path)}

def train_head(torch,objective_data,support,out):
    rows = objective_data["train"]
    eval_rows = objective_data["eval"]
    import worm_train_v8
    pairs = worm_train_v8.counterfactual_pairs(rows,support,"train")
    dev_pairs = worm_train_v8.counterfactual_pairs(eval_rows,support,"eval")
    torch.manual_seed(9612)
    rng = random.Random(9612)
    head = model_class(torch)()
    opt = torch.optim.AdamW(head.parameters(),lr=0.002,weight_decay=0.001)
    stages = []
    histories = []
    src_x,src_y = as_tensors(torch,rows)
    paired_a, paired_b = zip(*pairs)
    pair_x0,pair_y0 = as_tensors(torch,paired_a)
    pair_x1,pair_y1 = as_tensors(torch,paired_b)
    for stage in ("imitation","paired-counterfactual","onpolicy-replay"):
        if stage == "onpolicy-replay":
            replay = []
            for seed in TRAIN_SEEDS:
                for family in FAMILIES:
                    def select(obs):
                        with torch.no_grad():
                            scores = head(torch.tensor([features(obs)],dtype=torch.float32))[0]
                            legal = [a for a in obs["legal"] if a != "STOP" or obs["tick"] >= 112]
                            return max(legal,key=lambda a:float(scores[ACTIONS.index(a)]))
                    trace = support.worm.run(seed,"callback",family,128,selector=select)
                    for event in trace["events"]:
                        obs = event["observation"]
                        replay.append({"observation":obs,"action":support.teacher_choice(obs),
                                       "family":family,"seed":seed})
            if len(replay) < 64:
                raise RuntimeError("on-policy W0 replay unexpectedly short")
            rx,ry = as_tensors(torch,replay)
            (out/"onpolicy-replay-manifest.json").write_text(json.dumps({
                "synthetic_not_native":True,"seed_groups":list(TRAIN_SEEDS),
                "rows":len(replay),"actions":dict(Counter(r["action"] for r in replay)),
                "observations_sha256":hashlib.sha256(json.dumps(
                    [r["observation"] for r in replay],sort_keys=True,
                    separators=(",",":")).encode()).hexdigest()},indent=2)+"\n")
        for step in range(UPDATES_PER_STAGE):
            opt.zero_grad(set_to_none=True)
            if stage == "imitation":
                ids = balanced_indices(rows,64,rng)
                loss = torch.nn.functional.cross_entropy(head(src_x[ids]),src_y[ids])
            elif stage == "paired-counterfactual":
                ids = [rng.randrange(len(pairs)) for _ in range(32)]
                s0,s1 = head(pair_x0[ids]),head(pair_x1[ids])
                t0,t1 = pair_y0[ids],pair_y1[ids]
                ce = (torch.nn.functional.cross_entropy(s0,t0)+
                      torch.nn.functional.cross_entropy(s1,t1))*0.5
                diff = ((s0.gather(1,t0[:,None])-s0.gather(1,t1[:,None]))
                        -(s1.gather(1,t0[:,None])-s1.gather(1,t1[:,None]))).squeeze(1)
                loss = 0.55*ce+0.45*torch.nn.functional.softplus(2.0-diff).mean()
            else:
                rid = [rng.randrange(len(replay)) for _ in range(32)]
                tid = balanced_indices(rows,32,rng)
                loss = (torch.nn.functional.cross_entropy(head(rx[rid]),ry[rid])+
                        torch.nn.functional.cross_entropy(head(src_x[tid]),src_y[tid]))*0.5
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite genuine W9 loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(head.parameters(),3.0)
            if not torch.isfinite(grad).item() or grad.item() == 0:
                raise ValueError("zero/nonfinite W9 gradient")
            opt.step()
            if (step+1)%30 == 0:
                histories.append({"stage":stage,"update_in_stage":step+1,
                                  "global_step":len(histories)*30+30,
                                  "loss":round(float(loss.detach()),6),
                                  "grad_norm":round(float(grad),6)})
        assessment={"stage":stage,"updates":UPDATES_PER_STAGE,
                    "development":accuracy(torch,head,eval_rows),
                    "local_counterfactual":pair_metrics(torch,head,dev_pairs)}
        assessment["checkpoint"] = save_stage(torch,head,opt,out,stage,assessment)
        stages.append(assessment)
        write_json(out/"stage-progress.json",{"completed":stages,
                   "genuine_optimizer_updates":len(stages)*UPDATES_PER_STAGE})
    return head,stages,histories

def rollout_head(w0,torch,head,seed,family,steps=128):
    decisions=[]
    def selector(obs):
        with torch.no_grad():
            scores=head(torch.tensor([features(obs)],dtype=torch.float32))[0].tolist()
        legal=[a for a in obs["legal"] if a!="STOP" or obs["tick"]>=112]
        choice=max(legal,key=lambda a:scores[ACTIONS.index(a)])
        decisions.append({"tick":obs["tick"],"action":choice,
                          "legal":obs["legal"],"head_logits":scores})
        return choice
    rec=w0.run(seed,"callback",family,steps,selector=selector)
    assert all(e["raw_legal"] for e in rec["events"])
    rec["mode"]="sprocket-w9-aux-head"
    rec["engine"]="pinned-Python-W0-aux-head-not-native"
    rec["head_decisions"]=decisions
    return rec

def model_rollout(w0,torch,head,root,tok,adapter,token_ids,seed,family,fused):
    from worm_score_v5_eval import host_draw, sample_scores
    decisions=[]
    def selector(obs):
        prompt=objective.prompt(obs)
        prefix=adapter.encode_prompt(tok,prompt,objective.SYSTEM)
        x=torch.tensor([prefix],dtype=torch.long)
        with torch.no_grad():
            base=root(input_ids=x, attention_mask=torch.ones_like(x),use_cache=False).logits[0,-1,token_ids].float().tolist()
            extra=head(torch.tensor([features(obs)],dtype=torch.float32))[0].tolist()
        shifted=[base[i]+FUSION_WEIGHT*(extra[i]-sum(extra)/6) for i in range(6)] if fused else base
        legal=[a for a in obs["legal"] if a!="STOP" or obs["tick"]>=22]
        draw=host_draw(family,seed,obs["tick"],TEMP)
        choice,_=sample_scores(shifted,legal,TEMP,draw)
        decisions.append({"tick":obs["tick"],"root_six_logits":base,
                          "head_six_logits":extra,"fused":fused,
                          "selected":choice,"host_legal":legal,
                          "root_real_forward":True,"host_draw":draw})
        return choice
    rec=w0.run(seed,"callback",family,24,selector=selector)
    assert all(e["raw_legal"] and e["applied"]==d["selected"]
               for e,d in zip(rec["events"],decisions))
    rec["mode"]="sprocket-w9-fused" if fused else "sprocket-root-untuned"
    rec["engine"]="pinned-Python-W0-real-root-instruct-neural-forward-not-native"
    rec["model_score_decisions"]=decisions
    return rec

def main(args):
    started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    numerical,runtime=configure_cpu(1)
    import torch
    torch.set_num_threads(1)
    p=profile("sprocket",require_supported=True)
    assert p["model_id"]=="kandivault/sprocket-500m"
    assert p["revision"]=="eb157d4ddddc4619e661d4a2426bb2dd000fc668"
    assert p["subfolder"]=="" and p["framing"]["eos_token_id"]==5
    support=objective.load_support(args.support,args.w0)
    data,receipt=objective.dataset(support,args.output/"dataset")
    assert receipt["literal_prompt_intersection"]==0
    w0=support.worm
    preflight={"schema":FEATURE_VERSION,"source_sha":os.getenv("GITHUB_SHA","local"),
       "run_id":os.getenv("GITHUB_RUN_ID","local"),"run_attempt":os.getenv("GITHUB_RUN_ATTEMPT","local"),
       "model":p,"root_frozen":True,"lora_updates":0,
       "head_updates_planned":UPDATES_PER_STAGE*3,
       "w0_pin":objective.W0_COMMIT,"w0_sha256":sha256_file(args.w0),
       "support_pin":objective.SUPPORT_COMMIT,"support_sha256":sha256_file(args.support),
       "source_sha256":sha256_file(__file__),"features":FEATURE_VERSION,
       "train":receipt["train"],"development":receipt["eval"],
       "dataset_not_native":True,"sealed_final_opened":False,"numerical":numerical,"runtime":runtime}
    write_json(args.output/"preflight.json",preflight)
    head,stages,histories=train_head(torch,data,support,args.output)
    write_json(args.output/"optimizer-history.json",histories)
    rows=[]
    for seed in EVAL_SEEDS:
        for family in FAMILIES:
            for mode in ("random","wall","fidget"):
                rec=w0.run(seed,mode,family,128)
                w0.write_result(rec,args.output/"png-and-json")
                rows.append({"seed":seed,"family":family,"mode":mode,
                             "steps":rec["steps_executed"],"metrics":rec["metrics"]})
            rec=rollout_head(w0,torch,head,seed,family)
            w0.write_result(rec,args.output/"png-and-json")
            rows.append({"seed":seed,"family":family,"mode":"aux-head",
                         "steps":rec["steps_executed"],"metrics":rec["metrics"]})
    if not args.model:
        raise RuntimeError("W9 must fuse a genuinely pinned root model; head-only run is insufficient")
    import train as shared
    from worm_score_v5_eval import action_tokens
    root,tok=shared.load_base(args.model,p)
    root.eval()
    adapter=adapter_for(p)
    token_ids,token_map=action_tokens(tok)
    for family in FAMILIES:
        for fused in (False,True):
            rec=model_rollout(w0,torch,head,root,tok,adapter,token_ids,FUSION_SEED,family,fused)
            w0.write_result(rec,args.output/"png-and-json")
            rows.append({"seed":FUSION_SEED,"family":family,
                         "mode":"root-plus-head" if fused else "root-untuned",
                         "steps":rec["steps_executed"],"metrics":rec["metrics"],
                         "genuine_root_forwards":len(rec["model_score_decisions"])})
        write_json(args.output/"partial-progress.json",{"rows":rows,"status":"partial"})
    write_json(args.output/"summary.json",{"status":"complete",
         "schema":FEATURE_VERSION,"real_head_optimizer_updates":3*UPDATES_PER_STAGE,
         "root_model_optimizer_updates":0,"lora_updates":0,
         "root_model":p,"model_download_receipt":json.loads((args.model/"download_receipt.json").read_text()),
         "source_sha":os.getenv("GITHUB_SHA"),"stages":stages,
         "checkpoint_integrity":all(sha256_file(args.output/c["checkpoint"]["file"])==c["checkpoint"]["sha256"] for c in stages),
         "equal_world_seed_comparison":FUSION_SEED,"root_action_token_ids":token_map,
         "results":rows,"original_png_count":len(rows),
         "wall_seconds":time.monotonic()-started,"cpu_seconds":time.process_time(),
         "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "native":False,"sealed_final_opened":False})
    print(json.dumps({"status":"complete","real_head_updates":3*UPDATES_PER_STAGE,
        "root_model_forwards":sum(r.get("genuine_root_forwards",0) for r in rows),
        "original_png_count":len(rows)}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--w0",type=Path,required=True)
    parser.add_argument("--support",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        if args.output.exists():
            write_json(args.output/"failure.json",{"type":type(exc).__name__,
                "message":str(exc),"partial_or_failed":True})
        raise
