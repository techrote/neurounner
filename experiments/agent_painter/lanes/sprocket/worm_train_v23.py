#!/usr/bin/env python3
"""W23: train a tiny conditional router over frozen W20 material and W22 frontier experts.
Research-only synthetic host. No game/runtime, root Sprocket or production changes."""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, time
from pathlib import Path
import worm_train_v18 as w18
import worm_train_v19 as w19
import worm_train_v20 as w20
import worm_train_v21 as w21
import worm_train_v22 as w22

VERSION="sprocket-w23-frozen-expert-material-frontier-fusion-v1"
W22_SHA="74fbec5c6ff3ebb0f1ec31692327a9bcb2ad2135ae98a27db95c40bfb597b88f"
W20_SHA=w21.W20_SHA
W19_SHA=w20.W19_SHA
W0_SHA=w18.W0_SHA
TRAIN=(811,821,823,827)
HELDOUT=(6103,6113,6121)
WAVES=160
STAGES=("material-pair-gating","frontier-student-replay","closed-loop-retention")
NAMES=("W19","W20","W22","W23-stage1","W23-stage2",
       "W23-final","W23-final-T07","W23-final-T11","teacher","random","wall")

def fusion_class(torch):
    class Fusion(torch.nn.Module):
        def __init__(self,material,frontier):
            super().__init__()
            self.material_parent=material
            self.frontier_parent=frontier
            for expert in (material,frontier):
                expert.eval()
                for p in expert.parameters():p.requires_grad_(False)
            self.register_buffer("neighbor",torch.tensor([13,17,11,7]),persistent=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(160,48),torch.nn.Tanh(),
                                          torch.nn.Linear(48,1))
            self.residual=torch.nn.Sequential(torch.nn.Linear(160,48),torch.nn.Tanh(),
                                              torch.nn.Linear(48,6))
            for head in (self.gate,self.residual):
                torch.nn.init.zeros_(head[-1].weight)
                torch.nn.init.zeros_(head[-1].bias)
        def forward(self,x,m):
            n=x.shape[0]
            heading=x[:,100:104].argmax(1)
            cells=m.reshape(n,25,6)
            local=[cells[torch.arange(n,device=m.device),
                         self.neighbor[(heading+r)%4]] for r in (0,-1,1)]
            features=torch.cat([x,*local],dim=1)
            assert features.shape[1]==160
            with torch.no_grad():
                ml=self.material_parent(x,m)
                fl=self.frontier_parent(x,m)
            alpha=torch.sigmoid(self.gate(features))
            correction=0.85*torch.tanh(self.residual(features))
            return (1-alpha)*fl+alpha*ml+correction
    return Fusion

def parents(torch,args):
    for path,sha in ((args.w22,W22_SHA),(args.w20,W20_SHA),
                     (args.w19,W19_SHA),(args.w0,W0_SHA)):
        assert w18.sha(path)==sha,("checkpoint/source hash mismatch",str(path))
    base=w20.source(torch,args.w19,w19.VERSION)
    material=w21.load_w20(torch,args.w20,base)
    original=torch.load(args.w22,map_location="cpu",weights_only=True)
    assert original["version"]==w22.VERSION
    assert original["root_forwards"]==original["root_optimizer_steps"]==original["LoRA_updates"]==0
    assert not original["root_loaded"]
    frontier=w20.policy_class(torch)(base)
    frontier.load_state_dict(original["model"],strict=True)
    frontier.eval()
    return base,material,frontier

def eval_all(torch,model,rows,pairs):
    return {"ordinary":w18.evaluate(torch,model,rows),
            "frontier":w22.frontier_eval(torch,model,rows),
            "material":w20.pair_eval(torch,model,pairs)}

def snapshot(torch,model,mat,front):
    p=fusion_class(torch)(mat,front)
    p.load_state_dict(model.state_dict(),strict=True)
    p.eval()
    return p

def checkpoint(torch,model,opt,rng,out,name,stats):
    path=out/(name+".pt")
    torch.save({"version":VERSION,"stage":name,"model":model.state_dict(),
                "optimizer":opt.state_dict(),"random_state":rng.getstate(),
                "torch_rng":torch.random.get_rng_state(),
                "assessment":stats,"W22_sha256":W22_SHA,"W20_sha256":W20_SHA,
                "root_forwards":0,"root_optimizer_steps":0,"LoRA_updates":0,
                "native":False,"FINAL_opened":False},path)
    return {"file":path.name,"sha256":w18.sha(path),"bytes":path.stat().st_size}

def main(args):
    assert (w18.ROOT,w18.REVISION,w18.EOS)==(
        "kandivault/sprocket-500m","eb157d4ddddc4619e661d4a2426bb2dd000fc668",5)
    prev=set(w18.TRAIN+w18.HELDOUT+w19.TRAIN+w19.HELDOUT+
             w20.TRAIN+w20.HELDOUT+w21.TRAIN+w21.HELDOUT+
             w22.TRAIN+w22.HELDOUT)
    assert not prev.intersection(TRAIN+HELDOUT)
    assert not set(TRAIN).intersection(HELDOUT)
    if args.smoke_only:
        assert len(NAMES)==11 and WAVES==160
        print("W23 source-only preflight PASS: distinct seeds, research scope")
        return
    begin=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.manual_seed(232323);torch.set_num_threads(1)
    rng=random.Random(232323)
    base,material,frontier=parents(torch,args)
    w0=w18.w0_module(args.w0)
    w18.save(args.output/"provenance.json",{
        "version":VERSION,"source_sha":os.getenv("GITHUB_SHA","local"),
        "run_id":os.getenv("GITHUB_RUN_ID","local"),
        "script_sha256":w18.sha(__file__),
        "W22_checkpoint_sha256":W22_SHA,"W20_checkpoint_sha256":W20_SHA,
        "W19_checkpoint_sha256":W19_SHA,"W0_source_sha256":W0_SHA,
        "root":w18.ROOT,"revision":w18.REVISION,"EOS":w18.EOS,
        "root_loaded":False,"root_forwards":0,"LoRA_updates":0,
        "native":False,"FINAL_opened":False,"Draft_HOLD":True})
    # Matched material counterfactuals precede geometry overlay.
    _,train_pairs,material_train_receipts,_=w20.interventions(w0,torch,base,TRAIN)
    _,held_pairs,material_held_receipts,examples=w20.interventions(w0,torch,base,HELDOUT)
    witnesses=w20.verify_prefix_replay(w0,examples)
    assert len(train_pairs)>80 and len(held_pairs)>40 and len(witnesses)>=12
    fixture=w22.overlay_host(w0)
    w18.save(args.output/"obstacle-fixture.json",fixture)
    train,train_receipts=w21.gather(torch,w0,[
        ("teacher",None,0),("random",None,0),("W20",material,0),
        ("W22",frontier,0),("W22-T07",frontier,.7)],TRAIN)
    held,held_receipts=w21.gather(torch,w0,[
        ("teacher",None,0),("random",None,0),("W20",material,0),
        ("W22",frontier,0),("W22-T07",frontier,.7)],HELDOUT)
    x,m,y,legal=w18.tensorize(torch,train)
    plan,train_changed,_=w22.supervision(torch,train)
    _,held_changed,_=w22.supervision(torch,held)
    (a0,b0,y0,l0),(a1,b1,y1,l1)=w19.pair_tensors(torch,train_pairs)
    w18.save(args.output/"dataset.json",{
        "train_seeds":TRAIN,"heldout_seeds":HELDOUT,
        "train_rows":len(train),"heldout_rows":len(held),
        "train_material_pairs":len(train_pairs),"heldout_material_pairs":len(held_pairs),
        "planner_changes_train":train_changed,"planner_changes_heldout":held_changed,
        "legal_material_witnesses":witnesses,
        "train_receipts":train_receipts,"heldout_receipts":held_receipts,
        "material_train_receipts":material_train_receipts,
        "material_held_receipts":material_held_receipts,
        "source_W0_unmodified":True,"native":False})
    model=fusion_class(torch)(material,frontier)
    parameters=[p for p in model.parameters() if p.requires_grad]
    assert len(parameters)==8
    opt=torch.optim.AdamW(parameters,lr=.0008,weight_decay=.005)
    stages=[{"stage":"frozen-W20","assessment":eval_all(torch,material,held,held_pairs)},
            {"stage":"frozen-W22","assessment":eval_all(torch,frontier,held,held_pairs)}]
    history=[];snaps=[];replay=None
    for wave,name in enumerate(STAGES):
        if wave==2:
            replay,rec=w21.gather(torch,w0,[
                ("W23-stage2",snaps[-1],0),("W23-stage2-T07",snaps[-1],.7)],TRAIN)
            rx,rm,ry,rl=w18.tensorize(torch,replay)
            rp,_,_=w22.supervision(torch,replay)
            w18.save(args.output/"onpolicy-replay.json",{
                "rows":len(replay),"receipts":rec,
                "real_head_forwards":sum(r["actual_head_forwards"] for r in rec)})
        for step in range(WAVES):
            opt.zero_grad(set_to_none=True)
            def take(n,count):return [rng.randrange(n) for _ in range(count)]
            i=take(len(train),64);j=take(len(train_pairs),64)
            logits=model(x[i],m[i]).masked_fill(~legal[i],-1e4)
            planner_loss=torch.nn.functional.cross_entropy(logits,plan[i])
            teacher_loss=torch.nn.functional.cross_entropy(logits,y[i])
            pa=model(a0[j],b0[j]).masked_fill(~l0[j],-1e4)
            pb=model(a1[j],b1[j]).masked_fill(~l1[j],-1e4)
            material_loss=(torch.nn.functional.cross_entropy(pa,y0[j])+
                           torch.nn.functional.cross_entropy(pb,y1[j]))/2
            if wave==0:loss=.48*material_loss+.38*planner_loss+.14*teacher_loss
            elif wave==1:loss=.35*material_loss+.49*planner_loss+.16*teacher_loss
            else:
                k=take(len(replay),64)
                rlog=model(rx[k],rm[k]).masked_fill(~rl[k],-1e4)
                replay_loss=torch.nn.functional.cross_entropy(rlog,rp[k])
                loss=.36*material_loss+.32*planner_loss+.16*replay_loss+.16*teacher_loss
            assert bool(torch.isfinite(loss))
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(parameters,2.5)
            if not bool(torch.isfinite(grad)) or float(grad)<=0:
                raise ValueError("zero/nonfinite router gradient")
            opt.step()
            if (step+1)%20==0:
                history.append({"stage":name,"step":step+1,
                    "global_updates":wave*WAVES+step+1,
                    "loss":round(float(loss.detach()),6),
                    "gradient_norm":round(float(grad),6)})
                w18.save(args.output/"partial-optimizer-history.json",history)
        stats=eval_all(torch,model,held,held_pairs)
        cp=checkpoint(torch,model,opt,rng,args.output,name,stats)
        stages.append({"stage":name,"updates":WAVES,"assessment":stats,"checkpoint":cp})
        snaps.append(snapshot(torch,model,material,frontier))
        w18.save(args.output/"stage-progress.json",stages)
    controllers=[
        ("W19",base,0),("W20",material,0),("W22",frontier,0),
        ("W23-stage1",snaps[0],0),("W23-stage2",snaps[1],0),
        ("W23-final",model,0),("W23-final-T07",model,.7),
        ("W23-final-T11",model,1.1),
        ("teacher",None,0),("random",None,0),("wall",None,0)]
    assert tuple(label for label,_,_ in controllers)==NAMES
    images=args.output/"original-synthetic-png-json"
    images.mkdir()
    results=[]
    for seed in HELDOUT:
        for family in w18.FAMILIES:
            for label,p,temp in controllers:
                rec=w18.rollout(torch,w0,p,seed,family,label,temp)
                stem=f"{family}-{label}-{seed}"
                jp=images/(stem+".json");pp=images/(stem+".png")
                w18.save(jp,rec)
                pp.write_bytes(w18.material_png(w0,rec))
                results.append({"seed":seed,"family":family,"controller":label,
                    "metrics":rec["metrics"],"head_forwards":rec["original_head_forwards"],
                    "host_legal_filters":rec["additional_host_legal_filters"],
                    "sha_json":w18.sha(jp),"sha_png":w18.sha(pp),
                    "trajectory_sha256":hashlib.sha256(json.dumps(
                        rec["trajectory"],separators=(",",":")).encode()).hexdigest()})
        w18.save(args.output/"partial-gallery.json",results)
    assert len(results)==99
    assert all(r["metrics"]["invalid_raw_choices"]==0 for r in results)
    assert all(w18.sha(args.output/s["checkpoint"]["file"])==s["checkpoint"]["sha256"]
               for s in stages if "checkpoint" in s)
    aggregate={}
    for label in NAMES:
        rec=[r["metrics"] for r in results if r["controller"]==label]
        aggregate[label]={"worlds":len(rec),
            "mean_unique_cells":round(sum(v["unique_trail_cells"] for v in rec)/len(rec),6),
            "branches":sum(v["branches"] for v in rec),
            "retraces":sum(v["retraces"] for v in rec)}
    material_base=stages[1]["assessment"]["material"]["both_teacher_targets_correct"]
    material_final=stages[-1]["assessment"]["material"]["both_teacher_targets_correct"]
    spatial_base=aggregate["W22"]["mean_unique_cells"]
    spatial_final=aggregate["W23-final"]["mean_unique_cells"]
    gate=bool(material_final>=material_base*1.10 and spatial_final>=spatial_base*.97)
    w18.save(args.output/"summary.json",{
        "status":"complete","version":VERSION,"genuine_router_optimizer_updates":3*WAVES,
        "frozen_parents_unchanged":True,"root_forwards":0,
        "root_optimizer_steps":0,"LoRA_updates":0,"native":False,
        "sealed_FINAL":False,"promotion_allowed":False,
        "original_png_json_pairs":len(results),"results":results,
        "controller_aggregate":aggregate,"stages":stages,
        "compound_material_gain_and_spatial_retention_gate":gate,
        "minimum_material_pair_ratio_vs_W22":1.10,
        "minimum_unique_cell_ratio_vs_W22":.97,
        "actual_head_forwards":sum(r["head_forwards"] for r in results),
        "wall_seconds":time.monotonic()-begin,
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "external_synthetic_obstacles":True,
        "global_pathfinding_proven":False})
    print(json.dumps({"W23":"complete","updates":3*WAVES,
        "paintings":len(results),"compound_gate":gate}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("w22","w20","w19","w0","output"):p.add_argument("--"+n,type=Path)
    p.add_argument("--smoke-only",action="store_true")
    args=p.parse_args()
    if not args.smoke_only and not all((
        args.w22,args.w20,args.w19,args.w0,args.output)):
        p.error("SHA-admitted W22/W20/W19 checkpoints and original W0 required")
    try:main(args)
    except BaseException as exc:
        if args.output and args.output.exists():
            w18.save(args.output/"failure.json",
                     {"status":"failed","type":type(exc).__name__,"message":str(exc)})
        raise
