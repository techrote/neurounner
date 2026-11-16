"""DG008: genuinely train compositional game-law specialists, test unseen grafts.

Three frozen Qwen/DG005/DG006 parent components provide H/V/P states only at
inference; A/B material/motion alleles are synthetic, separately observed.
The training distribution NEVER contains A=B=1. Two specialist heads may add
their evidence at inference, whereas flat/blind comparators have no such
structural prior. A deliberately unseen A*B nonlinear physics interaction is
an unidentifiability negative control, not a target claimed generalisable.
No CyberSand game code, assets or production configuration are touched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.memplant.learned_game_007.experiment import (
    ACTIONS, DIRS, frames, offers, world, donor_states
)
from experiments.memplant.three_law.experiment import (
    advance, truth_after
)

N=6
D=15
SEQ=(0,1,2,0,1,2,0,1)


def expert(law, f, legal, hidden_interaction=False):
    """Constructed synthetic game teacher. Law bits: H,V,P,A,B."""
    h,v,p,a,b=map(int,law)
    scores=np.full(N,-1e7,dtype=np.float32)
    for j in range(N):
        if not legal[j]:
            continue
        z=f[j]
        if j<4:
            score=1.9*z[0]+.33*z[2]+.06*z[3]+.28*z[7]-.18*z[13]
            score+=z[5]*(1.12 if h else -1.12)
            score+=z[6]*(1.12 if v else -1.12)
            score+=a*(1.05*z[6]-.55*z[5])
            score+=b*(.78*z[2]+.45*z[3]-.3*z[13])
        elif j==4:
            score=.7+1.1*z[12]+.8*z[7]+(1.65 if (z[5] if p else z[4]) else -1.1)-2.5*z[11]
            score+=a*(.4*z[5]-.2)+b*(.3*(1-z[11]))
        else:
            score=-.65+.5*z[13]
        if hidden_interaction and a and b:
            # This hidden, *never observed in training* conjunction is
            # deliberately not inferable from either isolated donor.
            if j==5:
                score+=5.0
            if j==4:
                score-=1.6
        scores[j]=score
    return int(scores.argmax())


def samples(seed, count, split):
    d=frames(seed, count)
    if split=="train" or split=="seen":
        rng=random.Random(seed*13+91)
        ab=np.array([(0,0),(1,0),(0,1)],dtype=np.float32)
        aux=np.stack([ab[rng.randrange(3)] for _ in range(count)])
    elif split=="graft":
        aux=np.ones((count,2),dtype=np.float32)
    else:
        raise ValueError(split)
    laws=torch.cat((d["laws"],torch.tensor(aux)),-1)
    targets=torch.tensor([
        expert(laws[i].tolist(),d["features"][i].numpy(),d["legal"][i].numpy())
        for i in range(count)],dtype=torch.long)
    return {"laws":laws,"features":d["features"],"legal":d["legal"],
            "target":targets,"world_identifiers":d["identities"]}


class Factorized(nn.Module):
    """Separately trainable, additive law specialists on a shared base."""
    def __init__(self):
        super().__init__()
        self.base=nn.Sequential(nn.Linear(D+3,64),nn.Tanh(),nn.Linear(64,64),
                                nn.Tanh(),nn.Linear(64,1))
        self.a=nn.Sequential(nn.Linear(D+3,32),nn.Tanh(),nn.Linear(32,1))
        self.b=nn.Sequential(nn.Linear(D+3,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,law,features,legal):
        x=torch.cat((features,law[:,None,:3].expand(-1,N,-1)),-1)
        logit=self.base(x).squeeze(-1)
        logit+=law[:,3,None]*self.a(x).squeeze(-1)
        logit+=law[:,4,None]*self.b(x).squeeze(-1)
        return logit.masked_fill(~legal,-1e4)


class Flat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(D+5,64),nn.Tanh(),nn.Linear(64,64),
                               nn.Tanh(),nn.Linear(64,1))
    def forward(self,law,features,legal):
        x=torch.cat((features,law[:,None,:].expand(-1,N,-1)),-1)
        return self.net(x).squeeze(-1).masked_fill(~legal,-1e4)


def predict(model,data,law=None,blind=False):
    law=data["laws"] if law is None else law
    if blind:
        law=torch.zeros_like(law)+.5
    with torch.no_grad():
        out=model(law,data["features"],data["legal"]).argmax(-1)
    assert bool(data["legal"].gather(1,out[:,None]).all()),"illegal model output"
    return out


def evaluate(model,data,law=None,blind=False,hidden=False):
    out=predict(model,data,law,blind)
    if hidden:
        target=torch.tensor([expert(data["laws"][i].tolist(),
                  data["features"][i].numpy(),data["legal"][i].numpy(),True)
                  for i in range(len(out))],dtype=torch.long)
    else:
        target=data["target"]
    return {"correct":int((out==target).sum()),"n":len(out),
            "accuracy":round(float((out==target).float().mean()),6),
            "choices":{a:int((out==j).sum()) for j,a in enumerate(ACTIONS)}}


def fit(model,kind,tr,valid,seed,updates):
    torch.manual_seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.00005)
    gen=torch.Generator().manual_seed(seed+811)
    best=-1.0
    best_weights=None
    curve=[]
    for step in range(1,updates+1):
        idx=torch.randint(len(tr["target"]),(256,),generator=gen)
        x=tr["laws"][idx]
        if kind=="blind":
            x=torch.zeros_like(x)+.5
        opt.zero_grad()
        loss=F.cross_entropy(model(x,tr["features"][idx],tr["legal"][idx]),
                             tr["target"][idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),2)
        opt.step()
        if step%50==0 or step==updates:
            score=evaluate(model,valid,blind=(kind=="blind"))["accuracy"]
            curve.append({"step":step,"valid_seen":score,
                          "loss":round(float(loss.detach()),6)})
            if score>best:
                best=score
                best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    return {"best_seen_valid":best,"curve":curve}


def parent_probe(models,source,ref005,ref006,seed):
    """Frozen genuine Qwen-derived first 2 bits, 3rd synthetic, A/B unseen."""
    cases,idx,truth,initial,parents,provenance=donor_states(source,ref005,ref006,seed)
    gold=torch.tensor([truth_after(tuple(map(int,t)),SEQ)
                       for t in truth.tolist()],dtype=torch.float32)
    # Reuse the exact same 24 source families x eight new game worlds in all
    # parent conditions. Cases were historically inspected; not a fresh donor.
    rows=[]
    for k,i in enumerate(idx):
        for copy in range(8):
            w=world(seed*1000000+copy*10000+k)
            f,m=offers(w,w["start"],[w["start"]],set(),(k+copy)%4)
            rows.append((k,f,m))
    features=torch.from_numpy(np.stack([r[1] for r in rows]))
    legal=torch.from_numpy(np.stack([r[2] for r in rows]))
    laws=torch.tensor([list(gold[r[0]].tolist())+[1.,1.] for r in rows],
                      dtype=torch.float32)
    label=torch.tensor([expert(laws[j].tolist(),r[1],r[2])
                       for j,r in enumerate(rows)])
    data={"features":features,"legal":legal,"laws":laws,"target":label}
    zeros=initial.clone()
    zeros[:,:2]=0
    perm=torch.randperm(len(idx),generator=torch.Generator().manual_seed(seed+129))
    shuffled=initial.clone()
    shuffled[:,:2]=initial[perm,:2]
    sources={"real":initial,"zero":zeros,"shuffled":shuffled,
             "oracle":(2*truth-1)*6}
    results={}
    for parent_name,parent in parents.items():
        for mode,x in sources.items():
            with torch.no_grad():
                state=(torch.sigmoid(advance(parent,x,SEQ))>=.5).float()
            expanded=torch.tensor([list(state[r[0]].tolist())+[1.,1.] for r in rows])
            key=parent_name+"/"+mode
            results[key]={"state_joint_accuracy":round(float(
                (state==gold).all(-1).float().mean()),6)}
            for model_name,model in models.items():
                results[key][model_name]=evaluate(model,data,law=expanded,
                                                  blind=(model_name=="blind"))
    results["provenance"]=provenance
    results["source_cases"]=len(idx)
    results["worlds_per_source"]=8
    results["edit_sequence"]=list(SEQ)
    return results


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def render_rollout(model,name,law,world_seed,out):
    """Original 24-decision synthetic action trace, not a native game capture."""
    from PIL import Image, ImageDraw
    w=world(world_seed)
    pos=w["start"]
    history=[pos]
    painted=set()
    heading=0
    chosen=[]
    compared=[]
    for step in range(24):
        f,legal=offers(w,pos,history,painted,heading)
        ft=torch.from_numpy(f[None])
        lt=torch.from_numpy(legal[None])
        with torch.no_grad():
            action=int(model(torch.tensor([law],dtype=torch.float32),
                             ft,lt).argmax(-1))
        assert legal[action]
        gold=expert(law,f,legal)
        chosen.append(ACTIONS[action])
        compared.append(action==gold)
        if action<4:
            pos=(pos[0]+DIRS[action][0],pos[1]+DIRS[action][1])
            history.append(pos)
            heading=action
        elif action==4:
            painted.add(pos)
        elif len(history)>1:
            history.pop()
            pos=history[-1]
    # Render genuine derived world state from this specific original trace.
    scale=28
    img=Image.new("RGB",(9*scale,9*scale),(17,21,25))
    draw=ImageDraw.Draw(img)
    palette=[(111,111,118),(165,139,82),(55,103,156),(24,28,31)]
    for y in range(9):
        for x in range(9):
            pt=(x,y)
            col=palette[int(w["tiles"][y,x])]
            if w["protect"][y,x]:
                col=(127,55,66)
            if pt in painted:
                col=(240,191,77)
            draw.rectangle((x*scale,y*scale,(x+1)*scale-1,
                            (y+1)*scale-1),fill=col,
                           outline=(35,38,43))
    for k,(x,y) in enumerate(history):
        draw.ellipse((x*scale+9,y*scale+9,x*scale+18,y*scale+18),
                     fill=(123,214,151) if k<len(history)-1
                     else (247,249,251))
    gx,gy=w["goal"]
    draw.rectangle((gx*scale+11,gy*scale+11,
                    gx*scale+16,gy*scale+16),fill=(247,78,80))
    img.save(out)
    return {"model":name,"law":[int(b) for b in law],"world_seed":world_seed,
            "actions":chosen,"synthetic_steps":len(chosen),
            "painted_cells":len(painted),"final_position":list(pos),
            "exact_expert_matches_on_own_trajectory":sum(compared),
            "image":out.name,"image_sha256":digest(out)}


def produce(source,ref005,ref006,out,seed,updates,train_count=8500,
            valid_count=1500,sealed_count=2600):
    if not 100<=updates<=2000:
        raise ValueError("optimizer updates outside bounded 100..2000")
    if not 1000<=train_count<=20000:
        raise ValueError("world training budget invalid")
    torch.set_num_threads(2)
    start=time.monotonic()
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    tr=samples(seed+10,train_count,"train")
    valid=samples(seed+20,valid_count,"seen")
    seen=samples(seed+30,sealed_count,"seen")
    graft=samples(seed+40,sealed_count,"graft")
    assert not bool((tr["laws"][:,3:]==1).all(-1).any()),"graft leakage"
    assert bool((graft["laws"][:,3:]==1).all(-1).all())
    models={}
    train_reports={}
    checks={}
    for n,cls in (("factorized",Factorized),("flat",Flat),("blind",Flat)):
        torch.manual_seed(seed+{"factorized":11,"flat":12,"blind":13}[n])
        model=cls()
        train_reports[n]=fit(model,n,tr,valid,seed+{"factorized":11,"flat":12,"blind":13}[n],updates)
        models[n]=model
        file=out/(n+".pt")
        torch.save({"state_dict":model.state_dict(),"seed":seed,"kind":n,
                    "experiment":"DG008","train_pairs":["00","10","01"],
                    "heldout_pair":"11","true_laws_train":True},file)
        checks[n]=digest(file)
    evaluations={}
    for n,model in models.items():
        evaluations[n]={
            "seen":evaluate(model,seen,blind=(n=="blind")),
            "unseen_additive":evaluate(model,graft,blind=(n=="blind")),
            "unseen_hidden_synergy":evaluate(model,graft,blind=(n=="blind"),
                                             hidden=True)}
    # Visualise model-selected, sequential game-like actions on the SAME
    # world under separately observed A, B and unseen A+B law settings.
    rollouts=[]
    worlds_out=out/"rollouts"
    worlds_out.mkdir(exist_ok=True)
    for model_name,model in models.items():
        for combo in ((1,0),(0,1),(1,1)):
            law=(1,0,1)+combo
            file=worlds_out/(model_name+"-A"+str(combo[0])+"B"+str(combo[1])+".png")
            rollouts.append(render_rollout(
                model,model_name,law,seed+4321,file))
    parent=parent_probe(models,source,ref005,ref006,seed)
    report={"experiment":"DG008-heldout-material-motion-graft-v1","seed":seed,
      "updates_per_model":updates,
      "train_worlds":train_count,"valid_worlds":valid_count,
      "sealed_worlds_per_arm":sealed_count,
      "train_A_B_pairs":["00","10","01"],"sealed_A_B_pair":"11",
      "models_trained":["factorized","flat","blind"],
      "training":train_reports,"sealed_results":evaluations,
      "model_selected_synthetic_rollouts":rollouts,
      "real_donor_inherited_graft":parent,"checkpoints_sha256":checks,
      "elapsed_seconds":round(time.monotonic()-start,2),
      "limitations":["three H/V/P bits explicitly supervised in policy training",
       "A/B additional law bits explicitly observed, not extracted from neural donor",
       "factorized architectural prior biases toward additivity",
       "hidden A*B interaction is unidentifiable: no A=B=1 training examples",
       "synthetic game expert, not native CyberSand gameplay",
       "historically inspected Qwen donor sealed families",
       "no real heterogeneous cross-model KV transplant",
       "synthetic game output choices use host legal-action mask"]}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"seed":seed,"results":evaluations,
         "donor_contextual_real":parent["contextual/real"],
         "donor_contextual_zero":parent["contextual/zero"],
         "seconds":report["elapsed_seconds"]},indent=2))
    return report


def selftest():
    torch.set_num_threads(2)
    a=samples(91,40,"train")
    b=samples(91,40,"train")
    assert torch.equal(a["laws"],b["laws"]) and torch.equal(a["target"],b["target"])
    assert not bool((a["laws"][:,3:]==1).all(-1).any())
    g=samples(92,40,"graft")
    assert bool((g["laws"][:,3:]==1).all(-1).all())
    for i in range(40):
        assert g["legal"][i,int(g["target"][i])]
    for cls in (Factorized,Flat):
        m=cls()
        p=predict(m,g)
        assert len(p)==40
    f=g["features"][0].numpy()
    m=g["legal"][0].numpy()
    zero=expert((1,0,1,1,0),f,m)
    one=expert((1,0,1,0,1),f,m)
    both=expert((1,0,1,1,1),f,m)
    assert 0<=zero<6 and 0<=one<6 and 0<=both<6
    net=Factorized()
    r=fit(net,"factorized",a,a,83,2)
    assert len(r["curve"])==1
    # Algebraic graft invariant: specialist effects sum, independently of order.
    x=g["laws"][:2].clone()
    y=g["features"][:2]
    legal=g["legal"][:2]
    with torch.no_grad():
        base=x.clone()
        base[:,3:]=0
        ca=base.clone()
        cb=base.clone()
        ca[:,3]=1
        cb[:,4]=1
        full=base.clone()
        full[:,3:]=1
        assert torch.allclose(net(full,y,legal)+net(base,y,legal),
             net(ca,y,legal)+net(cb,y,legal),atol=1e-4)
    print("DG008 PAIR HOLDOUT / LEGALITY / OPTIMIZER / GRAFT INVARIANT PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--source")
    p.add_argument("--reference-dg005")
    p.add_argument("--reference-dg006")
    p.add_argument("--out")
    p.add_argument("--seed",type=int,default=26011611)
    p.add_argument("--updates",type=int,default=500)
    p.add_argument("--train-worlds",type=int,default=8500)
    a=p.parse_args()
    if a.selftest:
        selftest()
    else:
        if not all((a.source,a.reference_dg005,a.reference_dg006,a.out)):
            p.error("pinned source + both frozen checkpoints + out required")
        produce(a.source,a.reference_dg005,a.reference_dg006,
                a.out,a.seed,a.updates,train_count=a.train_worlds)
