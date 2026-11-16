"""Actual CPU SGD on synthetic WORM imitation examples; no HF weights implied.

Torch checkpoints/evidence are written OUTSIDE Git. Pretrained 20M-300M
baselines require separate pinned weight acquisition (see README).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from .core import ACTIONS, FAMILIES, FEATURES, SIZE, World, teacher, choose_random, choose_wall, digest, run, replay, raster


def generate(seed_groups: list[int]):
    xs, ys, ms, identities = [], [], [], []
    for seed in seed_groups:
        for fi, family in enumerate(FAMILIES):
            w = World(seed, family)
            rng = random.Random(seed * 1901 + fi * 13)
            for _ in range(w.budget):
                legal = w.legal()
                if not legal: break
                xs.append(w.observe())
                mask = np.zeros(len(ACTIONS),dtype=np.bool_)
                mask[list(legal)] = True
                ms.append(mask)
                y = int(teacher(w, rng))
                ys.append(y)
                identities.append((seed,family,len(w.actions),w.state_sha256(),ACTIONS[y]))
                w.apply(y)
                if w.stopped: break
    return (np.asarray(xs,np.float32),np.asarray(ys,np.int64),
            np.asarray(ms,np.bool_),identities)


def build_models(torch):
    nn=torch.nn
    class Linear(nn.Module):
        def __init__(self):
            super().__init__();self.fc=nn.Linear(FEATURES,5)
        def forward(self,x): return self.fc(x)
    class MLP(nn.Module):
        def __init__(self):
            super().__init__();self.net=nn.Sequential(nn.Linear(FEATURES,96),nn.Tanh(),nn.Linear(96,48),nn.ReLU(),nn.Linear(48,5))
        def forward(self,x):return self.net(x)
    class Conv(nn.Module):
        def __init__(self):
            super().__init__(); self.conv=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,16,3,padding=1),nn.ReLU(),nn.Flatten())
            self.fc=nn.Sequential(nn.Linear(16*7*7+15,64),nn.ReLU(),nn.Linear(64,5))
        def forward(self,x):return self.fc(torch.cat((self.conv(x[:,:147].reshape(-1,3,7,7)),x[:,147:]),dim=1))
    return {"linear":Linear(),"mlp":MLP(),"conv":Conv()}


def train_one(torch, model, train, dev, epochs=9):
    torch.set_num_threads(2)
    opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.002)
    tx=torch.from_numpy(train[0]); ty=torch.from_numpy(train[1]); tm=torch.from_numpy(train[2])
    vx=torch.from_numpy(dev[0]); vy=torch.from_numpy(dev[1]); vm=torch.from_numpy(dev[2])
    generator=torch.Generator().manual_seed(1717)
    losses=[];began=time.perf_counter()
    for e in range(epochs):
        model.train();total=0.0
        for ix in torch.randperm(len(tx),generator=generator).split(512):
            logits=model(tx[ix]).masked_fill(~tm[ix],-1e5)
            loss=torch.nn.functional.cross_entropy(logits,ty[ix])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
            total+=float(loss.detach())*len(ix)
        model.eval()
        with torch.inference_mode():
            val=model(vx).masked_fill(~vm,-1e5)
            acc=(val.argmax(dim=-1)==vy).float().mean().item()
            vloss=float(torch.nn.functional.cross_entropy(val,vy))
        losses.append({"epoch":e+1,"train_cross_entropy":total/len(tx),"dev_cross_entropy":vloss,"dev_accuracy":acc})
    return losses, time.perf_counter()-began


def selector_from_model(torch, model):
    model.eval()
    def select(w,rng):
        x=torch.from_numpy(w.observe()[None,:]);legal=w.legal()
        with torch.inference_mode():
            logits=model(x).squeeze(0).numpy()
        mask=np.full(5,-1e10,np.float64);mask[list(legal)]=logits[list(legal)]
        return int(np.argmax(mask))
    select.__name__=type(model).__name__.lower()
    return select


def save_experiment(output: Path, train_seeds=None, dev_seeds=None, test_seeds=None, epochs=9):
    import torch
    from PIL import Image,ImageDraw
    torch.manual_seed(42)
    np.random.seed(42)
    train_seeds=list(range(1001,1051)) if train_seeds is None else train_seeds
    dev_seeds=list(range(2001,2011)) if dev_seeds is None else dev_seeds
    test_seeds=[5001,5003,5009,5011,5021,5039,5051,5077] if test_seeds is None else test_seeds
    if set(train_seeds)&set(dev_seeds) or set(dev_seeds)&set(test_seeds) or set(train_seeds)&set(test_seeds):
        raise ValueError('Leakage: world seed shared across split')
    output.mkdir(parents=True,exist_ok=False)
    partitions={"TRAIN":train_seeds,"DEV":dev_seeds,"EVAL":test_seeds}
    data={k:generate(v) for k,v in partitions.items()}
    identities={k:{"groups":len(v),"examples":len(data[k][1]),
                    "identity_sha256":digest(data[k][3]),"seeds":v} for k,v in partitions.items()}
    models=build_models(torch)
    summary={"protocol":"tiny-game-policy-synthetic-v1", "framework":"torch", "torch_version":torch.__version__,
             "synthetic_only":True,"native_game_exec":False,"pretrained_backbone":False,
             "split_identity":identities,"model_comparisons":{},"world_size":96,
             "actions":list(ACTIONS),"train_model_outputs":{}}
    tests=data["EVAL"]
    tx=torch.from_numpy(tests[0]); ty=torch.from_numpy(tests[1]);mask=torch.from_numpy(tests[2])
    for name,model in models.items():
        tr_hist,elapsed=train_one(torch,model,data['TRAIN'],data['DEV'],epochs=epochs)
        model.eval()
        with torch.inference_mode():
            raw=model(tx)
            legal=raw.masked_fill(~mask,-1e5)
            preds=legal.argmax(dim=-1)
        illegal=int((~mask).gather(1,preds.view(-1,1)).sum())
        ck=output/f"{name}-synthetic-checkpoint.pt"
        torch.save({"model":name,"state_dict":model.state_dict(),"split_sha256":digest(identities),
                    "seed":42,"protocol":summary["protocol"]},ck)
        sha=hashlib.sha256(ck.read_bytes()).hexdigest()
        summary['model_comparisons'][name]={"parameters":sum(p.numel() for p in model.parameters()),
             "optimizer_updates":epochs*((len(data['TRAIN'][1])+511)//512),
             "train_seconds":round(elapsed,3),"test_teacher_accuracy":round(float((preds==ty).float().mean()),4),
             "test_illegal_predictions":illegal,"last_dev_accuracy":tr_hist[-1]['dev_accuracy'],
             "checkpoint_sha256":sha,"training_history":tr_hist}
        summary["train_model_outputs"][name]=ck.name
    policies={"teacher":teacher,"random":choose_random,"wall":choose_wall}
    for name,model in models.items():policies[name]=selector_from_model(torch,model)
    stats={p:[] for p in policies}
    masks={p:{} for p in policies}
    rows=[]
    for seed in test_seeds[:4]:
        row=[]
        for family in FAMILIES:
            for pname, fn in policies.items():
                w=World(seed,family)
                log=run(w,fn,seed*1237 + FAMILIES.index(family))
                replay(log)
                masks[pname][(seed,family)] = w.visited.copy()
                stats[pname].append({"seed":seed,"family":family,"unique":log['unique_cells'],
                                     "retrace":log['retrace_steps'],"branch_nodes":log['branch_nodes'],
                                     "stop":log['stop'],"actions":len(log['events'])})
                (output/"rollouts").mkdir(exist_ok=True)
                (output/"rollouts"/f"{seed}-{family}-{pname}.json").write_text(json.dumps(log,separators=(',',':'),sort_keys=True))
                im=raster(w,3)
                im.save(output/"rollouts"/f"{seed}-{family}-{pname}.png")
                if seed==test_seeds[0]:
                    row.append((f"{family} {pname}",im))
        if row: rows=row
    summary["rollout_metrics"]={p:{"n":len(rows),"mean_unique_cells":round(float(np.mean([x['unique'] for x in rows])),2),
                                     "mean_retrace_steps":round(float(np.mean([x['retrace'] for x in rows])),2),
                                     "mean_branch_nodes":round(float(np.mean([x['branch_nodes'] for x in rows])),2),
                                     "early_stop_count":sum(x['stop'] for x in rows)} for p,rows in stats.items()}
    for name in policies:
        overlaps=[]
        for seed in test_seeds[:4]:
            for i in range(len(FAMILIES)):
                for j in range(i+1,len(FAMILIES)):
                    a=masks[name][(seed,FAMILIES[i])]
                    b=masks[name][(seed,FAMILIES[j])]
                    overlaps.append(float(np.logical_and(a,b).sum())/float(np.logical_or(a,b).sum()))
        summary["rollout_metrics"][name]["mean_pairwise_family_iou"]=round(float(np.mean(overlaps)),4)
    # One same-world contact sheet. Keep source images independently in rollouts.
    if rows:
        ww=len(policies)*SIZE*3;hh=len(FAMILIES)*(SIZE*3+24)
        sheet=Image.new('RGB',(ww,hh),'white');d=ImageDraw.Draw(sheet)
        for i,(lab,im) in enumerate(rows):
            x=(i%len(policies))*SIZE*3;y=(i//len(policies))*(SIZE*3+24)
            sheet.paste(im,(x,y+24));d.text((x+3,y+4),lab,fill='black')
        sheet.save(output/f'comparison-seed{test_seeds[0]}.png')
    (output/'report.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"split":{k:v['examples'] for k,v in identities.items()},
                      "models":{k:{q:v[q] for q in ('parameters','test_teacher_accuracy','test_illegal_predictions')} for k,v in summary['model_comparisons'].items()},
                      "rollouts":summary['rollout_metrics']},indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--epochs',type=int,default=9)
    args=parser.parse_args()
    if not 1<=args.epochs<=20:parser.error('--epochs must be in [1,20]')
    save_experiment(args.out,epochs=args.epochs)

if __name__=='__main__':main()
