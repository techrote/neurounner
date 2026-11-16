"""M3: bounded synthetic recovery/branch/STOP curriculum and true CPU optimizer trials.

Uses the unchanged tiny_policy.core host contract, not native CyberSand/FINAL.
No third-party pretrained weights are claimed; small heads isolate curriculum effects.
All learned actions are selected from host-provided legal indices.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

from .core import ACTIONS, DELTA, FAMILIES, FEATURES, World, choose_random, choose_wall, digest, raster

CASE_NAMES = ("fork", "protected_gate", "switchback", "boxed_exit")
SPLITS = {"TRAIN": tuple(range(7300, 7340)),
          "DEV": tuple(range(8300, 8308)),
          "EVAL": tuple(range(9300, 9308))}
assert len(set(sum((list(v) for v in SPLITS.values()), []))) == sum(map(len, SPLITS.values()))
BUDGET = 96
EVAL_TURNS = 64


def world(seed: int, family: str, *, trapped: bool = False) -> World:
    """Same geometry for every family, with actual walls and protected-cell fixtures."""
    w = World(seed, family, budget=BUDGET)
    variant = seed % len(CASE_NAMES)
    if variant == 0:  # Path bifurcation, blocked corridor and two openings.
        w.blocked[43:57, 44] = True
        w.blocked[43:57, 53] = True
        w.blocked[43, 44:54] = w.blocked[56, 44:54] = True
        w.blocked[48, 44] = False
        w.blocked[51, 53] = False
        w.blocked[45:54, 50] = True
        w.blocked[47, 50] = w.blocked[52, 50] = False
    elif variant == 1:  # Protected gate means blocked is not the only forbidden plane.
        w.protected[44:55, 50] = True
        w.protected[49, 50] = False
        w.blocked[43, 45:55] = w.blocked[55, 45:55] = True
        w.blocked[44:56, 45] = True
        w.blocked[51, 45] = False
    elif variant == 2:  # Alternating wall ends and a forced retrace opportunity.
        w.blocked[45, 43:54] = True
        w.blocked[45, 46] = False
        w.blocked[52, 44:55] = True
        w.blocked[52, 51] = False
        w.protected[47:51, 52] = True
    else:  # Box with a single exit and an internal obstruction.
        w.blocked[43:55, 43] = True
        w.blocked[43:55, 54] = True
        w.blocked[43, 43:55] = w.blocked[54, 43:55] = True
        w.blocked[49, 43] = False
        w.protected[46:51, 51] = True
    w.blocked[48, 48] = w.protected[48, 48] = False
    if trapped:
        for dx, dy in DELTA[:4]:
            w.protected[w.y+dy, w.x+dx] = True
        assert w.legal() == (4,)
    return w


def geometry_hash(w: World) -> str:
    return hashlib.sha256(w.blocked.tobytes()+w.protected.tobytes()).hexdigest()


def oracle(w: World, rng: random.Random) -> int:
    """Imperfect small legal-action teacher: recovery, border-following and habits.

    Explicit STOP only in actual motion dead ends. No global map, native access,
    privileged final goal or complete-art sequence labels are used.
    """
    moves = [a for a in w.legal() if a != 4]
    if not moves:
        return 4
    t = len(w.actions)
    options = []
    for action in moves:
        dx, dy = DELTA[action]
        x, y = w.x+dx, w.y+dy
        prior = bool(w.visited[y, x])
        adjacent = sum(bool(w.blocked[y+oy,x+ox] or w.protected[y+oy,x+ox])
                       for ox, oy in DELTA[:4])
        novelty = 0
        # Bounded short-range accessibility heuristic: only local 5x5 cells.
        for ox, oy in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
                       (0, -2), (0, 2), (-2, 0), (2, 0)):
            xx, yy = x+ox, y+oy
            if 0 <= xx < 96 and 0 <= yy < 96:
                novelty += int(not (w.blocked[yy,xx] or w.protected[yy,xx] or w.visited[yy,xx]))
        straight = action == w.heading
        turn = action != w.heading
        # Intentional imperfections make family styles distinct; negative habits retained.
        if w.family == "Greebles":
            style = .7*turn + .75*(t % 13 == 0 and prior) + .15*adjacent
        elif w.family == "Pipework":
            style = 1.25*straight + .5*adjacent - .2*(t%11 == 0)
        else:
            style = 1.2*(straight if (t//8)%2 == 0 else turn) + .35*(action in (0,3))
        score = (2.7 * (not prior) - 1.0 * prior + .13 * novelty + style
                 + rng.uniform(-.15, .15))
        options.append((score, -action, action))
    return max(options)[2]


def rollout(w, fn, seed, *, steps=EVAL_TURNS, stop_guard=False):
    rng = random.Random(seed*7919 + FAMILIES.index(w.family)*17)
    events=[]
    original_geometry=geometry_hash(w)
    for _ in range(min(w.budget,steps)):
        legal=w.legal()
        if not legal:break
        pre=w.state_sha256()
        obs=w.observe()
        choice=int(fn(w,rng))
        # The guard is an explicit POLICY-SIDE intervention, not host-authority editing.
        if stop_guard and choice==4 and len(legal)>1:
            choice=min(a for a in legal if a != 4)
        if choice not in legal:raise ValueError("Illegal selected action")
        record={"before_sha256":pre,"feature_sha256":hashlib.sha256(obs.tobytes()).hexdigest(),
                "legal":[ACTIONS[a] for a in legal]}
        record.update(w.apply(choice))
        events.append(record)
        if w.stopped:break
    evidence={"seed":w.seed,"family":w.family,"case":CASE_NAMES[w.seed%4],
              "geometry_sha256":original_geometry,"budget":w.budget,"window":steps,
              "events":events,"final_sha256":w.state_sha256(),
              "unique":int(w.visited.sum()),"branches":w.branch_nodes(),
              "retraces":w.revisit_landings,"dead_ends":w.dead_ends,
              "stopped":bool(w.actions and w.actions[-1]=="STOP")}
    replay(evidence)
    return evidence


def replay(evidence):
    w=world(evidence["seed"],evidence["family"])
    if geometry_hash(w)!=evidence["geometry_sha256"] or evidence["budget"]!=BUDGET:
        raise ValueError("World geometry or budget corruption")
    if not 1<=evidence["window"]<=BUDGET or len(evidence["events"])>evidence["window"]:
        raise ValueError("Out-of-budget receipt")
    for row in evidence["events"]:
        if w.state_sha256()!=row["before_sha256"]:raise ValueError("Pre-state mismatch")
        if hashlib.sha256(w.observe().tobytes()).hexdigest()!=row["feature_sha256"]:
            raise ValueError("Observation mismatch")
        if [ACTIONS[a] for a in w.legal()]!=row["legal"]:raise ValueError("Legal menu mismatch")
        if row["action"] not in ACTIONS:raise ValueError("Unknown action string")
        receipt=w.apply(ACTIONS.index(row["action"]))
        if receipt != {k:row[k] for k in receipt}:raise ValueError("Action receipt mismatch")
    if (w.state_sha256()!=evidence["final_sha256"] or int(w.visited.sum())!=evidence["unique"]
            or w.branch_nodes()!=evidence["branches"] or w.revisit_landings!=evidence["retraces"]
            or w.dead_ends!=evidence["dead_ends"]
            or bool(w.actions and w.actions[-1]=="STOP")!=evidence["stopped"]):
        raise ValueError("Final-state/metrics mismatch")
    return w


def examples(seed_groups, *, turns=64):
    xs, ys, menus, weights, identities=[], [], [], [], []
    for seed in seed_groups:
        for family in FAMILIES:
            for exploration in ("oracle", "random", "wall"):
                w=world(seed,family)
                rng=random.Random(seed*217+FAMILIES.index(family)*13)
                for tick in range(turns):
                    legal=w.legal()
                    if not legal:break
                    x=w.observe()
                    label=oracle(w,rng)
                    if label not in legal:raise ValueError("Illegal teacher label")
                    mask=np.zeros(5,dtype=np.bool_);mask[list(legal)]=True
                    xs.append(x);ys.append(label);menus.append(mask)
                    # Bounded weighting to avoid treating every straight-line example equally.
                    weights.append(1.7 if label != w.heading else 1.0)
                    identities.append([seed,family,exploration,tick,w.state_sha256(),ACTIONS[label]])
                    if exploration=="oracle":a=label
                    elif exploration=="random":a=choose_random(w,rng)
                    else:a=choose_wall(w,rng)
                    w.apply(int(a))
                    if w.stopped:break
            # Real host STOP-only protected trap: never relabel ordinary movement as forced STOP.
            trapped=world(seed,family,trapped=True)
            x=trapped.observe();mask=np.zeros(5,dtype=np.bool_);mask[4]=True
            xs.append(x);ys.append(4);menus.append(mask);weights.append(1.0)
            identities.append([seed,family,"protected-trap",0,trapped.state_sha256(),"STOP"])
    return (np.asarray(xs,np.float32),np.asarray(ys,np.int64),
            np.asarray(menus,np.bool_),np.asarray(weights,np.float32),identities)


def models(torch):
    nn=torch.nn
    class Linear(nn.Module):
        def __init__(self):super().__init__();self.net=nn.Linear(FEATURES,5)
        def forward(self,x):return self.net(x)
    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(FEATURES,96),nn.GELU(),nn.Linear(96,48),nn.GELU(),nn.Linear(48,5))
        def forward(self,x):return self.net(x)
    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.cv=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,16,3,padding=1),nn.ReLU(),nn.Flatten())
            self.head=nn.Sequential(nn.Linear(16*49+15,64),nn.GELU(),nn.Linear(64,5))
        def forward(self,x):
            return self.head(torch.cat([self.cv(x[:,:147].reshape(-1,3,7,7)),x[:,147:]],1))
    return {"linear":Linear(),"mlp":Mlp(),"conv":Conv()}


def fit(torch, model, tr, dev, *, epochs):
    xx,yy,mm,ww=(torch.from_numpy(a) for a in tr[:4])
    dx,dy,dm=(torch.from_numpy(a) for a in dev[:3])
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=.002)
    hist=[];best=float("inf");best_state=None;updates=0
    for epoch in range(epochs):
        order=torch.randperm(len(xx),generator=torch.Generator().manual_seed(491+epoch))
        for indices in order.split(512):
            logits=model(xx[indices]).masked_fill(~mm[indices],-1e6)
            loss=(torch.nn.functional.cross_entropy(logits,yy[indices],reduction="none")*ww[indices]).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step();updates+=1
        model.eval()
        with torch.inference_mode():
            logits=model(dx).masked_fill(~dm,-1e6)
            loss=float(torch.nn.functional.cross_entropy(logits,dy))
            acc=float((logits.argmax(1)==dy).float().mean())
        hist.append({"epoch":epoch+1,"dev_cross_entropy":round(loss,6),"dev_agreement":round(acc,5)})
        if loss<best:
            best=loss;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.train()
    model.load_state_dict(best_state);model.eval()
    return hist,updates


def learned_policy(torch, model, *, guard_stop=False):
    model.eval()
    def pick(w,_rng):
        v=torch.from_numpy(w.observe()[None,:])
        with torch.inference_mode():logits=model(v).view(-1).tolist()
        legal=w.legal()
        if guard_stop and any(a!=4 for a in legal):
            legal=tuple(a for a in legal if a!=4)
        return max(legal,key=lambda i:(logits[i],-i))
    pick.__name__=type(model).__name__.lower()
    return pick


def manifest(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}


def experiment(out:Path, *, epochs=5, seeds=SPLITS, gallery_seeds=2):
    import torch
    from PIL import Image,ImageDraw
    torch.set_num_threads(2);torch.manual_seed(730)
    all_seeds=[s for groups in seeds.values() for s in groups]
    if len(set(all_seeds))!=len(all_seeds):raise ValueError('World-seed leakage across splits')
    if out.exists():raise ValueError('Output must be new and immutable')
    out.mkdir(parents=True)
    start=time.perf_counter()
    report={"status":"STARTED","protocol":"tiny-policy-m3-curriculum-cpu-v1",
            "synthetic_only":True,"native_game":False,"pretrained_weights":False,
            "unsealed_research_eval":True,"action_strings":list(ACTIONS),
            "world_size":96,"host_budget":BUDGET,"eval_window":EVAL_TURNS,
            "split_seeds":{k:list(v) for k,v in seeds.items()},"models":{}}
    try:
        data={k:examples(v) for k,v in seeds.items()}
        report["splits"]={k:{"samples":len(v[1]),"identity_sha256":digest(v[4]),
                              "action_counts":{ACTIONS[i]:int((v[1]==i).sum()) for i in range(5)}}
                          for k,v in data.items()}
        report["geometry"]={str(seed):{CASE_NAMES[seed%4]:geometry_hash(world(seed,FAMILIES[0]))}
                            for seed in all_seeds}
        policies={"oracle":oracle,"random":choose_random,"wall":choose_wall}
        initial_models=models(torch)
        policies['conv-untrained']=learned_policy(torch,copy.deepcopy(initial_models['conv']))
        ckdir=out/'checkpoints';ckdir.mkdir()
        for name, model in initial_models.items():
            began=time.perf_counter()
            hist,updates=fit(torch,model,data['TRAIN'],data['DEV'],epochs=epochs)
            torch.save({"state_dict":model.state_dict(),"architecture":name,
                        "train_identity":report['splits']['TRAIN']['identity_sha256'],
                        "protocol":report['protocol']},ckdir/(name+'.pt'))
            with torch.inference_mode():
                x,y,m=data['EVAL'][:3]
                pred=model(torch.from_numpy(x)).masked_fill(~torch.from_numpy(m),-1e6).argmax(1)
                agreement=float((pred==torch.from_numpy(y)).float().mean())
            report['models'][name]={"parameters":sum(p.numel() for p in model.parameters()),
                                    "optimizer_updates":updates,"seconds":round(time.perf_counter()-began,2),
                                    "eval_oracle_agreement":round(agreement,5),
                                    "head_checkpoint_sha256":hashlib.sha256((ckdir/(name+'.pt')).read_bytes()).hexdigest(),
                                    "history":hist}
            policies[name]=learned_policy(torch,model)
        # Each trained model has a guarded version, retaining raw legal STOP for contrast.
        policies['conv-stop-guard']=learned_policy(torch,initial_models['conv'],guard_stop=True)
        rows=[]
        receipts={}
        rollouts=out/'rollouts';rollouts.mkdir()
        for seed in seeds['EVAL']:
            for family in FAMILIES:
                for name, policy in policies.items():
                    w=world(seed,family)
                    evidence=rollout(w,policy,seed)
                    key=(seed,family,name)
                    receipts[key]=evidence
                    file=rollouts/f'{seed}-{family}-{name}'
                    file.with_suffix('.json').write_text(json.dumps(evidence,sort_keys=True)+'\n')
                    raster(w,3).save(file.with_suffix('.png'))
                    rows.append({"seed":seed,"family":family,"case":CASE_NAMES[seed%4],
                                 "policy":name,"steps":len(evidence['events']),"cells":evidence['unique'],
                                 "branches":evidence['branches'],"retraces":evidence['retraces'],
                                 "dead_ends":evidence['dead_ends'],"stopped":evidence['stopped']})
        for seed in list(seeds['EVAL'])[:gallery_seeds]:
            tile=288;names=tuple(policies)
            sheet=Image.new('RGB',(tile*len(names),(tile+20)*len(FAMILIES)),'white')
            draw=ImageDraw.Draw(sheet)
            for i,family in enumerate(FAMILIES):
                for j,name in enumerate(names):
                    proof=receipts[(seed,family,name)]
                    im=raster(replay(proof),3)
                    sheet.paste(im,(j*tile,i*(tile+20)+20))
                    draw.text((j*tile+2,i*(tile+20)+3),f'{family} {name} {proof["unique"]}',fill='black')
            sheet.save(out/f'm3-original-compare-{seed}.png')
        (out/'rollout_rows.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
        report['policies']={name:{"n":sum(r['policy']==name for r in rows),
                                   "mean_cells":round(statistics.mean(r['cells'] for r in rows if r['policy']==name),2),
                                   "mean_branches":round(statistics.mean(r['branches'] for r in rows if r['policy']==name),2),
                                   "mean_retraces":round(statistics.mean(r['retraces'] for r in rows if r['policy']==name),2),
                                   "early_stops":sum(r['stopped'] for r in rows if r['policy']==name),
                                   "mean_steps":round(statistics.mean(r['steps'] for r in rows if r['policy']==name),2)}
                            for name in policies}
        report['illegal_applied_actions']=0
        report['status']='COMPLETED'
    except BaseException as e:
        report['status']='FAILED';report['error']=repr(e)
        raise
    finally:
        report['wall_seconds']=round(time.perf_counter()-start,2)
        (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        (out/'manifest.json').write_text(json.dumps(manifest(out),indent=2,sort_keys=True)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--epochs',type=int,default=5)
    a=parser.parse_args()
    if not 1<=a.epochs<=20:parser.error('bounded epochs required')
    report=experiment(a.out,epochs=a.epochs)
    print(json.dumps({k:report[k] for k in ('status','splits','models','policies','wall_seconds')},indent=2))

if __name__=='__main__':main()
