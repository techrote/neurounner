"""M10: train a tiny family-sensitive router over two immutable M8 decision heads.

Synthetic 96x96 host only. Train/dev seeds and *geometric shape families* are
isolated from evaluation. Never reads native FINAL or edits production sources.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageDraw
from .core import World, ACTIONS, DELTA, FAMILIES, choose_random, raster
from .m3_curriculum import models
from .m5_topology_train import novel

PROTOCOL='tiny-policy-m10-ood-router-v1'
M8_HASH={'novel-mlp':'68defcf794d57392d898bf217efe2fd5afe73f1b91a0303a702000b2ed4c39e2',
         'style-mlp':'9a8f416c61d3bb3ee4e5c06ae0be0e566d9da02a2896a7c8526363481fc0033a'}
TRAIN=tuple(range(44000,44012));DEV=tuple(range(45000,45004));EVAL=tuple(range(46000,46006))
SHAPES=('ratchet','branch-rail','windmill','double-ladder','nested-u','fan-spokes')
POLICIES=('random','novel-greedy','m8-novel','m8-style','fixed-25','fixed-50','fixed-75','trained-router','local-teacher')
BUDGET=64

def sha(b):return hashlib.sha256(b).hexdigest()
def jdump(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False)

def world(seed,family,shape):
    if family not in FAMILIES or shape not in range(6) or seed not in TRAIN+DEV+EVAL:
        raise ValueError('Invalid world allocation')
    w=World(seed,family,budget=BUDGET)
    w.blocked[:]=True;w.protected[:]=False
    r=random.Random(seed*12391+shape*8971)
    def box(x0,y0,x1,y1):w.blocked[max(1,y0):min(95,y1+1),max(1,x0):min(95,x1+1)]=False
    def line(x0,y0,x1,y1,thick=0):
        if x0!=x1 and y0!=y1:raise ValueError('Diagonal segment')
        box(min(x0,x1)-thick,min(y0,y1)-thick,max(x0,x1)+thick,max(y0,y1)+thick)
    box(45,45,51,51)
    if shape==0:  # ratchets with crosscuts
        for i in range(6):
            y=24+i*9;line(26,y,72,y);line(26 if i%2 else 72,y,26 if i%2 else 72,y+9 if i<5 else y)
        line(48,48,48,24);line(48,48,72,48)
    elif shape==1:
        line(22,48,74,48)
        for i,x in enumerate(range(25,75,7)):
            endpoint=24+r.randrange(0,14) if i%2 else 61+r.randrange(0,13)
            line(x,48,x,endpoint)
            if i%3==0:line(x,endpoint,x+3,endpoint)
        line(36,30,64,30);line(36,67,64,67)
    elif shape==2:
        for x,y in ((20,28),(74,28),(20,71),(74,71)):
            line(48,48,x,48);line(x,48,x,y);line(x,y,48,y)
        line(48,48,48,20);line(48,48,48,77)
    elif shape==3:  # Evaluation-only disconnected-looking double ladder
        for x in (31,48,65):line(x,20,x,79)
        for y in range(24,78,9):
            line(31,y,65,y)
            if (y//9)%2:box(28,y-2,33,y+2)
        line(48,48,65,48)
    elif shape==4: # Evaluation-only nested U-shaped recovery traps
        for rad in (10,18,27):
            a,b=48-rad,48+rad
            line(a,a,a,b);line(a,b,b,b);line(b,b,b,a+5)
            line(a,a,b-6,a)
        line(48,48,48,75);line(48,48,74,48)
    else:  # Evaluation-only branch fan with long dead ends
        for j in range(9):
            x=24+j*6
            line(x,25,x,76)
            line(x,48,48,48)
            if j%2==0:line(x,25,x+2,25)
        line(24,62,72,62);line(48,48,48,24)
    # Seed-dependent protection gates, never cover the spawn.
    options=[]
    for y,x in np.argwhere(~w.blocked):
        if abs(int(x)-48)+abs(int(y)-48)>10 and r.random()<.018:options.append((int(y),int(x)))
    for y,x in options[:18]:w.protected[y,x]=True
    w.blocked[48,48]=False;w.protected[48,48]=False
    if not any(a!=4 for a in w.legal()):raise AssertionError('Trapped start')
    return w

def target(w):
    moves=[a for a in w.legal() if a!=4]
    if not moves:return 4
    t=len(w.actions)
    scores=[]
    for a in moves:
        dx,dy=DELTA[a];x,y=w.x+dx,w.y+dy
        fresh=not w.visited[y,x]
        future=sum(0<=x+ox<96 and 0<=y+oy<96 and not (w.blocked[y+oy,x+ox] or w.protected[y+oy,x+ox] or w.visited[y+oy,x+ox]) for ox,oy in ((0,1),(0,-1),(1,0),(-1,0),(0,2),(0,-2),(2,0),(-2,0)))
        straight=(a==w.heading)
        style= (1.7*int(not straight)+.4*int(a in (0,3))) if w.family=='Greebles' else (
            (1.9*int(straight)+.2*int(a in (1,3))) if w.family=='Pipework' else
            1.7*int(straight if (t//7)%2==0 else not straight)+.3*int(a in (0,3)))
        scores.append((4*int(fresh)+.22*future+style,-a,a))
    return max(scores)[2]

def load_heads(ckpt_dir):
    loaded={}
    for name,expected in M8_HASH.items():
        path=ckpt_dir/(name+'.pt')
        if sha(path.read_bytes())!=expected:raise ValueError('M8 frozen checkpoint hash mismatch '+name)
        ck=torch.load(path,weights_only=True,map_location='cpu')
        if ck['model']!=name or ck['protocol']!='tiny-policy-m8-style-distill-v1' or ck['updates']!=200:raise ValueError('Checkpoint provenance mismatch')
        net=models(torch)['mlp'];net.load_state_dict(ck['state_dict'],strict=True);net.eval()
        for p in net.parameters():p.requires_grad_(False)
        loaded[name]=net
    return loaded

def masked_logits(w,heads):
    x=torch.from_numpy(w.observe()[None,:]);mask=torch.zeros((1,5),dtype=torch.bool);mask[0,list(w.legal())]=True
    # Policy-side explicit STOP guard: admissible only when no movement possible.
    if len(w.legal())>1:mask[0,4]=False
    with torch.no_grad():
        one=heads['novel-mlp'](x).masked_fill(~mask,-1e5)[0]
        two=heads['style-mlp'](x).masked_fill(~mask,-1e5)[0]
    return x[0],mask[0],torch.stack((one.log_softmax(-1),two.log_softmax(-1)))

class Router(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gate=torch.nn.Sequential(torch.nn.Linear(162,24),torch.nn.Tanh(),torch.nn.Linear(24,1))
    def forward(self,observations,paired):
        lam=torch.sigmoid(self.gate(observations)).squeeze(-1)
        return paired[:,0,:]*(1-lam[:,None])+paired[:,1,:]*lam[:,None],lam

def dataset(seeds,shapes,heads,turns=36):
    xx=[];pp=[];labels=[];identities=[]
    for seed in seeds:
        for shape in shapes:
            for fam in FAMILIES:
                for p in ('novel','random','target'):
                    w=world(seed,fam,shape);rng=random.Random(seed+shape*997+FAMILIES.index(fam)*31)
                    for tick in range(turns):
                        if not w.legal():break
                        x,mask,paired=masked_logits(w,heads)
                        label=target(w)
                        if not mask[label]:raise AssertionError('Label illegal')
                        xx.append(x);pp.append(paired);labels.append(label)
                        identities.append((seed,shape,fam,p,tick,w.state_sha256()))
                        a=novel(w,rng) if p=='novel' else (choose_random(w,rng) if p=='random' else label)
                        w.apply(int(a))
                        if w.stopped:break
    return (torch.stack(xx),torch.stack(pp),torch.tensor(labels,dtype=torch.long)),sha(jdump(identities).encode())

def train(heads,out):
    torch.manual_seed(1211);torch.set_num_threads(1)
    (X,P,Y),trainid=dataset(TRAIN,(0,1,2),heads)
    (DX,DP,DY),devid=dataset(DEV,(0,1,2),heads)
    model=Router();optimizer=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.0005)
    rng=torch.Generator().manual_seed(1010)
    history=[];best=1e5;best_sd=None
    for update in range(300):
        idx=torch.randint(len(X),(256,),generator=rng)
        outlog,alpha=model(X[idx],P[idx]);loss=torch.nn.functional.cross_entropy(outlog,Y[idx])+.02*(alpha-.35).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (update+1)%25==0:
            with torch.no_grad():
                dlog,dalpha=model(DX,DP);dev=float(torch.nn.functional.cross_entropy(dlog,DY))
                agree=float((dlog.argmax(1)==DY).float().mean())
            history.append({'updates':update+1,'dev_ce':round(dev,6),'agreement':round(agree,5),'style_weight':round(float(dalpha.mean()),5)})
            if dev<best:best=dev;best_sd={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_sd);model.eval()
    ck={'protocol':PROTOCOL,'train_id':trainid,'dev_id':devid,'updates':300,'state_dict':best_sd,'m8_hash':M8_HASH}
    torch.save(ck,out/'router.pt')
    return model,{'updates':300,'training_examples':len(X),'dev_examples':len(DX),'train_id':trainid,'dev_id':devid,'checkpoint_sha256':sha((out/'router.pt').read_bytes()),'parameters':sum(p.numel() for p in model.parameters()),'history':history}

def choice(w,policy,heads,router,rng):
    if policy=='random':return choose_random(w,rng)
    if policy=='novel-greedy':return novel(w,rng)
    if policy=='local-teacher':return target(w)
    obs,mask,paired=masked_logits(w,heads)
    if policy=='m8-novel':logits=paired[0]
    elif policy=='m8-style':logits=paired[1]
    elif policy.startswith('fixed-'):
        lam=int(policy.split('-')[1])/100
        logits=paired[0]*(1-lam)+paired[1]*lam
    else:
        with torch.no_grad():logits,_=router(obs[None,:],paired[None,:,:]);logits=logits[0]
    return min((-float(logits[a]),a) for a in range(5) if mask[a])[1]

def rollout(seed,shape,family,policy,heads,router):
    w=world(seed,family,shape);geo=sha(w.blocked.tobytes()+w.protected.tobytes());events=[]
    rng=random.Random(seed*43+shape*1027+FAMILIES.index(family)*97)
    for _ in range(BUDGET):
        if not w.legal():break
        pre=w.state_sha256();ob=sha(w.observe().tobytes());legal=[ACTIONS[i] for i in w.legal()]
        a=int(choice(w,policy,heads,router,rng))
        if a not in w.legal():raise AssertionError('Illegal action')
        row={'pre_sha256':pre,'observation_sha256':ob,'legal':legal};row.update(w.apply(a));events.append(row)
        if w.stopped:break
    rec={'protocol':PROTOCOL,'seed':seed,'shape':shape,'shape_name':SHAPES[shape],'family':family,'policy':policy,
         'synthetic_only':True,'geometry_sha256':geo,'budget':BUDGET,'events':events,'final_sha256':w.state_sha256(),
         'unique':int(w.visited.sum()),'branches':w.branch_nodes(),'retraces':w.revisit_landings,
         'direction_changes':sum(a!=b for a,b in zip(w.actions,w.actions[1:]) if a!='STOP' and b!='STOP'),
         'early_stop':bool(w.actions and w.actions[-1]=='STOP')}
    return rec,w

def replay(rec):
    assert rec['protocol']==PROTOCOL and rec['synthetic_only'] is True and rec['seed'] in EVAL
    assert rec['shape'] in (3,4,5) and rec['policy'] in POLICIES and rec['budget']==BUDGET
    w=world(rec['seed'],rec['family'],rec['shape'])
    assert sha(w.blocked.tobytes()+w.protected.tobytes())==rec['geometry_sha256']
    for e in rec['events']:
        assert w.state_sha256()==e['pre_sha256'] and sha(w.observe().tobytes())==e['observation_sha256']
        assert e['legal']==[ACTIONS[i] for i in w.legal()] and e['action'] in e['legal']
        got=w.apply(ACTIONS.index(e['action']))
        assert all(e[k]==v for k,v in got.items())
    assert w.state_sha256()==rec['final_sha256'] and int(w.visited.sum())==rec['unique']
    assert w.branch_nodes()==rec['branches'] and w.revisit_landings==rec['retraces']
    return w

def family_iou(worlds):
    maps=[w.visited for w in worlds]
    scores=[]
    for a,b in ((0,1),(0,2),(1,2)):
        inter=np.logical_and(maps[a],maps[b]).sum();union=np.logical_or(maps[a],maps[b]).sum()
        scores.append(float(inter/union))
    return statistics.mean(scores)

def run(out,ckpt_dir):
    if out.exists():raise FileExistsError('No overwrite of immutable evidence')
    out.mkdir(parents=True);(out/'rollouts').mkdir()
    heads=load_heads(ckpt_dir);router,training=train(heads,out)
    # EVAL is never used for target training, dev selection, or gate tuning.
    rows=[];pairs=defaultdict(dict);gallery={}
    for seed in EVAL:
        for shape in (3,4,5):
            for fam in FAMILIES:
                for policy in POLICIES:
                    rec,w=rollout(seed,shape,fam,policy,heads,router)
                    chk=replay(rec)
                    assert np.array_equal(chk.visited,w.visited)
                    key=f'{seed}-{shape}-{fam}-{policy}'
                    (out/'rollouts'/f'{key}.json').write_text(jdump(rec)+'\n')
                    raster(w,1).save(out/'rollouts'/f'{key}.png')
                    rows.append({k:rec[k] for k in ('seed','shape','family','policy','unique','branches','retraces','direction_changes','early_stop')})
                    pairs[(seed,shape,policy)][fam]=w
                    if seed==EVAL[0] and shape==4:gallery[fam,policy]=w
    metrics={}
    for policy in POLICIES:
        group=[r for r in rows if r['policy']==policy]
        mean=lambda k:round(statistics.mean(r[k] for r in group),4)
        iou=[family_iou([pairs[s,sh,policy][f] for f in FAMILIES]) for s in EVAL for sh in (3,4,5)]
        metrics[policy]={k:mean(k) for k in ('unique','branches','retraces','direction_changes')}
        metrics[policy]['family_iou']=round(statistics.mean(iou),4)
        metrics[policy]['early_stop']=sum(r['early_stop'] for r in group)
    cols=len(POLICIES);canvas=Image.new('RGB',(cols*194,len(FAMILIES)*219),(250,249,246));d=ImageDraw.Draw(canvas)
    for fi,fam in enumerate(FAMILIES):
        for ci,pol in enumerate(POLICIES):
            d.text((ci*194+4,fi*219+2),fam[:4]+' '+pol,fill=(0,0,0))
            canvas.paste(raster(gallery[fam,pol],2),(ci*194,fi*219+22))
    canvas.save(out/'original-nested-u-46000.png')
    report={'protocol':PROTOCOL,'source_sha256':sha(Path(__file__).read_bytes()),'archived_dependencies_not_exact_head':True,
            'host':'reconstructed synthetic W0, not native','action_strings':ACTIONS,'m8_checkpoints':M8_HASH,
            'train_shapes':SHAPES[:3],'eval_shapes':SHAPES[3:],'train_seeds':TRAIN,'dev_seeds':DEV,'eval_seeds':EVAL,
            'training':training,'trajectories':len(rows),'decisions':sum(len(json.loads(p.read_text())['events']) for p in (out/'rollouts').glob('*.json')),
            'applied_illegal':0,'metrics':metrics,'promotion':'NONE'}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (out/'rows.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    manifest={str(p.relative_to(out)):sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--m8',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.output,a.m8),indent=2))
