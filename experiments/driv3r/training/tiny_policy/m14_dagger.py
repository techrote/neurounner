"""M14: bounded, research-only DAgger vs behavioral cloning for host-legal painting.

No native CyberSand use. The privileged BFS is *only* a label/reference; the
student gets 7x7 local synthetic observation and its current host waypoint.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import ACTIONS, DELTA, FEATURES, FAMILIES, World, raster

PROTOCOL='tiny-policy-m14-dagger-local-goal-v1'
TRAIN=range(71000,71012); DEV=range(72000,72004); EVAL=range(73000,73009)
BUDGET=96; OBS=FEATURES+5; CONTEXT=8
POLICIES=('oracle','greedy','bc-mlp','dagger-mlp','bc-gru','dagger-gru','random')
def sha(data):return hashlib.sha256(data).hexdigest()

def world(seed, family, split):
    if family not in FAMILIES or seed not in {'train':TRAIN,'dev':DEV,'eval':EVAL}[split]:
        raise ValueError('Disjoint split violation')
    w=World(seed,family,128)
    w.blocked[:]=True;w.protected[:]=False
    w.blocked[28:76,27:77]=False
    r=random.Random(seed*37+99)
    if split!='eval':
        # TRAIN: short staggered baffles / perforated rails / discrete islands.
        for k in range(7):
            x=33+k*5+r.randrange(-1,2)
            for y in range(32,72):
                if abs(y-48)<5 or (y+2*k)%14<5:continue
                w.blocked[y,x]=True
        for _ in range(14):
            x=r.randrange(31,72);y=r.randrange(31,71)
            if abs(x-48)+abs(y-48)>9:w.blocked[y:y+2,x:x+2]=True
    else:
        # EVAL: unseen U pockets and crossed rings; not the TRAIN baffle generator.
        variant=(seed-73000)%3
        if variant==0:
            for x in (34,45,57,68):
                for y in range(33,70):
                    if y in (39,40,51,52,63,64):continue
                    w.blocked[y,x]=True
            for y in (36,59):
                for x in range(33,71):
                    if x in (42,43,53,54,65,66):continue
                    w.blocked[y,x]=True
        elif variant==1:
            for x in range(35,69):
                w.blocked[36,x]=True;w.blocked[67,x]=True
            for y in range(36,68):
                w.blocked[y,35]=True;w.blocked[y,68]=True
            # Open offset gates, plus nested U-shaped cul-de-sacs.
            for x,y in ((35,49),(68,57),(47,36),(56,67)):
                w.blocked[y-1:y+2,x-1:x+2]=False
            for x in (41,61):
                w.blocked[41:60,x]=True
                w.blocked[41,41:62]=True
        else:
            for y in (35,43,55,65):
                w.blocked[y,33:71]=True
                gap=36+((y*3+seed)%25)
                w.blocked[y,gap:gap+5]=False
            for x in (37,59):
                w.blocked[35:66,x]=True
                w.blocked[46:51,x]=False
    for _ in range(9):
        x=r.randrange(31,72);y=r.randrange(31,71)
        if abs(x-48)+abs(y-48)>10:w.protected[y:y+2,x:x+2]=True
    w.blocked[45:52,45:52]=False;w.protected[45:52,45:52]=False
    return w

def motif(family):
    if family=='Greebles':return ((55,48),(55,40),(55,48),(64,48),(64,55),(64,48),(70,48),(70,40))
    if family=='Pipework':return ((54,48),(54,37),(65,37),(65,55),(72,55),(72,42),(61,42),(61,63))
    return ((58,48),(58,38),(39,38),(39,58),(58,58),(58,48),(69,48),(69,61))

def reachable(w):
    q=deque([(48,48)]);dist={(48,48):0}
    while q:
        x,y=q.popleft()
        for dx,dy in DELTA[:4]:
            xx,yy=x+dx,y+dy
            if not 0<=xx<96 or not 0<=yy<96 or (xx,yy) in dist or w.blocked[yy,xx] or w.protected[yy,xx]:continue
            dist[xx,yy]=dist[x,y]+1;q.append((xx,yy))
    if not dist:raise AssertionError('No reachable host cells')
    return [min(dist,key=lambda p:(abs(p[0]-gx)+abs(p[1]-gy),dist[p],p)) for gx,gy in motif(w.family)]

def advance(w,goals,idx):
    while idx<len(goals) and (w.x,w.y)==goals[idx]:idx+=1
    return idx

def obs(w,goals,idx):
    if idx==len(goals):gx,gy=w.x,w.y
    else:gx,gy=goals[idx]
    dx,dy=gx-w.x,gy-w.y
    return np.concatenate([w.observe(),np.array([dx/48,dy/48,(abs(dx)+abs(dy))/96,idx/len(goals),float(idx==len(goals))],np.float32)])

def oracle(w,goals,idx):
    if idx==len(goals):return 4
    goal=goals[idx]
    queue=deque([(w.x,w.y)]); first={(w.x,w.y):4}
    while queue:
        x,y=queue.popleft()
        if (x,y)==goal:return first[x,y]
        for a,(dx,dy) in enumerate(DELTA[:4]):
            nx,ny=x+dx,y+dy
            if not 0<=nx<96 or not 0<=ny<96 or (nx,ny) in first or w.blocked[ny,nx] or w.protected[ny,nx]:continue
            first[nx,ny]=a if (x,y)==(w.x,w.y) else first[x,y]
            queue.append((nx,ny))
    return 4

def greedy(w,goals,idx):
    if idx==len(goals):return 4
    gx,gy=goals[idx]
    moves=[a for a in w.legal() if a!=4]
    if not moves:return 4
    return min(moves,key=lambda a:(abs(gx-w.x-DELTA[a][0])+abs(gy-w.y-DELTA[a][1]),a!=w.heading,a))

class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(OBS,96),nn.ReLU(),nn.Linear(96,64),nn.ReLU(),nn.Linear(64,5))
    def forward(self,x):return self.net(x)
class GRU(nn.Module):
    def __init__(self):
        super().__init__();self.rnn=nn.GRU(OBS,48,batch_first=True);self.head=nn.Linear(48,5)
    def forward(self,x):return self.head(self.rnn(x)[0])
CTORS={'mlp':MLP,'gru':GRU}

def choose(w,goals,idx,policy,heads,hist,rng):
    if policy=='oracle':return oracle(w,goals,idx)
    if policy=='greedy':return greedy(w,goals,idx)
    if policy=='random':return rng.choice([a for a in w.legal() if a!=4] or [4])
    with torch.inference_mode():
        logits=heads[policy](torch.from_numpy(np.stack(hist[-CONTEXT:]))[None])[0,-1]
    return max(w.legal(),key=lambda a:(float(logits[a]),-a))

def collect(seeds,split,heads=None,dagger=False):
    data=[]
    for seed in seeds:
        for fam in FAMILIES:
            for episode in range(3 if heads is None else 2):
                w=world(seed,fam,split);goals=reachable(w);idx=0;xs=[];ys=[];ms=[]
                hist=[];rng=random.Random(seed*91+episode*19+FAMILIES.index(fam))
                for t in range(72):
                    idx=advance(w,goals,idx)
                    o=obs(w,goals,idx);hist.append(o)
                    a=oracle(w,goals,idx)
                    mask=np.zeros(5,np.bool_);mask[list(w.legal())]=True
                    xs.append(o);ys.append(a);ms.append(mask)
                    if heads is None:
                        move=a if episode==0 else greedy(w,goals,idx) if episode==1 else rng.choice([a for a in w.legal() if a!=4] or [4])
                    else:
                        # DAgger relabels STUDENT-VISITED states, never training on EVAL.
                        pol=('bc-mlp','bc-gru')[episode]
                        move=choose(w,goals,idx,pol,heads,hist,rng)
                        if rng.random()<.20:move=rng.choice([a for a in w.legal() if a!=4] or [4])
                    w.apply(int(move))
                    if w.stopped:break
                data.append((np.stack(xs),np.array(ys,np.int64),np.stack(ms)))
    return data

def tensorize(episodes):
    xs=[];ys=[];ms=[];vs=[]
    for xx,yy,mm in episodes:
        for k in range(0,len(yy),CONTEXT):
            n=min(CONTEXT,len(yy)-k)
            xs.append(np.pad(xx[k:k+n],((0,CONTEXT-n),(0,0))))
            ys.append(np.pad(yy[k:k+n],(0,CONTEXT-n)))
            ms.append(np.pad(mm[k:k+n],((0,CONTEXT-n),(0,0)),constant_values=True))
            vs.append(np.arange(CONTEXT)<n)
    return tuple(torch.from_numpy(np.asarray(a)) for a in (np.asarray(xs,np.float32),np.asarray(ys,np.int64),np.asarray(ms,np.bool_),np.asarray(vs,np.bool_)))

def optimize(net,dataset,steps,seed):
    X,Y,M,V=tensorize(dataset);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.002)
    generator=torch.Generator().manual_seed(seed)
    losses=[]
    for k in range(steps):
        ids=torch.randint(len(X),(32,),generator=generator)
        pred=net(X[ids]).masked_fill(~M[ids],-1e6)
        ce=nn.functional.cross_entropy(pred.reshape(-1,5),Y[ids].reshape(-1),reduction='none').reshape(-1,CONTEXT)
        loss=(ce*V[ids]).sum()/V[ids].sum()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
        if k%40==0:losses.append(round(float(loss.detach()),5))
    net.eval()
    return {'updates':steps,'train_masked_ce_samples':losses,'observations':sum(len(e[1]) for e in dataset)}

def rollout(seed,fam,policy,heads,render=None,recompute=True):
    w=world(seed,fam,'eval');goals=reachable(w);idx=0;hist=[];events=[]
    rng=random.Random(seed*103+FAMILIES.index(fam))
    for t in range(BUDGET):
        idx=advance(w,goals,idx);feature=obs(w,goals,idx);hist.append(feature)
        legal=list(w.legal());before=w.state_sha256()
        a=int(choose(w,goals,idx,policy,heads,hist,rng))
        assert a in legal,(seed,fam,policy,a,legal)
        applied=w.apply(a)
        events.append({'state':before,'observation':sha(feature.tobytes()),'legal':[ACTIONS[i] for i in legal],**applied})
        if w.stopped:break
    done=advance(w,goals,idx)
    record={'protocol':PROTOCOL,'seed':seed,'family':fam,'policy':policy,'synthetic_only':True,
            'geometry':sha(w.blocked.tobytes()+w.protected.tobytes()),'goals':goals,'events':events,
            'completed_goals':done,'cells':int(w.visited.sum()),'branches':w.branch_nodes(),
            'retraces':w.revisit_landings,'state':w.state_sha256()}
    if render:raster(w,2).save(render)
    return record

def replay(record,heads):
    r=rollout(record['seed'],record['family'],record['policy'],heads)
    if r!=record:raise AssertionError('Failed independent deterministic action/state replay')
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=160);args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable evidence path already exists')
    torch.manual_seed(1401);torch.set_num_threads(2);np.random.seed(1401)
    args.out.mkdir(parents=True);(args.out/'checkpoints').mkdir();(args.out/'rollouts').mkdir()
    bc=collect(TRAIN,'train');dev=collect(DEV,'dev')
    heads={};train={};losses={}
    for arch,ctor in CTORS.items():
        name='bc-'+arch;net=ctor();train[name]=optimize(net,bc,args.steps,1401+len(arch));heads[name]=net
    # Bounded one-pass DAgger: relabel BC student rollouts with the SAME privileged oracle.
    on_policy=collect(TRAIN,'train',heads,True)
    for arch,ctor in CTORS.items():
        name='dagger-'+arch;net=ctor();net.load_state_dict(heads['bc-'+arch].state_dict())
        train[name]=optimize(net,bc+on_policy,args.steps,5401+len(arch));heads[name]=net
    for name,net in heads.items():
        p=args.out/'checkpoints'/f'{name}.pt'
        torch.save({'protocol':PROTOCOL,'name':name,'state_dict':net.state_dict(),'updates':train[name]['updates'],'train_seed_range':[min(TRAIN),max(TRAIN)]},p)
        train[name]['checkpoint_sha256']=sha(p.read_bytes())
    rows=[]
    for seed in EVAL:
        for fam in FAMILIES:
            for pol in POLICIES:
                stem=f'{seed}-{fam}-{pol}'
                rec=rollout(seed,fam,pol,heads,args.out/'rollouts'/f'{stem}.png')
                replay(rec,heads)
                (args.out/'rollouts'/f'{stem}.json').write_text(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n')
                rows.append({k:rec[k] for k in ('seed','family','policy','cells','branches','retraces','completed_goals')})
    summary={p:{key:round(statistics.mean(r[key] for r in rows if r['policy']==p),3) for key in ('cells','branches','retraces','completed_goals')} for p in POLICIES}
    for arch in CTORS:
        delta=[]
        for seed in EVAL:
            def avg(pol):return statistics.mean(r['completed_goals'] for r in rows if r['policy']==pol and r['seed']==seed)
            delta.append(avg('dagger-'+arch)-avg('bc-'+arch))
        rng=random.Random(1401);boots=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(2000))
        summary['dagger-'+arch]['delta_vs_bc_goals']={'mean':round(statistics.mean(delta),3),'seed_cluster_bootstrap_95':[round(boots[49],3),round(boots[1949],3)]}
    report={'protocol':PROTOCOL,'synthetic_only':True,'limitations':'Host-preplanned goals and privileged BFS oracle; not native evidence or autonomous drawing; DAgger labels include BC rollout states but no EVAL exposure.',
            'seeds':{'train':list(TRAIN),'dev':list(DEV),'eval':list(EVAL)},'training':train,
            'episodes':{'bc':len(bc),'dagger':len(on_policy),'dev':len(dev)},'rollouts':len(rows),'results':summary}
    (args.out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (args.out/'rows.json').write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
    from PIL import Image
    for seed in list(EVAL)[:2]:
        img=Image.new('RGB',(192*len(POLICIES),192*len(FAMILIES)),(240,238,233))
        for fi,fam in enumerate(FAMILIES):
            for pi,pol in enumerate(POLICIES):
                with Image.open(args.out/'rollouts'/f'{seed}-{fam}-{pol}.png') as src:img.paste(src,(pi*192,fi*192))
        img.save(args.out/f'gallery-{seed}.png')
    manifest={p.relative_to(args.out).as_posix():sha(p.read_bytes()) for p in args.out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'results':summary,'rollouts':len(rows),'optimizer_updates':4*args.steps,'hashes':len(manifest)},indent=2))
if __name__=='__main__':main()
