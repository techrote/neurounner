"""M11: distinct temporal tiny policies trained for legal CyberSand-like trajectory decisions.

Research-only 96x96 synthetic host. Not a native-game experiment. TRAIN/DEV
and blind geometry families are disjoint; labels use privileged shortest-frontier
planning, but inference sees only core.World.observe() and host legal masks.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import hashlib, json, random, statistics, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import World, ACTIONS, DELTA, FAMILIES, FEATURES, choose_random, raster

PROTOCOL = 'tiny-policy-m11-memory-frontier-v1'
TRAIN = range(51000,51036)
DEV = range(52000,52008)
EVAL = range(53000,53012)
TURNS = 80
EVAL_TURNS = 96
CONTEXT = 16
ARCHS = ('mlp','gru','tcn')
POLICIES = ('planner','novel','random') + ARCHS


def sha(buf): return hashlib.sha256(buf).hexdigest()


def world(seed, family, split):
    """Entirely new corridor topology family; EVAL topologies never used in training."""
    allowed = {'train': TRAIN, 'dev': DEV, 'eval': EVAL}[split]
    if seed not in allowed or family not in FAMILIES: raise ValueError('Seed/split leakage')
    w = World(seed, family, budget=128)
    w.blocked[:]=True; w.protected[:]=False
    rng=random.Random(seed*9147+277)
    def box(x0,y0,x1,y1):
        x0,x1=sorted((x0,x1));y0,y1=sorted((y0,y1))
        w.blocked[max(2,y0):min(94,y1+1),max(2,x0):min(94,x1+1)]=False
    def line(x0,y0,x1,y1,thick=0):
        if x0!=x1 and y0!=y1: raise ValueError('nonorthogonal')
        box(x0-thick,y0-thick,x1+thick,y1+thick)
    shape=(seed-(53000 if split=='eval' else 51000 if split=='train' else 52000))%(2 if split=='eval' else 4)
    jitter=rng.randrange(-3,4)
    if split!='eval':
        if shape==0: # ring with perpendicular escape ribs
            for r in (10,22,32):
                line(48-r,48-r,48+r,48-r);line(48-r,48+r,48+r,48+r)
                line(48-r,48-r,48-r,48+r);line(48+r,48-r,48+r,48+r)
            line(48,48,48,16);line(48,48,80,48)
            for x in (30,66):line(x,27,x,69)
        elif shape==1: # rail switchyard with cul-de-sacs
            for x in (24,36,48,60,72):line(x,17,x,79)
            for y in (26,46,66):line(20,y,76,y)
            for x in (29,42,55,68):line(x,36,x,58)
        elif shape==2: # comb with staggered bridges
            line(15,48,80,48)
            for i,x in enumerate(range(19,78,5)):
                line(x,48,x,20+jitter if i%2 else 76-jitter)
            for y in (30,67):line(26,y,70,y)
        else: # four-room manifold with choke doors
            for x in (23,72):
                for y in (23,72):box(x-10,y-10,x+10,y+10)
            line(48,48,48,15);line(48,48,48,82)
            line(15,48,82,48);line(48,48,23,48);line(23,48,23,23)
            for y in (30,65):line(23,y,72,y)
    else:
        if shape==0: # untrained skewed double helix / return loops
            for x in range(18,80,9):
                line(x,17,x,79)
                if x<72:line(x,17 if x%2 else 79,x+9,17 if x%2 else 79)
            for y in (30,48,64):line(18,y,79,y)
            for x in (30,60):line(x,48,x,48+jitter)
        else: # untrained nested chambers with bottlenecks
            for radius in (11,21,32):
                box(48-radius,48-radius,48+radius,48-radius+1)
                box(48-radius,48+radius-1,48+radius,48+radius)
                box(48-radius,48-radius,48-radius+1,48+radius)
                box(48+radius-1,48-radius,48+radius,48+radius)
                line(48,48,48,48-radius);line(48,48,48+radius,48)
            for x in (26,70):line(x,25,x,70)
    box(46,46,50,50)
    free=np.argwhere(~w.blocked)
    for y,x in free:
        y=int(y);x=int(x)
        if abs(x-48)+abs(y-48)>18 and rng.random()<.012:
            w.protected[y,x]=True
    w.protected[48,48]=False
    w.blocked[48,48]=False
    if len(w.legal())<=1:raise ValueError('Trapped start')
    return w


def nearest_frontier(w):
    """Privileged global BFS teacher ONLY, never a model feature."""
    moves=[a for a in w.legal() if a!=4]
    if not moves:return 4
    # Local novelty/style preferences under a bounded deterministic tie break.
    fam=FAMILIES.index(w.family);turn=len(w.actions)
    def sty(a,endpoint):
        dx,dy=DELTA[a];x,y=endpoint
        if fam==0:return (a!=w.heading)*.48 + ((turn//7)%2==0)*(a in (0,3))*.15
        if fam==1:return (a==w.heading)*.60 + (a in (1,3))*.12
        return ((turn//9)%2==0)*(a==w.heading)*.45 + ((turn//9)%2==1)*(a!=w.heading)*.40
    start=(w.x,w.y)
    q=deque([(w.x,w.y,None,0)])
    seen={start};best=[];bestdepth=None
    while q:
        x,y,first,depth=q.popleft()
        if bestdepth is not None and depth>bestdepth:break
        if depth and not w.visited[y,x]:
            bestdepth=depth;best.append((sty(first,(x,y)),-first,first));continue
        if depth>60:continue
        for a,(dx,dy) in enumerate(DELTA[:4]):
            xx,yy=x+dx,y+dy
            if (xx,yy) in seen or not (1<=xx<95 and 1<=yy<95):continue
            if w.blocked[yy,xx] or w.protected[yy,xx]:continue
            seen.add((xx,yy));q.append((xx,yy,first if first is not None else a,depth+1))
    if best:return max(best)[2]
    return max(moves,key=lambda a:(sty(a,(w.x+DELTA[a][0],w.y+DELTA[a][1])),-a))


def novelty(w):
    moves=[a for a in w.legal() if a!=4]
    if not moves:return 4
    return max(moves,key=lambda a:(int(not w.visited[w.y+DELTA[a][1],w.x+DELTA[a][0]]),a==w.heading,-a))


class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Linear(FEATURES,96),nn.ReLU(),nn.Linear(96,64),nn.ReLU(),nn.Linear(64,5))
    def forward(self,x):return self.layers(x)

class GRU(nn.Module):
    def __init__(self):
        super().__init__();self.cell=nn.GRU(FEATURES,48,batch_first=True);self.head=nn.Linear(48,5)
    def forward(self,x):return self.head(self.cell(x)[0])

class TCN(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Linear(FEATURES,48)
        self.conv=nn.ModuleList([nn.Conv1d(48,48,3,dilation=d) for d in (1,2,4)])
        self.out=nn.Linear(48,5)
    def forward(self,x):
        x=self.embed(x).relu().transpose(1,2)
        for d,conv in zip((1,2,4),self.conv):
            x=(x+conv(nn.functional.pad(x,(2*d,0))).relu())
        return self.out(x.transpose(1,2))


def make_episodes(seeds,split):
    ep=[]; identities=[]; labels=Counter()
    for seed in seeds:
        for family in FAMILIES:
            for exploration in ('planner','random','novel'):
                w=world(seed,family,split);rng=random.Random(seed*47+FAMILIES.index(family)*13)
                ob=[];ys=[];masks=[]
                for t in range(TURNS):
                    legal=w.legal()
                    if not legal:break
                    action=nearest_frontier(w)
                    if action not in legal:raise ValueError('Invalid teacher action')
                    o=w.observe();m=np.zeros(5,np.bool_);m[list(legal)]=True
                    ob.append(o);ys.append(action);masks.append(m)
                    identities.append([seed,family,exploration,t,w.state_sha256(),ACTIONS[action]])
                    labels[ACTIONS[action]]+=1
                    if exploration=='planner':move=action
                    elif exploration=='random':move=choose_random(w,rng)
                    else:move=novelty(w)
                    w.apply(int(move))
                    if w.stopped:break
                ep.append((np.stack(ob),np.asarray(ys,np.int64),np.stack(masks)))
            # Host-legal STOP-only true protected trap
            trap=world(seed,family,split)
            for dx,dy in DELTA[:4]:trap.protected[trap.y+dy,trap.x+dx]=True
            assert trap.legal()==(4,)
            mm=np.zeros(5,np.bool_);mm[4]=True
            ep.append((trap.observe()[None,:],np.asarray([4],np.int64),mm[None,:]))
            identities.append([seed,family,'trap',0,trap.state_sha256(),'STOP']);labels['STOP']+=1
    meta={'seed_groups':list(seeds),'worlds':len(seeds)*len(FAMILIES),'examples':len(identities),
          'teacher_action_histogram':dict(labels), 'identity_sha256':sha(json.dumps(identities,separators=(',',':')).encode())}
    return ep,meta


def batches(episodes):
    # Non-overlapping truncated BPTT with explicit reset at 16-decision boundaries.
    xx=[];yy=[];mm=[];valid=[]
    for obs,y,mask in episodes:
        for k in range(0,len(y),CONTEXT):
            a=obs[k:k+CONTEXT];b=y[k:k+CONTEXT];m=mask[k:k+CONTEXT];n=len(b)
            xx.append(np.pad(a,((0,CONTEXT-n),(0,0))))
            yy.append(np.pad(b,(0,CONTEXT-n)))
            mm.append(np.pad(m,((0,CONTEXT-n),(0,0)),constant_values=True))
            valid.append(np.arange(CONTEXT)<n)
    return [torch.from_numpy(z) for z in (np.asarray(xx,np.float32),np.asarray(yy,np.int64),np.asarray(mm,np.bool_),np.asarray(valid,np.bool_))]


def train(out,epochs=8):
    torch.manual_seed(1117);np.random.seed(1117);torch.set_num_threads(2)
    tr,train_meta=make_episodes(TRAIN,'train');dv,dev_meta=make_episodes(DEV,'dev')
    X,Y,M,V=batches(tr);DX,DY,DM,DV=batches(dv)
    results={};heads={}
    (out/'checkpoints').mkdir(parents=True,exist_ok=True)
    for name,ctor in (('mlp',MLP),('gru',GRU),('tcn',TCN)):
        net=ctor();opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.002)
        history=[];best=float('inf');state=None;updates=0
        for e in range(epochs):
            order=torch.randperm(len(X),generator=torch.Generator().manual_seed(8800+e))
            net.train()
            for idx in order.split(32):
                logits=net(X[idx]).masked_fill(~M[idx],-1e6)
                ce=nn.functional.cross_entropy(logits.reshape(-1,5),Y[idx].reshape(-1),reduction='none').reshape(-1,CONTEXT)
                loss=(ce*V[idx]).sum()/V[idx].sum()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.5);opt.step();updates+=1
            net.eval()
            with torch.inference_mode():
                lv=[];av=[]
                for j in range(0,len(DX),32):
                    logits=net(DX[j:j+32]).masked_fill(~DM[j:j+32],-1e6)
                    yy=DY[j:j+32];valid=DV[j:j+32]
                    ce=nn.functional.cross_entropy(logits.reshape(-1,5),yy.reshape(-1),reduction='none').reshape_as(yy)
                    lv.append((ce*valid).sum().item());av.append(((logits.argmax(-1)==yy)&valid).sum().item())
                dev_ce=sum(lv)/int(DV.sum());acc=sum(av)/int(DV.sum())
            history.append({'epoch':e+1,'dev_cross_entropy':round(dev_ce,6),'dev_agreement':round(acc,6)})
            if dev_ce<best:
                best=dev_ce;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        net.load_state_dict(state);net.eval()
        payload={'protocol':PROTOCOL,'architecture':name,'train_identity':train_meta['identity_sha256'],
                 'dev_identity':dev_meta['identity_sha256'],'optimizer_updates':updates,'state_dict':state}
        p=out/'checkpoints'/f'{name}.pt';torch.save(payload,p)
        results[name]={'parameters':sum(p.numel() for p in net.parameters()),'optimizer_updates':updates,
                       'checkpoint_sha256':sha(p.read_bytes()),'dev_history':history}
        heads[name]=net
    return heads,{'train':train_meta,'dev':dev_meta,'training':results}


def decide(w,policy,heads,rng,obs_hist):
    if policy=='planner':return nearest_frontier(w)
    if policy=='random':return choose_random(w,rng)
    if policy=='novel':return novelty(w)
    net=heads[policy];hist=torch.from_numpy(np.stack(obs_hist[-CONTEXT:],axis=0))[None,:]
    with torch.inference_mode():logits=net(hist)[0,-1]
    return max(w.legal(),key=lambda a:(float(logits[a]),-a))


def rollout(seed,family,policy,heads):
    w=world(seed,family,'eval');base=sha(w.blocked.tobytes()+w.protected.tobytes())
    rng=random.Random(seed*131+FAMILIES.index(family)*79)
    history=[];events=[]
    for _ in range(EVAL_TURNS):
        legal=w.legal()
        if not legal:break
        obs=w.observe();history.append(obs)
        a=decide(w,policy,heads,rng,history)
        assert a in legal
        row={'before_sha256':w.state_sha256(),'observation_sha256':sha(obs.tobytes()),'legal':[ACTIONS[x] for x in legal]}
        row.update(w.apply(int(a)));events.append(row)
        if w.stopped:break
    changes=sum(w.actions[i]!=w.actions[i-1] for i in range(1,len(w.actions)) if w.actions[i]!='STOP' and w.actions[i-1]!='STOP')
    rec={'protocol':PROTOCOL,'synthetic_only':True,'seed':seed,'family':family,'policy':policy,
         'geometry_sha256':base,'budget':EVAL_TURNS,'events':events,'state_sha256':w.state_sha256(),
         'cells':int(w.visited.sum()),'branches':w.branch_nodes(),'retraces':w.revisit_landings,
         'steps':len(events),'turns':changes,'early_stop':bool(w.actions and w.actions[-1]=='STOP')}
    return rec,w


def replay(rec):
    assert rec['protocol']==PROTOCOL and rec['seed'] in EVAL and rec['family'] in FAMILIES and rec['policy'] in POLICIES
    assert rec['budget']==EVAL_TURNS and 1<=len(rec['events'])<=EVAL_TURNS
    w=world(rec['seed'],rec['family'],'eval')
    assert sha(w.blocked.tobytes()+w.protected.tobytes())==rec['geometry_sha256']
    for e in rec['events']:
        assert e['before_sha256']==w.state_sha256()
        assert e['observation_sha256']==sha(w.observe().tobytes())
        assert e['legal']==[ACTIONS[a] for a in w.legal()]
        a=ACTIONS.index(e['action']);assert a in w.legal()
        actual=w.apply(a);assert actual=={k:e[k] for k in actual}
    assert w.state_sha256()==rec['state_sha256'] and int(w.visited.sum())==rec['cells']
    assert w.branch_nodes()==rec['branches'] and w.revisit_landings==rec['retraces']
    return w


def evaluate(out,heads):
    rows=[];worlds={}
    p=out/'rollouts';p.mkdir(parents=True)
    start=time.perf_counter()
    for seed in EVAL:
        for family in FAMILIES:
            for policy in POLICIES:
                rec,w=rollout(seed,family,policy,heads)
                rw=replay(rec);assert np.array_equal(rw.visited,w.visited)
                stem=f'{seed}-{family}-{policy}'
                (p/f'{stem}.json').write_text(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n')
                raster(w,2).save(p/f'{stem}.png')
                rows.append({k:rec[k] for k in ('seed','family','policy','cells','branches','retraces','steps','turns','early_stop')})
                worlds[(seed,family,policy)]=w
    (out/'rows.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    # Original host PNG contact sheets, no generated or redrawn trajectories.
    from PIL import Image, ImageDraw
    for seed in list(EVAL)[:2]:
        tile=192; sheet=Image.new('RGB',(tile*len(POLICIES),tile*len(FAMILIES)),(243,240,234))
        for i,family in enumerate(FAMILIES):
            for j,policy in enumerate(POLICIES):
                im=raster(worlds[(seed,family,policy)],2)
                sheet.paste(im,(j*tile,i*tile))
        sheet.save(out/f'gallery-{seed}.png')
    summaries={}
    for policy in POLICIES:
        selected=[r for r in rows if r['policy']==policy]
        summaries[policy]={k:round(statistics.mean(float(r[k]) for r in selected),3) for k in ('cells','branches','retraces','turns')}
        summaries[policy]['early_stops']=sum(bool(r['early_stop']) for r in selected)
    # Cluster by seed, no per-family independence fiction.
    import itertools
    pair={'gru_minus_mlp':[],'tcn_minus_mlp':[],'gru_minus_novel':[]}
    for seed in EVAL:
        def avg(name):return statistics.mean(next(r['cells'] for r in rows if r['seed']==seed and r['family']==f and r['policy']==name) for f in FAMILIES)
        pair['gru_minus_mlp'].append(avg('gru')-avg('mlp'))
        pair['tcn_minus_mlp'].append(avg('tcn')-avg('mlp'))
        pair['gru_minus_novel'].append(avg('gru')-avg('novel'))
    for key,diffs in pair.items():
        rand=random.Random(1137);draws=[statistics.mean(rand.choices(diffs,k=len(diffs))) for _ in range(2000)]
        draws.sort();summaries[key]={'difference':round(statistics.mean(diffs),3),'seed_cluster_bootstrap_ci95':[round(draws[50],3),round(draws[1949],3)]}
    # Family-conditional painted footprint overlap (same geometry seed).
    for policy in POLICIES:
        ious=[]
        for seed in EVAL:
            for a,b in itertools.combinations(FAMILIES,2):
                x=worlds[(seed,a,policy)].visited;y=worlds[(seed,b,policy)].visited
                ious.append(float(np.logical_and(x,y).sum())/max(1,int(np.logical_or(x,y).sum())))
        summaries[policy]['cross_family_iou']=round(statistics.mean(ious),4)
    return rows,summaries,round(time.perf_counter()-start,2)


def manifest(out):
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json')
    data={p.relative_to(out).as_posix():sha(p.read_bytes()) for p in files}
    (out/'manifest.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    return len(data)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=8)
    args=ap.parse_args();out=args.out
    if out.exists():raise FileExistsError('Never overwrite trial evidence')
    out.mkdir(parents=True);t=time.perf_counter()
    heads,training=train(out,args.epochs)
    rows,summaries,eval_seconds=evaluate(out,heads)
    latency={}
    obs=torch.from_numpy(world(53000,FAMILIES[0],'eval').observe())[None,None,:].repeat(1,CONTEXT,1)
    for name,net in heads.items():
        with torch.inference_mode():
            for _ in range(20):net(obs)
            stamps=[]
            for _ in range(256):
                t0=time.perf_counter_ns();net(obs);stamps.append((time.perf_counter_ns()-t0)/1e6)
        stamps.sort();latency[name]={'median_ms':round(statistics.median(stamps),4),'p95_ms':round(stamps[243],4),'measurement':'local CPU batch1 16-state history forward only; host and raster excluded'}
    report={'protocol':PROTOCOL,'synthetic_only':True,'training':training,'heldout_eval_worlds':len(EVAL)*3,
            'rollouts':len(rows),'results':summaries,
            'qualification_limit':'Local replay-valid synthetic only; not native, not exact-repo-HEAD until hosted Actions execution.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    report['manifest_entries']=manifest(out)
    print(json.dumps({'runtime_only':{'latency':latency,'evaluation_seconds':eval_seconds,'total_seconds':round(time.perf_counter()-t,2)},'evidence':report},indent=2,sort_keys=True))

if __name__=='__main__':main()
