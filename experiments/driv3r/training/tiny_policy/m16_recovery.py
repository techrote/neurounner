"""M16: self-contained synthetic-only 32x32 recovery/legality experiment.

No native game integration, pretrained weights, external APIs, or engine authority.
Run: python -m experiments.driv3r.training.tiny_policy.m16_recovery --out NEW_DIR
Python 3.11+, numpy, torch, pillow. Seeds fixed; EVAL never enters optimizer.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

PROTOCOL = 'tiny-policy-m16-recovery-ood-v1'
FAMILIES=('Greebles','Pipework','Architecture')
DIRECTIONS=((0,-1),(0,1),(-1,0),(1,0),(0,0))
ACTIONS=('UP','DOWN','LEFT','RIGHT','STOP')
TRAIN=tuple(range(91000,91016)); DEV=tuple(range(92000,92004)); EVAL=tuple(range(93000,93012))
SIZE=32; BUDGET=64; INPUT=3*7*7+4+3+4+2; CONTEXT=8
POLICIES=('oracle','greedy','novelty','untrained','bc-mlp','recovery-mlp','bc-gru','recovery-gru','bc-tabu','recovery-tabu')

def sha(data):return hashlib.sha256(data).hexdigest()
def dump(path,value): path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')

class World:
    def __init__(self,seed,family,split):
        if seed not in {'train':TRAIN,'dev':DEV,'eval':EVAL}[split] or family not in FAMILIES:raise ValueError('split violation')
        self.seed=seed;self.family=family;self.split=split;self.x=self.y=16;self.heading=3;self.t=0
        self.block=np.ones((SIZE,SIZE),np.bool_);self.protect=np.zeros((SIZE,SIZE),np.bool_)
        self.visited=np.zeros((SIZE,SIZE),np.bool_);self.visited[16,16]=True;self.path=[(16,16)]
        rng=random.Random(seed*13+11)
        self.block[2:30,2:30]=False
        if split!='eval':
            # TRAIN/DEV: perforated straight baffles + isolated crates
            for x in (6,11,21,26):
                self.block[4:28,x]=True
                for y in (8+(seed+x)%5,18+(seed+x)%5,25):self.block[y-1:y+2,x]=False
            for k in range(9):
                x,y=rng.randrange(4,27),rng.randrange(4,27)
                if abs(x-16)+abs(y-16)>7:self.block[y:y+2,x:x+2]=True
        else:
            # EVAL: disjoint generators: nested U, alternating rings, offset comb
            variant=(seed-93000)%3
            if variant==0:
                for x in (5,12,20,27):
                    self.block[5:27,x]=True
                    self.block[5,x:min(30,x+6)]=True
                    self.block[10:13,x]=False;self.block[20:23,x]=False
            elif variant==1:
                for low,high in ((5,26),(9,22)):
                    self.block[low,low:high+1]=True;self.block[high,low:high+1]=True
                    self.block[low:high+1,low]=True;self.block[low:high+1,high]=True
                    self.block[low+3:low+7,low]=False;self.block[high,high-8:high-4]=False
            else:
                for k,y in enumerate((5,10,21,26)):
                    self.block[y,4:28]=True
                    gap=5+(seed+k*7)%18;self.block[y,gap:gap+4]=False
                self.block[5:27,7]=True;self.block[5:27,24]=True
                self.block[13:18,7]=False;self.block[20:25,24]=False
        for k in range(7):
            x,y=rng.randrange(4,27),rng.randrange(4,27)
            if abs(x-16)+abs(y-16)>6:self.protect[y:y+2,x:x+2]=True
        self.block[14:19,14:19]=False;self.protect[14:19,14:19]=False
        self.goals=self.project_goals()
        self.goal_idx=0;self.retraces=0

    def project_goals(self):
        if self.family=='Greebles': pts=((23,16),(23,9),(23,16),(8,16),(8,24),(24,24))
        elif self.family=='Pipework': pts=((24,16),(24,7),(7,7),(7,25),(25,25),(25,12))
        else: pts=((25,16),(25,6),(6,6),(6,25),(25,25),(16,16))
        reach={(16,16):0};q=deque([(16,16)])
        while q:
            x,y=q.popleft()
            for dx,dy in DIRECTIONS[:4]:
                xx,yy=x+dx,y+dy
                if 0<=xx<SIZE and 0<=yy<SIZE and (xx,yy) not in reach and not self.block[yy,xx] and not self.protect[yy,xx]:
                    reach[xx,yy]=reach[x,y]+1;q.append((xx,yy))
        return [min(reach,key=lambda pos:(abs(pos[0]-gx)+abs(pos[1]-gy),reach[pos],pos)) for gx,gy in pts]

    def progress(self):
        while self.goal_idx<len(self.goals) and (self.x,self.y)==self.goals[self.goal_idx]:self.goal_idx+=1
    def legal(self):
        if self.t>=BUDGET:return ()
        return tuple(a for a,(dx,dy) in enumerate(DIRECTIONS[:4]) if not self.block[self.y+dy,self.x+dx] and not self.protect[self.y+dy,self.x+dx])+(4,)
    def observe(self):
        plane=np.zeros((3,7,7),np.float32)
        for j in range(7):
            for i in range(7):
                x,y=self.x+i-3,self.y+j-3
                if not (0<=x<SIZE and 0<=y<SIZE):plane[0,j,i]=1
                else:
                    plane[0,j,i]=self.block[y,x];plane[1,j,i]=self.visited[y,x];plane[2,j,i]=self.protect[y,x]
        g=self.goals[self.goal_idx] if self.goal_idx<len(self.goals) else (self.x,self.y)
        extra=np.array([self.t/BUDGET,self.goal_idx/len(self.goals),self.retraces/BUDGET,float(self.goal_idx==len(self.goals)),(g[0]-self.x)/SIZE,(g[1]-self.y)/SIZE],np.float32)
        return np.concatenate((plane.reshape(-1),np.eye(4,dtype=np.float32)[self.heading],np.eye(3,dtype=np.float32)[FAMILIES.index(self.family)],extra))
    def state(self):
        return sha(self.block.tobytes()+self.protect.tobytes()+self.visited.tobytes()+json.dumps([self.x,self.y,self.t,self.heading,self.goal_idx,self.path],separators=(',',':')).encode())
    def apply(self,a):
        if a not in self.legal():raise ValueError('illegal action')
        before=(self.x,self.y);prior=self.state()
        if a!=4:
            dx,dy=DIRECTIONS[a];self.x+=dx;self.y+=dy;self.heading=a
            if self.visited[self.y,self.x]:self.retraces+=1
            self.visited[self.y,self.x]=True;self.path.append((self.x,self.y))
        self.t+=1;self.progress()
        return {'before':list(before),'before_sha256':prior,'action':ACTIONS[a],'after':[self.x,self.y],'state_sha256':self.state()}
    def oracle(self):
        if self.goal_idx==len(self.goals):return 4
        destination=self.goals[self.goal_idx];q=deque([(self.x,self.y)]);first={(self.x,self.y):4}
        while q:
            x,y=q.popleft()
            if (x,y)==destination:return first[x,y]
            for a,(dx,dy) in enumerate(DIRECTIONS[:4]):
                pos=(x+dx,y+dy)
                if pos in first or self.block[pos[1],pos[0]] or self.protect[pos[1],pos[0]]:continue
                first[pos]=a if (x,y)==(self.x,self.y) else first[x,y];q.append(pos)
        return 4
    def greedy(self):
        if self.goal_idx==len(self.goals):return 4
        gx,gy=self.goals[self.goal_idx];choices=[a for a in self.legal() if a!=4]
        return min(choices,key=lambda a:(abs(gx-self.x-DIRECTIONS[a][0])+abs(gy-self.y-DIRECTIONS[a][1]),a!=self.heading,a)) if choices else 4
    def novelty(self):
        choices=[a for a in self.legal() if a!=4]
        return min(choices,key=lambda a:(bool(self.visited[self.y+DIRECTIONS[a][1],self.x+DIRECTIONS[a][0]]),a!=self.heading,a)) if choices else 4
    def branches(self):
        graph={}
        for a,b in zip(self.path,self.path[1:]):graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
        return sum(len(neigh)>=3 for neigh in graph.values())
    def render(self,path):
        a=np.full((SIZE,SIZE,3),(238,235,226),np.uint8)
        a[self.block]=(66,66,68);a[self.protect]=(190,72,82)
        a[self.visited]={'Greebles':(36,133,112),'Pipework':(41,101,195),'Architecture':(184,113,36)}[self.family]
        a[self.y,self.x]=(255,220,70)
        Image.fromarray(a).resize((SIZE*4,SIZE*4),Image.Resampling.NEAREST).save(path)

class MLP(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(INPUT,80),nn.ReLU(),nn.Linear(80,48),nn.ReLU(),nn.Linear(48,5))
    def forward(self,x):return self.net(x)
class GRU(nn.Module):
    def __init__(self):super().__init__();self.rnn=nn.GRU(INPUT,48,batch_first=True);self.head=nn.Linear(48,5)
    def forward(self,x):return self.head(self.rnn(x)[0])

def collect(seeds,split,perturbed):
    episodes=[]
    for seed in seeds:
        for fam in FAMILIES:
            for e in (0,1):
                w=World(seed,fam,split);rng=random.Random(seed*19+e)
                x=[];y=[];mask=[]
                for t in range(56):
                    w.progress();x.append(w.observe());label=w.oracle()
                    legal=w.legal(); assert label in legal
                    y.append(label);mask.append([a in legal for a in range(5)])
                    if perturbed and (e==1 or rng.random()<.35):
                        move=rng.choice([a for a in legal if a!=4]) if rng.random()<.55 else w.greedy()
                    else:move=label
                    w.apply(int(move))
                    if move==4:break
                episodes.append((np.asarray(x,np.float32),np.asarray(y,np.int64),np.asarray(mask,np.bool_)))
    return episodes

def batches(episodes):
    xx=[];yy=[];mm=[];vv=[]
    for x,y,m in episodes:
        for k in range(0,len(y),CONTEXT):
            n=min(CONTEXT,len(y)-k)
            xx.append(np.pad(x[k:k+n],((0,CONTEXT-n),(0,0))))
            yy.append(np.pad(y[k:k+n],(0,CONTEXT-n)))
            mm.append(np.pad(m[k:k+n],((0,CONTEXT-n),(0,0)),constant_values=True))
            vv.append(np.arange(CONTEXT)<n)
    return tuple(torch.from_numpy(np.asarray(a)) for a in (xx,yy,mm,vv))

def optimize(net,episodes,steps,seed):
    X,Y,M,V=batches(episodes);gen=torch.Generator().manual_seed(seed)
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.002)
    losses=[]
    for k in range(steps):
        ix=torch.randint(len(X),(32,),generator=gen)
        logits=net(X[ix]).masked_fill(~M[ix],-1e6)
        loss=(nn.functional.cross_entropy(logits.reshape(-1,5),Y[ix].reshape(-1),reduction='none').reshape(-1,CONTEXT)*V[ix]).sum()/V[ix].sum()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
        if k%40==39:losses.append(round(float(loss.detach()),5))
    net.eval();return {'steps':steps,'samples':sum(len(x[1]) for x in episodes),'losses':losses}

def decide(w,policy,heads,hist):
    if policy in ('oracle','greedy','novelty'):return getattr(w,policy)()
    net=heads['bc-mlp' if policy=='bc-tabu' else 'recovery-mlp' if policy=='recovery-tabu' else policy]
    with torch.inference_mode():
        x=torch.from_numpy(np.stack(hist[-CONTEXT:]))[None]
        logits=net(x)[0,-1].numpy()
    legal=[a for a in w.legal() if a!=4]
    if w.goal_idx==len(w.goals):return 4
    if policy.endswith('-tabu'):
        # Locally computed host-legal rescoring, NOT privileged whole-world BFS.
        # Penalise immediate revisits, but permit backtracking through cul-de-sacs.
        for a in legal:
            dx,dy=DIRECTIONS[a];target=(w.x+dx,w.y+dy)
            logits[a]-=3.5*int(target in w.path[-12:])
            logits[a]-=.65*int(w.visited[target[1],target[0]])
    return max(legal,key=lambda a:(float(logits[a]),-a)) if legal else 4

def rollout(seed,family,policy,heads,out=None,split='eval'):
    w=World(seed,family,split);events=[];hist=[]
    for t in range(BUDGET):
        w.progress();obs=w.observe();hist.append(obs)
        legal=w.legal();a=decide(w,policy,heads,hist);assert a in legal
        event={'legal':[ACTIONS[i] for i in legal],'obs_sha256':sha(obs.tobytes()),**w.apply(int(a))}
        events.append(event)
        if a==4:break
    rec={'protocol':PROTOCOL,'seed':seed,'family':family,'split':split,'policy':policy,'events':events,
         'cells':int(w.visited.sum()),'waypoints':w.goal_idx,'retraces':w.retraces,'branches':w.branches(),'final_sha256':w.state(),'world_sha256':sha(w.block.tobytes()+w.protect.tobytes())}
    if out is not None:w.render(out)
    return rec

def replay_receipts(record):
    w=World(record['seed'],record['family'],record['split'])
    for e in record['events']:
        w.progress();obs=w.observe()
        if [ACTIONS[a] for a in w.legal()]!=e['legal'] or sha(obs.tobytes())!=e['obs_sha256']:raise AssertionError('bad observation/legal receipt')
        got=w.apply(ACTIONS.index(e['action']))
        if any(got[k]!=e[k] for k in got):raise AssertionError('bad transition receipt')
    if w.state()!=record['final_sha256'] or int(w.visited.sum())!=record['cells'] or w.goal_idx!=record['waypoints'] or w.branches()!=record['branches']:raise AssertionError('bad terminal replay')


def main():
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--steps',type=int,default=160);opt=a.parse_args()
    if opt.out.exists():raise FileExistsError('Immutable output already exists')
    torch.set_num_threads(2);torch.manual_seed(1601);np.random.seed(1601)
    (opt.out/'checkpoints').mkdir(parents=True);(opt.out/'rollouts').mkdir()
    bc=collect(TRAIN,'train',False);rec=collect(TRAIN,'train',True);dev=collect(DEV,'dev',True)
    heads={};logs={}
    heads['untrained']=MLP().eval()
    for name,ctor,data in [('bc-mlp',MLP,bc),('recovery-mlp',MLP,bc+rec),('bc-gru',GRU,bc),('recovery-gru',GRU,bc+rec)]:
        net=ctor();logs[name]=optimize(net,data,opt.steps,1601+len(name));heads[name]=net
        path=opt.out/'checkpoints'/f'{name}.pt'
        torch.save({'protocol':PROTOCOL,'name':name,'state':net.state_dict(),'train_seeds':[min(TRAIN),max(TRAIN)],'steps':opt.steps},path)
        logs[name]['checkpoint_sha256']=sha(path.read_bytes())
        DX,DY,DM,DV=batches(dev)
        with torch.inference_mode():
            dev_pred=net(DX).masked_fill(~DM,-1e6).argmax(-1)
            logs[name]['dev_masked_teacher_agreement']=round(float(((dev_pred==DY)&DV).sum()/DV.sum()),5)
        copy=ctor();copy.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state']);copy.eval()
        assert all(torch.equal(p,q) for p,q in zip(net.parameters(),copy.parameters()))
    rows=[]
    for seed in EVAL:
        for fam in FAMILIES:
            for pol in POLICIES:
                stem=f'{seed}-{fam}-{pol}'
                recd=rollout(seed,fam,pol,heads,opt.out/'rollouts'/f'{stem}.png')
                replay_receipts(recd)
                dump(opt.out/'rollouts'/f'{stem}.json',recd)
                rows.append({k:recd[k] for k in ('seed','family','policy','cells','waypoints','retraces','branches')})
    summary={p:{key:round(statistics.mean(r[key] for r in rows if r['policy']==p),4) for key in ('cells','waypoints','retraces','branches')} for p in POLICIES}
    rng=random.Random(1601);comparison={}
    for test,base in [('recovery-mlp','bc-mlp'),('recovery-gru','bc-gru'),('bc-tabu','bc-mlp'),('recovery-tabu','recovery-mlp')]:
        delta=[statistics.mean(r['waypoints'] for r in rows if r['seed']==seed and r['policy']==test)-statistics.mean(r['waypoints'] for r in rows if r['seed']==seed and r['policy']==base) for seed in EVAL]
        bootstrap=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(2000))
        comparison[f'{test}-vs-{base}']={'waypoint_delta':round(statistics.mean(delta),4),'seed_cluster_95':[round(bootstrap[49],4),round(bootstrap[1949],4)]}
    report={'protocol':PROTOCOL,'synthetic_only':True,'constraints':'32x32 single-cell paint, host goal plans; not native CyberSand; trained scratch heads, not pretrained LLMs; held-out OOD geometry disjoint from TRAIN/DEV, but research EVAL is not sealed FINAL.', 'splits':{'train':list(TRAIN),'dev':list(DEV),'eval':list(EVAL)},'training':logs,'train_episodes':len(bc),'recovery_episodes':len(rec),'dev_episodes':len(dev),'rollouts':len(rows),'results':summary,'paired':comparison}
    dump(opt.out/'report.json',report);dump(opt.out/'rows.json',rows)
    for seed in EVAL[:2]:
        tile=Image.new('RGB',(128*len(POLICIES),128*3),(238,235,226))
        for j,fam in enumerate(FAMILIES):
            for i,pol in enumerate(POLICIES):
                with Image.open(opt.out/'rollouts'/f'{seed}-{fam}-{pol}.png') as source:tile.paste(source,(128*i,128*j))
        tile.save(opt.out/f'gallery-{seed}.png')
    manifest={p.relative_to(opt.out).as_posix():sha(p.read_bytes()) for p in opt.out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    dump(opt.out/'manifest.json',manifest)
    print(json.dumps({'trained':list(heads),'updates':4*opt.steps,'rollouts':len(rows),'hashes':len(manifest),'results':summary,'paired':comparison},indent=2))
if __name__=='__main__':main()
