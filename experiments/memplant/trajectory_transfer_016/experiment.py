"""DG016: out-of-family random maze / timed-obstruction neural donor transfer.
External synthetic CyberSand-like host-legal game; no production imports or writes.
DONOR NOTE: signatures are 24 neural output probes, never transplanted KV state.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.memplant.trajectory_transfer_014 import experiment as base

SIDE = 9
ACTIONS = base.ACTIONS
DIRS = base.DIRS
CHANNELS = 12
HELD = (1, 1, 1)
TRAIN_LAWS = [law for law in itertools.product((-1, 1), repeat=3) if law != HELD]
MODES = ("inherited", "wrong_wet", "wrong_goal", "blind", "oracle")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def maze(seed, family="train"):
    """Random 4x4 spanning-tree maze, cycle augmentation, wet clusters and a timed gate.
    Sealed OOD mazes have fewer cycles than the training family.
    Both route alternatives and goal connectivity are checked after gate closure.
    """
    rng = np.random.default_rng(seed)
    assert family in ("train", "iid", "ood")
    loop_prob = .68 if family != "ood" else .27
    for attempt in range(100):
        wall = np.ones((SIDE, SIDE), dtype=bool)
        cells = [(x, y) for y in (1, 3, 5, 7) for x in (1, 3, 5, 7)]
        for x, y in cells:
            wall[y, x] = False
        visited = {cells[0]}
        stack = [cells[0]]
        while stack:
            x, y = stack[-1]
            unvisited = [(nx, ny) for nx, ny in ((x-2,y),(x+2,y),(x,y-2),(x,y+2))
                         if (nx, ny) in cells and (nx, ny) not in visited]
            if not unvisited:
                stack.pop()
                continue
            q = unvisited[int(rng.integers(len(unvisited)))]
            wall[(y+q[1])//2, (x+q[0])//2] = False
            visited.add(q)
            stack.append(q)
        for x, y in cells:
            for nx, ny in ((x+2,y),(x,y+2)):
                if (nx,ny) in cells and rng.random() < loop_prob:
                    wall[(y+ny)//2, (x+nx)//2] = False
        # Select spatially separated objectives and a closing gate on a short route.
        start = cells[int(rng.integers(len(cells)))]
        reachable = [(q,abs(q[0]-start[0])+abs(q[1]-start[1])) for q in cells if q!=start]
        reachable.sort(key=lambda x:x[1],reverse=True)
        goal = reachable[int(rng.integers(min(5,len(reachable))))][0]
        others = [q for q in cells if q not in (start,goal) and abs(q[0]-start[0])+abs(q[1]-start[1])>=4]
        alternate = others[int(rng.integers(len(others)))]
        wet = (rng.random((SIDE, SIDE)) < .34) & ~wall
        # Contiguous wet pockets, with variable advantage for material-sensitive routing.
        sy,sx = int(rng.integers(1,8)),int(rng.integers(1,8))
        for y in range(max(1,sy-2),min(8,sy+3)):
            for x in range(max(1,sx-2),min(8,sx+3)):
                if not wall[y,x] and rng.random()<.74:
                    wet[y,x]=True
        protected = (~wall) & (rng.random((SIDE,SIDE)) < .20)
        for q in (start,goal,alternate):
            protected[q[1],q[0]]=False
        world = dict(wall=wall,wet=wet,protected=protected,
                     start=start,goal=goal,alternate=alternate)
        # Gate chosen on unweighted start/goal routes; only pick non-cut vertices.
        dist = base.distances(world,goal)
        if start not in dist:
            continue
        candidates=[]
        for y in range(1,8):
            for x in range(1,8):
                q=(x,y)
                if wall[y,x] or q in (start,goal,alternate):
                    continue
                if dist.get(q,1e6)>dist[start] or dist.get(q,1e6)<1:
                    continue
                closed=wall.copy();closed[y,x]=True
                probe=dict(world,wall=closed)
                # All traversable cells must remain connected for retrace legality.
                connected = base.distances(probe,start)
                if len(connected)==int((~closed).sum()):
                    candidates.append(q)
        if not candidates:
            continue
        gate=candidates[int(rng.integers(len(candidates)))]
        world["gate"]=gate
        # Train vs OOD differ in both topology and obstruction timing.
        world["close_after"] = int(rng.integers(2,5) if family!="ood" else rng.integers(5,9))
        world["family"] = family
        return world
    raise RuntimeError(f"unable to produce alternate-route maze for seed {seed}, {family}")


class Game(base.Game):
    def __init__(self,world,laws):
        safe=dict(world,wall=world['wall'].copy())
        super().__init__(safe,laws)
        self.elapsed=0
        self.closed=False
        self.retraces=0
        self.closure_step=None
        self.stalled=False

    def legal(self):
        legal=super().legal()
        if len(self.history)>1:
            x,y=self.history[-2]
            legal[5] = not self.world['wall'][y,x]
        return legal

    def observe(self):
        z=np.zeros((CHANNELS,SIDE,SIDE),dtype=np.float32)
        w=self.world
        z[0]=w['wall']
        z[1]=w['wet']
        z[2]=w['protected']
        previous=self.history[-2] if len(self.history)>1 else self.pos
        for ch,q in ((3,self.pos),(4,w['goal']),(5,w['alternate']),
                     (6,w['start']),(7,previous),(9,w['gate'])):
            z[ch,q[1],q[0]]=1.
        for x,y in self.painted:
            z[8,y,x]=1.
        z[10,:,:]=float(self.closed)
        z[11,:,:]=min(1.,max(0.,(w['close_after']-self.elapsed)/8.))
        return z,self.legal()

    def oracle(self):
        action=super().oracle()
        if action is not None and action<4 and len(self.history)>1:
            dx,dy=DIRS[action]
            q=(self.pos[0]+dx,self.pos[1]+dy)
            if q == self.history[-2] and self.legal()[5]:
                return 5
        return action

    def step(self,action):
        mask=self.legal()
        if action is None or not mask[action]:
            self.illegal+=1
            raise ValueError(f'Illegal host action {action}')
        if action==5:
            self.history.pop()
            self.pos=self.history[-1]
            self.visits[self.pos]=self.visits.get(self.pos,0)+1
            x,y=self.pos
            wet=int(self.world['wet'][y,x])
            self.wet_steps+=wet
            self.travel_cost+=1+(6. if self.laws[0]>0 else .05)*wet
            self.retraces+=1
            self.steps+=1
        else:
            super().step(action)
        if action<4 or action==5:
            self.elapsed+=1
        if not self.closed and self.elapsed>=self.world['close_after']:
            x,y=self.world['gate']
            if self.pos!=(x,y):
                self.world['wall'][y,x]=True
                self.closed=True
                self.closure_step=self.steps


class Receiver(nn.Module):
    def __init__(self,conditioned):
        super().__init__()
        self.conditioned=conditioned
        self.net=nn.Sequential(nn.Linear(CHANNELS*SIDE*SIDE + (24 if conditioned else 0),192),
             nn.ReLU(),nn.Linear(192,128),nn.ReLU(),nn.Linear(128,6))
    def forward(self,x,legal,signatures=None):
        z=x.flatten(1)
        if self.conditioned:
            z=torch.cat([z,signatures],dim=-1)
        return self.net(z).masked_fill(~legal,-1e5)


def train_data(seed,bank,worlds):
    obs,legal,signatures,labels=[],[],[],[]
    for i in range(worlds):
        world=maze(seed+i*79,"train")
        for law in TRAIN_LAWS:
            game=Game(world,law)
            sig=base.signatures(bank,"narrow",law)
            for step in range(96):
                x,m=game.observe()
                a=game.oracle()
                if a is None or game.done:
                    break
                assert m[a]
                obs.append(x); legal.append(m); signatures.append(sig);labels.append(a)
                game.step(a)
    return tuple(torch.from_numpy(np.stack(v)) for v in (obs,legal,signatures))+(torch.tensor(labels,dtype=torch.long),)


def fit(seed,data,conditioned,updates):
    torch.manual_seed(seed)
    model=Receiver(conditioned)
    opt=torch.optim.AdamW(model.parameters(),lr=.0018)
    x,legal,signatures,labels=data
    rng=np.random.default_rng(seed+17)
    for _ in range(updates):
        ix=torch.from_numpy(rng.integers(0,len(labels),size=192))
        pred=model(x[ix],legal[ix],signatures[ix] if conditioned else None)
        loss=F.cross_entropy(pred,labels[ix])
        opt.zero_grad();loss.backward();opt.step()
    return model.eval()


def play(world,law,mode,bank,models,horizon):
    game=Game(world,law)
    if mode=="wrong_wet":
        donor=(-law[0],law[1],law[2])
    elif mode=="wrong_goal":
        donor=(law[0],-law[1],law[2])
    else:
        donor=law
    signature=base.signatures(bank,"gated",donor)
    trace=[game.pos]
    actions=[]
    for _ in range(horizon):
        x,legal=game.observe()
        if mode=="oracle":
            action=game.oracle()
        else:
            policy=models["blind" if mode=="blind" else "inherited"]
            with torch.no_grad():
                action=int(policy(torch.from_numpy(x[None]),torch.from_numpy(legal[None]),
                          torch.from_numpy(signature[None]) if mode!="blind" else None).argmax(-1)[0])
        if action is None:
            break
        assert legal[action]
        game.step(action)
        trace.append(game.pos)
        actions.append(ACTIONS[action])
        if game.done:
            break
    return dict(success=int(game.done),cost=round(game.travel_cost,4),steps=game.steps,
                closure=int(game.closed),closure_step=game.closure_step,
                retraces=game.retraces,illegal=game.illegal,wet=game.wet_steps,
                path=trace,actions=actions,painted=len(game.painted))


def summarize(rows):
    result={}
    for mode,vals in rows.items():
        result[mode]=dict(worlds=len(vals),success=sum(v['success'] for v in vals),
           mean_cost_all=round(float(np.mean([v['cost'] for v in vals])),4),
           mean_cost_success=round(float(np.mean([v['cost'] for v in vals if v['success']])),4)
                      if any(v['success'] for v in vals) else None,
           failure_penalized_mean_cost=round(float(np.mean([v['cost'] + (0 if v['success'] else 3*96) for v in vals])),4),
           total_retraces=sum(v['retraces'] for v in vals),
           closures=sum(v['closure'] for v in vals),
           illegal=sum(v['illegal'] for v in vals))
    return result


def evaluate(seed,bank,models,worlds,horizon):
    splits={}
    samples=[]
    for family in ("iid","ood"):
        rows={m:[] for m in MODES}
        comparisons=[]
        donor_changes=0;oracle_differences=0;true_follow=0;cf_follow=0;both_follow=0
        for i in range(worlds):
            w=maze(seed+(i*103)+(0 if family=="iid" else 10000000),family)
            law=HELD if i%2==0 else TRAIN_LAWS[(i//2)%len(TRAIN_LAWS)]
            trials={mode:play(w,law,mode,bank,models,horizon) for mode in MODES}
            for mode in MODES:
                rows[mode].append(trials[mode])
            if trials['inherited']['success'] and trials['wrong_wet']['success'] and trials['oracle']['success']:
                comparisons.append(dict(delta=trials['wrong_wet']['cost']-trials['inherited']['cost'],
                     correct_regret=trials['inherited']['cost']/max(.001,trials['oracle']['cost'])-1,
                     wrong_regret=trials['wrong_wet']['cost']/max(.001,trials['oracle']['cost'])-1))
            true_nav=next((a for a in trials['inherited']['actions'] if a in ACTIONS[:4]),None)
            wrong_nav=next((a for a in trials['wrong_wet']['actions'] if a in ACTIONS[:4]),None)
            donor_changes+=int(true_nav is not None and wrong_nav is not None and true_nav!=wrong_nav)
            # Independent oracle of the actual alternative physics, identical maze/clock.
            cf_oracle=play(w,(-law[0],law[1],law[2]),'oracle',bank,models,horizon)
            actual_oracle_nav=next((a for a in trials['oracle']['actions'] if a in ACTIONS[:4]),None)
            cf_oracle_nav=next((a for a in cf_oracle['actions'] if a in ACTIONS[:4]),None)
            if actual_oracle_nav is not None and cf_oracle_nav is not None and actual_oracle_nav!=cf_oracle_nav:
                oracle_differences+=1
                p=true_nav==actual_oracle_nav; q=wrong_nav==cf_oracle_nav
                true_follow+=int(p);cf_follow+=int(q);both_follow+=int(p and q)
            if i<4:
                samples.append(dict(family=family,seed=seed+i*103+(0 if family=='iid' else 10000000),
                   laws=law,wall=w['wall'].astype(int).tolist(),wet=w['wet'].astype(int).tolist(),
                   gate=w['gate'],close_after=w['close_after'],start=w['start'],goal=w['goal'],
                   alternate=w['alternate'],trials=trials))
        summary=summarize(rows)
        joint= dict(joint_success=len(comparisons),
           mean_wrong_minus_correct_cost=round(float(np.mean([v['delta'] for v in comparisons])),4) if comparisons else None,
           correct_lower_cost=sum(v['delta']>1e-6 for v in comparisons),
           wrong_lower_cost=sum(v['delta']<-1e-6 for v in comparisons),
           equal_cost=sum(abs(v['delta'])<=1e-6 for v in comparisons),
           mean_correct_regret=round(float(np.mean([v['correct_regret'] for v in comparisons])),4) if comparisons else None,
           mean_wrong_regret=round(float(np.mean([v['wrong_regret'] for v in comparisons])),4) if comparisons else None,
           donor_navigation_changes=donor_changes,
           oracle_wet_action_differences=oracle_differences,
           original_oracle_followed=true_follow,
           counterfactual_oracle_followed=cf_follow,
           both_oracles_followed=both_follow)
        splits[family]=dict(trajectories=summary,paired_counterfactual=joint,
           withheld_success={m:sum(v['success'] for j,v in enumerate(rows[m]) if j%2==0)
                              for m in MODES},withheld_worlds=(worlds+1)//2)
    return splits,samples


def svg_gallery(samples,path):
    palette={"inherited":"#2563eb","wrong_wet":"#d43c4a","wrong_goal":"#aa57a3",
             "blind":"#c39329","oracle":"#282e35"}
    output=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="780" viewBox="0 0 1000 780">',
        '<rect width="1000" height="780" fill="white"/>',
        '<text x="12" y="24" font-size="17">DG016 · real trained policy trajectories · IID vs OOD mazes</text>']
    for i,case in enumerate(samples[:6]):
        row=i//3;col=i%3;ox=12+330*col;oy=38+360*row
        output.append(f'<text x="{ox}" y="{oy+13}" font-size="12">{case["family"]} · gate at {case["close_after"]} steps · {case["laws"]}</text>')
        for y in range(SIDE):
            for x in range(SIDE):
                color='#394651' if case['wall'][y][x] else '#abd7eb' if case['wet'][y][x] else '#eee'
                output.append(f'<rect x="{ox+25*x}" y="{oy+20+25*y}" width="24" height="24" fill="{color}" stroke="#ccc"/>')
        gx,gy=case['gate']
        output.append(f'<rect x="{ox+25*gx+4}" y="{oy+24+25*gy}" width="16" height="16" stroke="#ba2727" stroke-width="2" fill="none"/>')
        for mode,tr in case['trials'].items():
            coords=' '.join(f'{ox+25*x+12},{oy+20+25*y+12}' for x,y in tr['path'])
            output.append(f'<polyline points="{coords}" stroke="{palette[mode]}" fill="none" stroke-width="2" opacity=".65"/>')
        output.append(f'<text x="{ox}" y="{oy+262}" font-size="11">'+
                 ' | '.join(f'{m[:2]}:{case["trials"][m]["success"]}' for m in MODES)+'</text>')
    output.append('</svg>')
    path.write_text('\n'.join(output)+'\n')


def selftest():
    for family in ('train','iid','ood'):
        for s in range(20):
            world=maze(503+s*37,family)
            assert world['start']!=world['goal']
            a=world['wall'].copy()
            a[world['gate'][1],world['gate'][0]]=True
            assert world['start'] in base.distances(dict(world,wall=a),world['goal'])
            for law in itertools.product((-1,1),repeat=3):
                game=Game(world,law)
                for t in range(96):
                    obs,legal=game.observe()
                    assert obs.shape==(CHANNELS,SIDE,SIDE)
                    assert np.isfinite(obs).all()
                    action=game.oracle()
                    if action is None:break
                    assert legal[action]
                    game.step(action)
                    if game.done:break
                assert game.done, (family,s,law,world['close_after'],t)
                assert game.closed or game.elapsed < world['close_after'] or game.pos==world['gate']
    print('DG016 SELFTEST PASS: 480 oracle-maze variants, legal RETRACE, completion; some OOD episodes finish before closure')


def run(seed,out,worlds,updates,sealed,horizon,donor_steps):
    torch.set_num_threads(2)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    t=time.monotonic()
    bank={};errors={}
    for kind in ('narrow','gated'):
        for family in range(3):
            for sign in (-1,1):
                d=base.train_donor(seed+1000+100*family+10*(sign+1)+(500 if kind=='gated' else 0),
                                   family,sign,kind=='gated',donor_steps)
                bank[kind,family,sign]=d
                errors[f'{kind}/{family}/{sign}']=round(d['error'],6)
                torch.save(dict(weights=d['model'].state_dict(),private_permutation=d['perm'].tolist(),
                            private_reflection=d['flip'].tolist(),mse=d['error']),
                            out/f'donor-{kind}-{family}-{sign}.pt')
    data=train_data(seed+2000,bank,worlds)
    models={'inherited':fit(seed+3000,data,True,updates),
            'blind':fit(seed+4000,data,False,updates)}
    for name,model in models.items():
        torch.save(model.state_dict(),out/f'receiver-{name}.pt')
    splits,traces=evaluate(seed+100000,bank,models,sealed,horizon)
    for name,s in splits.items():
        assert all(v['illegal']==0 for v in s['trajectories'].values())
        assert s['trajectories']['oracle']['success']==sealed, name
    (out/'trajectories.json').write_text(json.dumps(traces,indent=2)+'\n')
    svg_gallery(traces,out/'gallery.svg')
    report=dict(experiment='DG016',seed=seed,source_sha256=digest(__file__),
       parent_commit='66fc77fdabba020f09cfa1231f9910a974fc852b',
       pytorch=torch.__version__,train_worlds=worlds,training_states=len(data[3]),updates=updates,
       donor_training_steps=donor_steps,sealed_worlds_per_split=sealed,
       horizon=horizon,withheld_law='+++',donor_validation_mse=errors,
       features='12 visible grid channels; no precomputed path distances or oracle potentials',
       split_design='IID looped mazes with 2-4 move gates; OOD sparsely looped mazes with 5-8 move gates',
       source_neural_outputs=True,source_hidden_permutation=True,
       train_test_seed_overlap=0,results=splits,
       caveats='Synthetic Dijkstra imitation; learned donors approximate authored atomic laws; OOD shifts topology and timing together; retrieval is frozen donor outputs rather than weights/KV. No native CyberSand acceptance.',
       duration_seconds=round(time.monotonic()-t,2))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    hashes={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--seed',type=int,default=26011628)
    parser.add_argument('--out',default='/tmp/dg016')
    parser.add_argument('--train-worlds',type=int,default=300)
    parser.add_argument('--updates',type=int,default=800)
    parser.add_argument('--sealed',type=int,default=100)
    parser.add_argument('--horizon',type=int,default=96)
    parser.add_argument('--donor-steps',type=int,default=140)
    args=parser.parse_args()
    if args.selftest:selftest()
    else:run(args.seed,args.out,args.train_worlds,args.updates,args.sealed,args.horizon,args.donor_steps)
