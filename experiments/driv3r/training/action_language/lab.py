"""GAL-001: external, model-side CyberSand-like action-language experiment.

Synthetic 48x48 host, not Godot, W0, model-resident SLM, or a native receipt.
All coordinates are computed by the host from finite action IDs. No game writes.
Source reference: CyberSand #627 BODY-V2 / #611 frozen Q8 pilot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import zlib

W = H = 48
START = (8, 24)
INK = 58
DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
ARMS = ("cell", "sweep", "lift", "retrace", "anchor", "jog")
DECISION_CAP = 96  # synthetic only: NOT native admission
CURSOR_CAP = 384
INK_CAP = 128
FEATURES = ("bias", "ink", "span", "straight", "left", "right", "reverse",
            "open", "novelty", "persist", "near_ink", "endgame", "lift", "jog",
            "retrace", "anchor", "stop", "already_visited")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def png(path, world, scale=4):
    # Static generated material PNG, NEVER a native screenshot.
    palette = {0: (17, 23, 30), 3: (95, 101, 119), 5: (169, 66, 55), INK: (237, 192, 92)}
    raw = bytearray()
    for y in range(H):
        row = b''.join(bytes(palette[v]) * scale for v in world.material[y*W:(y+1)*W])
        for _ in range(scale):
            raw.extend(b'\x00' + row)
    def chunk(t, data):
        return struct.pack('>I', len(data)) + t + data + struct.pack('>I', zlib.crc32(t + data) & 0xffffffff)
    pixels = chunk(b'IHDR', struct.pack('>IIBBBBB', W*scale,H*scale,8,2,0,0,0))
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + pixels + chunk(b'IDAT',zlib.compress(raw,9)) + chunk(b'IEND',b''))


def world_for(seed, family):
    """Deterministic host-owned materials/protection. Never a CyberSand native World."""
    r = random.Random(f'{family}:{seed}')
    material = [0]*(W*H)
    protected = set()
    def put(x,y,value):
        if 0 <= x < W and 0 <= y < H and (x,y) != START:
            material[y*W+x] = value
            if value == 5: protected.add((x,y))
    for x in range(W):
        put(x,0,5);put(x,H-1,5)
    for y in range(H):
        put(0,y,5);put(W-1,y,5)
    if family == 'pillars':
        for x in (15,23,31,39):
            for y in range(5,43):
                if (y//7) % 3 != 1:
                    put(x,y,3)
    elif family == 'gates':
        for x in (14,24,34):
            opening = r.randrange(10,37)
            for y in range(4,44):
                if abs(y-opening)>3: put(x,y,5 if y%9 == 0 else 3)
    elif family == 'scaffold':
        for y in (11,19,29,37):
            for x in range(10,44):
                if x%13 not in (3,4,5): put(x,y,3)
    elif family == 'scatter':
        for _ in range(90):
            x,y = r.randrange(3,45),r.randrange(3,45)
            if abs(x-START[0])+abs(y-START[1])>4:
                put(x,y,5 if r.random()<.25 else 3)
    elif family != 'empty':
        raise ValueError(f'unknown family {family}')
    return material, protected


class HostRefusal(ValueError):
    pass


class World:
    def __init__(self, seed, family, arm, *, lease=None):
        if arm not in ARMS: raise HostRefusal('unknown action arm')
        self.seed, self.family, self.arm = seed, family, arm
        self.lease = lease or f'lease-{seed}-{family}-{arm}'
        self.material, self.protected = world_for(seed,family)
        self.original = self.material.copy()
        self.cursor, self.heading = START, 'E'
        self.history = [START]
        self.own = set(); self.visits = {START: 1}; self.anchors=[]
        self.decisions = self.logical_moves = 0
        self.stopped = False
        self.trace = []
        self.turns = []; self.streak = 0

    def idx(self, p): return p[1]*W+p[0]
    def inside(self,p): return 0 <= p[0] < W and 0 <= p[1] < H
    def clear(self,p): return self.inside(p) and self.material[self.idx(p)] == 0 and p not in self.protected
    def liftable(self,p): return self.inside(p) and p not in self.protected and (self.clear(p) or p in self.own)
    def _path(self, dirs):
        cur=self.cursor; result=[]
        for d in dirs:
            dx,dy=DIRS[d];cur=(cur[0]+dx,cur[1]+dy);result.append(cur)
        return tuple(result)

    def legal(self):
        if self.stopped: return {}
        actions={'STOP': ()}
        if self.decisions >= DECISION_CAP: return actions
        remaining=CURSOR_CAP-self.logical_moves
        if remaining > 0:
            maxspan=1 if self.arm=='cell' else 4
            for d in DIRS:
                prefix=[]
                for n in range(1,min(maxspan,remaining)+1):
                    step=self._path([d]*n)[-1]
                    if not self.clear(step): break
                    prefix.append(step)
                    if len(self.own)+n <= INK_CAP:
                        actions[f'INK_{d}_{n}']=tuple(prefix)
                if self.arm in ('lift','retrace','anchor','jog'):
                    lifted=[]
                    for n in range(1,min(maxspan,remaining)+1):
                        step=self._path([d]*n)[-1]
                        if not self.liftable(step): break
                        lifted.append(step)
                        actions[f'LIFT_{d}_{n}']=tuple(lifted)
            if self.arm == 'jog' and remaining >= 2 and len(self.own)+2 <= INK_CAP:
                for a in DIRS:
                    for b in DIRS:
                        if a == b or (DIRS[a][0]+DIRS[b][0] == 0 and DIRS[a][1]+DIRS[b][1] == 0):
                            continue
                        path=self._path([a,b])
                        if all(self.clear(q) for q in path): actions[f'JOG_{a}_{b}']=path
            if self.arm in ('retrace','anchor','jog') and len(self.history)>1:
                prior=self.history[-2]
                if sum(abs(u-v) for u,v in zip(prior,self.cursor))==1 and self.liftable(prior):
                    actions['RETRACE_1']=(prior,)
        if self.arm in ('anchor','jog') and len(self.anchors)<4 and self.cursor not in self.anchors:
            actions['SAVE_ANCHOR']=()
        return dict(sorted(actions.items()))

    def offer(self):
        actions=self.legal()
        token=sha({'lease':self.lease,'seed':self.seed,'family':self.family,
                   'arm':self.arm,'cursor':self.cursor,'heading':self.heading,
                   'material_sha':sha(self.material),'decisions':self.decisions,
                   'moves':self.logical_moves,'actions':actions})
        return actions,token

    def features(self, name, path):
        head=name.split('_')
        action=head[0]
        d = head[1] if action in ('INK','LIFT','JOG') else self.heading
        x,y=self.cursor
        if getattr(self,'_feat_turn',None) != self.decisions:
            self._feat_turn=self.decisions
            self._feat_near=sum(1 for px,py in self.own if abs(px-x)+abs(py-y)<=3)/24.0
            self._feat_open=sum(1 for h in DIRS if self.clear(self._path([h])[0]))/4
        near=self._feat_near
        openness=self._feat_open
        turn='straight' if d == self.heading else 'reverse' if all(DIRS[d][i]==-DIRS[self.heading][i] for i in (0,1)) else 'right' if DIRS[self.heading][0]*DIRS[d][1]-DIRS[self.heading][1]*DIRS[d][0] > 0 else 'left'
        visit=sum(self.visits.get(c,0) for c in path)/max(1,len(path))
        novelty=sum(c not in self.visits for c in path)/max(1,len(path))
        vals={"bias":1,"ink":int(action in ('INK','JOG')),"span":len(path)/4,
              "straight":int(turn=='straight'),"left":int(turn=='left'),
              "right":int(turn=='right'),"reverse":int(turn=='reverse'),
              "open":openness,"novelty":novelty,"persist":min(self.streak,4)/4 if turn=='straight' else 0,
              "near_ink":near,"endgame":self.decisions/DECISION_CAP,
              "lift":int(action=='LIFT'),"jog":int(action=='JOG'),
              "retrace":int(action=='RETRACE'),"anchor":int(action=='SAVE'),
              "stop":int(action=='STOP'),"already_visited":min(visit,4)/4}
        return tuple(vals[k] for k in FEATURES)

    def apply(self, name, token, *, original_response=None):
        actions, now=self.offer()
        if token != now or name not in actions: raise HostRefusal('stale/foreign/illegal action')
        if original_response is None: original_response=name  # local tiny-policy choice record only
        if not isinstance(original_response,str) or not original_response: raise HostRefusal('missing original output')
        before=sha(self.material); prior=self.cursor; heading=self.heading
        path=actions[name]; parts=name.split('_'); kind=parts[0]
        was_straight=(kind in ('INK','LIFT','JOG') and parts[1]==self.heading)
        if kind in ('INK','JOG'):
            if any(not self.clear(p) for p in path) or len(self.own)+len(path)>INK_CAP:
                raise HostRefusal('unsafe swept write')
            for p in path:
                self.material[self.idx(p)]=INK;self.own.add(p)
        if kind=='RETRACE':
            if not self.history or self.history[-2]!=path[-1]: raise HostRefusal('nonlocal retrace')
            self.history.pop()
        else:self.history.extend(path)
        if path:
            self.cursor=path[-1]
            self.logical_moves+=len(path)
            self.heading=parts[-1] if kind=='JOG' else parts[1] if kind in ('INK','LIFT') else self.heading
            for p in path: self.visits[p]=self.visits.get(p,0)+1
        if kind=='SAVE':self.anchors.append(self.cursor)
        if kind=='STOP':self.stopped=True
        self.streak=self.streak+1 if was_straight else 0
        if kind in ('INK','JOG'):self.turns.append((heading,self.heading))
        self.decisions+=1
        receipt={'evidence_class':'external_synthetic_trained_tiny_policy','native_receipts':0,
                 'resident_slm_processes':0,'lease':self.lease,'turn':self.decisions-1,
                 'action_id':name,'original_response':original_response,'offered_sha256':token,
                 'path':path,'before_material_sha256':before,'after_material_sha256':sha(self.material),
                 'cursor_before':prior,'cursor_after':self.cursor,'logical_moves':self.logical_moves,
                 'new_ink':len(path) if kind in ('INK','JOG') else 0,'STOP':self.stopped}
        self.trace.append(receipt)
        return receipt

    def metrics(self):
        painted=len(self.own)
        paint_calls=sum(t['action_id'].startswith(('INK_','JOG_')) for t in self.trace)
        bends=sum(a!=b for a,b in self.turns)
        return {'decisions':self.decisions,'new_ink':painted,'paint_calls':paint_calls,
                'ink_per_decision':round(painted/max(1,self.decisions),4),
                'bends':bends,'bends_per_32_ink':round(32*bends/max(1,painted),3),
                'logical_moves':self.logical_moves,'STOP_selected':self.stopped,
                'unique_visited':len(self.visits),
                'actions':{k:sum(t['action_id'].startswith(k) for t in self.trace)
                           for k in ('INK','LIFT','JOG','SAVE','RETRACE','STOP')}}


class TinyPolicy:
    """Genuine trainable linear legal-catalog policy; not an SLM or LoRA."""
    def __init__(self):self.weights=[0.0]*len(FEATURES)
    def probabilities(self,vecs,temp=1):
        logits=[sum(x*w for x,w in zip(v,self.weights))/temp for v in vecs]
        m=max(logits); exps=[math.exp(min(30,z-m)) for z in logits];s=sum(exps)
        return [x/s for x in exps]
    def choose(self,world,rng,*,temp=1):
        options,token=world.offer()
        names=list(options);vecs=[world.features(n,options[n]) for n in names]
        probs=self.probabilities(vecs,temp)
        u=rng.random();cum=0;pick=len(names)-1
        for i,p in enumerate(probs):
            cum+=p
            if u<=cum:pick=i;break
        return names[pick],token,names,vecs,probs,pick


def reward(world,receipt,phi):
    # Cultural/visual interest *proxy*, not a measure of actual artistic value.
    kind=receipt['action_id'].split('_')[0]
    ink=receipt['new_ink']
    r=0.36*ink/4 + 0.24*phi[FEATURES.index('novelty')]
    r+=0.14*phi[FEATURES.index('open')]
    if kind in ('INK','JOG'):
        r+=0.09*(phi[FEATURES.index('left')]+phi[FEATURES.index('right')])
        r+=0.12*phi[FEATURES.index('straight')]*(1-phi[FEATURES.index('persist')])
    elif kind=='LIFT': r+=0.03 if world.logical_moves<INK_CAP else -0.06
    elif kind=='RETRACE':r+=0.01
    elif kind=='SAVE':r+=0.015
    elif kind=='STOP':r+=0.18 if world.decisions>=DECISION_CAP or len(world.own)>=INK_CAP else -0.25
    r-=0.11*phi[FEATURES.index('already_visited')]
    return r


def episode(policy,seed,family,arm,*,training=False,lr=.06,entropy=.015,temperature=1):
    w=World(seed,family,arm)
    rng=random.Random(f'actor:{arm}:{seed}:{int(training)}')
    baseline=.25;updates=0
    while not w.stopped and w.decisions<DECISION_CAP:
        action,token,names,vecs,p,pick=policy.choose(w,rng,temp=temperature)
        receipt=w.apply(action,token,original_response=f'tiny-linear-policy:{action}')
        if training:
            r=reward(w,receipt,vecs[pick]);adv=max(-1,min(1,r-baseline));baseline=.95*baseline+.05*r
            expectation=[sum(prob*vec[j] for prob,vec in zip(p,vecs)) for j in range(len(FEATURES))]
            for j in range(len(FEATURES)):
                # REINFORCE gradient plus explicit softmax entropy bonus (via weak uniform pressure).
                entropy_grad=sum((1/len(p)-prob)*vec[j] for prob,vec in zip(p,vecs))
                policy.weights[j]+=lr*(adv*(vecs[pick][j]-expectation[j]) + entropy*entropy_grad)
                policy.weights[j]=max(-6,min(6,policy.weights[j]))
            updates+=1
        if w.decisions>DECISION_CAP:raise AssertionError('host decision ceiling violation')
    return w, updates


def train(arm,episodes_count=120,base_seed=10000):
    policy=TinyPolicy();families=('empty','pillars','gates','scaffold','scatter')
    updates=0
    for i in range(episodes_count):
        _,up=episode(policy,base_seed+i,families[i%len(families)],arm,training=True)
        updates+=up
    return policy,updates


def evaluate(policy,arm,seed_base=90000,count=24,*,out=None):
    results=[];worlds=[];families=('empty','pillars','gates','scaffold','scatter')
    for i in range(count):
        family=families[i%len(families)];w,_=episode(policy,seed_base+i,family,arm)
        assert all(t['native_receipts']==0 for t in w.trace)
        m=w.metrics();m.update({'seed':seed_base+i,'family':family,'trace_sha256':sha(w.trace),
                                'material_sha256':sha(w.material),'synthetic_only':True})
        results.append(m);worlds.append(w)
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        for i in (0,1,2,3,4):
            w=worlds[i];png(out/f'{arm}-{w.family}-{w.seed}.png',w)
            (out/f'{arm}-{w.family}-{w.seed}.json').write_text(json.dumps(w.trace,sort_keys=True,indent=2)+'\n')
    return results


def summarize(rows):
    return {'episodes':len(rows),
            **{k:round(sum(r[k] for r in rows)/len(rows),4) for k in
               ('decisions','new_ink','ink_per_decision','paint_calls','bends','bends_per_32_ink','logical_moves','unique_visited')},
            'early_stop_fraction':round(sum(r['STOP_selected'] and r['decisions']<DECISION_CAP for r in rows)/len(rows),4),
            'trace_digest':sha([r['trace_sha256'] for r in rows])}


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    all_metrics={'schema':'gal-001-external-synthetic/1','not_native':True,
       'not_synthetic_W0':True,'not_SLM':True,'not_native_receipt':True,
       'training_seed_base':10000,'heldout_seed_base':90000,'heldout_overlap':False,
       'train_episodes_per_arm':args.train,'heldout_worlds_per_arm':args.eval,
       'arms':{},'checks':{'action_menus_host_owned':True,'source_scope':'model_side_only'}}
    for arm in ARMS:
        policy,updates=train(arm,args.train)
        rows=evaluate(policy,arm,count=args.eval,out=out/'gallery')
        control=evaluate(TinyPolicy(),arm,count=args.eval)
        checkpoint={'schema':'gal-001-plain-linear/1','arm':arm,'feature_names':FEATURES,'weights':policy.weights,
             'updates':updates,'training_seed_base':10000,'heldout_seed_base':90000}
        checkpath=out/f'{arm}.checkpoint.json';checkpath.write_text(json.dumps(checkpoint,sort_keys=True,indent=2)+'\n')
        rowpath=out/f'{arm}.heldout.json';rowpath.write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
        all_metrics['arms'][arm]={**summarize(rows),'untrained_control':summarize(control),'updates':updates,
             'checkpoint_sha256':hashlib.sha256(checkpath.read_bytes()).hexdigest(),
             'row_evidence_sha256':hashlib.sha256(rowpath.read_bytes()).hexdigest()}
    (out/'metrics.json').write_text(json.dumps(all_metrics,sort_keys=True,indent=2)+'\n')
    print(json.dumps(all_metrics,indent=2,sort_keys=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',type=int,default=120)
    parser.add_argument('--eval',type=int,default=24)
    parser.add_argument('--output',default='gal001-run')
    args=parser.parse_args()
    if not 1<=args.train<=1000 or not 5<=args.eval<=100: parser.error('bounded train/eval required')
    run(args)
