"""C9: offline bounded host-feedback persistence for CyberSand-like worms.

Optional 26th research-only input is an explicitly host-issued recovery-context
TTL, not a material permission or a changed shared/native observation schema.
Frozen C8 remains exact unless host cue and that context are both present.
Only BACKTRACK score can move; real material Undo is never implemented.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from experiment import ARMS, sha
from adversarial_cues import source_hash
from topology_transfer import ALL_FAMILIES, TRAIN_FAMILIES, OOD_FAMILIES, EXITS, build, get_c7, TransferGate, transform

SCHEMA='guidance-recovery-persistent-feedback/1'
HOST_CONTEXT_STEPS=9


def load_c8(base,c4,c7,c8):
    policy=TransferGate(get_c7(base,c4,c7))
    policy.delta.load_state_dict(load_file(str(c8)))
    return policy.eval()


class Context:
    """Host-owned, fixed-bounded feedback life; never inferred by model."""
    def __init__(self, world, authorized=True, steps=HOST_CONTEXT_STEPS):
        if not 1 <= steps <= 12: raise ValueError('bounded context must be 1..12')
        self.remaining=steps if authorized and world.goal_change and world.recovery_cue_ttl else 0
        self.initial_goal=tuple(world.goal)
    def active(self,world):
        return bool(self.remaining>0 and world.recovery_cue_ttl>0 and tuple(world.goal)==self.initial_goal
                    and not world.stopped and world.distance()>0)
    def advance(self,world):
        if self.remaining: self.remaining-=1
        if tuple(world.goal)!=self.initial_goal or world.stopped or world.distance()==0:
            self.remaining=0


def observe(world,arm,context=False):
    cs=world.candidates(arm)
    original=np.asarray([c.features for c in cs],np.float32)
    extra=np.full((10,1),1. if context else 0.,dtype=np.float32)
    x=np.concatenate((original,extra),axis=-1)
    assert x.shape==(10,26) and np.array_equal(x[:,:25],original)
    return cs,x


class PersistentGate(nn.Module):
    """Frozen C8 plus cue-and-context-gated BACKTRACK-only correction."""
    def __init__(self,prior,width=20):
        super().__init__()
        self.prior=prior
        for p in self.prior.parameters():p.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(26,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def forward(self,x):
        prior=self.prior(x[...,:25])
        activation=x[...,2]*x[...,23]*x[...,25]
        return prior + activation*self.delta(x).squeeze(-1)


def action(model,w,arm,context=False,rng=None,temperature=.80):
    cs,x=observe(w,arm,context)
    with torch.no_grad(): scores=model(torch.from_numpy(x if isinstance(model,PersistentGate) else x[:,:25])).detach().numpy().astype(np.float64)
    illegal_top1=not cs[int(scores.argmax())].legal
    scores=np.where(np.asarray([c.legal for c in cs],bool),scores,-1e8)
    if rng is None: i=int(scores.argmax())
    else:
        probs=np.exp((scores-scores.max())/temperature); probs/=probs.sum()
        i=int(rng.choice(len(cs),p=probs))
    return cs[i],bool(illegal_top1),x


def state_after(seed,personality,family,rotation,mirror,outcome,k):
    """Actual host-applied legal history, NOT manually edited future features."""
    w=build(seed,personality,family,rotation,mirror,outcome)
    w.issue_recovery_cue(12)
    ctx=Context(w)
    for j in range(k):
        cs=w.candidates(0)
        if outcome=='reverse':
            target=next(c for c in cs if c.action[0]=='BACKTRACK')
        else:
            path=EXITS[family]
            if j>=len(path):raise ValueError('productive path too short')
            dest=transform(path[j],rotation,mirror)
            target=next(c for c in cs if c.action[0]=='INK' and c.target==dest)
        assert target.legal and w.apply(target)
        ctx.advance(w)
    assert w.invalid_attempts==0 and ctx.active(w)
    return w,ctx


def data(n=180,seed=750031,steps=(0,1,2,3,4)):
    xs=[];ys=[];receipt=[]
    for i in range(n):
        family=TRAIN_FAMILIES[(i//2)%len(TRAIN_FAMILIES)]
        person=1 if i%2 else -1;rot=(i//4)%4;mirror=bool((i//16)%2)
        seed0=seed+i*1951
        for outcome in ('reverse','ahead'):
            s=seed0+(1000000 if outcome=='ahead' else 0)
            for k in steps:
                # Real forward exit has only four nodes; at k=4 the goal is
                # reached and feedback naturally expires, so exclude it.
                if outcome=='ahead' and k>=len(EXITS[family]):continue
                w,ctx=state_after(s,person,family,rot,mirror,outcome,k)
                for arm in ARMS:
                    cs,x=observe(w,arm,ctx.active(w))
                    assert cs[8].action[0]=='BACKTRACK' and cs[8].legal
                    assert x[8,23]==1 and x[8,25]==1
                    assert x[8,18]==(1. if k==0 else 0.)
                    xs.append(x);ys.append(int(outcome=='reverse'))
                    receipt.append({'seed':s,'family':family,'outcome':outcome,'k':k,'personality':person,
                      'arm':arm,'world':source_hash(w),'material':sha(sorted(w.protected)),
                      'menu':sha([(c.action,c.target,c.legal) for c in cs]),
                      'obs':hashlib.sha256(x.tobytes()).hexdigest()})
    return np.asarray(xs,np.float32),np.asarray(ys,np.float32),receipt


def optimize(model,X,Y,epochs=16,seed=9923,lr=.0018):
    torch.manual_seed(seed); torch.set_num_threads(1)
    xt=torch.from_numpy(X);yt=torch.from_numpy(Y)
    opt=torch.optim.AdamW(model.delta.parameters(),lr=lr,weight_decay=.005)
    rng=np.random.default_rng(seed)
    losses=[];updates=0
    for _ in range(epochs):
        order=rng.permutation(len(X));loss_sum=0.
        for start in range(0,len(order),128):
            sel=order[start:start+128];batch=xt[sel]
            logits=model(batch).masked_fill(batch[:,:,4]<.5,-1e6)
            contrast=logits[:,8]-torch.logsumexp(torch.cat((logits[:,:8],logits[:,9:]),1),1)
            loss=nn.functional.binary_cross_entropy_with_logits(contrast,yt[sel])
            loss=loss+.0015*model.delta(batch[:,8,:]).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.delta.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            updates+=1;loss_sum+=float(loss.detach())*len(sel)
        losses.append(round(loss_sum/len(X),6))
    return model.eval(),updates,losses


def rollout(models,n=18,seed=1010001,horizon=24,temperature=.8):
    """Source+RNG matched; disclose divergence after differently chosen actions."""
    rows=[]
    for family in ALL_FAMILIES:
      for i in range(n):
        s=seed+i*1607+ALL_FAMILIES.index(family)*300000
        rot=i%4;mir=bool((i//4)%2)
        for persona in (-1,1):
         for arm in ARMS:
          for outcome in ('reverse','ahead'):
           for cue in (False,True):
            for name,model in models.items():
             w=build(s+(1000000 if outcome=='ahead' else 0),persona,family,rot,mir,outcome)
             world_sha=source_hash(w);material_sha=sha(sorted(w.protected))
             ink0=set(w.inked)
             if cue:w.issue_recovery_cue(12)
             ctx=Context(w)
             rng=np.random.default_rng(s+int(arm*100)*43+int(cue)*941+(1000000 if outcome=='ahead' else 0))
             trace=[];left=right=0;backs=0;longest=run=0;raw_invalid=0
             for tick in range(horizon):
                # Use optional context for C9 only; C8 receives unchanged 25 features.
                pending=ctx.active(w) if name=='C9' else False
                selected,raw,x=action(model,w,arm,pending,rng,temperature)
                raw_invalid+=int(raw)
                prev=w.last_dir;old_ink=set(w.inked)
                assert w.apply(selected) and w.invalid_attempts==0
                assert old_ink<=w.inked and material_sha==sha(sorted(w.protected))
                if selected.action[0] in ('INK','LIFT'):
                    turn=prev[0]*selected.action[2]-prev[1]*selected.action[1]
                    left+=int(turn>0);right+=int(turn<0)
                if selected.action[0]=='BACKTRACK':backs+=1;run+=1;longest=max(longest,run)
                else:run=0
                trace.append({'tick':tick,'action':selected.action,'target':selected.target,
                 'cursor':w.pos,'history_depth':len(w.history),'context':bool(pending),
                 'cue_ttl':w.recovery_cue_ttl,'ink_delta':len(w.inked-old_ink),
                 'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
                ctx.advance(w)
                if w.stopped or w.distance()==0:break
             rows.append({'model':name,'family':family,'seed':w.seed,'personality':persona,
                'arm':arm,'outcome':outcome,'cue':cue,'source_sha256':world_sha,
                'material_sha256':material_sha,'final_ink_sha256':sha(sorted(w.inked)),
                'reached':w.distance()==0,'end_distance':w.distance(),'steps':len(trace),
                'backtracks':backs,'longest_backtrack_run':longest,
                'left':left,'right':right,'ink_added':len(w.inked-ink0),
                'raw_illegal_top1':raw_invalid,'host_rejected':w.invalid_attempts,
                'trace':trace})
    return rows


def stats(rows):
    result={}
    for family in ALL_FAMILIES:
      for outcome in ('reverse','ahead'):
       for cue in (False,True):
        for arm in ARMS:
         ss=[z for z in rows if z['family']==family and z['outcome']==outcome and
                     z['cue']==cue and z['arm']==arm]
         entry={}
         for name in sorted({z['model'] for z in ss}):
            select=[z for z in ss if z['model']==name]
            tw=sum(z['left']+z['right'] for z in select)
            bias=lambda p: (sum(z['left']-z['right'] for z in select if z['personality']==p)/
                            max(1,sum(z['left']+z['right'] for z in select if z['personality']==p)))
            entry[name]={'n':len(select),'reached':sum(z['reached'] for z in select),
                'mean_end_distance':round(float(np.mean([z['end_distance'] for z in select])),4),
                'mean_backtrack_run':round(float(np.mean([z['longest_backtrack_run'] for z in select])),4),
                'personality_turn_gap':round(bias(1)-bias(-1),6),
                'host_rejected':sum(z['host_rejected'] for z in select),
                'raw_illegal_top1':sum(z['raw_illegal_top1'] for z in select)}
         result[f'{family}/{outcome}/{arm:.2f}/cue={int(cue)}']=entry
    return result


def check_noninterference(before,after,n=10):
    inactive=active_other=0
    for i in range(n):
     for family in ALL_FAMILIES:
      for outcome in ('reverse','ahead'):
       w=build(900101+i*881,1 if i%2 else -1,family,i%4,bool(i%2),outcome)
       for arm in ARMS:
        for cue,context in ((False,False),(False,True),(True,False)):
            v=copy.deepcopy(w)
            if cue:v.issue_recovery_cue(12)
            _,x=observe(v,arm,context)
            with torch.no_grad():
                b=before(torch.from_numpy(x[:,:25]));a=after(torch.from_numpy(x))
            assert torch.equal(a,b)
            inactive+=1
        v=copy.deepcopy(w);v.issue_recovery_cue(12)
        _,x=observe(v,arm,True)
        with torch.no_grad():a=after(torch.from_numpy(x));b=before(torch.from_numpy(x[:,:25]))
        assert torch.equal(a[:8],b[:8]) and a[9]==b[9]
        active_other+=9
    return {'inactive_identical_cases':inactive,'active_other_scores_identical':active_other}


def run(base,c4,c7,c8,out,n=180,epochs=16,rollout_n=18):
    torch.set_num_threads(1)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    old=load_c8(base,c4,c7,c8)
    torch.manual_seed(4579)
    student=PersistentGate(load_c8(base,c4,c7,c8))
    X,Y,receipts=data(n)
    student,steps,losses=optimize(student,X,Y,epochs)
    intact=check_noninterference(old,student)
    traces=rollout({'C8':old,'C9':student},n=rollout_n)
    ck=folder/'guidance_recovery_persistent_feedback.safetensors'
    provenance={'schema':SCHEMA,'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),
                'c4_sha256':hashlib.sha256(Path(c4).read_bytes()).hexdigest(),
                'c7_sha256':hashlib.sha256(Path(c7).read_bytes()).hexdigest(),
                'c8_sha256':hashlib.sha256(Path(c8).read_bytes()).hexdigest()}
    save_file({k:v.detach().cpu().contiguous() for k,v in student.delta.state_dict().items()},
       str(ck),metadata={'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))})
    summary=stats(traces)
    metrics={'schema':SCHEMA,'synthetic_only':True,'d1_trained':False,'native_material_undo':False,
        'training_cases':len(X),'training_source_groups':n,'optimizer_steps':steps,
        'regularized_loss':losses,'training_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
        'sources_sha256':sha(receipts),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
        'provenance':provenance,'noninterference':intact,'rollout_episodes':len(traces),
        'raw_trajectory_sha256':sha(traces),'summary':summary,
        'host_rejected':sum(z['host_rejected'] for z in traces),
        'raw_illegal_top1':sum(z['raw_illegal_top1'] for z in traces)}
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    (folder/'raw_trajectories.json').write_text(json.dumps(traces,separators=(',',':'),sort_keys=True)+'\n')
    (folder/'source_receipts.json').write_text(json.dumps(receipts,separators=(',',':'),sort_keys=True)+'\n')
    return metrics


def main():
    p=argparse.ArgumentParser()
    for name in ('base','c4','c7','c8','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--n',type=int,default=180)
    p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--rollout-n',type=int,default=18)
    a=p.parse_args()
    m=run(a.base,a.c4,a.c7,a.c8,a.out,a.n,a.epochs,a.rollout_n)
    print(json.dumps({'optimizer_steps':m['optimizer_steps'],'loss_start':m['regularized_loss'][0],
       'loss_end':m['regularized_loss'][-1],'checkpoint':m['checkpoint_sha256'],
       'episodes':m['rollout_episodes'],'host_rejected':m['host_rejected']},sort_keys=True))

if __name__=='__main__': main()
