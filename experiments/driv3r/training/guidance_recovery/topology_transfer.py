"""C8: training/evaluating correction transfer beyond C7's straight/fork templates.

Entirely synthetic external game-like World; no CyberSand game edits, native Undo,
new shared observation contract, or model inference against resident worms.
The learned residual can affect BACKTRACK only, with host cue+goal-change gating.
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

from experiment import ARMS, N, World, sha
from adversarial_cues import load_prior, source_hash
from personality_gate import BacktrackOnlyGate

SCHEMA = 'guidance-recovery-topology-transfer/1'
TRAIN_FAMILIES = ('hairpin', 'staircase')
OOD_FAMILIES = ('hook', 'zigzag')
ALL_FAMILIES = TRAIN_FAMILIES + OOD_FAMILIES

# Every path is orthogonally adjacent. Last point is initial cursor position.
# The geometric shapes are structurally different from C6/C7's straight trail.
TRAILS = {
 'hairpin': ((3,5),(4,5),(5,5),(5,6),(5,7),(6,7),(7,7)),
 'staircase': ((3,4),(4,4),(4,5),(5,5),(5,6),(6,6),(6,7),(7,7)),
 'hook': ((3,6),(4,6),(5,6),(5,7),(5,8),(6,8),(7,8),(7,7)),
 'zigzag': ((3,3),(4,3),(4,4),(5,4),(5,5),(6,5),(6,6),(7,6),(7,7)),
}
EXITS = {
 'hairpin': ((8,7),(9,7),(10,7),(10,6)),
 'staircase': ((8,7),(8,8),(9,8),(10,8)),
 'hook': ((8,7),(9,7),(9,6),(10,6)),
 'zigzag': ((8,7),(8,8),(9,8),(9,9)),
}
SIDE = {
 'hairpin': ((7,6),(8,6),(9,6)),
 'staircase': ((7,8),(7,9),(8,9)),
 'hook': ((7,6),(7,5),(8,5)),
 'zigzag': ((7,8),(6,8),(6,9)),
}


def transform(p, rotation, mirror, shift=(0,0)):
    x,y=p
    if mirror:x=14-x
    for _ in range(rotation % 4):x,y=14-y,x
    return x+shift[0],y+shift[1]


def build(seed, personality, family, rotation, mirror, outcome, distractor=True):
    """Construct a material-protected, navigable host world from a real trail.

    For outcome=reverse, new goal lies on the existing painted history.
    For outcome=ahead, goal is down an open exit. Both options are host-legal.
    Misleading cue means same geometry and objective as uncued, but cue asserted.
    """
    if family not in ALL_FAMILIES:raise ValueError('unknown family')
    if outcome not in ('reverse','ahead'):raise ValueError('unknown outcome')
    path=TRAILS[family];exit_path=EXITS[family]
    # Controlled topology: legal corridors plus one wrong-looking side branch.
    available=set(path)|set(exit_path)
    if distractor:available|=set(SIDE[family])
    # A small pocket varies geometry while preserving the first-action menu.
    if seed % 3==0:
        extra=(11,6) if family in ('hairpin','hook') else (10,9)
        available.add(extra)
    w=World(seed,'islands',personality)
    allowed={transform(p,rotation,mirror) for p in available}
    w.protected={(x,y) for x in range(N) for y in range(N)} - allowed
    tr=lambda p:transform(p,rotation,mirror)
    w.history=[tr(p) for p in path[:-1]]
    w.pos=tr(path[-1]);w.inked=set(w.history)|{w.pos}
    w.last_dir=(w.pos[0]-w.history[-1][0],w.pos[1]-w.history[-1][1])
    w.goal=tr(path[0] if outcome=='reverse' else exit_path[-1])
    w.goal_change=True;w.goals_seen=2;w.steps=len(path)+2;w.stuck=2
    w.previous_distance=w.distance()
    assert w.history and w.allowed(w.history[-1]) and w.allowed(tr(exit_path[0]))
    assert (abs(exit_path[0][0]-path[-1][0])+abs(exit_path[0][1]-path[-1][1]))==1
    return w


def get_c7(base, c4, c7):
    old=BacktrackOnlyGate(load_prior(base,c4))
    old.gate.load_state_dict(load_file(str(c7)))
    return old.eval()


class TransferGate(nn.Module):
    """One learned scalar applied only to legal host BACKTRACK candidate.

    Frozen C7 remains exact when no cue or no goal switch. Other candidate
    scores are bit-identical even with cue: corrective influence is bounded.
    """
    def __init__(self, prior, width=16):
        super().__init__()
        self.prior=prior
        for p in self.prior.parameters():p.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(25,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def forward(self,x):
        baseline=self.prior(x)
        active=x[...,2]*x[...,18]*x[...,23]
        return baseline + active*self.delta(x).squeeze(-1)


def observe(w,arm):
    cs=w.candidates(arm)
    x=np.asarray([c.features for c in cs],np.float32)
    assert x.shape==(10,25)
    return cs,x


def train_data(n=160,seed=620033):
    x=[];y=[];sources=[]
    for i in range(n):
        s=seed+i*1709;persona=1 if i%2 else -1
        family=TRAIN_FAMILIES[(i//2)%2]
        rot=(i//4)%4;mirror=bool((i//16)%2)
        for outcome in ('reverse','ahead'):
            w=build(s+(0 if outcome=='reverse' else 1000000),persona,family,rot,mirror,outcome)
            un=copy.deepcopy(w); cued=copy.deepcopy(w);cued.issue_recovery_cue(8)
            for arm in ARMS:
                no,xno=observe(un,arm);cs,xi=observe(cued,arm)
                assert np.array_equal(xi[:, :16],xno[:, :16])
                assert tuple((c.action,c.target,c.legal) for c in cs)==tuple((c.action,c.target,c.legal) for c in no)
                assert cs[8].action[0]=='BACKTRACK' and cs[8].legal
                assert xi[8,18]==1 and xi[8,23]==1
                x.append(xi);y.append(1 if outcome=='reverse' else 0)
                sources.append({'seed':w.seed,'family':family,'outcome':outcome,'arm':arm,
                  'source_sha256':source_hash(w),
                  'material_sha256':sha(sorted(w.protected)),
                  'menu_sha256':sha([(c.action,c.target,c.legal) for c in cs]),
                  'observation_sha256':hashlib.sha256(xi.tobytes()).hexdigest()})
    return np.asarray(x,np.float32), np.asarray(y,np.float32),sources


def optimize(model,X,Y,epochs=12,seed=82111,lr=.0015):
    torch.manual_seed(seed);torch.set_num_threads(1)
    xt=torch.from_numpy(X);yt=torch.from_numpy(Y)
    assert xt.shape[1:]==(10,25)
    opt=torch.optim.AdamW(model.delta.parameters(),lr=lr,weight_decay=.004)
    rng=np.random.default_rng(seed);losses=[];steps=0
    for _ in range(epochs):
        order=rng.permutation(len(xt));total=0.
        for start in range(0,len(order),96):
            ix=order[start:start+96]
            logits=model(xt[ix]).masked_fill(xt[ix,:,4]<.5,-1e5)
            contrast=logits[:,8]-torch.logsumexp(torch.cat((logits[:,:8],logits[:,9:]),dim=1),dim=1)
            ce=nn.functional.binary_cross_entropy_with_logits(contrast,yt[ix])
            penalty=.002*model.delta(xt[ix,8,:]).square().mean()
            loss=ce+penalty
            loss.backward();torch.nn.utils.clip_grad_norm_(model.delta.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            steps+=1;total+=float(loss.detach())*len(ix)
        losses.append(round(total/len(xt),6))
    model.eval();return model,steps,losses


def greedy(model,w,arm):
    cs,x=observe(w,arm)
    with torch.no_grad():score=model(torch.from_numpy(x)).numpy()
    illegal_top1=not cs[int(score.argmax())].legal
    score=np.where([c.legal for c in cs],score,-1e7)
    return cs[int(score.argmax())],illegal_top1,x


def first_choices(model,n=48,seed=930007):
    rows=[]
    for fam in ALL_FAMILIES:
        for i in range(n):
            s=seed+i*1999+ALL_FAMILIES.index(fam)*300000
            p=1 if i%2 else -1;rot=i%4;mirror=bool((i//4)%2)
            for outcome in ('reverse','ahead'):
                w=build(s+(1000000 if outcome=='ahead' else 0),p,fam,rot,mirror,outcome)
                for arm in ARMS:
                    for cue in (False,True):
                        view=copy.deepcopy(w)
                        if cue:view.issue_recovery_cue(8)
                        cs,x=observe(view,arm)
                        action,raw,_=greedy(model,view,arm)
                        answer='BACKTRACK' if outcome=='reverse' else 'INK'
                        correct=action.action[0]==answer and (answer=='BACKTRACK' or action.target in
                                  {transform(q,rot,mirror) for q in (EXITS[fam][0],SIDE[fam][0])})
                        applied=view.apply(action)
                        assert applied and view.invalid_attempts==0
                        assert sha(sorted(view.protected))==sha(sorted(w.protected))
                        assert w.inked<=view.inked
                        rows.append({'family':fam,'seed':w.seed,'arm':arm,'personality':p,
                          'rotation':rot,'mirror':mirror,'outcome':outcome,'cue':cue,
                          'source_sha256':source_hash(w),
                          'material_sha256':sha(sorted(w.protected)),
                          'menu_sha256':sha([(c.action,c.target,c.legal) for c in cs]),
                          'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                          'action':action.action,'target':action.target,'correct':bool(correct),
                          'raw_illegal_top1':bool(raw),'host_rejected':not applied})
    return rows


def results(rows):
    stats={}
    for fam in ALL_FAMILIES:
      for outcome in ('reverse','ahead'):
       for arm in ARMS:
        for cue in (False,True):
         sub=[r for r in rows if r['family']==fam and r['outcome']==outcome and r['arm']==arm and r['cue']==cue]
         stats[f'{fam}/{outcome}/{arm:.2f}/cue={int(cue)}']={
           'correct':sum(z['correct'] for z in sub),'total':len(sub),
           'illegal_raw_top1':sum(z['raw_illegal_top1'] for z in sub),
           'host_rejected':sum(z['host_rejected'] for z in sub)}
    return stats


def noninterference(old,new,seeds=12):
    baseline_equal=0;other_exact=0
    with torch.no_grad():
        for i in range(seeds):
            for fam in ALL_FAMILIES:
                for outcome in ('reverse','ahead'):
                    w=build(660001+i*719,i%2*2-1,fam,i%4,bool(i%2),outcome)
                    for arm in ARMS:
                        for cue,change in ((False,True),(True,False),(False,False)):
                            v=copy.deepcopy(w)
                            if cue:v.issue_recovery_cue(8)
                            v.goal_change=change
                            _,x=observe(v,arm);t=torch.from_numpy(x)
                            if not torch.equal(old(t),new(t)):
                                raise AssertionError('noninterference failure')
                            baseline_equal+=1
                        v=copy.deepcopy(w);v.issue_recovery_cue(8)
                        _,x=observe(v,arm);t=torch.from_numpy(x)
                        a=old(t);b=new(t)
                        for ix in list(range(8))+[9]:
                            if a[ix]!=b[ix]:raise AssertionError('draw preference changed')
                            other_exact+=1
    return {'inactive_exact':baseline_equal,'active_non_backtrack_exact':other_exact}


def stochastic(models, n=18, seed=991901, turns=24, temperature=.8):
    """Matched-source, matched-RNG host-executed trajectories; eventual divergence disclosed."""
    rows=[]
    for family in ALL_FAMILIES:
      for i in range(n):
        s=seed+i*1601+ALL_FAMILIES.index(family)*300000
        rotation=i%4;mirror=bool((i//4)%2)
        for person in (-1,1):
         for arm in ARMS:
          for outcome in ('reverse','ahead'):
           for cue in (False,True):
            for name,model in models.items():
             w=build(s+(1000000 if outcome=='ahead' else 0),person,family,rotation,mirror,outcome)
             original=source_hash(w);mat_sha=sha(sorted(w.protected));ink_before=set(w.inked)
             if cue:w.issue_recovery_cue(8)
             rng=np.random.default_rng(s+int(arm*100)*43+int(cue)*941+(1000000 if outcome=='ahead' else 0))
             moves=[];left=right=0;unwinds=0
             for t in range(turns):
                cs,x=observe(w,arm)
                with torch.no_grad():score=model(torch.from_numpy(x)).numpy().astype(np.float64)
                score=np.where([c.legal for c in cs],score,-1e8)
                score=np.exp((score-score.max())/temperature);score/=score.sum()
                idx=int(rng.choice(len(cs),p=score));choice=cs[idx]
                prev=w.last_dir;ink0=set(w.inked)
                ok=w.apply(choice)
                assert ok and w.invalid_attempts==0
                if choice.action[0] in ('INK','LIFT'):
                    signed=prev[0]*choice.action[2]-prev[1]*choice.action[1]
                    left+=int(signed>0);right+=int(signed<0)
                unwinds+=int(choice.action[0]=='BACKTRACK')
                moves.append({'action':choice.action,'target':choice.target,'cursor':w.pos,
                  'ink_delta':len(w.inked-ink0),'history_depth':len(w.history)})
                if w.stopped or w.distance()==0:break
             assert sha(sorted(w.protected))==mat_sha
             assert ink_before<=w.inked
             rows.append({'model':name,'family':family,'seed':w.seed,'personality':person,
                 'source_sha256':original,'material_sha256':mat_sha,'rotation':rotation,'mirror':mirror,
                 'arm':arm,'cue':cue,'outcome':outcome,
                 'reached':w.distance()==0,'end_distance':w.distance(),'steps':len(moves),
                 'left':left,'right':right,'unwinds':unwinds,'ink_added':len(w.inked-ink_before),
                 'host_rejected':w.invalid_attempts,'trace':moves})
    return rows


def summarize_stochastic(rows):
    result={}
    for family in ALL_FAMILIES:
      for outcome in ('reverse','ahead'):
       for arm in ARMS:
        for cue in (False,True):
         stats={}
         for model in ('C7','C8'):
            subset=[r for r in rows if r['model']==model and r['family']==family and
                    r['outcome']==outcome and r['arm']==arm and r['cue']==cue]
            if not subset:continue
            turns=sum(r['left']+r['right'] for r in subset)
            style=(sum(r['left']-r['right'] for r in subset)/turns) if turns else 0
            persona_bias={}
            for persona in (-1,1):
                ss=[r for r in subset if r['personality']==persona]
                trn=sum(z['left']+z['right'] for z in ss)
                persona_bias[persona]=(sum(z['left']-z['right'] for z in ss)/trn) if trn else 0.
            stats[model]={'n':len(subset),'reached':sum(r['reached'] for r in subset),
               'mean_end_distance':round(np.mean([r['end_distance'] for r in subset]),4),
               'mean_ink_added':round(np.mean([r['ink_added'] for r in subset]),4),
               'turn_bias':round(style,6),
               'personality_gap':round(persona_bias[1]-persona_bias[-1],6),'host_rejected':sum(r['host_rejected'] for r in subset)}
         result[f'{family}/{outcome}/{arm:.2f}/cue={int(cue)}']=stats
    return result


def run(base, c4, c7, out, train_n=160, heldout_n=48, epochs=12, rollout_n=18):
    torch.set_num_threads(1)
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    old=get_c7(base,c4,c7)
    X,Y,sources=train_data(n=train_n)
    before=first_choices(old,n=heldout_n)
    torch.manual_seed(15771)
    new=TransferGate(get_c7(base,c4,c7))
    new,steps,losses=optimize(new,X,Y,epochs=epochs)
    after=first_choices(new,n=heldout_n)
    intact=noninterference(old,new)
    trajectories=stochastic({'C7':old,'C8':new},n=rollout_n)
    checkpoint=path/'guidance_recovery_topology_transfer.safetensors'
    provenance={'schema':SCHEMA,'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),
      'c4_sha256':hashlib.sha256(Path(c4).read_bytes()).hexdigest(),
      'c7_sha256':hashlib.sha256(Path(c7).read_bytes()).hexdigest()}
    save_file({k:v.detach().cpu().contiguous() for k,v in new.delta.state_dict().items()},str(checkpoint),
      metadata={'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))})
    metrics={'schema':SCHEMA,'synthetic_only':True,'training_families':TRAIN_FAMILIES,
      'ood_families':OOD_FAMILIES,'train_groups':train_n,'train_cases':len(X),
      'optimizer_steps':steps,'losses':losses,'training_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
      'train_sources_sha256':sha(sources),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      'provenance':provenance,'noninterference':intact,
      'before_first':results(before),'after_first':results(after),
      'before_first_sha256':sha(before),'after_first_sha256':sha(after),
      'stochastic':summarize_stochastic(trajectories),'stochastic_sha256':sha(trajectories),
      'raw_illegal_top1_before':sum(r['raw_illegal_top1'] for r in before),
      'raw_illegal_top1_after':sum(r['raw_illegal_top1'] for r in after),
      'host_rejected':sum(r['host_rejected'] for r in trajectories)}
    for name,rows in (('before_first',before),('after_first',after),('stochastic',trajectories)):
        (path/(name+'.json')).write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (path/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    return metrics


def main():
    p=argparse.ArgumentParser()
    for key in ('base','c4','c7','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--train-n',type=int,default=160)
    p.add_argument('--heldout-n',type=int,default=48)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--rollout-n',type=int,default=18)
    a=p.parse_args()
    m=run(a.base,a.c4,a.c7,a.out,a.train_n,a.heldout_n,a.epochs,a.rollout_n)
    print(json.dumps({'steps':m['optimizer_steps'],'checkpoint_sha256':m['checkpoint_sha256'],
      'train_loss_initial':m['losses'][0],'train_loss_final':m['losses'][-1],
      'noninterference':m['noninterference'],'host_rejected':m['host_rejected']},sort_keys=True))

if __name__=='__main__':main()
