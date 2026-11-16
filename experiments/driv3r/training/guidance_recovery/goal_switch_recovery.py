"""C6 offline goal-switch feedback calibration for CyberSand-like worm policies.

No game code, shared contracts, native Undo, material permissions or live inference.
Experimental adapter freezes the C4 policy and activates only for a simultaneous
host correction cue and explicit goal-change bit in the existing 25-feature menu.
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
from safetensors.torch import save_file
from experiment import ARMS, World, sha
from hard_recovery import rotate, add_supervised
from adversarial_cues import load_prior, source_hash, pair_view
from positive_correction import candidates_fingerprint

SCHEMA = 'guidance-recovery-goal-switch/1'


def world(seed, personality, rotation, mirror, outcome, branch=False, branch_shape="upper"):
    """Painted history into an orthogonal corridor; new goal can reverse or continue.

    Both directions are legal. Cue is advisory and cannot change either menu.
    The lower-side branch topology is excluded from training and held out for transfer.
    """
    if outcome not in ('reverse', 'ahead'): raise ValueError(outcome)
    tr = lambda p: rotate(p, rotation, mirror)
    path = [(3,7),(4,7),(5,7),(6,7),(7,7)]
    available = set(path + [(x,7) for x in range(8,14)])
    available.update((3,y) for y in range(3,8))
    available.update((x,3) for x in range(3,14))
    available.update((13,y) for y in range(3,8))
    if branch:
        if branch_shape not in ('upper','lower'):raise ValueError(branch_shape)
        y=5 if branch_shape=='upper' else 9
        available.update([(7,6 if branch_shape=='upper' else 8),(7,y)])
        available.update((x,y) for x in range(8,14))
        available.update([(13,6),(13,8)])
    w = World(seed, 'islands', personality)
    w.protected = {(x,y) for x in range(15) for y in range(15)} - {tr(p) for p in available}
    w.pos = tr(path[-1]); w.history = [tr(p) for p in path[:-1]]
    w.inked = set(w.history + [w.pos])
    w.last_dir = (tr((8,7))[0]-w.pos[0],tr((8,7))[1]-w.pos[1])
    w.steps = 6; w.stuck = 2
    w.goal = tr((3,3) if outcome=='reverse' else (13,7))
    w.goal_change = True
    w.goals_seen = 2
    w.previous_distance = w.distance()
    assert w.allowed(w.history[-1]) and w.allowed(tr((8,7)))
    return w


class GoalSwitchAdapter(nn.Module):
    """Preserve exact legacy logits unless cue AND changed-goal are both observed."""
    def __init__(self, prior, width=16):
        super().__init__()
        self.prior = prior
        for p in self.prior.parameters(): p.requires_grad_(False)
        self.residual = nn.Sequential(nn.Linear(25,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        old = self.prior(x)
        flag = x[...,18]*x[...,23]
        return old + flag*self.residual(x).squeeze(-1)


def choose(model, w, arm):
    cs = w.candidates(arm)
    x=torch.tensor([c.features for c in cs],dtype=torch.float32)
    with torch.no_grad(): scores=model(x).detach().numpy()
    raw_illegal = not cs[int(np.argmax(scores))].legal
    scores=np.where(np.asarray([c.legal for c in cs],bool),scores,-1e5)
    return cs[int(np.argmax(scores))], bool(raw_illegal), x


def dataset(seed=73091, n=240, mix_branches=False):
    """World-isolated source pairing; training rotations 0/1 only."""
    X=[]; Y=[]; rows=[]
    for i in range(n):
        s=seed+i*1171; p=1 if i%2 else -1
        rot=i%2; mirror=bool((i//2)%2)
        for outcome in ('reverse','ahead'):
            branch=bool(mix_branches and ((i//2)%2))
            w=world(s+(0 if outcome=='reverse' else 500000),p,rot,mirror,outcome,branch=branch)
            for arm in ARMS:
                no,cued=pair_view(w,arm)
                assert source_hash(no)==source_hash(cued)
                assert candidates_fingerprint(no,arm)==candidates_fingerprint(cued,arm)
                xx=np.asarray([c.features for c in cued.candidates(arm)],np.float32)
                ys=[]; add_supervised([],ys,cued,arm,
                   'BACKTRACK' if outcome=='reverse' else 'INK',
                   None if outcome=='reverse' else rotate((8,7),rot,mirror), mass=.95)
                # Preserve two viable drawing choices rather than force a unique
                # stroke through the held-out junction branch.
                if branch and outcome=='ahead':
                    cs=cued.candidates(arm)
                    side=rotate((7,6),rot,mirror)
                    idx=next(j for j,c in enumerate(cs) if c.action[0]=='INK' and c.target==side and c.legal)
                    main=next(j for j,c in enumerate(cs) if c.action[0]=='INK' and c.target==rotate((8,7),rot,mirror) and c.legal)
                    v=ys[0].copy()
                    share=.35 if p==1 else .55
                    v[idx]+=share*v[main];v[main]*=(1-share)
                    ys[0]=v
                X.append(xx);Y.append(ys[0]);rows.append({'seed':w.seed,'branch':branch,'outcome':outcome,
                   'arm':arm,'world_sha256':source_hash(w),
                   'menu_sha256':candidates_fingerprint(w,arm),'observation_sha256':hashlib.sha256(xx.tobytes()).hexdigest()})
    return np.asarray(X,np.float32), np.asarray(Y,np.float32),rows


def train(model, X,Y, epochs=12,seed=4701,lr=.002):
    torch.manual_seed(seed);torch.set_num_threads(1)
    xt=torch.from_numpy(X);yt=torch.from_numpy(Y)
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.residual.parameters(),lr=lr,weight_decay=.003)
    losses=[];steps=0
    for _ in range(epochs):
        order=rng.permutation(len(X));total=0.
        for st in range(0,len(X),96):
            ids=order[st:st+96]
            logits=model(xt[ids]); logits=logits.masked_fill(xt[ids,:,4]<.5,-1e4)
            ce=-(yt[ids]*torch.log_softmax(logits,dim=-1)).sum(-1).mean()
            penalty=.004*(model.residual(xt[ids]).squeeze(-1)**2).mean()
            loss=ce+penalty
            loss.backward();torch.nn.utils.clip_grad_norm_(model.residual.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(ids);steps+=1
        losses.append(round(total/len(X),6))
    model.eval();return model,steps,losses


def evaluate(model, n=40,seed=990031, branch=False, rotation_set=(2,3),branch_shape="upper"):
    rows=[]
    for i in range(n):
        s=seed+i*919;p=-1 if i%2 else 1;rot=rotation_set[i%len(rotation_set)];mir=bool((i//2)%2)
        for outcome in ('reverse','ahead'):
            w=world(s+(0 if outcome=='reverse' else 500000),p,rot,mir,outcome,branch=branch,branch_shape=branch_shape)
            for arm in ARMS:
                for cue in (False,True):
                    a=copy.deepcopy(w)
                    if cue:a.issue_recovery_cue(9)
                    c,raw_bad,x=choose(model,a,arm)
                    wanted='BACKTRACK' if outcome=='reverse' else 'INK'
                    target=rotate((8,7),rot,mir) if outcome=='ahead' else None
                    productive=(c.target==target or (branch and c.target==rotate((7,6 if branch_shape=='upper' else 8),rot,mir)))
                    right=(c.action[0]==wanted and (target is None or productive))
                    accepted=a.apply(c)
                    rows.append({'seed':w.seed,'rotation':rot,'mirror':mir,'branch':branch,
                      'branch_shape':branch_shape if branch else 'none',
                      'outcome':outcome,'arm':arm,'cue':cue,
                      'material_sha256':sha(sorted(w.protected)),'world_sha256':source_hash(w),
                      'menu_sha256':candidates_fingerprint(w,arm),
                      'observation_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                      'chosen_action':c.action,'target':c.target,'correct':bool(right),
                      'raw_illegal_top1':raw_bad,'accepted':bool(accepted),
                      'history_after':list(a.history),'inked_after_sha256':sha(sorted(a.inked))})
    return rows


def summary(rows):
    res={}
    for branch,shape in sorted(set((r['branch'],r['branch_shape']) for r in rows)):
        for outcome in ('reverse','ahead'):
            for arm in ARMS:
                for cue in (False,True):
                    a=[r for r in rows if r['branch']==branch and r['branch_shape']==shape and r['outcome']==outcome and r['arm']==arm and r['cue']==cue]
                    if not a:continue
                    res[f'{"branch_"+shape if branch else "corridor"}/{outcome}/{arm:.2f}/{"cue" if cue else "none"}']={
                       'correct':sum(r['correct'] for r in a),'total':len(a),
                       'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in a),'host_rejected':sum(not r['accepted'] for r in a)}
    return res


def personality_probe(model,seeds=32):
    """Pair distinct personalities on identical geometry, not post-divergence paths."""
    result={}
    for shape in ('corridor','upper','lower'):
        for cue in (False,True):
            for arm in ARMS:
                differed=0
                for i in range(seeds):
                    choices=[]
                    for p in (-1,1):
                        w=world(870000+i*779,p,2+i%2,bool(i%2),'ahead',
                                branch=shape!='corridor',branch_shape=shape if shape!='corridor' else 'upper')
                        if cue:w.issue_recovery_cue(9)
                        c,_,_=choose(model,w,arm);choices.append(c.action)
                    differed+=int(choices[0]!=choices[1])
                result[f'{shape}/{arm:.2f}/{"cue" if cue else "none"}']={
                    'personality_action_divergence':differed,'pairs':seeds}
    return result


def rollout_probe(model,seeds=12,turns=12):
    """Actual synthetic host-applied cursor turns, never material rollback."""
    records=[]
    for shape in ('corridor','upper','lower'):
        for i in range(seeds):
            for arm in ARMS:
                for cue in (False,True):
                    s=781000+i*997;rot=2+i%2;mir=bool(i%2)
                    w=world(s,1 if i%2 else -1,rot,mir,'reverse',
                            branch=shape!='corridor',branch_shape=shape if shape!='corridor' else 'upper')
                    initial_hash=source_hash(w)
                    material_hash=sha(sorted(w.protected))
                    ink0=set(w.inked)
                    if cue:w.issue_recovery_cue(9)
                    trace=[{'pos':w.pos,'action':'START'}];rejected=0
                    for t in range(turns):
                        c,_,_=choose(model,w,arm)
                        ok=w.apply(c);rejected+=int(not ok)
                        trace.append({'pos':w.pos,'action':c.action[0],
                             'target':c.target,'accepted':bool(ok)})
                        if not ok or w.stopped:break
                    assert material_hash==sha(sorted(w.protected))
                    assert ink0<=w.inked
                    expected_junction=rotate((3,7),rot,mir)
                    records.append({'schema':SCHEMA,'branch_shape':shape,'seed':s,'personality':w.personality,
                       'rotation':rot,'mirror':mir,'arm':arm,'cue':cue,
                       'source_sha256':initial_hash,'material_sha256':material_hash,
                       'goal':w.goal,'history_unwound':expected_junction in [z['pos'] for z in trace],
                       'first_four_backtracks':all(z['action']=='BACKTRACK' for z in trace[1:5]),
                       'goal_reached':w.distance()==0,'final_distance':w.distance(),
                       'new_ink_cells':len(w.inked-ink0),'host_rejected':rejected,
                       'trace':trace})
    return records


def rollout_summary(rows):
    summary={}
    for shape in ('corridor','upper','lower'):
        for arm in ARMS:
            for cue in (False,True):
                z=[r for r in rows if r['branch_shape']==shape and r['arm']==arm and r['cue']==cue]
                summary[f'{shape}/{arm:.2f}/{"cue" if cue else "none"}']={
                    'total':len(z),'four_backtracks':sum(r['first_four_backtracks'] for r in z),
                    'junction_reached':sum(r['history_unwound'] for r in z),
                    'goal_reached':sum(r['goal_reached'] for r in z),
                    'mean_final_distance':round(sum(r['final_distance'] for r in z)/len(z),4),
                    'host_rejected':sum(r['host_rejected'] for r in z)}
    return summary


def run(base,prior_adapter,out,n=240,heldout=40,epochs=12,mix_branches=False):
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before=load_prior(base,prior_adapter)
    X,Y,sources=dataset(n=n,mix_branches=mix_branches)
    baseline=evaluate(before,heldout)
    baseline_branch=evaluate(before,heldout,seed=890031,branch=True)
    baseline_lower=evaluate(before,heldout,seed=790031,branch=True,branch_shape="lower")
    torch.manual_seed(4720) # deterministic first-layer initialization BEFORE optimizer
    model=GoalSwitchAdapter(load_prior(base,prior_adapter))
    model,steps,losses=train(model,X,Y,epochs=epochs)
    after=evaluate(model,heldout)
    after_branch=evaluate(model,heldout,seed=890031,branch=True)
    after_lower=evaluate(model,heldout,seed=790031,branch=True,branch_shape="lower")
    style_before=personality_probe(before);style_after=personality_probe(model)
    walks_before=rollout_probe(before);walks_after=rollout_probe(model)
    path=out/'guidance_recovery_goal_switch_adapter.safetensors'
    provenance={'schema':SCHEMA,'seed':4701,'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),
        'c4_adapter_sha256':hashlib.sha256(Path(prior_adapter).read_bytes()).hexdigest()}
    save_file({k:v.detach().cpu().contiguous() for k,v in model.residual.state_dict().items()},
        str(path),metadata={'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))})
    metrics={'schema':SCHEMA,'synthetic_only':True,'native_undo':False,'changed_game_code':False,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'base_sha256':provenance['base_sha256'],'prior_adapter_sha256':provenance['c4_adapter_sha256'],
       'new_adapter_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'training_branch_mix':mix_branches,
       'dataset_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
       'source_receipt_sha256':sha(sources),'examples':len(X),'steps':steps,'losses':losses,
       'style_before':style_before,'style_after':style_after,
       'rollout_before':rollout_summary(walks_before),'rollout_after':rollout_summary(walks_after),
       'rollout_before_sha256':sha(walks_before),'rollout_after_sha256':sha(walks_after),
       'before':summary(baseline+baseline_branch+baseline_lower),'after':summary(after+after_branch+after_lower),
       'baseline_receipt_sha256':sha(baseline+baseline_branch+baseline_lower),'adapted_receipt_sha256':sha(after+after_branch+after_lower)}
    (out/'goal_switch_walks_before.json').write_text(json.dumps(walks_before,sort_keys=True,separators=(',',':'))+'\n')
    (out/'goal_switch_walks_after.json').write_text(json.dumps(walks_after,sort_keys=True,separators=(',',':'))+'\n')
    (out/'goal_switch_metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    (out/'goal_switch_before_receipts.json').write_text(json.dumps(baseline+baseline_branch+baseline_lower,sort_keys=True,separators=(',',':'))+'\n')
    (out/'goal_switch_after_receipts.json').write_text(json.dumps(after+after_branch+after_lower,sort_keys=True,separators=(',',':'))+'\n')
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--previous',required=True)
    p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=240)
    p.add_argument('--heldout',type=int,default=40);p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--mix-branches',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.base,a.previous,a.out,a.n,a.heldout,a.epochs,a.mix_branches),sort_keys=True))

if __name__=='__main__':main()
