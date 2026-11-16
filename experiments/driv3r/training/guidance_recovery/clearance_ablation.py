"""Optional host-verified branch clearance for the offline CyberSand-like worm.

Experimental *observation* only: no material permissions, executor, native or
Workbench modifications. A frozen cue-calibrated head retains all existing
scores when cues/evidence are absent; new AdamW adapter is evidence gated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from experiment import ARMS, DIRECTIONS, World, sha
from adversarial_cues import (load_prior, near_wall_false_world,
    decoy_true_world, source_hash, pair_view)
from positive_correction import warning_world, candidates_fingerprint
from hard_recovery import false_cue_world, add_supervised, rotate

SCHEMA = 'guidance-recovery-optional-clearance/1'
FEATURES = 28
RADIUS = 5


def clearance(w, candidate, radius=RADIUS):
    """Bounded legal-geometry lookaround beyond a proposed paint stroke.

    Inspect the *candidate endpoint*, excluding the cursor-history corridor.
    Never infer permissions from the head: we only read w.allowed(). Returned
    values are non-authoritative and do not mutate w or the candidate menu.
    """
    if radius < 1 or radius > 8:
        raise ValueError('radius out of bounded range')
    if candidate.action[0] != 'INK' or not candidate.legal or candidate.target in w.history:
        return (0., 0.)
    if not w.allowed(candidate.target):
        return (0., 0.)
    forbidden = set(w.history) | {w.pos}
    seen={candidate.target}
    todo=deque([(candidate.target,1)])
    farthest=1
    while todo:
        point, depth=todo.popleft()
        farthest=max(farthest,depth)
        if depth>=radius:
            continue
        for dx,dy in DIRECTIONS:
            nxt=(point[0]+dx, point[1]+dy)
            if nxt not in seen and nxt not in forbidden and w.allowed(nxt):
                seen.add(nxt)
                todo.append((nxt,depth+1))
    return (round(farthest/radius,6), round(min(len(seen),radius)/radius,6))


def observe(w, arm, evidence=True):
    """Extra evidence is OPTIONAL and cannot alter host action legality."""
    entries=w.candidates(arm)
    original=np.asarray([c.features for c in entries],dtype=np.float32)
    result=np.zeros((len(entries),FEATURES),dtype=np.float32)
    result[:,:25]=original
    if evidence:
        result[:,25:27]=np.asarray([clearance(w,c) for c in entries],np.float32)
        result[:,27]=1.0
    assert len(entries)==10
    assert np.array_equal(result[:,:25],original)
    return result


class EvidenceGatedPolicy(nn.Module):
    """Exact fallback to frozen prior under no-cue OR absent evidence."""
    def __init__(self, prior, width=32):
        super().__init__()
        self.prior=prior
        for p in self.prior.parameters(): p.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(FEATURES,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self,x):
        old=self.prior(x[...,:25])
        return old+(x[...,23]*x[...,27])*self.delta(x).squeeze(-1)


def choose(model,w,arm,evidence=True):
    items=w.candidates(arm)
    x=observe(w,arm,evidence)
    with torch.no_grad():scores=model(torch.from_numpy(x)).cpu().numpy()
    bad=not items[int(np.argmax(scores))].legal
    scores=np.where(np.asarray([c.legal for c in items],bool),scores,-1e4)
    return items[int(np.argmax(scores))], bool(bad), x


def construct(seed, persona, depth, rotation, mirror):
    """True, identical-input productive+decoy pairs, independent open control."""
    return {
      'true': warning_world(seed,persona,depth,rotation,mirror)[0],
      'straight_false':near_wall_false_world(seed+100000,persona,depth,rotation,mirror,'straight')[0],
      'straight_decoy_true':decoy_true_world(seed+100000,persona,depth,rotation,mirror,'straight')[0],
      'corner_false':near_wall_false_world(seed+200000,persona,depth,rotation,mirror,'corner')[0],
      'corner_decoy_true':decoy_true_world(seed+200000,persona,depth,rotation,mirror,'corner')[0],
      'open_false':false_cue_world(seed+300000,persona),
    }


def expected_choice(w, family):
    if family=='open_false':
        return ('INK',(7,7))
    if family.endswith('decoy_true') or family=='true':
        return ('BACKTRACK',None)
    forward=[z for z in w.candidates(0) if z.action[0]=='INK' and z.legal and z.target!=w.history[-1]]
    assert len(forward)==1,(family,w.pos)
    return ('INK',forward[0].target)


def data(seed=60217, instances=240):
    """Train source-level matched labels; withhold deeper-depth eval identities."""
    rng=random.Random(seed)
    xx=[];yy=[];sources=[];labels=[]
    for i in range(instances):
        s=seed+i*1739; persona=(-1 if i%2 else 1)
        depth=rng.choice((2,3));rotation=rng.randrange(4); mirror=bool(rng.randrange(2))
        arm=ARMS[i%3]
        worlds=construct(s,persona,depth,rotation,mirror)
        for family,w in worlds.items():
            no,yes=pair_view(w,arm)
            assert source_hash(no)==source_hash(yes)
            assert candidates_fingerprint(no,arm)==candidates_fingerprint(yes,arm)
            # Oracle labels are never inferred from cue strength; the future
            # geometry differs and is independently protected by the host.
            kind,target=expected_choice(w,family)
            X=observe(yes,arm,True)
            y=[];chosen=add_supervised([],y,yes,arm,kind,target,mass=.96)
            assert len(y)==1 and chosen.legal
            xx.append(X);yy.append(y[0])
            labels.append((family, s, w.seed, arm, source_hash(w),kind,target))
            sources.append({'family':family,'source':source_hash(w),'seed':w.seed,
              'world_material_sha256':sha(sorted(w.protected)),
              'menu_sha256':candidates_fingerprint(w,arm),'evidence_sha256':hashlib.sha256(X.tobytes()).hexdigest()})
    x=np.asarray(xx,np.float32);y=np.asarray(yy,np.float32)
    assert x.shape==(instances*6,10,FEATURES)
    assert y.shape==(instances*6,10)
    assert np.all(x[:,:,23]==1) and np.all(x[:,:,27]==1)
    return x,y,labels,sources


def train(model,x,y,epochs=18,seed=4403,lr=.002):
    torch.manual_seed(seed); torch.set_num_threads(1)
    opt=torch.optim.AdamW(model.delta.parameters(),lr=lr,weight_decay=.002)
    xt=torch.from_numpy(x);yt=torch.from_numpy(y)
    rng=np.random.default_rng(seed);losses=[];steps=0
    for _ in range(epochs):
        order=rng.permutation(len(x));total=0.
        for start in range(0,len(x),128):
            ids=order[start:start+128]
            scores=model(xt[ids]);probs=torch.log_softmax(scores,dim=-1)
            loss=-(yt[ids]*probs).sum(dim=-1).mean()
            # Preserve policy diversity by minimizing only necessary deltas.
            change=scores-model.prior(xt[ids,:,:25])
            loss=loss+.003*change.square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.delta.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(ids);steps+=1
        losses.append(round(total/len(x),6))
    model.eval()
    return model,steps,losses


def evaluate(model,seeds=32,seed_start=920001,with_evidence=True):
    """New identities/depth, same-source guidance/cue controls and raw receipts."""
    records=[]
    for i in range(seeds):
        seed=seed_start+i*977; persona=1 if i%2==0 else -1
        rotation=2+i%2;mirror=bool((i//2)%2)
        worlds=construct(seed,persona,4,rotation,mirror)
        for arm in ARMS:
            for family,w in worlds.items():
                for cue in (False,True):
                    v=copy.deepcopy(w)
                    if cue:v.issue_recovery_cue(9)
                    start_hash=source_hash(v)
                    action,raw_bad,x=choose(model,v,arm,with_evidence)
                    kind,target=expected_choice(w,family)
                    expected_kind='INK' if not cue else kind
                    if family=='open_false': expected_kind='INK'
                    # The original no-cue path is free to explore, but expected
                    # correction is only measured on cue-present conditions.
                    correct=(action.action[0]==expected_kind and (target is None or action.target==target))
                    can_apply=copy.deepcopy(v)
                    prior_protected=sha(sorted(can_apply.protected));prior_ink=sha(sorted(can_apply.inked))
                    accepted=can_apply.apply(action)
                    assert accepted and sha(sorted(can_apply.protected))==prior_protected
                    assert can_apply.invalid_attempts==0
                    assert candidates_fingerprint(w,arm)==candidates_fingerprint(v,arm)
                    records.append({'source':start_hash,'seed':seed,'family':family,
                      'arm':arm,'personality':persona,'cue':cue,'evidence':with_evidence,
                      'world_sha256':source_hash(w),'material_sha256':prior_protected,
                      'initial_ink_sha256':prior_ink,
                      'menu_sha256':candidates_fingerprint(v,arm),
                      'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                      'clearance_forward':round(float(max(x[j,25] for j,c in enumerate(v.candidates(arm))
                        if c.action[0]=='INK' and c.legal and c.target not in v.history)),6),
                      'action':action.action,'target':action.target,
                      'correct':bool(correct),'raw_illegal_top1':raw_bad,
                      'host_accepted':accepted,
                      'accepted_action_ink_sha256':sha(sorted(can_apply.inked))})
    return records


def summarize(records):
    out={}
    for arm in ARMS:
        for family in ('true','straight_false','straight_decoy_true',
                       'corner_false','corner_decoy_true','open_false'):
            group=[r for r in records if r['arm']==arm and r['family']==family and r['cue']]
            out[f'{family}/{arm:.2f}']={
                'n':len(group),'cue_correct':sum(z['correct'] for z in group),
                'raw_illegal_top1':sum(z['raw_illegal_top1'] for z in group),
                'host_rejected':sum(not z['host_accepted'] for z in group)}
    return out



def long_range_decoy(seed, persona, rotation, mirror, shape):
    """A 2-depth productive-to-blocked pair aliased EVEN under radius-5.

    The obstruction begins beyond the 5-step evidence horizon. This is a
    deliberately untrained long-range negative control, not a fair oracle
    question for the bounded-head observation contract.
    """
    productive=near_wall_false_world(seed,persona,2,rotation,mirror,shape)[0]
    hidden=(12,7) if shape=='straight' else (12,8)
    stopped=copy.deepcopy(productive)
    stopped.protected.add(rotate(hidden,rotation,mirror))
    assert source_hash(stopped)!=source_hash(productive)
    for arm in ARMS:
        assert candidates_fingerprint(productive,arm)==candidates_fingerprint(stopped,arm)
        assert np.array_equal(observe(productive,arm,True),observe(stopped,arm,True))
    return productive,stopped


def evaluate_limits(model,seeds=32):
    """Original synthetic receipts exposing finite lookahead and corruption."""
    limit=[];corruption=[]
    for i in range(seeds):
        s=970001+i*1319;persona=1 if i%2==0 else -1
        rotation=2+i%2;mirror=bool((i//2)%2)
        for arm in ARMS:
            for shape in ('straight','corner'):
                good,bad=long_range_decoy(s+100000,persona,rotation,mirror,shape)
                for fam,w in (('long_viable',good),('long_hidden_trap',bad)):
                    w.issue_recovery_cue(9)
                    c,raw,x=choose(model,w,arm,True)
                    want='INK' if fam=='long_viable' else 'BACKTRACK'
                    limit.append({'seed':s,'family':shape+'/'+fam,'arm':arm,
                       'world_sha256':source_hash(w),'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                       'menu_sha256':candidates_fingerprint(w,arm),'action':c.action,
                       'target':c.target,'correct':c.action[0]==want,'raw_illegal_top1':raw})
                productive=near_wall_false_world(s+200000,persona,4,rotation,mirror,shape)[0]
                decoy=decoy_true_world(s+200000,persona,4,rotation,mirror,shape)[0]
                productive.issue_recovery_cue(9);decoy.issue_recovery_cue(9)
                assert np.array_equal(observe(productive,arm,False),observe(decoy,arm,False))
                for fam,w,other,answer in (('viable',productive,decoy,'INK'),('decoy',decoy,productive,'BACKTRACK')):
                    clean=observe(w,arm,True);wrong=clean.copy()
                    wrong[:,25:27]=observe(other,arm,True)[:,25:27]
                    cs=w.candidates(arm)
                    with torch.no_grad():logits=model(torch.from_numpy(wrong)).numpy()
                    logits=np.where([c.legal for c in cs],logits,-1e4)
                    c=cs[int(np.argmax(logits))]
                    corruption.append({'seed':s,'family':shape+'/'+fam,'arm':arm,
                      'world_sha256':source_hash(w),'clean_observation_sha256':hashlib.sha256(clean.tobytes()).hexdigest(),
                      'spoof_observation_sha256':hashlib.sha256(wrong.tobytes()).hexdigest(),
                      'menu_sha256':candidates_fingerprint(w,arm),'action':c.action,
                      'correct':c.action[0]==answer})
    return limit,corruption


def style_rollout(model,seeds=24,turns=30):
    """Genuinely sampled held-out game-like worlds and cue/goal changes.

    This is *not* a matched-state intervention after paths diverge. The same
    random-number stream and starting world are retained for each arm/model.
    """
    out=[]
    for i in range(seeds):
        seed=990001+i*1511; family=('forks','switchback','clumps')[i%3]
        for personality in (-1,1):
            for arm in ARMS:
                for cue in (False,True):
                    w=World(seed,family,personality)
                    start=source_hash(w);initial_material=sha(sorted(w.protected))
                    rng=np.random.default_rng(seed+personality*701+int(arm*100)*11)
                    events=[];turn_positive=0;turn_negative=0
                    for tick in range(turns):
                        if tick==6 and cue:w.issue_recovery_cue(8)
                        if tick==15:w.change_goal()
                        candidates=w.candidates(arm)
                        x=observe(w,arm,True)
                        with torch.no_grad():l=model(torch.from_numpy(x)).numpy().astype(np.float64)
                        l=np.where(np.asarray([c.legal for c in candidates]),l,-1e5)
                        l-=l.max();p=np.exp(l/.9);p/=p.sum()
                        chosen=candidates[int(rng.choice(len(candidates),p=p))]
                        before=w.pos;prev_dir=w.last_dir
                        assert w.apply(chosen) and w.invalid_attempts==0
                        assert sha(sorted(w.protected))==initial_material
                        kind=chosen.action[0];dx,dy=chosen.action[1:]
                        signed_turn=prev_dir[0]*dy-prev_dir[1]*dx if kind in ('INK','LIFT') else 0
                        turn_positive+=int(signed_turn>0);turn_negative+=int(signed_turn<0)
                        events.append((tick,kind,before,w.pos,chosen.target,
                            bool(w.recovery_cue_ttl),w.goal))
                        if w.stopped:break
                    out.append({'seed':seed,'family':family,'personality':personality,
                        'arm':arm,'cue':cue,'source_sha256':start,
                        'material_sha256':initial_material,'final_ink_sha256':sha(sorted(w.inked)),
                        'steps':len(events),'unique_ink':len(w.inked),'goal_distance':w.distance(),
                        'left_turns':turn_positive,'right_turns':turn_negative,'stopped':w.stopped,
                        'host_rejected':w.invalid_attempts,'events':events})
    return out


def summarize_style(records):
    out={}
    for arm in ARMS:
        for cue in (False,True):
            group=[r for r in records if r['arm']==arm and r['cue']==cue]
            by_persona={p:[r for r in group if r['personality']==p] for p in (-1,1)}
            def bias(items):
                turns=sum(r['left_turns']+r['right_turns'] for r in items)
                return (sum(r['left_turns']-r['right_turns'] for r in items)/turns) if turns else 0.
            out[f'{arm:.2f}/cue={int(cue)}']={'episodes':len(group),
               'mean_unique_ink':round(float(np.mean([r['unique_ink'] for r in group])),4),
               'mean_goal_distance':round(float(np.mean([r['goal_distance'] for r in group])),4),
               'mean_steps':round(float(np.mean([r['steps'] for r in group])),4),
               'turn_bias_plus':round(bias(by_persona[1]),6),
               'turn_bias_minus':round(bias(by_persona[-1]),6),
               'signed_personality_separation':round(bias(by_persona[1])-bias(by_persona[-1]),6),
               'host_rejected':sum(r['host_rejected'] for r in group),
               'stopped':sum(r['stopped'] for r in group)}
    return out


def run(base,previous,out,instances=240,seeds=32,epochs=18):
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prior=load_prior(base,previous)
    source_prior=EvidenceGatedPolicy(prior);source_prior.eval()
    X,Y,labels,sources=data(instances=instances)
    baseline=evaluate(source_prior,seeds)
    torch.manual_seed(22451)
    adapted=EvidenceGatedPolicy(load_prior(base,previous))
    adapted,steps,losses=train(adapted,X,Y,epochs=epochs)
    updated=evaluate(adapted,seeds)
    withheld=evaluate(adapted,seeds,with_evidence=False)
    source0=evaluate(adapted,seeds,seed_start=940001,with_evidence=True)
    long_negative,spoof_negative=evaluate_limits(adapted,seeds)
    sampled_before=style_rollout(source_prior,seeds=24)
    sampled_after=style_rollout(adapted,seeds=24)
    path=out/'guidance_recovery_clearance_adapter.safetensors'
    metadata={'research':json.dumps({'schema':SCHEMA,'train_seed':60217,
      'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),
      'prior_adapter_sha256':hashlib.sha256(Path(previous).read_bytes()).hexdigest()},sort_keys=True,separators=(',',':'))}
    save_file({k:v.detach().cpu().contiguous() for k,v in adapted.delta.state_dict().items()},str(path),metadata=metadata)
    result={'schema':SCHEMA,'synthetic_only':True,'native_material_undo':False,'trained_d1':False,
      'training_instances':instances,'training_observations':len(X),'optimizer_steps':steps,
      'cross_entropy_regularized':losses,'seed_train':60217,'seed_eval':920001,
      'source_file_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),
      'prior_adapter_sha256':hashlib.sha256(Path(previous).read_bytes()).hexdigest(),
      'new_adapter_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'dataset_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
      'sources_sha256':sha(sources),'labels_sha256':sha(labels),
      'baseline':summarize(baseline),'with_evidence':summarize(updated),
      'heldout_long_range_negative':{name:{'n':len([r for r in long_negative if r['family']==name]),
        'correct':sum(r['correct'] for r in long_negative if r['family']==name)}
        for name in sorted({r['family'] for r in long_negative})},
      'evidence_corruption_negative':{name:{'n':len([r for r in spoof_negative if r['family']==name]),
        'correct':sum(r['correct'] for r in spoof_negative if r['family']==name)}
        for name in sorted({r['family'] for r in spoof_negative})},
      'sampled_before':summarize_style(sampled_before),'sampled_after':summarize_style(sampled_after),
      'evidence_withheld':summarize(withheld),'second_holdout':summarize(source0),
      'receipts_sha256':{k:sha(v) for k,v in (('baseline',baseline),('updated',updated),
        ('withheld',withheld),('second',source0))},
      'numpy':np.__version__,'torch':torch.__version__}
    for name,records in [('baseline',baseline),('updated',updated),
            ('withheld',withheld),('second',source0),('long_negative',long_negative),
            ('spoof_negative',spoof_negative),('sampled_before',sampled_before),
            ('sampled_after',sampled_after)]:
        (out/(name+'_receipts.json')).write_text(json.dumps(records,sort_keys=True,indent=2)+'\n')
    (out/'clearance_metrics.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--instances',type=int,default=240)
    p.add_argument('--seeds',type=int,default=32)
    p.add_argument('--epochs',type=int,default=18)
    a=p.parse_args()
    print(json.dumps(run(a.base,a.previous,a.out,a.instances,a.seeds,a.epochs),indent=2,sort_keys=True))

if __name__=='__main__':main()
