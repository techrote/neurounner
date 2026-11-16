"""C14: replay-backed counterfactual trajectory preference trainer.

Only external synthetic CyberSand-like host states are used. The policy is
frozen C12 plus a new bounded, cue-active BACKTRACK-only adapter. All choices
are host-legal and no native simulation, assets or game content are changed.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import ARMS,sha
from adversarial_cues import source_hash
from topology_transfer import build,TRAIN_FAMILIES
from signed_feedback import HostAdvice,observe
from personality_retention_core import RetentionGate,load_c12
SCHEMA='guidance-recovery-trajectory-preferences/1'
PREREQUISITE_KEYS=('base','c4','c7','c8','c9','c11','c12')


def params(model):
    return model.delta.parameters()

def legal_probs(model,x,cs,temperature=.8):
    with torch.no_grad(): scores=model(torch.from_numpy(x)).numpy().astype(np.float64)
    legal=np.array([c.legal for c in cs],bool)
    # Illegal unmasked logits are diagnostic only, not applied.
    invalid=not legal[int(scores.argmax())]
    scores=np.where(legal,scores,-1e8)
    z=np.exp((scores-scores.max())/temperature)
    return z/z.sum(),invalid

def outcome_reward(w,initial_distance,used,steps):
    # Only host-observed trajectory outcomes. No full-map oracle.
    reached=float(w.distance()==0)
    progress=(initial_distance-w.distance())/max(1.,initial_distance)
    return 2.*reached+.35*progress-.012*steps-.015*used

def suffix(prior,w,advice,arm,forced_idx,seed,horizon=22):
    # Deep-copy world+host advice *together* to preserve context identity.
    w,advice=copy.deepcopy((w,advice))
    initdist=w.distance();protected=sha(sorted(w.protected));initial_ink=set(w.inked)
    cs,_=observe(w,arm,advice)
    assert cs[forced_idx].legal
    forced_action=cs[forced_idx].action
    assert w.apply(cs[forced_idx])
    advice.advance(w)
    rng=np.random.default_rng(seed)
    trace=[{'tick':0,'action':forced_action,'distance':w.distance(),'cursor':w.pos}]
    illegal=0
    for tick in range(1,horizon):
        if w.stopped or w.distance()==0:break
        menu,x=observe(w,arm,advice)
        prob,bad=legal_probs(prior,x,menu);illegal+=int(bad)
        selected=menu[int(rng.choice(len(menu),p=prob))]
        assert w.apply(selected)
        advice.advance(w)
        trace.append({'tick':tick,'action':selected.action,'distance':w.distance(),'cursor':w.pos})
    assert w.invalid_attempts==0 and sha(sorted(w.protected))==protected and initial_ink<=w.inked
    return outcome_reward(w,initdist,len(w.inked-initial_ink),len(trace)),bool(w.distance()==0),trace,illegal

def data(prior,n=28,seed=411701,horizon=22,replicates=2):
    X=[];Y=[];WEIGHTS=[];receipts=[];stats={'pairs':0,'reached_back':0,'reached_forward':0,'ties':0,
                                             'positive_advantages':0,'negative_advantages':0,
                                             'illegal_unmasked_predictions':0}
    torch.set_num_threads(1)
    for i in range(n):
        family=TRAIN_FAMILIES[i%len(TRAIN_FAMILIES)]
        person=1 if (i//2)%2 else -1
        rotation=i%4;mirror=bool((i//4)%2)
        for outcome in ('reverse','ahead'):
            s=seed+i*7307+(1000000 if outcome=='ahead' else 0)
            for arm in ARMS:
                for directive in ('backtrack','continue'):
                    # Exact same geometry/menu for different host advice strengths.
                    for strength in (.35,1.):
                        w=build(s,person,family,rotation,mirror,outcome)
                        advice=HostAdvice(w)
                        getattr(advice,'issue_'+directive)(w,strength=strength,ttl=9)
                        cs,x=observe(w,arm,advice)
                        if not cs[8].legal:continue
                        p,illegal=legal_probs(prior,x,cs)
                        candidates=[j for j,c in enumerate(cs) if c.legal and j!=8 and c.action[0]!='STOP']
                        if not candidates:continue
                        alt=max(candidates,key=lambda j:p[j])
                        init_world=source_hash(w)
                        rewards=[[],[]]; reached=[[],[]]; sample_traces=[]
                        for k in range(replicates):
                            rng_seed=s+int(arm*100)*71+k*2971+(0 if directive=='backtrack' else 83121)
                            for branch,j in enumerate((8,alt)):
                                ret,done,trace,bad=suffix(prior,w,advice,arm,j,rng_seed,horizon)
                                rewards[branch].append(ret);reached[branch].append(done)
                                stats['illegal_unmasked_predictions']+=bad
                                if k==0:sample_traces.append(trace)
                        advantage=float(np.mean(rewards[0])-np.mean(rewards[1]))
                        stats['pairs']+=1
                        stats['reached_back']+=sum(reached[0]);stats['reached_forward']+=sum(reached[1])
                        if abs(advantage)<.02:stats['ties']+=1
                        if advantage>.02:stats['positive_advantages']+=1
                        if advantage<-.02:stats['negative_advantages']+=1
                        # Outcome-derived soft preference; downweight exact ties.
                        # Lower cap avoids forcing deterministic action selection.
                        label=float(1/(1+np.exp(-1.5*advantage)))
                        weight=float(.15+min(1.0,abs(advantage)))
                        X.append(x);Y.append(label);WEIGHTS.append(weight)
                        receipts.append({'family':family,'seed':s,'personality':person,'rotation':rotation,
                          'mirror':mirror,'outcome':outcome,'arm':arm,'directive':directive,
                          'strength':strength,'world_sha256':init_world,
                          'protected_sha256':sha(sorted(w.protected)),
                          'menu_sha256':sha([(c.action,c.target,c.legal) for c in cs]),
                          'obs_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                          'forced_back':cs[8].action,'forced_forward':cs[alt].action,
                          'paired_rewards':[[round(float(v),5) for v in rr] for rr in rewards],
                          'paired_successes':reached,'advantage':round(advantage,6),'soft_target':round(label,6),
                          'sample_trace_back':sample_traces[0],
                          'sample_trace_forward':sample_traces[1]})
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(WEIGHTS,np.float32),receipts,stats

def fit(candidate,prior,X,Y,W,epochs=10,lr=.0007,seed=411733,regularization=.2):
    torch.manual_seed(seed);torch.set_num_threads(1)
    xt=torch.from_numpy(X);yt=torch.from_numpy(Y);wt=torch.from_numpy(W)
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(params(candidate),lr=lr,weight_decay=.012)
    with torch.no_grad():
        prior_logits=prior(xt).masked_fill(xt[:,:,4]<.5,-1e7)
        orig_margin=prior_logits[:,8]-torch.logsumexp(torch.cat((prior_logits[:,:8],prior_logits[:,9:]),1),1)
    losses=[];updates=0
    for e in range(epochs):
        order=rng.permutation(len(X));total=0.
        for start in range(0,len(X),64):
            idx=order[start:start+64];x=xt[idx]
            logits=candidate(x).masked_fill(x[:,:,4]<.5,-1e7)
            margin=logits[:,8]-torch.logsumexp(torch.cat((logits[:,:8],logits[:,9:]),1),1)
            # Weighted continuous action-value difference, + anchored KL.
            ce=nn.functional.binary_cross_entropy_with_logits(margin,yt[idx],reduction='none')
            drift=(torch.sigmoid(margin)-torch.sigmoid(orig_margin[idx])).square()
            delta=candidate.delta(x[:,8,:]).square().squeeze()
            loss=(ce*wt[idx]).mean()+regularization*drift.mean()+.005*delta.mean()
            loss.backward();nn.utils.clip_grad_norm_(params(candidate),1.5)
            opt.step();opt.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(idx);updates+=1
        losses.append(round(total/len(X),6))
    return candidate.eval(),updates,losses

def train_and_save(paths,out,n=28,epochs=20):
    torch.set_num_threads(1);torch.manual_seed(411733)
    paths=[Path(p) for p in paths]
    prior=load_c12(paths)
    X,Y,W,receipts,counts=data(prior,n=n)
    candidate=RetentionGate(load_c12(paths),width=12,scale=2.)
    candidate,steps,losses=fit(candidate,prior,X,Y,W,epochs=epochs)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    predecessors={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(PREREQUISITE_KEYS,paths)}
    ckpt=out/'guidance_recovery_trajectory_outcomes.safetensors'
    save_file({k:v.detach().contiguous().cpu() for k,v in candidate.delta.state_dict().items()},str(ckpt),
        metadata={'provenance':json.dumps({'schema':SCHEMA,'predecessors':predecessors,'width':12,'scale':2.},sort_keys=True,separators=(',',':'))})
    return {'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
      'predecessors':predecessors,'training_menus':len(X),'paired_suffix_rollouts':len(X)*4,
      'training_observations_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()+W.tobytes()).hexdigest(),
      'training_receipts_sha256':sha(receipts),'diagnostics':counts,'adamw_updates':steps,'losses':losses}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in PREREQUISITE_KEYS+('out',):p.add_argument('--'+k,required=True)
    p.add_argument('--n',type=int,default=28);p.add_argument('--epochs',type=int,default=20)
    a=p.parse_args()
    print(json.dumps(train_and_save([getattr(a,k) for k in PREREQUISITE_KEYS],a.out,a.n,a.epochs),sort_keys=True))
