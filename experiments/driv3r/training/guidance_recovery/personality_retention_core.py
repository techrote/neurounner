"""C13: frozen-C12 on-policy recovery distillation checkpoint trainer.

Research-only, synthetic CyberSand-like world. The frozen C12 policy is not modified.
Only BACKTRACK scores can change, under a live host directive (either sign).
Host legal menus and World.apply remain authoritative; no paint/Undo game access.
"""
from __future__ import annotations
import argparse, hashlib, json, copy
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from experiment import ARMS, DIRECTIONS, sha
from adversarial_cues import source_hash
from signed_feedback import HostAdvice, observe
from topology_transfer import TRAIN_FAMILIES, OOD_FAMILIES, ALL_FAMILIES, build
from selective_continue import SelectiveContinue, load_c11

SCHEMA='guidance-recovery-personality-retention/1'


def load_c12(paths):
    base,c4,c7,c8,c9,c11,c12=map(Path,paths)
    prior=SelectiveContinue(load_c11(base,c4,c7,c8,c9,c11),scale=5.)
    prior.delta.load_state_dict(load_file(str(c12)))
    return prior.eval()


class RetentionGate(nn.Module):
    """Bounded residual on only candidate 8 when a host directive is live.

    Drawing candidates 0..7 and STOP are unchanged even under advice.
    An absent, expired or cancelled advice event gives exact frozen C12 logits.
    """
    def __init__(self,prior,width=12,scale=2.):
        super().__init__()
        self.prior=prior
        for p in prior.parameters(): p.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(29,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        self.scale=float(scale)

    def forward(self,x):
        old=self.prior(x)
        enabled=x[...,2]*x[...,4]*(x[...,26].abs()>.5).to(x.dtype)*x[...,27]
        return old + enabled*self.scale*torch.tanh(self.delta(x).squeeze(-1))


def shortest_dist(w,start):
    if start==w.goal: return 0
    if not w.allowed(start):return 999
    q=deque([(start,0)]);seen={start}
    while q:
        p,d=q.popleft()
        for dx,dy in DIRECTIONS:
            v=(p[0]+dx,p[1]+dy)
            if v in seen or not w.allowed(v):continue
            if v==w.goal:return d+1
            seen.add(v);q.append((v,d+1))
    return 999


def teacher_backtrack(w,cs):
    """Full-world oracle used for labels only (never given as model input).

    Skip ties and permanently isolated targets; do not force deterministic
    BACKTRACK if a forward route is equally short. A positive label requires
    strict path advantage over every legal forward action, not just proximity.
    """
    if not cs[8].legal:return None
    b=shortest_dist(w,cs[8].target)
    f=min((shortest_dist(w,c.target) for c in cs[:8] if c.legal),default=999)
    if b==999 and f==999:return None
    if b<f:return 1.
    if f<b:return 0.
    # LIFT/INK can retrace the same adjacent cell as BACKTRACK but append
    # a new cursor-history item, leaving a pathological loop. In a tie,
    # prefer *clean stack unwind* when the goal is already on history.
    if w.goal in w.history:return 1.
    return 0.


def gather(prior,n=180,seed=293001,horizon=9):
    """On-policy state aggregation; only hairpin/staircase used to train.

    Collect matched host advisories on observed legal menus at each cursor
    state. This deliberately avoids imagined target actions: actions are
    actually executed through the synthetic host before the next sample.
    """
    rows=[];X=[];Y=[];W=[]
    torch.set_num_threads(1)
    for i in range(n):
        fam=TRAIN_FAMILIES[(i//2)%2]
        person=1 if i%2 else -1
        rot=(i//4)%4;mirror=bool((i//16)%2)
        for outcome in ('reverse','ahead'):
            s=seed+i*9731+(1000000 if outcome=='ahead' else 0)
            for arm in ARMS:
                for directive in ('backtrack','continue'):
                    w=build(s,person,fam,rot,mirror,outcome)
                    advice=HostAdvice(w)
                    getattr(advice,'issue_'+directive)(w,strength=1.,ttl=9)
                    rng=np.random.default_rng(s+int(arm*100)*79+(0 if directive=='backtrack' else 9013))
                    initial_protected=sha(sorted(w.protected))
                    for t in range(horizon):
                        cs,x=observe(w,arm,advice)
                        sign=advice.active(w)
                        if sign==0 or not cs[8].legal or w.distance()==0:break
                        y=teacher_backtrack(w,cs)
                        if y is not None:
                            X.append(x);Y.append(y)
                            # stronger penalty for failing an obvious forced route,
                            # mild evidence when a side route invites personal choice
                            W.append(1. if y else .8)
                            rows.append({'world_sha256':source_hash(w),'protected_sha256':initial_protected,
                               'menu_sha256':sha([(c.action,c.target,c.legal) for c in cs]),
                               'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                               'family':fam,'seed':s,'personality':person,'rotation':rot,
                               'mirror':mirror,'arm':arm,'directive':directive,'outcome':outcome,
                               'tick':t,'oracle_backtrack':y})
                        with torch.no_grad():logits=prior(torch.from_numpy(x)).numpy().astype(np.float64)
                        legal=np.asarray([c.legal for c in cs])
                        logits=np.where(legal,logits,-1e8)
                        probs=np.exp((logits-logits.max())/.8);probs/=probs.sum()
                        chosen=cs[int(rng.choice(len(cs),p=probs))]
                        assert w.apply(chosen) and w.invalid_attempts==0
                        assert sha(sorted(w.protected))==initial_protected
                        advice.advance(w)
                        if w.stopped or w.distance()==0:break
    assert X and {int(y) for y in Y}=={0,1}
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(W,np.float32),rows


def fit(model,X,Y,W,epochs=12,seed=130013,lr=.0012,alpha=.05):
    torch.manual_seed(seed);torch.set_num_threads(1)
    rng=np.random.default_rng(seed)
    tx=torch.from_numpy(X);ty=torch.from_numpy(Y);tw=torch.from_numpy(W)
    optim=torch.optim.AdamW(model.delta.parameters(),lr=lr,weight_decay=.012)
    losses=[];steps=0
    for _ in range(epochs):
        order=rng.permutation(len(X));total=0.
        for j in range(0,len(X),128):
            idx=order[j:j+128];x=tx[idx]
            logits=model(x).masked_fill(x[:,:,4]<.5,-1e8)
            margin=logits[:,8]-torch.logsumexp(torch.cat((logits[:,:8],logits[:,9:]),1),1)
            bce=nn.functional.binary_cross_entropy_with_logits(margin,ty[idx],reduction='none')
            # Keep the frozen personality-sensitive policy in charge except
            # when cue guidance provides a compelling advantage.
            loss=(bce*tw[idx]).mean()+alpha*(model.delta(x[:,8,:])**2).mean()
            loss.backward();nn.utils.clip_grad_norm_(model.delta.parameters(),1.5)
            optim.step();optim.zero_grad(set_to_none=True)
            steps+=1;total+=float(loss.detach())*len(idx)
        losses.append(round(total/len(X),6))
    return model.eval(),steps,losses



def train_and_save(paths, out, n=144, epochs=12):
    torch.set_num_threads(1)
    prior=load_c12(paths)
    torch.manual_seed(130013)
    candidate=RetentionGate(load_c12(paths))
    X,Y,W,receipts=gather(prior,n=n)
    candidate,steps,losses=fit(candidate,X,Y,W,epochs=epochs)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    predecessors={key:hashlib.sha256(Path(p).read_bytes()).hexdigest() for key,p in
        zip(('base','C4','C7','C8','C9','C11','C12'),paths)}
    ckpt=out/'guidance_recovery_personality_retention.safetensors'
    save_file({k:v.detach().contiguous().cpu() for k,v in candidate.delta.state_dict().items()},str(ckpt),
        metadata={'provenance':json.dumps({'schema':SCHEMA,'predecessors':predecessors,'scale':candidate.scale},sort_keys=True,separators=(',',':'))})
    summary={'schema':SCHEMA,'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
       'predecessors':predecessors,'training_observations':len(X),'source_groups':n,
       'training_input_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()+W.tobytes()).hexdigest(),
       'training_receipts_sha256':sha(receipts),'positive_labels':int(Y.sum()),
       'optimizer_updates':steps,'losses':losses,'synthetic_only':True}
    (out/'training_summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser()
    for key in ('base','c4','c7','c8','c9','c11','c12','out'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--n',type=int,default=144);ap.add_argument('--epochs',type=int,default=12)
    a=ap.parse_args()
    print(json.dumps(train_and_save([getattr(a,k) for k in ('base','c4','c7','c8','c9','c11','c12')],a.out,a.n,a.epochs),sort_keys=True))

if __name__=='__main__':main()
