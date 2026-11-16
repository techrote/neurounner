"""C7: event-gated BACKTRACK-only learned residual; synthetic DRIV3R game policy.

The host remains the sole authority for legal materials/actions.  No real Undo:
BACKTRACK pops synthetic cursor history but retains every painted cell.
"""
from __future__ import annotations
import argparse, hashlib, json, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import ARMS, sha
from adversarial_cues import load_prior, source_hash
from goal_switch_recovery import (world, dataset, evaluate, summary,
    personality_probe, rollout_probe, rollout_summary, choose)

SCHEMA = 'guidance-recovery-personality-gate/1'

class BacktrackOnlyGate(nn.Module):
    """Learn feedback without changing relative preferences among drawing actions."""
    def __init__(self, prior, width=12):
        super().__init__()
        self.prior = prior
        for p in self.prior.parameters(): p.requires_grad_(False)
        self.gate = nn.Sequential(nn.Linear(25,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self,x):
        old = self.prior(x)
        # Cued objective transition only; no other candidate scores are altered.
        active = x[...,2]*x[...,18]*x[...,23]
        return old + active*self.gate(x).squeeze(-1)


def train(model, X, Y, epochs=24, seed=61731, lr=.002):
    torch.manual_seed(seed); torch.set_num_threads(1)
    xt = torch.from_numpy(X); yt=torch.from_numpy(Y)
    assert xt.shape[1:]==(10,25) and yt.shape[1:]==(10,)
    assert torch.all(xt[:,8,2]==1) and torch.all(xt[:,8,23]==1)
    label = yt[:,8] > .5  # reverse requires BACKTRACK; ahead must preserve INK
    opt=torch.optim.AdamW(model.gate.parameters(),lr=lr,weight_decay=.003)
    rng=np.random.default_rng(seed); losses=[]; steps=0
    for _ in range(epochs):
        order=rng.permutation(len(xt)); accum=0.
        for start in range(0,len(xt),96):
            ids=order[start:start+96]
            logits=model(xt[ids]).masked_fill(xt[ids,:,4]<.5,-1e4)
            contrast=logits[:,8]-torch.logsumexp(torch.cat((logits[:,:8],logits[:,9:]),dim=1),dim=1)
            ce=nn.functional.binary_cross_entropy_with_logits(contrast,label[ids].float())
            # Weak regularisation avoids an arbitrarily large imperative cue.
            penalty=.0008*(model.gate(xt[ids,8,:]).squeeze(-1)**2).mean()
            loss=ce+penalty
            loss.backward();torch.nn.utils.clip_grad_norm_(model.gate.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            accum+=float(loss.detach())*len(ids);steps+=1
        losses.append(round(accum/len(xt),6))
    model.eval();return model,steps,losses


def check_exact_noninterference(model, seeds=12):
    checked=0
    with torch.no_grad():
        for i in range(seeds):
            for outcome in ('reverse','ahead'):
                for arm in ARMS:
                    for cue,change in ((False,True),(True,False),(False,False)):
                        w=world(830000+i*1009,1 if i%2 else -1,2+i%2,bool(i%2),outcome,True)
                        if cue:w.issue_recovery_cue(9)
                        w.goal_change=change
                        x=torch.tensor([c.features for c in w.candidates(arm)],dtype=torch.float32)
                        assert torch.equal(model(x),model.prior(x))
                        checked+=1
    return checked


def check_ink_rank_preservation(model, seeds=12):
    checked=0;max_score_diff=0.
    with torch.no_grad():
        for i in range(seeds):
            for outcome in ('reverse','ahead'):
                for arm in ARMS:
                    w=world(730000+i*907,1 if i%2 else -1,2+i%2,bool(i%2),outcome,True)
                    w.issue_recovery_cue(9)
                    x=torch.tensor([c.features for c in w.candidates(arm)],dtype=torch.float32)
                    a=model.prior(x); b=model(x)
                    for j in range(len(x)):
                        if j==8:continue
                        diff=float((b[j]-a[j]).abs());max_score_diff=max(max_score_diff,diff)
                        assert b[j]==a[j]
                        checked+=1
    return {'checked':checked,'max_score_diff':max_score_diff}


def run(base,previous,out,n=240,heldout=40,epochs=24):
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before=load_prior(base,previous)
    X,Y,training_sources=dataset(n=n,mix_branches=True)
    before_eval=[evaluate(before,n=heldout),evaluate(before,n=heldout,seed=890031,branch=True),
                 evaluate(before,n=heldout,seed=790031,branch=True,branch_shape='lower')]
    torch.manual_seed(7263)
    model=BacktrackOnlyGate(load_prior(base,previous))
    model,steps,losses=train(model,X,Y,epochs=epochs)
    after_eval=[evaluate(model,n=heldout),evaluate(model,n=heldout,seed=890031,branch=True),
                evaluate(model,n=heldout,seed=790031,branch=True,branch_shape='lower')]
    style_before=personality_probe(before);style_after=personality_probe(model)
    walks_before=rollout_probe(before);walks_after=rollout_probe(model)
    ck=out/'guidance_recovery_personality_gate.safetensors'
    provenance={'schema':SCHEMA,'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),
                'prior_adapter_sha256':hashlib.sha256(Path(previous).read_bytes()).hexdigest()}
    save_file({k:v.detach().cpu().contiguous() for k,v in model.gate.state_dict().items()},str(ck),
              metadata={'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))})
    for tag,recs in (('before',sum(before_eval,[])),('after',sum(after_eval,[]))):
        (out/f'{tag}_first_choices.json').write_text(json.dumps(recs,sort_keys=True,separators=(',',':'))+'\n')
    for tag,recs in (('before',walks_before),('after',walks_after)):
        (out/f'{tag}_walks.json').write_text(json.dumps(recs,sort_keys=True,separators=(',',':'))+'\n')
    result={'schema':SCHEMA,'native_undo':False,'native_game_used':False,'paid_compute':False,
            'training_examples':len(X),'training_sources':len({z['world_sha256'] for z in training_sources}),
            'training_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
            'sources_sha256':sha(training_sources),'optimizer_steps':steps,'losses':losses,
            'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
            'base_sha256':provenance['base_sha256'],'prior_adapter_sha256':provenance['prior_adapter_sha256'],
            'noninterference':check_exact_noninterference(model),
            'ink_rank_preservation':check_ink_rank_preservation(model),
            'before':summary(sum(before_eval,[])),'after':summary(sum(after_eval,[])),
            'style_before':style_before,'style_after':style_after,
            'walks_before':rollout_summary(walks_before),'walks_after':rollout_summary(walks_after),
            'before_receipts_sha256':sha(sum(before_eval,[])),'after_receipts_sha256':sha(sum(after_eval,[])),
            'before_walks_sha256':sha(walks_before),'after_walks_sha256':sha(walks_after)}
    (out/'metrics.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True);p.add_argument('--previous',required=True)
    p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=240)
    p.add_argument('--heldout',type=int,default=40);p.add_argument('--epochs',type=int,default=24)
    a=p.parse_args()
    r=run(a.base,a.previous,a.out,a.n,a.heldout,a.epochs)
    print(json.dumps({'checkpoint_sha256':r['checkpoint_sha256'],'training_examples':r['training_examples'],
                     'optimizer_steps':r['optimizer_steps'],'losses':r['losses'],'style_before':r['style_before'],
                     'style_after':r['style_after'],'walks_after':r['walks_after']},sort_keys=True))

if __name__=='__main__':main()
