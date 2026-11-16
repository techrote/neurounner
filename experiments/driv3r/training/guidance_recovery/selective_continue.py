"""C12 selective refusal of false CONTINUE cues; CyberSand-like research only.

Frozen C11 and earlier checkpoints are required. Only legal BACKTRACK score may
change; no production/Workbench integration or native material Undo.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from experiment import ARMS,sha
from adversarial_cues import source_hash
from persistent_feedback import state_after
from signed_feedback import HostAdvice,SignedGate,observe,load_c9
from topology_transfer import TRAIN_FAMILIES
SCHEMA='guidance-recovery-selective-continue/1'

def load_c11(base,c4,c7,c8,c9,c11):
    prior=SignedGate(load_c9(base,c4,c7,c8,c9))
    prior.delta.load_state_dict(load_file(str(c11)))
    return prior.eval()

class SelectiveContinue(nn.Module):
    def __init__(self,prior,width=16,scale=5.):
        super().__init__();self.prior=prior
        for p in self.prior.parameters():p.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(29,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        self.scale=float(scale)
    def forward(self,x):
        baseline=self.prior(x)
        enable=x[...,2]*x[...,4]*(x[...,26]<-.5).to(x.dtype)*x[...,27]
        return baseline+enable*self.scale*torch.tanh(self.delta(x).squeeze(-1))

def examples(n=144,seed=750013):
    xx=[];yy=[];receipts=[]
    for i in range(n):
        fam=TRAIN_FAMILIES[(i//2)%len(TRAIN_FAMILIES)]
        person=1 if i%2 else -1;rotation=(i//4)%4;mirror=bool((i//16)%2)
        for outcome in ('reverse','ahead'):
            s=seed+i*1709+(1000000 if outcome=='ahead' else 0)
            for k in (0,1,2,3):
                w,_=state_after(s,person,fam,rotation,mirror,outcome,k)
                for confidence in (.35,1.):
                    advice=HostAdvice(w);advice.issue_continue(w,confidence,ttl=6);advice.age=k
                    for arm in ARMS:
                        cs,x=observe(w,arm,advice)
                        assert cs[8].legal and cs[8].action[0]=='BACKTRACK'
                        assert x.shape==(10,29) and x[8,26]==-1 and x[8,27]==confidence
                        xx.append(x);yy.append(float(outcome=='reverse'))
                        receipts.append({'seed':w.seed,'family':fam,'personality':person,
                           'rotation':rotation,'mirror':mirror,'outcome':outcome,
                           'depth':k,'arm':arm,'confidence':confidence,
                           'world_sha256':source_hash(w),
                           'protected_sha256':sha(sorted(w.protected)),
                           'menu_sha256':sha([(c.action,c.target,c.legal) for c in cs]),
                           'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    return np.asarray(xx,np.float32),np.asarray(yy,np.float32),receipts

def optimize(model,X,Y,epochs=10,seed=120612,lr=.0016):
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed);xt=torch.from_numpy(X);yt=torch.from_numpy(Y)
    opt=torch.optim.AdamW(model.delta.parameters(),lr=lr,weight_decay=.008)
    losses=[];steps=0
    for epoch in range(epochs):
        perm=rng.permutation(len(X));total=0.
        for start in range(0,len(X),128):
            idx=perm[start:start+128];batch=xt[idx]
            logits=model(batch).masked_fill(batch[:,:,4]<.5,-1e6)
            back=logits[:,8]-torch.logsumexp(torch.cat((logits[:,:8],logits[:,9:]),1),1)
            loss=nn.functional.binary_cross_entropy_with_logits(back,yt[idx])
            loss=loss+.004*(model.delta(batch[:,8,:]).square().mean())
            loss.backward();nn.utils.clip_grad_norm_(model.delta.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            steps+=1;total+=float(loss.detach())*len(idx)
        losses.append(round(total/len(X),6))
    return model.eval(),steps,losses

def run(paths,out,n=144,epochs=10):
    paths=[Path(p) for p in paths]
    torch.set_num_threads(1);torch.manual_seed(120612)
    model=SelectiveContinue(load_c11(*paths),scale=5.)
    X,Y,receipts=examples(n)
    model,steps,losses=optimize(model,X,Y,epochs)
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    ckpt=dest/'guidance_recovery_selective_continue.safetensors'
    preds={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(('base','C4','C7','C8','C9','C11'),paths)}
    meta={'schema':SCHEMA,'predecessors':preds,'scale':5.}
    save_file({k:v.detach().cpu().contiguous() for k,v in model.delta.state_dict().items()},str(ckpt),
      metadata={'provenance':json.dumps(meta,sort_keys=True,separators=(',',':'))})
    result={'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        'training_input_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
        'training_receipts_sha256':sha(receipts),'training_observations':len(X),
        'adamw_steps':steps,'losses':losses,'predecessors':preds}
    (dest/'training_summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('base','c4','c7','c8','c9','c11','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--n',type=int,default=144);ap.add_argument('--epochs',type=int,default=10)
    a=ap.parse_args()
    print(json.dumps(run([getattr(a,k) for k in ('base','c4','c7','c8','c9','c11')],a.out,a.n,a.epochs),sort_keys=True))
