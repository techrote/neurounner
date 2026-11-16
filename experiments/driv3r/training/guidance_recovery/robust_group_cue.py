"""C16: group-robust, outcome-replay cue arbitration (research-only).

Frozen C12; no CyberSand native/game mutations. Ground-truth cue validity is
never a model feature: it is used to balance training *risk groups* only.
Only host-legal BACKTRACK logit can change, under active signed host advice.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import sha
from topology_transfer import build
from adversarial_cues import source_hash
from signed_feedback import HostAdvice, observe
from personality_retention_core import load_c12
from trajectory_outcomes import evaluate, summarize, personality

KEYS = ('base','c4','c7','c8','c9','c11','c12')
SCHEMA = 'guidance-recovery-cue-group-risk/1'

def evidence(x):
    # Host-legal local action menu ONLY. No topology truth, reward, or cue-validity bit.
    legal=x[:,:8,4]>.5
    invalid=torch.full_like(x[:,:8,7],-1e3)
    p=torch.where(legal,x[:,:8,7],invalid).max(1).values.clamp(-1,1)
    options=legal.sum(1).to(x.dtype)/8.
    blocked=torch.where(legal,x[:,:8,22],invalid).max(1).values.clamp(0,1)
    return torch.stack([x[:,8,7],p,x[:,8,7]-p,blocked,options],1)

class GroupGate(nn.Module):
    """Residual only on index 8; other logits bit-identical, including STOP."""
    def __init__(self,prior,scale=1.25,width=16):
        super().__init__()
        self.prior=prior
        for p in prior.parameters():p.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(34,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        self.scale=float(scale)
    def forward(self,x):
        if x.ndim==2:return self.forward(x.unsqueeze(0)).squeeze(0)
        old=self.prior(x)
        new=old.clone()
        inp=torch.cat((x[:,8,:],evidence(x)),1)
        # Host-controlled enable: action type + legality + signed live cue + strength.
        enabled=x[:,8,2]*x[:,8,4]*(x[:,8,26].abs()>.5).to(x.dtype)*x[:,8,27]
        new[:,8]=old[:,8]+enabled*self.scale*torch.tanh(self.delta(inp).squeeze(1))
        return new

def load_replays(receipts):
    X=[];labels=[];weights=[];groups=[];checked=0
    for r in receipts:
        w=build(r['seed'],r['personality'],r['family'],r['rotation'],r['mirror'],r['outcome'])
        advice=HostAdvice(w)
        getattr(advice,'issue_'+r['directive'])(w,strength=r['strength'],ttl=9)
        cs,x=observe(w,r['arm'],advice)
        assert source_hash(w)==r['world_sha256']
        assert sha(sorted(w.protected))==r['protected_sha256']
        assert sha([(c.action,c.target,c.legal) for c in cs])==r['menu_sha256']
        assert hashlib.sha256(x.tobytes()).hexdigest()==r['obs_sha256']
        a=float(r['advantage'])
        group=2*int(r['outcome']=='ahead')+int(r['directive']=='continue')
        X.append(x);labels.append(1/(1+np.exp(-1.5*a)))
        weights.append(.15+min(1.,abs(a)));groups.append(group);checked+=1
    return tuple(np.asarray(z,d) for z,d in ((X,np.float32),(labels,np.float32),(weights,np.float32),(groups,np.int64)))+(checked,)

def fit(model,prior,X,Y,W,G,epochs=24,alpha=.5,seed=166016,lr=.0008):
    torch.manual_seed(seed);torch.set_num_threads(1)
    tx=torch.from_numpy(X);ty=torch.from_numpy(Y);tw=torch.from_numpy(W);tg=torch.from_numpy(G)
    with torch.no_grad():
        old=prior(tx).masked_fill(tx[:,:,4]<.5,-1e7)
        pm=old[:,8]-torch.logsumexp(torch.cat((old[:,:8],old[:,9:]),1),1)
        pp=torch.sigmoid(pm)
    optim=torch.optim.AdamW(model.delta.parameters(),lr=lr,weight_decay=.012)
    rng=np.random.default_rng(seed);losses=[];steps=0
    for epoch in range(epochs):
        order=rng.permutation(len(tx));total=0.
        for offset in range(0,len(tx),96):
            idx=order[offset:offset+96];x=tx[idx]
            logits=model(x).masked_fill(x[:,:,4]<.5,-1e7)
            margin=logits[:,8]-torch.logsumexp(torch.cat((logits[:,:8],logits[:,9:]),1),1)
            lossrow=nn.functional.binary_cross_entropy_with_logits(margin,ty[idx],reduction='none')*tw[idx]
            # Group DRO surrogate: penalize worst of all four outcome/directive
            # groups rather than pooling common easy cases into the average.
            gl=torch.stack([lossrow[tg[idx]==k].mean() for k in range(4) if bool((tg[idx]==k).any())])
            worst=torch.logsumexp(gl*4.,0)/4.
            drift=(torch.sigmoid(margin)-pp[idx]).square().mean()
            regular=model.delta(torch.cat((x[:,8,:],evidence(x)),1)).square().mean()
            loss=.5*lossrow.mean()+.5*worst+alpha*drift+.002*regular
            loss.backward();nn.utils.clip_grad_norm_(model.delta.parameters(),1.5)
            optim.step();optim.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(idx);steps+=1
        losses.append(round(total/len(tx),6))
    return model.eval(),steps,losses

def guards(prior,model):
    with torch.no_grad():
        for case in range(4):
            w=build(913007+case*991,-1 if case%2 else 1,('hairpin','hook')[case%2],case,False,'reverse' if case<2 else 'ahead')
            a=HostAdvice(w)
            if case==1:a.issue_backtrack(w,strength=.7)
            if case==2:a.issue_continue(w,strength=.9)
            if case==3:a.issue_backtrack(w,strength=1.);a.cancel(w)
            cs,x=observe(w,.65,a)
            t=torch.from_numpy(x)
            p=prior(t);q=model(t)
            assert torch.equal(p[:8],q[:8]) and torch.equal(p[9:],q[9:])
            if case in (0,3):assert torch.equal(p,q)
            assert cs[8].action[0]=='BACKTRACK'
    return {'inactive_exact':True,'nonbacktrack_exact':True,'signed_host_only':True}

def run(paths,receipts_path,out,epochs=24,eval_n=8,validate_n=4):
    torch.set_num_threads(1);paths=[Path(p) for p in paths];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prior=load_c12(paths)
    receipts=json.loads(Path(receipts_path).read_text())
    X,Y,W,G,n=load_replays(receipts)
    hash_inputs=hashlib.sha256(X.tobytes()+Y.tobytes()+W.tobytes()+G.tobytes()).hexdigest()
    candidates={}
    metrics={}
    for name,alpha in (('C16-A',.2),('C16-B',3.)):
        torch.manual_seed(166016)
        model=GroupGate(load_c12(paths))
        model,updates,losses=fit(model,prior,X,Y,W,G,epochs=epochs,alpha=alpha)
        assert guards(prior,model)
        path=out/(name+'.safetensors')
        provenance={'schema':SCHEMA,'alpha':alpha,'model':name,'inputs':hash_inputs,
                    'predecessors':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)}}
        save_file({k:v.detach().cpu().contiguous() for k,v in model.delta.state_dict().items()},str(path),
                  metadata={'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))})
        candidates[name]=model
        metrics[name]={'alpha':alpha,'updates':updates,'losses':losses,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    # Validation and final fresh test seeds are mutually disjoint and withheld
    # from the source-train hairpin/staircase examples.
    val=evaluate({'C12':prior,**candidates},n=validate_n,seed=690161,horizon=40)
    test=evaluate({'C12':prior,**candidates},n=eval_n,seed=785117,horizon=48)
    results={'schema':SCHEMA,'training_replays':n,'train_groups':dict(zip(('rev_back','rev_continue','ahead_back','ahead_continue'),map(int,np.bincount(G,minlength=4)))),
      'training_data_sha256':hash_inputs,'source_receipt_sha256':hashlib.sha256(Path(receipts_path).read_bytes()).hexdigest(),
      'models':metrics,'validation':summarize(val),'test':summarize(test),'personality_test':personality(test),
      'validation_rows':len(val),'test_rows':len(test),'guard':guards(prior,candidates['C16-A']),
      'train_family':['hairpin','staircase'],'withheld_family':['hook','zigzag']}
    for name,obj in (('C16_results.json',results),('C16_validation_trajectories.json',val),('C16_test_trajectories.json',test)):
        (out/name).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in KEYS:ap.add_argument('--'+k,required=True)
    ap.add_argument('--receipts',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--eval-n',type=int,default=8);ap.add_argument('--validate-n',type=int,default=4)
    args=ap.parse_args()
    print(json.dumps(run([getattr(args,k) for k in KEYS],args.receipts,args.out,args.epochs,args.eval_n,args.validate_n),sort_keys=True)[:4000])
