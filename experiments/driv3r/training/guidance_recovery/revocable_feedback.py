"""C10: bounded feedback countermand + selectively learned BACKTRACK residual.

External CyberSand-like host only. Countermand is a HOST event, never a model
instruction, material rollback or permission grant. All C9 prerequisites frozen.
"""
from __future__ import annotations
import argparse, hashlib, json, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from experiment import ARMS, sha
from adversarial_cues import source_hash
from persistent_feedback import load_c8, PersistentGate, Context, state_after
from topology_transfer import ALL_FAMILIES,TRAIN_FAMILIES,OOD_FAMILIES, build

SCHEMA='guidance-recovery-revocable-feedback/1'


def load_c9(base,c4,c7,c8,c9):
    prior=PersistentGate(load_c8(base,c4,c7,c8))
    prior.delta.load_state_dict(load_file(str(c9)))
    return prior.eval()


class HostFeedback:
    """Research-only, host-controlled authority, not inferred from target label.

    Strength is explicitly host-specified and can be revoked at any point.
    The default does not infer correctness by looking into hidden geometry.
    """
    def __init__(self,world,strength=1.,steps=9):
        if not 0.<=strength<=1.: raise ValueError('strength must be in [0,1]')
        self.context=Context(world,steps=steps)
        self.strength=float(strength)
        self.revoked=False
        self.age=0
    def active(self,w):return self.context.active(w) and not self.revoked and self.strength>0
    def revoke(self,w):
        self.revoked=True
        self.context.remaining=0
        w.recovery_cue_ttl=0 # explicit host countermand; resets advice, not legality
    def advance(self,w):
        self.age+=1;self.context.advance(w)


def observe(w,arm,feedback=None):
    cs=w.candidates(arm)
    base=np.asarray([c.features for c in cs],np.float32)
    effective=bool(feedback is not None and feedback.active(w))
    confidence=(feedback.strength if effective else 0.)
    age=(min(feedback.age,12)/12 if feedback is not None else 0.)
    # Original 25 features unchanged. C9 context is independent of cue truth.
    arr=np.concatenate([base,np.full((10,1),float(effective),np.float32),
       np.full((10,1),confidence,np.float32),np.full((10,1),age,np.float32)],axis=1)
    assert arr.shape==(10,28) and np.array_equal(base,arr[:,:25])
    return cs,arr


class RevocableGate(nn.Module):
    """Freeze C9, learn bounded one-candidate residual, NEVER alter action legality."""
    def __init__(self,prior,width=16,scale=6.):
        super().__init__()
        self.prior=prior
        for p in self.prior.parameters():p.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(28,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        self.scale=scale
    def forward(self,x):
        context=x[...,25]*x[...,26]
        # Preserve the exact original C9 computation for full-trust or
        # inactive-only batches; cloning to multiply by 1 changed FP last bits.
        if not torch.any(context):
            return self.prior(x[...,:26])
        if torch.all(x[...,26]==1):
            prior_input=x[...,:26]
        else:
            prior_input=x[...,:26].clone()
            prior_input[...,25]*=x[...,26]
        logits=self.prior(prior_input)
        enabled=x[...,2]*x[...,23]*context
        return logits + enabled*self.scale*torch.tanh(self.delta(x).squeeze(-1))


def make_data(n=180,seed=1300301):
    X=[];Y=[];receipts=[]
    for i in range(n):
      fam=TRAIN_FAMILIES[(i//2)%2];person=1 if i%2 else -1
      rot=(i//4)%4;mir=bool((i//16)%2)
      for outcome in ('reverse','ahead'):
       s=seed+i*1823+(1000000 if outcome=='ahead' else 0)
       for k in (0,1,2,3):
        w,ctx=state_after(s,person,fam,rot,mir,outcome,k)
        for confidence in (0.55,1.):
         feedback=HostFeedback(w,strength=confidence)
         feedback.context=ctx # original host-issued context survived k legal steps
         feedback.age=k
         for arm in ARMS:
          cs,x=observe(w,arm,feedback)
          assert cs[8].legal and x[8,2]==1
          X.append(x);Y.append(float(outcome=='reverse'))
          receipts.append({'seed':s,'family':fam,'personality':person,'arm':arm,
             'outcome':outcome,'age':k,'confidence':confidence,
             'source_sha256':source_hash(w),'material_sha256':sha(sorted(w.protected)),
             'menu_sha256':sha([(c.action,c.target,c.legal) for c in cs]),
             'obs_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),receipts


def train(model,X,Y,epochs=10,seed=4399,lr=.0012,reverse_weight=1.):
    torch.manual_seed(seed);torch.set_num_threads(1)
    xt=torch.from_numpy(X);yt=torch.from_numpy(Y)
    opt=torch.optim.AdamW(model.delta.parameters(),lr=lr,weight_decay=.005)
    rng=np.random.default_rng(seed);losses=[];steps=0
    for epoch in range(epochs):
      perm=rng.permutation(len(X));total=0.
      for k in range(0,len(X),128):
       ix=perm[k:k+128];bx=xt[ix]
       out=model(bx).masked_fill(bx[...,4]<.5,-1e6)
       contrast=out[:,8]-torch.logsumexp(torch.cat([out[:,:8],out[:,9:]],dim=1),dim=1)
       # Proper Bernoulli calibration vs legal alternatives; no hardcoded action override.
       loss=nn.functional.binary_cross_entropy_with_logits(contrast,yt[ix],pos_weight=torch.tensor(reverse_weight))
       loss=loss+.003*(model.delta(bx[:,8,:])**2).mean()
       loss.backward();nn.utils.clip_grad_norm_(model.delta.parameters(),2.)
       opt.step();opt.zero_grad(set_to_none=True)
       total+=float(loss.detach())*len(ix);steps+=1
      losses.append(round(total/len(X),6))
    return model.eval(),steps,losses


def choose(model,w,arm,feedback,rng,temperature=.8):
    cs,x=observe(w,arm,feedback)
    with torch.no_grad():
      if isinstance(model,RevocableGate): y=model(torch.from_numpy(x)).numpy().astype(np.float64)
      else:
       c9_input=x[:,:26].copy();c9_input[:,25]*=x[:,26]
       y=model(torch.from_numpy(c9_input)).numpy().astype(np.float64)
    legal=np.asarray([c.legal for c in cs],bool)
    raw_bad=not legal[y.argmax()]
    y=np.where(legal,y,-1e8)
    logits=y/temperature
    probs=np.exp(logits-logits.max());probs/=probs.sum()
    return cs[int(rng.choice(len(cs),p=probs))],bool(raw_bad),hashlib.sha256(x.tobytes()).hexdigest()


def evaluate(models,n=12,seed=1600401,horizon=24,temperature=.8,confidence=1.,strength_by_model=None):
    rows=[]
    for fam in ALL_FAMILIES:
      for i in range(n):
       s=seed+i*1709+ALL_FAMILIES.index(fam)*250000
       rot=i%4;mir=bool((i//4)%2)
       for person in (-1,1):
        for arm in ARMS:
         for outcome in ('reverse','ahead'):
          for mode in ('no_cue','live','countermand_2'):
           for name,model in models.items():
            w=build(s+(1000000 if outcome=='ahead' else 0),person,fam,rot,mir,outcome)
            w0=source_hash(w);material=sha(sorted(w.protected));first_ink=set(w.inked)
            if mode!='no_cue':w.issue_recovery_cue(12)
            fb=HostFeedback(w,strength=(strength_by_model or {}).get(name,confidence))
            rng=np.random.default_rng(s+int(arm*100)*43+(1000000 if outcome=='ahead' else 0)+{'no_cue':0,'live':941,'countermand_2':1941}[mode])
            trace=[];left=right=back=run=longest=rawbad=0
            for tick in range(horizon):
              if mode=='countermand_2' and tick==2:fb.revoke(w)
              active=fb.active(w)
              c,raw,obs=choose(model,w,arm,fb,rng,temperature)
              rawbad+=raw
              old_ink=set(w.inked);prev=w.last_dir
              assert w.apply(c) and w.invalid_attempts==0
              assert old_ink<=w.inked and material==sha(sorted(w.protected))
              if c.action[0] in ('INK','LIFT'):
                turn=prev[0]*c.action[2]-prev[1]*c.action[1]
                left+=int(turn>0);right+=int(turn<0)
              if c.action[0]=='BACKTRACK':back+=1;run+=1;longest=max(longest,run)
              else:run=0
              trace.append({'turn':tick,'action':c.action,'cursor':w.pos,
                'ink_delta':len(w.inked-old_ink),'active':active,
                'revoked':fb.revoked,'obs_sha256':obs})
              fb.advance(w)
              if w.stopped or w.distance()==0:break
            rows.append({'model':name,'family':fam,'seed':w.seed,'personality':person,'arm':arm,
              'outcome':outcome,'cue_mode':mode,'source_sha256':w0,'material_sha256':material,
              'final_ink_sha256':sha(sorted(w.inked)),'reached':w.distance()==0,
              'end_distance':w.distance(),'steps':len(trace),'backtracks':back,
              'longest_backtrack_run':longest,'left':left,'right':right,
              'ink_added':len(w.inked-first_ink),'raw_illegal_top1':rawbad,
              'host_rejected':w.invalid_attempts,'trace':trace})
    return rows


def summarize(rows):
    summary={}
    for m in sorted({r['model'] for r in rows}):
     for family_group, families in [('trained',TRAIN_FAMILIES),('withheld',OOD_FAMILIES),('all',ALL_FAMILIES)]:
      for outcome in ('reverse','ahead'):
       for mode in ('no_cue','live','countermand_2'):
        s=[r for r in rows if r['model']==m and r['family'] in families and r['outcome']==outcome and r['cue_mode']==mode]
        def bias(p):
          z=[r for r in s if r['personality']==p]; return sum(r['left']-r['right'] for r in z)/max(1,sum(r['left']+r['right'] for r in z))
        summary[f'{m}/{family_group}/{outcome}/{mode}']={
          'n':len(s),'reached':sum(r['reached'] for r in s),
          'mean_steps':round(float(np.mean([r['steps'] for r in s])),3),
          'mean_backtracks':round(float(np.mean([r['backtracks'] for r in s])),3),
          'mean_distance':round(float(np.mean([r['end_distance'] for r in s])),3),
          'personality_turn_gap':round(bias(1)-bias(-1),6),
          'host_rejected':sum(r['host_rejected'] for r in s)}
    return summary


def gate_checks(old,new,n=10):
    inactive=other=countermand=0
    for i in range(n):
      for fam in ALL_FAMILIES:
       for outcome in ('reverse','ahead'):
        w=build(110011+i*971,1 if i%2 else -1,fam,i%4,bool(i%2),outcome)
        for arm in ARMS:
         for cued in (False,True):
          q=copy.deepcopy(w)
          if cued:q.issue_recovery_cue(12)
          fb=HostFeedback(q)
          if cued:fb.revoke(q)
          _,x=observe(q,arm,fb)
          with torch.no_grad():
            a=new(torch.from_numpy(x));b=old(torch.from_numpy(x[:,:26]))
          assert torch.equal(a,b)
          inactive+=1
         q=copy.deepcopy(w);q.issue_recovery_cue(12)
         fb=HostFeedback(q)
         _,x=observe(q,arm,fb)
         with torch.no_grad():
          a=new(torch.from_numpy(x));b=old(torch.from_numpy(x[:,:26]))
         assert torch.equal(a[:8],b[:8]) and a[9]==b[9]
         other+=9
         fb.revoke(q);_,x=observe(q,arm,fb)
         assert np.all(x[:,25]==0) and np.all(x[:,26]==0)
         countermand+=1
    return {'inactive_identical_cases':inactive,'active_nonbacktrack_exact':other,
            'host_countermands_exercised':countermand}


def run(base,c4,c7,c8,c9,out,n=180,epochs=10,eval_n=12,reverse_weight=1.,scale=6.):
    torch.set_num_threads(1);p=Path(out);p.mkdir(parents=True,exist_ok=True)
    old=load_c9(base,c4,c7,c8,c9)
    torch.manual_seed(5011)
    new=RevocableGate(load_c9(base,c4,c7,c8,c9),scale=scale)
    X,Y,receipts=make_data(n)
    new,steps,losses=train(new,X,Y,epochs,reverse_weight=reverse_weight)
    checks=gate_checks(old,new)
    traces=evaluate({'C9':old,'C10':new},n=eval_n)
    ck=p/'guidance_recovery_revocable_feedback.safetensors'
    sources={'schema':SCHEMA,'predecessors':{label:hashlib.sha256(Path(path).read_bytes()).hexdigest()
       for label,path in [('base',base),('C4',c4),('C7',c7),('C8',c8),('C9',c9)]}}
    save_file({k:v.cpu().detach().contiguous() for k,v in new.delta.state_dict().items()},str(ck),
        metadata={'provenance':json.dumps(sources,sort_keys=True,separators=(',',':'))})
    metrics={'schema':SCHEMA,'synthetic_only':True,'native_actions':False,'d1_weights_updated':False,
      'training_cases':len(X),'source_groups':n,'optimizer_updates':steps,
      'reverse_weight':reverse_weight,'residual_scale':scale,
      'losses':losses,'training_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
      'receipts_sha256':sha(receipts),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
      'predecessors':sources,'gate_checks':checks,'episodes':len(traces),
      'trace_sha256':sha(traces),'results':summarize(traces)}
    (p/'metrics.json').write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    (p/'training_source_receipts.json').write_text(json.dumps(receipts,sort_keys=True,separators=(',',':'))+'\n')
    (p/'raw_trajectories.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    return metrics


def main():
    parser=argparse.ArgumentParser()
    for name in ('base','c4','c7','c8','c9','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--n',type=int,default=180)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--eval-n',type=int,default=12)
    parser.add_argument('--reverse-weight',type=float,default=1.)
    parser.add_argument('--scale',type=float,default=6.)
    args=parser.parse_args()
    result=run(args.base,args.c4,args.c7,args.c8,args.c9,args.out,args.n,args.epochs,args.eval_n,args.reverse_weight,args.scale)
    print(json.dumps({'optimizer_updates':result['optimizer_updates'],
      'loss_first':result['losses'][0],'loss_final':result['losses'][-1],
      'episodes':result['episodes'],'checkpoint_sha256':result['checkpoint_sha256']},sort_keys=True))

if __name__=='__main__':main()
