"""C11: host-signed recovery and explicit continue advice, offline CyberSand-like worms.

Model-only research. A host instruction is never permission or real material Undo.
C9 and its prerequisites are frozen. Only BACKTRACK scores may change.
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
from persistent_feedback import PersistentGate, Context, load_c8, state_after
from topology_transfer import ALL_FAMILIES, TRAIN_FAMILIES, OOD_FAMILIES, build

SCHEMA = 'guidance-recovery-signed-feedback/1'


def load_c9(base,c4,c7,c8,c9):
    model=PersistentGate(load_c8(base,c4,c7,c8))
    model.delta.load_state_dict(load_file(str(c9)))
    return model.eval()


class HostAdvice:
    """Explicit bounded host events; synthetic observation, not native contract."""
    def __init__(self,w):
        self.directive=0
        self.strength=0.
        self.age=0
        self.remaining=0
        self.initial_goal=tuple(w.goal)
        self.context=None

    def issue_backtrack(self,w,strength=1.,ttl=9):
        self._validate(strength,ttl)
        if strength==0.:
            self.cancel(w);return
        w.issue_recovery_cue(12)
        self.context=Context(w,steps=ttl)
        self.directive=1
        self.strength=float(strength)
        self.age=0
        self.remaining=ttl
        self.initial_goal=tuple(w.goal)

    def issue_continue(self,w,strength=1.,ttl=6):
        self._validate(strength,ttl)
        if strength==0.:
            self.cancel(w);return
        # Host supersedes its previous advisory, without reversing applied ink.
        w.recovery_cue_ttl=0
        self.context=None
        self.directive=-1
        self.strength=float(strength)
        self.age=0
        self.remaining=ttl
        self.initial_goal=tuple(w.goal)

    def cancel(self,w):
        w.recovery_cue_ttl=0
        self.context=None
        self.directive=0
        self.strength=0.
        self.remaining=0

    @staticmethod
    def _validate(strength,ttl):
        if not 0.<=strength<=1. or not 1<=ttl<=12:
            raise ValueError('host advice strength/ttl out of bounds')

    def active(self,w):
        if not self.remaining or self.strength<=0. or tuple(w.goal)!=self.initial_goal or w.stopped or w.distance()==0:
            return 0
        if self.directive==1:
            return 1 if self.context is not None and self.context.active(w) else 0
        return -1 if self.directive==-1 else 0

    def advance(self,w):
        self.age+=1
        self.remaining=max(0,self.remaining-1)
        if self.context is not None:
            self.context.advance(w)
        if tuple(w.goal)!=self.initial_goal or w.stopped or w.distance()==0:
            self.remaining=0


def observe(w,arm,advice=None):
    cs=w.candidates(arm)
    original=np.asarray([c.features for c in cs],dtype=np.float32)
    sign=advice.active(w) if advice is not None else 0
    confidence=advice.strength if sign else 0.
    age=min(advice.age,12)/12 if sign else 0.
    # First 26 features are C9. Only positive BACKTRACK context enables C9 correction.
    context=float(sign==1)
    x=np.concatenate((original,np.full((10,1),context,np.float32),
      np.full((10,1),sign,np.float32),np.full((10,1),confidence,np.float32),
      np.full((10,1),age,np.float32)),axis=1)
    assert x.shape==(10,29) and np.array_equal(x[:,:25],original)
    return cs,x


class SignedGate(nn.Module):
    def __init__(self,prior,width=12,scale=1.,variant="bounded"):
        if variant not in ("bounded","symmetric"): raise ValueError("variant")
        super().__init__()
        self.prior=prior
        for param in self.prior.parameters(): param.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(29,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        self.scale=float(scale)
        self.variant=variant

    def forward(self,x):
        baseline=self.prior(x[...,:26])
        # Only the BACKTRACK candidate may respond. Never affect host legality.
        selector=(x[...,26]<0).to(x.dtype) if self.variant=="bounded" else x[...,26].abs()
        enabled=x[...,2]*x[...,4]*selector*x[...,27]
        return baseline + enabled*self.scale*torch.tanh(self.delta(x).squeeze(-1))


def training_data(n=144,seed=202611716):
    xs=[];ys=[];receipts=[]
    for i in range(n):
        fam=TRAIN_FAMILIES[(i//2)%2]
        person=1 if i%2 else -1
        rot=(i//4)%4;mirror=bool((i//16)%2)
        for outcome in ('reverse','ahead'):
            s=seed+i*1793+(1000000 if outcome=='ahead' else 0)
            for depth in (0,1,2,3):
                w,ctx=state_after(s,person,fam,rot,mirror,outcome,depth)
                for strength in (.45,1.):
                    advice=HostAdvice(w)
                    if outcome=='reverse':
                        advice.issue_backtrack(w,strength)
                        advice.context=ctx
                    else:
                        advice.issue_continue(w,strength)
                    advice.age=depth
                    for arm in ARMS:
                        cs,x=observe(w,arm,advice)
                        assert cs[8].legal and x[8,2]==1 and x[8,26]!=0
                        xs.append(x);ys.append(int(outcome=='reverse'))
                        receipts.append({'source_sha256':source_hash(w),'family':fam,'seed':s,
                            'personality':person,'rotation':rot,'mirror':mirror,'depth':depth,
                            'outcome':outcome,'arm':arm,'strength':strength,
                            'material_sha256':sha(sorted(w.protected)),
                            'menu_sha256':sha([(c.action,c.target,c.legal) for c in cs]),
                            'obs_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    return np.asarray(xs,np.float32),np.asarray(ys,np.float32),receipts


def train(model,X,Y,epochs=10,seed=88011,lr=.0012):
    torch.set_num_threads(1);torch.manual_seed(seed)
    xt=torch.from_numpy(X);yt=torch.from_numpy(Y)
    opt=torch.optim.AdamW(model.delta.parameters(),lr=lr,weight_decay=.005)
    rng=np.random.default_rng(seed);losses=[];updates=0
    for _ in range(epochs):
        order=rng.permutation(len(X));loss_sum=0.
        for k in range(0,len(X),128):
            ix=order[k:k+128];x=xt[ix]
            logits=model(x).masked_fill(x[:,:,4]<.5,-1e6)
            back=logits[:,8]-torch.logsumexp(torch.cat((logits[:,:8],logits[:,9:]),1),1)
            loss=nn.functional.binary_cross_entropy_with_logits(back,yt[ix])
            loss=loss+.001*model.delta(x[:,8,:]).square().mean()
            loss.backward();nn.utils.clip_grad_norm_(model.delta.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            updates+=1;loss_sum+=float(loss.detach())*len(ix)
        losses.append(round(loss_sum/len(X),6))
    return model.eval(),updates,losses


def choose(model,w,arm,advice,rng,temperature=.8):
    cs,x=observe(w,arm,advice)
    with torch.no_grad():
        inp=torch.from_numpy(x if isinstance(model,SignedGate) else x[:,:26])
        scores=model(inp).numpy().astype(np.float64)
    illegal_top1=not cs[int(scores.argmax())].legal
    scores=np.where([c.legal for c in cs],scores,-1e8)
    p=np.exp((scores-scores.max())/temperature);p/=p.sum()
    idx=int(rng.choice(len(cs),p=p))
    return cs[idx],illegal_top1,hashlib.sha256(x.tobytes()).hexdigest()


def evaluate(models,n=12,seed=2560001,horizon=28,temperature=.8,release_ticks=(0,1,2,4),strength=1.):
    rows=[]
    for fam in ALL_FAMILIES:
      for i in range(n):
        s=seed+i*1901+ALL_FAMILIES.index(fam)*300000
        rot=i%4;mirror=bool((i//4)%2)
        for person in (-1,1):
         for arm in ARMS:
          for outcome in ('reverse','ahead'):
           modes=('no_cue','live') if outcome=='reverse' else ('no_cue','live')+tuple('release_'+str(k) for k in release_ticks)
           for mode in modes:
            for name,model in models.items():
             w=build(s+(1000000 if outcome=='ahead' else 0),person,fam,rot,mirror,outcome)
             original_sha=source_hash(w);materials_sha=sha(sorted(w.protected));initial_ink=set(w.inked)
             advice=HostAdvice(w)
             if mode!='no_cue': advice.issue_backtrack(w,strength)
             rng=np.random.default_rng(s+int(arm*100)*67+(1000000 if outcome=='ahead' else 0)+
               {'no_cue':0,'live':970,'release_0':1300,'release_1':1301,'release_2':1302,'release_4':1304}[mode])
             trace=[];bad=0;turn_left=turn_right=post_back=0;revoked=False
             for tick in range(horizon):
                if mode.startswith('release_') and tick==int(mode.split('_')[1]) and not w.stopped and w.distance()>0:
                    # C9 is the cancel-only control. C11 sees an explicit CONTINUE cue.
                    if name=='C11':advice.issue_continue(w,strength,ttl=6)
                    else:advice.cancel(w)
                    revoked=True
                directive=advice.active(w)
                before=set(w.inked);last_dir=w.last_dir
                c,raw,obs=choose(model,w,arm,advice,rng,temperature)
                bad+=int(raw)
                assert w.apply(c) and w.invalid_attempts==0
                assert before<=w.inked and materials_sha==sha(sorted(w.protected))
                if c.action[0] in ('INK','LIFT'):
                    turn=last_dir[0]*c.action[2]-last_dir[1]*c.action[1]
                    turn_left+=int(turn>0);turn_right+=int(turn<0)
                if revoked and c.action[0]=='BACKTRACK':post_back+=1
                trace.append({'tick':tick,'cursor':w.pos,'action':c.action,'active_directive':directive,
                   'history_depth':len(w.history),'ink_delta':len(w.inked-before),
                   'observation_sha256':obs})
                advice.advance(w)
                if w.stopped or w.distance()==0: break
             rows.append({'model':name,'family':fam,'seed':w.seed,'personality':person,
               'arm':arm,'outcome':outcome,'mode':mode,'source_sha256':original_sha,
               'material_sha256':materials_sha,'final_ink_sha256':sha(sorted(w.inked)),
               'goal_reached':w.distance()==0,'final_distance':w.distance(),
               'turns':len(trace),'post_release_backtracks':post_back,
               'left':turn_left,'right':turn_right,'ink_added':len(w.inked-initial_ink),
               'raw_illegal_top1':bad,'host_rejected':w.invalid_attempts,'trace':trace})
    return rows


def summarize(rows):
    output={}
    for name in sorted({r['model'] for r in rows}):
     for family_group,fams in (('trained',TRAIN_FAMILIES),('withheld',OOD_FAMILIES),('all',ALL_FAMILIES)):
      for outcome in ('reverse','ahead'):
       for mode in (('no_cue','live') if outcome=='reverse' else ('no_cue','live','release_0','release_1','release_2','release_4')):
        sub=[r for r in rows if r['model']==name and r['family'] in fams and r['outcome']==outcome and r['mode']==mode]
        def bias(person):
            part=[r for r in sub if r['personality']==person]
            return sum(r['left']-r['right'] for r in part)/max(1,sum(r['left']+r['right'] for r in part))
        output[f'{name}/{family_group}/{outcome}/{mode}']={
            'n':len(sub),'goal_reached':sum(r['goal_reached'] for r in sub),
            'mean_turns':round(float(np.mean([r['turns'] for r in sub])),3),
            'mean_distance':round(float(np.mean([r['final_distance'] for r in sub])),3),
            'mean_post_release_backtracks':round(float(np.mean([r['post_release_backtracks'] for r in sub])),3),
            'personality_turn_gap':round(bias(1)-bias(-1),6),
            'host_rejected':sum(r['host_rejected'] for r in sub),
            'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in sub)}
    return output


def guard(old,new):
    total=0;nonback=0;revoked=0
    for fam in ALL_FAMILIES:
      for person in (-1,1):
       for arm in ARMS:
        for outcome in ('reverse','ahead'):
         w=build(870011,person,fam,0,False,outcome)
         for typ in ('no_cue','continue','reverse','revoked'):
            v=copy.deepcopy(w);advice=HostAdvice(v)
            if typ=='continue':advice.issue_continue(v)
            if typ in ('reverse','revoked'):advice.issue_backtrack(v)
            if typ=='revoked':advice.cancel(v)
            _,x=observe(v,arm,advice)
            with torch.no_grad():
               a=new(torch.from_numpy(x));b=old(torch.from_numpy(x[:,:26]))
            assert torch.equal(a[:8],b[:8]) and a[9]==b[9]
            nonback+=9
            if typ in ('no_cue','revoked'):
                assert torch.equal(a,b)
                total+=1
            if typ=='revoked':revoked+=1
    return {'exact_inactive_pairs':total,'exact_other_action_scores':nonback,'revocations':revoked}


def run(base,c4,c7,c8,c9,out,n=144,epochs=10,eval_n=12,variant="bounded"):
    torch.set_num_threads(1)
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    old=load_c9(base,c4,c7,c8,c9)
    torch.manual_seed(63012)
    model=SignedGate(load_c9(base,c4,c7,c8,c9),scale=(5. if variant=="symmetric" else 1.),variant=variant)
    X,Y,receipts=training_data(n)
    model,steps,losses=train(model,X,Y,epochs)
    checks=guard(old,model)
    traces=evaluate({'C9':old,'C11':model},n=eval_n)
    checkpoint=dest/'guidance_recovery_signed_feedback.safetensors'
    provenance={'schema':SCHEMA,'predecessors':{label:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for label,path in [('base',base),('C4',c4),('C7',c7),('C8',c8),('C9',c9)]}}
    save_file({k:v.detach().cpu().contiguous() for k,v in model.delta.state_dict().items()},
          str(checkpoint),metadata={'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))})
    result={'schema':SCHEMA,'research_only':True,'native_execution':False,'d1_training':False,
        'training_cases':len(X),'training_source_groups':n,'training_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
        'training_receipt_sha256':sha(receipts),'optimizer_updates':steps,'losses':losses,
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'predecessors':provenance,'gate_checks':checks,'residual_scale':model.scale,'variant':variant,
        'trajectory_count':len(traces),
        'trajectories_sha256':sha(traces),'results':summarize(traces)}
    (dest/'metrics.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (dest/'raw_trajectories.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    (dest/'training_receipts.json').write_text(json.dumps(receipts,sort_keys=True,separators=(',',':'))+'\n')
    return result


def main():
    parser=argparse.ArgumentParser()
    for x in ('base','c4','c7','c8','c9','out'):parser.add_argument('--'+x,required=True)
    parser.add_argument('--n',type=int,default=144)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--eval-n',type=int,default=12)
    parser.add_argument('--variant',choices=('bounded','symmetric'),default='bounded')
    args=parser.parse_args()
    print(json.dumps({k:v for k,v in run(args.base,args.c4,args.c7,args.c8,args.c9,args.out,args.n,args.epochs,args.eval_n,args.variant).items()
        if k in ('training_cases','optimizer_updates','checkpoint_sha256','trajectory_count','losses')},sort_keys=True))


if __name__=='__main__':main()
