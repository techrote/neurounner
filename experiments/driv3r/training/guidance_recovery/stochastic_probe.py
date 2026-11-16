"""C7 stochastic 16-turn independent-seed probe; host validates every move."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from experiment import ARMS, sha
from adversarial_cues import load_prior, source_hash
from goal_switch_recovery import GoalSwitchAdapter, world
from personality_gate import BacktrackOnlyGate

SCHEMA='guidance-recovery-stochastic-probe/1'

def load_models(base, adapter, c6, c7):
    original=load_prior(base,adapter).eval()
    old=GoalSwitchAdapter(load_prior(base,adapter));old.residual.load_state_dict(load_file(str(c6)));old.eval()
    new=BacktrackOnlyGate(load_prior(base,adapter));new.gate.load_state_dict(load_file(str(c7)));new.eval()
    return {'C4':original,'C6':old,'C7':new}

def sample(model,w,arm,rng,temperature=.7):
    cs=w.candidates(arm)
    x=torch.tensor([c.features for c in cs],dtype=torch.float32)
    with torch.no_grad(): logits=model(x).numpy()
    logits=np.where(np.asarray([c.legal for c in cs],bool),logits,-1e9)/temperature
    logits=np.exp(logits-logits.max());logits/=logits.sum()
    return cs[int(rng.choice(len(cs),p=logits))]

def run(models,seeds=24,turns=16):
    torch.set_num_threads(1)
    receipts=[]
    for shape in ('corridor','upper','lower'):
        for i in range(seeds):
            seed=947000+i*1327; person=1 if i%2 else -1
            rot=2+i%2;mirror=bool((i//2)%2)
            for arm in ARMS:
                for cue in (False,True):
                    for name,model in models.items():
                        w=world(seed,person,rot,mirror,'reverse',branch=(shape!='corridor'),branch_shape=shape if shape!='corridor' else 'upper')
                        orig_hash=source_hash(w);mat_hash=sha(sorted(w.protected));ink0=set(w.inked)
                        if cue:w.issue_recovery_cue(9)
                        rng=np.random.default_rng(1043+i*1009+int(arm*100)*43+int(cue)*991)
                        trajectory=[];rejected=0
                        for _ in range(turns):
                            c=sample(model,w,arm,rng)
                            ok=w.apply(c);rejected+=int(not ok)
                            trajectory.append({'action':c.action,'target':c.target,'pos':w.pos,'accepted':bool(ok)})
                            if not ok or w.stopped or w.distance()==0:break
                        assert mat_hash==sha(sorted(w.protected))
                        assert ink0<=w.inked
                        receipts.append({'schema':SCHEMA,'model':name,'seed':seed,'personality':person,
                            'rotation':rot,'mirror':mirror,'shape':shape,'arm':arm,'cue':cue,
                            'source_sha256':orig_hash,'materials_sha256':mat_hash,
                            'goal_reached':w.distance()==0,'end_distance':w.distance(),
                            'new_ink_cells':len(w.inked-ink0),'rejected':rejected,
                            'trace':trajectory})
    result={}
    for shape in ('corridor','upper','lower'):
        for arm in ARMS:
            for cue in (False,True):
                for model in models:
                    r=[z for z in receipts if z['shape']==shape and z['arm']==arm and z['cue']==cue and z['model']==model]
                    result[f'{shape}/{arm:.2f}/{int(cue)}/{model}']={
                        'n':len(r),'reached':sum(z['goal_reached'] for z in r),
                        'mean_final_distance':round(sum(z['end_distance'] for z in r)/len(r),4),
                        'rejected':sum(z['rejected'] for z in r),
                        'mean_new_ink':round(sum(z['new_ink_cells'] for z in r)/len(r),4)}
    return receipts,result

def main():
    p=argparse.ArgumentParser()
    for name in ('base','adapter','c6','c7','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--seeds',type=int,default=24);p.add_argument('--turns',type=int,default=16)
    a=p.parse_args();p_out=Path(a.out);p_out.mkdir(parents=True,exist_ok=True)
    models=load_models(a.base,a.adapter,a.c6,a.c7)
    rec,summary=run(models,seeds=a.seeds,turns=a.turns)
    (p_out/'stochastic_receipts.json').write_text(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n')
    metrics={'schema':SCHEMA,'synthetic_only':True,'native_undo':False,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'receipts_sha256':sha(rec),'seeds':a.seeds,'turns':a.turns,'episodes':len(rec),'summary':summary}
    (p_out/'stochastic_metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'episodes':len(rec),'receipts_sha256':metrics['receipts_sha256'],'summary':summary},sort_keys=True))

if __name__=='__main__':main()
