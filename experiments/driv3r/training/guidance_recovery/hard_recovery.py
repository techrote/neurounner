"""Offline adversarial recovery curriculum for the CyberSand-like worm head.

Synthetic host-only training and matched testing. No native actions, no real Undo.
The purpose is to challenge the previous saturated six-step recovery measure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from experiment import World, ARMS, Scorer, make_data, policy_index, evaluation, aggregate, sha

SCHEMA = 'guidance-recovery-hard-trap/1'


def rotate(p, quarter_turns, mirror):
    x, y = p
    if mirror:
        y = 14-y
    for _ in range(quarter_turns % 4):
        x, y = 14-y, x
    return x, y


def trap_world(seed, personality, depth=3, rotation=0, mirror=False):
    """A host-legally constructed cul-de-sac plus the counterintuitive escape branch."""
    if depth not in (2, 3, 4):
        raise ValueError('depth')
    junction = (5, 7)
    stem = [(5+i, 7) for i in range(depth+1)]
    exit_route = [(5, y) for y in (6, 5, 4)]
    exit_route += [(x, 4) for x in range(6, 14)]
    exit_route += [(13, y) for y in (5, 6, 7)]
    corridor = set(stem+exit_route)
    tr = lambda p: rotate(p, rotation, mirror)
    w = World(seed, 'bars', personality)
    w.protected = {(x,y) for x in range(15) for y in range(15) if (x,y) not in {tr(p) for p in corridor}}
    w.pos = tr(junction)
    w.goal = tr((13, 7))
    w.last_dir = (tr((6, 7))[0]-tr((5, 7))[0], tr((6, 7))[1]-tr((5, 7))[1])
    w.history.clear()
    for a,b in zip(stem[:-1],stem[1:]):
        target=tr(b)
        choice=next(c for c in w.candidates(0) if c.action[0]=='LIFT' and c.target==target)
        assert w.apply(choice)
    assert w.pos == tr(stem[-1]) and len(w.history)==depth
    # World.distance() cannot see the occluding protected barrier; host detected a dead end.
    w.stuck=2
    return w,tr(junction),[tr(p) for p in exit_route]


def false_cue_world(seed, personality):
    """Recovery-suggestion counterexample on an open, productive branch."""
    w=World(seed,'islands',personality)
    w.protected=set()
    w.pos=(6,7)
    w.goal=(13,7)
    w.history=[(5,7)]
    w.last_dir=(1,0)
    w.stuck=0
    return w


def target_distribution(w, candidates, chosen, mass=.96):
    legal=[i for i,c in enumerate(candidates) if c.legal]
    assert chosen in legal
    prob=np.zeros(len(candidates),np.float32)
    prob[legal]=(1-mass)/len(legal)
    prob[chosen]+=mass
    return prob


def add_supervised(X,Y,w,arm,kind,target=None,mass=.96):
    c=w.candidates(arm)
    if kind=='BACKTRACK':
        hits=[i for i,v in enumerate(c) if v.action[0]=='BACKTRACK' and v.legal]
    elif kind=='INK':
        hits=[i for i,v in enumerate(c) if v.action[0]=='INK' and v.target==target and v.legal]
    else:
        raise ValueError(kind)
    assert len(hits)==1, (kind,target,w.pos)
    X.append([v.features for v in c]);Y.append(target_distribution(w,c,hits[0]))
    return c[hits[0]]


def build_curriculum(seed=15017, instances=240):
    """Lawful backtracking with sham cues; depth 4 and rotations 2-3 are held out."""
    rng=random.Random(seed)
    X=[];Y=[];labels=[]
    for i in range(instances):
        depth=rng.choice((2,3));rotation=rng.randrange(2);mirror=bool(rng.randrange(2))
        personality=(-1 if i%2 else 1)
        arm=ARMS[i%3]
        w,junction,exit_route=trap_world(seed+i*101,personality,depth,rotation,mirror)
        if i%4 != 0:
            w.issue_recovery_cue(depth+2)
        for t in range(depth):
            a=add_supervised(X,Y,w,arm,'BACKTRACK')
            labels.append(('trap',i,t,arm,personality))
            assert w.apply(a)
        assert w.pos==junction and not w.history
        choice=add_supervised(X,Y,w,arm,'INK',exit_route[0],mass=.92)
        labels.append(('branch',i,depth,arm,personality))
        assert w.apply(choice)
        # Demonstrate a portion of lawful continuation rather than teaching only retreat.
        for t,p in enumerate(exit_route[1:5]):
            a=add_supervised(X,Y,w,arm,'INK',p,mass=.88)
            labels.append(('continue',i,t,arm,personality))
            assert w.apply(a)
        # A spurious cue must not blindly force BACKTRACK on an open productive path.
        good=false_cue_world(seed+i*101,personality)
        if i%2==0:good.issue_recovery_cue(3)
        a=add_supervised(X,Y,good,arm,'INK',(7,7))
        labels.append(('false_cue',i,0,arm,personality))
        assert good.apply(a)
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),labels


def fine_tune(initial,hard_x,hard_y,seed=807,epochs=14,lr=.0008):
    """Train on hard traps mixed with original imitation to reduce style collapse."""
    torch.manual_seed(seed);torch.set_num_threads(1)
    original_x,original_y,_=make_data(episodes=140,seed=20261116)
    # balanced repetition of the hard teacher, while retaining original stochastic style.
    x=np.concatenate([original_x,hard_x,hard_x])
    y=np.concatenate([original_y,hard_y,hard_y])
    XT=torch.from_numpy(x);YT=torch.from_numpy(y)
    model=Scorer();model.load_state_dict(initial.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.002)
    rng=np.random.default_rng(seed);loss=[];steps=0
    for epoch in range(epochs):
        perm=rng.permutation(len(x));total=0.
        for a in range(0,len(x),192):
            idx=perm[a:a+192]
            logits=model(XT[idx]);p=YT[idx]
            err=-(p*torch.log_softmax(logits,dim=1)).sum(dim=1).mean()
            err.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            total+=float(err.detach())*len(idx);steps+=1
        loss.append(round(total/len(x),5))
    return model,steps,loss,len(x)


def run_trap(model,seed,personality,depth,rotation,mirror,arm,cue,mode='student'):
    w,junction,exit_route=trap_world(seed,personality,depth,rotation,mirror)
    initial_mat=tuple(sorted(w.inked));initial_protected=sha(sorted(w.protected))
    if cue:w.issue_recovery_cue(depth+2)
    rng=np.random.default_rng(seed*7+13+personality*31)
    recovered_at=None;clean_at=None;first_branch_at=None;stop=False;choices=[];raw_bad=0
    for step in range(depth+12):
        c=w.candidates(arm)
        idx,bad=policy_index(model,w,c,arm,rng,mode)
        raw_bad+=bad
        action=c[idx]
        choices.append((action.action[0],action.target))
        assert w.apply(action)
        if w.pos==junction and recovered_at is None:recovered_at=step+1
        if w.pos==junction and not w.history and clean_at is None:clean_at=step+1
        if w.pos==exit_route[0] and first_branch_at is None:first_branch_at=step+1
        if w.stopped:stop=True;break
    return {'seed':seed,'personality':personality,'depth':depth,'rotation':rotation,'mirror':mirror,
       'arm':arm,'cue':cue,'start_material_sha':sha(initial_mat),'geometry_sha':initial_protected,
       'reached_junction':recovered_at,'clean_backtrack':clean_at,'first_escape_branch':first_branch_at,
       'painted_cells':len(w.inked),'illegal_host_edits':w.invalid_attempts,'raw_illegal_top1':raw_bad,
       'stop':stop,'events':choices,'end_material_sha':sha(sorted(w.inked))}


def evaluate_hard(model,seeds=16,mode='student'):
    records=[]
    for i in range(seeds):
        seed=911000+i*113
        for depth in (2,3,4):
            for personality in (-1,1):
                rotation=i%4;mirror=bool((i//4)%2)
                for arm in ARMS:
                    for cue in (False,True):
                        records.append(run_trap(model,seed,personality,depth,rotation,mirror,arm,cue,mode))
    return records


def paired_cue_response(records):
    """Compare matched cue toggles without pretending later paths stay same-state."""
    out={}
    for arm in ARMS:
        grp={}
        for r in records:
            if r['arm'] != arm:continue
            key=(r['seed'],r['personality'],r['depth'],r['rotation'],r['mirror'])
            grp.setdefault(key,{})[r['cue']]=bool(r['clean_backtrack'] is not None and r['clean_backtrack']<=r['depth']+2)
        pairs=[v for v in grp.values() if len(v)==2]
        out[f'{arm:.2f}']={'pairs':len(pairs),'cue_only_successes':sum(v[True] and not v[False] for v in pairs),'uncued_only_successes':sum(v[False] and not v[True] for v in pairs),'both_successes':sum(v[False] and v[True] for v in pairs)}
    return out

def summarize(records):
    out={}
    for arm in ARMS:
        for cue in (False,True):
            arr=[r for r in records if r['arm']==arm and r['cue']==cue]
            n=len(arr)
            out[f'{arm:.2f}/cue={int(cue)}']={
                'n':n,
                'junction_within_depth_plus_2':sum(r['reached_junction'] is not None and r['reached_junction']<=r['depth']+2 for r in arr),
                'clean_history_unwind_within_depth_plus_2':sum(r['clean_backtrack'] is not None and r['clean_backtrack']<=r['depth']+2 for r in arr),
                'escaped_branch_within_depth_plus_7':sum(r['first_escape_branch'] is not None and r['first_escape_branch']<=r['depth']+7 for r in arr),
                'mean_ink_cells':round(sum(r['painted_cells'] for r in arr)/n,3),
                'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in arr),
                'host_rejected_actions':sum(r['illegal_host_edits'] for r in arr),
                'stopped':sum(r['stop'] for r in arr),
            }
    return out


def false_cue_preference(model,n=80):
    result={}
    for arm in ARMS:
        for cue in (False,True):
            back=0;paint=0
            for i in range(n):
                w=false_cue_world(801000+i,1 if i%2 else -1)
                if cue:w.issue_recovery_cue(3)
                c=w.candidates(arm)
                with torch.no_grad():score=model(torch.tensor([v.features for v in c],dtype=torch.float32)).numpy()
                score=np.where([v.legal for v in c],score,-1e4)
                pred=c[int(np.argmax(score))].action[0]
                back+=(pred=='BACKTRACK');paint+=(pred=='INK')
            result[f'{arm:.2f}/false_cue={int(cue)}']={'backtrack_greedy':back,'ink_greedy':paint,'n':n}
    return result


def load_initial(checkpoint):
    model=Scorer();model.load_state_dict(load_file(str(checkpoint)))
    model.eval();return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--train-instances',type=int,default=240)
    ap.add_argument('--eval-seeds',type=int,default=16)
    ap.add_argument('--epochs',type=int,default=14)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    baseline=load_initial(args.checkpoint)
    X,Y,labels=build_curriculum(instances=args.train_instances)
    baseline_records=evaluate_hard(baseline,args.eval_seeds)
    model,steps,loss,training_examples=fine_tune(baseline,X,Y,epochs=args.epochs)
    updated_records=evaluate_hard(model,args.eval_seeds)
    outck=args.out/'guidance_recovery_hard_finetuned.safetensors'
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(outck),metadata={'schema':SCHEMA,'seed':'807'})
    metrics={'schema':SCHEMA,'source_scope':'experiments/driv3r/training/guidance_recovery/**',
      'synthetic_only':True,'no_native_undo':True,'initial_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      'checkpoint_sha256':hashlib.sha256(outck.read_bytes()).hexdigest(),'checkpoint_bytes':outck.stat().st_size,
      'hard_dataset_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
      'training_labels_sha256':sha(labels),'hard_examples':len(X),'mixed_training_examples':training_examples,
      'adamw_optimizer_steps':steps,'loss_by_epoch':loss,'eval_seeds':args.eval_seeds,
      'baseline_summary':summarize(baseline_records),'finetuned_summary':summarize(updated_records),
      'false_cue_baseline':false_cue_preference(baseline),'false_cue_finetuned':false_cue_preference(model),
      'baseline_record_sha256':sha(baseline_records),'finetuned_record_sha256':sha(updated_records),
      'paired_cue_baseline':paired_cue_response(baseline_records),'paired_cue_finetuned':paired_cue_response(updated_records),
      'heldout_depth4_baseline':summarize([r for r in baseline_records if r['depth']==4]),
      'heldout_depth4_finetuned':summarize([r for r in updated_records if r['depth']==4]),
      'heldout_rotations23_baseline':summarize([r for r in baseline_records if r['rotation'] in (2,3)]),
      'heldout_rotations23_finetuned':summarize([r for r in updated_records if r['rotation'] in (2,3)]),
      'original_heldout_baseline':aggregate(evaluation(baseline,seeds=8)[0]),
      'original_heldout_finetuned':aggregate(evaluation(model,seeds=8)[0]),
      'training_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'torch_version':torch.__version__,'numpy_version':np.__version__}
    (args.out/'hard_metrics.json').write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    (args.out/'hard_trajectories.json').write_text(json.dumps({'baseline':baseline_records,'finetuned':updated_records},sort_keys=True,indent=2)+'\n')
    print(json.dumps(metrics,sort_keys=True,indent=2))

if __name__=='__main__':main()
