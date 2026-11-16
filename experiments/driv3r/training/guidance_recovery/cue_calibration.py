"""Counterfactual host-cue calibration for the synthetic DRIV3R worm head.

Pairs the SAME world, legal menu, personality and guidance arm with/without a
non-authoritative recovery cue. Tests cue stability in genuine traps and cue
rejection in productive corridors. Does not call CyberSand native/Workbench.
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

from experiment import ARMS, Scorer, make_data, policy_index, evaluation, aggregate, sha
from hard_recovery import (add_supervised, trap_world, false_cue_world,
                           evaluate_hard, summarize, paired_cue_response,
                           false_cue_preference)

SCHEMA = 'guidance-recovery-cue-calibration/1'


def paired_curriculum(seed=33017, instances=240):
    """Paired clean-state targets; rotation 2/3 and depth 4 are test-only.

    The two cues do not change World or legality. Both arms must learn to
    backtrack at a hard dead-end and to ignore recovery cues on productive
    paths. The pair labels are intentionally identical; this is calibration,
    not proof that a model understands correct host advice in general.
    """
    rng = random.Random(seed)
    x=[]; y=[]; labels=[]
    for i in range(instances):
        depth=rng.choice((2,3)); rotation=rng.randrange(2)
        mirror=bool(rng.randrange(2)); personality=1 if i%2 else -1
        arm=ARMS[i%3]
        for cue in (False,True):
            w,junction,exit_route=trap_world(seed+i*103,personality,depth,rotation,mirror)
            if cue:w.issue_recovery_cue(depth+6)
            for step in range(depth):
                action=add_supervised(x,y,w,arm,'BACKTRACK',mass=.97)
                labels.append(('trap',i,cue,step,arm))
                assert w.apply(action)
            assert w.pos==junction and not w.history
            for step,p in enumerate(exit_route[:5]):
                action=add_supervised(x,y,w,arm,'INK',p,mass=.93)
                labels.append(('branch',i,cue,step,arm))
                assert w.apply(action)
            # No progress-based oracle for the false cue: the corridor is open
            # and continuing to paint is the productive decision.
            good=false_cue_world(seed+i*103,personality)
            good.stuck=i%4
            if cue:good.issue_recovery_cue(6)
            action=add_supervised(x,y,good,arm,'INK',(7,7),mass=.97)
            labels.append(('false_cue',i,cue,0,arm,good.stuck))
            assert good.apply(action)
    return np.asarray(x,np.float32),np.asarray(y,np.float32),labels


def train(initial, x, y, seed=911, epochs=10, lr=.00045, style_episodes=120):
    """Paired hard labels plus older stochastic imitation as a style anchor."""
    torch.manual_seed(seed);torch.set_num_threads(1)
    old_x,old_y,_=make_data(episodes=style_episodes,seed=20261116)
    inputs=np.concatenate([old_x,x,x]); targets=np.concatenate([old_y,y,y])
    XT=torch.from_numpy(inputs);YT=torch.from_numpy(targets)
    model=Scorer();model.load_state_dict(initial.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.002)
    rng=np.random.default_rng(seed);losses=[];steps=0
    for _ in range(epochs):
        order=rng.permutation(len(inputs));running=0.
        for a in range(0,len(inputs),192):
            batch=order[a:a+192]
            logits=model(XT[batch]);gold=YT[batch]
            loss=-(gold*torch.log_softmax(logits,dim=1)).sum(dim=1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            steps+=1;running+=float(loss.detach())*len(batch)
        losses.append(round(running/len(inputs),5))
    model.eval();return model,steps,losses,len(inputs)


def matched_first_step(model, seeds=32):
    """Non-trajectory observation: same legal actions before any action."""
    out={}
    for arm in ARMS:
        summary={'pairs':0,'trapped_uncued_backtrack':0,'trapped_cued_backtrack':0,
                 'trapped_changed_first_action':0,'false_uncued_backtrack':0,
                 'false_cued_backtrack':0,'false_changed_first_action':0}
        for i in range(seeds):
            for family in ('trapped','false'):
                indices=[]
                for cue in (False,True):
                    if family=='trapped':
                        w,_,_=trap_world(882001+i*127,(-1 if i%2 else 1),4,(i%4),bool((i//4)%2))
                    else:
                        w=false_cue_world(882001+i*127,(-1 if i%2 else 1))
                        w.stuck=i%4
                    if cue:w.issue_recovery_cue(6)
                    candidates=w.candidates(arm)
                    with torch.no_grad():
                        values=model(torch.tensor([c.features for c in candidates],dtype=torch.float32)).numpy()
                    values=np.where([c.legal for c in candidates],values,-1e4)
                    indices.append(candidates[int(np.argmax(values))].action[0])
                prefix='trapped' if family=='trapped' else 'false'
                summary[f'{prefix}_uncued_backtrack']+=indices[0]=='BACKTRACK'
                summary[f'{prefix}_cued_backtrack']+=indices[1]=='BACKTRACK'
                summary[f'{prefix}_changed_first_action']+=indices[0]!=indices[1]
                if family=='trapped':summary['pairs']+=1
        out[f'{arm:.2f}']=summary
    return out


def run(checkpoint, out_dir, train_instances=240, eval_seeds=16, epochs=10):
    torch.set_num_threads(1)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    base=Scorer();base.load_state_dict(load_file(str(checkpoint)));base.eval()
    x,y,labels=paired_curriculum(instances=train_instances)
    base_records=evaluate_hard(base,seeds=eval_seeds)
    tuned,steps,losses,count=train(base,x,y,epochs=epochs)
    tuned_records=evaluate_hard(tuned,seeds=eval_seeds)
    out_path=out_dir/'guidance_recovery_cue_calibrated.safetensors'
    save_file({k:v.detach().cpu().contiguous() for k,v in tuned.state_dict().items()},str(out_path),metadata={'provenance':json.dumps({'schema':SCHEMA,'seed':911},sort_keys=True,separators=(',',':'))})
    metrics={'schema':SCHEMA,'synthetic_only':True,'no_native_undo':True,
      'initial_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
      'checkpoint_sha256':hashlib.sha256(out_path.read_bytes()).hexdigest(),
      'paired_examples':len(x),'training_examples':count,'optimizer_steps':steps,
      'epoch_losses':losses,'paired_dataset_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
      'labels_sha256':sha(labels),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'baseline_hard':summarize(base_records),'updated_hard':summarize(tuned_records),
      'baseline_cue_pairs':paired_cue_response(base_records),'updated_cue_pairs':paired_cue_response(tuned_records),
      'baseline_first_step':matched_first_step(base),'updated_first_step':matched_first_step(tuned),
      'baseline_false_cue':false_cue_preference(base),'updated_false_cue':false_cue_preference(tuned),
      'baseline_original_holdout':aggregate(evaluation(base,seeds=8)[0]),
      'updated_original_holdout':aggregate(evaluation(tuned,seeds=8)[0]),
      'base_traces_sha256':sha(base_records),'tuned_traces_sha256':sha(tuned_records),
      'torch_version':torch.__version__,'numpy_version':np.__version__}
    (out_dir/'cue_calibration_metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    (out_dir/'cue_calibration_trajectories.json').write_text(json.dumps({'baseline':base_records,'calibrated':tuned_records},sort_keys=True)+'\n')
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--train-instances',type=int,default=240)
    ap.add_argument('--eval-seeds',type=int,default=16)
    ap.add_argument('--epochs',type=int,default=10)
    args=ap.parse_args()
    print(json.dumps(run(args.checkpoint,args.out,args.train_instances,args.eval_seeds,args.epochs),indent=2,sort_keys=True))

if __name__=='__main__':main()
