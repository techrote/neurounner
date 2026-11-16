"""Offline early-warning correction curriculum for the DRIV3R synthetic worm.

The SAME host World state is duplicated with/without a non-authoritative cue.
At a two-step-lookahead trap, a valid cue should switch the preferred action
from exploratory INK to material-preserving BACKTRACK, but an erroneous cue
in an open corridor should not cause retreat. This is a narrow synthetic
cue interpretation experiment, not native host advice or material Undo.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from experiment import ARMS, Scorer, make_data, policy_index, evaluation, aggregate, sha
from hard_recovery import (
    add_supervised, trap_world, false_cue_world, evaluate_hard, summarize,
    paired_cue_response,
)

SCHEMA = 'guidance-recovery-positive-correction/1'


def warning_world(seed, personality, depth=3, rotation=0, mirror=False):
    """A forward legal cell is a real cul-de-sac; backtracking remains legal.

    Return a pre-terminal cursor with an unpainted, legally enterable forward
    tip. The fatal obstruction is *beyond* the tip, thus the model does not
    get an illegal-action oracle. Only the host warning conveys the reason to
    turn around early. The branch behind the junction is safe and accessible.
    """
    w, junction, exit_route = trap_world(seed, personality, depth, rotation, mirror)
    tip = w.pos
    undo_cursor = next(c for c in w.candidates(0) if c.action[0] == 'BACKTRACK' and c.legal)
    assert w.apply(undo_cursor)
    w.stuck = 0  # warning is anticipatory, not triggered by stuckness
    w.recovery_cue_ttl = 0
    forward = [c for c in w.candidates(0) if c.action[0]=='INK' and c.target == tip and c.legal]
    assert len(forward)==1 and len(w.history)==depth-1
    assert w.pos != junction and w.distance(tip)<w.distance()
    assert not w.inked  # Backtrack was a cursor move, never native material rollback
    return w, junction, tip, exit_route


def candidates_fingerprint(w, arm):
    c=w.candidates(arm)
    return sha([(v.action,v.target,v.legal) for v in c])


def _target_and_apply(xs, ys, labels, w, arm, kind, target, label, mass=.97):
    a=add_supervised(xs,ys,w,arm,kind,target,mass)
    labels.append((label,kind,tuple(a.target),arm))
    assert w.apply(a)
    assert w.invalid_attempts==0


def make_correction_data(seed=51457, instances=288):
    """Train pairs by *source*, not by individual divergent downstream steps.

    Depth four, quarter-turn rotations 2/3 and independent seeds remain held
    out. True warnings and false alarms have the same cue encoding; world
    geometry and cursor history must discriminate between them.
    """
    rng=random.Random(seed)
    x=[];y=[];labels=[];identity=[]
    for i in range(instances):
        depth=rng.choice((2,3))
        rot=rng.randrange(2)
        mirror=bool(rng.randrange(2))
        persona=-1 if i%2 else 1
        arm=ARMS[i%len(ARMS)]
        identity.append((seed+i*1031,depth,rot,mirror,persona,arm))
        original,junction,tip,route=warning_world(seed+i*1031,persona,depth,rot,mirror)
        no_cue=copy.deepcopy(original)
        yes_cue=copy.deepcopy(original)
        yes_cue.issue_recovery_cue(depth+8)
        assert candidates_fingerprint(no_cue,arm)==candidates_fingerprint(yes_cue,arm)
        assert sha(sorted(no_cue.protected))==sha(sorted(yes_cue.protected))
        assert tuple(no_cue.history)==tuple(yes_cue.history)
        # Without the host's early warning exploration is permissible. The
        # exploratory step becomes a single extra material-ink decision.
        _target_and_apply(x,y,labels,no_cue,arm,'INK',tip,'unwarned-explore')
        for step in range(depth):
            _target_and_apply(x,y,labels,no_cue,arm,'BACKTRACK',None,'unwarned-recover')
        assert no_cue.pos==junction and not no_cue.history
        # With genuine future-dead-end advice, save one probe and unwind now.
        for step in range(depth-1):
            _target_and_apply(x,y,labels,yes_cue,arm,'BACKTRACK',None,('warned-switch' if step == 0 else 'warned-recover'))
        assert yes_cue.pos==junction and not yes_cue.history
        for w,tag in ((no_cue,'unwarned'),(yes_cue,'warned')):
            for p in route[:4]:
                _target_and_apply(x,y,labels,w,arm,'INK',p,tag+'-escape',mass=.92)
        # Repeated genuine false cue in a distinct, productive open geometry.
        base=false_cue_world(seed+i*1031,persona)
        base.stuck=i%3
        a=copy.deepcopy(base)
        b=copy.deepcopy(base);b.issue_recovery_cue(depth+8)
        assert candidates_fingerprint(a,arm)==candidates_fingerprint(b,arm)
        for w,tag in ((a,'no-false-cue'),(b,'reject-false-cue')):
            _target_and_apply(x,y,labels,w,arm,'INK',(7,7),tag)
    return np.asarray(x,np.float32),np.asarray(y,np.float32),labels,identity


class CueGatedPolicy(nn.Module):
    """Frozen prior policy, with a trained cue-only residual scoring adapter.

    Because the residual is multiplied by the *observed* cue flag, this head
    cannot change no-cue logits, goal pursuit or original personality. The
    cue still never modifies legality, host material authority or World.
    """
    def __init__(self, base):
        super().__init__()
        self.base=base
        for p in self.base.parameters():p.requires_grad_(False)
        self.adapter=nn.Sequential(nn.Linear(25,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        baseline=self.base(x)
        flag=x[...,23]
        return baseline+flag*self.adapter(x).squeeze(-1)


def train(initial, x, y, labels, seed=1209, epochs=14, lr=.0018):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model=CueGatedPolicy(initial)
    # Only advice-exposed states can train the adapter. Balance early positive
    # correction switches against false alarms, avoiding blind cue obedience.
    named={'warned-switch':5,'warned-recover':1,'warned-escape':1,'reject-false-cue':5}
    ids=np.asarray([i for i,z in enumerate(labels) for _ in range(named.get(z[0],0))],dtype=np.int64)
    assert len(ids) and all(x[i,0,23] == 1.0 for i in ids)
    xt=torch.from_numpy(x[ids]);yt=torch.from_numpy(y[ids])
    opt=torch.optim.AdamW(model.adapter.parameters(),lr=lr,weight_decay=.002)
    rng=np.random.default_rng(seed)
    losses=[];steps=0
    for _ in range(epochs):
        order=rng.permutation(len(ids));total=0.
        for start in range(0,len(ids),192):
            batch=order[start:start+192]
            logits=model(xt[batch]); gold=yt[batch]
            loss=-(gold*torch.log_softmax(logits,dim=1)).sum(dim=1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            steps+=1;total+=float(loss.detach())*len(batch)
        losses.append(round(total/len(ids),5))
    model.eval()
    return model,steps,losses,len(ids)


def greedy_choice(model,w,arm):
    c=w.candidates(arm)
    with torch.no_grad():scores=model(torch.tensor([z.features for z in c],dtype=torch.float32)).numpy()
    raw_bad=not c[int(np.argmax(scores))].legal
    scores=np.where([z.legal for z in c],scores,-1e4)
    return c[int(np.argmax(scores))], bool(raw_bad)


def evaluate_pairs(model, seeds=24):
    """Use identical heldout source World for every no/yes comparison.

    This is a *greedy first-decision* diagnostic, not a resident SLM or native
    A/B. Depth=4 and rotations 2/3 are withheld from supervised curriculum.
    """
    records=[]
    for i in range(seeds):
        seed=867100+i*811
        persona=-1 if i%2 else 1
        rot=2+(i%2); mirror=bool((i//2)%2)
        world,junction,tip,route=warning_world(seed,persona,4,rot,mirror)
        for arm in ARMS:
            for family in ('true_warning','false_alarm'):
                orig=world if family=='true_warning' else false_cue_world(seed,persona)
                base_hash=sha({'pos':orig.pos,'history':orig.history,'protected':sorted(orig.protected),
                               'ink':sorted(orig.inked),'goal':orig.goal,'personality':orig.personality,
                               'seed':orig.seed,'steps':orig.steps,'stuck':orig.stuck})
                pair=[]
                for cue in (False,True):
                    w=copy.deepcopy(orig)
                    if cue:w.issue_recovery_cue(9)
                    action,bad=greedy_choice(model,w,arm)
                    pair.append((w,action,bad,cue))
                assert candidates_fingerprint(pair[0][0],arm)==candidates_fingerprint(pair[1][0],arm)
                for w,choice,bad,cue in pair:
                    correct = ((choice.action[0]=='BACKTRACK') if family=='true_warning' and cue else
                               (choice.action[0]=='INK' and choice.target== (tip if family=='true_warning' else (7,7))))
                    records.append({'family':family,'seed':seed,'personality':persona,'rotation':rot,'mirror':mirror,
                       'arm':arm,'cue':cue,'world_sha256':base_hash,'legal_menu_sha256':candidates_fingerprint(w,arm),
                       'choice':choice.action,'target':choice.target,'raw_illegal_top1':bad,'oracle_first_action':bool(correct)})
    return records


def summarize_pairs(records):
    out={}
    for arm in ARMS:
        for family in ('true_warning','false_alarm'):
            grp=[r for r in records if r['arm']==arm and r['family']==family]
            a={r['seed']:r for r in grp if not r['cue']}
            b={r['seed']:r for r in grp if r['cue']}
            assert a.keys()==b.keys()
            for s in a:
                assert a[s]['world_sha256']==b[s]['world_sha256']
                assert a[s]['legal_menu_sha256']==b[s]['legal_menu_sha256']
            out[f'{arm:.2f}/{family}']={'pairs':len(a),
              'no_cue_correct':sum(a[s]['oracle_first_action'] for s in a),
              'cue_correct':sum(b[s]['oracle_first_action'] for s in a),
              'changed_action':sum((a[s]['choice']!=b[s]['choice']) for s in a),
              'desired_switch':sum((a[s]['oracle_first_action'] and b[s]['oracle_first_action']) for s in a),
              'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in grp)}
    return out


def rollout_warnings(model,seeds=24):
    """Bounded original per-turn synthetic World receipts, no native material Undo.

    Unlike the greedy first-choice test, records the actual coupled sequel:
    each successful action changes cursor/ink/history, so conditions need not
    remain same-state after turn one. Do not interpret this as a formal A/B.
    """
    records=[]
    for i in range(seeds):
        seed=867100+i*811
        personality=-1 if i%2 else 1
        rotation=2+(i%2);mirror=bool((i//2)%2)
        initial,junction,tip,route=warning_world(seed,personality,4,rotation,mirror)
        for arm in ARMS:
            for cue in (False,True):
                w=copy.deepcopy(initial)
                if cue:w.issue_recovery_cue(9)
                clean_at=None;branch_at=None;events=[]
                for turn in range(17):
                    c,bad=greedy_choice(model,w,arm)
                    assert w.apply(c), 'host rejected a candidate selected from its own legal menu'
                    events.append({'turn':turn+1,'action':c.action,'target':c.target,
                        'history_depth':len(w.history),'ink_count':len(w.inked),
                        'cue_ttl_after':w.recovery_cue_ttl,'raw_illegal_top1':bad,
                        'material_sha256':sha(sorted(w.inked))})
                    if w.pos==junction and not w.history and clean_at is None:clean_at=turn+1
                    if w.pos==route[0] and branch_at is None:branch_at=turn+1
                    if w.stopped:break
                records.append({'seed':seed,'personality':personality,'rotation':rotation,'mirror':mirror,
                    'arm':arm,'cue':cue,'initial_world_sha256':sha({'protected':sorted(initial.protected),
                       'cursor':initial.pos,'history':initial.history,'goal':initial.goal,'ink':sorted(initial.inked)}),
                    'clean_junction_turn':clean_at,'first_escape_branch_turn':branch_at,
                    'painted_cells':len(w.inked),'host_rejected_actions':w.invalid_attempts,
                    'raw_illegal_top1':sum(z['raw_illegal_top1'] for z in events),
                    'steps':len(events),'stopped':w.stopped,'events':events})
    return records


def summarize_rollouts(records):
    stats={}
    for arm in ARMS:
        for cue in (False,True):
            group=[r for r in records if r['arm']==arm and r['cue']==cue]
            done=[r['clean_junction_turn'] for r in group if r['clean_junction_turn'] is not None]
            escape=[r['first_escape_branch_turn'] for r in group if r['first_escape_branch_turn'] is not None]
            stats[f'{arm:.2f}/cue={int(cue)}']={'episodes':len(group),'clean_count':len(done),
                'mean_clean_steps_success_only':round(float(np.mean(done)),4) if done else None,
                'escape_count':len(escape),'mean_escape_steps_success_only':round(float(np.mean(escape)),4) if escape else None,
                'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in group),
                'host_rejected_actions':sum(r['host_rejected_actions'] for r in group)}
    return stats


def run(checkpoint,out_dir,train_instances=288,eval_seeds=24,epochs=14):
    torch.set_num_threads(1)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    initial=Scorer();initial.load_state_dict(load_file(str(checkpoint)));initial.eval()
    x,y,labels,ids=make_correction_data(instances=train_instances)
    before=evaluate_pairs(initial,seeds=eval_seeds)
    old_rollouts=rollout_warnings(initial,seeds=eval_seeds)
    model,steps,losses,count=train(initial,x,y,labels,epochs=epochs)
    after=evaluate_pairs(model,seeds=eval_seeds)
    new_rollouts=rollout_warnings(model,seeds=eval_seeds)
    path=out/'guidance_recovery_positive_cue_adapter.safetensors'
    metadata={'provenance':json.dumps({'schema':SCHEMA,'seed':1209,
      'base_checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()},
      sort_keys=True,separators=(',',':'))}
    save_file({k:v.detach().cpu().contiguous() for k,v in model.adapter.state_dict().items()},str(path),metadata=metadata)
    # Small comparison slice with explicit paired hashes and oracle outcomes;
    # full 576 first-decision receipts are written in the sibling artifact.
    metric={'schema':SCHEMA,'synthetic_only':True,'real_native_undo':False,
      'input_checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
      'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'dataset_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
      'labels_sha256':sha(labels),'source_ids_sha256':sha(ids),
      'training_examples':count,'paired_examples':len(x),'adapter_only':True,'frozen_base':True,'optimizer_steps':steps,'losses':losses,
      'before':summarize_pairs(before),'after':summarize_pairs(after),
      'before_sha256':sha(before),'after_sha256':sha(after),
      'before_early_warning_rollout':summarize_rollouts(old_rollouts),
      'after_early_warning_rollout':summarize_rollouts(new_rollouts),
      'before_rollouts_sha256':sha(old_rollouts),'after_rollouts_sha256':sha(new_rollouts),
      'prior_hard':summarize(evaluate_hard(initial,seeds=8)),
      'updated_hard':summarize(evaluate_hard(model,seeds=8)),
      'prior_style':aggregate(evaluation(initial,seeds=8)[0]),
      'updated_style':aggregate(evaluation(model,seeds=8)[0]),
      'torch':torch.__version__,'numpy':np.__version__}
    (out/'positive_correction_metrics.json').write_text(json.dumps(metric,sort_keys=True,indent=2)+'\n')
    (out/'positive_correction_first_action_receipts.json').write_text(json.dumps({'before':before,'after':after},sort_keys=True,indent=2)+'\n')
    (out/'positive_correction_rollout_receipts.json').write_text(json.dumps({'before':old_rollouts,'after':new_rollouts},sort_keys=True,indent=2)+'\n')
    return metric


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--train-instances',type=int,default=288)
    parser.add_argument('--eval-seeds',type=int,default=24)
    parser.add_argument('--epochs',type=int,default=14)
    args=parser.parse_args()
    print(json.dumps(run(args.checkpoint,args.out,args.train_instances,args.eval_seeds,args.epochs),sort_keys=True,indent=2))

if __name__=='__main__':main()
