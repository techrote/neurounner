"""Game-like near-wall false-alert hardening for DRIV3R's frozen worm policy.

Research-only synthetic host. Match cue/no-cue World and legal menus; only
cue-gated adapter weights learn. Separate genuine ahead-of-dead-end warning,
misleading near-wall advice with viable forward continuation, and easy open
corridor false alarms. BACKTRACK changes cursor history, never painted material.
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
from safetensors.torch import load_file, save_file

from experiment import ARMS, Scorer, sha, evaluation, aggregate
from hard_recovery import add_supervised, false_cue_world, rotate
from positive_correction import (
    CueGatedPolicy, candidates_fingerprint, greedy_choice, warning_world,
    rollout_warnings, summarize_rollouts,
)

SCHEMA = 'guidance-recovery-adversarial-cues/1'


def near_wall_false_world(seed, personality, depth=3, rotation=0, mirror=False, shape='straight'):
    """Open the next tip's exit: danger-looking geometry is still productive.

    The original no-escape world and the altered passable version differ only
    in host material cells beyond the currently offered brush movement. The
    legal action IDs at the cursor stay the same. No permissions are inferred
    or granted by the correction model.
    """
    w, junction, tip, route = warning_world(seed, personality, depth, rotation, mirror)
    if shape == 'straight':
        extension = [(x,7) for x in range(6+depth,14)]
    elif shape == 'corner':
        extension = [(5+depth,8)] + [(x,8) for x in range(6+depth,14)] + [(13,7)]
    else:
        raise ValueError('unsupported exit shape')
    w.protected.difference_update(rotate(p,rotation,mirror) for p in extension)
    assert w.material_at(tip)=='AIR'
    assert any(c.action[0]=='INK' and c.target==tip and c.legal for c in w.candidates(0))
    # Explicitly assert a *traversable* viable continuation to the goal.
    probe=copy.deepcopy(w)
    planned=[tip]+[rotate(p,rotation,mirror) for p in extension]
    for target in planned:
        legal=[c for c in probe.candidates(0) if c.action[0]=='INK' and c.target==target and c.legal]
        assert len(legal)==1,(shape,target)
        assert probe.apply(legal[0])
    assert probe.pos==w.goal
    return w, junction, tip, planned


def decoy_true_world(seed, personality, depth=4, rotation=2, mirror=False, shape='straight'):
    """Visually near-identical to a productive exit, but that exit ends next cell.

    The near-wall candidate's one-step-adjacent clearance feature is identical
    to its productive counterpart. Without deeper map observation, no policy
    seeing only the 25-feature legal menu can distinguish the required choice.
    """
    w,junction,tip,route=warning_world(seed,personality,depth,rotation,mirror)
    extension=(6+depth,7) if shape=='straight' else (5+depth,8)
    w.protected.remove(rotate(extension,rotation,mirror))
    assert w.material_at(rotate(extension,rotation,mirror))=='AIR'
    return w,junction,tip


def source_hash(w):
    return sha({'seed':w.seed,'personality':w.personality,'pos':w.pos,
       'history':w.history,'protected':sorted(w.protected),'ink':sorted(w.inked),
       'goal':w.goal,'last_dir':w.last_dir,'stuck':w.stuck,'steps':w.steps})


def pair_view(world, arm):
    a=copy.deepcopy(world)
    b=copy.deepcopy(world);b.issue_recovery_cue(9)
    # World identity before cue is identical; cue does not modify materials,
    # cursor history, legal menus or source identity.
    assert sorted(a.protected)==sorted(b.protected)
    assert a.history==b.history and a.pos==b.pos and a.goal==b.goal
    assert candidates_fingerprint(a,arm)==candidates_fingerprint(b,arm)
    assert all(x.legal==y.legal for x,y in zip(a.candidates(arm),b.candidates(arm)))
    return a,b


def generate(seed=404221, instances=384):
    rng=random.Random(seed)
    X=[];Y=[];labels=[];sources=[]
    for i in range(instances):
        s=seed+i*719
        personality=1 if i%2==0 else -1
        depth=rng.choice((2,3));rotation=rng.randrange(4);mirror=bool(rng.randrange(2))
        arm=ARMS[i%len(ARMS)]
        # World-level split. Every supervised source appears only in TRAIN;
        # evaluations use disjoint seeds, deeper held-out stem and rotations.
        worlds=[('true',warning_world(s,personality,depth,rotation,mirror)[0],None),
                ('straight_false',near_wall_false_world(s+100000,personality,depth,rotation,mirror,'straight')[0],None),
                ('corner_false',near_wall_false_world(s+200000,personality,depth,rotation,mirror,'corner')[0],None),
                ('open_false',false_cue_world(s+300000,personality),None)]
        for family,w,_ in worlds:
            target = (7,7) if family=='open_false' else None
            # For true/near-wall sources, use the precise ahead-of-cursor tip,
            # selected from identical source geometry without model influence.
            if family!='open_false':
                backwards = w.history[-1]
                tip_candidates=[c for c in w.candidates(0) if c.action[0]=='INK' and c.legal
                    and c.target!=backwards]
                assert len(tip_candidates)==1,(family,w.pos,[z.target for z in tip_candidates])
                target=tip_candidates[0].target
            no,yes=pair_view(w,arm)
            sources.append({'source':source_hash(w),'seed':w.seed,'family':family,'arm':arm,
                            'depth':depth,'rotation':rotation,'mirror':mirror})
            for cue,view in ((False,no),(True,yes)):
                kind='BACKTRACK' if family=='true' and cue else 'INK'
                chosen=add_supervised(X,Y,view,arm,kind,None if kind=='BACKTRACK' else target,mass=.98)
                labels.append({'family':family,'cue':cue,'kind':kind,'source':source_hash(w),
                               'action':chosen.action,'target':chosen.target})
    x=np.asarray(X,dtype=np.float32);y=np.asarray(Y,dtype=np.float32)
    assert x.shape==(instances*8,10,25) and y.shape==(instances*8,10)
    return x,y,labels,sources


def load_prior(base_path,adapter_path):
    base=Scorer();base.load_state_dict(load_file(str(base_path)));base.eval()
    model=CueGatedPolicy(base)
    model.adapter.load_state_dict(load_file(str(adapter_path)))
    model.eval()
    return model


def tune(model,x,y,labels,epochs=14,seed=2207,lr=.001):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    # Prior only updates cue-on residual; draw personality and unguided head
    # cannot be altered. 2x true positive corrections, 2x near-wall controls.
    count={'true':3,'straight_false':3,'corner_false':3,'open_false':1}
    ids=np.asarray([i for i,v in enumerate(labels) if v['cue']
                    for _ in range(count[v['family']])],dtype=np.int64)
    assert ids.size and np.all(x[ids,:,23]==1)
    xt=torch.from_numpy(x[ids]);yt=torch.from_numpy(y[ids])
    length=len(xt)
    opt=torch.optim.AdamW(model.adapter.parameters(),lr=lr,weight_decay=.003)
    rng=np.random.default_rng(seed);losses=[];steps=0
    for _ in range(epochs):
        order=rng.permutation(length);total=0.
        for start in range(0,length,192):
            batch=order[start:start+192]
            logit=model(xt[batch])
            loss=-(yt[batch]*torch.log_softmax(logit,dim=1)).sum(dim=1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),2.)
            opt.step();opt.zero_grad(set_to_none=True)
            steps+=1;total+=float(loss.detach())*len(batch)
        losses.append(round(total/length,6))
    model.eval()
    return model,steps,length,losses


def evaluate(model,seeds=32):
    records=[]
    for i in range(seeds):
        seed=800001+i*977
        personality=1 if i%2==0 else -1
        rotation=2+i%2;mirror=bool((i//2)%2)
        for arm in ARMS:
            scenarios=[('true',warning_world(seed,personality,4,rotation,mirror)[0]),
                       ('straight_false',near_wall_false_world(seed+100000,personality,4,rotation,mirror,'straight')[0]),
                       ('corner_false',near_wall_false_world(seed+200000,personality,4,rotation,mirror,'corner')[0]),
                       ('open_false',false_cue_world(seed+300000,personality)),
                       ('decoy_straight_true',decoy_true_world(seed+100000,personality,4,rotation,mirror,'straight')[0]),
                       ('decoy_corner_true',decoy_true_world(seed+200000,personality,4,rotation,mirror,'corner')[0])]
            # Deliberately impossible-to-resolve examples: a hidden passage
            # ends one cell farther than the 25-feature one-step clearance.
            # Identical candidate observations but conflicting oracle labels.
            by_name=dict(scenarios)
            for normal,decoy in (('straight_false','decoy_straight_true'),
                                 ('corner_false','decoy_corner_true')):
                for cue in (False,True):
                    x=copy.deepcopy(by_name[normal]);y=copy.deepcopy(by_name[decoy])
                    if cue:x.issue_recovery_cue(9);y.issue_recovery_cue(9)
                    assert candidates_fingerprint(x,arm)==candidates_fingerprint(y,arm)
                    assert np.array_equal(np.asarray([c.features for c in x.candidates(arm)]),
                                          np.asarray([c.features for c in y.candidates(arm)]))
                    assert source_hash(x)!=source_hash(y)
            for family,w in scenarios:
                a,b=pair_view(w,arm)
                for cue,v in ((False,a),(True,b)):
                    c,raw_illegal=greedy_choice(model,v,arm)
                    if family=='open_false': expected='INK';expected_target=(7,7)
                    else:
                        expected='BACKTRACK' if (family=='true' or family.startswith('decoy_')) and cue else 'INK'
                        expected_target=next(z.target for z in v.candidates(0) if z.action[0]=='INK'
                                         and z.legal and z.target!=v.history[-1]) if expected=='INK' else None
                    correct=(c.action[0]==expected and (expected_target is None or c.target==expected_target))
                    records.append({'family':family,'seed':seed,'depth':4,'rotation':rotation,'mirror':mirror,
                      'personality':personality,'arm':arm,'cue':cue,
                      'world_sha256':source_hash(w),'legal_menu_sha256':candidates_fingerprint(v,arm),
                      'action':c.action,'target':c.target,'correct_first_action':bool(correct),
                      'raw_illegal_top1':raw_illegal})
    return records


def summarize(records):
    stats={}
    for family in ('true','straight_false','corner_false','open_false','decoy_straight_true','decoy_corner_true'):
        for arm in ARMS:
            subset=[r for r in records if r['family']==family and r['arm']==arm]
            false=[r for r in subset if not r['cue']];true=[r for r in subset if r['cue']]
            assert len(false)==len(true)
            a={r['seed']:r for r in false};b={r['seed']:r for r in true}
            for s in a:
                assert a[s]['world_sha256']==b[s]['world_sha256']
                assert a[s]['legal_menu_sha256']==b[s]['legal_menu_sha256']
            stats[f'{family}/{arm:.2f}']={'paired':len(a),
              'no_cue_correct':sum(a[s]['correct_first_action'] for s in a),
              'cue_correct':sum(b[s]['correct_first_action'] for s in a),
              'changed_action':sum(a[s]['action']!=b[s]['action'] for s in a),
              'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in subset)}
    return stats


def run(base,adapter,out,instances=384,seeds=32,epochs=14):
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    original=load_prior(base,adapter)
    X,Y,labels,sources=generate(instances=instances)
    before=evaluate(original,seeds)
    updated=load_prior(base,adapter)
    updated,steps,n,loss=tune(updated,X,Y,labels,epochs=epochs)
    after=evaluate(updated,seeds)
    path=out/'guidance_recovery_adversarial_adapter.safetensors'
    metadata={'provenance':json.dumps({'schema':SCHEMA,'seed':2207,
                'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),
                'prior_adapter_sha256':hashlib.sha256(Path(adapter).read_bytes()).hexdigest()},
                sort_keys=True,separators=(',',':'))}
    save_file({k:v.detach().cpu().contiguous() for k,v in updated.adapter.state_dict().items()},
              str(path),metadata=metadata)
    style_before=aggregate(evaluation(original,seeds=8)[0])
    style_after=aggregate(evaluation(updated,seeds=8)[0])
    original_rollouts=rollout_warnings(original,seeds=min(12,seeds))
    updated_rollouts=rollout_warnings(updated,seeds=min(12,seeds))
    metrics={'schema':SCHEMA,'synthetic_only':True,'real_material_undo':False,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),
             'prior_adapter_sha256':hashlib.sha256(Path(adapter).read_bytes()).hexdigest(),
             'updated_adapter_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'dataset_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
             'labels_sha256':sha(labels),'sources_sha256':sha(sources),'train_instances':instances,
             'training_examples':n,'optimizer_steps':steps,'losses':loss,
             'before':summarize(before),'after':summarize(after),
             'indistinguishable_decoy_pairs_per_arm':2*seeds,
             'deceptive_decoy_trained':False,
             'before_first_action_sha256':sha(before),'after_first_action_sha256':sha(after),
             'style_before':style_before,'style_after':style_after,
             'before_true_warning_rollouts':summarize_rollouts(original_rollouts),
             'after_true_warning_rollouts':summarize_rollouts(updated_rollouts),
             'before_rollouts_sha256':sha(original_rollouts),'after_rollouts_sha256':sha(updated_rollouts),
             'numpy':np.__version__,'torch':torch.__version__}
    (out/'adversarial_metrics.json').write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    (out/'adversarial_first_action_receipts.json').write_text(json.dumps({'before':before,'after':after},sort_keys=True,indent=2)+'\n')
    (out/'adversarial_true_warning_rollouts.json').write_text(json.dumps({'before':original_rollouts,'after':updated_rollouts},sort_keys=True,indent=2)+'\n')
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--instances',type=int,default=384)
    p.add_argument('--seeds',type=int,default=32)
    p.add_argument('--epochs',type=int,default=14)
    a=p.parse_args()
    print(json.dumps(run(a.base,a.adapter,a.out,a.instances,a.seeds,a.epochs),sort_keys=True,indent=2))

if __name__=='__main__':main()
