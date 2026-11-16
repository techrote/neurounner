#!/usr/bin/env python3
"""TRAIN-only synthetic geodesic/recovery teacher and CPU closed-loop transfer.

Research-only. No native gameplay contracts or production checkpoint outputs.
Source baseline: closed_loop_transfer_probe.py at CyberSand PR #638 c8263253.
"""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
import zipfile

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from closed_loop_transfer_probe import (ACTIONS, DIRS, FAMILIES, SOURCE_SHA, make_world,
                                        legal_state, observation, png, train_model, sha)
from learnability_probe_scoped import features

SCHEMA = 'cybersand/synthetic-goal-recovery-curriculum-v1'
TRAIN_SEEDS = tuple(range(1000, 1120))
EVAL_SEEDS = (61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
              113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179)
BUDGET = 192


def distance_field(w):
    """Teacher-only shortest distance on (x,y,heading); never exported in prompts."""
    queue = deque(); dist = {}
    for h in range(4):
        key = (*w['goal'], h); dist[key] = 0; queue.append(key)
    blocked = w['obstacles'] | w['protected']
    while queue:
        x, y, h = queue.popleft()
        dx, dy = DIRS[h]
        prev = (x-dx, y-dy)
        if not (0 <= prev[0] < 96 and 0 <= prev[1] < 96) or prev in blocked:
            continue
        for turn in (0, -1, 1):
            key = (*prev, (h-turn) % 4)
            if key not in dist:
                dist[key] = dist[(x, y, h)] + 1
                queue.append(key)
    return dist


def oracle(w, pos, heading, visited, history, dist, legal):
    if pos == w['goal']: return 'STOP', {}
    moves, fresh, _ = legal_state(w, pos, heading, visited, history)
    scored = []
    for action, (target, nh) in moves.items():
        dd = dist.get((*target, nh), 10**6)
        scored.append((dd, int(target in visited), ACTIONS.index(action), action))
    if history and 'RETRACE' in legal:
        oldpos, oldhead = history[-1]
        # A history-return can be cheaper than navigating forward from a detour.
        olddist = dist.get((*oldpos, oldhead), 10**6)
        scored.append((olddist, 1, ACTIONS.index('RETRACE'), 'RETRACE'))
    if not scored:return 'STOP', {}
    scored.sort()
    choice = scored[0][3]
    negatives = {a: {'excess_geodesic_steps': max(0, dd-scored[0][0]),
                     'already_visited': bool(been)}
                 for dd, been, _, a in scored[1:]}
    return choice, negatives


def apply(w, pos, heading, visited, history, action):
    moves, fresh, legal = legal_state(w, pos, heading, visited, history)
    if action not in legal:raise AssertionError('host-unsafe action')
    if action == 'STOP':return pos, heading
    if action == 'RETRACE':return history.pop()
    arm = next(a for a in ('RIGHT', 'LEFT', 'STRAIGHT') if a in fresh) if action == 'BRANCH' else action
    history.append((pos, heading)); target, newh = moves[arm]
    if target in w['protected'] or target in w['obstacles']:
        raise AssertionError('protected/blocked cell entered')
    return target, newh


def teacher_data(seedset, budget=BUDGET):
    rows = []; worlds = []; hard = Counter(); labels = Counter(); counters=Counter()
    for seed in seedset:
        w = make_world(seed); distances = distance_field(w)
        worlds.append({'seed': seed, 'goal': w['goal'], 'shortest_goal': w['shortest_goal'],
                       'obstacles': len(w['obstacles']), 'protected': len(w['protected']),
                       'fingerprint': sha(json.dumps(sorted(map(list, w['obstacles'] | w['protected']))).encode())})
        for k, family in enumerate(FAMILIES):
            pos = w['start']; heading = 0; visited = {pos}; history=[]; past=[]
            rng = random.Random(seed*4567 + k*37 + 192321)
            for tick in range(budget):
                moves, fresh, legal = legal_state(w, pos, heading, visited, history)
                obs = observation(w, pos, heading, visited, past, legal, tick, budget, family)
                label, negatives = oracle(w,pos,heading,visited,history,distances,legal)
                if label not in legal: raise AssertionError('illegal oracle label')
                rows.append((obs, label, seed, negatives))
                labels[label]+=1; hard['legal_distractors']+=len(negatives)
                if label == 'RETRACE':hard['retrace_targets']+=1
                if label=='STOP':counters['stops']+=1;break
                # Host applies a bounded, seeded exploratory perturbation, then
                # the teacher labels the ACTUAL subsequent recovery observation.
                action = label
                alternatives = [a for a in moves if a != label]
                if 6 < tick < budget-12 and rng.random() < .18 and alternatives:
                    action=rng.choice(alternatives); counters['perturbations']+=1
                pos,heading=apply(w,pos,heading,visited,history,action)
                visited.add(pos);past.append(action)
                if pos == w['goal']: counters['goal_arrivals']+=1
            counters['episodes']+=1
    return rows, worlds, dict(hard), dict(labels), dict(counters)


def augmented_features(obs):
    result=features(obs, True)
    dx,dy=obs['goal_delta']; heading=obs['heading']
    # Explicit relative goal orientation rather than exposing privileged BFS.
    forward=dx*DIRS[heading][0]+dy*DIRS[heading][1]
    right=dx*DIRS[(heading+1)%4][0]+dy*DIRS[(heading+1)%4][1]
    for key,val in (('forward',forward),('right',right)):
        result['relative:'+key+':sign:'+str((val>0)-(val<0))]=1
        result['relative:'+key+':magnitude:'+str(min(12,abs(val)//5))]=1
    return result


def write_train_export(rows, worlds, destination):
    """Immutable TRAIN-only chat corpus; removes same-state cross-family duplications.

    No hidden geodesic/teacher distances or application receipts in prompts.
    Hard negatives remain a distinct supervised-analysis channel, not targets.
    """
    if destination.exists():raise FileExistsError('refuse overwrite of train artifact')
    clean = []; seen = set()
    for obs, label, seed, negatives in rows:
        normalized = {k:v for k,v in obs.items() if k != 'family'}
        key = sha(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode())
        if key in seen:continue
        seen.add(key)
        assert label in obs['legal']
        clean.append((obs,label,seed,negatives,key))
    system=('Select exactly one host-legal action for a synthetic 96x96 '
            'CyberSand-style worm. Goal-directed navigation is the objective; '
            'protected cells are immutable. Answer with one action token.')
    messages = []; provenance = []; distractors = []
    for i,(obs,label,seed,negatives,key) in enumerate(clean):
        messages.append(json.dumps({'messages':[
            {'role':'system','content':system},
            {'role':'user','content':json.dumps(obs,sort_keys=True,separators=(',',':'))},
            {'role':'assistant','content':label}]},sort_keys=True,separators=(',',':')))
        provenance.append(json.dumps({'index':i,'split':'TRAIN',
            'source_world_group_id':f'goal_recovery:{seed}',
            'observable_state_sha256':key,'label':label,
            'teacher':'synthetic-privileged-oriented-geodesic-v1',
            'evidence_class':'synthetic-host-applied-teacher-label',
            'is_native':False,'is_official_dev':False},sort_keys=True,separators=(',',':')))
        distractors.append(json.dumps({'index':i,'correct_action':label,
            'hard_negatives':negatives,'evidence_class':'synthetic-teacher-distractors'},
            sort_keys=True,separators=(',',':')))
    members={name:('\n'.join(lines)+'\n').encode() for name,lines in (
        ('train.messages.jsonl',messages),('train.provenance.jsonl',provenance),
        ('train.hard_negatives.jsonl',distractors))}
    manifest={'schema':'cybersand-goal-recovery-train-export-v1',
        'data_class':'synthetic-train-only-no-native',
        'eligible_model_training':True,
        'world_count':len(worlds),'row_count':len(clean),
        'duplicate_normalized_states_rejected':len(rows)-len(clean),
        'seed_range':[min(w['seed'] for w in worlds),max(w['seed'] for w in worlds)],
        'worlds':worlds,
        'members_sha256':{name:sha(payload) for name,payload in sorted(members.items())},
        'restrictions':['never train on eval results','privileged geodesic is teacher-only',
                        'family labels are not evidence of learned artistic style']}
    members['manifest.json']=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(members.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert all(sha(z.read(name))==d for name,d in manifest['members_sha256'].items())
    return {'artifact_name':destination.name,'sha256':sha(destination.read_bytes()),
            'rows':len(clean),'duplicates_removed':len(rows)-len(clean),
            'hard_negative_records':len(distractors),'world_groups':len(worlds)}


def fit(rows, epochs=24):
    vec=DictVectorizer(sort=True); x=vec.fit_transform(augmented_features(r[0]) for r in rows)
    mdl=SGDClassifier(loss='log_loss', alpha=.0001, max_iter=epochs, tol=None,
                      random_state=17, shuffle=True, learning_rate='optimal')
    mdl.fit(x,[r[1] for r in rows]);return vec,mdl,{'epochs':int(mdl.n_iter_),
        'optimizer_samples_seen':int(mdl.t_),'training_examples':len(rows), 'feature_count':x.shape[1]}


def decide(policy,obs,visit,vec=None,model=None,rng=None):
    if policy=='random':return rng.choice([a for a in obs['legal'] if a!='STOP'] or ['STOP'])
    feat=features(obs,True) if policy=='v1' else augmented_features(obs)
    scores=model.decision_function(vec.transform([feat]))[0]
    labels=list(model.classes_)
    return max((a for a in obs['legal'] if a in labels),
               key=lambda a:(scores[labels.index(a)],-ACTIONS.index(a)))


def rollout(w, family, policy, vec, model, budget=BUDGET):
    pos=w['start'];heading=0;visited={pos};history=[];past=[];path=[pos];
    rng=random.Random(92031+w['seed']*111+sum(map(ord,family)))
    novelty=[]; recovery=0; reversals=0; events=Counter();first_goal=None
    for tick in range(budget):
        moves,fresh,legal=legal_state(w,pos,heading,visited,history)
        obs=observation(w,pos,heading,visited,past,legal,tick,budget,family)
        action=decide(policy,obs,visited,vec,model,rng)
        if action not in legal:raise AssertionError('illegal model action')
        events[action]+=1
        if action=='STOP':break
        oldpos=pos
        pos,heading=apply(w,pos,heading,visited,history,action)
        if action=='RETRACE':reversals+=1
        isfresh=pos not in visited
        if not isfresh:recovery+=1
        novelty.append(int(isfresh));visited.add(pos);past.append(action);path.append(pos)
        if pos == w['goal'] and first_goal is None:first_goal=tick+1
    return {'policy':policy,'seed':w['seed'],'family':family,
            'turns':len(novelty),'unique_cells':len(visited),'goal_tick':first_goal,
            'revisited_steps':recovery,'retrace_steps':reversals,
            'novel_by_32':[sum(novelty[i:i+32]) for i in range(0,budget,32)],
            'actions':dict(sorted(events.items())),'path':path}


def run(v1, out, train_seeds=TRAIN_SEEDS, eval_seeds=EVAL_SEEDS, budget=BUDGET):
    if out.exists():raise FileExistsError('immutable experiment destination exists')
    if not 32<=budget<=256 or not train_seeds or not eval_seeds:raise ValueError('bounded seeds/budget')
    if set(train_seeds)&set(eval_seeds):raise ValueError('world split leakage')
    if sha(v1.read_bytes())!=SOURCE_SHA['v1']:raise ValueError('v1 train ZIP pin mismatch')
    examples,worlds,hards,labels,counters=teacher_data(train_seeds,budget)
    vec,model,fit_meta=fit(examples)
    vvec,vmodel,vfit_meta=train_model(v1,'v1')
    out.mkdir(parents=True);(out/'png').mkdir()
    export=write_train_export(examples,worlds,out/'train_only_goal_recovery.zip')
    rollouts=[];images=[];eval_worlds=[]
    for seed in eval_seeds:
        w=make_world(seed)
        eval_worlds.append({'seed':seed,'goal':w['goal'],'shortest_goal':w['shortest_goal']})
        for family in FAMILIES:
            for policy in ('recovery','v1','random'):
                a,b=(vec,model) if policy=='recovery' else (vvec,vmodel) if policy=='v1' else (None,None)
                r=rollout(w,family,policy,a,b,budget)
                if len(images)<18 and seed in eval_seeds[:2]:
                    filename=f'world{seed}-{family}-{policy}.png';pixels=png(w,r)
                    (out/'png'/filename).write_bytes(pixels)
                    images.append({'file':filename,'sha256':sha(pixels)})
                del r['path'];rollouts.append(r)
    summary={}
    for policy in ('recovery','v1','random'):
        a=[r for r in rollouts if r['policy']==policy]
        summary[policy]={'runs':len(a),'goal_arrivals':sum(r['goal_tick'] is not None for r in a),
                         'mean_unique_cells':round(sum(r['unique_cells'] for r in a)/len(a),3),
                         'mean_retrace_steps':round(sum(r['retrace_steps'] for r in a)/len(a),3),
                         'mean_novel_by_32':[round(sum(r['novel_by_32'][j] for r in a)/len(a),3)
                                               for j in range(budget//32)]}
    report={'schema':SCHEMA,'evidence_class':'external-synthetic-real-CPU-SGD-and-host-applied-inference',
            'source_pins':{'v1_train_zip_sha256':SOURCE_SHA['v1'],
                           'imported_baseline':'PR638:c8263253, closed_loop_transfer_probe.py'},
            'warning':'TRAIN-only synthetic geodesic teacher has privileged hidden-world knowledge for LABELS. No privileged teacher fields in prompts. Neither native, Sprocket/LFM nor official DEV evidence.',
            'train_split':{'world_seeds':list(train_seeds),'worlds':worlds,'examples':len(examples),
                           'labels':labels,'hard_negatives':hards,'episode_counts':counters,
                           'model':fit_meta,'export':export,'training_obs_digest':sha('\n'.join(json.dumps([o,l,s],sort_keys=True)
                                                        for o,l,s,_ in examples).encode())},
            'eval_split':{'seeds':list(eval_seeds),'worlds':eval_worlds,'budget':budget,
                          'source_class':'new seeds / same held-out procedural world generator',
                          'never_in_training':True},
            'baseline_model':vfit_meta,'summary':summary,'rollouts':rollouts,
            'original_png_inventory':images,'native_events':0,'model_artifact_promotions':0}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(args.v1,args.output)['summary'],indent=2,sort_keys=True))
