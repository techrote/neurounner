#!/usr/bin/env python3
"""Independent, executable synthetic CyberSand-like *closed-loop* style corpus.

Host-legal paint/branch/retrace actions actually alter a seeded 96x96 world.
Every row is synthetic heuristic supervision; none is native, model inference,
or evidence of model training. Only TRAIN rows may be used as targets.
No imports from, or modifications to, game/runtime files; stdlib only.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import zlib

SCHEMA = 'cybersand.driv3r.game-corpus/family-closed-loop-1'
EVIDENCE = 'synthetic-host-applied-family-steering-not-native-or-model'
ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'STOP')
FAMILIES = ('greebles', 'pipework', 'architecture')
SCENARIOS = ('corridor', 'deceptive', 'protected', 'deadend', 'motif', 'long_history')
SPLITS = ('train', 'dev', 'heldout')
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def canonical(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=True, separators=(',', ':'))


def digest(v):
    return hashlib.sha256(canonical(v).encode('utf-8')).hexdigest()


def seed_for(*items):
    return int(digest(items)[:16], 16)


def split_for(group):
    v = seed_for('corpus-split-v1', group) % 100
    return 'train' if v < 80 else 'dev' if v < 90 else 'heldout'


def make_world(seed, scenario):
    """New named synthetic variant; not a native material readback or v1 replay."""
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    r = random.Random(seed_for('family-closed-loop-world-v1', seed, scenario))
    size = 96
    mid = 48
    start = (7, mid)
    obstacles = set()
    for _ in range(8 + seed % 8):
        w, h = r.randint(2, 11), r.randint(2, 10)
        x, y = r.randint(13, 92-w), r.randint(3, 92-h)
        obstacles.update((xx, yy) for yy in range(y, y+h) for xx in range(x, x+w))
    if scenario in ('corridor', 'deceptive', 'deadend'):
        for x in range(17, 35):
            obstacles.add((x, mid-2))
            obstacles.add((x, mid+2))
        for y in range(mid-2, mid+3):
            obstacles.add((35, y))
        obstacles.discard((21, mid-2))
    if scenario == 'motif':
        for xx in range(28, 84, 8):
            for yy in range(14, 83, 15):
                if r.random() < .5:
                    obstacles.update((xx+a, yy+b) for a in range(2) for b in range(5))
    protected = set()
    if scenario in ('protected', 'long_history'):
        for x in range(25, 70):
            for y in range(mid-16, mid+17):
                if (x*3+y+seed) % 13 == 0:
                    protected.add((x,y))
    safe = {(x, y) for x in range(4, 14) for y in range(mid-5, mid+6)}
    obstacles.difference_update(safe)
    protected.difference_update(safe)
    protected.difference_update(obstacles)
    materials = {}
    for y in range(size):
        for x in range(size):
            p = (x, y)
            if p in obstacles:
                m = 'stone'
            elif p in protected:
                m = 'metal'
            else:
                band = (x//8 + y//13 + seed % 7) % 10
                m = 'sand' if band < 4 else 'water' if band == 4 else 'empty'
            if m != 'empty':
                materials[p] = m
    # A reachable distant goal, verified on this world's *actual* occupancy.
    distances = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS:
            p = (x+dx, y+dy)
            if (0 <= p[0] < size and 0 <= p[1] < size and
                    p not in obstacles and p not in protected and p not in distances):
                distances[p] = distances[(x,y)] + 1
                q.append(p)
    choices = [p for p, d in distances.items() if d >= (55 if scenario == 'long_history' else 30)]
    if not choices:
        raise RuntimeError('unreachable synthetic world')
    choices.sort(key=lambda p: (abs(p[0]-72)+abs(p[1]-mid), p))
    goal = r.choice(choices[:min(80,len(choices))])
    difficulty = ('hard' if scenario in ('deadend','protected','long_history') or distances[goal] >= 90
                  else 'medium' if scenario in ('deceptive','motif') or distances[goal] >= 65 else 'easy')
    world = {'schema': SCHEMA, 'seed': seed, 'scenario':scenario, 'size':size,
             'difficulty':difficulty,'reachable_goal_shortest_path_steps':distances[goal],
             'start':list(start), 'goal':list(goal),
             'obstacles':[list(p) for p in sorted(obstacles)],
             'protected':[list(p) for p in sorted(protected)],
             'material_sha256':digest([materials.get((x,y),'empty') for y in range(size) for x in range(size)]),
             'source':'external-fully-synthetic-96x96-family-closed-loop-1'}
    return world, obstacles, protected, materials


def legal_moves(pos, heading, visited, history, obstacles, protected, size=96):
    moves = {}
    for name,rot in (('STRAIGHT',0),('LEFT',-1),('RIGHT',1)):
        d=(heading+rot)%4
        p=(pos[0]+DIRS[d][0], pos[1]+DIRS[d][1])
        if 0<=p[0]<size and 0<=p[1]<size and p not in obstacles and p not in protected:
            moves[name]=(p,d)
    fresh={a for a,(p,_) in moves.items() if p not in visited}
    legal=list(moves)
    if len(fresh)>=2 and len(history)<2048:
        legal.append('BRANCH')
    if history:
        legal.append('RETRACE')
    legal.append('STOP')
    return moves, fresh, legal


def local_observation(pos, heading, visited, obstacles, protected, mats, legal, last, family, tick, budget, goal):
    symbolic=[]; material=[]; immutable=[]
    for dy in range(-2,3):
        s=''; m=[]; p=''
        for dx in range(-2,3):
            cell=(pos[0]+dx,pos[1]+dy)
            out=not(0<=cell[0]<96 and 0<=cell[1]<96)
            s+=('@' if dx==dy==0 else '#' if out or cell in obstacles or cell in protected else
                '+' if cell in visited else '.')
            m.append('boundary' if out else mats.get(cell,'empty'))
            p+='1' if cell in protected else '0'
        symbolic.append(s);material.append(m);immutable.append(p)
    return {'x':pos[0],'y':pos[1],'heading':heading,'local_5x5':symbolic,
            'local_material_5x5':material,'local_protected_5x5':immutable,
            'legal':legal,'last_actions':last[-4:],'family':family,'tick':tick,
            'budget_remaining':budget-tick,'goal_delta':[goal[0]-pos[0],goal[1]-pos[1]],
            'goal_reached':pos==goal}


def style_policy(family, tick, legal, moves, fresh, pos, visited, goal, last, rng):
    """Softmax on host-legal actions, with distinct family motifs and recovery.

    This is a deliberately imperfect, stochastic *procedural teacher*, not an
    architecture planner or a learned policy; no target leakage into observations.
    """
    scores={}
    for action in legal:
        if action == 'STOP':
            score = -1000.0  # budget exhaustion ends the rollout externally
        elif action == 'RETRACE':
            score = 18.0 if not fresh else -16.0
            if last[-3:].count('RETRACE') >= 2:
                score -= 4.0
        else:
            arm = (next(a for a in ('RIGHT','LEFT','STRAIGHT') if a in fresh)
                   if action=='BRANCH' else action)
            p,_=moves[arm]
            dist=abs(goal[0]-p[0])+abs(goal[1]-p[1])
            score = (4.8 if p not in visited else -10.0) - .035*dist
            if family == 'greebles':
                score += {'STRAIGHT':-.8,'LEFT':5.0,'RIGHT':5.0,'BRANCH':7.5}[action]
                if action == ('LEFT' if tick%6<3 else 'RIGHT'):
                    score += 2.2
            elif family == 'pipework':
                score += {'STRAIGHT':8.8,'LEFT':-1.3,'RIGHT':-1.3,'BRANCH':-4.2}[action]
                if tick%21 in (0,1,2) and action in ('LEFT','RIGHT'):
                    score += 6.2
            else:  # architecture: terraced turns + occasional supporting branches
                phase=(tick//9)%4
                preferred=('STRAIGHT','LEFT','STRAIGHT','RIGHT')[phase]
                score += 8.5 if action==preferred else -1.2
                if action=='BRANCH':
                    score += 5.0 if tick%18 in (0,1,2,3) else -4.5
                if action=='STRAIGHT' and tick%9 in (4,5,6):
                    score += 2.2
        # Independent yet deterministic jitter makes this a genuine seeded walk.
        scores[action]=score + rng.uniform(-1.7,1.7)
    non_stop=[a for a in legal if a!='STOP']
    if not non_stop:
        return 'STOP', scores
    highest=max(scores[a] for a in non_stop)
    weights=[math.exp(max(-40.0, min(40.0,(scores[a]-highest)/2.15))) for a in non_stop]
    pick=rng.random()*sum(weights)
    for a,weight in zip(non_stop,weights):
        pick-=weight
        if pick<=0:
            return a, scores
    return non_stop[-1], scores


def rollout(seed, scenario, family, budget=192, steering_family=None):
    if family not in FAMILIES or not 32<=budget<=4096:
        raise ValueError('family or budget out of range')
    control_family = steering_family or family
    if control_family not in FAMILIES:
        raise ValueError('unknown controlling style')
    w,obstacles,protected,mats=make_world(seed, scenario)
    group=f'{scenario}:{seed}'
    split=split_for(group)
    wid=digest(w)[:24]
    eid=digest([SCHEMA, group, family, budget])[:24]
    pos=tuple(w['start']);heading=0; visited={pos}; stack=[]
    last=[]; trail=[pos]; branches=set();rows=[]; goal=tuple(w['goal'])
    r=random.Random(seed_for('closed-loop-family-teacher', seed, scenario, budget))
    for tick in range(budget):
        moves,fresh,legal=legal_moves(pos,heading,visited,stack,obstacles,protected)
        obs=local_observation(pos,heading,visited,obstacles,protected,mats,legal,last,family,tick,budget,goal)
        if tick == budget-1:
            action, scores = 'STOP', {a:(1.0 if a=='STOP' else -1.0) for a in legal}
        else:
            action, scores=style_policy(control_family,tick,legal,moves,fresh,pos,visited,goal,last,r)
        if action not in legal:
            raise AssertionError('illegal teacher choice')
        prior=(pos,heading)
        before=pos
        if action=='STOP':
            pass
        elif action=='RETRACE':
            pos,heading=stack.pop()
        else:
            arm=next(a for a in ('RIGHT','LEFT','STRAIGHT') if a in fresh) if action=='BRANCH' else action
            if action=='BRANCH':
                branches.add(pos)
            stack.append(prior)
            pos,heading=moves[arm]
            visited.add(pos)
        last.append(action);trail.append(pos)
        receipt={'selected':action,'applied':action,'raw_legal':True,
                 'position_before':list(before),'position_after':list(pos),
                 'host_emulation':'external-synthetic-not-native',
                 'stop_reason':('budget_bound' if action=='STOP' and tick==budget-1 else None)}
        signature=digest([obs['local_5x5'],obs['local_material_5x5'],obs['local_protected_5x5'],
                          obs['heading'],obs['goal_delta'],obs['family'],obs['legal'],obs['last_actions']])
        neg=[{'action':a,'kind':'host_illegal'} for a in ACTIONS if a not in legal]
        neg += [{'action':a,'kind':('heuristic_dispreferred_legal' if scores[a] <= scores[action]-3 else 'legal_not_chosen')}
                for a in legal if a!=action][:2]
        rows.append({'schema':SCHEMA,'record_id':f'{eid}:{tick:04d}', 'world_id':wid,
                     'world_group_id':group,'episode_id':eid,'split':split,
                     'evidence_class':EVIDENCE,'scenario':scenario,'family':family,
                     'observation':obs, 'input_sha256':signature,
                     'target_action':action,'target_origin':'seeded-synthetic-closed-loop-policy',
                     'hard_negatives':neg,'receipt':receipt})
        if action=='STOP':
            break
    w.update({'world_id':wid,'world_group_id':group,'split':split})
    ep={'schema':SCHEMA,'episode_id':eid,'world_id':wid,'world_group_id':group,
        'split':split,'scenario':scenario,'family':family,'seed':seed,
        'step_budget':budget,'steps':len(rows),'trail':[list(x) for x in trail],
        'branch_points':[list(x) for x in sorted(branches)],
        'unique_trail_cells':len(visited), 'terminated_by_stop':bool(last and last[-1]=='STOP'),
        'goal_visited':goal in visited,'retraces':last.count('RETRACE'),
        'evidence_class':EVIDENCE}
    return w,ep,rows


def image_bytes(w,ep,scale=2):
    obstacles={tuple(p) for p in w['obstacles']}
    protected={tuple(p) for p in w['protected']}
    trail={tuple(p) for p in ep['trail']}
    branches={tuple(p) for p in ep['branch_points']}
    tip=tuple(ep['trail'][-1]);goal=tuple(w['goal'])
    payload=bytearray()
    for y in range(96):
        line=bytearray()
        for x in range(96):
            p=(x,y)
            rgb=((250,211,68) if p==tip else (88,236,115) if p==goal else
                 (236,105,210) if p in branches else (69,75,90) if p in obstacles else
                 (216,75,69) if p in protected else (77,181,231) if p in trail else (15,20,30))
            line.extend(bytes(rgb)*scale)
        for _ in range(scale):
            payload.append(0);payload.extend(line)
    def chunk(name,data):
        return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',96*scale,96*scale,8,2,0,0,0))+
            chunk(b'IDAT',zlib.compress(payload,9))+chunk(b'IEND',b''))


def write_jsonl(path,rows):
    data=(''.join(canonical(x)+'\n' for x in rows)).encode()
    path.write_bytes(data)
    return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'rows':len(rows)}


def produce(out, worlds_per_scenario=10, budget=192, image_worlds=2):
    if not 10<=worlds_per_scenario<=1000 or not 32<=budget<=4096 or not 0<=image_worlds<=20:
        raise ValueError('unbounded generation')
    if out.exists():
        raise FileExistsError('output must be new/immutable')
    quotas={'train':worlds_per_scenario-2*(worlds_per_scenario//10),
            'dev':worlds_per_scenario//10, 'heldout':worlds_per_scenario//10}
    ws={s:[] for s in SPLITS}; eps={s:[] for s in SPLITS}; decisions={s:[] for s in SPLITS}
    images={}; paired=[]; groups=set(); dup=Counter(); appearance=Counter()
    for scenario in SCENARIOS:
        counts=Counter();attempts=0; per_scenario_images=0
        while any(counts[s]<quotas[s] for s in SPLITS):
            attempts+=1
            if attempts>worlds_per_scenario*500:
                raise RuntimeError('split quotas cannot be populated')
            seed=seed_for('closed-loop-world-sequence',scenario,attempts)%1_000_000_007
            group=f'{scenario}:{seed}'
            split=split_for(group)
            if counts[split]>=quotas[split] or group in groups:
                continue
            family_runs=[rollout(seed,scenario,f,budget) for f in FAMILIES]
            # State occupancy alone, not differently colored tips/branches, defines divergence.
            masks=[{tuple(p) for p in ep['trail']} for _,ep,_ in family_runs]
            distinct=len({digest(sorted(m)) for m in masks})
            overlaps=[]
            for i,j in ((0,1),(0,2),(1,2)):
                overlaps.append(round(len(masks[i]&masks[j])/max(1,len(masks[i]|masks[j])),5))
            ablation=[rollout(seed,scenario,f,budget,steering_family='pipework')[1] for f in FAMILIES]
            ablated_distinct=len({digest(sorted({tuple(p) for p in ep['trail']})) for ep in ablation})
            if ablated_distinct!=1:
                raise AssertionError('label-only ablation changed host-applied footprints')
            paired.append({'ablation_fixed_pipework_distinct_footprints':ablated_distinct,
                           'world_group_id':group,'split':split,'scenario':scenario,
                           'distinct_footprints':distinct,'mean_pairwise_jaccard':round(sum(overlaps)/3,5),
                           'new_cells_by_family':{f:len(m) for f,m in zip(FAMILIES,masks)}})
            if per_scenario_images < image_worlds:
                for w,ep,_ in family_runs:
                    filename=f'{split}_{scenario}_{seed}_{ep["family"]}.png'
                    images[filename]=image_bytes(w,ep)
                per_scenario_images+=1
            counts[split]+=1;groups.add(group)
            w=family_runs[0][0];ws[split].append(w)
            for _,ep,rows in family_runs:
                eps[split].append(ep);decisions[split].extend(rows)
    # Identical prompt-visible contexts across splits are NOT allowed into training.
    seen={}; remove={s:set() for s in SPLITS}
    for s in SPLITS:
        for i,row in enumerate(decisions[s]):
            key=row['input_sha256']
            if key in seen and seen[key]!=s:
                # Retain DEV/heldout original receipts for sequence reproducibility;
                # quarantine the collision as unsuitable for cross-split scoring.
                remove[s].add(i)
                dup['cross_split_observable_collision']+=1
            else:
                seen[key]=s
    out.mkdir(parents=True)
    receipts={}
    for s in SPLITS:
        receipts[f'{s}.worlds.jsonl']=write_jsonl(out/f'{s}.worlds.jsonl',ws[s])
        receipts[f'{s}.episodes.jsonl']=write_jsonl(out/f'{s}.episodes.jsonl',eps[s])
        # Unfiltered original full sequences have a separate ledger.
        receipts[f'{s}.receipts.jsonl']=write_jsonl(out/f'{s}.receipts.jsonl',decisions[s])
        cleaned=[row for i,row in enumerate(decisions[s]) if i not in remove[s]]
        if s=='train':
            # Also remove within-TRAIN exact duplicates for supervised export.
            dedup=set();safe=[]
            for row in cleaned:
                if row['input_sha256'] in dedup:
                    dup['same_split_duplicate_train_input']+=1
                    continue
                dedup.add(row['input_sha256']);safe.append(row)
            cleaned=safe
        eligible_name=('train.eligible.jsonl' if s=='train' else f'{s}.evaluation-eligible.jsonl')
        receipts[eligible_name]=write_jsonl(out/eligible_name,cleaned)
    receipts['paired-footprints.jsonl']=write_jsonl(out/'paired-footprints.jsonl',paired)
    for filename,data in images.items():
        (out/filename).write_bytes(data)
        receipts[filename]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    manifest={'schema':SCHEMA,'source':'self-contained-new-synthetic-variant-not-game-or-v1-replay',
              'generator_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'read_only_reference':{'corpus_v1_git_blob':'ade2605097cde034476fd842ca5d170ad5dc9959',
                                     'source':'existing external corpus.py, not native gameplay'},
              'contract':'96x96 host-legal-action-loop compatible W0-style experiment; NO native execution',
              'parameters':{'worlds_per_scenario':worlds_per_scenario,'budget':budget,
                            'image_worlds_per_scenario':image_worlds},
              'evidence_class':EVIDENCE,'trainer_boundary':'TRAIN eligible only; DEV/heldout evaluation only; never prompt with receipts, targets or negatives',
              'counts':{'world_groups':len(groups),'episodes':sum(len(eps[s]) for s in SPLITS),
                        'full_decisions':{s:len(decisions[s]) for s in SPLITS},
                        'eligible_decisions':{s:receipts['train.eligible.jsonl' if s=='train' else f'{s}.evaluation-eligible.jsonl']['rows'] for s in SPLITS},
                        'split_groups':{s:len(ws[s]) for s in SPLITS},
                        'difficulty_bands':dict(Counter(w['difficulty'] for s in SPLITS for w in ws[s])),
                        'family_footprints':dict(Counter(p['distinct_footprints'] for p in paired)),
                        'label_only_ablation_footprints':dict(Counter(p['ablation_fixed_pipework_distinct_footprints'] for p in paired)),
                        'duplicate_controls':dict(dup),'pngs':len(images)},
              'files':receipts,'claims':{'native_steps':0,'model_updates':0,'host_legal_closed_loop_steering':True}}
    (out/'manifest.json').write_text(canonical(manifest)+'\n')
    return manifest


def audit(out, replay=True):
    m=json.loads((out/'manifest.json').read_text())
    for filename,record in m['files'].items():
        raw=(out/filename).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['sha256'] or len(raw)!=record['bytes']:
            raise ValueError('source integrity failure: '+filename)
        if filename.endswith('.jsonl') and len(raw.splitlines())!=record['rows']:
            raise ValueError('source row-count integrity failure: '+filename)
    groups={}; input_by_split={}; episodes_by_group={}
    for split in SPLITS:
        worlds=[json.loads(s) for s in (out/f'{split}.worlds.jsonl').read_text().splitlines()]
        episodes=[json.loads(s) for s in (out/f'{split}.episodes.jsonl').read_text().splitlines()]
        receipts=[json.loads(s) for s in (out/f'{split}.receipts.jsonl').read_text().splitlines()]
        eligible_file=('train.eligible.jsonl' if split=='train' else f'{split}.evaluation-eligible.jsonl')
        eligible=[json.loads(s) for s in (out/eligible_file).read_text().splitlines()]
        worlds_by_id={w['world_id']:w for w in worlds}
        if len(worlds_by_id)!=len(worlds):
            raise ValueError('duplicate world-id ledger')
        for w in worlds:
            group=w['world_group_id']
            if group in groups or split_for(group)!=split:
                raise ValueError('world split leakage')
            groups[group]=split
        receipt_by_id={}
        for row in receipts:
            if row['record_id'] in receipt_by_id:
                raise ValueError('duplicate receipt identifier')
            receipt_by_id[row['record_id']]=row
            if row['target_action'] not in row['observation']['legal'] or not row['receipt']['raw_legal']:
                raise ValueError('illegal action receipt')
            if row['split']!=split or groups.get(row['world_group_id'])!=split:
                raise ValueError('receipt split mislabel')
        for row in eligible:
            original=receipt_by_id.get(row['record_id'])
            if original is None or canonical(row)!=canonical(original):
                raise ValueError('eligible row not present identically in source receipts')
            if row['split']!=split or groups.get(row['world_group_id'])!=split:
                raise ValueError('eligible split mislabel')
            key=row['input_sha256']
            if key in input_by_split and input_by_split[key]!=split:
                raise ValueError('cross-split input leakage')
            if split=='train' and key in input_by_split:
                raise ValueError('duplicate TRAIN prompt')
            input_by_split[key]=split
        if replay:
            source={}
            for ep in episodes:
                w,e,rows=rollout(ep['seed'],ep['scenario'],ep['family'],ep['step_budget'])
                if canonical(e)!=canonical(ep):
                    raise ValueError('replay mismatch: episode')
                if canonical(w)!=canonical(worlds_by_id.get(e['world_id'])):
                    raise ValueError('replay mismatch: world')
                source[e['episode_id']]=rows
                episodes_by_group.setdefault(e['world_group_id'],{})[e['family']]=e
            recorded={}
            for row in receipts:
                recorded.setdefault(row['episode_id'],[]).append(row)
            for eid,rows in source.items():
                if canonical(rows)!=canonical(recorded.get(eid,[])):
                    raise ValueError('replay mismatch: receipt')
    if replay:
        pair_rows=[json.loads(s) for s in (out/'paired-footprints.jsonl').read_text().splitlines()]
        if len(pair_rows)!=len(groups):
            raise ValueError('paired footprint group count mismatch')
        verified=set()
        for pair in pair_rows:
            group=pair['world_group_id']
            if group in verified or groups.get(group)!=pair['split']:
                raise ValueError('paired footprint split or repetition mismatch')
            verified.add(group)
            episodes=episodes_by_group.get(group,{})
            if set(episodes)!=set(FAMILIES):
                raise ValueError('paired footprint missing family')
            masks=[{tuple(p) for p in episodes[f]['trail']} for f in FAMILIES]
            distinct=len({digest(sorted(m)) for m in masks})
            overlaps=[round(len(masks[i]&masks[j])/max(1,len(masks[i]|masks[j])),5)
                      for i,j in ((0,1),(0,2),(1,2))]
            if (pair['distinct_footprints']!=distinct or
                pair['new_cells_by_family']!={f:len(m) for f,m in zip(FAMILIES,masks)} or
                pair['mean_pairwise_jaccard']!=round(sum(overlaps)/3,5) or
                pair['ablation_fixed_pipework_distinct_footprints']!=1):
                raise ValueError('replay mismatch: paired footprint')
    return {'audit':'PASS','world_groups':len(groups),'distinct_eligible_inputs':len(input_by_split),
            'source_replay':replay,'files':len(m['files'])}


def main():
    p=argparse.ArgumentParser(description='Real synthetic closed-loop family drawings, not native/game changes')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worlds-per-scenario',type=int,default=10)
    p.add_argument('--budget',type=int,default=192)
    p.add_argument('--image-worlds',type=int,default=2)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--export-train',type=Path,help='new destination for replay-audited TRAIN-only messages')
    p.add_argument('--per-episode',type=int,default=16)
    a=p.parse_args()
    if a.export_train is not None:
        print(canonical(export_train_only(a.output,a.export_train,a.per_episode)))
        return
    result=audit(a.output) if a.audit_only else produce(a.output,a.worlds_per_scenario,a.budget,a.image_worlds)
    print(canonical(result if a.audit_only else {'counts':result['counts'],'audit':audit(a.output)}))


# Separate, explicit trainer entry point (not invoked by corpus generation).
# TRAIN source rows deliberately retain receipts for science; those fields
# MUST NOT appear in model prompt content. DEV/heldout remain read-only.
OBS_KEYS = frozenset(('x','y','heading','local_5x5','local_material_5x5',
                      'local_protected_5x5','legal','last_actions','family',
                      'tick','budget_remaining','goal_delta','goal_reached'))
SYSTEM = ('Choose exactly one action from the host legal menu in the supplied '
          'synthetic CyberSand-like observation; output only the action name. '
          'Protected cells and obstacles must not be changed.')


def export_train_only(source_root, destination, per_episode=16):
    """Make compact, prompt-safe, provenance-pinned TRAIN messages after replay."""
    if not 1 <= per_episode <= 256:
        raise ValueError('unbounded export')
    if destination.exists():
        raise FileExistsError('training export is immutable; use a new destination')
    audit(source_root, replay=True)
    manifest_raw=(source_root/'manifest.json').read_bytes()
    data=(source_root/'train.eligible.jsonl').read_text().splitlines()
    groups={}
    for line in data:
        row=json.loads(line)
        if row['split']!='train' or row['evidence_class']!=EVIDENCE:
            raise ValueError('non-TRAIN data in TRAIN source')
        observation=row['observation']
        if set(observation)!=OBS_KEYS or row['target_action'] not in observation['legal']:
            raise ValueError('unapproved observation or target')
        groups.setdefault(row['episode_id'],[]).append(row)
    messages=[]; provenance=[]; histogram=Counter(); selected_groups=set()
    for eid,rows in sorted(groups.items()):
        # Stable per-record ranking, with a STRAIGHT cap to counter the long-run
        # class imbalance without inventing or relabeling a teacher decision.
        rank=lambda row: digest(['closed-loop-export-v1',row['record_id']])
        ranked=sorted(rows,key=rank)
        by_label=Counter(); chosen=[]
        for row in ranked:
            action=row['target_action']
            if by_label[action] >= max(1, (per_episode+1)//2):
                continue
            chosen.append(row);by_label[action]+=1
            if len(chosen)>=per_episode:
                break
        for row in chosen:
            obs={k:row['observation'][k] for k in sorted(OBS_KEYS)}
            action=row['target_action']
            messages.append({'messages':[{'role':'system','content':SYSTEM},
                                         {'role':'user','content':canonical(obs)},
                                         {'role':'assistant','content':action}]})
            provenance.append({'index':len(provenance),'split':'train',
                               'world_group_id':row['world_group_id'],
                               'episode_id':eid,'source_record_id':row['record_id'],
                               'input_sha256':row['input_sha256'],
                               'label':action,'evidence_class':EVIDENCE,
                               'model_updates':0,'native_events':0})
            histogram[action]+=1;selected_groups.add(row['world_group_id'])
    destination.mkdir(parents=True)
    files={'train.messages.jsonl':write_jsonl(destination/'train.messages.jsonl',messages),
           'train.provenance.jsonl':write_jsonl(destination/'train.provenance.jsonl',provenance)}
    summary={'schema':SCHEMA+'/train-export-1','source_manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
             'source_eligible_sha256':hashlib.sha256((source_root/'train.eligible.jsonl').read_bytes()).hexdigest(),
             'split':'train-only','per_episode_cap':per_episode,'rows':len(messages),
             'world_groups':len(selected_groups),'action_histogram':dict(histogram),
             'provenance_class':EVIDENCE,'native_events':0,'model_updates':0,
             'prompt_allowlist':sorted(OBS_KEYS),
             'explicit_exclusions':['DEV','heldout','all source receipts','scores','negative labels','world ledgers'],
             'files':files}
    (destination/'export-manifest.json').write_text(canonical(summary)+'\n')
    return summary


if __name__=='__main__':
    main()
