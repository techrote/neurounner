#!/usr/bin/env python3
"""External CyberSand-inspired GAME corpus producer; no native/game mutations.

All task labels are synthetic host-heuristic labels, never evidence of a model,
Godot, material simulation, or actual native PAINT acceptance. Stdlib only.
"""
from __future__ import annotations
import argparse
from collections import deque, Counter
import hashlib
import json
from pathlib import Path
import random
import struct
import zlib

VERSION = 'cybersand.driv3r.game-corpus/1'
SOURCE = {
    'repo': 'techrote/cybersand',
    'common_w0_ref': 'research/612-worm-common',
    'common_w0_blob_sha': '8979b6f0296e81051194bacc907992d38e006605',
    'common_w0_contract': 'cybersand.worm.observation-action/0-experimental',
    'source_kind': 'read-only W0 contract; synthetic variant, NOT native readback',
}
ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'STOP')
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
SCENARIOS = ('corridor', 'deceptive', 'protected', 'deadend', 'motif', 'long_history')
MATERIALS = ('empty', 'sand', 'stone', 'metal', 'water')
SPLITS = ('train', 'dev', 'heldout')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def seed_for(*parts):
    return int(hashlib.sha256(canonical(parts).encode('utf-8')).hexdigest()[:16], 16)


def partition(world_group):
    v = seed_for('corpus-split-v1', world_group) % 100
    return 'train' if v < 80 else 'dev' if v < 90 else 'heldout'


def world(seed, scenario, size=96):
    if scenario not in SCENARIOS or size != 96:
        raise ValueError('v1 world requires a known scenario and 96x96 grid')
    rng = random.Random(seed_for('world-v1', seed, scenario))
    mid = size // 2
    start = (7, mid)
    obstacles = set()
    protected = set()
    # Shared rectangular debris and varied reachability without rewriting game data.
    for _ in range(9 + seed % 6):
        w, h = rng.randint(2, 12), rng.randint(2, 11)
        x, y = rng.randint(11, size - w - 3), rng.randint(3, size - h - 3)
        obstacles.update((xx, yy) for yy in range(y, y+h) for xx in range(x, x+w))
    if scenario in ('corridor', 'deadend', 'deceptive'):
        # Approach offers exit choices; the eastern throat is a cul-de-sac.
        for x in range(13, 28):
            obstacles.add((x, mid-1)); obstacles.add((x, mid+1))
        for y in (mid-1, mid, mid+1):
            obstacles.add((28, y))
        obstacles.discard((13, mid-1))  # observable side escape
    if scenario == 'protected':
        # Protected material is immutable AND non-traversable in this variant.
        for x in range(23, 57):
            for y in range(mid-12, mid+12):
                if (x+y+seed) % 5 == 0:
                    protected.add((x, y))
        for x in range(23, 57):
            protected.discard((x, mid-13))
    if scenario in ('motif', 'long_history'):
        for xx in range(25, 68, 7):
            for yy in range(12, 80, 15):
                if rng.random() < 0.45:
                    obstacles.update((xx+a, yy+b) for a in range(2) for b in range(6))
    safe_area = {(x, y) for x in range(4, 13) for y in range(mid-4, mid+5)}
    obstacles.difference_update(safe_area); protected.difference_update(safe_area)
    protected.difference_update(obstacles)
    # Select a real reachable goal; never declare disconnected goals achievable.
    passable = lambda p: 0 <= p[0] < size and 0 <= p[1] < size and p not in obstacles and p not in protected
    q = deque([start]); distance = {start: 0}
    while q:
        p = q.popleft()
        for dx, dy in DIRECTIONS:
            n = (p[0]+dx, p[1]+dy)
            if passable(n) and n not in distance:
                distance[n] = distance[p] + 1
                q.append(n)
    candidates = [p for p, d in distance.items() if d >= (45 if scenario == 'long_history' else 25)]
    if not candidates:
        raise RuntimeError('world has no reachable distant goal')
    candidates.sort(key=lambda p: (abs(p[0]-70)+abs(p[1]-mid), -distance[p], p))
    goal = rng.choice(candidates[:min(100, len(candidates))])
    mats = {}
    for y in range(size):
        for x in range(size):
            if (x, y) in obstacles:
                mat = 'stone'
            elif (x, y) in protected:
                mat = 'metal'
            else:
                band = (x//9 + y//12 + seed%7) % 11
                mat = 'sand' if band in (0,1,2,3,4) else 'water' if band == 5 else 'empty'
            if mat != 'empty':
                mats[(x, y)] = mat
    base_world = {'size': size, 'start': list(start), 'goal': list(goal),
                  'scenario': scenario, 'generator_seed': seed,
                  'obstacles': [list(p) for p in sorted(obstacles)],
                  'protected': [list(p) for p in sorted(protected)],
                  'material_grid_sha256': digest([mats.get((x,y), 'empty') for y in range(size) for x in range(size)]),
                  'material_palette': list(MATERIALS)}
    return base_world, obstacles, protected, mats, distance


def candidate_state(pos, heading, visited, history, obstacles, protected, size):
    x, y = pos; moves = {}
    for name, rot in (('STRAIGHT',0),('LEFT',-1),('RIGHT',1)):
        d = (heading+rot)%4
        nextp = (x+DIRECTIONS[d][0], y+DIRECTIONS[d][1])
        if 0 <= nextp[0] < size and 0 <= nextp[1] < size and nextp not in obstacles and nextp not in protected:
            moves[name] = (nextp, d)
    fresh = [a for a, (p, _) in moves.items() if p not in visited]
    legal = list(moves)
    if len(fresh) >= 2 and len(history) < 2048:
        legal.append('BRANCH')
    if history:
        legal.append('RETRACE')
    legal.append('STOP')
    return moves, fresh, legal


def local_grid(pos, visited, obstacles, protected, mats, size):
    x, y = pos; symbolic = []; material = []; immutable = []
    for dy in range(-2, 3):
        line = ''; mline = []; pline = ''
        for dx in range(-2, 3):
            p = (x+dx, y+dy)
            out = not (0 <= p[0] < size and 0 <= p[1] < size)
            line += ('@' if dx == dy == 0 else
                     '#' if out or p in obstacles or p in protected else
                     '+' if p in visited else '.')
            mline.append('boundary' if out else mats.get(p, 'empty'))
            pline += '1' if p in protected else '0'
        symbolic.append(line); material.append(mline); immutable.append(pline)
    return symbolic, material, immutable


def goal_distances(goal, obstacles, protected, size):
    q = deque([goal]); d = {goal: 0}
    while q:
        x, y = q.popleft()
        for dx, dy in DIRECTIONS:
            p = (x+dx, y+dy)
            if 0 <= p[0] < size and 0 <= p[1] < size and p not in obstacles and p not in protected and p not in d:
                d[p] = d[(x,y)]+1; q.append(p)
    return d


def teacher(legal, moves, fresh, visited, history, goal, distances, tick, scenario, rng, pos):
    """Bounded *synthetic heuristic*, not an optimal-policy assertion."""
    if (pos == goal and scenario != 'long_history') or (scenario == 'long_history' and goal in visited and tick >= 128):
        return 'STOP', {a: (1000 if a == 'STOP' else -1000) for a in legal}
    scores = {}
    for action in legal:
        if action == 'STOP':
            score = -1000.0
        elif action == 'RETRACE':
            # Unwind real stack dead ends. Never fabricate an illegal backstep.
            score = 250.0 if not fresh else -500.0
        else:
            arm = action
            if action == 'BRANCH':
                arm = next((a for a in ('RIGHT','LEFT','STRAIGHT') if a in fresh), 'STRAIGHT')
            destination = moves[arm][0]
            score = -4.0 * distances.get(destination, 10000)
            score += 28.0 if destination not in visited else -75.0
            if action == 'BRANCH':
                # Branches should be occasional structures, not 60% of labels.
                score += (14.0 if tick % 17 == 0 else -24.0)
            if scenario in ('motif', 'long_history') and action in ('LEFT', 'RIGHT'):
                score += 4.0
            if scenario == 'deceptive' and tick < 15 and action == 'STRAIGHT':
                score += 26.0
            if scenario == 'deadend' and tick < 23 and action == 'STRAIGHT':
                score += 80.0
        scores[action] = round(score, 5)
    best = max(scores, key=lambda a: (scores[a], -ACTIONS.index(a)))
    # An occasional deterministic wobble supplies original texture variation.
    if scenario in ('motif', 'long_history') and tick % 19 == 7:
        alternate = [a for a in ('LEFT','RIGHT','BRANCH') if a in legal and scores[a] > scores[best] - 8.0]
        if alternate:
            best = rng.choice(alternate)
    return best, scores

def difficulty(world_record, distance):
    d = distance[tuple(world_record['goal'])]
    scenario = world_record['scenario']
    if scenario in ('deadend', 'protected', 'long_history') or d >= 95:
        return 'hard'
    if scenario in ('deceptive', 'motif') or d >= 70:
        return 'medium'
    return 'easy'

def episode(seed, scenario, family, policy='heuristic', budget=160):
    if budget < 4 or budget > 4096 or family not in ('greebles','pipework','architecture'):
        raise ValueError('invalid budget or family')
    w, obstacles, protected, mats, reach = world(seed, scenario)
    group = f'{scenario}:{seed}'
    split = partition(group)
    world_id = digest(w)[:24]
    eid = digest([group, family, policy, budget])[:24]
    pos = tuple(w['start']); heading = 0; history = []; visited = {pos}
    past = []; rows = []; trail = [pos]; branches = set(); stop = False
    gd = goal_distances(tuple(w['goal']), obstacles, protected, w['size'])
    rng = random.Random(seed_for('teacher-v1', group, family, budget))
    for tick in range(budget):
        moves, fresh, legal = candidate_state(pos, heading, visited, history, obstacles, protected, w['size'])
        local, local_mats, local_protected = local_grid(pos, visited, obstacles, protected, mats, w['size'])
        observation = {'x': pos[0], 'y': pos[1], 'heading': heading, 'local_5x5': local,
                       'legal': legal, 'last_actions': past[-4:], 'family': family, 'tick': tick,
                       'local_material_5x5': local_mats, 'local_protected_5x5': local_protected,
                       'goal_delta': [w['goal'][0]-pos[0],w['goal'][1]-pos[1]],
                       'goal_reached': pos == tuple(w['goal']), 'budget_remaining': budget-tick}
        action, scores = teacher(legal, moves, fresh, visited, history, tuple(w['goal']), gd,
                                 tick, scenario, rng, pos)
        # Negatives include both legal-but-heuristically-worse and host-refused actions.
        legal_distractors = [a for a in legal if a != action and scores[a] < scores[action]-0.01]
        refused = [a for a in ACTIONS if a not in legal]
        negatives = [{'action': a, 'kind': 'heuristic_less_preferred_legal'} for a in legal_distractors[:2]]
        negatives += [{'action': a, 'kind': 'host_illegal'} for a in refused[:2]]
        before = pos; prior = (pos, heading)
        if action == 'STOP':
            stop = True
        elif action == 'RETRACE':
            pos, heading = history.pop()
        else:
            direction_action = action
            if action == 'BRANCH':
                branches.add(pos)
                direction_action = next(a for a in ('RIGHT','LEFT','STRAIGHT') if a in fresh)
            history.append(prior)
            pos, heading = moves[direction_action]
            visited.add(pos)
        past.append(action); trail.append(pos)
        input_signature = digest([observation['local_5x5'], observation['local_material_5x5'],
                                  observation['local_protected_5x5'], observation['goal_delta'],
                                  observation['heading'], observation['family'], legal, past[-5:-1]])
        rows.append({'record_id':f'{eid}:{tick:04d}', 'schema': VERSION, 'split':split,
                     'world_group_id':group, 'world_id':world_id, 'episode_id':eid,
                     'evidence_class':'synthetic-host-heuristic', 'scenario':scenario,
                     'family':family, 'observation':observation, 'input_sha256':input_signature,
                     'target_action':action, 'target_origin':'deterministic-synthetic-heuristic-not-native',
                     'candidate_scores':scores, 'hard_negatives':negatives,
                     'receipt':{'raw_choice':action,'applied':action,'raw_legal':True,
                                'position_before':list(before), 'position_after':list(pos)}})
        if stop:
            break
    w.update({'schema':VERSION, 'split':split,'world_id':world_id,'world_group_id':group,
              'difficulty':difficulty(w, reach), 'provenance':SOURCE})
    info = {'schema':VERSION,'split':split,'world_group_id':group,'world_id':world_id,
            'episode_id':eid,'scenario':scenario,'family':family,'policy':policy,
            'difficulty':w['difficulty'],'steps':len(rows),'step_budget':budget,
            'goal_reached':tuple(w['goal']) in visited, 'at_goal_at_stop':pos==tuple(w['goal']), 'terminated_by_stop':stop,
            'unique_trail_cells':len(visited),'branches':len(branches),
            'retraces':past.count('RETRACE'), 'protected_count':len(protected),
            'evidence_class':'synthetic-host-heuristic','trail':[list(p) for p in trail],
            'branch_points':[list(p) for p in sorted(branches)]}
    return w, info, rows


def png_bytes(w, ep, scale=2):
    size=w['size']; obstacles={tuple(p) for p in w['obstacles']}; protect={tuple(p) for p in w['protected']}
    trail={tuple(p) for p in ep['trail']}; branch={tuple(p) for p in ep['branch_points']}
    goal=tuple(w['goal']); tip=tuple(ep['trail'][-1]); pixels=bytearray()
    for y in range(size):
        line=bytearray()
        for x in range(size):
            p=(x,y)
            color=((246,204,79) if p==tip else (100,240,120) if p==goal else
                   (224,103,223) if p in branch else (65,70,85) if p in obstacles else
                   (219,80,70) if p in protect else (75,180,225) if p in trail else (17,22,31))
            line.extend(bytes(color)*scale)
        for _ in range(scale):
            pixels.extend(b'\x00'); pixels.extend(line)
    def chunk(name, payload):
        return struct.pack('>I',len(payload))+name+payload+struct.pack('>I',zlib.crc32(name+payload)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',size*scale,size*scale,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels,9))+chunk(b'IEND',b'')


def exact_or_near_duplicate(a, b):
    # Keep scenario semantics; reject nearly identical world masks across split groups.
    if a['scenario'] != b['scenario']:
        return False
    x={tuple(p) for p in a['obstacles']}|{tuple(p) for p in a['protected']}
    y={tuple(p) for p in b['obstacles']}|{tuple(p) for p in b['protected']}
    return len(x&y)/max(len(x|y),1) > 0.985


def write_jsonl(path, rows):
    payload=''.join(canonical(r)+'\n' for r in rows).encode('utf-8')
    path.write_bytes(payload)
    return {'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'rows':len(rows)}


def build(out:Path, worlds_per_scenario=20, budget=160, png_limit=12):
    if not 3 <= worlds_per_scenario <= 1000 or not 4 <= budget <= 4096 or not 0 <= png_limit <= 1000:
        raise ValueError('unbounded request')
    out.mkdir(parents=True,exist_ok=True)
    quotas={'train':worlds_per_scenario - 2*(worlds_per_scenario//10),
            'dev':worlds_per_scenario//10,'heldout':worlds_per_scenario//10}
    # Always include meaningful DEV and heldout groups, even for small trials.
    if not quotas['dev'] or not quotas['heldout']:
        raise ValueError('worlds_per_scenario must be at least 10')
    worlds_by_split={s:[] for s in SPLITS}; episodes_by_split={s:[] for s in SPLITS};
    decisions_by_split={s:[] for s in SPLITS}; seen_input={s:set() for s in SPLITS}
    seen_worlds=[]; discarded=Counter(); pngs=[]; gallery_candidates=[]
    for scenario in SCENARIOS:
        got=Counter(); attempts=0
        while any(got[s] < quotas[s] for s in SPLITS):
            attempts+=1
            if attempts>worlds_per_scenario*500:
                raise RuntimeError('split population stalled')
            seed=seed_for('world-sequence',scenario,attempts)%1_000_000_007
            group=f'{scenario}:{seed}'; split=partition(group)
            if got[split]>=quotas[split]:
                continue
            families=('greebles','pipework','architecture')
            w,e,rows=episode(seed,scenario,families[0],budget=budget if scenario!='long_history' else max(256,budget))
            if any(exact_or_near_duplicate(w, other) for other in seen_worlds):
                discarded['near_identical_world']+=1; continue
            # Three independent steering styles on the same geometry, but always
            # in ONE split; no style/world confounding or cross-split leakage.
            variants=[(w,e,rows)]+[episode(seed,scenario,fam,budget=budget if scenario!='long_history' else max(256,budget)) for fam in families[1:]]
            accepted=[]
            for world_variant, ep, decisions in variants:
                filtered=[]
                for row in decisions:
                    key=row['input_sha256']
                    if key in seen_input[split]:
                        discarded['duplicate_local_state']+=1;continue
                    seen_input[split].add(key);filtered.append(row)
                accepted.append((world_variant,ep,filtered))
            if not any(x[2] for x in accepted):
                discarded['empty_episode']+=1;continue
            seen_worlds.append(w); worlds_by_split[split].append(w)
            got[split]+=1
            for world_variant,ep,filtered in accepted:
                episodes_by_split[split].append(ep); decisions_by_split[split].extend(filtered)
                gallery_candidates.append((split,scenario,world_variant,ep))
    # Guard identical observation inputs across TRAIN/DEV/heldout, not only world seeds.
    cross={}
    for split in SPLITS:
        unique=[]
        for row in decisions_by_split[split]:
            key=row['input_sha256']
            if key in cross and cross[key] != split:
                discarded['cross_split_duplicate_input']+=1;continue
            cross[key]=split; unique.append(row)
        decisions_by_split[split]=unique
    # Visual samples span scenarios AND splits before repeating a family/world.
    selected=[]; used=set()
    for wanted_split in ('train','heldout','dev'):
        for wanted_scenario in SCENARIOS:
            if len(selected)>=png_limit:break
            choice=next((item for item in gallery_candidates if item[0]==wanted_split and item[1]==wanted_scenario),None)
            if choice is not None:
                selected.append(choice);used.add((choice[2]['world_id'],choice[3]['family']))
    for candidate in gallery_candidates:
        if len(selected)>=png_limit:break
        key=(candidate[2]['world_id'],candidate[3]['family'])
        if key not in used: selected.append(candidate);used.add(key)
    for split,scenario,w,ep in selected:
        name=f"{split}_{scenario}_{ep['family']}_{w['world_id']}.png"
        (out/name).write_bytes(png_bytes(w,ep));pngs.append(name)
    inventory={}
    for split in SPLITS:
        for kind,data in (('worlds',worlds_by_split),('episodes',episodes_by_split),('decisions',decisions_by_split)):
            filename=f'{split}.{kind}.jsonl'
            inventory[filename]=write_jsonl(out/filename,data[split])
    for name in pngs:
        raw=(out/name).read_bytes();inventory[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    groups_by_split={s:{w['world_group_id'] for w in worlds_by_split[s]} for s in SPLITS}
    assert not (groups_by_split['train']&groups_by_split['dev'] or groups_by_split['train']&groups_by_split['heldout'] or groups_by_split['dev']&groups_by_split['heldout'])
    manifest={'schema':VERSION,'seed_strategy':'sha256 namespaced deterministic Python 3 stdlib',
              'split_strategy':'sha256(world_group_id) 80/10/10, quotas per scenario',
              'source':SOURCE,'parameters':{'worlds_per_scenario':worlds_per_scenario,'step_budget':budget,'png_limit':png_limit},
              'counts':{'worlds':{s:len(worlds_by_split[s]) for s in SPLITS},
                        'episodes':{s:len(episodes_by_split[s]) for s in SPLITS},
                        'decisions':{s:len(decisions_by_split[s]) for s in SPLITS},
                        'hard_negatives':{s:sum(len(r['hard_negatives']) for r in decisions_by_split[s]) for s in SPLITS},
                        'discarded':dict(discarded)},
              'files':inventory,'claims':{'native_godot_records':0,'trained_model_records':0,
                                         'synthetic_training_examples':sum(len(x) for x in decisions_by_split.values()),
                                         'heldout_not_used_for_training':True}}
    (out/'manifest.json').write_text(canonical(manifest)+'\n',encoding='utf-8')
    return manifest


def audit(folder):
    root=json.loads((folder/'manifest.json').read_text())
    for name,receipt in root['files'].items():
        payload=(folder/name).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=receipt['sha256'] or len(payload)!=receipt['bytes']:
            raise ValueError(f'corrupt or modified corpus file: {name}')
        if name.endswith('.jsonl') and len(payload.splitlines()) != receipt['rows']:
            raise ValueError(f'row count mismatch: {name}')
    groups=[]; unique=set()
    for s in SPLITS:
        for line in (folder/f'{s}.worlds.jsonl').read_text().splitlines():
            record=json.loads(line)
            if record['split']!=s or record['world_group_id'] in unique:
                raise ValueError('leaking world group across splits')
            unique.add(record['world_group_id']);groups.append(record)
    seen={}
    for s in SPLITS:
        for line in (folder/f'{s}.decisions.jsonl').read_text().splitlines():
            row=json.loads(line); key=row['input_sha256']
            if key in seen and seen[key]!=s:
                raise ValueError('leaking duplicate model input across splits')
            seen[key]=s
            if row['target_action'] not in row['observation']['legal'] or not row['receipt']['raw_legal']:
                raise ValueError('illegal synthetic teacher receipt')
    return {'audit':'PASS','world_groups':len(groups),'distinct_inputs':len(seen),'files':len(root['files'])}


def main():
    p=argparse.ArgumentParser(description='Standalone synthetic CyberSand GAME corpus factory; NEVER changes native game')
    p.add_argument('--output',type=Path,default=Path('game-corpus-v1'))
    p.add_argument('--worlds-per-scenario',type=int,default=20)
    p.add_argument('--budget',type=int,default=160)
    p.add_argument('--png-limit',type=int,default=12)
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    result=audit(args.output) if args.audit_only else build(args.output,args.worlds_per_scenario,args.budget,args.png_limit)
    if not args.audit_only:
        result={'built':result['counts'],'audit':audit(args.output)}
    print(canonical(result))


if __name__=='__main__':
    main()
