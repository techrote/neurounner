"""Source-replayed, bounded SOFT waypoint steering in the D-2 synthetic worm.

No native World, no native history, no model, no synthetic material Undo.
Existing frozen scene/seed/tick states stay untouched. Goal selection depends
only on those states and legal geometry, not on outcomes. Every arm sees the
same start state, random numbers, hard mask, pen cadence and 32-decision budget.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from functools import lru_cache
from pathlib import Path

from worm_sketch import DIRS, RNG, canonical, digest, rgba_png
from worm_horizon_fixtures import SCENES, SEEDS, preregistration
from worm_pulse_study import TICKS, score
from worm_pulse_horizon_study import baseline, baseline_move, HORIZON

VERSION = 'd2.worm-waypoint-steering/1'
QUALIFICATION = 'DEVELOPMENT_SYNTHETIC_SOURCE_REPLAYED_NOT_NATIVE'
START_TICKS = (60,)  # bounded first pilot: one frozen decision state per scene/seed
ARMS = {'sham': (0, 12), 'single_4096': (4096, 1),
        'hold4_4096': (4096, 4), 'hold12_4096': (4096, 12),
        'hold12_8192': (8192, 12)}
SOURCES = ('worm_waypoint_study.py', 'worm_pulse_horizon_study.py',
           'worm_pulse_study.py', 'worm_horizon_study.py',
           'worm_horizon_fixtures.py', 'worm_scent_study.py', 'worm_sketch.py')


def _ink_hash(ink):
    return digest(canonical(sorted(map(list, ink))))


def distance_from(target, allowed):
    """Integer cardinal geodesic, not straight-line clearance through obstacles."""
    if target not in allowed:
        raise ValueError('target not legal')
    out = {target: 0}
    q = deque([target])
    while q:
        p = q.popleft()
        for dx, dy in DIRS:
            n = (p[0]+dx, p[1]+dy)
            if n in allowed and n not in out:
                out[n] = out[p] + 1
                q.append(n)
    return out


@lru_cache(maxsize=48)
def frozen_state(scene, seed, tick):
    if scene not in SCENES or seed not in SEEDS or tick not in START_TICKS:
        raise ValueError('not a registered DEVELOPMENT state')
    spec, allowed, guide, source, states = baseline(scene, seed)
    pos, heading, visits, ink, rng_state, _, _ = states[tick]
    allowed = frozenset(allowed)
    # Independent-of-outcomes, input-defined waypoint: reachable northeast
    # relative to the source-replayed cursor, never fabricated across a wall.
    origin_dist = distance_from(pos, allowed)
    desired = (pos[0]+8, pos[1]-8)
    feasible = [p for p, d in origin_dist.items() if 6 <= d <= 20]
    if not feasible:
        raise ValueError('registered state has no meaningful waypoint')
    goal = min(feasible, key=lambda p: (abs(p[0]-desired[0]) + abs(p[1]-desired[1]),
                                        abs(origin_dist[p]-16), p[1], p[0]))
    to_goal = distance_from(goal, allowed)
    if pos not in to_goal:
        raise ValueError('goal not reachable from source cursor')
    # Validate original baseline for all 32 ticks using the already independently
    # checked first-class source replay, before running any intervention.
    b = baseline_move(scene, seed, tick)
    original = [e for e in source['events'] if e['kind'] == 'STEP']
    slice_path = [tuple(original[tick]['from_cursor'])] + [tuple(e['to_cursor'])
                                                          for e in original[tick:tick+HORIZON]]
    if b['path'] != slice_path or _ink_hash(b['ink']) != original[tick+HORIZON-1]['material_after_sha256']:
        raise ValueError('frozen source shadow mismatch')
    return dict(spec=spec, allowed=allowed, guide=guide, pos=pos,
                heading=heading, visits=visits, ink=ink, rng_state=rng_state,
                goal=goal, distances=to_goal, baseline=b)


def run_case(scene, seed, tick, arm):
    if arm not in ARMS:
        raise ValueError('unregistered steering arm')
    state = frozen_state(scene, seed, tick)
    pos, heading = state['pos'], state['heading']
    visits, ink = dict(state['visits']), set(state['ink'])
    starting_ink = set(ink)
    goal, distances = state['goal'], state['distances']
    delta, duration = ARMS[arm]
    rng = RNG(seed)
    rng.state = state['rng_state']  # frozen source has consumed noise at tick
    initial_rng = RNG(seed)
    for _ in range(tick*4):
        initial_rng.next()
    path, events = [pos], []
    min_distance, first_reached_tick = distances[pos], None
    for step in range(HORIZON):
        absolute_tick = tick + step
        legal = [(d, (pos[0]+v[0], pos[1]+v[1])) for d,v in enumerate(DIRS)
                 if (pos[0]+v[0], pos[1]+v[1]) in state['allowed']]
        if not legal:
            raise ValueError('source-registered horizon unexpectedly blocked')
        if step == 0:
            noise = {d: initial_rng.next() % 73 for d in range(4)}
            if initial_rng.state != rng.state:
                raise ValueError('source noise mismatch')
        else:
            noise = {d: rng.next() % 73 for d in range(4)}
        # No hard legality change, no direct movement override: add scalar
        # preference to ONE geodesically productive legal action. Ties are
        # cardinal-ID deterministic and independent of baseline's winner.
        productive = min(legal, key=lambda x: (distances.get(x[1], 1<<30), x[0]))
        applied = delta if step < duration else 0
        def value(item):
            d, p = item
            g = min(65535, state['guide'][p] + applied) if item == productive else state['guide'][p]
            return (score(absolute_tick, seed, heading, visits, noise, d, p, g), -d)
        direction, q = max(legal, key=value)
        prior = _ink_hash(ink)
        pen = ('down' if absolute_tick % state['spec']['pen_rhythm']['period'] >=
               state['spec']['pen_rhythm']['lifted'] else 'up')
        if pen == 'down':
            ink.add(q)
        current = _ink_hash(ink)
        events.append(dict(tick=absolute_tick, kind='STEP',
                           from_cursor=list(pos), to_cursor=list(q), heading=direction,
                           legal_catalogue=[[d, list(p)] for d,p in legal],
                           goal=list(goal), goal_distance_after=distances.get(q),
                           productive_candidate=[productive[0], list(productive[1])],
                           applied_guide_delta_u16=applied,
                           original_guide_u16=state['guide'][q],
                           pen=pen, material_before_sha256=prior,
                           material_after_sha256=current,
                           native_receipt=None, model_decision=None))
        pos, heading = q, direction
        visits[q] = visits.get(q, 0) + 1
        path.append(q)
        if pos in distances:
            min_distance = min(min_distance, distances[pos])
        if pos == goal and first_reached_tick is None:
            first_reached_tick = step
    events.append(dict(kind='UNDO_MATERIAL_EDIT_UNAVAILABLE', executed=False,
                       native_receipt=None, material_after_sha256=_ink_hash(ink)))
    base = state['baseline']
    bp = base['path']
    baseline_ink = base['ink']
    bnew, new = baseline_ink-starting_ink, ink-starting_ink
    metrics = dict(steps=HORIZON, changed_first_decision=path[1] != bp[1],
                   mismatched_cursor_steps=sum(a != b for a,b in zip(path[1:], bp[1:])),
                   start_distance=distances[state['pos']],
                   final_distance=distances.get(pos), minimum_distance=min_distance,
                   reached_goal=first_reached_tick is not None,
                   first_reached_step=first_reached_tick,
                   baseline_minimum_distance=min(distances.get(p, 99999) for p in bp),
                   unique_cursor_cells=len(set(path)), repeats=len(path)-len(set(path)),
                   novel_ink_only=len(new-bnew), omitted_baseline_ink=len(bnew-new),
                   total_material_cells=len(ink),
                   material_sha256=_ink_hash(ink), path_sha256=digest(canonical(path)))
    rec = dict(version=VERSION, qualification=QUALIFICATION,
               input=dict(scene=scene, seed=seed, tick=tick, arm=arm,
                          delta_u16=delta, duration_decisions=duration, horizon=HORIZON,
                          goal=list(goal), input_fixture_sha256=digest(canonical(state['spec'])),
                          baseline_start_material_sha256=_ink_hash(starting_ink),
                          baseline_start_visits_sha256=digest(canonical(sorted(
                              [list(p)+[n] for p,n in state['visits'].items()])))),
               metrics=metrics, path=[list(p) for p in path], events=events,
               native=None, model=None, native_replay_verified=False)
    return dict(record=rec, original_ink=starting_ink, final_ink=ink,
                allowed=state['allowed'], baseline_ink=baseline_ink,
                goal=goal, guide=state['guide'])


def render(outcome):
    w,h=80,80
    allowed, ink, goal = outcome['allowed'], outcome['final_ink'], outcome['goal']
    start = tuple(outcome['record']['path'][0]); endpoint = tuple(outcome['record']['path'][-1])
    def base(x,y):
        p=(x,y)
        return bytes((255, 200, 53, 255)) if p == goal else (
            bytes((84, 210, 228, 255)) if p == endpoint else
            bytes((233, 215, 176, 255)) if p in ink else
            bytes((79, 67, 75, 255)) if p not in allowed else bytes((22,32,43,255)))
    return {'original.png': rgba_png(w,h,lambda x,y:
                bytes((250, 112, 90, 255)) if (x,y)==start else
                bytes((255, 200, 53, 255)) if (x,y)==goal else
                bytes((79, 67, 75, 255)) if (x,y) not in allowed else
                bytes((233,215,176,255)) if (x,y) in outcome['original_ink'] else bytes((22,32,43,255))),
            'result.png': rgba_png(w,h,base)}


def artifacts(outcome):
    return {'case.json':canonical(outcome['record']), **render(outcome)}


@lru_cache(maxsize=1)
def source_hashes():
    base=Path(__file__).parent
    return {name:digest((base/name).read_bytes()) for name in SOURCES}


@lru_cache(maxsize=1)
def input_freeze_hash():
    return digest(canonical(preregistration()))


def manifest(art):
    return dict(qualification=QUALIFICATION,
                source_sha256=source_hashes(),
                input_freeze_sha256=input_freeze_hash(),
                native=None, model=None, native_replay_verified=False,
                artifacts={k:digest(v) for k,v in art.items()})


def verify(folder):
    folder=Path(folder)
    rec=json.loads((folder/'case.json').read_bytes())
    arg=rec['input']
    if folder.name != f"{arg['scene']}-seed{arg['seed']}-tick{arg['tick']}-{arg['arm']}":
        raise ValueError('case name drift')
    out=run_case(arg['scene'],arg['seed'],arg['tick'],arg['arm'])
    expected=artifacts(out)
    if rec != out['record'] or json.loads((folder/'manifest.json').read_bytes()) != manifest(expected):
        raise ValueError('source-verified case mismatch')
    if set(p.name for p in folder.iterdir()) != set(expected)|{'manifest.json'}:
        raise ValueError('artifact set mismatch')
    for name,raw in expected.items():
        if (folder/name).read_bytes()!=raw:
            raise ValueError('tampered source-replayed bytes: '+name)
    return True


def corpus(root):
    root=Path(root); root.mkdir(parents=True, exist_ok=True)
    rows=[]
    for scene in SCENES:
        for seed in SEEDS:
            for tick in START_TICKS:
                for arm in ARMS:
                    out=run_case(scene,seed,tick,arm)
                    folder=root/f'{scene}-seed{seed}-tick{tick}-{arm}'
                    folder.mkdir(parents=True, exist_ok=True)
                    b=artifacts(out)
                    for n,v in b.items(): (folder/n).write_bytes(v)
                    m=manifest(b); (folder/'manifest.json').write_bytes(canonical(m))
                    if not verify(folder): raise ValueError('case not source-verifiable')
                    rows.append(dict(case=folder.name, input=out['record']['input'],
                                     metrics=out['record']['metrics'], manifest_sha256=digest(canonical(m))))
    stats={arm:dict(cases=sum(x['input']['arm']==arm for x in rows),
                    reach=sum(x['metrics']['reached_goal'] for x in rows if x['input']['arm']==arm),
                    mean_minimum_distance_sum=sum(x['metrics']['minimum_distance'] for x in rows if x['input']['arm']==arm),
                    mean_final_distance_sum=sum(x['metrics']['final_distance'] for x in rows if x['input']['arm']==arm),
                    differing_first_steps=sum(x['metrics']['changed_first_decision'] for x in rows if x['input']['arm']==arm),
                    mismatched_cursor_steps=sum(x['metrics']['mismatched_cursor_steps'] for x in rows if x['input']['arm']==arm))
           for arm in ARMS}
    result=dict(version=VERSION,qualification=QUALIFICATION,
                input_freeze_sha256=input_freeze_hash(),
                source_sha256=source_hashes(), statistics=stats, records=rows,
                native=None, model=None, native_replay_verified=False)
    (root/'index.json').write_bytes(canonical(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    a=p.parse_args();print(digest(canonical(corpus(a.out))))
