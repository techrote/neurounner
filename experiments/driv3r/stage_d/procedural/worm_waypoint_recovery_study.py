"""Predeclared DEVELOPMENT-only geodesic waypoint recovery for a drawing worm.

Frozen starting worlds, seeds, goals, guidance, and source-replayed baseline
come from worm_waypoint_study. No model, native edit, receipt, or native Undo.
A synthetic deterministic geodesic oracle is a *control*, not a learned policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from functools import lru_cache

from worm_sketch import RNG, DIRS, canonical, digest, rgba_png
from worm_pulse_study import score
from worm_waypoint_study import (frozen_state, _ink_hash, START_TICKS,
                                 HORIZON, run_case as prior_run_case)
from worm_horizon_fixtures import SCENES, SEEDS, preregistration

VERSION = 'd2.worm-waypoint-recovery/1'
QUALIFICATION = 'DEVELOPMENT_SYNTHETIC_SOURCE_REPLAYED_NOT_NATIVE'
ARMS = ('sham', 'hold12_8192', 'full32_8192', 'full32_16384',
        'rescue_hysteresis', 'oracle_geodesic')
SOURCES = ('worm_waypoint_recovery_study.py', 'test_worm_waypoint_recovery_study.py',
           'worm_waypoint_study.py',
           'worm_pulse_horizon_study.py', 'worm_pulse_study.py',
           'worm_horizon_study.py', 'worm_horizon_fixtures.py',
           'worm_scent_study.py', 'worm_sketch.py')


def _guidance(arm, step, plateau, boosting, boost_steps):
    """Defined before testing. Escalation is bounded by the shared 32 steps."""
    if arm == 'sham' or arm == 'oracle_geodesic':
        return 0, boosting, boost_steps
    if arm == 'hold12_8192':
        return (8192 if step < 12 else 0), boosting, boost_steps
    if arm == 'full32_8192':
        return 8192, boosting, boost_steps
    if arm == 'full32_16384':
        return 16384, boosting, boost_steps
    # Rescue state can only inspect *previous actual cursor progress*.
    # Starts gently, escalates on four non-improving steps; active burst lasts
    # at most six steps, with an explicit <=12 boosted-decision budget.
    if not boosting and plateau >= 4 and boost_steps < 12:
        boosting = 6
    if boosting:
        boosting -= 1
        boost_steps += 1
        return 24576, boosting, boost_steps
    return 4096, boosting, boost_steps


def run_case(scene: str, seed: int, tick: int, arm: str):
    if arm not in ARMS:
        raise ValueError('unknown predeclared recovery arm')
    state = frozen_state(scene, seed, tick)
    pos, heading = state['pos'], state['heading']
    visits, ink = dict(state['visits']), set(state['ink'])
    starting_ink = set(ink)
    goal, distances = state['goal'], state['distances']
    rng = RNG(seed)
    rng.state = state['rng_state']
    first_noise_rng = RNG(seed)
    for _ in range(tick * 4):
        first_noise_rng.next()
    steps = []
    path = [pos]
    reached_step = None
    best = distances[pos]
    plateau = boosting = boost_steps = 0
    total_guide = choice_overrides = productive_choices = 0
    for step in range(HORIZON):
        legal = [(d, (pos[0]+v[0], pos[1]+v[1])) for d,v in enumerate(DIRS)
                 if (pos[0]+v[0], pos[1]+v[1]) in state['allowed']]
        if not legal:
            raise ValueError('frozen geometry unexpectedly has no legal actions')
        if step == 0:
            noise = {d: first_noise_rng.next() % 73 for d in range(4)}
            if first_noise_rng.state != rng.state:
                raise ValueError('source RNG mismatch')
        else:
            noise = {d: rng.next() % 73 for d in range(4)}
        # All arms offered identical legal actions and same deterministic noise.
        target = min(legal, key=lambda pair: (distances.get(pair[1], 1<<30), pair[0]))
        delta, boosting, boost_steps = _guidance(arm, step, plateau,
                                                boosting, boost_steps)
        def evaluate(pair):
            d, q = pair
            guide = min(65535, state['guide'][q] + delta) if pair == target else state['guide'][q]
            return (score(tick + step, seed, heading, visits, noise, d, q, guide), -d)
        natural = max(legal, key=lambda pair: (score(tick+step, seed, heading,
                             visits, noise, pair[0], pair[1], state['guide'][pair[1]]), -pair[0]))
        chosen = target if arm == 'oracle_geodesic' else max(legal, key=evaluate)
        d, q = chosen
        prior = _ink_hash(ink)
        pen = ('down' if (tick+step) % state['spec']['pen_rhythm']['period'] >=
               state['spec']['pen_rhythm']['lifted'] else 'up')
        if pen == 'down':
            ink.add(q)
        after = _ink_hash(ink)
        if chosen != natural:
            choice_overrides += 1
        if chosen == target:
            productive_choices += 1
        if delta:
            total_guide += 1
        new_best = min(best, distances.get(q, 1<<30))
        plateau = 0 if new_best < best else plateau + 1
        best = new_best
        steps.append(dict(step=step, absolute_tick=tick+step, kind='STEP',
                          from_cursor=list(pos), to_cursor=list(q), heading=d,
                          legal_catalogue=[[i,list(p)] for i,p in legal],
                          productive_candidate=[target[0],list(target[1])],
                          original_guide_u16=state['guide'][q],
                          applied_guide_delta_u16=delta,
                          oracle_forced=(arm == 'oracle_geodesic'),
                          rng_after_noise=rng.state,
                          goal_distance_after=distances.get(q),
                          pen=pen, material_before_sha256=prior,
                          material_after_sha256=after,
                          native_receipt=None, model_decision=None))
        pos, heading = q, d
        visits[q] = visits.get(q, 0) + 1
        path.append(q)
        if q == goal and reached_step is None:
            reached_step = step
    if arm == 'sham':
        src = state['baseline']
        if path != src['path'] or ink != src['ink']:
            raise ValueError('source-replayed sham failed baseline correspondence')
    if arm == 'hold12_8192':
        prior = prior_run_case(scene, seed, tick, 'hold12_8192')
        if path != [tuple(q) for q in prior['record']['path']] or ink != prior['final_ink']:
            raise ValueError('12-step arm changed prior qualified policy')
    if boost_steps > 12:
        raise ValueError('recovery exceeded declared escalation budget')
    metrics = dict(start_distance=distances[state['pos']],
                   min_distance=best, final_distance=distances.get(pos),
                   reached_goal=reached_step is not None,
                   first_reached_step=reached_step, interventions=total_guide,
                   rescue_boost_steps=boost_steps,
                   changed_vs_natural=choice_overrides,
                   geodesic_choices=productive_choices,
                   unique_cursor_cells=len(set(path)),
                   repeats=len(path)-len(set(path)),
                   painted_cells_after=len(ink),
                   new_painted_cells=len(ink-starting_ink),
                   material_sha256=_ink_hash(ink),
                   path_sha256=digest(canonical(path)))
    record=dict(version=VERSION, qualification=QUALIFICATION,
                input=dict(scene=scene, seed=seed, tick=tick, arm=arm,
                           horizon=HORIZON, goal=list(goal),
                           frozen_fixture_sha256=digest(canonical(state['spec'])),
                           original_material_sha256=_ink_hash(starting_ink)),
                metrics=metrics, path=[list(q) for q in path], events=steps + [
                    dict(kind='UNDO_MATERIAL_EDIT_UNAVAILABLE', executed=False,
                         native_receipt=None, material_after_sha256=_ink_hash(ink))],
                native=None, model=None, native_replay_verified=False)
    return dict(record=record, original_ink=starting_ink, ink=ink, allowed=state['allowed'],
                goal=goal, source_guide=state['guide'])


def artifacts(out):
    record=out['record']; allowed=out['allowed']; goal=out['goal']
    start=tuple(record['path'][0]); last=tuple(record['path'][-1])
    def background(x,y, ink):
        q=(x,y)
        return bytes((83,68,76,255) if q not in allowed else
                     (230,215,181,255) if q in ink else (22,32,43,255))
    original=rgba_png(80,80,lambda x,y: (bytes((255,196,70,255)) if (x,y)==goal else
                   bytes((253,91,110,255)) if (x,y)==start else
                   background(x,y,out['original_ink'])))
    final=rgba_png(80,80,lambda x,y: (bytes((255,196,70,255)) if (x,y)==goal else
                bytes((84,210,228,255)) if (x,y)==last else
                background(x,y,out['ink'])))
    return {'case.json':canonical(record), 'original.png':original, 'result.png':final}

@lru_cache(maxsize=1)
def source_hashes():
    root=Path(__file__).parent
    return {f:digest((root/f).read_bytes()) for f in SOURCES}

@lru_cache(maxsize=1)
def frozen_input_hash():
    return digest(canonical(preregistration()))


def manifest(blobs):
    return dict(qualification=QUALIFICATION, input_freeze_sha256=frozen_input_hash(),
                source_sha256=source_hashes(), native=None, model=None,
                native_replay_verified=False,
                artifacts={f:digest(v) for f,v in blobs.items()})


def case_name(r):
    v=r['input'];return f"{v['scene']}-seed{v['seed']}-tick{v['tick']}-{v['arm']}"


def verify(folder):
    folder=Path(folder)
    rec=json.loads((folder/'case.json').read_bytes())
    v=rec['input']
    if folder.name != case_name(rec): raise ValueError('case-name drift')
    out=run_case(v['scene'],v['seed'],v['tick'],v['arm'])
    blobs=artifacts(out)
    if rec != out['record'] or json.loads((folder/'manifest.json').read_bytes()) != manifest(blobs):
        raise ValueError('source-replay or manifest disagreement')
    if set(f.name for f in folder.iterdir()) != set(blobs)|{'manifest.json'}:
        raise ValueError('unexpected artifact set')
    for name,raw in blobs.items():
        if (folder/name).read_bytes() != raw:
            raise ValueError('source-replay byte mismatch: '+name)
    return True


def corpus(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for scene in SCENES:
        for seed in SEEDS:
            for tick in START_TICKS:
                for arm in ARMS:
                    out=run_case(scene,seed,tick,arm);rec=out['record']
                    folder=root/case_name(rec);folder.mkdir(exist_ok=True)
                    blobs=artifacts(out)
                    for name,blob in blobs.items(): (folder/name).write_bytes(blob)
                    m=manifest(blobs);(folder/'manifest.json').write_bytes(canonical(m))
                    verify(folder)
                    rows.append(dict(case=folder.name, metrics=rec['metrics'],
                                     manifest_sha256=digest(canonical(m))))
    summary={arm:dict(cases=sum(row['case'].endswith('-'+arm) for row in rows),
                      reached=sum(row['metrics']['reached_goal'] for row in rows if row['case'].endswith('-'+arm)),
                      min_distance_sum=sum(row['metrics']['min_distance'] for row in rows if row['case'].endswith('-'+arm)),
                      new_ink_sum=sum(row['metrics']['new_painted_cells'] for row in rows if row['case'].endswith('-'+arm)),
                      interventions_sum=sum(row['metrics']['interventions'] for row in rows if row['case'].endswith('-'+arm)),
                      boost_sum=sum(row['metrics']['rescue_boost_steps'] for row in rows if row['case'].endswith('-'+arm)))
             for arm in ARMS}
    result=dict(version=VERSION, qualification=QUALIFICATION,
                source_sha256=source_hashes(), input_freeze_sha256=frozen_input_hash(),
                cases=rows, summary=summary)
    (root/'../summary.json').resolve().write_bytes(canonical(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    args=p.parse_args(); print(json.dumps(corpus(args.out)['summary'],sort_keys=True,indent=2))
