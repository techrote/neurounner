"""Source-fenced, matched 32-decision continuations of D-2 synthetic steering pulses.

One counterfactual guide bump is applied only at a preregistered decision tick.
Everything after that decision uses the original scalar field, exact tick/RNG noise,
permission mask, and pen cadence. This is DEVELOPMENT, never native/model evidence.
The source-verified strong_x16 trajectory is reconstructed at every decision before
any fork is run. Disagreements fail closed, not silently corrected.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from worm_sketch import DIRS, RNG, canonical, digest, rgba_png
from worm_horizon_fixtures import SCENES, SEEDS, fixture, preregistration
from worm_horizon_study import run_case as source_run_case
from worm_pulse_study import TICKS, score

VERSION = 'd2.worm-pulse-horizon/1'
DELTAS = (0, 2048, 4096, 8192)
HORIZON = 32
SOURCES = ('worm_pulse_horizon_study.py', 'worm_pulse_study.py',
           'worm_horizon_study.py', 'worm_horizon_fixtures.py',
           'worm_scent_study.py', 'worm_sketch.py')
QUALIFICATION = 'DEVELOPMENT_SYNTHETIC_SOURCE_REPLAYED'


def _hash_ink(ink):
    return digest(canonical(sorted(map(list, ink))))


@lru_cache(maxsize=12)
def baseline(scene, seed):
    spec, allowed, guide = fixture(scene, seed)
    source = source_run_case(scene, seed, 'strong_x16')
    expected = [e for e in source['events'] if e['kind'] == 'STEP']
    if len(expected) < max(TICKS) + HORIZON:
        raise ValueError('frozen source terminated before registered shadow horizon')
    pos, heading, visits, ink, rng = tuple(spec['origin']), spec['initial_heading'], {tuple(spec['origin']): 1}, set(), RNG(seed)
    states = []
    for tick, event in enumerate(expected):
        legal = [(d, (pos[0] + v[0], pos[1] + v[1])) for d, v in enumerate(DIRS)
                 if (pos[0] + v[0], pos[1] + v[1]) in allowed]
        noise = {d: rng.next() % 73 for d in range(4)}
        if not legal:
            raise ValueError('unexpected empty legal catalogue')
        def value(item):
            d, q = item
            return (score(tick, seed, heading, visits, noise, d, q, guide[q]), -d)
        d, q = max(legal, key=value)
        if (event['tick'] != tick or event['kind'] != 'STEP'
                or tuple(event['from_cursor']) != pos or event['heading'] != d
                or tuple(event['to_cursor']) != q
                or event['legal_catalogue'] != [[a, list(b)] for a, b in legal]
                or event['material_before_sha256'] != _hash_ink(ink)):
            raise ValueError('exact source baseline mismatch at tick ' + str(tick))
        states.append((pos, heading, dict(visits), frozenset(ink), rng.state, d, q))
        if event['pen'] == 'down':
            ink.add(q)
        if event['material_after_sha256'] != _hash_ink(ink):
            raise ValueError('source baseline material mismatch at tick ' + str(tick))
        pos, heading = q, d
        visits[q] = visits.get(q, 0) + 1
    if (len(states) != source['metrics']['steps'] or pos != tuple(source['trail'][-1])
            or _hash_ink(ink) != source['metrics']['material_sha256']):
        raise ValueError('baseline length/material final mismatch')
    return spec, frozenset(allowed), guide, source, tuple(states)


def _move(scene, seed, start_tick, delta, horizon=HORIZON):
    if scene not in SCENES or seed not in SEEDS or start_tick not in TICKS or delta not in DELTAS or horizon != HORIZON:
        raise ValueError('unregistered shadow case')
    spec, allowed, guide, source, states = baseline(scene, seed)
    pos, heading, old_visits, old_ink, rng_state, expected_dir, expected_q = states[start_tick]
    visits, ink = dict(old_visits), set(old_ink)
    rng = RNG(seed)
    rng.state = rng_state  # At the start tick noise has ALREADY been consumed by the frozen baseline.
    path, events, painted_segment = [pos], [], set()
    target_dir = None
    initial_legal = [(d, (pos[0] + v[0], pos[1] + v[1])) for d, v in enumerate(DIRS)
                     if (pos[0] + v[0], pos[1] + v[1]) in allowed]
    for choice in ((heading + 3) % 4, (heading + 1) % 4, heading, (heading + 2) % 4):
        if any(d == choice for d, _ in initial_legal):
            target_dir = choice
            break
    if target_dir is None:
        raise ValueError('frozen baseline has no legal target')
    target_point = next(q for d, q in initial_legal if d == target_dir)
    for tick in range(start_tick, start_tick + horizon):
        legal = [(d, (pos[0] + v[0], pos[1] + v[1])) for d, v in enumerate(DIRS)
                 if (pos[0] + v[0], pos[1] + v[1]) in allowed]
        if not legal:
            events.append(dict(tick=tick, kind='STOP', reason='no cardinal permitted move', native_receipt=None))
            break
        # The frozen baseline already consumed four random values for start_tick.
        if tick == start_tick:
            # Reconstruct noise independently from the seed; states capture
            # the RNG *after* source baseline noise was consumed.
            original_rng = RNG(seed)
            for _ in range(start_tick * 4):
                original_rng.next()
            noise = {d: original_rng.next() % 73 for d in range(4)}
            if original_rng.state != rng.state:
                raise ValueError('baseline noise-state custody mismatch')
        else:
            noise = {d: rng.next() % 73 for d in range(4)}
        def value(item):
            d, q = item
            g = (min(65535, guide[q] + delta)
                 if tick == start_tick and d == target_dir else guide[q])
            return (score(tick, seed, heading, visits, noise, d, q, g), -d)
        d, q = max(legal, key=value)
        before = _hash_ink(ink)
        pen = 'down' if tick % spec['pen_rhythm']['period'] >= spec['pen_rhythm']['lifted'] else 'up'
        if pen == 'down':
            ink.add(q)
            painted_segment.add(q)
        after = _hash_ink(ink)
        events.append(dict(tick=tick, kind='STEP', from_cursor=list(pos),
                           to_cursor=list(q), legal_catalogue=[[a, list(b)] for a, b in legal],
                           heading=d, pen=pen, guidance_u16=guide[q],
                           pulsed_guide_u16=min(65535, guide[q]+delta) if tick == start_tick and d == target_dir else guide[q],
                           first_step_pulse=tick == start_tick, material_before_sha256=before,
                           material_after_sha256=after, native_receipt=None, model_decision=None))
        path.append(q)
        pos, heading = q, d
        visits[q] = visits.get(q, 0) + 1
    return dict(path=path, events=events, ink=ink, ink_segment=painted_segment,
                allowed=allowed, starting_ink=set(old_ink), starting_cursor=states[start_tick][0],
                target_dir=target_dir, target_point=target_point, guide=guide, source=source,
                frozen_spec=spec)


@lru_cache(maxsize=48)
def baseline_move(scene, seed, start_tick):
    return _move(scene, seed, start_tick, 0)


def run_case(scene, seed, start_tick, delta):
    baseline_arm = baseline_move(scene, seed, start_tick)
    fork = baseline_arm if delta == 0 else _move(scene, seed, start_tick, delta)
    bpath, fpath = baseline_arm['path'], fork['path']
    if len(bpath) != HORIZON + 1 or len(fpath) != HORIZON + 1:
        raise ValueError('unexpected STOP within preregistered horizon')
    # Source identity checks: sham must be an exact slice of the committed baseline.
    original_steps = [e for e in baseline_arm['source']['events'] if e['kind'] == 'STEP']
    true_path = [tuple(original_steps[start_tick]['from_cursor'])] + [tuple(e['to_cursor']) for e in original_steps[start_tick:start_tick+HORIZON]]
    if bpath != true_path:
        raise ValueError('zero-pulse short-horizon trajectory disagrees with source')
    if any(q not in fork['allowed'] for q in fpath):
        raise ValueError('a fork crossed a protected pixel')
    changed = fpath[1] != bpath[1]
    if not changed and (bpath != fpath or fork['ink'] != baseline_arm['ink']):
        raise ValueError('identical first decisions produced unequal deterministic outcomes')
    bpaint, fpaint = baseline_arm['ink_segment'], fork['ink_segment']
    common = bpaint & fpaint
    origin = baseline_arm['starting_ink']
    # Distinguish re-inking old material from genuinely novel material.
    bnew, fnew = bpaint - origin, fpaint - origin
    rec = dict(version=VERSION, qualification=QUALIFICATION,
               input=dict(scene=scene, seed=seed, tick=start_tick, delta_u16=delta,
                          horizon=HORIZON, baseline_policy='strong_x16',
                          frozen_input_sha256=digest(canonical(baseline_arm['frozen_spec']))),
               pulse=dict(target_dir=fork['target_dir'], target_point=list(fork['target_point']),
                          original_target_guide_u16=fork['guide'][fork['target_point']],
                          applied_target_guide_u16=min(65535, fork['guide'][fork['target_point']] + delta),
                          duration_decisions=1),
               metrics=dict(initial_decision_changed=changed,
                            baseline_new_paint=len(bnew), fork_new_paint=len(fnew),
                            novel_fork_only=len(fnew - bnew), omitted_baseline=len(bnew - fnew),
                            common_new_paint=len(bnew & fnew),
                            segment_ink_overlap_ppm=round(1000000*len(common)/max(1,len(bpaint | fpaint))),
                            baseline_unique_cursor=len(set(bpath)), fork_unique_cursor=len(set(fpath)),
                            baseline_turns=sum(a['heading'] != b['heading'] for a,b in zip(baseline_arm['events'],baseline_arm['events'][1:])),
                            fork_turns=sum(a['heading'] != b['heading'] for a,b in zip(fork['events'],fork['events'][1:])),
                            end_cursor_equal=bpath[-1] == fpath[-1],
                            mismatched_cursor_ticks=sum(a != b for a,b in zip(bpath[1:],fpath[1:])),
                            material_after_equal=baseline_arm['ink'] == fork['ink']),
               before_material_sha256=_hash_ink(origin),
               baseline_material_sha256=_hash_ink(baseline_arm['ink']),
               fork_material_sha256=_hash_ink(fork['ink']),
               baseline_path=[list(q) for q in bpath], fork_path=[list(q) for q in fpath],
               baseline_events=baseline_arm['events'], fork_events=fork['events'],
               source_baseline_sha256=baseline_arm['source']['metrics']['trajectory_sha256'],
               native_receipt=None, model_decision=None, native_replay_verified=False,
               material_undo_executed=False)
    return rec, baseline_arm, fork


def artifacts(rec, base, fork):
    allowed, ink0 = base['allowed'], base['starting_ink']
    w, h = base['frozen_spec']['canvas']
    def pixel(ink, x, y):
        return bytes((77,69,86,255) if (x,y) not in allowed else
                     (230,217,184,255) if (x,y) in ink else (22,32,43,255))
    bs, fs = base['ink'], fork['ink']
    def delta_pixel(x,y):
        if (x,y) not in allowed: return bytes((77,69,86,255))
        q=(x,y)
        if q in bs and q in fs: return bytes((230,217,184,255))
        if q in bs: return bytes((245,111,99,255))
        if q in fs: return bytes((76,206,184,255))
        return bytes((22,32,43,255))
    return {'case.json':canonical(rec),
            'original.png':rgba_png(w,h,lambda x,y:pixel(ink0,x,y)),
            'baseline.png':rgba_png(w,h,lambda x,y:pixel(bs,x,y)),
            'fork.png':rgba_png(w,h,lambda x,y:pixel(fs,x,y)),
            'difference.png':rgba_png(w,h,delta_pixel)}


def case_id(rec):
    s = rec['input']
    return f"{s['scene']}-seed{s['seed']}-tick{s['tick']}-pulse{s['delta_u16']}"


@lru_cache(maxsize=1)
def source_hashes():
    return {name: digest((Path(__file__).parent / name).read_bytes()) for name in SOURCES}


@lru_cache(maxsize=1)
def prereg_sha256():
    return digest(canonical(preregistration()))


def manifest(rec, data):
    return dict(qualification=QUALIFICATION,
                preregistration_sha256=prereg_sha256(),
                source_sha256=source_hashes(),
                artifacts={k:digest(v) for k,v in data.items()},
                model=None, native=None, native_replay_verified=False)


def verify(folder):
    folder=Path(folder)
    m=json.loads((folder/'manifest.json').read_bytes())
    r=json.loads((folder/'case.json').read_bytes())
    if folder.name != case_id(r):
        raise ValueError('case ID mismatch')
    inp=r['input']
    recomputed=run_case(inp['scene'], inp['seed'], inp['tick'], inp['delta_u16'])
    source_rec, base, fork=recomputed
    blobs=artifacts(*recomputed)
    if r != source_rec or m != manifest(source_rec, blobs):
        raise ValueError('source-recomputed record or manifest mismatch')
    if {p.name for p in folder.iterdir()} != set(blobs) | {'manifest.json'}:
        raise ValueError('missing or added evidence file')
    for n, v in blobs.items():
        if (folder/n).read_bytes() != v:
            raise ValueError('source-recomputed evidence bytes mismatch: '+n)
    return True


def corpus(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    cases=[]
    for scene in SCENES:
        for seed in SEEDS:
            for tick in TICKS:
                for delta in DELTAS:
                    rec, base, fork=run_case(scene,seed,tick,delta)
                    name=case_id(rec)
                    folder=root/name
                    folder.mkdir(parents=True, exist_ok=True)
                    b=artifacts(rec,base,fork)
                    for n,v in b.items(): (folder/n).write_bytes(v)
                    m=manifest(rec,b)
                    (folder/'manifest.json').write_bytes(canonical(m))
                    cases.append(dict(case=name, tick=tick, delta=delta,
                                      initial_decision_changed=rec['metrics']['initial_decision_changed'],
                                      metrics=rec['metrics'],
                                      manifest_sha256=digest(canonical(m))))
    summary={str(delta):dict(n=sum(x['delta']==delta for x in cases),
                             initial_changes=sum(x['initial_decision_changed'] for x in cases if x['delta']==delta),
                             added_new_paint=sum(x['metrics']['novel_fork_only'] for x in cases if x['delta']==delta),
                             omitted_baseline=sum(x['metrics']['omitted_baseline'] for x in cases if x['delta']==delta),
                             total_cursor_mismatch=sum(x['metrics']['mismatched_cursor_ticks'] for x in cases if x['delta']==delta),
                             ending_cursor_different=sum(not x['metrics']['end_cursor_equal'] for x in cases if x['delta']==delta))
             for delta in DELTAS}
    report=dict(version=VERSION,qualification=QUALIFICATION,preregistration_sha256=prereg_sha256(),
                source_sha256=source_hashes(), summary=summary, cases=cases,
                note='One-turn pulse, 32-step synthetic continuation; no native/model/artistic validation')
    (root/'index.json').write_bytes(canonical(report))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(digest(canonical(corpus(Path(args.out)))))
