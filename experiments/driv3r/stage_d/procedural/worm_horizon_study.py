"""Input-frozen, post-freeze D-2 synthetic drawing-worm escape-horizon A/B.

No native/model execution, host receipts, historical Undo, or claimed aesthetic
success. This records only cardinal legal synthetic brush movement. Unmodified
source-frozen worm_horizon_fixtures defines inputs independently of outcomes.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from worm_sketch import DIRS, RNG, canonical, digest, rgba_png
from worm_horizon_fixtures import SCENES, SEEDS, fixture, preregistration

VERSION = 'd2.worm-escape-horizon/1'
ARMS = ('strong_x16', 'short_escape', 'sustained_escape', 'sham_sustained', 'always_novel')
WINDOW = 12
MIN_UNIQUE = 10
PROFILES = {'short_escape': (4, 5, 24, 8, 120, 1),
            'sustained_escape': (12, 12, 64, 12, 200, 0),
            'sham_sustained': (12, 12, 64, 12, 0, 16)}


def run_case(scene: str, seed: int, arm: str, steps: int = 192):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    if type(steps) is not int or not 1 <= steps <= 512:
        raise ValueError('steps out of bounded range')
    source_spec, allowed, guide = fixture(scene, seed)
    pos = tuple(source_spec['origin'])
    heading = source_spec['initial_heading']
    visits = {pos: 1}
    recent = deque([pos], maxlen=WINDOW)
    rng = RNG(seed)
    trail, material, events = [pos], set(), []
    active, escape_start, cooldown_until, fresh = False, -1, 0, set()
    activations = releases = stuck = 0
    min_ticks, exit_fresh, timeout, cooldown, novelty_penalty, escape_gain = PROFILES.get(arm, (0, 0, 0, 0, 0, 16))
    for tick in range(steps):
        legal = [(d, (pos[0]+v[0], pos[1]+v[1])) for d, v in enumerate(DIRS)
                 if (pos[0]+v[0], pos[1]+v[1]) in allowed]
        if not legal:
            events.append(dict(tick=tick, kind='STOP', reason='no cardinal permitted move'))
            break
        unique_recent = len(set(recent))
        if arm in PROFILES and not active and len(recent) == WINDOW and unique_recent <= MIN_UNIQUE and tick >= cooldown_until:
            active, escape_start, fresh = True, tick, set()
            activations += 1
            events.append(dict(kind='ESCAPE_ENTER', tick=tick, recent_unique=unique_recent,
                               material_changed=False, native_receipt=None))
        if active and tick > escape_start:
            elapsed = tick - escape_start
            if (elapsed >= min_ticks and len(fresh) >= exit_fresh) or elapsed >= timeout:
                reason = 'fresh' if elapsed >= min_ticks and len(fresh) >= exit_fresh else 'timeout'
                active, cooldown_until = False, tick + cooldown
                releases += 1
                events.append(dict(kind='ESCAPE_EXIT', tick=tick, reason=reason,
                                   escape_ticks=elapsed, fresh_cells=len(fresh),
                                   material_changed=False, native_receipt=None))
        phase = (tick // 13 + seed) % 4
        noise = {d: rng.next() % 73 for d in range(4)}
        profile_on = active and arm in PROFILES
        gain = escape_gain if profile_on else (1 if arm == 'always_novel' else 16)
        extra = novelty_penalty if profile_on else (120 if arm == 'always_novel' else 0)
        def score(candidate):
            d, q = candidate
            bend = (d - heading) % 4
            heading_bonus = 42 if bend == 0 else 12 if bend in (1, 3) else -12
            oddity = 40 if d == phase and tick % 29 < 9 else 0
            return (gain * (guide[q] // 1024) + heading_bonus + oddity
                    - (28 + extra) * min(3, visits.get(q, 0)) + noise[d])
        direction, target = max(legal, key=lambda a: (score(a), -a[0]))
        before = digest(canonical(sorted(map(list, material))))
        down = tick % source_spec['pen_rhythm']['period'] >= source_spec['pen_rhythm']['lifted']
        new = target not in visits
        if down:
            material.add(target)
        after = digest(canonical(sorted(map(list, material))))
        events.append(dict(tick=tick, kind='STEP', from_cursor=list(pos),
                           to_cursor=list(target), heading=direction,
                           legal_catalogue=[[d, list(q)] for d, q in legal],
                           guide_u16=guide[target], gain=gain, extra_revisit_penalty=extra,
                           escape_mode=profile_on and arm not in ('sham_sustained',),
                           first_visit=new, pen='down' if down else 'up',
                           material_before_sha256=before, material_after_sha256=after,
                           changed=before != after, native_receipt=None, model_decision=None))
        pos, heading = target, direction
        trail.append(pos)
        recent.append(pos)
        if new and active:
            fresh.add(pos)
        visits[pos] = visits.get(pos, 0) + 1
        stuck = stuck + 1 if visits[pos] >= 5 else 0
        if stuck >= 8:
            events.append(dict(tick=tick, kind='STOP', reason='repeated local oscillation'))
            break
    events.append(dict(tick=len(trail) - 1, kind='UNDO_MATERIAL_EDIT_UNAVAILABLE',
                       executed=False, native_receipt=None))
    moves = [e for e in events if e['kind'] == 'STEP']
    headings = [e['heading'] for e in moves]
    advantage = round(sum(e['guide_u16'] - sum(guide[tuple(q)] for _, q in e['legal_catalogue']) // len(e['legal_catalogue']) for e in moves) / len(moves)) if moves else 0
    spec = dict(version=VERSION, input=source_spec, arm=arm, steps=steps,
                profile=list(PROFILES[arm]) if arm in PROFILES else None,
                window=WINDOW, min_unique=MIN_UNIQUE)
    metrics = dict(steps=len(moves), painted_cells=len(material),
                   unique_cursor_cells=len(set(trail)),
                   repeats=len(trail)-len(set(trail)),
                   turns=sum(a != b for a, b in zip(headings, headings[1:])),
                   escape_triggers=activations, escape_exits=releases,
                   escape_steps=sum(e['escape_mode'] for e in moves),
                   guide_advantage_u16=advantage,
                   stop=next((e['reason'] for e in events if e['kind'] == 'STOP'), 'budget'),
                   material_sha256=digest(canonical(sorted(map(list, material)))),
                   trajectory_sha256=digest(canonical(trail)))
    return dict(spec=spec, material=material, trail=trail, events=events,
                permitted=allowed, guide=guide, metrics=metrics)


def blobs(outcome):
    spec, allowed, guide, material = (outcome[x] for x in ('spec', 'permitted', 'guide', 'material'))
    w, h = spec['input']['canvas']
    return {'recipe.json': canonical(spec), 'events.json': canonical(outcome['events']),
            'trajectory.json': canonical(outcome['trail']), 'metrics.json': canonical(outcome['metrics']),
            'guidance.u16.json': canonical([guide[x, y] for y in range(h) for x in range(w)]),
            'original.png': rgba_png(w, h, lambda x, y:
                bytes((77, 69, 86, 255)) if (x, y) not in allowed else bytes((22, 32, 43, 255))),
            'material.png': rgba_png(w, h, lambda x, y:
                bytes((77, 69, 86, 255)) if (x, y) not in allowed else
                bytes((230, 217, 184, 255)) if (x, y) in material else bytes((22, 32, 43, 255)))}


def verify(folder: Path):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    spec = json.loads((folder / 'recipe.json').read_bytes())
    if (manifest['qualification'] != 'DEVELOPMENT_SYNTHETIC_ONLY'
            or manifest['native'] is not None or manifest['model'] is not None
            or manifest['native_replay_verified'] is not False):
        raise ValueError('synthetic provenance misrepresented')
    dependencies = {'worm_horizon_fixtures.py': Path(__import__('worm_horizon_fixtures').__file__),
                    'worm_sketch.py': Path(__import__('worm_sketch').__file__),
                    'worm_scent_study.py': Path(__import__('worm_scent_study').__file__)}
    expected_sources = {'worm_horizon_study.py': digest(Path(__file__).read_bytes()),
                        **{n: digest(p.read_bytes()) for n, p in dependencies.items()}}
    if manifest['sources'] != expected_sources:
        raise ValueError('exact source mismatch')
    out = run_case(spec['input']['scene'], spec['input']['seed'], spec['arm'], spec['steps'])
    b = blobs(out)
    if spec != out['spec'] or manifest['metrics'] != out['metrics']:
        raise ValueError('input/metrics drift')
    if manifest['artifacts'] != {k: digest(v) for k, v in b.items()}:
        raise ValueError('artifact manifest mismatch')
    if {p.name for p in folder.iterdir()} != set(b) | {'manifest.json'}:
        raise ValueError('missing or extra artifact')
    for k, v in b.items():
        if (folder / k).read_bytes() != v:
            raise ValueError('tampered original bytes: '+k)
    return True


def corpus(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dependencies = {'worm_horizon_fixtures.py': Path(__import__('worm_horizon_fixtures').__file__),
                    'worm_sketch.py': Path(__import__('worm_sketch').__file__),
                    'worm_scent_study.py': Path(__import__('worm_scent_study').__file__)}
    sources = {'worm_horizon_study.py': digest(Path(__file__).read_bytes()),
               **{n: digest(p.read_bytes()) for n, p in dependencies.items()}}
    records = []
    for scene in SCENES:
        for seed in SEEDS:
            for arm in ARMS:
                out = run_case(scene, seed, arm)
                case = f'{scene}-seed{seed}-{arm}'
                folder = root / case
                folder.mkdir(parents=True, exist_ok=True)
                b = blobs(out)
                for name, value in b.items():
                    (folder / name).write_bytes(value)
                manifest = dict(qualification='DEVELOPMENT_SYNTHETIC_ONLY',
                                native=None, model=None, native_replay_verified=False,
                                sources=sources, input_freeze_sha256=digest(canonical(preregistration())),
                                metrics=out['metrics'], artifacts={k: digest(v) for k, v in b.items()})
                (folder / 'manifest.json').write_bytes(canonical(manifest))
                if not verify(folder):
                    raise ValueError('replay failed')
                records.append(dict(case=case, scene=scene, seed=seed, arm=arm,
                                    metrics=out['metrics'], manifest_sha256=digest(canonical(manifest))))
    report = dict(version=VERSION, qualification='DEVELOPMENT_SYNTHETIC_ONLY',
                  input_freeze_sha256=digest(canonical(preregistration())),
                  outcomes=records, native=None, model=None)
    (root / 'index.json').write_bytes(canonical(report))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(digest(canonical(corpus(Path(a.out)))))
