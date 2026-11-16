"""Source-replayed same-state soft-guidance pulse probes for the D-2 worm.

A pulse is a ONE-STEP counterfactual on a baseline cursor state, not a native
stroke or an edited World. Inputs are the existing frozen DEVELOPMENT fixtures.
This post-hoc diagnostic cannot establish learned-model or artistic quality.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from worm_sketch import DIRS, RNG, canonical, digest, rgba_png
from worm_horizon_fixtures import SCENES, SEEDS, fixture, preregistration
from worm_horizon_study import run_case

VERSION = 'd2.worm-pulse-causal-shadow/1'
TICKS = (12, 60, 108, 156)
DELTAS = (0, 2048, 8192, 32768)
SOURCES = ('worm_pulse_study.py', 'worm_horizon_study.py',
           'worm_horizon_fixtures.py', 'worm_scent_study.py', 'worm_sketch.py')


def score(tick, seed, heading, visits, noise, direction, point, guide_value):
    bend = (direction - heading) % 4
    forward = 42 if bend == 0 else 12 if bend in (1, 3) else -12
    phase = (tick // 13 + seed) % 4
    odd = 40 if direction == phase and tick % 29 < 9 else 0
    return 16 * (guide_value // 1024) + forward + odd - 28 * min(3, visits.get(point, 0)) + noise[direction]


@lru_cache(maxsize=12)
def probes(scene: str, seed: int):
    spec, allowed, guide = fixture(scene, seed)
    baseline = run_case(scene, seed, 'strong_x16')
    step_events = [e for e in baseline['events'] if e['kind'] == 'STEP']
    pos = tuple(spec['origin'])
    heading, visits, rng, ink = spec['initial_heading'], {pos: 1}, RNG(seed), set()
    result = []
    for tick, event in enumerate(step_events):
        legal = [(d, (pos[0] + v[0], pos[1] + v[1])) for d, v in enumerate(DIRS)
                 if (pos[0] + v[0], pos[1] + v[1]) in allowed]
        noise = {d: rng.next() % 73 for d in range(4)}
        choose = lambda pulse_dir=None, pulse_value=None: max(
            legal, key=lambda x: (score(tick, seed, heading, visits, noise, x[0], x[1],
                     pulse_value if x[0] == pulse_dir else guide[x[1]]), -x[0]))
        original_dir, original_pos = choose()
        # Fails closed when source provenance or event ordering drifts.
        if (event['tick'] != tick or tuple(event['from_cursor']) != pos
                or event['heading'] != original_dir or tuple(event['to_cursor']) != original_pos
                or event['legal_catalogue'] != [[d, list(q)] for d, q in legal]
                or event['material_before_sha256'] != digest(canonical(sorted(map(list, ink))))):
            raise ValueError('source-frozen baseline replay diverged')
        if tick in TICKS:
            # Request a left turn when legal, otherwise right/forward/reverse;
            # rule is independent of the baseline *decision*, avoiding winner-selection bias.
            choices = ((heading + 3) % 4, (heading + 1) % 4, heading, (heading + 2) % 4)
            target_dir = next(d for d in choices if any(a == d for a, _ in legal))
            target = next(q for d, q in legal if d == target_dir)
            state = dict(scene=scene, seed=seed, tick=tick, cursor=list(pos),
                         heading=heading, legal_catalogue=[[d, list(q)] for d, q in legal],
                         visits=sorted([list(q) + [n] for q, n in visits.items()]),
                         baseline_choice=[original_dir, list(original_pos)],
                         target=[target_dir, list(target)],
                         material_before_sha256=event['material_before_sha256'],
                         original_guide_u16=guide[target],
                         guide_sha256=spec['guide_u16_sha256'],
                         input_sha256=digest(canonical(spec)),
                         rng_state_after_noise=rng.state,
                         model_decision=None, native_receipt=None)
            for delta in DELTAS:
                bumped = min(65535, guide[target] + delta)
                chosen_dir, chosen_pos = choose(target_dir, bumped)
                rec = dict(version=VERSION, qualification='DEVELOPMENT_SYNTHETIC_SHADOW',
                           state=state, delta_u16=delta, pulsed_guide_u16=bumped,
                           chosen=[chosen_dir, list(chosen_pos)],
                           changed=chosen_dir != original_dir,
                           target_selected=chosen_dir == target_dir,
                           material_changed=False, native_replay_verified=False)
                result.append((rec, set(ink), set(allowed)))
        # Advance the ORIGINAL branch only. No counterfactual state leaks into later ticks.
        if event['pen'] == 'down':
            ink.add(original_pos)
        if event['material_after_sha256'] != digest(canonical(sorted(map(list, ink)))):
            raise ValueError('baseline material receipt chain changed')
        pos, heading = original_pos, original_dir
        visits[pos] = visits.get(pos, 0) + 1
    return tuple(result)


def views(rec, ink, allowed):
    w, h = 80, 80
    cursor = tuple(rec['state']['cursor'])
    target = tuple(rec['state']['target'][1])
    chosen = tuple(rec['chosen'][1])
    def base(x, y):
        return (77, 69, 86, 255) if (x, y) not in allowed else (
            (230, 217, 184, 255) if (x, y) in ink else (22, 32, 43, 255))
    return {'original.png': rgba_png(w, h, lambda x, y: bytes(base(x, y))),
            'choice-overlay.png': rgba_png(w, h, lambda x, y: bytes(
                (86, 206, 224, 255) if (x, y) == chosen else
                (255, 175, 74, 255) if (x, y) == target else
                (253, 85, 114, 255) if (x, y) == cursor else base(x, y)))}


def artifact(rec, ink, allowed):
    return {'probe.json': canonical(rec), **views(rec, ink, allowed)}


@lru_cache(maxsize=1)
def source_hashes():
    base = Path(__file__).parent
    return {name: digest((base / name).read_bytes()) for name in SOURCES}


def case_id(rec):
    s = rec['state']
    return f"{s['scene']}-seed{s['seed']}-tick{s['tick']}-pulse{rec['delta_u16']}"


@lru_cache(maxsize=1)
def input_freeze_hash():
    return digest(canonical(preregistration()))


def manifest(rec, blob):
    return dict(qualification='DEVELOPMENT_SYNTHETIC_SHADOW',
                input_freeze_sha256=input_freeze_hash(),
                source_sha256=source_hashes(),
                native=None, model=None, native_replay_verified=False,
                artifacts={name: digest(value) for name, value in blob.items()})


def replay_case(folder: Path):
    folder = Path(folder)
    m = json.loads((folder / 'manifest.json').read_bytes())
    r = json.loads((folder / 'probe.json').read_bytes())
    s = r['state']
    if folder.name != case_id(r):
        raise ValueError('case identity mismatch')
    if (s['scene'] not in SCENES or s['seed'] not in SEEDS or
            s['tick'] not in TICKS or r['delta_u16'] not in DELTAS):
        raise ValueError('unregistered input')
    original = next((x for x in probes(s['scene'], s['seed']) if case_id(x[0]) == folder.name), None)
    if original is None:
        raise ValueError('source replay missing case')
    rebuilt = artifact(*original)
    if m != manifest(original[0], rebuilt) or set(p.name for p in folder.iterdir()) != set(rebuilt) | {'manifest.json'}:
        raise ValueError('manifest, source or provenance drift')
    for name, raw in rebuilt.items():
        if (folder / name).read_bytes() != raw:
            raise ValueError('source-replayed artifact mismatch: ' + name)
    return True


def corpus(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary = []
    for scene in SCENES:
        for seed in SEEDS:
            for rec, ink, allowed in probes(scene, seed):
                folder = root / case_id(rec)
                folder.mkdir(parents=True, exist_ok=True)
                blobs = artifact(rec, ink, allowed)
                for filename, raw in blobs.items():
                    (folder / filename).write_bytes(raw)
                m = manifest(rec, blobs)
                (folder / 'manifest.json').write_bytes(canonical(m))
                summary.append(dict(case=folder.name, delta_u16=rec['delta_u16'],
                                    baseline_target=rec['state']['baseline_choice'][0] == rec['state']['target'][0],
                                    changed=rec['changed'], target_selected=rec['target_selected'],
                                    manifest_sha256=digest(canonical(m))))
    by_delta = {str(d): dict(n=sum(x['delta_u16'] == d for x in summary),
                            changed=sum(x['changed'] for x in summary if x['delta_u16'] == d),
                            target_selected=sum(x['target_selected'] for x in summary if x['delta_u16'] == d),
                            alternate_target_selected=sum(x['target_selected'] for x in summary
                                                          if x['delta_u16'] == d and not x['baseline_target']))
                for d in DELTAS}
    report = dict(version=VERSION, qualification='DEVELOPMENT_SYNTHETIC_SHADOW',
                  input_freeze_sha256=input_freeze_hash(),
                  source_sha256=source_hashes(), summary=by_delta, records=summary,
                  native=None, model=None)
    (root / 'index.json').write_bytes(canonical(report))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    print(digest(canonical(corpus(Path(args.out)))))
