"""Assessor-blind (not cryptographically secret) visual preference pilot.

Reconstruct source-fenced synthetic 32-step pulse continuations, then render
anonymous LEFT/RIGHT comparisons without policy labels. Never mix custody
(arm identities, hashes, inputs) into the reviewer packet. This is DEVELOPMENT,
not model/native inference or evidence of artistic superiority.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from worm_sketch import canonical, digest, rgba_png
from worm_horizon_fixtures import SCENES, SEEDS
from worm_pulse_horizon_study import run_case, source_hashes, prereg_sha256

VERSION = 'd2.worm-blind-gallery/1'
PULSES = (0, 4096, 8192)
TICKS_BY_SEED = {23: 12, 79: 108}
QUALIFICATION = 'DEVELOPMENT_SYNTHETIC_SOURCE_REPLAYED_NOT_NATIVE'


def plan():
    """Predeclared 12 inputs x 3 arms; no selection by observed success."""
    tasks = [(scene, seed, TICKS_BY_SEED[seed], pulse)
             for scene in SCENES for seed in SEEDS for pulse in PULSES]
    return sorted(tasks, key=lambda t: digest(canonical(list(t))))


def _art(allowed, ink, width, height):
    dark = bytes((22, 32, 43, 255))
    blocked = bytes((77, 69, 86, 255))
    painted = bytes((230, 217, 184, 255))
    return rgba_png(width, height, lambda x, y:
                    blocked if (x, y) not in allowed else
                    painted if (x, y) in ink else dark)


def _pngs(case, baseline, fork, baseline_left):
    allowed = baseline['allowed']
    w, h = baseline['frozen_spec']['canvas']
    original = _art(allowed, baseline['starting_ink'], w, h)
    original_baseline = _art(allowed, baseline['ink'], w, h)
    intervention = _art(allowed, fork['ink'], w, h)
    return dict(context=original,
                left=original_baseline if baseline_left else intervention,
                right=intervention if baseline_left else original_baseline)


def _build_item(num, task):
    scene, seed, tick, pulse = task
    rec, baseline, fork = run_case(scene, seed, tick, pulse)
    left_is_baseline = num % 2 == 0  # balanced 18/18, independently ordered tasks
    pics = _pngs(rec, baseline, fork, left_is_baseline)
    if pulse == 0:
        assert pics['left'] == pics['right']  # sham catch trial
    label = f'item-{num+1:03d}'
    info = dict(item=label, scene=scene, seed=seed, tick=tick,
                pulse_u16=pulse, baseline_side='left' if left_is_baseline else 'right',
                first_decision_changed=rec['metrics']['initial_decision_changed'],
                mismatch_ticks=rec['metrics']['mismatched_cursor_ticks'],
                novel_paint=rec['metrics']['novel_fork_only'],
                omitted_paint=rec['metrics']['omitted_baseline'],
                baseline_material_sha256=rec['baseline_material_sha256'],
                intervention_material_sha256=rec['fork_material_sha256'],
                frozen_input_sha256=rec['input']['frozen_input_sha256'],
                image_sha256={name:digest(buf) for name,buf in pics.items()})
    return label, pics, info


def corpus(destination):
    destination = Path(destination)
    review, custody = destination / 'review', destination / 'custody'
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('destination must be empty to avoid stale/unblinded assets')
    review.mkdir(parents=True, exist_ok=True)
    custody.mkdir(parents=True, exist_ok=True)
    tasks = plan()
    items, assignments = [], []
    for i, task in enumerate(tasks):
        label, imgs, info = _build_item(i, task)
        for kind, data in imgs.items():
            (review / f'{label}-{kind}.png').write_bytes(data)
        items.append(dict(item=label,
                          context=f'{label}-context.png',
                          left=f'{label}-left.png', right=f'{label}-right.png',
                          preference='unrated', difference='unrated',
                          weirdness='unrated'))
        assignments.append(info)
    ballot = dict(version=VERSION, status='UNRATED',
                  instructions=('Compare two 32-step line drawings using their shared '
                                'unlabelled start context. No model or native PAINT is '
                                'shown. Treat identical pairs as valid catches.'),
                  preference_choices=['left', 'tie', 'right'],
                  difference_choices=['none', 'small', 'clear'],
                  weirdness_choices=['left', 'tie', 'right'], items=items)
    # Neither original source names nor arm/delta/seed labels are in the review packet.
    (review / 'ballot.json').write_bytes(canonical(ballot))
    custody_record = dict(version=VERSION, qualification=QUALIFICATION,
                          prerequisite_pr_head='07399c60873e35a7e12e4660bc63a926c9f5d701',
                          frozen_source_sha256=source_hashes(),
                          frozen_preregistration_sha256=prereg_sha256(),
                          plan_sha256=digest(canonical(tasks)),
                          review_ballot_sha256=digest(canonical(ballot)),
                          assignments=assignments,
                          note='Hold this custody file away from assessors until their ballots are frozen; no native/model/art evaluation yet.')
    (custody / 'assignments.json').write_bytes(canonical(custody_record))
    return custody_record


def verify(destination):
    root = Path(destination)
    manifest = root/'custody'/'assignments.json'
    if not manifest.is_file(): raise ValueError('missing custody')
    import json
    stored = json.loads(manifest.read_bytes())
    if stored['frozen_source_sha256'] != source_hashes() or stored['frozen_preregistration_sha256'] != prereg_sha256():
        raise ValueError('source/preregistration custody mismatch')
    if stored['plan_sha256'] != digest(canonical(plan())) or len(stored['assignments']) != len(plan()):
        raise ValueError('plan mismatch')
    items = stored['assignments']
    if sum(i['baseline_side'] == 'left' for i in items) != len(items)//2:
        raise ValueError('side imbalance')
    expected_paths = {'ballot.json'}
    for j, task in enumerate(plan()):
        name, imgs, info = _build_item(j, task)
        if info != items[j]: raise ValueError('original source reconstruction mismatch')
        for kind, b in imgs.items():
            p = root/'review'/f'{name}-{kind}.png'
            expected_paths.add(p.name)
            if not p.is_file() or p.read_bytes() != b:
                raise ValueError('review bytes differ from source: '+p.name)
    expected_ballot = dict(version=VERSION, status='UNRATED',
                           instructions=('Compare two 32-step line drawings using their shared '
                                         'unlabelled start context. No model or native PAINT is '
                                         'shown. Treat identical pairs as valid catches.'),
                           preference_choices=['left', 'tie', 'right'],
                           difference_choices=['none', 'small', 'clear'],
                           weirdness_choices=['left', 'tie', 'right'],
                           items=[dict(item=f'item-{k+1:03d}',
                                       context=f'item-{k+1:03d}-context.png',
                                       left=f'item-{k+1:03d}-left.png',
                                       right=f'item-{k+1:03d}-right.png',
                                       preference='unrated', difference='unrated', weirdness='unrated')
                                  for k in range(len(plan()))])
    if ((root/'review'/'ballot.json').read_bytes() != canonical(expected_ballot)
        or stored['review_ballot_sha256'] != digest(canonical(expected_ballot))
        or {p.name for p in (root/'review').iterdir()} != expected_paths):
        raise ValueError('review index or file-list mismatch')
    if {p.name for p in (root/'custody').iterdir()} != {'assignments.json'}:
        raise ValueError('unexpected custody file')
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    corpus(args.out)
    print('verified:', verify(args.out))
