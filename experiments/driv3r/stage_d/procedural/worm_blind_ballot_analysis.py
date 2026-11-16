"""Unblind only a complete, source-verified D-2 DEVELOPMENT visual ballot.

The reviewer receives review/ without custody/. This offline processor requires
both the filled anonymous ballot and the original source-replayable custody.
No policy is trained, no images are judged by this module, and no native model
or material action is executed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from worm_sketch import canonical, digest
from worm_blind_gallery_study import VERSION as GALLERY_VERSION, verify

VERSION = 'd2.worm-blind-ballot-analysis/1'
PREF = {'left', 'tie', 'right'}
DIFF = {'none', 'small', 'clear'}
WEIRD = {'left', 'tie', 'right', 'unrated'}


def _read_object(path):
    data = json.loads(Path(path).read_bytes())
    if not isinstance(data, dict):
        raise ValueError('ballot must be a JSON object')
    return data


def _validate(root, filled):
    """Block unblinding if any ratings, identifiers or source assets drift."""
    if not verify(root):
        raise ValueError('source/asset verification failed')
    template = _read_object(Path(root) / 'review' / 'ballot.json')
    if set(filled) != set(template) or filled.get('version') != GALLERY_VERSION:
        raise ValueError('ballot header/version mismatch')
    if filled['status'] != 'RATED':
        raise ValueError('ballot must be declared RATED')
    for key in ('instructions', 'preference_choices', 'difference_choices', 'weirdness_choices'):
        if filled[key] != template[key]:
            raise ValueError('review protocol changed: ' + key)
    if not isinstance(filled['items'], list) or len(filled['items']) != 36:
        raise ValueError('complete 36-item ballot required')
    for expected, actual in zip(template['items'], filled['items']):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('ballot item schema drift')
        for key in ('item', 'context', 'left', 'right'):
            if actual[key] != expected[key]:
                raise ValueError('item order or file identity changed')
        if actual['preference'] not in PREF or actual['difference'] not in DIFF:
            raise ValueError('missing/invalid preference or difference rating')
        if actual['weirdness'] not in WEIRD:
            raise ValueError('invalid optional weirdness rating')
    return template


def evaluate(root, filled_path):
    """Return descriptive, fully enumerated study metrics; never infer artistry.

    Deliberately refuses partial ballots even for a zero-pulse sham. Prevents
    outcome-driven item exclusions and inadvertent early arm disclosure.
    """
    root = Path(root)
    filled = _read_object(filled_path)
    template = _validate(root, filled)
    custody = _read_object(root / 'custody' / 'assignments.json')
    by_item = {a['item']: a for a in custody['assignments']}
    if len(by_item) != 36 or set(by_item) != {x['item'] for x in template['items']}:
        raise ValueError('custody item set mismatch')

    bins = defaultdict(list)
    rows = []
    for mark in filled['items']:
        ref = by_item[mark['item']]
        pulse = ref['pulse_u16']
        preference = mark['preference']
        baseline_side = ref['baseline_side']
        if preference == 'tie':
            decision = 'tie'
        elif preference == baseline_side:
            decision = 'baseline'
        else:
            decision = 'intervention'
        identical = ref['image_sha256']['left'] == ref['image_sha256']['right']
        unexpected = identical and (mark['difference'] != 'none' or mark['preference'] != 'tie'
                                    or mark['weirdness'] not in ('tie', 'unrated'))
        row = dict(item=ref['item'], pulse_u16=pulse, scene=ref['scene'],
                   seed=ref['seed'], tick=ref['tick'],
                   first_decision_changed=ref['first_decision_changed'],
                   images_identical=identical,
                   preference_for=decision, visible_difference=mark['difference'],
                   subjective_weirdness_side=mark['weirdness'],
                   identical_image_disagreement=unexpected)
        rows.append(row)
        bins[pulse].append(row)

    if set(bins) != {0, 4096, 8192} or any(len(bins[p]) != 12 for p in bins):
        raise ValueError('arm denominators changed')
    arms = {}
    for pulse in sorted(bins):
        part = bins[pulse]
        arms[str(pulse)] = dict(
            total=len(part),
            preferences={k:sum(r['preference_for'] == k for r in part)
                         for k in ('intervention', 'baseline', 'tie')},
            visible_differences={k:sum(r['visible_difference'] == k for r in part)
                                 for k in ('none', 'small', 'clear')},
            initial_decision_changed=sum(bool(r['first_decision_changed']) for r in part),
            physically_identical_pairs=sum(r['images_identical'] for r in part),
            identical_image_disagreements=sum(r['identical_image_disagreement'] for r in part),
            per_scene={s:dict(total=sum(r['scene'] == s for r in part),
                              intervention_preferred=sum(r['scene'] == s and r['preference_for'] == 'intervention'
                                                         for r in part))
                       for s in sorted({r['scene'] for r in part})})
    sham = arms['0']
    flagged = sum(r['identical_image_disagreement'] for r in rows)
    return dict(
        version=VERSION, qualification='DEVELOPMENT_SYNTHETIC_HUMAN_BALLOT_NOT_NATIVE',
        frozen_source_head=custody['prerequisite_pr_head'],
        gallery_plan_sha256=custody['plan_sha256'],
        gallery_custody_sha256=digest(canonical(custody)),
        source_blank_ballot_sha256=digest(canonical(template)),
        rated_ballot_sha256=digest(canonical(filled)),
        number_of_rated_items=len(rows), arms=arms,
        sham_catch=dict(total=sham['total'], false_difference=12-sham['visible_differences']['none'],
                        forced_preference=12-sham['preferences']['tie'],
                        identical_image_disagreements=sham['identical_image_disagreements']),
        all_identical_image_disagreements=flagged,
        interpretive_warning=('All 36 items retained including shams/unchanged cases. '
                              'Preference is descriptive, not proof of artistic superiority; '
                              'items share geometries/seeds and are not independent samples. '
                              'Any identical-image disagreement must be disclosed.'),
        records=rows)


def write_analysis(root, filled_path, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('refuse overwrite of prior assessment')
    result = evaluate(root, filled_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gallery', required=True)
    p.add_argument('--rated-ballot', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = write_analysis(a.gallery, a.rated_ballot, a.out)
    print('rated items:', r['number_of_rated_items'])
    print('sham catch:', r['sham_catch'])
    print('report SHA256:', digest(Path(a.out).read_bytes()))
