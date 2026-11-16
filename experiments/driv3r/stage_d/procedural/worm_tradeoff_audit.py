"""Post-hoc, source-replayed diagnostic of frozen D-2 drawing-worm sketches.

No learning, native receipts, human art judgement, or confirmatory claim. The
input cohort and policy choices were already inspected before these secondary
metrics were selected. Reject absent/tampered original synthetic receipts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from worm_horizon_fixtures import SCENES, SEEDS
from worm_horizon_study import ARMS, verify
from worm_sketch import canonical, digest

VERSION = 'd2.worm-tradeoff-posthoc/1'


def read_case(folder: Path):
    if not verify(folder):
        raise ValueError('source replay failed')
    recipe = json.loads((folder / 'recipe.json').read_bytes())
    events = json.loads((folder / 'events.json').read_bytes())
    trail = [tuple(point) for point in json.loads((folder / 'trajectory.json').read_bytes())]
    metrics = json.loads((folder / 'metrics.json').read_bytes())
    guide = json.loads((folder / 'guidance.u16.json').read_bytes())
    width, height = recipe['input']['canvas']
    if len(guide) != width * height:
        raise ValueError('invalid guidance dimensions')
    steps = [e for e in events if e['kind'] == 'STEP']
    if len(steps) != metrics['steps'] or len(trail) != len(steps) + 1:
        raise ValueError('wrong event count')
    material = {tuple(e['to_cursor']) for e in steps if e['pen'] == 'down'}
    if len(material) != metrics['painted_cells']:
        raise ValueError('inconsistent material count')
    hi = 0
    for e in steps:
        x, y = e['to_cursor']
        legal = [guide[q[1] * width + q[0]] for _, q in e['legal_catalogue']]
        if e['guide_u16'] != guide[y * width + x]:
            raise ValueError('guide event does not match original raster')
        hi += int(e['guide_u16'] == max(legal))
    if material:
        xs, ys = zip(*material)
        bbox = (max(xs)-min(xs)+1)*(max(ys)-min(ys)+1)
    else:
        bbox = 0
    record = {
        'painted_cells': len(material),
        'turns': metrics['turns'],
        'repeated_visits': metrics['repeats'],
        'guide_advantage_u16': metrics['guide_advantage_u16'],
        'max_guide_choices': hi,
        'steps': len(steps),
        'ink_bbox_area': bbox,
        'distinct_directed_edges': len({(trail[i], trail[i+1]) for i in range(len(trail)-1)}),
    }
    return recipe, trail, material, record


def paired_report(root: Path):
    root = Path(root)
    expected = {f'{s}-seed{k}-{a}' for s in SCENES for k in SEEDS for a in ARMS}
    actual = {p.name for p in root.iterdir() if p.is_dir()}
    if actual != expected:
        raise ValueError('missing/unexpected case folders: ' + repr(sorted(expected ^ actual)))
    rows = []
    for s in SCENES:
        for k in SEEDS:
            records = {}
            input_ids = set()
            for arm in ARMS:
                recipe, trail, material, metrics = read_case(root / f'{s}-seed{k}-{arm}')
                if recipe['arm'] != arm or recipe['input']['scene'] != s or recipe['input']['seed'] != k:
                    raise ValueError('case recipe mismatch')
                input_ids.add(digest(canonical(recipe['input'])))
                records[arm] = (trail, material, metrics)
            if len(input_ids) != 1:
                raise ValueError('non-matched initial input')
            if records['sham_sustained'][0:2] != records['strong_x16'][0:2]:
                raise ValueError('sham does not preserve baseline')
            base_t, base_m, b = records['strong_x16']
            new_t, new_m, n = records['sustained_escape']
            first = next((i for i, (x, y) in enumerate(zip(base_t, new_t)) if x != y), None)
            intersection = len(base_m & new_m)
            union = len(base_m | new_m)
            rows.append({
                'scene': s, 'seed': k, 'input_sha256': next(iter(input_ids)),
                'baseline': b, 'sustained': n,
                'delta_painted': n['painted_cells'] - b['painted_cells'],
                'delta_repeats': n['repeated_visits'] - b['repeated_visits'],
                'delta_guide_advantage_u16': n['guide_advantage_u16'] - b['guide_advantage_u16'],
                'delta_max_guide_choices': n['max_guide_choices'] - b['max_guide_choices'],
                'new_ink_not_in_baseline': len(new_m-base_m),
                'baseline_ink_missing_in_new': len(base_m-new_m),
                'ink_jaccard_numerator': intersection,
                'ink_jaccard_denominator': union,
                'first_cursor_divergence_step': first,
                'sham_identical': True,
            })
    totals = {key: sum(r[key] for r in rows) for key in (
        'delta_painted', 'delta_repeats', 'delta_guide_advantage_u16',
        'delta_max_guide_choices', 'new_ink_not_in_baseline',
        'baseline_ink_missing_in_new')}
    totals['paired_improvements'] = sum(r['delta_painted'] > 0 for r in rows)
    totals['paired_regressions'] = sum(r['delta_painted'] < 0 for r in rows)
    totals['paired_unchanged'] = sum(r['delta_painted'] == 0 for r in rows)
    totals['guide_advantage_regressions'] = sum(r['delta_guide_advantage_u16'] < 0 for r in rows)
    totals['local_max_choice_decreases'] = sum(r['delta_max_guide_choices'] < 0 for r in rows)
    return {'version': VERSION, 'qualification': 'POSTHOC_DEVELOPMENT_SYNTHETIC_ONLY',
            'not_preregistered': ['ink_overlap', 'local_max_guide_choices', 'ink_bbox_area', 'directed_edges'],
            'native': None, 'model': None, 'human_art_judgment': None,
            'cases': rows, 'totals': totals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = paired_report(Path(args.corpus))
    out = canonical(report)
    Path(args.out).write_bytes(out)
    print(digest(out))


if __name__ == '__main__':
    main()
