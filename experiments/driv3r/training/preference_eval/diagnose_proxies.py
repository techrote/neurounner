"""P1 falsification: repeated-control and cross-family geometric-proxy diagnostics.

Uses ONLY original source-fenced P0 image-derived numerical features. Trains
small logistic proxy rankers on one family, evaluates on the other. This is
not a gameplay trajectory test, human preference vote, or native readback.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import rank

PAINTED_GEOMETRY_KEYS = ('cells', 'xspan', 'yspan', 'edges', 'junctions', 'cycles',
                         'endpoints', 'components', 'diagonal_contacts', 'western_cells')
CONTROL_NAMES = ('random', 'fidget', 'wall')
AXES = ('spatial_extent', 'local_weave')


def signature(item):
    """Geometry summary, NOT an original PNG or turn-sequence digest."""
    f = item['features']
    return tuple(f[key] for key in PAINTED_GEOMETRY_KEYS)


def diagnose(path):
    worlds = rank.validate(path)
    rows = [item for world in worlds.values() for item in world]
    counts = Counter(item['controller'] for item in rows)
    if counts != Counter({'learned': 8, 'random': 8, 'fidget': 8, 'wall': 8}):
        raise ValueError('P0 original-controller cohort incomplete')
    families = sorted({item['model_family'] for item in rows})
    if families != ['lfm_w13', 'sprocket_f3']:
        raise ValueError('unexpected family or source identity')
    wall = [item for item in rows if item['controller'] == 'wall']
    clusters = defaultdict(list)
    for item in wall:
        clusters[signature(item)].append(item['id'])
    report = {
        'schema': 'driv3r-preference-proxy-falsification/v1',
        'dataset_sha256': rank.DATA_SHA,
        'evidence_class': 'original source-cited PNG geometry of completed synthetic-W0 episodes',
        'model_forwards_executed_here': 0,
        'native_godot_receipts': 0,
        'human_owner_ratings': 0,
        'trajectory_recovery_or_stop_measured': False,
        'causal_model_observation_sensitivity_proven': False,
        'claims': {
            'source_worlds': len(worlds),
            'learned_model_images': counts['learned'],
            'model_free_control_images': sum(counts[role] for role in CONTROL_NAMES),
            'distinct_wall_control_feature_vectors': len(clusters),
            'largest_repeated_wall_feature_vector': max(map(len, clusters.values())),
            'wall_feature_aliases': [sorted(ids) for ids in clusters.values() if len(ids) > 1],
            'wall_comparison_warning': 'Painted-shape geometry summaries repeat across model families/worlds even where obstacle contacts differ; holdout-world accuracy is not independent morphological generalization',
            'loop_warning': '4-neighbour cycles in final painted pixels are not observed chronological pen loops or intentional revisits',
        },
        'axes': {}
    }
    for axis in AXES:
        pairs, ties = rank.create_pairs(worlds, axis)
        transfer = {}
        for held_out in families:
            train = [pair for pair in pairs if pair['a']['model_family'] != held_out]
            test = [pair for pair in pairs if pair['a']['model_family'] == held_out]
            if not train or not test or any(p['a']['model_family'] != p['b']['model_family'] for p in pairs):
                raise ValueError('source family isolation violated')
            model = rank.fit(train)
            transfer[held_out] = {'training_family': next(f for f in families if f != held_out),
                                  'train_pairs': len(train), **rank.evaluate(model, test)}
        matched = {}
        for control in CONTROL_NAMES:
            n = 0
            wins = 0
            ties_count = 0
            for world, items in sorted(worlds.items()):
                lookup = {item['controller']: item for item in items}
                margin = rank.proxy(lookup['learned'], axis) - rank.proxy(lookup[control], axis)
                if abs(margin) < 0.002:
                    ties_count += 1
                else:
                    n += 1
                    wins += margin > 0
            matched[control] = {'non_tied_pairs': n, 'learned_proxy_wins': wins, 'abstained_ties': ties_count}
        report['axes'][axis] = {
            'same_world_pairs': len(pairs), 'abstained_ties': len(ties),
            'matched_learned_vs_control': matched,
            'leave_entire_model_family_out': transfer,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(Path(__file__).with_name('source_features.csv')))
    ap.add_argument('--output', default=None)
    a = ap.parse_args()
    result = diagnose(a.data)
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if a.output:
        Path(a.output).write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
