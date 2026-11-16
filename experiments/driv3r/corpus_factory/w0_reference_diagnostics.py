#!/usr/bin/env python3
"""Read-only, evaluation-only diagnostics for quarantined original W0 corpora.

Never emits action supervision or runnable TRAIN samples. Input is an audited
w0_eval_decisions.py ZIP; the original source archive remains authoritative.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

SCHEMA = 'cybersand.game-corpus.w0-diagnostics/1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _controller(entry):
    name = entry['json_member'].rsplit('/', 1)[-1]
    if entry['origin_kind'] == 'model-assisted':
        return 'model-assisted'
    for label in ('fidget', 'random', 'wall'):
        if f'-{label}-' in name:
            return label
    raise ValueError('unrecognized control arm')


def analyze(source_zip):
    source_zip = Path(source_zip)
    with ZipFile(source_zip) as z:
        names = set(z.namelist())
        if names != {'manifest.json', 'w0-evaluation-decisions.jsonl'} or z.testzip() is not None:
            raise ValueError('malformed evaluation-only ZIP')
        manifest = json.loads(z.read('manifest.json'))
        raw = z.read('w0-evaluation-decisions.jsonl')
    receipt = manifest['files']['w0-evaluation-decisions.jsonl']
    if (len(raw) != receipt['bytes'] or hashlib.sha256(raw).hexdigest() != receipt['sha256']
            or manifest['schema'] != 'cybersand.game-corpus.w0-evaluation-export/1'
            or manifest['split'] != 'evaluation-reference-only'
            or manifest['teacher_labels'] != 0 or manifest['train_examples'] != 0
            or manifest['native_game_records'] != 0 or manifest['model_optimizer_steps'] != 0
            or manifest['archive_crc'] != 'PASS'):
        raise ValueError('evaluation quarantine/inventory mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != receipt['rows'] or len(rows) != manifest['host_legal_receipts']:
        raise ValueError('decision count mismatch')
    origins = {e['json_member']: e for e in manifest['origin_traces']}
    if len(origins) != manifest['original_trace_count']:
        raise ValueError('duplicate or missing provenance roots')
    grouped = defaultdict(list)
    for r in rows:
        source = origins.get(r['source_json_member'])
        if (source is None or r['split'] != 'evaluation-reference-only'
                or r['train_eligible'] is not False or r['teacher_label'] is not None
                or r['evidence_class'] != 'W0-SYNTHETIC-INFERENCE-EVALUATION-ONLY'
                or r['source_zip_sha256'] != manifest['source_zip_sha256']
                or r['source_json_sha256'] != source['json_sha256']
                or r['source_png_sha256'] != source['png_sha256']
                or r['origin_kind'] != source['origin_kind']
                or r['family'] != source['family'] or r['seed'] != source['seed']):
            raise ValueError('source lineage or TRAIN quarantine mismatch')
        obs = r['observation']
        action = r['observed_applied_action_not_teacher']
        if (action not in obs['legal'] or obs['tick'] != r['tick']
                or type(r['tick']) is not int or type(obs['x']) is not int
                or type(obs['y']) is not int or len(r['host_position_after']) != 2
                or not all(type(i) is int and 0 <= i < 96 for i in r['host_position_after'])
                or not (0 <= obs['x'] < 96 and 0 <= obs['y'] < 96)):
            raise ValueError('malformed or illegal W0 observation/action')
        grouped[r['source_json_member']].append(r)
    if set(grouped) != set(origins):
        raise ValueError('source traces absent')
    trajectories = []
    for member, source in sorted(origins.items()):
        group = sorted(grouped[member], key=lambda r: r['tick'])
        if len(group) != source['events'] or [r['tick'] for r in group] != list(range(source['events'])):
            raise ValueError('non-contiguous observation ticks')
        observed_actions = Counter(r['observed_applied_action_not_teacher'] for r in group)
        if dict(observed_actions) != source['observed_action_histogram']:
            raise ValueError('action histogram differs from origin manifest')
        if any((a['host_position_after'] != [b['observation']['x'], b['observation']['y']])
               for a, b in zip(group, group[1:])):
            raise ValueError('non-contiguous host positions')
        first = group[0]['observation']
        seen = {(first['x'], first['y'])}
        novel_by_window = [0, 0, 0, 0]
        repeat_by_window = [0, 0, 0, 0]
        for r in group:
            destination = tuple(r['host_position_after'])
            index = min(r['tick'] // 32, 3)
            if destination in seen:
                repeat_by_window[index] += 1
            else:
                novel_by_window[index] += 1
            seen.add(destination)
        n = len(group)
        entropy = -sum((v / n) * math.log2(v / n) for v in observed_actions.values())
        trajectories.append({
            'family': source['family'], 'seed': source['seed'],
            'arm': _controller(source), 'origin_kind': source['origin_kind'],
            'events': n, 'unique_cells': len(seen), 'novel_arrivals': sum(novel_by_window),
            'novel_by_32_tick_window': novel_by_window,
            'repeated_by_32_tick_window': repeat_by_window,
            'actions': dict(sorted(observed_actions.items())),
            'action_entropy_bits': round(entropy, 4),
            'source_json_sha256': source['json_sha256'],
            'source_png_sha256': source['png_sha256'],
        })
    worlds = defaultdict(dict)
    for t in trajectories:
        key = f"{t['family']}:{t['seed']}"
        if t['arm'] in worlds[key]:
            raise ValueError('duplicate policy arm for matched world')
        worlds[key][t['arm']] = t
    if not all(set(arms) == {'model-assisted', 'fidget', 'random', 'wall'} for arms in worlds.values()):
        raise ValueError('missing matched policy control arm')
    comparison = []
    for world, arms in sorted(worlds.items()):
        model = arms['model-assisted']
        comparison.append({'world': world,
                           'model_unique_cells': model['unique_cells'],
                           'model_novel_by_window': model['novel_by_32_tick_window'],
                           'control_unique_cells': {k: arms[k]['unique_cells']
                                                    for k in ('fidget', 'random', 'wall')},
                           'model_minus_random_unique_cells':
                               model['unique_cells'] - arms['random']['unique_cells']})
    return {'schema': SCHEMA, 'data_class': 'evaluation-reference-only',
            'trainer_eligible_records': 0, 'teacher_labels': 0,
            'source_zip_sha256': manifest['source_zip_sha256'],
            'input_manifest_sha256': hashlib.sha256(canonical(manifest).encode()).hexdigest(),
            'source_decision_jsonl_sha256': receipt['sha256'],
            'world_count': len(worlds), 'source_trajectories': len(trajectories),
            'decision_receipts': len(rows), 'trajectories': trajectories,
            'matched_worlds': comparison,
            'limitations': ['These are previously executed W0 model/control decisions, NEVER teacher targets.',
                            'The synthetic W0 host is not native Godot material simulation.',
                            'Four matched worlds are descriptive, not statistical generalisation.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_quarantined_eval_zip', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable evaluation report destination exists')
    result = analyze(args.original_quarantined_eval_zip)
    args.output.write_text(canonical(result) + '\n', encoding='utf-8')
    print(canonical({'report': str(args.output), 'worlds': result['world_count'],
                     'receipts': result['decision_receipts'],
                     'model_minus_random_unique':
                         [w['model_minus_random_unique_cells'] for w in result['matched_worlds']]}))


if __name__ == '__main__':
    main()
