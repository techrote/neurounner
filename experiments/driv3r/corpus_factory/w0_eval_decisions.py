#!/usr/bin/env python3
"""Read-only W0 decision evidence, never optimizer/TRAIN/teacher records.

Decision-level complement to ingest_w0.py's trace summaries.
Original model and control actions are evaluation observations, NOT target truth.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ACTIONS = frozenset(('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'STOP'))
CONTRACT = 'cybersand.worm.observation-action/0-experimental'
CLASS = 'W0-SYNTHETIC-INFERENCE-EVALUATION-ONLY'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def extract(archive: Path, output: Path, max_events: int = 4096):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if output.exists() or output == archive:
        raise ValueError('refusing existing output or original archive overwrite')
    if max_events < 1 or max_events > 4096:
        raise ValueError('invalid maximum events')
    original_sha = digest(archive.read_bytes())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('original ZIP CRC failure')
        names = set()
        for member in z.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or member.filename in names:
                raise ValueError('unsafe or repeated original member')
            names.add(member.filename)
        inventories = 0
        for sums_name in sorted(n for n in names if n.endswith('/results/SHA256SUMS.json')):
            prefix = sums_name.rsplit('/', 1)[0] + '/'
            for rel, expected_sha in json.loads(z.read(sums_name)).items():
                name = prefix + rel
                if name not in names or digest(z.read(name)) != expected_sha:
                    raise ValueError(f'original member SHA mismatch: {name}')
                inventories += 1
        rows, traces = [], []
        originals = sorted(n for n in names if '/original-png-and-json/' in n and n.endswith('.json'))
        if not originals:
            raise ValueError('no W0 original traces')
        for name in originals:
            raw = z.read(name)
            trace = json.loads(raw)
            if trace.get('contract') != CONTRACT or trace.get('size') != 96:
                raise ValueError(f'wrong W0 contract/size: {name}')
            events, positions = trace['events'], trace['trajectory']
            if (type(trace.get('steps_executed')) is not int
                    or len(events) != trace['steps_executed']
                    or not 1 <= len(events) <= max_events
                    or len(positions) != len(events) + 1
                    or trace.get('step_budget', 0) < len(events)):
                raise ValueError(f'invalid W0 trajectory bound: {name}')
            png_name = name[:-5] + '.png'
            if png_name not in names:
                raise ValueError(f'original PNG missing: {png_name}')
            png = z.read(png_name)
            if not png.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError(f'invalid original PNG: {png_name}')
            trace_sha, png_sha = digest(raw), digest(png)
            kind = 'model-assisted' if 'score_receipts' in trace else 'model-free-control'
            if kind == 'model-assisted' and len(trace['score_receipts']) != len(events):
                raise ValueError(f'model score inventory mismatch: {name}')
            actions = Counter()
            for i, event in enumerate(events):
                obs = event['observation']
                legal = obs['legal']
                if (obs.get('tick') != i
                        or [obs.get('x'), obs.get('y')] != positions[i]
                        or event.get('position_after') != positions[i + 1]
                        or obs.get('family') != trace['family']
                        or len(obs.get('local_5x5', [])) != 5
                        or obs['local_5x5'][2][2] != '@'
                        or len(legal) != len(set(legal))
                        or not set(legal).issubset(ACTIONS)
                        or event['applied'] not in legal
                        or event.get('raw_legal') is not (event.get('raw_choice') in legal)):
                    raise ValueError(f'W0 host observation/receipt mismatch: {name}:{i}')
                actions[event['applied']] += 1
                rows.append({
                    'schema': 'cybersand.game-corpus.w0-evaluation-decision/1',
                    'evidence_class': CLASS, 'split': 'evaluation-reference-only',
                    'train_eligible': False, 'teacher_label': None,
                    'source_zip_sha256': original_sha, 'source_json_member': name,
                    'source_json_sha256': trace_sha, 'source_png_sha256': png_sha,
                    'origin_kind': kind, 'seed': trace['seed'], 'family': trace['family'],
                    'tick': i, 'observation': obs,
                    'observed_applied_action_not_teacher': event['applied'],
                    'host_position_after': event['position_after'],
                })
            traces.append({'json_member': name, 'json_sha256': trace_sha,
                           'png_member': png_name, 'png_sha256': png_sha,
                           'origin_kind': kind, 'seed': trace['seed'],
                           'family': trace['family'], 'events': len(events),
                           'observed_action_histogram': dict(sorted(actions.items()))})
    output.mkdir(parents=True, exist_ok=False)
    content = ''.join(canonical(row) + '\n' for row in rows).encode('utf8')
    (output / 'w0-evaluation-decisions.jsonl').write_bytes(content)
    manifest = {
        'schema': 'cybersand.game-corpus.w0-evaluation-export/1',
        'source_zip_sha256': original_sha,
        'split': 'evaluation-reference-only', 'native_game_records': 0,
        'train_examples': 0, 'teacher_labels': 0,
        'model_optimizer_steps': 0, 'host_legal_receipts': len(rows),
        'original_trace_count': len(traces), 'original_png_count': len(traces),
        'sha256_member_receipts_checked': inventories,
        'archive_crc': 'PASS', 'origin_traces': traces,
        'files': {'w0-evaluation-decisions.jsonl': {
            'sha256': digest(content), 'bytes': len(content), 'rows': len(rows)}},
        'restriction': 'Evaluation only: never train on observed model/control actions; original archive is authoritative.',
    }
    (output / 'manifest.json').write_text(canonical(manifest) + '\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description='Source-pinned W0 evaluation-only decision quarantine')
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-events', type=int, default=4096)
    args = p.parse_args()
    result = extract(args.archive, args.output, args.max_events)
    print(canonical({k: v for k, v in result.items() if k != 'origin_traces'}))


if __name__ == '__main__':
    main()
