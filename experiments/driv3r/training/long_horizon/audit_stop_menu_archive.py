"""Read-only, source-pinned independent audit of archived STOP-margin episodes.

This script performs host World replay and file hash verification. It does not
retrain weights, re-infer model logits, run native CyberSand, or open held-out.
Usage: python -m experiments.driv3r.training.long_horizon.audit_stop_menu_archive ARTIFACT.zip
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from .world import World, digest, replay, specs
from .train import MAX_RECOVERIES

VERSION = 'driv3r-stop-menu-external-audit/1'
SEEDS = (584, 585)
ARMS = ('executed_clock-margin0p0', 'dual_clock-margin0p0',
        'dual_clock-margin1p0', 'dual_clock-margin2p5')

def sha(b):
    return hashlib.sha256(b).hexdigest()

def audit(path):
    path = Path(path)
    dev = {s.ident: s for s in specs('development', 1)}
    # Import the exact source World, not a guessed model implementation.
    world_source = Path(__file__).with_name('world.py')
    source_hash = sha(world_source.read_bytes())
    if source_hash != '6801c401c356869ed6535fec5da4a627871cb05fff9c95d6726224a492db678a':
        raise ValueError('world source drift: cannot independently replay')
    counts = Counter()
    by_cohort = defaultdict(dict)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or z.testzip() is not None:
            raise ValueError('duplicate or CRC-corrupt zip entries')
        for seed in SEEDS:
            prefix = f'stop-margin-seed{seed}/'
            report = json.loads(z.read(prefix + 'report.json'))
            if (report['seed'] != seed or report['episodes'] != 128 or
                report['native_receipts'] != 0 or not report['structural_heldout_sealed']):
                raise ValueError('incorrect provenance')
            if report['source_sha256']['world.py'] != source_hash:
                raise ValueError('pinned original world does not match')
            manifest = json.loads(z.read(prefix + 'SHA256SUMS.json'))
            expected = {n[len(prefix):] for n in names if n.startswith(prefix)
                        and n != prefix+'SHA256SUMS.json' and not n.endswith('/')}
            if expected != set(manifest):
                raise ValueError('missing or extra archive members')
            for short, digest_expected in manifest.items():
                if sha(z.read(prefix+short)) != digest_expected:
                    raise ValueError(f'corrupted original evidence {short}')
                counts['sha_files'] += 1
            for short in manifest:
                if short in ('report.json',) or not short.endswith('.json'):
                    continue
                e = json.loads(z.read(prefix+short))
                if (e['version'] != 'driv3r-stop-margin/1' or e['split'] != 'development'
                    or e['native_receipts'] != 0 or e['world_id'] not in dev):
                    raise ValueError('bad episode identity/heldout contamination')
                if e['horizon'] not in (128, 160) or e['style'] not in ('wave','wave_jag'):
                    raise ValueError('unexpected evaluation factor')
                if e['trajectory_sha'] != digest(e['events']):
                    raise ValueError('falsified transcript hash')
                w = replay(dev[e['world_id']],e['events'],MAX_RECOVERIES)
                if (e['base_sha'] != w.base_sha() or e['final_sha'] != w.state_sha()
                    or e['metrics'] != w.metrics(e['horizon'])):
                    raise ValueError('bad World replay/geometry')
                if len(e['events']) != len(e['decisions']):
                    raise ValueError('truncated action/probability list')
                for rec, event in zip(e['decisions'],e['events']):
                    if set(rec['p']) != set(event['legal']):
                        raise ValueError('model distribution not on exact host menu')
                    if abs(sum(rec['p'].values())-1) > 1e-6:
                        raise ValueError('unnormalized model distribution')
                    if rec['chosen'] not in event['legal']:
                        raise ValueError('nonlegal preferred action')
                    if 'STOP' in event['legal'] and any(x.startswith('MOVE_') for x in event['legal']):
                        counts['stop_with_move_offered'] += 1
                    if event['chosen'] == 'STOP':
                        counts['stops'] += 1
                        if event['legal'] == ['STOP']:
                            counts['forced_terminal_stops'] += 1
                        else:
                            counts['optional_stops'] += 1
                    counts['events'] += 1
                counts['episodes'] += 1
                cohort = (seed,e['world_id'],e['style'],e['horizon'],e['forced'])
                if e['variant'] in by_cohort[cohort]:
                    raise ValueError('duplicate experiment arm')
                by_cohort[cohort][e['variant']] = e
        for arms in by_cohort.values():
            if set(arms) != set(ARMS):
                raise ValueError('missing matched arm')
            base = arms['dual_clock-margin0p0']
            for other in ('dual_clock-margin1p0','dual_clock-margin2p5'):
                e = arms[other]
                if e['events'] != base['events']:
                    counts['margin_trajectory_differences'] += 1
                if e['stop_overrides']:
                    counts['margin_stop_overrides'] += e['stop_overrides']
            counts['matched_quartets'] += 1
    return {'version': VERSION, 'original_zip_sha256':sha(path.read_bytes()),
            'world_source_sha256': source_hash,
            'counts':dict(sorted(counts.items())),
            'assurance':'source-World replay, CRC, SHA, legal menu and trajectory equality; NOT independently recomputed neural logits',
            'native_receipts':0, 'structural_heldout_read':False}

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('original_zip',type=Path)
    a=p.parse_args();print(json.dumps(audit(a.original_zip),sort_keys=True,indent=2))
