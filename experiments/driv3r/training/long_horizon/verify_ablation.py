"""Independent offline verifier for the saved development-only 128-step evidence.

Validates all SHA256SUMS, immutable world spec IDs, per-step replay receipts,
legal menus, neural checkpoint selection and logged probabilities. No native
backend, model hub, training, or heldout evaluations are invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .compare_curricula import ARMS
from .policy import Ranker
from .train import source_pins, sha_file, aggregate
from .world import Spec, World, replay, digest


def verify(root: Path, enforce_source: bool = True) -> dict:
    report = json.loads((root/'report.json').read_text())
    manifest = json.loads((root/'SHA256SUMS.json').read_text())
    if report['heldout_used_in_optimizer_or_evaluation'] or report['original_v1_final_heldout_reused']:
        raise ValueError('Invalid heldout separation claim')
    actual = {str(p.relative_to(root)):sha_file(p)
              for p in sorted(root.rglob('*')) if p.is_file()
              and p.name != 'SHA256SUMS.json'}
    if manifest != actual:
        raise ValueError('Artifact SHA256 manifest mismatch')
    if enforce_source:
        expected = {**source_pins(), 'compare_curricula.py':sha_file(Path(__file__).with_name('compare_curricula.py'))}
        if expected != report['source_sha256']:
            raise ValueError('Exact source SHA256 mismatch')
    source_specs = {}
    for desc in report['worlds']:
        spec = Spec('development',desc['family'],desc['seed'],desc['mode'],desc['challenge'])
        if spec.ident != desc['id'] or World.create(spec).base_sha() != desc['base_sha256']:
            raise ValueError('World source/spec mismatch')
        source_specs[spec.ident] = spec
    if not source_specs:
        raise ValueError('Empty development fixture cohort')
    ckpts = {}
    for entry in report['training']:
        if entry['updates'] != 3*entry['stages'][0]['optimizer']['updates']:
            raise ValueError('Unexpected optimizer update total')
        cp = root/entry['checkpoint_filename']
        if sha_file(cp) != entry['checkpoint_sha256']:
            raise ValueError('Wrong checkpoint bytes')
        model = Ranker.restore(cp)
        if model.updates != entry['updates']:
            raise ValueError('Checkpoint update count mismatch')
        ckpts[(str(entry['seed']),entry['curriculum'])] = model
    audited = Counter()
    for seed_text, by_horizon in report['results'].items():
        if '128' not in by_horizon:
            raise ValueError('No 128-decision cohort')
        for arm in ARMS:
            folder = root/'trajectories'/f'seed{seed_text}'/arm
            files = sorted(folder.glob('*.json'))
            pngs = sorted(folder.glob('*.png'))
            if len(files) != len(source_specs) or len(pngs) != len(source_specs):
                raise ValueError('Incomplete original trajectories/gallery')
            evidence = []
            for path in files:
                run = json.loads(path.read_text())
                if run['split'] != 'development' or run['horizon'] != 128 or run['native_executed']:
                    raise ValueError('Evidence boundary violation')
                spec = source_specs.get(run['spec_id'])
                if spec is None or run['world_base_sha256'] != World.create(spec).base_sha():
                    raise ValueError('Unmatched world/source')
                if str(run['policy_seed']) != seed_text:
                    raise ValueError('Run seed mismatch')
                if run['metrics']['decisions'] > 128 or run['metrics']['illegal_host_actions']:
                    raise ValueError('Invalid game decision count')
                model = ckpts.get((seed_text,arm))
                if arm in ARMS[:2] and model is None:
                    raise ValueError('Missing provenance-pinned trained checkpoint')
                if arm not in ARMS[:2] and model is not None:
                    raise ValueError('Inappropriate model attribution')
                world = World.create(spec)
                original_events = []
                for event in run['events']:
                    rec = {k:v for k,v in event.items() if k != 'policy_probabilities'}
                    if model is not None:
                        pred, prob = model.choose(world,128)
                        if pred != rec['chosen'] or prob != event.get('policy_probabilities'):
                            raise ValueError('Neural action/softmax does not match checkpoint')
                    elif 'policy_probabilities' in event:
                        raise ValueError('Controls cannot have neural probabilities')
                    if world.apply(rec['chosen']) != rec:
                        raise ValueError('Per-event source-replay divergence')
                    original_events.append(rec)
                if digest(original_events) != run['trajectory_sha256']:
                    raise ValueError('Trajectory SHA256 mismatch')
                if world.state_sha() != run['final_state_sha256']:
                    raise ValueError('Final world state mismatch')
                if world.metrics(128) != run['metrics']:
                    raise ValueError('World metrics mismatch')
                if sorted(run['action_distribution'].items()) != sorted(Counter(e['chosen'] for e in run['events']).items()):
                    raise ValueError('Action distribution mismatch')
                evidence.append(run)
                audited[arm] += 1
            agg = aggregate(evidence)
            original = by_horizon['128'][arm]
            for key in ('episodes','total_decisions','mean_ink_cells','mean_branch_cells',
                        'mean_recoveries','early_stops','all_host_legal',
                        'no_protected_or_occupied_edits','action_distribution'):
                if agg[key] != original[key]:
                    raise ValueError(f'Aggregated evidence mismatch: {key}')
    return {'status':'PASS','source_matched':enforce_source,
            'trajectories_audited':sum(audited.values()),
            'by_arm':dict(sorted(audited.items())),
            'ckpts_verified':len(ckpts),'manifest_entries':len(manifest),
            'heldout_opened':False,'native_receipts':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root),sort_keys=True))
