"""Predeclared DEV-only 36/72/128 curriculum vs equal-update flat-128 ablation.

Trains real NumPy action ranker checkpoints and compares them on matched
synthetic worlds, with persistent and STOP-inclusive legal random controls.
Neither native execution nor structurally held-out outcomes are opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from .policy import Ranker
from .train import (episode, aggregate, HORIZONS, source_pins,
                    sha_file, make_corpus)
from .world import World, specs, MOVE

VERSION = 'driv3r-long-horizon-curriculum-ablation/1'
ARMS = ('staged_36_72_128', 'flat_128', 'teacher_mixture',
        'random_legal', 'random_legal_persistent')


def training_plan(arm: str):
    if arm == 'staged_36_72_128':
        return HORIZONS
    if arm == 'flat_128':
        return (128, 128, 128)
    raise ValueError('Not a trainable curriculum')


def train_policy(arm: str, seed: int, worlds: int, updates: int, output: Path):
    if not (1 <= worlds <= 9 and 1 <= updates <= 180):
        raise ValueError('Bounded research cap exceeded')
    policy = Ranker(seed=seed, hidden=24)
    record = []
    for stage, horizon in enumerate(training_plan(arm)):
        X, masks, labels, corpus = make_corpus(horizon, seed, worlds)
        # Identical optimizer schedule and RNG between arms; ONLY the
        # training horizon changes. Same initial weights for matched seed.
        optimization = policy.optimize(
            X, masks, labels, steps=updates, batch=64,
            lr=.007 if stage < 2 else .004, seed=seed+stage*917)
        record.append({'stage': stage, 'horizon': horizon,
                       'examples': len(labels),
                       'train_worlds_sha256': corpus['original_worlds_sha256'],
                       'label_distribution': corpus['labels'],
                       'interventions': corpus['off_policy_interventions'],
                       'optimizer': optimization})
    ckpt = output/f'{arm}-seed{seed}.npz'
    policy.checkpoint(ckpt)
    loaded = Ranker.restore(ckpt)
    if loaded.updates != 3*updates or any(
            not np.array_equal(loaded.params[k], policy.params[k])
            for k in policy.params):
        raise AssertionError('Checkpoint replay differs from trained weights')
    return loaded, {'curriculum': arm, 'seed': seed,
                    'updates': loaded.updates,
                    'parameters': loaded.parameter_count,
                    'checkpoint_sha256': sha_file(ckpt),
                    'checkpoint_filename': ckpt.name,
                    'stages': record}


def teacher_conditional_agreement(worlds, ranker: Ranker, seed: int):
    """Teacher-on-its-own-states imitation, NOT an on-policy rollout score."""
    tally = Counter()
    mismatch_by_mode = Counter()
    for spec in worlds:
        teacher_run = episode(spec, 128, 'teacher_mixture', None, seed)
        world = World.create(spec)
        for event in teacher_run['events']:
            chosen, _ = ranker.choose(world, 128)
            expected = event['chosen']
            tally['total'] += 1
            tally['correct'] += int(chosen == expected)
            if chosen != expected:
                mismatch_by_mode[spec.mode] += 1
                tally['missed_recovery'] += int(expected in ('BACKTRACK', 'RELOCATE_START'))
            applied = world.apply(expected)
            if applied != event:
                raise AssertionError('Teacher conditional trace replay failed')
    return {'teacher_state_decisions': tally['total'],
            'matches': tally['correct'],
            'accuracy': round(tally['correct']/max(1,tally['total']), 6),
            'missed_teacher_recovery': tally['missed_recovery'],
            'mismatch_by_mode': dict(sorted(mismatch_by_mode.items()))}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-count', type=int, default=2)
    parser.add_argument('--seed-start', type=int, default=570)
    parser.add_argument('--train-worlds-per-family', type=int, default=9)
    parser.add_argument('--updates-per-stage', type=int, default=180)
    parser.add_argument('--dev-worlds-per-family', type=int, default=2)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Immutable artifact destination already exists')
    if not (1 <= args.seed_count <= 2 and
            570 <= args.seed_start <= 571 and args.seed_start + args.seed_count <= 572 and
            1 <= args.train_worlds_per_family <= 9 and
            1 <= args.updates_per_stage <= 180 and
            1 <= args.dev_worlds_per_family <= 3):
        raise ValueError('Resource budget exceeded')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    # Development ONLY. No calls to specs('heldout') in this runner.
    dev = specs('development', args.dev_worlds_per_family)
    dev += [replace(s, challenge='trap_corridor') for s in dev if s.seed % 2 == 0]
    if any(s.split != 'development' for s in dev):
        raise AssertionError('Development split seal broken')
    report = {
        'version': VERSION,
        'research_only': True,
        'model_families_executed': ['NumPy-625-parameter-tiny-action-ranker'],
        'model_families_not_executed': ['d1-2B','LFM2.5-350M','Sprocket-500M'],
        'native_receipts': 0, 'paid_external_compute': False,
        'heldout_used_in_optimizer_or_evaluation': False,
        'original_v1_final_heldout_reused': False,
        'source_sha256': {**source_pins(),
            'compare_curricula.py': sha_file(Path(__file__))},
        'protocol': 'same seed+init+Adam schedule, 540 updates, only corpus horizon differs',
        'worlds': [{'id': s.ident,'seed':s.seed,'family':s.family,
                    'mode':s.mode,'challenge':s.challenge,
                    'base_sha256':World.create(s).base_sha()} for s in dev],
        'horizons': list(HORIZONS), 'arms': list(ARMS),
        'training': [], 'results': {},
        'resource_caps': {'seed_count': 2, 'train_worlds_per_family': 9,
                          'updates_per_stage': 180,
                          'dev_worlds_per_family': 3,
                          'native_64_logical_move_limit': 'unchanged; synthetic only'},
    }
    for seed in range(args.seed_start, args.seed_start+args.seed_count):
        trained = {}
        for arm in ARMS[:2]:
            policy, record = train_policy(arm, seed,
                                         args.train_worlds_per_family,
                                         args.updates_per_stage, args.output)
            trained[arm] = policy
            report['training'].append(record)
        scores = {}
        for horizon in HORIZONS:
            scores[str(horizon)] = {}
            for arm in ARMS:
                results = []
                for spec in dev:
                    # Store complete original 128-step trajectories + 4x gallery.
                    dest = args.output/'trajectories'/f'seed{seed}'/arm if horizon == 128 else None
                    run = episode(spec, horizon,
                                  'trained_tiny_mlp' if arm in trained else arm,
                                  trained.get(arm), seed, dest)
                    # Per-arm checkpoint linkage is explicit in the top-level run table.
                    results.append(run)
                agg = aggregate(results)
                agg['source_replay_qualified'] = True
                agg['action_count_check'] = (sum(agg['action_distribution'].values()) ==
                                             agg['total_decisions'])
                agg['mean_turns'] = round(sum(r['metrics']['turns'] for r in results)/len(results), 4)
                agg['mean_logical_cursor_moves'] = round(sum(r['metrics']['logical_cursor_moves']
                                                             for r in results)/len(results), 4)
                agg['mean_ink_per_decision'] = round(sum(r['metrics']['unique_painted_cells']
                                                             / r['metrics']['decisions']
                                                             for r in results)/len(results), 5)
                agg['hard_negative_cases'] = [
                    {'world':r['spec_id'], 'family':r['family'],
                     'mode':r['mode'], 'challenge':r['challenge'],
                     'ink':r['metrics']['unique_painted_cells'],
                     'recoveries':r['metrics']['recoveries'],
                     'stopped_early':r['metrics']['stopped_early'],
                     'trajectory_sha256':r['trajectory_sha256']}
                    for r in results if r['metrics']['unique_painted_cells'] < horizon*.6
                    or r['metrics']['stopped_early']]
                scores[str(horizon)][arm] = agg
        for arm, model in trained.items():
            scores['128'][arm]['teacher_conditional_agreement'] = teacher_conditional_agreement(dev, model, seed)
        report['results'][str(seed)] = scores
    report['wall_seconds'] = round(time.perf_counter()-started, 4)
    report['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (args.output/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    digest_map = {str(p.relative_to(args.output)):sha_file(p)
                  for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'SHA256SUMS.json').write_text(json.dumps(digest_map, sort_keys=True, indent=2)+'\n')
    print(json.dumps({
        'result': 'real_cpu_train_and_matched_development_evaluation',
        'wall_seconds': report['wall_seconds'],
        'peak_process_rss_kib': report['peak_process_rss_kib'],
        'checkpoints': [{'arm':r['curriculum'],'seed':r['seed'],'sha256':r['checkpoint_sha256']}
                        for r in report['training']],
        'dev_128': {seed:{arm:{k:scores['128'][arm][k] for k in
                    ('mean_ink_cells','mean_branch_cells','mean_recoveries','early_stops','mean_decision_latency_ms')}
                   for arm in ARMS} for seed,scores in report['results'].items()},
    }, sort_keys=True))
    return report


if __name__ == '__main__':
    main()
