#!/usr/bin/env python3
"""DG020/H22 independently recomputed receipt-level causal-selectivity contract.

Pure standard-library audit. It never trains, infers, or modifies sealed data.
Usage: python h22_receipt_contract_audit.py PATH/TO/h22-evidence
This is not a neural checkpoint replay; use h22_independent_receipt_replay.py
for genuine donor and receiver checkpoint evaluation.
"""
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

BUCKETS = ('cheap_edge_divergent', 'expensive_edge_divergent',
           'wet_rich_invariant', 'wet_poor_invariant', 'equal_wet_invariant')
LAWS = (0.35, 2.30)


def require(value, description):
    if not value:
        raise AssertionError(description)


def oracle(ob, law):
    costs = [sum((LAWS[law] if tile[0] > .5 else 1.) + tile[1]
                 for tile in route) for route in ob]
    return costs.index(min(costs)), costs


def inspect_cell(path, ref, observed):
    entries = [json.loads(s) for s in path.read_text().splitlines()]
    require(len(entries) == 160, f'{path.name}: expected 160 receipts')
    fields = Counter()
    problems = []
    negatives = []
    for i, r in enumerate(entries):
        require((r['seed'], r['family'], r['bucket'], r['index']) ==
                (ref[0], ref[1], ref[2], i), f'{path.name}:{i} metadata mismatch')
        obs = r['observation']
        require(len(obs) == 2 and all(len(route) == 12 for route in obs),
                f'{path.name}:{i}: illegal route shape')
        require(all(len(tile) == 4 and all(math.isfinite(v) for v in tile)
                    for route in obs for tile in route), f'{path.name}:{i}: observation')
        raw = b''.join(struct.pack('<f', f) for route in obs for tile in route for f in tile)
        digest = hashlib.sha256(raw).hexdigest()
        require(r['observation_sha256'] == digest, f'{path.name}:{i}: observation hash')
        require(digest not in observed, f'{path.name}:{i}: duplicate observation')
        observed.add(digest)
        require(r['legal_mask'] == [1, 1], f'{path.name}:{i}: legal mask')
        truth = [oracle(obs, law) for law in (0, 1)]
        truth_actions = [t[0] for t in truth]
        require(r['oracle_actions'] == truth_actions, f'{path.name}:{i}: oracle actions')
        for law in (0, 1):
            for route in (0, 1):
                require(abs(r['oracle_route_costs'][law][route] - truth[law][1][route]) < 1e-5,
                        f'{path.name}:{i}: route cost')
        predicted = [int(max(range(2), key=lambda a: r['logits'][j][a])) for j in (0, 1)]
        foreign = [int(max(range(2), key=lambda a: r['foreign_logits'][j][a])) for j in (0, 1)]
        require(r['actions'] == predicted, f'{path.name}:{i}: logit action mismatch')
        require(r['foreign_actions'] == foreign, f'{path.name}:{i}: foreign logit mismatch')
        require(r['both_correct'] == (predicted == truth_actions), f'{path.name}:{i}: both')
        require(r['law_swap_changed'] == (predicted[0] != predicted[1]), f'{path.name}:{i}: flip')
        require(r['foreign_sham_changed'] == (predicted != foreign), f'{path.name}:{i}: sham')
        placebo = [predicted[j] != r['dry_placebo_actions'][j] for j in (0, 1)]
        require(r['dry_placebo_changed'] == placebo, f'{path.name}:{i}: placebo')
        require(len(r['shuffled_actions']) == 2 and len(r['shuffled_donor_selection']) == 2,
                f'{path.name}:{i}: shuffle shape')
        require(all(a in (0, 1) for a in predicted + foreign + r['dry_placebo_actions']
                    + r['shuffled_actions'] + [r['zero_action'], r['source_blind_action']]),
                f'{path.name}:{i}: illegal action')
        for k, condition in {
            'both': predicted == truth_actions,
            'flip': predicted[0] != predicted[1],
            'law0': predicted[0] == truth_actions[0],
            'law1': predicted[1] == truth_actions[1],
            'foreign_same_law_sham': predicted != foreign,
            'dry_only_placebo_any': any(placebo),
            'dry_only_placebo_law0': placebo[0],
            'dry_only_placebo_law1': placebo[1],
            'shuffled_preinference_both': r['shuffled_actions'] == truth_actions,
            'zero_both': r['zero_action'] == truth_actions[0] == truth_actions[1],
            'source_blind_both': r['source_blind_action'] == truth_actions[0] == truth_actions[1],
        }.items():
            fields[k] += int(condition)
        if predicted != truth_actions or predicted != foreign or any(placebo):
            negatives.append(i)
            problems.append((r['seed'], r['family'], r['bucket'], i, digest))
    actual = dict(fields)
    actual.update(n=160, illegal=0)
    for k, v in actual.items():
        require(ref[3]['metrics'][k] == v, f'{path.name}: metric {k} {v} != {ref[3]["metrics"][k]}')
    require(ref[3]['negative_indices'] == negatives, f'{path.name}: negative index mismatch')
    div = ref[2].endswith('divergent')
    m = ref[3]['metrics']
    passed = (actual['both'] / 160 >= (.90 if div else .94)
              and (actual['flip'] / 160 >= .95 if div else actual['flip'] / 160 <= .05)
              and min(actual['law0'], actual['law1']) / 160 >= .94
              and actual['foreign_same_law_sham'] / 160 <= .05
              and actual['dry_only_placebo_any'] / 160 <= .05
              and (not div or (actual['shuffled_preinference_both'] / 160 <= .60
                               and actual['zero_both'] == 0 and actual['source_blind_both'] == 0))
              and m['max_probe_mse'] <= .02)
    # MSE is a source-claimed aggregate, independently checked by neural replay only.
    require(ref[3]['verdict'] == ('PASS' if passed else 'FAIL'),
            f'{path.name}: cell verdict mismatch')
    return actual, problems, passed


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('evidence', type=Path)
    args = cli.parse_args()
    directory = args.evidence
    report = json.loads((directory / 'report.json').read_text())
    hashes = json.loads((directory / 'sha256.json').read_text())
    require(len(hashes) == 31, 'expected 31 sealed evidence manifest entries')
    for name, expected in hashes.items():
        require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected,
                f'sealed manifest mismatch: {name}')
    seeds = report['receiver_seeds']
    families = list(report['families'])
    require(len(seeds) == 3 and len(families) == 2, 'unexpected cohort shape')
    require(report['thresholds']['invariant_both'] == .94, 'changed threshold')
    observed, totals, problems = set(), Counter(), []
    count = 0
    passed = 0
    for seed in seeds:
        for fam in families:
            for bucket in BUCKETS:
                f = directory / f'receipts-{seed}-{fam}-{bucket}.jsonl'
                metrics, negatives, cell_ok = inspect_cell(
                    f, (seed, fam, bucket, report['results'][str(seed)][fam][bucket]), observed)
                count += 1
                passed += int(cell_ok)
                problems.extend(negatives)
                prefix = 'divergent_' if bucket.endswith('divergent') else 'invariant_'
                totals[prefix + 'both'] += metrics['both']
                totals[prefix + 'flip'] += metrics['flip']
                totals['foreign_source_changes'] += metrics['foreign_same_law_sham']
                totals['dry_placebo_changes'] += metrics['dry_only_placebo_any']
                if bucket.endswith('divergent'):
                    totals['shuffled_divergent_both'] += metrics['shuffled_preinference_both']
    require(count == 30 and passed == report['cells_passed'], 'aggregate cell mismatch')
    require(len(observed) == 4800 == report['worlds'], 'world-count mismatch')
    require(report['scientific_verdict'] == ('PASS' if passed == count else 'FAIL'),
            'reported gate verdict mismatch')
    result = dict(verdict=report['scientific_verdict'], cells=f'{passed}/{count}',
                  pairs=len(observed), receipts_hashed=len(hashes),
                  results=dict(totals), failures=problems,
                  limitation='Receipt and oracle contract only; no neural checkpoint inference or original source verification')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
