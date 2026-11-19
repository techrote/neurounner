#!/usr/bin/env python3
"""DG020 H49: retrospective numerical margin inventory of authentic H40/H45 receipts.

Read-only. Does not infer a new physical law, manufacture donors, retrain, or
change the historical scientific acceptance verdict. Run with the extracted
DG020-H48-EVIDENCE directory; use numpy for stored float32 score inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

H40_SHA = '5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d'
H47_MARGIN_SHA = '77e96494144c2015dab45ad4823dfbc17056b1c064cdd2e5a793e1c324da7bee'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    rows_path = root / 'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl'
    margins_path = root / 'h47-output/h47_all_float32_neural_route_margins.npz'
    h48_path = root / 'h48-output/h48_cross_runtime_audit.json'
    assert digest(rows_path) == H40_SHA, 'Original H40 observations changed'
    assert digest(margins_path) == H47_MARGIN_SHA, 'Original H47 logit arrays changed'
    h48 = json.loads(h48_path.read_text())
    assert h48['file_sha256'] == H47_MARGIN_SHA
    assert h48['all_scores_checked'] == 7526400
    assert h48['decision_disagreements'] == 1
    # Empirical maximum from one distinct CPU replay, NOT a universal bound.
    empirical = float(h48['worst_abs_score_drift'])
    assert empirical == 7.62939453125e-5
    rows = [json.loads(s) for s in rows_path.open()]
    assert len(rows) == 4800
    data = np.load(margins_path)
    result = {'scope': 'exposed H40/H45 numerical diagnostic only',
              'h40_receipts_sha256': H40_SHA,
              'h47_margins_sha256': H47_MARGIN_SHA,
              'empirical_one_cpu_max_abs_drift': empirical,
              'seeds': {}}
    for seed in (26011639, 26011640, 26011641):
        original = [r for r in rows if r['seed'] == seed]
        assert len(original) == 1600
        truth = np.asarray([r['oracle_actions'] for r in original], dtype=np.int8).T
        assert truth.shape == (2, 1600)
        divergent = np.asarray([r['bucket'].endswith('divergent') for r in original])
        assert int(divergent.sum()) == 640 and int((~divergent).sum()) == 960
        assert all(r['pair_same_geometry_goal_legal_mask'] and r['legal_mask'] == [1, 1] for r in original)
        group = {}
        for model in ('h40', 'h45'):
            endpoints = data[f'{model}_{seed}_endpoint_margins']
            assert endpoints.shape == (2, 2, 4, 1600)
            model_result = {}
            for axis, label in enumerate(('original', 'aligned')):
                score = endpoints[axis]
                error = (score > 0) != truth[:, None, :]
                absolute = np.abs(score.astype(np.float64))
                model_result[label] = {
                    'genuine_endpoint_margins': int(score.size),
                    'incorrect_actions': int(error.sum()),
                    'min_abs_margin': float(absolute.min()),
                    'within_empirical_drift': int((absolute <= empirical).sum()),
                    'within_empirical_drift_divergent': int((absolute[:, :, divergent] <= empirical).sum()),
                    'within_empirical_drift_invariant': int((absolute[:, :, ~divergent] <= empirical).sum())}
            if model == 'h45':
                interior = data[f'{model}_{seed}_offgrid_margins']
                assert interior.shape == (2, 4, 192, 1600)
                near = np.abs(interior.astype(np.float64)) <= empirical
                model_result['synthetic_interior'] = {
                    'margins': int(interior.size),
                    'within_empirical_drift': int(near.sum()),
                    'exact_zeros': int((interior == 0).sum())}
            group[model] = model_result
        result['seeds'][str(seed)] = group
    result['totals'] = {}
    for model in ('h40', 'h45'):
        for label in ('original', 'aligned'):
            series = [result['seeds'][str(seed)][model][label] for seed in (26011639, 26011640, 26011641)]
            result['totals'][f'{model}_{label}'] = {
                'genuine_endpoint_margins': sum(s['genuine_endpoint_margins'] for s in series),
                'incorrect_actions': sum(s['incorrect_actions'] for s in series),
                'within_empirical_drift': sum(s['within_empirical_drift'] for s in series),
                'minimum_abs_margin': min(s['min_abs_margin'] for s in series)}
    off = [result['seeds'][str(seed)]['h45']['synthetic_interior'] for seed in (26011639,26011640,26011641)]
    result['totals']['h45_synthetic_interior'] = {
        'margins': sum(s['margins'] for s in off),
        'within_empirical_drift': sum(s['within_empirical_drift'] for s in off),
        'exact_zeros': sum(s['exact_zeros'] for s in off)}
    assert result['totals']['h40_original']['within_empirical_drift'] == 0
    assert result['totals']['h40_aligned']['within_empirical_drift'] == 0
    assert result['totals']['h45_original']['within_empirical_drift'] == 1390
    assert result['totals']['h45_aligned']['within_empirical_drift'] == 1392
    assert result['totals']['h45_synthetic_interior']['within_empirical_drift'] == 102943
    assert all(v['incorrect_actions'] == 0 for k,v in result['totals'].items() if 'endpoint_margins' in v)
    result['verdict'] = 'PASS: retrospective inventory; FAIL: no general cross-runtime bit-exactness or strengthened scientific gate'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path, help='Unpacked DG020-H48-EVIDENCE directory')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    document = json.dumps(report, sort_keys=True, indent=2) + '\n'
    if args.output:
        args.output.write_text(document)
    print(document)
