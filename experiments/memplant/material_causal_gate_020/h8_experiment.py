"""DG020 H8 preregistered median-conditioned material law gate.
Scientific preregistration: https://github.com/techrote/neurounner/pull/253#issuecomment-6941114781
Research-only: no CyberSand or DRIV3R edits; prior H5/H6/H7 FAILs remain preserved.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import h6_experiment as h6
from experiment import Receiver, TILES, WET_COST, oracle

SEEDS = h6.SEEDS
TRAIN_FAMILIES = h6.TRAIN_FAMILIES
HELDOUT_FAMILIES = ('interleave_edges', 'shifted_nested')
BUCKETS = h6.BUCKETS
THRESHOLDS = h6.THRESHOLDS
CONSISTENCY_WEIGHT = 1.0  # Identical to H7, not tuned on H8 acceptance.
_raw_forecast = h6.donor_forecast
_original_wet_route = h6.wet_route


def median_condition(obs, forecasts):
    """Remove per-tile nuisance noise; infer each material cost from donor outputs only.

    The material mask and cell fees are already in the receiver's ordinary
    observation. No true costs, law identifiers, answers or family IDs are read.
    """
    wet = obs[..., 0] > .5
    fee = obs[..., 1]
    no_fee = (forecasts - fee).reshape(len(obs), -1)
    wet_flat = wet.reshape(len(obs), -1)
    assert wet_flat.any(dim=1).all() and (~wet_flat).any(dim=1).all()
    nan = torch.full_like(no_fee, float('nan'))
    wet_cost = torch.nanmedian(torch.where(wet_flat, no_fee, nan), dim=1).values
    dry_cost = torch.nanmedian(torch.where(~wet_flat, no_fee, nan), dim=1).values
    return torch.where(wet, wet_cost[:, None, None], dry_cost[:, None, None]) + fee


def conditioned_forecast(donor, obs):
    return median_condition(obs, _raw_forecast(donor, obs))


def h8_wet_route(rng, n, family):
    if family not in HELDOUT_FAMILIES:
        return _original_wet_route(rng, n, family)
    order = (np.array([0,6,1,7,2,8,3,9,4,10,5,11]) if family == 'interleave_edges'
             else np.array([0,3,6,9,1,4,7,10,2,5,8,11]))
    order = (order + int(rng.integers(TILES))) % TILES
    if rng.integers(2):
        order = (TILES - 1 - order) % TILES
    a = np.zeros(TILES, dtype='f4')
    a[order[:n]] = 1
    assert int(a.sum()) == n
    return a


# All samplers/oracles/paired-input controls are exactly the H6 implementation.
h6.donor_forecast = conditioned_forecast
h6.wet_route = h8_wet_route
h6.HELDOUT_FAMILIES = HELDOUT_FAMILIES


def fit(seed, worlds, donors, updates=1000):
    """Unchanged Receiver(3->16->8->1), H7 weights/objectives, H8 inputs."""
    torch.manual_seed(seed + 999)
    rng = np.random.default_rng(seed + 101)
    obs = torch.from_numpy(worlds)
    forecast = [[conditioned_forecast(d, obs) for d in replicas] for replicas in donors]
    oracle_costs = [torch.from_numpy(np.asarray([oracle(x, k)[2] for x in worlds], dtype='f4')) for k in (0, 1)]
    tile_truth = [torch.from_numpy((np.where(worlds[..., 0] > .5, WET_COST[k], 1.) + worlds[..., 1]).astype('f4')) for k in (0, 1)]
    receiver = Receiver().train()
    optimizer = torch.optim.AdamW(receiver.parameters(), lr=.002, weight_decay=.0001)
    for _ in range(updates):
        indices = torch.from_numpy(rng.integers(0, len(obs), 192))
        loss = 0.
        for law in (0, 1):
            oracle_delta = oracle_costs[law][indices, 1] - oracle_costs[law][indices, 0]
            target_action = torch.where(oracle_delta > 0, torch.zeros_like(oracle_delta), torch.ones_like(oracle_delta)).long()
            advantages = []
            for replica in (0, 1):
                tile = receiver.tile(torch.stack((obs[indices, ..., 0], obs[indices, ..., 1], forecast[law][replica][indices]), -1)).squeeze(-1)
                logits = -tile.sum(-1)
                advantage = logits[:, 0] - logits[:, 1]
                advantages.append(advantage)
                loss += .25 * (3.0 * F.smooth_l1_loss(advantage, oracle_delta, beta=.15)
                               + .35 * F.mse_loss(tile, tile_truth[law][indices])
                               + .15 * F.cross_entropy(logits, target_action))
            loss += CONSISTENCY_WEIGHT * F.mse_loss(advantages[0], advantages[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return receiver.eval()


def enrich_receipts(receipt_path, donors):
    rows = [json.loads(line) for line in Path(receipt_path).read_text().splitlines()]
    obs = torch.from_numpy(np.asarray([r['observation'] for r in rows], dtype='f4'))
    raw = [[_raw_forecast(d, obs).numpy() for d in replicas] for replicas in donors]
    with Path(receipt_path).open('w') as w:
        for i, row in enumerate(rows):
            row['raw_donor_forecasts'] = [raw[k][0][i].tolist() for k in (0, 1)]
            row['raw_sham_forecasts'] = [raw[k][1][i].tolist() for k in (0, 1)]
            row['conditioning'] = 'median-of-trained-donor-material-forecast-minus-observed-fee'
            w.write(json.dumps(row, separators=(',', ':')) + '\n')


def run(seed, checkpoint_root, out, train_each=320, eval_each=160, updates=1000):
    assert seed in SEEDS and (train_each, eval_each, updates) == (320, 160, 1000), 'H8 preregistration locked'
    torch.set_num_threads(2)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    donors, blind = h6.load_donors(seed, checkpoint_root)
    train_worlds, training_attempts = [], {}
    for fi, family in enumerate(TRAIN_FAMILIES):
        for bi, bucket in enumerate(BUCKETS):
            near, attempts = h6.sample(seed + 1_000_000 + fi*7777 + bi*111, family, bucket, train_each, True)
            train_worlds.append(near)
            training_attempts[f'{family}-{bucket}'] = attempts
            moderate, attempts = h6.sample(seed + 1_900_000 + fi*7777 + bi*111, family, bucket, train_each//4, False)
            train_worlds.append(moderate)
            training_attempts[f'{family}-{bucket}-wide'] = attempts
    x = np.concatenate(train_worlds).astype('f4')
    assert len(x) == 4000
    train_hashes = {h6.sha(w.tobytes()) for w in x}
    receiver = fit(seed, x, donors, updates)
    torch.save(receiver.state_dict(), out/'receiver-median-conditioned.pt')
    results, test_hashes = {}, set()
    passed = True
    for fi, family in enumerate(HELDOUT_FAMILIES):
        results[family] = {}
        for bi, bucket in enumerate(BUCKETS):
            worlds, attempts = h6.sample(seed + 67_000_000 + fi*113000 + bi*9920, family, bucket, eval_each, True)
            hashes = {h6.sha(w.tobytes()) for w in worlds}
            assert len(hashes) == eval_each
            assert not train_hashes.intersection(hashes), 'train/test collision'
            assert not test_hashes.intersection(hashes), 'duplicate heldout observation'
            test_hashes.update(hashes)
            receipts = out/f'receipts-{family}-{bucket}.jsonl'
            result = h6.evaluate(seed, family, bucket, worlds, donors, receiver, blind, receipts)
            enrich_receipts(receipts, donors)
            result['attempts'] = attempts
            results[family][bucket] = result
            passed = passed and result['verdict'] == 'PASS' and result['max_donor_mse'] <= THRESHOLDS['donor_mse_max']
    report = dict(name='DG020 H8 median-conditioned trained material-law donors', seed=seed,
                  scientific_verdict='PASS' if passed else 'FAIL',
                  preregistration='https://github.com/techrote/neurounner/pull/253#issuecomment-6941114781',
                  training_families=TRAIN_FAMILIES, heldout_families=HELDOUT_FAMILIES,
                  training_worlds=len(x), unique_train_hashes=len(train_hashes),
                  unique_test_hashes=len(test_hashes), train_test_collisions=0,
                  updates=updates, train_each=train_each, eval_each=eval_each,
                  consistency_weight=CONSISTENCY_WEIGHT, thresholds=THRESHOLDS,
                  conditioning='per-material within-world median of independent trained donor predictions minus observed fee',
                  training_attempts=training_attempts,
                  model='same-size Receiver; two same-law trained donor replicas, conditioned separately',
                  source_sha256=h6.sha_file(__file__), h6_source_sha256=h6.sha_file(h6.__file__),
                  torch=torch.__version__, numpy=np.__version__, results=results)
    (out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    checksums = {p.name:h6.sha_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(checksums, indent=2) + '\n')
    print(json.dumps(dict(seed=seed, verdict=report['scientific_verdict'],
          cells={f:{b:r['metrics']|{'verdict':r['verdict']} for b,r in by.items()} for f,by in results.items()})), flush=True)
    return passed


def audit(seed, checkpoint_root, out):
    """Independent raw-donor, conditioned-input, trained-weight and oracle replay."""
    torch.set_num_threads(2)
    out = Path(out)
    donors, _ = h6.load_donors(seed, checkpoint_root)
    receiver = Receiver().eval()
    receiver.load_state_dict(torch.load(out/'receiver-median-conditioned.pt', map_location='cpu', weights_only=True))
    report = json.loads((out/'report.json').read_text())
    assert report['source_sha256'] == h6.sha_file(__file__)
    manifest = json.loads((out/'sha256.json').read_text())
    assert all(h6.sha_file(out/name) == digest for name,digest in manifest.items())
    count = 0
    for family in HELDOUT_FAMILIES:
        for bucket in BUCKETS:
            rows = [json.loads(s) for s in (out/f'receipts-{family}-{bucket}.jsonl').read_text().splitlines()]
            assert len(rows) == THRESHOLDS['n']
            obs = np.asarray([r['observation'] for r in rows], dtype='f4')
            tx = torch.from_numpy(obs)
            with torch.no_grad():
                raw = [[_raw_forecast(d, tx) for d in reps] for reps in donors]
                forecasts = [[median_condition(tx, r) for r in reps] for reps in raw]
                logits = [receiver(tx, forecasts[k][0]) for k in (0, 1)]
                sham = [receiver(tx, forecasts[k][1]).argmax(-1).numpy() for k in (0, 1)]
            for i, r in enumerate(rows):
                assert h6.sha(obs[i].tobytes()) == r['observation_sha256']
                assert r['identical_observation_both_laws'] and r['legal_mask'] == [1, 1]
                acts = []
                for k in (0, 1):
                    action, gap, costs = oracle(obs[i], k)
                    assert action == r['oracle_actions'][k]
                    assert abs(gap - r['oracle_gaps'][k]) < 1e-4
                    assert np.allclose(costs, r['oracle_route_costs'][k], atol=5e-5, rtol=1e-6)
                    for rep in (0, 1):
                        key = 'raw_donor_forecasts' if rep == 0 else 'raw_sham_forecasts'
                        assert np.allclose(raw[k][rep][i], r[key][k], atol=5e-5)
                    assert np.allclose(forecasts[k][0][i], r['donor_forecasts'][k], atol=5e-5)
                    assert np.allclose(forecasts[k][1][i], r['sham_forecasts'][k], atol=5e-5)
                    assert np.allclose(logits[k][i], r['logits'][k], atol=5e-5)
                    assert int(sham[k][i]) == r['sham_actions'][k]
                    acts.append(int(logits[k][i].argmax()))
                assert acts == r['actions']
                assert (r['oracle_actions'][0] != r['oracle_actions'][1]) == bucket.endswith('divergent')
                count += 1
    assert count == 1600
    audit_report = dict(seed=seed, receipts_replayed=count, decisions_replayed=count*2,
                        manifest_verified=len(manifest), science=report['scientific_verdict'],
                        code_sha256=h6.sha_file(__file__))
    (out/'independent-audit.json').write_text(json.dumps(audit_report, indent=2)+'\n')
    print(json.dumps(audit_report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--checkpoint-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--train-each', type=int, default=320)
    parser.add_argument('--eval-each', type=int, default=160)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        audit(args.seed, args.checkpoint_root, args.out)
    else:
        run(args.seed, args.checkpoint_root, args.out, args.train_each, args.eval_each, args.updates)
