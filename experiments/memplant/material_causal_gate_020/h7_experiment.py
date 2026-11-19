"""DG020 H7: frozen trained law donors, unchanged receiver with replica-consistent conditioning.
Preregistered: https://github.com/techrote/neurounner/pull/253#issuecomment-6940018175
Research-only. Previous H5/H6 scientific FAILs remain preserved.
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
HELDOUT_FAMILIES = ('outer_lobes', 'center_pockets')
BUCKETS = h6.BUCKETS
THRESHOLDS = h6.THRESHOLDS
CONSISTENCY_WEIGHT = 1.0  # Locked before H7 inference.

_original_wet_route = h6.wet_route

def h7_wet_route(rng, n, family):
    if family not in HELDOUT_FAMILIES:
        return _original_wet_route(rng, n, family)
    a = np.zeros(TILES, dtype='f4')
    if family == 'outer_lobes':
        order = np.array([0, 11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6])
    else:
        order = np.array([5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0, 11])
    order = (order + int(rng.integers(TILES))) % TILES
    if rng.integers(2):
        order = (TILES - 1 - order) % TILES
    a[order[:n]] = 1
    assert int(a.sum()) == n
    return a

# Reuse H6's independently computed oracle, sealed-pair sampler, negative
# controls and complete raw JSONL receipt writer unchanged.
h6.wet_route = h7_wet_route
h6.HELDOUT_FAMILIES = HELDOUT_FAMILIES

def fit_replica_consistent(seed, worlds, donors, updates=1000):
    """Same Receiver(3->16->8->1), trained with both same-law neural replicas."""
    torch.manual_seed(seed + 999)
    rng = np.random.default_rng(seed + 101)
    obs = torch.from_numpy(worlds)
    forecast = [[h6.donor_forecast(d, obs) for d in replicas] for replicas in donors]
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
                loss = loss + .25 * (
                    3.0 * F.smooth_l1_loss(advantage, oracle_delta, beta=.15)
                    + .35 * F.mse_loss(tile, tile_truth[law][indices])
                    + .15 * F.cross_entropy(logits, target_action)
                )
            loss = loss + CONSISTENCY_WEIGHT * F.mse_loss(advantages[0], advantages[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return receiver.eval()

def run(seed, checkpoint_root, out, train_each=320, eval_each=160, updates=1000):
    assert seed in SEEDS and train_each == 320 and eval_each == 160 and updates == 1000, 'H7 registered configuration locked'
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
    receiver = fit_replica_consistent(seed, x, donors, updates)
    torch.save(receiver.state_dict(), out/'receiver-replica-consistent.pt')
    results, test_hashes = {}, set()
    passed = True
    for fi, family in enumerate(HELDOUT_FAMILIES):
        results[family] = {}
        for bi, bucket in enumerate(BUCKETS):
            worlds, attempts = h6.sample(seed + 57_000_000 + fi*113000 + bi*9920, family, bucket, eval_each, True)
            hashes = {h6.sha(w.tobytes()) for w in worlds}
            assert len(hashes) == eval_each
            assert not train_hashes.intersection(hashes), 'train/test observation collision'
            assert not test_hashes.intersection(hashes), 'duplicate heldout observation'
            test_hashes.update(hashes)
            result = h6.evaluate(seed, family, bucket, worlds, donors, receiver, blind, out/f'receipts-{family}-{bucket}.jsonl')
            result['attempts'] = attempts
            results[family][bucket] = result
            passed = passed and result['verdict'] == 'PASS' and result['max_donor_mse'] <= THRESHOLDS['donor_mse_max']
    report = dict(name='DG020 H7 same-law donor replica consistency', seed=seed,
                  scientific_verdict='PASS' if passed else 'FAIL',
                  preregistration='https://github.com/techrote/neurounner/pull/253#issuecomment-6940018175',
                  training_families=TRAIN_FAMILIES, heldout_families=HELDOUT_FAMILIES,
                  training_worlds=len(x), unique_train_hashes=len(train_hashes),
                  unique_test_hashes=len(test_hashes), train_test_collisions=0,
                  updates=updates, train_each=train_each, eval_each=eval_each,
                  consistency_weight=CONSISTENCY_WEIGHT, thresholds=THRESHOLDS,
                  training_attempts=training_attempts,
                  model='same-size Receiver; both same-law trained donor replicas',
                  source_sha256=h6.sha_file(__file__), h6_source_sha256=h6.sha_file(h6.__file__),
                  torch=torch.__version__, numpy=np.__version__, results=results)
    (out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    checksums = {p.name:h6.sha_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(checksums, indent=2) + '\n')
    print(json.dumps(dict(seed=seed, verdict=report['scientific_verdict'],
          cells={f:{b:r['metrics']|{'verdict':r['verdict']} for b,r in by.items()} for f,by in results.items()})), flush=True)
    return passed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--checkpoint-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--train-each', type=int, default=320)
    parser.add_argument('--eval-each', type=int, default=160)
    parser.add_argument('--updates', type=int, default=1000)
    args = parser.parse_args()
    run(args.seed, args.checkpoint_root, args.out, args.train_each, args.eval_each, args.updates)
