"""MZ-008: history-dependent external CyberSand-style game policy probe.

Pure model research: no native, Godot, Workbench or game content edits.
Beacon on ticks 0-1, hazard inverter on 4-5; decision on tick 8.
The current observation is identical for opposite earlier beacons.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import time

ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'PENUP', 'STOP')
SCHEMA = 'MZ008-DELAYED-CUE-GAME-v1'
T, D = 9, 32
TARGETS = (1, 2, 3, 4)


def world(base: int, cue: int, invert: int, author: int):
    if any(v not in (0, 1) for v in (cue, invert, author)):
        raise ValueError('binary counterfactuals only')
    rng = random.Random(720171 + 10007 * base)
    matrix = [[rng.uniform(-.12, .12) for _ in range(D)] for _ in range(T)]
    for t, x in enumerate(matrix):
        x[8:15] = [1., 1., 1., 1., 1., 0., 1.]
        x[15] = (base % 4) / 3.
        x[16] = ((base * 3) % 11 - 5) / 5.
        x[17] = ((base * 7) % 11 - 5) / 5.
        x[18:24] = [0.] * 6
        x[19] = float(t == T - 1)
        x[24:32] = [rng.uniform(-1., 1.) for _ in range(8)]
    # Redundant views of each clue, but neither is present at decision time.
    matrix[0][20 + cue] = 3.
    matrix[1][20 + cue] = 3.
    matrix[4][22 + invert] = 3.
    matrix[5][22 + invert] = 3.
    matrix[-1][18] = float(author)
    target = TARGETS[author * 2 + (cue ^ invert)]
    assert matrix[-1][8 + target] == 1.
    return matrix, target, dict(base=base, cue=cue, invert=invert, author=author)


def dataset(ids):
    return [world(base, cue, inv, mode) for base in ids
            for cue in (0, 1) for inv in (0, 1) for mode in (0, 1)]


def models(torch, hidden=128, layers=2):
    nn = torch.nn
    class MemoryPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(D)
            self.gru = nn.GRU(D, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden + D),
                                      nn.Linear(hidden + D, 96), nn.GELU(),
                                      nn.Linear(96, len(ACTIONS)))
        def forward(self, x):
            z = self.norm(x)
            h = self.gru(z)[0][:, -1, :]
            return self.head(torch.cat((h, z[:, -1, :]), dim=-1))
    control = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 128),
                            nn.GELU(), nn.Linear(128, len(ACTIONS)))
    return MemoryPolicy(), control


def balanced_scores(pred, rows):
    totals, hits = {}, {}
    confusion = {a: {b: 0 for b in ACTIONS} for a in ACTIONS}
    invalid = 0
    for p, (x, label, _meta) in zip(pred, rows):
        totals[label] = totals.get(label, 0) + 1
        hits[label] = hits.get(label, 0) + int(p == label)
        confusion[ACTIONS[label]][ACTIONS[p]] += 1
        invalid += int(x[-1][8 + p] != 1.)
    recall = [hits.get(k, 0) / totals[k] for k in TARGETS]
    return dict(accuracy=round(sum(hits.values()) / len(rows), 6),
                macro_recall=round(sum(recall) / len(recall), 6),
                raw_illegal=invalid, decisions=len(rows), confusion=confusion)


def pair_scores(pred, rows):
    keyed = {(m['base'], m['invert'], m['author'], m['cue']): p
             for p, (_x, _y, m) in zip(pred, rows)}
    correct_pairs = switches = count = 0
    for (base, inv, mode, cue), p0 in keyed.items():
        if cue != 0:
            continue
        p1 = keyed[base, inv, mode, 1]
        correct_pairs += int(p0 == TARGETS[2 * mode + inv]
                             and p1 == TARGETS[2 * mode + (1 ^ inv)])
        switches += int(p0 != p1)
        count += 1
    return dict(both_correct=correct_pairs, pairs=count, switched=switches,
                both_correct_rate=round(correct_pairs / count, 6),
                switched_rate=round(switches / count, 6))


def intervention(data, torch, kind):
    x = data.clone()
    if kind == 'erase_history':
        x[:, :-1, :] = 0.
    elif kind == 'remove_beacon':
        x[:, :2, 20:22] = 0.
    elif kind == 'remove_inversion':
        x[:, 4:6, 22:24] = 0.
    elif kind == 'reverse_early_beacon':
        x[:, :2, 20:22] = x[:, :2, 20:22].flip(-1)
    else:
        raise ValueError(kind)
    return x


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(60851)
    gen = torch.Generator().manual_seed(60853)
    if min(args.train_worlds, args.eval_worlds, args.updates, args.batch) < 1:
        raise ValueError('positive budgets required')
    train_ids = range(1000, 1000 + args.train_worlds)
    dev_ids = range(3000, 3000 + args.eval_worlds)
    test_ids = range(5000, 5000 + args.eval_worlds)
    assert not (set(train_ids) & set(dev_ids) or set(train_ids) & set(test_ids)
                or set(dev_ids) & set(test_ids))
    train, dev, test = [dataset(ids) for ids in (train_ids, dev_ids, test_ids)]
    def tensors(rows):
        return (torch.tensor([r[0] for r in rows], dtype=torch.float32),
                torch.tensor([r[1] for r in rows], dtype=torch.long))
    X, y = tensors(train)
    XD, _ = tensors(dev)
    XT, _ = tensors(test)
    net, control = models(torch, args.hidden, args.layers)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    ctrl_opt = torch.optim.AdamW(control.parameters(), lr=args.lr)
    losses = {'recurrent': [], 'current_only': []}
    updates_ms = []
    for _ in range(args.updates):
        ids = torch.randint(len(X), (args.batch,), generator=gen)
        for name, m, optimizer, obs in (
            ('recurrent', net, opt, X),
            ('current_only', control, ctrl_opt, X[:, -1, :])
        ):
            tic = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = m(obs[ids])
            loss = torch.nn.functional.cross_entropy(logits, y[ids])
            if not torch.isfinite(loss):
                raise RuntimeError(name + ' nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            optimizer.step()
            losses[name].append(round(float(loss.detach()), 6))
            if name == 'recurrent':
                updates_ms.append((time.perf_counter() - tic) * 1000)
    net.eval()
    control.eval()
    def predict(m, data):
        with torch.inference_mode():
            return m(data).argmax(-1).tolist()
    with torch.inference_mode():
        for _ in range(8):
            net(XT[:1])
        times = []
        for _ in range(100):
            tic = time.perf_counter()
            net(XT[:1])
            times.append(1000 * (time.perf_counter() - tic))
    predictions = {
        name: predict(net, data) for name, data in (
            ('recurrent', XT),
            ('erase_history', intervention(XT, torch, 'erase_history')),
            ('remove_beacon', intervention(XT, torch, 'remove_beacon')),
            ('remove_inversion', intervention(XT, torch, 'remove_inversion')),
            ('beacon_flip', intervention(XT, torch, 'reverse_early_beacon')),
        )
    }
    predictions['current_only'] = predict(control, XT[:, -1, :])
    scores = {name: {**balanced_scores(p, test), 'cue_pairs': pair_scores(p, test)}
              for name, p in predictions.items()}
    dev_score = balanced_scores(predict(net, XD), dev)
    flips = sum(pred == TARGETS[2 * meta['author'] + (1 - (meta['cue'] ^ meta['invert']))]
                for pred, (_x, _target, meta) in zip(predictions['beacon_flip'], test))
    params = {name: sum(p.numel() for p in m.parameters()) for name, m in
              (('recurrent', net), ('current_only', control))}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = os.getenv('GITHUB_SHA', 'LOCAL-UNVERIFIED')
    ck = out / 'memory_policy.pt'
    torch.save(dict(schema=SCHEMA, source_sha=sha,
                    architecture=dict(hidden=args.hidden, layers=args.layers),
                    action_contract=ACTIONS, weights=net.state_dict(),
                    control=control.state_dict(), updates=args.updates,
                    train_worlds=args.train_worlds), ck)
    report = dict(
        schema=SCHEMA, source_sha=sha,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        provenance='from-scratch recurrent synthetic delayed-cue game decisions',
        architecture=dict(hidden=args.hidden, layers=args.layers), parameters=params,
        training=dict(updates=args.updates, batch=args.batch, train_examples=len(train),
                      dev_examples=len(dev), test_examples=len(test),
                      train_worlds=args.train_worlds, dev_worlds=args.eval_worlds),
        dev=dev_score, scores=scores,
        beacon_flip_target_correct=round(flips / len(test), 6),
        CPU_unbatched_forward_p50_ms=round(statistics.median(times), 3),
        CPU_unbatched_forward_p95_ms=round(sorted(times)[94], 3),
        CPU_train_step_p50_ms=round(statistics.median(updates_ms), 3),
        peak_RSS_MiB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        losses=losses,
        checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
        restrictions=['external synthetic training, not native CyberSand simulation',
                      'full paired counterfactual world groups kept within each split',
                      'both presentations of each earlier clue are ablated',
                      'host legal action mask is authoritative',
                      'no game code/assets/provider defaults changed; Draft/HOLD'])
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'REPORT.md').write_text(
        '# MZ-008 — delayed-cue memory game-policy probe\n\n'
        + 'Source: ' + sha + '\n\n'
        + 'Parameters: ' + str(params) + '\n'
        + 'Real updates: ' + str(args.updates) + '\n'
        + 'Held-out recurrent: ' + str(scores['recurrent']['accuracy']) + '\n'
        + 'Current-only: ' + str(scores['current_only']['accuracy']) + '\n'
        + 'History erased: ' + str(scores['erase_history']['accuracy']) + '\n'
        + 'Synthetic research only, no game changes; Draft/HOLD.\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'losses'},
                     indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='mz008-evidence')
    p.add_argument('--train-worlds', type=int, default=96)
    p.add_argument('--eval-worlds', type=int, default=24)
    p.add_argument('--updates', type=int, default=550)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=.002)
    run(p.parse_args())


if __name__ == '__main__':
    main()
