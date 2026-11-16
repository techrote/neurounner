"""MZ-009: held-out partially observed material-fork game-policy research.

Independent synthetic environment only; never modifies CyberSand gameplay code.
A beacon and later inversion determine a latent permitted material.  At three
successive forks, the visible candidate materials change, but the earlier rule
is absent.  Wrong routes collide with protected terrain and require RETRACE.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, statistics, time
from dataclasses import dataclass
from pathlib import Path

ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'RETRACE', 'STOP')
SCHEMA = 'MZ009-MATERIAL-FORK-MEMORY-v1'
D = 32
MAX_STEPS = 40


class World:
    def __init__(self, seed, beacon, invert):
        self.seed, self.beacon, self.invert = seed, beacon, invert
        rng = random.Random(seed * 99991 + 1337)
        self.left_materials = [rng.randrange(2) for _ in range(3)]
        self.decor = [rng.random() for _ in range(6)]
        self.stage = 'corridor'
        self.corridor = 0
        self.fork = 0
        self.tick = 0
        self.collisions = 0
        self.painted = 0
        self.done = False
        self.failed = False
        self.previous_action = 0

    def legal(self):
        if self.stage == 'corridor': return [1, 0, 0, 0, 1]
        if self.stage == 'fork': return [0, 1, 1, 0, 1]
        if self.stage == 'hazard': return [0, 0, 0, 1, 1]
        return [0, 0, 0, 0, 1]

    def observe(self):
        x = [0.] * D
        stage_idx = {'corridor': 0, 'fork': 1, 'hazard': 2, 'complete': 3}[self.stage]
        x[stage_idx] = 1.
        x[4] = self.fork / 3.
        x[5] = self.corridor / 8.
        x[6] = self.tick / MAX_STEPS
        x[7] = self.collisions / 3.
        x[8:13] = self.legal()
        if self.stage == 'fork':
            x[13 + self.left_materials[self.fork]] = 1.
            x[15 + (1 - self.left_materials[self.fork])] = 1.
        if self.tick in (0, 1): x[17 + self.beacon] = 2.
        if self.tick in (2, 3): x[19 + self.invert] = 2.
        x[21 + self.previous_action] = 1.
        x[26:32] = self.decor
        return x

    def preferred_action(self):
        if self.stage == 'corridor': return 0
        if self.stage == 'hazard': return 3
        if self.stage == 'complete': return 4
        safe_material = self.beacon ^ self.invert
        return 1 if self.left_materials[self.fork] == safe_material else 2

    def step(self, action):
        if self.done: raise RuntimeError('step after terminal')
        if not self.legal()[action]: raise ValueError('unmasked illegal action')
        self.previous_action = action
        self.tick += 1
        if action == 4:
            self.done = True
            self.failed = self.stage != 'complete'
        elif self.stage == 'corridor':
            self.corridor += 1
            if self.corridor >= (8 if self.fork == 0 else 6):
                self.stage = 'fork'
        elif self.stage == 'fork':
            if action == self.preferred_action():
                self.painted += 1
                self.fork += 1
                self.corridor = 0
                self.stage = 'complete' if self.fork == 3 else 'corridor'
            else:
                self.collisions += 1
                self.stage = 'hazard'
        elif self.stage == 'hazard':
            self.stage = 'fork'
        return self.done


def episode(seed, beacon, invert, injected_fork=-1):
    w = World(seed, beacon, invert)
    obs, targets = [], []
    injected = False
    for _ in range(MAX_STEPS):
        x = w.observe()
        teacher = w.preferred_action()
        obs.append(x)
        targets.append(teacher)
        forced = (w.stage == 'fork' and w.fork == injected_fork and not injected)
        action = (3 - teacher) if forced else teacher  # LEFT <-> RIGHT
        injected |= forced
        w.step(action)
        if w.done: break
    assert w.done and not w.failed and w.painted == 3
    return obs, targets, (seed, beacon, invert, injected_fork)


def dataset(seeds, injection=(-1, 0, 1)):
    return [episode(seed, cue, inv, bad) for seed in seeds
            for cue in range(2) for inv in range(2) for bad in injection]


def to_tensors(torch, rows):
    x = torch.zeros((len(rows), MAX_STEPS, D), dtype=torch.float32)
    y = torch.full((len(rows), MAX_STEPS), -100, dtype=torch.long)
    for i, (seq, labels, _) in enumerate(rows):
        assert len(seq) <= MAX_STEPS
        x[i, :len(seq)] = torch.tensor(seq)
        y[i, :len(seq)] = torch.tensor(labels)
    return x, y


def networks(torch, hidden=128, layers=2):
    nn = torch.nn
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(D)
            self.core = nn.GRU(D, hidden, layers, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(D + hidden),
                                      nn.Linear(D + hidden, 96), nn.GELU(),
                                      nn.Linear(96, len(ACTIONS)))
        def forward(self, x):
            z = self.norm(x)
            h = self.core(z)[0]
            return self.head(torch.cat((z, h), dim=-1))
    class Reactive(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 128),
                                     nn.GELU(), nn.Linear(128, len(ACTIONS)))
        def forward(self, x): return self.net(x)
    return Recurrent(), Reactive()


def class_scores(logits, y, x):
    valid = y != -100
    pred = logits.argmax(-1)
    forks = valid & (x[:, :, 1] > 0)
    masks = x[:, :, 8:13]
    illegal = ((masks.gather(-1, pred.unsqueeze(-1)).squeeze(-1) == 0) & valid).sum().item()
    return dict(all_accuracy=round(float(((pred == y) & valid).sum()) / int(valid.sum()), 5),
                fork_accuracy=round(float(((pred == y) & forks).sum()) / max(1, int(forks.sum())), 5),
                fork_decisions=int(forks.sum()), raw_illegal=int(illegal))


def play(torch, net, seed, cue, inv, force_error=False, context_limit=None):
    w = World(seed, cue, inv)
    past = []
    raw_illegal = 0
    used_injection = False
    fork_pred = []
    fork_events = []
    for _ in range(MAX_STEPS):
        obs = w.observe()
        past.append(obs)
        features = past if context_limit is None else past[-context_limit:]
        with torch.inference_mode():
            logits = net(torch.tensor([features], dtype=torch.float32))[0, -1]
        raw = int(logits.argmax())
        legal = w.legal()
        raw_illegal += (not legal[raw])
        action = int(logits.masked_fill(torch.tensor([not b for b in legal]), -1e9).argmax())
        if w.stage == 'fork':
            fork_pred.append(action)
            if force_error and w.fork == 0 and not used_injection:
                action = 3 - w.preferred_action()
                used_injection = True
        if w.stage == 'fork':
            fork_events.append(dict(fork=w.fork, action=ACTIONS[action],
                                    safe=ACTIONS[w.preferred_action()],
                                    material_left=w.left_materials[w.fork],
                                    hit=action == w.preferred_action()))
        w.step(action)
        if w.done: break
    return dict(success=w.done and not w.failed and w.painted == 3,
                painted=w.painted, collisions=w.collisions,
                illegal_raw=raw_illegal, decisions=w.tick, fork_choices=fork_pred, fork_events=fork_events,
                clean_success=w.done and not w.failed and w.painted == 3 and w.collisions == 0)


def evaluate_rollouts(torch, net, seeds, context_limit=None, force_error=False):
    episodes = [play(torch, net, s, cue, inv, force_error, context_limit)
                for s in seeds for cue in range(2) for inv in range(2)]
    n = len(episodes)
    return dict(episodes=n, successes=sum(r['success'] for r in episodes),
                success_rate=round(sum(r['success'] for r in episodes)/n, 5),
                clean_success_rate=round(sum(r['clean_success'] for r in episodes)/n, 5),
                painted=sum(r['painted'] for r in episodes),
                collisions=sum(r['collisions'] for r in episodes),
                raw_illegal=sum(r['illegal_raw'] for r in episodes),
                mean_decisions=round(sum(r['decisions'] for r in episodes)/n, 3),
                sample=episodes[:8])



def build_gallery(torch, nets, contexts, seeds):
    """Original source-traceable fork decisions; red marks protected collision."""
    titles = ('GRU full history', 'Reactive MLP', 'GRU last observation', 'GRU four observations')
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="450" viewBox="0 0 980 450">',
             '<rect width="980" height="450" fill="#151e2a"/>',
             '<text x="16" y="25" fill="white" font-size="17">MZ-009 | three material forks, closed-loop original actions</text>']
    for row, (net, context, title) in enumerate(zip(nets, contexts, titles)):
        ybase = 52 + row * 96
        parts.append(f'<text x="16" y="{ybase+14}" fill="#e2e8f0" font-size="12">{title}</text>')
        for col, seed in enumerate(seeds):
            cue, inv = col % 2, (col // 2) % 2
            trace = play(torch, net, seed, cue, inv, context_limit=context)
            xbase = 186 + col * 196
            parts.append(f'<rect x="{xbase}" y="{ybase}" width="180" height="78" rx="5" fill="#27364a"/>')
            parts.append(f'<text x="{xbase+5}" y="{ybase+13}" fill="#fff" font-size="10">world {seed} / cue {cue} / inv {inv}</text>')
            for e in trace['fork_events']:
                idx = e['fork']; x = xbase + 20 + idx * 50
                # Left/right candidate material dots: distinct mineral colours.
                parts.append(f'<circle cx="{x}" cy="{ybase+36}" r="6" fill="{("#b8a16a", "#91bbc7")[e["material_left"]]}"/>')
                parts.append(f'<circle cx="{x}" cy="{ybase+58}" r="6" fill="{("#91bbc7", "#b8a16a")[e["material_left"]]}"/>')
                if e['action'] in ('LEFT', 'RIGHT'):
                    yy = ybase + (36 if e['action'] == 'LEFT' else 58)
                    parts.append(f'<circle cx="{x}" cy="{yy}" r="9" fill="none" stroke="{("#2dd4bf" if e["hit"] else "#fb7185")}" stroke-width="2"/>')
            parts.append(f'<text x="{xbase+5}" y="{ybase+74}" fill="#e2e8f0" font-size="10">clean {int(trace["clean_success"])} / hits {trace["collisions"]}</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(59009)
    gen = torch.Generator().manual_seed(59011)
    train = dataset(range(1000, 1000 + args.train_seeds))
    dev = dataset(range(3000, 3000 + args.eval_seeds))
    test = dataset(range(5000, 5000 + args.eval_seeds))
    X, Y = to_tensors(torch, train)
    XD, YD = to_tensors(torch, dev)
    XT, YT = to_tensors(torch, test)
    net, baseline = networks(torch, args.hidden, args.layers)
    params = {'recurrent': sum(p.numel() for p in net.parameters()),
              'current_only': sum(p.numel() for p in baseline.parameters())}
    optimizers = [torch.optim.AdamW(n.parameters(), lr=args.lr) for n in (net, baseline)]
    weight = torch.tensor([.5, 3., 3., 1.5, .5])
    losses = {'recurrent': [], 'current_only': []}
    update_ms = []
    for _ in range(args.updates):
        ids = torch.randint(len(X), (args.batch,), generator=gen)
        for name, model, opt in [('recurrent', net, optimizers[0]),
                                  ('current_only', baseline, optimizers[1])]:
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits = model(X[ids])
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), Y[ids],
                                                      weight=weight, ignore_index=-100)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses[name].append(round(float(loss.detach()), 6))
            if name == 'recurrent': update_ms.append((time.perf_counter() - start)*1000)
    net.eval(); baseline.eval()
    with torch.inference_mode():
        scores = {'recurrent': class_scores(net(XT), YT, XT),
                  'current_only': class_scores(baseline(XT), YT, XT),
                  'dev': class_scores(net(XD), YD, XD)}
        flip = XT.clone()
        flip[:, :2, 17:19] = flip[:, :2, 17:19].flip(-1)
        normal_preds = net(XT).argmax(-1)
        flipped_preds = net(flip).argmax(-1)
        forkmask = (XT[:, :, 1] > 0) & (YT != -100)
        fork_zero = forkmask & (XT[:, :, 4] == 0)
        firstfork = fork_zero & (torch.cumsum(fork_zero.int(), dim=1) == 1)
        flip_fraction = float(((flipped_preds != normal_preds) & firstfork).sum()) / max(1,int(firstfork.sum()))
    rollout_seeds = range(7000, 7000+args.rollout_seeds)
    rollouts = {
        'recurrent': evaluate_rollouts(torch, net, rollout_seeds),
        'current_only': evaluate_rollouts(torch, baseline, rollout_seeds),
        'recurrent_context_1': evaluate_rollouts(torch, net, rollout_seeds, context_limit=1),
        'recurrent_context_4': evaluate_rollouts(torch, net, rollout_seeds, context_limit=4),
        'recurrent_context_8': evaluate_rollouts(torch, net, rollout_seeds, context_limit=8),
        'recurrent_forced_wrong_first_fork': evaluate_rollouts(torch, net, rollout_seeds, force_error=True),
    }
    gallery = build_gallery(torch, [net, baseline, net, net],
                            [None, None, 1, 4], range(7000, 7004))
    for _ in range(10): play(torch, net, 9000, 0, 0)
    times = []
    for _ in range(60):
        t = time.perf_counter()
        with torch.inference_mode(): net(XT[:1, :8])
        times.append(1000*(time.perf_counter()-t))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sha = os.getenv('GITHUB_SHA', 'LOCAL-UNVERIFIED')
    checkpoint = out/'policy.pt'
    torch.save(dict(schema=SCHEMA, source_sha=sha, hidden=args.hidden,
                    layers=args.layers, actions=ACTIONS,
                    recurrent=net.state_dict(), current_only=baseline.state_dict()), checkpoint)
    report = dict(schema=SCHEMA, source_sha=sha, parameters=params,
                  training=dict(seeds=args.train_seeds, eval_seeds=args.eval_seeds,
                                updates=args.updates, batch=args.batch, learning_rate=args.lr,
                                train_episodes=len(train), test_episodes=len(test)),
                  scores=scores, first_fork_beacon_flip_action_switch_rate=round(flip_fraction, 5),
                  rollouts=rollouts,
                  cpu_forward_8_step_p50_ms=round(statistics.median(times), 4),
                  cpu_forward_8_step_p95_ms=round(sorted(times)[56], 4),
                  train_step_p50_ms=round(statistics.median(update_ms), 4),
                  peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  losses=losses, limitations=['synthetic procedural host, no native CyberSand',
                      'obstacle painting represented by hidden protected-material collision',
                      'teacher-forced classification versus separately closed-loop actual actions',
                      'matched updates not matched compute', 'Draft/HOLD; no game changes'])
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'trajectories.svg').write_text(gallery)
    (out/'REPORT.md').write_text('# MZ-009 partially observed material fork game policy\n\n'+
        f"Source: `{sha}`\n\nRecurrent TEST fork accuracy: {scores['recurrent']['fork_accuracy']}\n\n"+
        f"Reactive TEST fork accuracy: {scores['current_only']['fork_accuracy']}\n\n"+
        f"Closed-loop success: {rollouts['recurrent']['success_rate']} vs {rollouts['current_only']['success_rate']}\n\n"+
        'Synthetic only; no production game code changed. Draft/HOLD.\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'losses'},indent=2),flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='mz009-evidence')
    p.add_argument('--train-seeds', type=int, default=160)
    p.add_argument('--eval-seeds', type=int, default=24)
    p.add_argument('--rollout-seeds', type=int, default=24)
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=.003)
    run(p.parse_args())

if __name__ == '__main__': main()