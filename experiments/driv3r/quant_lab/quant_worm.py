#!/usr/bin/env python3
"""QL-001: independent, CyberSand-style SYNTHETIC quantized game worms.

Not GGUF/IQ kernels or native CyberSand. The quantized paths store actual bit-packed
weights, then dequantize them for ordinary CPU matmul; no artificial speed claims.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "RETRACE", "BRANCH", "STOP")
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
FAMILIES = ("GREEBLES", "PIPEWORK", "ARCHITECTURE")
SIZE = 25
FEATS = 27


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


class WormWorld:
    def __init__(self, seed, family):
        self.seed, self.family = seed, family
        r = random.Random(seed)
        self.grid = np.zeros((SIZE, SIZE), np.uint8)
        for y in range(SIZE):
            for x in range(SIZE):
                v = r.random()
                self.grid[y, x] = 1 if v < 0.085 else 2 if v < 0.115 else 0
        k = SIZE // 2
        self.grid[k-2:k+3, k-2:k+3] = 0
        self.pos = (k, k)
        self.heading = r.randrange(4)
        self.path = [self.pos]
        self.visited = {self.pos}
        self.ink = {self.pos}
        self.branches = 0
        self.stopped = False
        self.actions = []

    def neighbor(self, delta):
        dx, dy = DIRS[(self.heading + delta) % 4]
        return self.pos[0] + dx, self.pos[1] + dy

    def state(self, xy):
        x, y = xy
        if x < 0 or y < 0 or x >= SIZE or y >= SIZE:
            return -1
        if self.grid[y, x] == 2:
            return -2
        if self.grid[y, x] == 1:
            return -1
        return 0 if xy in self.visited else 1

    def legal(self):
        if self.stopped:
            return ("STOP",)
        ans = [name for d, name in ((0, "STRAIGHT"), (-1, "LEFT"), (1, "RIGHT"))
               if self.state(self.neighbor(d)) >= 0]
        if len(self.path) > 1:
            ans.append("RETRACE")
        if len(self.ink) >= 4 and sum(self.state(self.neighbor(d)) == 1 for d in (0, -1, 1)) >= 2:
            ans.append("BRANCH")
        ans.append("STOP")
        return tuple(ans)

    def features(self, tick):
        local = [self.state(self.neighbor(d)) for d in (0, -1, 1)]
        x, y = self.pos
        values = []
        for st in local:
            values.extend([float(st == -2), float(st == -1), float(st == 0), float(st == 1)])
        values.extend([float(self.heading == k) for k in range(4)])
        values.extend([float(self.family == k) for k in FAMILIES])
        values.extend([min(tick, 64)/64, min(len(self.ink), 64)/64,
                       min(len(self.path), 64)/64, min(self.branches, 16)/16,
                       (x - SIZE / 2) / SIZE, (y - SIZE / 2) / SIZE,
                       float(len(self.legal()) - 1)/5, float(len(self.actions) > 0 and self.actions[-1] == 'RETRACE')])
        assert len(values) == FEATS
        return np.asarray(values, dtype=np.float32)

    def teacher(self, tick):
        legal = self.legal()
        if tick >= 40 or len(self.ink) >= 52:
            return "STOP"
        if "BRANCH" in legal and tick % (9 if self.family == 'GREEBLES' else 6) == 5:
            return 'BRANCH'
        wishes = {'GREEBLES': ('LEFT', 'STRAIGHT', 'RIGHT'),
                  'PIPEWORK': ('STRAIGHT', 'RIGHT', 'LEFT'),
                  'ARCHITECTURE': ('RIGHT', 'STRAIGHT', 'LEFT')}[self.family]
        if tick % 7 in (1, 2, 3):
            wishes = ('STRAIGHT',) + tuple(x for x in wishes if x != 'STRAIGHT')
        for v in wishes:
            if v in legal and self.state(self.neighbor({'LEFT': -1, 'RIGHT': 1, 'STRAIGHT': 0}[v])) == 1:
                return v
        for v in wishes:
            if v in legal:
                return v
        return 'RETRACE' if 'RETRACE' in legal else 'STOP'

    def apply(self, choice):
        if choice not in self.legal():
            raise ValueError(f'host-illegal {choice}')
        prev = self.pos
        if choice == 'STOP':
            self.stopped = True
        elif choice == 'RETRACE':
            self.path.pop()
            self.pos = self.path[-1]
        else:
            if choice == 'BRANCH':
                choices = [d for d in (-1, 1, 0) if self.state(self.neighbor(d)) == 1]
                assert choices
                delta = choices[self.branches % len(choices)]
                self.branches += 1
            else:
                delta = {'STRAIGHT': 0, 'LEFT': -1, 'RIGHT': 1}[choice]
            self.heading = (self.heading + delta) % 4
            self.pos = self.neighbor(0)
            if self.state(self.pos) < 0:
                raise AssertionError('painted forbidden cell')
            self.path.append(self.pos)
            self.visited.add(self.pos)
            self.ink.add(self.pos)
        self.actions.append(choice)
        return prev, self.pos


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(FEATS, 64)
        self.b = torch.nn.Linear(64, 64)
        self.c = torch.nn.Linear(64, len(ACTIONS))

    def forward(self, x):
        return self.c(torch.tanh(self.b(torch.tanh(self.a(x)))))


def collect(seeds, rng_seed):
    rng = random.Random(rng_seed)
    rows = []
    hist = Counter()
    for seed in seeds:
        for family in FAMILIES:
            world = WormWorld(seed, family)
            for tick in range(48):
                legal = world.legal()
                label = world.teacher(tick)
                assert label in legal
                rows.append((world.features(tick), ACTIONS.index(label)))
                hist[label] += 1
                action = rng.choice(legal) if rng.random() < .23 else label
                world.apply(action)
                if world.stopped:
                    break
    x = torch.from_numpy(np.stack([v for v, _ in rows]))
    y = torch.tensor([a for _, a in rows], dtype=torch.long)
    return (x, y), dict(hist)


def quant_numpy(weight, bits):
    a = np.asarray(weight, dtype=np.float32)
    if bits == 32:
        return a.copy(), None, None
    flat = a.reshape(a.shape[0], -1)
    if bits == 1:
        scale = np.mean(np.abs(flat), axis=1).astype(np.float32)
        code = (flat >= 0).astype(np.uint8)
        rec = (code.astype(np.float32) * 2 - 1) * scale[:, None]
    else:
        level = (1 << (bits - 1)) - 1
        scale = (np.max(np.abs(flat), axis=1) / level).astype(np.float32)
        safe = np.where(scale > 0, scale, 1.0)
        signed = np.clip(np.rint(flat / safe[:, None]), -level, level).astype(np.int32)
        code = (signed + level).astype(np.uint8)
        rec = signed.astype(np.float32) * scale[:, None]
    return rec.reshape(a.shape), code, scale


def pack_codes(codes, bits):
    values = codes.reshape(-1).astype(np.uint8)
    b = np.unpackbits(values[:, None], axis=1, bitorder='little')[:, :bits].reshape(-1)
    return np.packbits(b, bitorder='little')


def unpack_codes(data, bits, n):
    raw = np.unpackbits(np.asarray(data, np.uint8), bitorder='little')[:n*bits]
    return (raw.reshape(-1, bits).astype(np.uint16) * (2 ** np.arange(bits, dtype=np.uint16))).sum(axis=1).astype(np.uint8)


def save_packed(root, name, model, levels):
    arrays = {}
    metadata = {}
    raw_bytes = 0
    for layer, bits in zip('abc', levels):
        w = model.state_dict()[layer+'.weight'].detach().numpy().copy()
        b = model.state_dict()[layer+'.bias'].detach().numpy().copy()
        arrays[layer+'_bias'] = b
        if bits == 32:
            arrays[layer+'_fp32'] = w
            raw_bytes += w.nbytes + b.nbytes
        else:
            _, codes, scale = quant_numpy(w, bits)
            packed = pack_codes(codes, bits)
            arrays[layer+'_packed'] = packed
            arrays[layer+'_scale'] = scale
            raw_bytes += packed.nbytes + scale.nbytes + b.nbytes
        metadata[layer] = {'bits': bits, 'shape': list(w.shape)}
    path = root/(name+'.npz')
    np.savez(path, **arrays)
    return path, metadata, raw_bytes


def load_packed(path, metadata):
    out = Policy()
    sd = {}
    with np.load(path, allow_pickle=False) as f:
        for layer, spec in metadata.items():
            bits, shape = spec['bits'], spec['shape']
            if bits == 32:
                rec = f[layer+'_fp32']
            else:
                data = f[layer+'_packed']
                code = unpack_codes(data, bits, math.prod(shape)).reshape(shape[0], -1)
                scale = f[layer+'_scale']
                if bits == 1:
                    rec = ((code.astype(np.float32) * 2 - 1) * scale[:, None]).reshape(shape)
                else:
                    level = (1 << (bits - 1)) - 1
                    rec = ((code.astype(np.float32)-level) * scale[:, None]).reshape(shape)
            sd[layer+'.weight'] = torch.from_numpy(np.asarray(rec, np.float32).copy())
            sd[layer+'.bias'] = torch.from_numpy(f[layer+'_bias'].copy())
    out.load_state_dict(sd)
    out.eval()
    return out


def qat_repair(original, data, bits, steps, seed):
    torch.manual_seed(seed)
    model = copy.deepcopy(original)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    x, y = data
    losses = []
    for i in range(steps):
        idx = torch.randint(len(x), (128,))
        act = x[idx]
        for layer in 'abc':
            mod = getattr(model, layer)
            wt = mod.weight
            q = torch.from_numpy(quant_numpy(wt.detach().numpy(), bits)[0])
            act = torch.nn.functional.linear(act, wt + (q - wt).detach(), mod.bias)
            if layer != 'c':
                act = torch.tanh(act)
        loss = torch.nn.functional.cross_entropy(act, y[idx])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    return model, {'optimizer': 'AdamW', 'steps': steps, 'loss_first': losses[0],
                   'loss_last': losses[-1], 'nonzero_optimizer_moment_tensors': sum(
                   int(torch.count_nonzero(s['exp_avg']).item() > 0) for s in optim.state.values())}


def sparse_channels(original, fraction):
    model = copy.deepcopy(original)
    with torch.no_grad():
        scores = model.a.weight.abs().sum(dim=1)
        indices = torch.argsort(scores)[:round(len(scores)*fraction)]
        model.a.weight[indices] = 0
        model.a.bias[indices] = 0
        model.b.weight[:, indices] = 0
    return model


def draw(world, path, scale=13):
    colors = {0: (243, 239, 221), 1: (79, 82, 94), 2: (220, 89, 96)}
    img = Image.new('RGB', (SIZE*scale, SIZE*scale))
    pix = img.load()
    for y in range(SIZE):
        for x in range(SIZE):
            p = (x, y)
            rgb = (246, 181, 65) if p == world.pos else (46, 103, 142) if p in world.ink else colors[int(world.grid[y, x])]
            for yy in range(y*scale, (y+1)*scale):
                for xx in range(x*scale, (x+1)*scale):
                    pix[xx, yy] = rgb
    img.save(path, optimize=True)


def rollout(model, policy, family, seed, out, horizon=64):
    world = WormWorld(seed, family)
    r = random.Random(1000003*seed)  # same host draw stream across all family counterfactuals
    steps = []
    for tick in range(horizon):
        if world.stopped:
            break
        legal = world.legal()
        values = None
        raw = None
        if model is not None:
            with torch.no_grad():
                values = model(torch.from_numpy(world.features(tick)).unsqueeze(0))[0].tolist()
            raw = ACTIONS[int(np.argmax(values))]
            masked = [v if a in legal else -1e9 for a, v in zip(ACTIONS, values)]
            action = r.choice(legal) if r.random() < 0.10 else ACTIONS[int(np.argmax(masked))]
        elif policy == 'teacher':
            action = world.teacher(tick)
        elif policy == 'random':
            action = r.choice(legal)
        elif policy == 'wall':
            action = next(a for a in ('STRAIGHT', 'RIGHT', 'LEFT', 'BRANCH', 'RETRACE', 'STOP') if a in legal)
        else:
            preferences = ('LEFT', 'RIGHT', 'BRANCH', 'STRAIGHT', 'RETRACE', 'STOP') if tick % 2 else ('RIGHT', 'LEFT', 'BRANCH', 'STRAIGHT', 'RETRACE', 'STOP')
            action = next(a for a in preferences if a in legal)
        pos = list(world.pos)
        world.apply(action)
        steps.append({'tick': tick, 'position': pos, 'legal': list(legal),
                      'raw_top': raw, 'chosen': action,
                      'scores': None if values is None else [round(float(v), 7) for v in values],
                      'next_position': list(world.pos)})
    png = out/'gallery'/f'{policy}-{family.lower()}-{seed}.png'
    draw(world, png)
    counts = Counter(world.actions)
    return ({'policy': policy, 'family': family, 'seed': seed, 'turns': len(steps),
             'unique_cells': len(world.ink), 'branch_count': counts['BRANCH'],
             'retrace_count': counts['RETRACE'], 'stop_count': counts['STOP'],
             'unmasked_legal_rate': None if model is None else sum(v['raw_top'] in v['legal'] for v in steps)/max(1, len(steps)),
             'action_histogram': dict(counts), 'png': str(png.relative_to(out)),
             'png_sha256': digest(png)}, steps)


def timing(model, x, n=400):
    torch.set_num_threads(1)
    q = x[:min(len(x), 96)]
    with torch.no_grad():
        for _ in range(12):
            model(q[0:1])
        start = time.perf_counter()
        times = []
        for i in range(n):
            t = time.perf_counter_ns()
            model(q[i % len(q):i % len(q)+1])
            times.append((time.perf_counter_ns()-t)/1e6)
    return {'same_process_p50_ms': round(statistics.median(times), 5),
            'same_process_p95_ms': round(sorted(times)[math.ceil(.95*len(times))-1], 5),
            'decisions_per_second': round(n/(time.perf_counter()-start), 1)}


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(44177)
    np.random.seed(44177)
    out = Path(args.out)
    (out/'gallery').mkdir(parents=True, exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    train, train_hist = collect(range(1100, 1164), 8003)
    held, held_hist = collect(range(8100, 8124), 8004)
    assert not set(range(1100,1164)) & set(range(8100,8124))
    net = Policy()
    untuned = copy.deepcopy(net)  # independently stored pre-optimizer negative control
    opt = torch.optim.AdamW(net.parameters(), lr=0.004)
    losses = []
    for i in range(args.steps):
        ix = torch.randint(0,len(train[0]),(128,))
        loss = torch.nn.functional.cross_entropy(net(train[0][ix]), train[1][ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    torch.save({'state_dict':net.state_dict(), 'optimizer': opt.state_dict(),
                'source_sha': os.environ.get('GITHUB_SHA','local')}, out/'models'/'fp32-trained.pt')
    repair, repair_diag = qat_repair(net, train, 2, args.repair_steps, 9941)
    mutants = sparse_channels(net, .30)
    variants = {
        'untuned-fp32': (untuned, (32,32,32)),
        'untuned-q1': (untuned, (1,1,1)),
        'fp32': (net, (32,32,32)),
        'q8': (net, (8,8,8)),
        'q4': (net, (4,4,4)),
        'q3': (net, (3,3,3)),
        'q2': (net, (2,2,2)),
        'q1': (net, (1,1,1)),
        'mixed-q1q4q4': (net,(1,4,4)),
        'mixed-q4q1q4': (net,(4,1,4)),
        'q2-sparse30': (mutants,(2,2,2)),
        'q2-qat-repair': (repair,(2,2,2)),
    }
    results = {}
    all_traces = []
    baselines = {x: None for x in ('random','wall','fidget','teacher')}
    def evaluation(model, name, description):
        episodes = []
        for seed in (8101,8107,8113):
            for family in FAMILIES:
                entry, steps = rollout(model, name, family, seed, out)
                episodes.append(entry)
                all_traces.append({'policy': name, 'seed': seed, 'family': family, 'steps': steps})
        action_counts = Counter()
        for e in episodes:
            action_counts.update(e['action_histogram'])
        return {'episodes':episodes, 'mean_unique_cells':round(statistics.mean(e['unique_cells'] for e in episodes),3),
                'mean_turns':round(statistics.mean(e['turns'] for e in episodes),3),
                'action_counts':dict(action_counts), 'description':description}
    for control in baselines:
        baselines[control] = evaluation(None, control, 'No model; same obstacle seeds/horizon, divergent states')
    with torch.no_grad():
        for name, (source, levels) in variants.items():
            path, meta, byte_count = save_packed(out/'models',name,source,levels)
            loaded = load_packed(path, meta)
            ref = torch.cat([torch.from_numpy(quant_numpy(source.state_dict()[layer+'.weight'].numpy(), bits)[0]).reshape(-1)
                             for layer,bits in zip('abc',levels)])
            got = torch.cat([loaded.state_dict()[layer+'.weight'].reshape(-1) for layer in 'abc'])
            assert torch.equal(ref,got), name+' packing roundtrip failure'
            measure = timing(loaded, held[0], 250)
            pred = loaded(held[0]).argmax(1)
            denom = len(held[1])
            same = int((pred==held[1]).sum())
            action_support = {ACTIONS[i]: int((held[1] == i).sum()) for i in range(len(ACTIONS))}
            per_action = {ACTIONS[i]: {'n': action_support[ACTIONS[i]],
                'correct': int(((pred == i) & (held[1] == i)).sum())}
                for i in range(len(ACTIONS))}
            for a, row in per_action.items():
                row['recall'] = None if row['n'] == 0 else round(row['correct'] / row['n'], 6)
            supported = [v['recall'] for v in per_action.values() if v['recall'] is not None]
            predicted_hist = {a: int((pred == i).sum()) for i, a in enumerate(ACTIONS)}
            score = evaluation(loaded, name, 'Host-legal masked local-action steering, free STOP, no native painting')
            results[name] = {'quant_bits':list(levels), 'packed_weight_and_bias_bytes':byte_count,
                             'actual_archive_bytes':path.stat().st_size, 'archive_sha256':digest(path),
                             'layer_shapes_and_bits':meta,'heldout_teacher_accuracy':round(same/denom,5),
                             'heldout_count':denom,'heldout_balanced_recall_supported_classes':round(statistics.mean(supported),6),
                             'heldout_per_action':per_action,'heldout_prediction_counts':predicted_hist,
                             'warm_cpu':measure, **score}
    (out/'decisions.json').write_text(json.dumps(all_traces,separators=(',', ':'))+'\n')
    result = {'schema':'driv3r-quant-lab-001/1','source_sha':os.environ.get('GITHUB_SHA','local'),
              'game_type':'CyberSand-inspired SYNTHETIC toy; NOT native CyberSand/Workbench; not GGUF IQ kernel',
              'objective':'tiny trained local-action neural controller, NOT quantized pretrained SLM',
              'quant_note':'real bit-packed scalar 1/2/3/4/8 weights; packed weights dequantized to float32 for PyTorch CPU matmul; no packed lowbit acceleration',
              'resident_rss_note':'process-wide RSS is dominated by PyTorch overhead, do not interpret as quant-only model RSS',
              'seed':44177,'horizon':64,'train_world_seeds':'1100..1163',
              'heldout_world_seeds':'8100..8123','visual_world_seeds':[8101,8107,8113],
              'train_samples':len(train[1]),'heldout_samples':len(held[1]),
              'train_histogram':train_hist,'heldout_histogram':held_hist,
              'heldout_majority_teacher_accuracy':round(max(held_hist.values()) / len(held[1]),6),
              'heldout_unsupported_actions':[a for a in ACTIONS if a not in held_hist],
              'training':{'optimizer':'AdamW','steps':args.steps,'loss_first':losses[0],
                          'loss_last':losses[-1], 'nonzero_optimizer_moment_tensors':sum(
                              int(torch.count_nonzero(s['exp_avg']).item()>0) for s in opt.state.values())},
              'repair':repair_diag,'baselines':baselines,'variants':results,
              'raw_decisions_sha256':digest(out/'decisions.json'),
              'rss_kib_at_end_linux':int(next((l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')),0)) if Path('/proc/self/status').exists() else None,
              'fp32_checkpoint_sha256':digest(out/'models'/'fp32-trained.pt')}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest = [{'path':str(p.relative_to(out)), 'sha256':digest(p), 'bytes':p.stat().st_size}
                for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json']
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','source_sha':result['source_sha'],
                      'optimizer_updates':args.steps,'repair_steps':args.repair_steps,
                      'training_loss':[losses[0],losses[-1]],
                      'heldout_size':len(held[1]),
                      'packed':{k:{'bytes':v['packed_weight_and_bias_bytes'],
                                   'accuracy':v['heldout_teacher_accuracy'],
                                   'cells':v['mean_unique_cells'],
                                   'turns':v['mean_turns']} for k,v in results.items()}}))


def self_test():
    w = WormWorld(100, 'GREEBLES')
    try:
        w.apply('MAGIC')
    except ValueError:
        pass
    else:
        raise AssertionError('host illegal action accepted')
    for bits in (1,2,3,4,8):
        ar = np.random.default_rng(33).standard_normal((11,19)).astype(np.float32)
        rec, codes, scales = quant_numpy(ar,bits)
        tmp = pack_codes(codes,bits)
        back = unpack_codes(tmp,bits,codes.size).reshape(codes.shape)
        assert np.array_equal(codes,back)
        assert np.isfinite(rec).all()
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        model = Policy()
        for n in (1,2,3,4,8,32):
            p,m,_ = save_packed(Path(t),'test',model,(n,n,n))
            dec = load_packed(p,m)
            for layer in 'abc':
                expected = quant_numpy(model.state_dict()[layer+'.weight'].numpy(), n)[0]
                assert np.array_equal(dec.state_dict()[layer+'.weight'].numpy(),expected)
    # Model-free random actions MUST be identical across style labels in a fixed world.
    # This is a negative-control guard for family-conditioned morphology claims.
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as _td:
        (Path(_td)/'gallery').mkdir()
        controls = [rollout(None, 'random', family, 503, Path(_td))[1] for family in FAMILIES]
        assert all([x['chosen'] for x in controls[0]] == [x['chosen'] for x in other] for other in controls[1:])
    tr,h = collect(range(10,13),42)
    assert len(tr[1]) > 10 and h
    print('QL-001 self-test PASS: host legality, quant packed roundtrip, synthetic train examples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',default='quant-lab-out')
    parser.add_argument('--steps',type=int,default=450)
    parser.add_argument('--repair-steps',type=int,default=100)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run(args)
