#!/usr/bin/env python3
"""QL-002: bounded, game-legal BRANCH/RETRACE supervision under extreme scalar quant.

Independent *synthetic* research, not CyberSand native W0, not GGUF IQ codecs.
The same explicit game-state curriculum is used for both training baselines.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from quant_worm import (ACTIONS, FAMILIES, WormWorld, Policy, quant_numpy,
                        save_packed, load_packed, sparse_channels, rollout, timing,
                        digest)


def ql2_teacher(w, tick):
    legal = w.legal()
    if tick >= 44 or len(w.ink) >= 55:
        return 'STOP'
    # First-class route recovery: explicit legal local dead ends and short deliberate
    # backward excursions. No RETRACE label is invented for an illegal state.
    fresh = [a for d, a in ((0, 'STRAIGHT'), (-1, 'LEFT'), (1, 'RIGHT'))
             if a in legal and w.state(w.neighbor(d)) == 1]
    family_phase = FAMILIES.index(w.family) * 3
    if 'RETRACE' in legal and (not fresh or
          (len(w.path) >= 6 and (tick + family_phase) % 13 in (0, 1))):
        return 'RETRACE'
    if 'BRANCH' in legal and ((tick + family_phase) % 7 in (2, 4)):
        return 'BRANCH'
    preferred = {
        'GREEBLES': ('LEFT', 'STRAIGHT', 'RIGHT'),
        'PIPEWORK': ('STRAIGHT', 'RIGHT', 'LEFT'),
        'ARCHITECTURE': ('RIGHT', 'STRAIGHT', 'LEFT')
    }[w.family]
    if tick % 9 in (2, 3, 4):
        preferred = ('STRAIGHT',) + tuple(a for a in preferred if a != 'STRAIGHT')
    for a in preferred:
        if a in fresh:
            return a
    for a in preferred:
        if a in legal:
            return a
    return 'RETRACE' if 'RETRACE' in legal else 'STOP'


def examples(seeds, exploration_seed):
    rng = random.Random(exploration_seed)
    features, labels, legals = [], [], []
    hist, invalid = Counter(), 0
    for seed in seeds:
        for family in FAMILIES:
            w = WormWorld(seed, family)
            for tick in range(48):
                legal = w.legal()
                label = ql2_teacher(w, tick)
                invalid += int(label not in legal)
                assert label in legal
                features.append(w.features(tick))
                labels.append(ACTIONS.index(label))
                legals.append([int(a in legal) for a in ACTIONS])
                hist[label] += 1
                # Curriculum explores recoverable paths. Unsupervised random STOP
                # is deliberately excluded while *teacher* STOP remains legal.
                explore = [a for a in legal if a != 'STOP']
                chosen = rng.choice(explore) if explore and rng.random() < .23 else label
                w.apply(chosen)
                if w.stopped:
                    break
    x = torch.from_numpy(np.stack(features))
    y = torch.as_tensor(labels, dtype=torch.long)
    masks = torch.tensor(legals, dtype=torch.bool)
    return (x, y, masks), dict(hist), invalid


def trained_model(data, steps, seed, balanced):
    torch.manual_seed(seed)
    model = Policy()
    optim = torch.optim.AdamW(model.parameters(), lr=.0035)
    x, y, _ = data
    counts = torch.bincount(y, minlength=len(ACTIONS))
    if balanced:
        # Exact class balancing for minibatch sampling; no fake RETRACE/BRANCH
        # states are manufactured. All samples come from actual host legal menus.
        weights = (1.0 / counts[y].float()).double()
        weights /= weights.sum()
    losses = []
    for i in range(steps):
        idx = (torch.multinomial(weights, 128, replacement=True)
               if balanced else torch.randint(len(y), (128,)))
        logits = model(x[idx])
        loss = torch.nn.functional.cross_entropy(logits, y[idx])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    diag = {'updates': steps, 'seed':seed, 'balanced_minibatches':balanced,
            'train_histogram':{a:int(counts[i]) for i,a in enumerate(ACTIONS)},
            'loss_first':losses[0], 'loss_last':losses[-1],
            'moment_tensors_nonzero':sum(int(torch.count_nonzero(s['exp_avg'])>0)
                for s in optim.state.values())}
    return model, optim.state_dict(), diag


def balanced_qat(base, data, steps, seed, bits=2):
    torch.manual_seed(seed)
    net = copy.deepcopy(base)
    opt = torch.optim.AdamW(net.parameters(), lr=.0008)
    x, y, _ = data
    counts = torch.bincount(y, minlength=len(ACTIONS))
    weights = (1. / counts[y].float()).double()
    weights /= weights.sum()
    losses = []
    for _ in range(steps):
        ix = torch.multinomial(weights, 128, replacement=True)
        z = x[ix]
        for layer in 'abc':
            mod = getattr(net, layer)
            wt = mod.weight
            quant = torch.from_numpy(quant_numpy(wt.detach().numpy(), bits)[0])
            z = torch.nn.functional.linear(z, wt + (quant - wt).detach(), mod.bias)
            if layer != 'c':
                z = torch.tanh(z)
        loss = torch.nn.functional.cross_entropy(z, y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return net, opt.state_dict(), {'updates':steps, 'bits':bits,
          'balanced_minibatches':True, 'loss_first':losses[0],
          'loss_last':losses[-1], 'moment_tensors_nonzero':sum(
          int(torch.count_nonzero(s['exp_avg'])>0) for s in opt.state.values())}


def assess(net, held):
    x,y,legal = held
    with torch.no_grad():
        logits = net(x)
        top = logits.argmax(dim=1)
        legal_top = legal.gather(1, top[:,None]).flatten()
        masked = logits.masked_fill(~legal, -1e9).argmax(dim=1)
    classes = {}
    for i,name in enumerate(ACTIONS):
        mask = (y == i)
        count = int(mask.sum())
        classes[name] = {'count':count, 'unmasked_recall':
            None if count == 0 else round(float((top[mask] == i).float().mean()),5),
            'host_masked_recall':None if count == 0 else
                 round(float((masked[mask] == i).float().mean()),5)}
    return {'heldout_n':len(y),
        'unmasked_teacher_accuracy':round(float((top == y).float().mean()),5),
        'host_masked_teacher_accuracy':round(float((masked == y).float().mean()),5),
        'unmasked_top_action_legal_rate':round(float(legal_top.float().mean()),5),
        'heldout_actions':classes,
        'predicted_top_counts':{a:int((top == i).sum()) for i,a in enumerate(ACTIONS)}}


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(65112)
    np.random.seed(65112)
    root = Path(args.out)
    for part in ('models','gallery'):
        (root/part).mkdir(parents=True,exist_ok=True)
    train, train_hist, bad_train = examples(range(2100,2180), 33102)
    held, held_hist, bad_held = examples(range(9100,9132), 33103)
    assert not (set(range(2100,2180)) & set(range(9100,9132)))
    assert bad_train == bad_held == 0
    assert all(train_hist.get(a,0) > 0 and held_hist.get(a,0) > 0
               for a in ACTIONS), (train_hist,held_hist)
    regular, r_opt, r_diag = trained_model(train,args.steps,65112,False)
    balanced, b_opt, b_diag = trained_model(train,args.steps,65112,True)
    repaired, q_opt, q_diag = balanced_qat(balanced,train,args.repair_steps,6423)
    for tag, model, opt, desc in (
            ('regular',regular,r_opt,r_diag),('balanced',balanced,b_opt,b_diag),
            ('balanced-q2-qat',repaired,q_opt,q_diag)):
        torch.save({'state_dict':model.state_dict(),'optimizer':opt,
                    'training':desc,'source_sha':os.environ.get('GITHUB_SHA','local')},
                    root/'models'/f'{tag}-optimizer.pt')
    variant_defs = {
        'regular-fp32':(regular,(32,32,32)),
        'regular-q1':(regular,(1,1,1)),
        'balanced-fp32':(balanced,(32,32,32)),
        'balanced-q4':(balanced,(4,4,4)),
        'balanced-q2':(balanced,(2,2,2)),
        'balanced-q1':(balanced,(1,1,1)),
        'balanced-mixed-q1q4q4':(balanced,(1,4,4)),
        'balanced-q2-carve30':(sparse_channels(balanced,.30),(2,2,2)),
        'balanced-q2-qat':(repaired,(2,2,2)),
    }
    records, all_steps = {}, []
    for name,(model,levels) in variant_defs.items():
        archive, shapes, raw_bytes = save_packed(root/'models',name,model,levels)
        loaded = load_packed(archive,shapes)
        # Byte-packed reconstitution is checked independently per layer.
        for layer,bits in zip('abc',levels):
            expected = quant_numpy(model.state_dict()[layer+'.weight'].numpy(),bits)[0]
            actual = loaded.state_dict()[layer+'.weight'].numpy()
            assert np.array_equal(actual,expected),(name,layer)
        evidence = assess(loaded,held)
        episodes = []
        for seed in (9103,9111,9127):
            for family in FAMILIES:
                e, steps = rollout(loaded,name,family,seed,root,horizon=64)
                episodes.append(e)
                all_steps.append({'variant':name,'seed':seed,'family':family,'steps':steps})
        hist = Counter()
        for e in episodes:
            hist.update(e['action_histogram'])
        records[name] = {
            'bits':levels,'packed_payload_bytes':raw_bytes,
            'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive),
            'model_quant_shapes':shapes, **evidence,
            'warm_cpu_same_runner':timing(loaded,held[0],150),
            'visual_episodes':episodes,
            'mean_cells':round(statistics.mean(e['unique_cells'] for e in episodes),3),
            'mean_turns':round(statistics.mean(e['turns'] for e in episodes),3),
            'actions':dict(hist)}
    controls = {}
    for kind in ('random','wall','fidget'):
        runs=[]
        for seed in (9103,9111,9127):
            for family in FAMILIES:
                e, steps=rollout(None,kind,family,seed,root,horizon=64)
                runs.append(e)
                all_steps.append({'variant':kind,'seed':seed,'family':family,'steps':steps})
        controls[kind] = runs
    (root/'raw_decisions.json').write_text(json.dumps(all_steps,separators=(',',':'))+'\n')
    result = {'schema':'ql002-branch-retrace-1','source_sha':os.environ.get('GITHUB_SHA','local'),
      'kind':'synthetic CyberSand-inspired local action toy, NOT native W0/Godot, NOT GGUF IQ',
      'experiment':'equal-step regular vs class-balanced actual host-legal data; quant+Q2 STE repair',
      'seeds':{'train':'2100..2179','heldout':'9100..9131','visual':[9103,9111,9127]},
      'train_n':len(train[1]),'heldout_n':len(held[1]),
      'train_hist':train_hist,'heldout_hist':held_hist,
      'training':{'regular':r_diag,'balanced':b_diag,'q2-qat':q_diag},
      'variants':records,'controls':controls,
      'raw_decisions_sha256':digest(root/'raw_decisions.json'),
      'cpu_note':'packed weights dequantized to FP32; no packed-inference speed claims',
      'art_note':'STOP legal from turn zero; different episode lengths; no human artistic scoring'}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    inventory=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,
                'sha256':digest(p)} for p in sorted(root.rglob('*'))
                if p.is_file() and p.name!='MANIFEST.json']
    (root/'MANIFEST.json').write_text(json.dumps(inventory,indent=2)+'\n')
    summary={'source_sha':result['source_sha'],'train_hist':train_hist,'held_hist':held_hist,
        'samples':[len(train[1]),len(held[1])], 'train':[r_diag,b_diag,q_diag],
        'variants':{k:{'accuracy':v['unmasked_teacher_accuracy'],
        'branch_recall':v['heldout_actions']['BRANCH']['unmasked_recall'],
        'retrace_recall':v['heldout_actions']['RETRACE']['unmasked_recall'],
        'stop_recall':v['heldout_actions']['STOP']['unmasked_recall'],
        'cells':v['mean_cells'],'turns':v['mean_turns'],
        'bytes':v['packed_payload_bytes']} for k,v in records.items()}}
    print(json.dumps(summary,sort_keys=True))


def self_test():
    for seed in (15, 17, 21):
        w=WormWorld(seed,'GREEBLES')
        assert ql2_teacher(w,0) in w.legal()
        for tick in range(44):
            action=ql2_teacher(w,tick)
            assert action in w.legal()
            w.apply(action)
            if w.stopped:break
    data,hist,illegal=examples(range(2200,2205),505)
    assert illegal == 0 and len(data[1]) > 50
    assert hist.get('RETRACE',0) > 0 and hist.get('BRANCH',0) > 0, hist
    assert all(int(data[2][i, data[1][i]]) for i in range(len(data[1])))
    # Distinct world-seed sample counts may differ but eval is fresh-seed only.
    assert set(range(2100,2180)).isdisjoint(range(9100,9132))
    print('QL002 self-test PASS',dict(hist))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='ql002-out')
    ap.add_argument('--steps',type=int,default=350)
    ap.add_argument('--repair-steps',type=int,default=75)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    self_test() if args.self_test else run(args)
