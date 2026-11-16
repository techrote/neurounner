#!/usr/bin/env python3
"""QL-003: real low-bit × sparse/franken game-policy crossing.

Research-only CPU synthetic CyberSand-like world; NOT native game W0, GGUF IQ,
accelerated 1-bit kernels, or a production provider. Host legality always wins.
No STOP suppression. Original first-pass PNGs retained in immutable CI artifacts.
"""
import argparse
import copy
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from quant_worm import (ACTIONS, FAMILIES, WormWorld, Policy, quant_numpy,
                        save_packed, load_packed, sparse_channels, rollout,
                        timing, digest)
from ql002_branch_retrace import examples, trained_model, balanced_qat, assess


def state_hash(net):
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode('ascii') + b'\0')
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def carve_b(net, fraction):
    """Zero low-magnitude b-neurons and connected c input channels."""
    out = copy.deepcopy(net)
    with torch.no_grad():
        strengths = out.b.weight.abs().sum(dim=1)
        selected = torch.argsort(strengths)[:round(len(strengths)*fraction)]
        out.b.weight[selected, :] = 0
        out.b.bias[selected] = 0
        out.c.weight[:, selected] = 0
    return out


def frank(left, right, selected):
    """Exchange whole corresponding layers from same-initialization siblings."""
    result = copy.deepcopy(left)
    with torch.no_grad():
        for letter in selected:
            getattr(result, letter).load_state_dict(
                getattr(right, letter).state_dict())
    return result


class OpeningPrior(torch.nn.Module):
    """Frozen initial-world score vector; SAME model and host, no live state updates."""
    def __init__(self, net, seed, family):
        super().__init__()
        initial = WormWorld(seed, family)
        with torch.no_grad():
            self.register_buffer('scores', net(torch.from_numpy(
                initial.features(0)).unsqueeze(0))[0].detach().clone())

    def forward(self, x):
        return self.scores.unsqueeze(0).expand(x.shape[0], -1)


def morphology(steps):
    points = [tuple(s['position']) for s in steps]
    if steps:
        points.append(tuple(steps[-1]['next_position']))
    distinct = len(set(points))
    if points:
        xs, ys = zip(*points)
        bbox_area = (max(xs)-min(xs)+1)*(max(ys)-min(ys)+1)
    else:
        bbox_area = 0
    actions = [s['chosen'] for s in steps]
    max_streak, current = 0, 0
    for i, action in enumerate(actions):
        current = current+1 if i and action == actions[i-1] else 1
        max_streak = max(max_streak, current)
    return {'visited_positions':distinct, 'path_positions':len(points),
            'repeat_fraction':round(1-distinct/max(1,len(points)),5),
            'bounding_box_area':bbox_area, 'max_action_streak':max_streak,
            'stopped':bool(actions and actions[-1]=='STOP'),
            'action_counts':dict(Counter(actions))}


def variants(regular, balanced, repair):
    return {
        'bal-fp32':(balanced,(32,32,32)),
        'bal-q4':(balanced,(4,4,4)),
        'bal-q3':(balanced,(3,3,3)),
        'bal-q2':(balanced,(2,2,2)),
        'bal-q1':(balanced,(1,1,1)),
        'bal-q1a-q4bc':(balanced,(1,4,4)),
        'bal-q4a-q1b-q4c':(balanced,(4,1,4)),
        'bal-q4ab-q1c':(balanced,(4,4,1)),
        'bal-q2a-q4bc':(balanced,(2,4,4)),
        'bal-q4a-q2b-q4c':(balanced,(4,2,4)),
        'bal-a-carve30-q1':(sparse_channels(balanced,.30),(1,1,1)),
        'bal-b-carve30-q1':(carve_b(balanced,.30),(1,1,1)),
        'frank-bal-a-reg-bc-q1':(frank(balanced,regular,'bc'),(1,1,1)),
        'frank-bal-ab-reg-c-q2':(frank(balanced,regular,'c'),(2,2,2)),
        'frank-reg-a-bal-bc-q1':(frank(regular,balanced,'bc'),(1,1,1)),
        'bal-q1-qat40':(repair,(1,1,1)),
    }


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(65112)
    np.random.seed(65112)
    root = Path(args.out)
    for subdir in ('models','gallery'):
        (root/subdir).mkdir(parents=True, exist_ok=True)
    train, train_hist, train_illegal = examples(range(2100,2180),33102)
    held, held_hist, held_illegal = examples(range(9100,9132),33103)
    assert train_illegal == held_illegal == 0
    assert all(held_hist.get(a,0)>0 for a in ACTIONS)
    regular, regular_opt, regular_diag = trained_model(train,args.steps,65112,False)
    balanced, balanced_opt, balanced_diag = trained_model(train,args.steps,65112,True)
    repair, repair_opt, repair_diag = balanced_qat(balanced,train,args.q1_repair_steps,6427,bits=1)
    training = {'regular':regular_diag, 'balanced':balanced_diag,
                'q1-repair':repair_diag}
    for name,net,opt,diag in (
        ('regular',regular,regular_opt,regular_diag),
        ('balanced',balanced,balanced_opt,balanced_diag),
        ('q1-repair',repair,repair_opt,repair_diag)):
        torch.save({'state_dict':net.state_dict(),'optimizer':opt,'training':diag,
                    'source_sha':os.environ.get('GITHUB_SHA','local'),
                    'weights_sha256':state_hash(net)},
                   root/'models'/(name+'-optimizer.pt'))

    family_seeds=(9103,9111,9127)
    baselines={}
    raw=[]
    for control in ('random','wall','fidget'):
        episodes=[]
        for seed in family_seeds:
            for family in FAMILIES:
                e,steps=rollout(None,control,family,seed,root,horizon=64)
                e['morphology']=morphology(steps)
                episodes.append(e)
                raw.append({'variant':control,'mode':'model-free',
                            'seed':seed,'family':family,'steps':steps})
        baselines[control]=episodes

    reports={}
    reference=None
    for name,(original,bits) in variants(regular,balanced,repair).items():
        path,shapes,payload=save_packed(root/'models',name,original,bits)
        loaded=load_packed(path,shapes)
        for layer,b in zip('abc',bits):
            expected=quant_numpy(original.state_dict()[layer+'.weight'].numpy(),b)[0]
            actual=loaded.state_dict()[layer+'.weight'].numpy()
            assert np.array_equal(expected,actual),(name,layer)
        held_report=assess(loaded,held)
        with torch.no_grad():
            x,y,legal=held
            logits=loaded(x)
            masked=logits.masked_fill(~legal,-1e9).argmax(dim=1)
            if name=='bal-fp32':
                reference=masked.clone()
            assert reference is not None
            diffs=int((masked!=reference).sum())
        episodes=[]
        modes={}
        for mode in ('live','opening-prior'):
            selected=[]
            for seed in family_seeds:
                for family in FAMILIES:
                    policy=name if mode=='live' else name+'-opening-prior'
                    model=loaded if mode=='live' else OpeningPrior(loaded,seed,family)
                    e,steps=rollout(model,policy,family,seed,root,horizon=64)
                    e['morphology']=morphology(steps)
                    e['mode']=mode
                    episodes.append(e)
                    selected.append(e)
                    raw.append({'variant':name,'mode':mode,'seed':seed,
                                'family':family,'steps':steps})
            modes[mode]={
                'mean_unique_cells':round(statistics.mean(e['unique_cells'] for e in selected),4),
                'mean_turns':round(statistics.mean(e['turns'] for e in selected),4),
                'stopped_episodes':sum(e['morphology']['stopped'] for e in selected),
                'mean_bbox_area':round(statistics.mean(e['morphology']['bounding_box_area'] for e in selected),4),
                'mean_repeat_fraction':round(statistics.mean(e['morphology']['repeat_fraction'] for e in selected),5),
                'action_histogram':dict(sum((Counter(e['action_histogram']) for e in selected),Counter()))}
        reports[name]={
            **held_report,'bits':list(bits),'source_state_hash':state_hash(original),
            'loaded_state_hash':state_hash(loaded),
            'archive_sha256':digest(path),'packed_payload_bytes':payload,
            'npz_archive_bytes':path.stat().st_size,
            'same_state_masked_disagreement_vs_bal_fp32_n':diffs,
            'same_state_masked_disagreement_vs_bal_fp32_rate':round(diffs/len(held[1]),6),
            'warm_dequant_fp32_cpu':timing(loaded,held[0],150),
            'modes':modes,'visual_episodes':episodes}
    (root/'raw_decisions.json').write_text(json.dumps(raw,separators=(',',':'))+'\n')
    results={
        'schema':'ql003-franken-lowbit-1',
        'source_sha':os.environ.get('GITHUB_SHA','local'),
        'kind':'synthetic CyberSand-style local-action toy; NOT W0 native/Godot, not llama.cpp IQ',
        'experiment':'same-initialization trained sibling Franken layers crossed with per-layer scalar Q1/Q2/Q3/Q4 and linked-neuron carving',
        'training':training, 'training_state_sha256':{
            'regular':state_hash(regular),'balanced':state_hash(balanced),
            'q1-repair':state_hash(repair)},
        'train_n':len(train[1]),'heldout_n':len(held[1]),
        'train_hist':train_hist,'heldout_hist':held_hist,
        'world_seeds':{'train':'2100..2179','heldout':'9100..9131','visual':list(family_seeds)},
        'variants':reports,'controls':baselines,
        'raw_decisions_sha256':digest(root/'raw_decisions.json'),
        'qualification':'All first-pass actions apply through host legality; free STOP from tick 0; score receipts and original source-faithful PNG',
        'causal_caveat':'Matched held-out action contexts use identical observations/legal masks; live versus opening-prior rollouts diverge after first differing action, not same-state causal comparisons.',
        'speed_caveat':'Real bitpacked payloads decoded to FP32 CPU for inference. RSS/latency do not measure low-bit kernels.',
        'rss_kib_process_wide':int(next((s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')),0)) if Path('/proc/self/status').exists() else None
    }
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    inventory=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p)}
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']
    (root/'MANIFEST.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','sha':results['source_sha'],
       'train_n':len(train[1]),'heldout_n':len(held[1]),
       'training':training,'variant_count':len(reports),'original_png_count':len(list((root/'gallery').glob('*.png'))),
       'variants':{k:{'agreement_change':v['same_state_masked_disagreement_vs_bal_fp32_rate'],
           'branch_recall':v['heldout_actions']['BRANCH']['unmasked_recall'],
           'retrace_recall':v['heldout_actions']['RETRACE']['unmasked_recall'],
           'stop_recall':v['heldout_actions']['STOP']['unmasked_recall'],
           'live_cells':v['modes']['live']['mean_unique_cells'],
           'frozen_cells':v['modes']['opening-prior']['mean_unique_cells'],
           'bytes':v['packed_payload_bytes']} for k,v in reports.items()}},
       sort_keys=True))


def self_test():
    torch.manual_seed(21)
    a,b=Policy(),Policy()
    with torch.no_grad():
        a.a.weight.fill_(1)
        b.a.weight.fill_(-1)
    f=frank(a,b,'a')
    assert torch.all(f.a.weight==-1)
    assert not torch.all(a.a.weight==-1)
    s=carve_b(a,.30)
    assert (s.b.weight.abs().sum(dim=1)==0).sum().item()==round(64*.30)
    assert (s.c.weight.abs().sum(dim=0)==0).sum().item()>=round(64*.30)
    sample=WormWorld(9103,'GREEBLES')
    prior=OpeningPrior(a,9103,'GREEBLES')
    z=torch.from_numpy(sample.features(0))[None,:]
    with torch.no_grad():
        assert torch.allclose(prior(z),a(z))
        assert torch.allclose(prior(z),prior(torch.zeros_like(z)))
    assert len(variants(a,b,a))>=12
    print('QL003 self-test PASS: surgery, donor immutability, frozen-opening-logits')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='ql003-out')
    p.add_argument('--steps',type=int,default=350)
    p.add_argument('--q1-repair-steps',type=int,default=40)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    self_test() if args.self_test else run(args)
