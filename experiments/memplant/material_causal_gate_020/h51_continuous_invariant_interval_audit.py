#!/usr/bin/env python3
"""Posthoc H51: enumerate ALL 1-D ReLU kinks for invariant H40 worlds.

Requires extracted original H50-EVIDENCE directory as first positional argument.
Uses genuine trained donor endpoint scalar costs and frozen 209-parameter
receiver checkpoints. Never scores synthetic divergent oracle labels.
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import torch


def knots(fee, state, lo, hi):
    w,b,v,d,*_=state
    a=w[:,2]; q=b+w[:,0]+fee*(w[:,1]+w[:,2])
    boundaries={lo,hi}
    boundaries.update(-q[i]/a[i] for i in range(len(a)) if a[i] and lo < -q[i]/a[i] < hi)
    boundaries=sorted(boundaries)
    second=[]
    for left,right in zip(boundaries,boundaries[1:]):
        mask=a*((left+right)/2)+q>0
        aa=v[:,mask]@a[mask]; bb=d+v[:,mask]@q[mask]
        second.extend(-bb[i]/aa[i] for i in range(len(aa)) if aa[i] and left < -bb[i]/aa[i] < right)
    return set(boundaries+second)


def route_margins(world, costs, state):
    w,b,v,d,u,z=state
    totals=np.zeros((2,len(costs)))
    for route in range(2):
        for tile in world[route]:
            wet,fee=tile[:2]
            value=(costs+fee) if wet>.5 else np.full(len(costs),1.+fee)
            inp=np.column_stack((np.full(len(costs),wet),np.full(len(costs),fee),value))
            h=np.maximum(0,inp@w.T+b)
            h=np.maximum(0,h@v.T+d)
            totals[route]-=(h@u.T+z).ravel()
    return totals[1]-totals[0]


def main(root, output):
    # Load upstream exact-byte controls and all historical checkpoint digests.
    helper_path=root/'h50_independent_receiver_precision_test.py'
    spec=importlib.util.spec_from_file_location('original_h50',helper_path)
    h50=importlib.util.module_from_spec(spec);spec.loader.exec_module(h50)
    receipts=root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl'
    assert h50.sha(receipts)==h50.H40_RECEIPT_SHA
    rows=[json.loads(line) for line in receipts.open()]
    assert len(rows)==4800
    profiles=json.loads((root/'h47-output/h47_results.json').read_text())['raw_genuine_donor_endpoint_costs']
    assert len(profiles)==4
    lower=min(v[0] for v in profiles.values());upper=max(v[1] for v in profiles.values())
    result={'classification':'EXPOSED_ONLY_NOT_PROSPECTIVE','counts':{},'witnesses':[]}
    for kind,checkpoints in h50.RECEIVERS.items():
        for seed,file in checkpoints.items():
            cp=root/file
            assert h50.sha(cp)==h50.CHECKPOINT_SHA[kind][seed]
            weights=torch.load(cp,map_location='cpu',weights_only=True)
            assert sum(t.numel() for t in weights.values())==209
            params=tuple(weights[f'tile.{i}.{attr}'].numpy().astype('f8') for i,attr in ((0,'weight'),(0,'bias'),(2,'weight'),(2,'bias'),(4,'weight'),(4,'bias')))
            worlds=[r for r in rows if str(r['seed'])==seed and not r['bucket'].endswith('divergent')]
            assert len(worlds)==960
            for r in worlds:
                assert len(set(r['oracle_actions']))==1
                target=r['oracle_actions'][0]
                world=np.asarray(r['world_observation'],dtype='f4').astype('f8')
                full={lower,upper}
                for tile in world.reshape(-1,4):
                    if tile[0]>.5:
                        full.update(knots(float(tile[1]),params,lower,upper))
                full=np.asarray(sorted(full))
                for source,(a,b) in sorted(profiles.items()):
                    key=f'{kind}/{seed}/{source}'
                    costs=np.concatenate(([a],full[(full>a)&(full<b)],[b]))
                    signed=route_margins(world,costs,params)*(1 if target else -1)
                    # Signed margin is affine on each true ReLU activation segment.
                    bad=(signed[:-1]<-1e-10)|(signed[1:]<-1e-10)
                    result['counts'].setdefault(key,0)
                    if bad.any():
                        result['counts'][key]+=1
                        idx=int(np.argmin(signed))
                        result['witnesses'].append({'condition':kind,'seed':seed,'profile':source,
                            'sha256':r['observation_sha256'],'index':r['index'],
                            'world_seed':r['world_seed'],'bucket':r['bucket'],
                            'cost':float(costs[idx]),'signed_margin':float(signed[idx]),
                            'expected_action':target})
    result['totals']={kind:sum(v for k,v in result['counts'].items() if k.startswith(kind+'/')) for kind in h50.RECEIVERS}
    assert result['totals']=={'h40':568,'h45':20},result['totals']
    assert len(result['witnesses'])==588
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'totals':result['totals'],'witnesses':len(result['witnesses'])}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('evidence_root',type=Path);ap.add_argument('--output',type=Path,default=Path('h51_replay.json'))
    args=ap.parse_args();main(args.evidence_root,args.output)
