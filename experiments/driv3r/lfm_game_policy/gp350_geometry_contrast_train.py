#!/usr/bin/env python3
"""GP350-005: fresh LFM350 LoRA on matched forward-ray geometry contrasts.

Original W0 observed states supply positions, headings, legal menus and context.
Three different TWO-CELL-DISTANCE forward-ray markers are explicitly SYNTHETIC:
free '.', obstacle '#', visited '+'. All have unchanged legal actions and
immediate-neighbour pixels; this is not native evidence or human labelling.
The reused GP350-003 optimizer is identical, but the curriculum and receipts
are independently versioned; this is a fresh parent, NOT a checkpoint resume.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import lfm350 as c
import gp350_triad as t

SCHEMA='driv3r.lfm350.gp350005-ray-contrast/1'
TRAIN_SEEDS=tuple(range(3101,3113))
DEV_SEEDS=(9103,9119,9137)
RAY_MARKERS=('.', '#', '+')
TARGETS={
    'greebles':('BRANCH','LEFT','RIGHT'),
    'pipework':('STRAIGHT','RIGHT','LEFT'),
    'architecture':('STRAIGHT','LEFT','RIGHT'),
}

def ray_variant(base, marker):
    # The immediate neighbours and host legal menu remain exactly unchanged.
    dx,dy=((1,0),(0,1),(-1,0),(0,-1))[base['heading']%4]
    y,x=2+2*dy,2+2*dx
    lines=[list(line) for line in base['local_5x5']]
    lines[y][x]=marker
    variant=dict(base,local_5x5=[''.join(line) for line in lines])
    assert variant['legal']==base['legal']
    assert all(variant[k]==base[k] for k in base if k!='local_5x5')
    assert all(variant['local_5x5'][2+dy][2+dx]==base['local_5x5'][2+dy][2+dx]
               for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)))
    return variant

def geometry_triples(w0, seeds):
    triples=[]
    seen=set()
    for seed in seeds:
        for family in c.FAMILIES:
            for item in c.collect_world(w0,seed,family,steps=18):
                base=item['observation']
                if c.nearby(base,0)!='.':
                    continue
                desired=TARGETS[family]
                if not set(desired).issubset(set(base['legal'])):
                    continue
                # De-duplicate identical observed prompts, without mixing world splits.
                key=(seed,c.canonical(base))
                if key in seen:
                    continue
                seen.add(key)
                # Every seventh triple is a non-contrast control, preserving
                # the GP350-003 optimizer's original 3:1 sampling strata.
                noncontrast=(len(triples)%7==0)
                trio=[]
                for marker,action in zip(RAY_MARKERS,desired):
                    obs=ray_variant(base,marker)
                    label='STRAIGHT' if noncontrast else action
                    assert label in obs['legal']
                    trio.append({'seed':seed,'observation':obs,'target':label,
                        'synthetic':True,'W0_observed_state':obs==base,
                        'W0_observed_parent':True,
                        'synthetic_forward_ray_marker':marker,
                        'synthetic_label_not_native':True})
                assert len({a['target'] for a in trio}) in (1,3)
                assert len({tuple(a['observation']['legal']) for a in trio})==1
                triples.append(trio)
    return triples

_original_probe=t.probe

def geometry_probe(model,tok,ids,triples,n=18):
    p=_original_probe(model,tok,ids,triples,n)
    p['geometry_contrast_triples']=sum(len({x['target'] for x in r['choices']})==3 for r in p['cases'])
    p['three_distinct_legal_choices']=sum(len({x['legal'] for x in r['choices']})==3 for r in p['cases'])
    p['all_geometry_labels_matched']=sum(all(x['match'] for x in r['choices']) for r in p['cases']
                                        if len({x['target'] for x in r['choices']})==3)
    p['heldout_geometry_is_synthetic']=True
    return p

def self_test(w0):
    tr=geometry_triples(w0, TRAIN_SEEDS)
    dev=geometry_triples(w0, DEV_SEEDS)
    assert len(tr)>=80 and len(dev)>=20,(len(tr),len(dev))
    assert {r[0]['seed'] for r in tr}.isdisjoint({r[0]['seed'] for r in dev})
    assert sum(len({x['target'] for x in r})==3 for r in tr)>=24
    assert any(len({x['target'] for x in r})==1 for r in tr)
    for r in tr+dev:
        assert {x['observation']['family'] for x in r} in [set([f]) for f in c.FAMILIES]
        for x in r:
            assert x['target'] in x['observation']['legal']
    print(c.canonical({'GP350005':'PRETRAIN_PASS','train_triples':len(tr),
        'dev_triples':len(dev),'train_contrasts':sum(len({x['target'] for x in r})==3 for r in tr),
        'synthetic_geometry':True}))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--w0',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--deadline',type=int,default=1600)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    t.SCHEMA=SCHEMA
    t.STEPS=72
    t.make_triples=geometry_triples
    t.probe=geometry_probe
    if a.self_test:
        self_test(c.load_w0(a.w0))
    else:
        self_test(c.load_w0(a.w0))
        t.run(a)

if __name__=='__main__':
    main()
