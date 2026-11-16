"""GAL-010b: fixed-weight material sensor knockout on 40 new world seeds.

No training, teacher or goal lookup at inference. Independent host receipt replay
and matched common-random-stream comparisons. External synthetic host only.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import statistics
from material_corridors import (ARMS,KINDS,Model,CorridorWorld,episode,sha,digest,
                                summarise,TRAIN_BASE)

SEED=1230000
N_DEFAULT=40
PANELS=('material_trained_visible','material_trained_occluded','occluded_trained_occluded')

def evaluate(checkpoints,out,n=N_DEFAULT):
    if type(n)!=int or not 5<=n<=64:raise ValueError('bounded seed count')
    checkpoints=Path(checkpoints);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows={};traces={};hashes={}
    for arm in ARMS:
        models={}
        for train_view in ('material','occluded'):
            f=checkpoints/f'{arm}_{train_view}.json';cp=json.loads(f.read_text())
            if cp['arm']!=arm or cp['view']!=train_view:raise ValueError('mislabelled checkpoint')
            model=Model();model.weights=cp['weights'];models[train_view]=model
            hashes[f'{arm}:{train_view}']=digest(f)
        for kind in KINDS:
            for panel in PANELS:
                view='material' if panel=='material_trained_visible' else 'occluded'
                model=models['occluded' if panel=='occluded_trained_occluded' else 'material']
                key=f'{arm}:{kind}:{panel}'
                rows[key]=[];traces[key]=[]
                for j in range(n):
                    w,r=episode(model,arm,view,kind,SEED+j)
                    rows[key].append(r);traces[key].append(w.trace)
    (out/'episodes.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (out/'receipts.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    summary={k:summarise(v) for k,v in rows.items()}
    paired={}
    for arm in ARMS:
        for kind in KINDS:
            a=rows[f'{arm}:{kind}:material_trained_visible']
            b=rows[f'{arm}:{kind}:material_trained_occluded']
            c=rows[f'{arm}:{kind}:occluded_trained_occluded']
            for label,control in [('fixed_weight_knockout',b),('retrained_occluded',c)]:
                if any((x['seed'],x['world_sha256'])!=(y['seed'],y['world_sha256']) for x,y in zip(a,control)):
                    raise AssertionError('world custody mismatch')
                paired[f'{arm}:{kind}:{label}']={
                    'mean_coverage_delta_visible_minus_control':round(statistics.mean(x['coverage']-y['coverage'] for x,y in zip(a,control)),4),
                    'visible_only_completes':sum(x['complete'] and not y['complete'] for x,y in zip(a,control)),
                    'control_only_completes':sum(y['complete'] and not x['complete'] for x,y in zip(a,control)),
                    'changed_action_trajectories':sum([t['action_id'] for t in traces[f'{arm}:{kind}:material_trained_visible'][i]] !=
                        [t['action_id'] for t in traces[f'{arm}:{kind}:{"material_trained_occluded" if label=="fixed_weight_knockout" else "occluded_trained_occluded"}'][i]] for i in range(n))}
    m={'schema':'gal-010b-fixed-weight-knockout/1','seeds':[SEED,SEED+n-1],
       'checkpoint_sha256':hashes,'source_sha256':{f:digest(Path(__file__).with_name(f)) for f in ('lab.py','material_corridors.py','sensor_knockout.py')},
       'episodes_sha256':digest(out/'episodes.json'),'receipts_sha256':digest(out/'receipts.json'),
       'summary':summary,'paired':paired,'native_receipts':0,'resident_slms':0}
    (out/'metrics.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
    return m

def audit(checkpoints,out):
    out=Path(out);m=json.loads((out/'metrics.json').read_text());root=Path(checkpoints)
    if m['source_sha256']!={f:digest(Path(__file__).with_name(f)) for f in ('lab.py','material_corridors.py','sensor_knockout.py')}:
        raise ValueError('source drift')
    for k,h in m['checkpoint_sha256'].items():
        arm,view=k.split(':')
        if digest(root/f'{arm}_{view}.json')!=h:raise ValueError('checkpoint drift')
    for s in (out/'SHA256SUMS').read_text().splitlines():
        h,name=s.split('  ',1)
        if name not in ('episodes.json','receipts.json','metrics.json') or digest(out/name)!=h:raise ValueError('artifact drift')
    ep=json.loads((out/'episodes.json').read_text());traces=json.loads((out/'receipts.json').read_text())
    count=0
    for key,rs in ep.items():
        arm,kind,panel=key.split(':');view='material' if panel=='material_trained_visible' else 'occluded'
        for row,trace in zip(rs,traces[key]):
            seed=row['seed'];w=CorridorWorld(seed,kind,arm,f'gal010-eval:{arm}:{view}:{kind}:{seed}')
            for t in trace:
                offered,token=w.offer()
                if t['action_id'] not in offered or t['offered_sha256']!=token:raise ValueError('illegal menu')
                if sha(w.apply(t['action_id'],token,original_response=t['original_response']))!=sha(t):
                    raise ValueError('changed receipt')
                count+=1
            if sha(w.trace)!=row['receipts_sha256'] or sha(w.material)!=row['material_sha256']:
                raise ValueError('changed material')
    return {'episodes':sum(len(v) for v in ep.values()),'verified_host_receipts':count}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',required=True);p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=N_DEFAULT);a=p.parse_args()
    print(json.dumps(evaluate(a.checkpoints,a.out,a.n)['paired'],indent=2))
