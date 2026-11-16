"""Independent deterministic replay of stored policy probabilities and world effects."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .history_motif_lab import Adapter, PROBES, STYLES, ARMS, VERSION, EVAL_HORIZONS
from .train import MAX_RECOVERIES,sha_file,source_pins
from .world import MOVE,World,digest,specs

def verify_episode(record,model):
    if (record['version']!=VERSION or record['split']!='development' or
        record['style'] not in STYLES or record['horizon'] not in EVAL_HORIZONS or
        record['native_receipts']!=0):raise ValueError('forbidden episode identity')
    possibilities=[s for s in specs('development',1) if s.ident==record['world_id']]
    if len(possibilities)!=1:raise ValueError('non-development fixture')
    spec=possibilities[0]
    if record['switch_to'] is not None and (record['switch_to'] not in STYLES or record['horizon']<128):
        raise ValueError('invalid switch')
    w=World.create(spec)
    if w.base_sha()!=record['base_sha256'] or digest(record['events'])!=record['trajectory_sha256']:
        raise ValueError('forged source or events')
    if len(record['events'])!=len(record['probabilities']):raise ValueError('missing policy decisions')
    counts=Counter()
    for i,(event,p0) in enumerate(zip(record['events'],record['probabilities'])):
        command=record['style'] if record['switch_to'] is None or i<64 else record['switch_to']
        menu,p=model.distribution(w,command)
        chosen=menu[int(np.argmax(p))]
        if (p0['turn']!=i or p0['command']!=command or p0['preferred']!=chosen or
            set(p0['distribution'])!=set(menu) or
            any(abs(p0['distribution'][name]-float(p[j]))>2e-8 for j,name in enumerate(menu)) or
            abs(sum(p0['distribution'].values())-1)>1e-6):
            raise ValueError('forged checkpoint probabilities')
        if record['force'] and i in PROBES:
            alternatives=[a for a in menu if a in MOVE and a!=chosen]
            if alternatives:chosen=alternatives[0]
        if event['chosen']!=chosen or w.apply(chosen,MAX_RECOVERIES)!=event:
            raise ValueError('forged world action/material transition')
        counts[chosen]+=1
    moves=[e['chosen'][-1] for e in w.decisions if e['chosen'] in MOVE]
    segments=[];prev=None;length=0
    for heading in moves:
        if heading==prev:length+=1
        else:
            if length:segments.append(length)
            prev=heading;length=1
    if length:segments.append(length)
    target={'arc':9,'wave':4,'jag':2}[record['style']]
    segment_score=round(sum(abs(v-target)<=1 for v in segments)/max(1,len(segments)),6)
    if (segments!=record['segment_lengths'] or segment_score!=record['segment_target_fraction']):
        raise ValueError('forged motif segments or consistency score')
    if (w.state_sha()!=record['final_sha256'] or w.metrics(record['horizon'])!=record['metrics'] or
        [list(x) for x in sorted(w.ink)]!=record['ink_cells'] or
        dict(sorted(counts.items()))!=record['action_distribution'] or
        any(p in w.materials or p in w.protected for p in w.ink)):
        raise ValueError('geometry/summary falsification')
    return w

def main(root):
    root=Path(root);report=json.loads((root/'report.json').read_text())
    if report['native_executed'] or not report['sealed_heldout_unread'] or not report['train_only']:
        raise ValueError('experiment boundary violated')
    check={**source_pins(),'history_motif_lab.py':sha_file(Path(__file__).with_name('history_motif_lab.py'))}
    if report['source_sha256']!=check:raise ValueError('source pin mismatch')
    manifest=json.loads((root/'SHA256SUMS.json').read_text())
    actual={f.name:sha_file(f) for f in root.iterdir() if f.is_file() and f.name!='SHA256SUMS.json'}
    if actual!=manifest:raise ValueError('changed or missing artifact bytes')
    models={}
    for arm in ARMS:
        cp=root/f'{arm}.npz'
        if sha_file(cp)!=report['checkpoints'][arm]['sha256']:raise ValueError('checkpoint mismatch')
        models[arm]=Adapter.load(cp)
    checked=Counter(); decisions=0
    for file in root.glob('*.json'):
        if file.name in ('report.json','SHA256SUMS.json'):continue
        record=json.loads(file.read_text())
        arm=file.name.split('-')[0]
        if arm not in models:raise ValueError('unknown arm')
        verify_episode(record,models[arm]);checked[arm]+=1;decisions+=len(record['events'])
    if sum(checked.values())!=report['episodes'] or set(checked)!=set(ARMS):
        raise ValueError('missing or duplicate episodes')
    return {'verified_episodes':dict(checked),'verified_decisions':decisions,
            'files_sha256_verified':len(actual),'native_receipts':0}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args()
    print(json.dumps(main(args.root),sort_keys=True))
