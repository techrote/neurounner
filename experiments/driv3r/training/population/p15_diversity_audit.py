"""Independent P15 checkpoint replay and conditioned trajectory diversity audit.

No retraining. Re-executes every heldout four-decision game policy mission
from saved weights, independent of the P15 training summary's rollout loop.
"""
from __future__ import annotations
import itertools
import json
import statistics
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from population_lab import NICHES
from population_p14 import mission_pair,choose
from population_p15 import MODES,build_model


def audit(root):
    root=Path(root)
    records=[]
    torch.set_num_threads(1)
    for cohort in sorted(root.glob('cohort_*')):
        training=json.loads((cohort/'results.json').read_text())
        models={}
        for mode in MODES:
            checkpoint=torch.load(cohort/'models'/f'{mode}.pt',map_location='cpu',weights_only=True)
            model=build_model(mode,checkpoint['width'])
            model.load_state_dict(checkpoint['state_dict']);model.eval();models[mode]=model
        for mode,model in models.items():
            by_context={}; successes=0; actions=0
            for seed in training['splits']['holdout']:
                for card in (0,1):
                    for side in ('E','W'):
                        for order_id in (0,1):
                            key=(seed,card,side,order_id)
                            traces=[]
                            for niche in NICHES:
                                episode=mission_pair(seed,card,9,'heldout_c',niche,'heldout_far',side)[order_id]
                                w=deepcopy(episode['initial']); path=[]
                                sham=torch.Generator().manual_seed(24117)
                                for tick in range(4):
                                    if w.stopped:break
                                    a=choose(model,mode,w,niche,list(w.history),sham)
                                    assert a in w.legal()
                                    w.step(a);path.append(a);actions+=1
                                successes+=tuple(path)==episode['desired']
                                assert not (w.painted & (w.obstacles|w.protected))
                                traces.append(tuple(path))
                            by_context[key]=traces
            differences=[];unique=[]
            for traces in by_context.values():
                unique.append(len(set(traces)))
                # Respect early stopping: pad missing decisions with sentinel.
                for a,b in itertools.combinations(traces,2):
                    differences.append(sum((a[i] if i<len(a) else 'PAD') !=
                                           (b[i] if i<len(b) else 'PAD') for i in range(4))/4)
            expected=training['models'][mode]['closed_loop']
            assert expected['episodes']==len(by_context)*len(NICHES)
            assert expected['actions']==actions
            assert round(successes/expected['episodes'],6)==expected['mission_success']
            records.append(dict(seed=training['seed'],mode=mode,
                    episodes=expected['episodes'],actions=actions,successes=successes,
                    legal_rate=1.,contexts=len(by_context),
                    mean_unique_niche_trajectories=round(statistics.mean(unique),6),
                    niche_hamming=round(statistics.mean(differences),6),
                    unique_distribution=dict(Counter(unique))))
    aggregates={}
    for mode in MODES:
        rows=[r for r in records if r['mode']==mode]
        aggregates[mode]=dict(niche_hamming=round(statistics.mean(r['niche_hamming'] for r in rows),6),
            mean_unique=round(statistics.mean(r['mean_unique_niche_trajectories'] for r in rows),6),
            episodes=sum(r['episodes'] for r in rows),actions=sum(r['actions'] for r in rows))
    return dict(source='P15 saved-checkpoint independent replay',native=False,
                metric='normalized Hamming of 4-action trajectories across six niche-condition labels on fixed world/history/goal',
                note='Niche targets intentionally differ by parity. This is not artistic-behavior diversity.',
                cohorts=records,aggregate=aggregates,
                total_episodes=sum(r['episodes'] for r in records),
                total_actions=sum(r['actions'] for r in records))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('out')
    args=p.parse_args(); d=audit(args.folder)
    Path(args.out).write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(dict(aggregate=d['aggregate'],episodes=d['total_episodes'],actions=d['total_actions']),indent=2))
