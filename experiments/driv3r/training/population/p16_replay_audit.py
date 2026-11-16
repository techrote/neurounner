"""Independent checkpoint replay for P16: full post-survey model-driven missions.

Kept separate from population_p16.run() to cross-check saved checkpoint
weights, world seeds, legality, protection and reported mission totals.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,statistics
from collections import Counter
from copy import deepcopy
from pathlib import Path
import torch
from population_lab import NICHES
from population_p14 import mission_pair,choose
from population_p16 import MODES,build_model


def audit(root):
    root=Path(root)
    torch.set_num_threads(1)
    results=[]
    for cohort in sorted(root.glob('cohort_*')):
        meta=json.loads((cohort/'results.json').read_text())
        for mode in MODES:
            path=cohort/'models'/f'{mode}.pt'
            stored=torch.load(path,map_location='cpu',weights_only=True)
            assert stored['seed']==meta['seed'] and stored['mode']==mode
            assert stored['source']=='synthetic-P0' and not stored['native']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['models'][mode]['training']['checkpoint_sha256']
            model=build_model(mode,stored['width'])
            model.load_state_dict(stored['state_dict'])
            model.eval()
            contexts={};successes=0;actions=0;legal=0;protected_hits=0
            for seed in meta['splits']['holdout']:
                for card in (0,1):
                    for goal in ('E','W'):
                        for order_id in (0,1):
                            paths=[]
                            for niche in NICHES:
                                ep=mission_pair(seed,card,9,'heldout_c',niche,'heldout_far',goal)[order_id]
                                w=deepcopy(ep['initial']); actual=[]
                                sham=torch.Generator().manual_seed(24117)
                                for step in range(4):
                                    if w.stopped: break
                                    a=choose(model,mode,w,niche,list(w.history),sham)
                                    assert a in w.legal()
                                    actual.append(a);w.step(a);actions+=1;legal+=1
                                protected_hits+=len(w.painted & (w.protected | w.obstacles))
                                successes+=tuple(actual)==ep['desired']
                                paths.append(tuple(actual))
                            contexts[(seed,card,goal,order_id)]=paths
            distances=[];unique=[]
            for paths in contexts.values():
                unique.append(len(set(paths)))
                for a,b in itertools.combinations(paths,2):
                    distances.append(sum((a[i] if i<len(a) else 'PAD') !=
                                         (b[i] if i<len(b) else 'PAD') for i in range(4))/4)
            original=meta['models'][mode]['mission']
            assert original['episodes']==len(contexts)*len(NICHES)
            assert original['actions']==actions
            assert original['legal_rate']==1 and protected_hits==0
            assert round(successes/original['episodes'],6)==original['mission_success']
            results.append(dict(seed=meta['seed'],mode=mode,
                episodes=original['episodes'],actions=actions,legal=legal,
                painted_protected=protected_hits,successes=successes,
                mission_success=round(successes/original['episodes'],6),
                niche_hamming=round(statistics.mean(distances),6),
                unique_niches=round(statistics.mean(unique),6),
                unique_histogram=dict(Counter(unique)),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    aggregate={}
    for mode in MODES:
        rr=[r for r in results if r['mode']==mode]
        aggregate[mode]=dict(episodes=sum(r['episodes'] for r in rr),
            actions=sum(r['actions'] for r in rr),
            successes=sum(r['successes'] for r in rr),
            legal_rate=1.0,
            niche_hamming=round(statistics.mean(r['niche_hamming'] for r in rr),6),
            unique_niche_trajectories=round(statistics.mean(r['unique_niches'] for r in rr),6))
    return dict(format='population-p16-independent-checkpoint-replay',native=False,
        no_game_integration=True,cohorts=results,aggregate=aggregate,
        episodes=sum(r['episodes'] for r in results),
        actions=sum(r['actions'] for r in results),
        protected_hits=sum(r['painted_protected'] for r in results),
        metric='Niche-conditioned Hamming across six labels; NOT artistic-niche validation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('output')
    args=p.parse_args()
    result=audit(args.directory)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(episodes=result['episodes'],actions=result['actions'],
        protected_hits=result['protected_hits'],
        diversity={m:r['niche_hamming'] for m,r in result['aggregate'].items()}),sort_keys=True))
