"""Independently replay original TRAP-PREVENT development trajectories from restored weights."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from . import trap_avoidance_lab as lab
from .world import Spec, World, digest
from .train import MAX_RECOVERIES, sha_file


def verify(root:Path):
    summary=json.loads((root/'report.json').read_text())
    if summary['version']!=lab.VERSION or summary['no_heldout_opened'] is not True or summary['native_receipts']!=0:
        raise ValueError('campaign provenance/gate mismatch')
    if summary['source_sha256'] != sha_file(Path(lab.__file__)):
        raise ValueError('exact experiment source mismatch')
    for n,h in summary['source_pins'].items():
        if sha_file(Path(lab.__file__).parent/n)!=h:raise ValueError('source pin mismatch: '+n)
    manifests=json.loads((root/'SHA256SUMS.json').read_text())
    original_files={str(x.relative_to(root)):sha_file(x) for x in root.rglob('*') if x.is_file() and x.name!='SHA256SUMS.json'}
    if manifests!=original_files:raise ValueError('artifact SHA256 manifest does not cover exactly original files')
    models={}
    for arm,m in summary['checkpoints'].items():
        path=root/m['name']
        if sha_file(path)!=m['sha256']:raise ValueError('checkpoint hash')
        models[arm]=lab.LinearRanker.load(path)
        if models[arm].updates!=m['optimizer_updates']:raise ValueError('optimizer count')
    count=choices=forced_count=0
    stats=Counter()
    for p in sorted((root/'trajectories').glob('*.json')):
        e=json.loads(p.read_text())
        if e['version']!=lab.VERSION or e['spec']['split']!='development' or e['native_receipts']!=0:
            raise ValueError('trajectory provenance')
        sp=Spec(e['spec']['split'],e['spec']['family'],e['spec']['seed'],e['spec']['mode'],e['spec']['challenge'])
        if sp.ident!=e['spec']['ident']: raise ValueError('world id')
        w=World.create(sp)
        if w.base_sha()!=e['base_sha256']:raise ValueError('world material mismatch')
        dist=Counter();near=avoidable=zero=forced=0
        for i,step in enumerate(e['events']):
            if w.stopped:raise ValueError('stopped yet events continued')
            model=models[e['arm']]
            action,pmap=model.select(w,e['horizon'])
            if action!=step['model_choice']:raise ValueError('checkpoint logits disagree on policy decision')
            if list(pmap)!=list(step['probabilities']):raise ValueError('probability menu mismatch')
            if not all(np.isclose(pmap[k],step['probabilities'][k],rtol=1e-10,atol=1e-10) for k in pmap):
                raise ValueError('probability source mismatch')
            candidate=lab.forced_alternative(w,action,e['seed'],i) if e['forced'] and i in lab.FORCED_INDICES else None
            chosen=candidate or action
            if step['intervention'] != (candidate is not None) or step['event']['chosen']!=chosen:
                raise ValueError('intervention/action mismatch')
            s=lab.sensor(w,chosen) if chosen in lab.MOVE else None
            expected=None if s is None else [round(float(v),8) for v in s]
            if expected!=step['local_risk']:raise ValueError('local risk mismatch')
            move_options=[a for a in w.menu() if a in lab.MOVE]
            if chosen in lab.MOVE and s[2] and len(move_options)>1:near+=1
            if chosen in lab.MOVE and s[0]<.5 and any(lab.sensor(w,a)[0]>s[0] for a in move_options):avoidable+=1
            if chosen in lab.MOVE and s[0]<.25:zero+=1
            actual=w.apply(chosen,MAX_RECOVERIES)
            if actual!=step['event']:raise ValueError('original world transition/host receipt mismatch')
            dist[chosen]+=1;forced+=int(candidate is not None)
        if e['metrics']!=w.metrics(e['horizon']) or e['final_state_sha256']!=w.state_sha():
            raise ValueError('post-trajectory state metric mismatch')
        if e['trajectory_sha256']!=digest([q['event'] for q in e['events']]):raise ValueError('trajectory hash')
        if dict(sorted(dist.items()))!=e['action_distribution'] or forced!=e['forced_applied']:
            raise ValueError('action distribution/intervention mismatch')
        if (near,avoidable,zero)!=(e['forced_dead_end_choices'],e['avoidable_low_exit_choices'],e['zero_fresh_successor_choices']):
            raise ValueError('risk score mismatch')
        count+=1;choices+=len(e['events']);forced_count+=forced;stats.update(dist)
    if count!=summary['episodes'] or choices!=summary['sequential_choices']:
        raise ValueError('campaign total disagreement')
    return {'original_episodes_verified':count,'original_sequential_choices':choices,'forced_moves_verified':forced_count,
            'file_hashes_verified':len(manifests),
            'action_distribution':dict(sorted(stats.items())),'checkpoint_hashes':[q['sha256'] for q in summary['checkpoints'].values()],
            'native_receipts':0,'heldout_untouched':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');a=parser.parse_args()
    print(json.dumps(verify(Path(a.root)),sort_keys=True,indent=2))
