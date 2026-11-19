#!/usr/bin/env python3
"""H26 post-hoc *negative* margin falsification on already exposed worlds.

No new prospective acceptance, no retraining, no changed law/geometry/weights.
Fee offsets are applied identically to the TWO matched law arms; oracle action
class remains identical before/after. Exact-cost donor is a diagnostic ablation,
NEVER a genuine-donor scientific success. Original H26 30/30 remains intact.
"""
import argparse, hashlib, io, json
from pathlib import Path
import numpy as np
import torch
from h26_original_checkpoints import checkpoint_bytes
from h26_git_custody_replay import (
    SEEDS,SOURCES,LAWS,FAMILIES,BUCKETS,Receiver,open_h22,trained_features,worlds,score
)

TARGETS=(None,.005,.001,.0001)

def cost(x):
    wet=x[...,0]>.5
    return np.stack([(np.where(wet,law,1.)+x[...,1]).sum(-1,dtype='f8') for law in LAWS],axis=1)

def narrow(x,target):
    if target is None:return x.copy()
    y=x.copy()
    d=cost(x)[:,:,0]-cost(x)[:,:,1]
    closest=np.where(np.abs(d[:,0])<np.abs(d[:,1]),d[:,0],d[:,1])
    delta=(np.sign(closest)*target-closest)/24.
    y[:,0,:,1]+=delta[:,None];y[:,1,:,1]-=delta[:,None]
    if y[...,1].min()<0 or y[...,1].max()>.30:raise AssertionError('invalid physical fee')
    if not np.array_equal(cost(y).argmin(-1),cost(x).argmin(-1)):
        raise AssertionError('oracle action changed under fee compression')
    return y

def main(out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    donors=trained_features(open_h22())
    # Ablation only: replace neural predictions with exact material costs; does
    # not count as donor inheritance or model success.
    analytic={src:{0:np.array([1.,.35],dtype='f4'),1:np.array([1.,2.30],dtype='f4')} for src in SOURCES}
    totals={}
    rows=0
    with (out/'h26-posthoc-margin-raw.jsonl').open('w') as receipts:
        for seed in SEEDS:
            model=Receiver().eval()
            model.load_state_dict(torch.load(io.BytesIO(checkpoint_bytes(seed)),weights_only=True,map_location='cpu'))
            for family in FAMILIES:
                for bucket in BUCKETS:
                    original=worlds(seed,family,bucket)
                    source_hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in original]
                    for target in TARGETS:
                        obs=narrow(original,target)
                        oracle=cost(obs).argmin(-1)
                        is_div=bucket.endswith('divergent')
                        if not np.all((oracle[:,0]!=oracle[:,1])==is_div):
                            raise AssertionError('oracle category changed')
                        primary=score(model,obs,donors,SOURCES[0])
                        foreign=score(model,obs,donors,SOURCES[1])
                        exact=score(model,obs,analytic,SOURCES[0])
                        actions=primary.argmax(-1)
                        foreign_actions=foreign.argmax(-1)
                        analytic_actions=exact.argmax(-1)
                        k=str(target)
                        group=totals.setdefault(k,{'exposed_cells':30,'pairs':0,'divergent_both_correct':0,
                            'necessary_divergent_flips':0,'invariant_both_correct':0,
                            'wrong_invariant_action_flips':0,'same_law_foreign_changes':0,
                            'analytic_ablation_wrong_invariant_flips':0,
                            'analytic_ablation_disagreements':0,'cell_failures':[]})
                        both=int((actions==oracle).all(-1).sum())
                        flips=int((actions[:,0]!=actions[:,1]).sum())
                        wrong=int(((actions[:,0]!=actions[:,1])&(oracle[:,0]==oracle[:,1])).sum())
                        sham=int((actions!=foreign_actions).any(-1).sum())
                        awrong=int(((analytic_actions[:,0]!=analytic_actions[:,1])&(oracle[:,0]==oracle[:,1])).sum())
                        mismatch=int((analytic_actions!=actions).any(-1).sum())
                        group['pairs']+=160
                        group['same_law_foreign_changes']+=sham
                        group['analytic_ablation_disagreements']+=mismatch
                        group['analytic_ablation_wrong_invariant_flips']+=awrong
                        if is_div:
                            group['divergent_both_correct']+=both
                            group['necessary_divergent_flips']+=flips
                        else:
                            group['invariant_both_correct']+=both
                            group['wrong_invariant_action_flips']+=wrong
                        if both<151:
                            group['cell_failures'].append({'seed':seed,'family':family,'bucket':bucket,
                                'both_correct':both,'wrong_invariant_flips':wrong})
                        for i in range(160):
                            receipts.write(json.dumps({'seed':seed,'family':family,'bucket':bucket,
                                'target_minimum_margin':target,'index':i,'original_observation_sha256':source_hashes[i],
                                'transformed_observation_sha256':hashlib.sha256(obs[i].tobytes()).hexdigest(),
                                'oracle':oracle[i].tolist(),'genuine_actions':actions[i].tolist(),
                                'foreign_genuine_donor_actions':foreign_actions[i].tolist(),
                                'perfect_cost_diagnostic_actions':analytic_actions[i].tolist(),
                                'genuine_logits':primary[i].tolist()},separators=(',',':'))+'\n')
                            rows+=1
    if totals['None']['divergent_both_correct']!=1920 or totals['None']['invariant_both_correct']!=2880:
        raise AssertionError('H26 original restricted gate unexpected regression')
    result={'status':'POSTHOC_DIAGNOSTIC_ONLY','prospective_h26_scientific_disposition':'ORIGINAL_30_OF_30_PASS_UNCHANGED',
            'trained_donors_and_receivers':True,'original_h26_weight_hashes_preserved':True,
            'claim_boundary':'Only previously exposed H26 synthetic world instances, no new holdout, thresholds not preregistered',
            'cells':totals,'raw_rows':rows,
            'raw_sha256':hashlib.sha256((out/'h26-posthoc-margin-raw.jsonl').read_bytes()).hexdigest()}
    (out/'h26-posthoc-margin-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'raw_rows':rows,
        'wrong_invariant_flips_by_margin':{k:v['wrong_invariant_action_flips'] for k,v in totals.items()},
        'analytic_ablation_wrong_flips_by_margin':{k:v['analytic_ablation_wrong_invariant_flips'] for k,v in totals.items()}},indent=2))
    if totals['0.001']['wrong_invariant_action_flips']<=0:
        raise AssertionError('Expected previously observed H26 margin failure vanished; investigate CPU/model drift')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,default=Path('h26-posthoc-output'))
    main(a.parse_args().out)
