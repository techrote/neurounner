#!/usr/bin/env python3
"""H50 exposed-only receiver-precision audit with actual frozen neural checkpoints.

The 8 genuine donor endpoints are fixed to H47's archived trained-network
probe outputs.  The world, network weights and donor scalar input are identical
between CPU float32 and float64.  This tests numerical receiver sensitivity,
NOT prospective law-family generalization or donor training reproducibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

RECEIVERS = {
    'h40': {
        '26011639': 'h36-output/receiver-h36-anchored-26011639.pt',
        '26011640': 'h36-output/receiver-h36-anchored-26011640.pt',
        '26011641': 'h39-output/receiver-h39-margin-sweep-best-26011641.pt',
    },
    'h45': {
        '26011639': 'h45-output/h45-possible-feasible-209param-receiver-26011639-knee2.pt',
        '26011640': 'h45-output/h45-possible-feasible-209param-receiver-26011640-knee2.pt',
        '26011641': 'h45-output/h45-possible-feasible-209param-receiver-26011641-two-knee-2-0.pt',
    },
}
H40_RECEIPT_SHA = '5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d'
H47_MARGINS_SHA = '77e96494144c2015dab45ad4823dfbc17056b1c064cdd2e5a793e1c324da7bee'
CHECKPOINT_SHA = {
    'h40': {'26011639':'6ec90371ecfb7fdcc7291959bc1e9a0d98df71836cef3100a2205f3dfea3fa2d', '26011640':'dc2b2fbc7b9fa04303915cc92255bdcb0fb15e7218c9e96a4e5e24717317ed05', '26011641':'b19af59d00a8b7a76d20e379b165211066415ea9bab4f2170e5bf069814089f4'},
    'h45': {'26011639':'0647f98131e30026e5f8211570cda73adcf6172a8e6238430f7b73efbac5df1f', '26011640':'7eaae96ba2b8e73bbfdb144f1d86eb78d101d887463b5002e6a2291d819458a6', '26011641':'acec1cc9ac7ac221131e60b515d2c3025f20cd9b95afd134c27b48d23f63a542'},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def margin(state,obs,scalar,dtype):
    obs=obs.to(dtype)
    w=obs[...,0]>.5
    fee=obs[...,1]
    scalar=torch.tensor(float(np.float32(scalar)),dtype=dtype)
    cost=torch.where(w,scalar+fee,torch.ones_like(fee)+fee)
    q=torch.stack((obs[...,0],fee,cost),dim=-1)
    for j in (0,2):
        q=F.relu(F.linear(q,state[f'tile.{j}.weight'].to(dtype),state[f'tile.{j}.bias'].to(dtype)))
    last=F.linear(q,state['tile.4.weight'].to(dtype),state['tile.4.bias'].to(dtype)).squeeze(-1)
    route=-last.sum(-1)
    return (route[:,1]-route[:,0]).detach().numpy()

def audit(root):
    torch.set_num_threads(1)
    report=json.loads((root/'h47-output/h47_results.json').read_text())
    receipts=root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl'
    margins=root/'h47-output/h47_all_float32_neural_route_margins.npz'
    assert sha(receipts)==H40_RECEIPT_SHA and sha(margins)==H47_MARGINS_SHA
    rows=[json.loads(s) for s in receipts.open()]
    assert len(rows)==4800
    saved=np.load(margins)
    donor=report['raw_genuine_donor_endpoint_costs']
    keys=sorted(donor)
    assert len(keys)==4 and len(report['genuine_donor_checkpoint_sha256'])==8
    pooled=report['common_unlabelled_pooled_endpoint_costs']
    outcome={'classification':'H50 exposed-only independent CPU float64 receiver sensitivity audit; no new prospective cohort',
      'source_h40_receipts_sha256':H40_RECEIPT_SHA,'source_h47_margins_sha256':H47_MARGINS_SHA,
      'torch':torch.__version__,'numpy':np.__version__,'receiver_conditions':{}}
    for kind,paths in RECEIVERS.items():
        for seed,path in paths.items():
            cp=root/path
            expected=CHECKPOINT_SHA[kind][seed]
            assert sha(cp)==expected,(cp,sha(cp),expected)
            state=torch.load(cp,map_location='cpu',weights_only=True)
            assert sum(t.numel() for t in state.values())==209
            subset=[r for r in rows if str(r['seed'])==seed]
            assert len(subset)==1600
            obs=torch.from_numpy(np.asarray([r['world_observation'] for r in subset],dtype='f4'))
            oracle=np.asarray([r['oracle_actions'] for r in subset],dtype='i1').T
            assert oracle.shape==(2,1600)
            div=np.asarray([r['bucket'].endswith('divergent') for r in subset])
            assert div.sum()==640 and (~div).sum()==960
            check= saved[f'{kind}_{seed}_endpoint_margins'];assert check.shape==(2,2,4,1600)
            stat={'checkpoint_sha256':expected,'receiver_parameters':209,'observations':1600,'divergent_pairs':int(div.sum()),'invariant_pairs':int((~div).sum()),'conditions':{}}
            for mode in ('raw','aligned'):
                d32=[];d64=[]; archived=[]
                for law in (0,1):
                    for profile in keys:
                        p=donor[profile]
                        c=p[law] if mode=='raw' else float(np.float32(pooled[0]+(p[law]-p[0])*(pooled[1]-pooled[0])/(p[1]-p[0])))
                        v32=margin(state,obs,c,torch.float32)
                        v64=margin(state,obs,c,torch.float64)
                        d32.append(v32);d64.append(v64)
                        archived.append(check[0 if mode=='raw' else 1,law,keys.index(profile)])
                d32=np.asarray(d32).reshape(2,4,1600)
                d64=np.asarray(d64).reshape(2,4,1600)
                archived=np.asarray(archived).reshape(2,4,1600)
                flip=(d32>0)!=(d64>0)
                archived_flip=(archived>0)!=(d64>0)
                condition={'total_endpoint_decisions':int(d32.size),
                  'local_f32_to_f64_action_changes':int(flip.sum()),
                  'archived_f32_to_f64_action_changes':int(archived_flip.sum()),
                  'local_f32_to_archived_action_changes':int(((d32>0)!=(archived>0)).sum()),
                  'f64_oracle_action_errors':int(((d64>0)!=oracle[:,None,:]).sum()),
                  'f64_divergent_action_errors':int((((d64>0)!=oracle[:,None,:])[:,:,div]).sum()),
                  'f64_invariant_action_errors':int((((d64>0)!=oracle[:,None,:])[:,:,~div]).sum()),
                  'worst_abs_f64_minus_f32_margin':float(np.max(np.abs(d64-d32))),
                  'min_abs_f64_margin':float(np.min(np.abs(d64))),
                  'cross_dtype_action_changes_by_law':[int(flip[i].sum()) for i in range(2)],
                  'cross_dtype_action_changes_by_family':{f:int(flip[:,:,np.asarray([r['family']==f for r in subset])].sum()) for f in ('interleaved_braid','reverse_bifurcation')},
                  'examples':[]}
                for law,k,n in np.argwhere(archived_flip)[:8]:
                    condition['examples'].append({'law':int(law),'donor_profile':keys[k],
                      'world_index':int(n),'bucket':subset[n]['bucket'],
                      'archived_f32_margin':float(archived[law,k,n]),
                      'local_f32_margin':float(d32[law,k,n]),
                      'f64_margin':float(d64[law,k,n])})
                stat['conditions'][mode]=condition
            outcome['receiver_conditions'][f'{kind}/{seed}']=stat
    outcome['totals']={}
    for kind in RECEIVERS:
        for mode in ('raw','aligned'):
            stats=[outcome['receiver_conditions'][f'{kind}/{seed}']['conditions'][mode] for seed in RECEIVERS[kind]]
            outcome['totals'][f'{kind}_{mode}']={k:sum(q[k] for q in stats) for k in (
              'total_endpoint_decisions','local_f32_to_f64_action_changes',
              'archived_f32_to_f64_action_changes','local_f32_to_archived_action_changes',
              'f64_oracle_action_errors','f64_divergent_action_errors','f64_invariant_action_errors')}
            outcome['totals'][f'{kind}_{mode}']['min_abs_f64_margin']=min(q['min_abs_f64_margin'] for q in stats)
    outcome['H40_original_restricted_gate_not_retested']=True
    ok=all(t['total_endpoint_decisions']==38400 and
           t['local_f32_to_f64_action_changes']==0 and
           t['archived_f32_to_f64_action_changes']==0 and
           t['local_f32_to_archived_action_changes']==0 and
           t['f64_oracle_action_errors']==0
           for t in outcome['totals'].values())
    outcome['verdict']='PASS: exposed-only receiver precision audit completed' if ok else 'FAIL: cross-precision or original-action regression'
    outcome['scientific_prospective_verdict']='NOT_RETESTED'
    return outcome

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('evidence_root',type=Path);ap.add_argument('--output',type=Path)
    args=ap.parse_args();r=audit(args.evidence_root.resolve());s=json.dumps(r,indent=2,sort_keys=True)+'\n'
    if args.output:args.output.write_text(s)
    print(json.dumps({'verdict':r['verdict'],'totals':r['totals']},indent=2))
