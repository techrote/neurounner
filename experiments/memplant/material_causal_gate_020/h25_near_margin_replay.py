#!/usr/bin/env python3
"""H25 posthoc near-boundary counterfactual replay. NOT a prospective gate.

Usage: python h25_near_margin_replay.py --h25-dir H25_UNZIP --capsule-dir H22_CAPSULE_UNZIP
Require genuine donor and frozen original H25 checkpoint bytes. Never retrain.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

CHECKPOINT_SHA={
 26011639:'49a74ad64fbfd79a4f5771bfa71158186b5025da038e621c87edc0331c234df3',
 26011640:'21df8ec5e5a1a9d16aa6fefd42054eb987a3092719521b7335e32699e7d789c3',
 26011641:'093750d8e0f9b35affe5c831b0c01024a002638e37672fd367db57b3d6093018'}
H25_SHA='b1a385dc4c9bceea4aa5b3bf2f8f22382db1e9f0af7b646b755f631815a00853'
H23_SHA='41475d535ce4005e1f94c57385891b2f90b87f01e5f265d4ee0eec482bfb8358'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--h25-dir',required=True,type=Path);p.add_argument('--capsule-dir',required=True,type=Path)
 a=p.parse_args();h25=a.h25_dir;capsule=a.capsule_dir
 assert sha(h25/'h25_selectivity_repair_trial.py')==H25_SHA
 assert sha(h25/'h23_paired_margin_trial.py')==H23_SHA
 for line in (capsule/'MANIFEST.sha256').read_text().splitlines():
  digest,path=line.split(None,1);assert sha(capsule/path.strip(' *'))==digest
 sys.path.insert(0,str(h25.resolve()));sys.path.insert(0,str((capsule/'source').resolve()))
 import h23_paired_margin_trial as h
 torch.set_num_threads(2)
 values=h.probe_donors(capsule/'source')
 totals={str(edge):{'n':0,'genuine_wrong_pairs':0,'perfect_feature_wrong_pairs':0,'different_actions':0} for edge in (.01,.001,.0001)}
 breakdown=[]
 for seed,expect in CHECKPOINT_SHA.items():
  cp=h25/'h25-output'/f'receiver-h25-{seed}.pt'
  assert sha(cp)==expect,'wrong original H25 checkpoint'
  receiver=h.Receiver().eval();receiver.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
  assert sum(p.numel() for p in receiver.parameters())==209
  rows=[json.loads(s) for s in (h25/'h25-output'/f'h25-receipts-{seed}.jsonl').read_text().splitlines()]
  for fam in ('coprime_hop','nested_sweep'):
   for bucket in ('wet_rich_invariant','wet_poor_invariant','equal_wet_invariant'):
    group=[r for r in rows if r['family']==fam and r['bucket']==bucket]
    assert len(group)==160
    x=np.asarray([r['observation'] for r in group],dtype='f4')
    base=h.oracle_cost(x).argmin(-1)
    assert np.array_equal(base,np.asarray([r['oracle_actions'] for r in group]))
    for edge in (.01,.001,.0001):
     z=x.copy()
     cost=h.oracle_cost(z);difference=cost[:,:,0]-cost[:,:,1]
     margin=np.where(abs(difference[:,0])<abs(difference[:,1]),difference[:,0],difference[:,1])
     shift=(np.sign(margin)*edge-margin)/24.
     z[:,0,:,1]+=shift[:,None];z[:,1,:,1]-=shift[:,None]
     assert z[...,1].min()>=0 and z[...,1].max()<=.30
     oracle=h.oracle_cost(z).argmin(-1)
     assert np.array_equal(oracle,base)
     with torch.no_grad():
      obs=torch.from_numpy(z)
      gd=h.donor_features(z,values,h.SOURCE_SEED)
      pf=[np.where(z[...,0]>.5,v,1.).astype('f4')+z[...,1] for v in h.LAWS]
      genuine=torch.stack([receiver(obs,torch.from_numpy(f)) for f in gd],dim=1).argmax(-1).numpy()
      perfect=torch.stack([receiver(obs,torch.from_numpy(f)) for f in pf],dim=1).argmax(-1).numpy()
     badg=(genuine!=oracle).any(-1)
     badp=(perfect!=oracle).any(-1)
     changing=(genuine!=perfect).any(-1)
     assert np.array_equal(oracle[:,0],oracle[:,1])
     assert np.array_equal(badg,genuine[:,0]!=genuine[:,1]),'wrong action without law flip'
     key=str(edge);t=totals[key]
     t['n']+=len(z);t['genuine_wrong_pairs']+=int(badg.sum());t['perfect_feature_wrong_pairs']+=int(badp.sum());t['different_actions']+=int(changing.sum())
     breakdown.append(dict(seed=seed,family=fam,bucket=bucket,epsilon=edge,n=len(z),genuine_failures=int(badg.sum()),perfect_feature_failures=int(badp.sum())))
 print(json.dumps(dict(label='POSTHOC_ONLY_NOT_PROSPECTIVE',totals=totals,per_cell=breakdown),indent=2))

if __name__=='__main__':main()
