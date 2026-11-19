#!/usr/bin/env python3
"""Compact independent H42 posthoc cross-interpolation reproducer.

Requires only archived H40 evidence directory (H42 evidence subset suffices).
This is NOT a genuine intermediate-law donor trial or a prospective gate.
"""
import json,hashlib,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn

H40_SHA='5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d'
RECEIVERS={
 '26011639':('h36-output/receiver-h36-anchored-26011639.pt','6ec90371ecfb7fdcc7291959bc1e9a0d98df71836cef3100a2205f3dfea3fa2d'),
 '26011640':('h36-output/receiver-h36-anchored-26011640.pt','dc2b2fbc7b9fa04303915cc92255bdcb0fb15e7218c9e96a4e5e24717317ed05'),
 '26011641':('h39-output/receiver-h39-margin-sweep-best-26011641.pt','b19af59d00a8b7a76d20e379b165211066415ea9bab4f2170e5bf069814089f4')}

class Donor(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(4,24),nn.Tanh(),nn.Linear(24,24),nn.Tanh(),nn.Linear(24,1))
 def forward(self,x):return self.net(x).squeeze(-1)
class Receiver(nn.Module):
 def __init__(self):
  super().__init__();self.tile=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
 def forward(self,obs,donor):
  return -self.tile(torch.stack((obs[...,0],obs[...,1],donor),-1)).squeeze(-1).sum(-1)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def demand(value,message):
 if not value:raise AssertionError(message)

def replay(root):
 torch.set_num_threads(2)
 receipt=root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl'
 demand(sha(receipt)==H40_SHA,'H40 receipt corruption')
 rows=[json.loads(r) for r in receipt.read_text().splitlines()]
 demand(len(rows)==4800,'H40 world count')
 meta=json.loads((root/'h40-heldout/h34-donor-training.json').read_text())
 nodes,weights=np.polynomial.legendre.leggauss(5)
 probe=np.empty((2,25,4),dtype='f4')
 for material in (0,1):
  probe[material,:,0]=material;probe[material,:,1]=.15
  probe[material,:,2:]=[(a,b) for a in nodes for b in nodes]
 w=np.array([a*b/4 for a in weights for b in weights])
 donors={}
 with torch.no_grad():
  for trained in meta['trained']:
   path=root/'h40-heldout'/trained['checkpoint']
   demand(sha(path)==trained['checkpoint_sha256'],'original donor checkpoint corruption')
   m=Donor().eval();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
   prediction=(m(torch.from_numpy(probe)).numpy().astype('f8')@w-.15).astype('f4')
   donors.setdefault(f"{trained['source']}-rep{trained['replica']}",{})[trained['law']]=prediction
 demand(len(donors)==4 and all(set(v)=={0,1} for v in donors.values()),'genuine donor pair count')
 # Single scalar extracted from each genuine donor network, as in original H40.
 costs={k:[float(p[l][1])-float(p[0][0])+1 for l in (0,1)] for k,p in donors.items()}
 alpha=np.linspace(0,1,65);physical_cost=.35+1.95*alpha
 report=Counter(); buckets={}
 with torch.no_grad():
  for seed,(relative,expected_hash) in RECEIVERS.items():
   path=root/relative;demand(sha(path)==expected_hash,'frozen receiver checkpoint corruption')
   m=Receiver().eval();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
   demand(sum(p.numel() for p in m.parameters())==209,'receiver architecture changed')
   data=[r for r in rows if str(r['seed'])==seed]
   demand(len(data)==1600,'seed world count')
   x=np.stack([np.array(r['world_observation'],dtype='f4') for r in data]);xt=torch.from_numpy(x)
   wet=x[...,0]>.5;fee=x[...,1]
   countdiff=wet[:,0].sum(-1)-wet[:,1].sum(-1)
   feediff=fee[:,0].sum(-1,dtype='f8')-fee[:,1].sum(-1,dtype='f8')
   truth=(feediff[None,:]+countdiff[None,:]*(physical_cost[:,None]-1)>0).astype(int)
   response=[]
   for k in sorted(costs):
    c0,c1=costs[k];values=[]
    for a in alpha:
     d=np.where(wet,c0*(1-a)+c1*a,1.).astype('f4')+fee
     values.append(m(xt,torch.from_numpy(d)).numpy())
    response.append(np.stack(values))
   scores=np.stack(response) # [donor,alpha,world,route]
   actions=scores.argmax(-1)
   endpoints=np.stack([np.asarray(r['neural_actions']) for r in data],axis=1)
   demand(np.array_equal(actions[:,[0,-1]].transpose(0,2,1),endpoints),'genuine H40 endpoint actions differ')
   neural=np.stack([np.asarray(r['neural_logits'],dtype='f4') for r in data],axis=1)
   peak=max(float(np.max(np.abs(scores[:,0]-neural[:,:,0]))),float(np.max(np.abs(scores[:,-1]-neural[:,:,1]))))
   demand(peak<=1.1e-5,'CPU float drift beyond tolerance')
   report['endpoint_peak_drift_micro']=max(report['endpoint_peak_drift_micro'],round(peak*1e6,4))
   wrong=actions!=truth[None]
   report['interpolation_wrong']+=int(wrong.sum())
   report['interpolation_total']+=int(wrong.size)
   report['endpoint_correct']+=int((actions[:,[0,-1]]==truth[None,[0,-1]]).sum())
   for i,r in enumerate(data):
    bucket=r['bucket']; v=buckets.setdefault(bucket,Counter());v['worlds']+=1
    errors=int(wrong[:,:,i].sum());v['wrong_action_grid_points']+=errors
    if errors:v['worlds_with_wrong_action']+=1
    crossings=np.count_nonzero(np.diff(actions[:,:,i],axis=1),axis=1)
    if 'invariant' in bucket:
     v['inappropriate_interpolated_transitions']+=int(np.count_nonzero(crossings>0))
    v['multiple_crossings']+=int(np.count_nonzero(crossings>1))
    report['worlds']+=1
 report['buckets']=dict((b,dict(v)) for b,v in buckets.items())
 return dict(report)

if __name__=='__main__':
 root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
 result=replay(root)
 print(json.dumps({'status':'H42 POSTHOC MECHANISM FAIL / H40 original PASS unchanged',**result},indent=2))
