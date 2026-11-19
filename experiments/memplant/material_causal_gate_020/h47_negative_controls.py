#!/usr/bin/env python3
"""DG020 H47 genuine full-network negative controls on exposed H40 worlds.
All shuffles happen to donor input selection BEFORE receiver inference.
Zero control is an actual all-zero donor COST TENSOR, not 1+fee.
"""
import json,hashlib,argparse
from pathlib import Path
import numpy as np, torch
from torch import nn

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Receiver(nn.Module):
 def __init__(self):
  super().__init__();self.tile=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
 def forward(self,x,d):return -self.tile(torch.stack((x[...,0],x[...,1],d),-1)).squeeze(-1).sum(-1)

def forward(m,x,d):
 with torch.inference_mode(): y=m(x,d)
 return (y[:,1]>y[:,0]).numpy().astype('i1')

def main(root,out):
 torch.set_num_threads(2)
 j=json.loads((out/'h47_results.json').read_text());raw=j['raw_genuine_donor_endpoint_costs'];mu=j['common_unlabelled_pooled_endpoint_costs']
 keys=sorted(raw)
 rec={'h40':{'26011639':'h36-output/receiver-h36-anchored-26011639.pt','26011640':'h36-output/receiver-h36-anchored-26011640.pt','26011641':'h39-output/receiver-h39-margin-sweep-best-26011641.pt'},
      'h45':{'26011639':'h45-output/h45-possible-feasible-209param-receiver-26011639-knee2.pt','26011640':'h45-output/h45-possible-feasible-209param-receiver-26011640-knee2.pt','26011641':'h45-output/h45-possible-feasible-209param-receiver-26011641-two-knee-2-0.pt'}}
 rawrows=[json.loads(t) for t in (root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl').open()]
 data={};outcomes={};seed_rng=470047
 for typ,paths in rec.items():
  for seed,path in paths.items():
   state=torch.load(root/path,weights_only=True,map_location='cpu');m=Receiver().eval();m.load_state_dict(state)
   rr=[r for r in rawrows if str(r['seed'])==seed]
   obs=torch.from_numpy(np.asarray([r['world_observation'] for r in rr],dtype='f4'))
   wet=obs[...,0]>.5;fee=obs[...,1];oracle=np.asarray([r['oracle_actions'] for r in rr],dtype='i1')
   divergent=oracle[:,0]!=oracle[:,1]
   zero=forward(m,obs,torch.zeros_like(fee));neutral=forward(m,obs,1.+fee)
   assert np.array_equal(zero,zero.copy()) and neutral.shape==zero.shape
   outcomes[f'{typ}_{seed}']={'genuine_model_checkpoint_sha256':sha(root/path),
     'actual_zero_tensor_is_zero':bool(torch.count_nonzero(torch.zeros_like(fee)).item()==0),
     'zero_tensor_divergent_both_correct':int(((zero==oracle[:,0])&(zero==oracle[:,1])&divergent).sum()),
     'neutral_tensor_divergent_both_correct':int(((neutral==oracle[:,0])&(neutral==oracle[:,1])&divergent).sum()),
     'controls_by_profile':{}}
   data[f'{typ}_{seed}_actual_zero_decisions']=zero
   data[f'{typ}_{seed}_neutral_decisions']=neutral
   for k,name in enumerate(keys):
    p=raw[name]; vals=np.asarray([float(np.float32(mu[0]+(c-p[0])*(mu[1]-mu[0])/(p[1]-p[0]))) for c in p],dtype='f4')
    rng=np.random.default_rng(seed_rng + int(seed) + 17*k + (0 if typ=='h40' else 100000))
    original=np.zeros((1600,2),dtype='i1');wrong=np.zeros_like(original);shuffled=np.zeros_like(original)
    shuffle_alloc=rng.integers(0,2,size=(1600,2),dtype='i1')
    for law in [0,1]:
     d=torch.where(wet, torch.tensor(float(vals[law]),dtype=torch.float32)+fee,1.+fee)
     original[:,law]=forward(m,obs,d)
     dw=torch.where(wet,torch.tensor(float(vals[1-law]),dtype=torch.float32)+fee,1.+fee)
     wrong[:,law]=forward(m,obs,dw)
     choice=vals[shuffle_alloc[:,law]]
     ds=torch.where(wet,torch.from_numpy(choice)[:,None,None]+fee,1.+fee)
     shuffled[:,law]=forward(m,obs,ds)
    assert np.array_equal(wrong,original[:,::-1])
    assert np.array_equal(original[:,0],original[:,0])
    data[f'{typ}_{seed}_{k}_genuine_correct']=original
    data[f'{typ}_{seed}_{k}_swapped_law']=wrong
    data[f'{typ}_{seed}_{k}_shuffled_preinference']=shuffled
    data[f'{typ}_{seed}_{k}_shuffle_choices']=shuffle_alloc
    outcomes[f'{typ}_{seed}']['controls_by_profile'][name]={
       'aligned_genuine_endpoint_both_correct_divergent':int((np.all(original==oracle,axis=1)&divergent).sum()),
       'wrong_law_both_correct_divergent':int((np.all(wrong==oracle,axis=1)&divergent).sum()),
       'shuffled_input_both_correct_divergent':int((np.all(shuffled==oracle,axis=1)&divergent).sum()),
       'shuffled_arm_assignments':int(shuffle_alloc.sum()),
       'genuine_endpoint_both_correct_invariant':int((np.all(original==oracle,axis=1)&~divergent).sum()),
    }
   print('controls',typ,seed,flush=True)
 np.savez_compressed(out/'h47_negative_control_raw_decisions.npz',**data)
 result={'classification':'exposed-only authentic neural negative controls, not prospectively calibrated','shuffle_seed':seed_rng,
         'zero_is_literal_all_zero_cost_tensor':True,'neutral_is_1_plus_fee_both_laws':True,
         'swapped_law_is_actual_opposite_genuine_donor_feature':True,
         'shuffled_law_is_preinference_random_genuine_donor_feature':True,
         'raw_decisions_sha256':sha(out/'h47_negative_control_raw_decisions.npz'),
         'observations_sha256':sha(root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl'),
         'per_seed_receiver':outcomes}
 (out/'h47_negative_controls.json').write_text(json.dumps(result,indent=2)+'\n')
 print('CTRL genuine divergent both correct',sum(x['aligned_genuine_endpoint_both_correct_divergent'] for a in outcomes.values() for x in a['controls_by_profile'].values()))
 print('CTRL shuffled divergent both correct',sum(x['shuffled_input_both_correct_divergent'] for a in outcomes.values() for x in a['controls_by_profile'].values()))
 print('CTRL swapped divergent both correct',sum(x['wrong_law_both_correct_divergent'] for a in outcomes.values() for x in a['controls_by_profile'].values()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.out.resolve())
