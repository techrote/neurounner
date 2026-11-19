#!/usr/bin/env python3
"""H46 capsule verifier: independently replay every bit-identical input/contradictory-label witness.
Run from the extracted owner H46 archive: python h46_verify_exact_input_collision.py .
CPU-only; never a prospective neural science gate.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,24),nn.Tanh(),nn.Linear(24,24),nn.Tanh(),nn.Linear(24,1))
    def forward(self,x): return self.net(x).squeeze(-1)
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
    def forward(self,o,c):
        x=torch.stack((o[...,0],o[...,1],c),-1)
        return -self.tile(x).squeeze(-1).sum(-1)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(root):
    torch.set_num_threads(2)
    raw=root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl'
    assert sha(raw)=='5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d'
    worlds=[json.loads(s) for s in raw.open()]
    byseed={s:[w for w in worlds if str(w['seed'])==s] for s in ('26011639','26011640','26011641')}
    meta=json.loads((root/'h40-heldout/h34-donor-training.json').read_text())
    nodes,weights=np.polynomial.legendre.leggauss(5)
    q=np.empty((2,25,4),dtype='f4')
    for wet in (0,1):
        q[wet,:,0]=wet;q[wet,:,1]=0.15
        q[wet,:,2:]=[(a,b) for a in nodes for b in nodes]
    qw=np.array([a*b/4 for a in weights for b in weights],dtype='f8')
    estimated={}
    with torch.inference_mode():
        for e in meta['trained']:
            path=root/'h40-heldout'/e['checkpoint']
            assert sha(path)==e['checkpoint_sha256']
            d=D().eval();d.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            signal=(d(torch.from_numpy(q)).numpy().astype('f8')@qw-0.15).astype('f4')
            estimated.setdefault(f"{e['source']}-rep{e['replica']}",{})[e['law']]=signal
    cc={k:(float(np.float32(v[0][1]-v[0][0]+1.)),
           float(np.float32(v[1][1]-v[0][0]+1.))) for k,v in estimated.items()}
    audit=json.loads((root/'h46-output/h46_identifiability_results.json').read_text())
    assert cc=={k:tuple(v) for k,v in audit['donor_cost_endpoint_profiles'].items()}
    witness=root/'h46-output/h46_all_bit_identical_opposite_oracle_witnesses.jsonl'
    assert sha(witness)==audit['all_witness_receipts_sha256']
    w=[json.loads(s) for s in witness.open()]
    assert len(w)==audit['total_verified_contradictory_pairs']==11286
    checkpoints={'26011639':'h45-possible-feasible-209param-receiver-26011639-knee2.pt',
                 '26011640':'h45-possible-feasible-209param-receiver-26011640-knee2.pt',
                 '26011641':'h45-possible-feasible-209param-receiver-26011641-two-knee-2-0.pt'}
    checked=0; unique=set()
    for seed,file in checkpoints.items():
        path=root/'h45-output'/file
        expected=next(z['h45_receiver_sha256'] for z in audit['worlds'] if z['seed']==seed)
        assert sha(path)==expected
        rec=R().eval();rec.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        assert sum(p.numel() for p in rec.parameters())==209
        selected=[item for item in w if item['seed']==seed]
        for j in range(0,len(selected),128):
            batch=selected[j:j+128]
            xs=np.stack([np.asarray(byseed[seed][r['index']]['world_observation'],dtype='f4') for r in batch])
            x=torch.from_numpy(xs)
            costs=[]
            for k,item in enumerate(batch):
                o=xs[k];wet=o[...,0]>.5
                dw=int(wet[0].sum()-wet[1].sum())
                df=float(o[0,:,1].sum(dtype='f8')-o[1,:,1].sum(dtype='f8'))
                alpha0=float(item['physical_alpha_below'])
                alpha1=float(item['physical_alpha_above'])
                c0,c1=cc[item['source_below']],cc[item['source_above']]
                v0=np.float32((1-alpha0)*c0[0]+alpha0*c0[1])
                v1=np.float32((1-alpha1)*c1[0]+alpha1*c1[1])
                assert v0.view('u4')==v1.view('u4')
                assert format(int(v0.view('u4')),'08x')==item['cost_bits']
                a0=int(df+dw*(0.35+1.95*alpha0-1)>0)
                a1=int(df+dw*(0.35+1.95*alpha1-1)>0)
                assert a0==item['expected_below'] and a1==item['expected_above'] and a0!=a1
                assert byseed[seed][item['index']]['observation_sha256']==item['observation_sha256']
                costs.append(v0);unique.add((seed,item['index']))
            f=x[...,1];wet=x[...,0]>.5
            c=torch.tensor(np.asarray(costs,dtype='f4')).view(-1,1,1)
            d=torch.where(wet,c+f,1.+f)
            t=torch.stack((x[...,0],x[...,1],d),-1).numpy()
            with torch.inference_mode():logits=rec(x,d).numpy().astype('f4')
            for k,item in enumerate(batch):
                assert hashlib.sha256(t[k].tobytes()).hexdigest()==item['complete_tensor_sha256']
                assert np.array_equal(logits[k].view('u4'),np.asarray(item['receiver_logits_float32'],dtype='f4').view('u4'))
                assert int(logits[k,1]>logits[k,0])==item['predicted_both']
            checked+=len(batch)
    assert len(unique)==1920 and checked==11286
    print('H46 exact-input contradictory-oracle witnesses: 11286/11286 PASS across 1920 divergent exposed worlds')
    print('Scientific gate: NOT RETESTED. Universally exact nominal synthetic-alpha labels are non-identifiable.')

if __name__=='__main__':main(Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve())
