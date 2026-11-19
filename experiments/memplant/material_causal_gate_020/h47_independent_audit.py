#!/usr/bin/env python3
"""Independent H47 full-score audit. No import of experimental implementation.
Recompute 5x5 genuine donor probes and manually evaluate frozen neural nets;
then independently score every original/aligned endpoint and synthetic offgrid logit.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F

SH='5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d'
REC={'26011639':'h36-output/receiver-h36-anchored-26011639.pt',
     '26011640':'h36-output/receiver-h36-anchored-26011640.pt',
     '26011641':'h39-output/receiver-h39-margin-sweep-best-26011641.pt'}
CAND={'26011639':'h45-output/h45-possible-feasible-209param-receiver-26011639-knee2.pt',
     '26011640':'h45-output/h45-possible-feasible-209param-receiver-26011640-knee2.pt',
     '26011641':'h45-output/h45-possible-feasible-209param-receiver-26011641-two-knee-2-0.pt'}

def hashfile(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
def neural_donor(state,x):
    for j in (0,2):
        x=torch.tanh(F.linear(x,state[f'net.{j}.weight'],state[f'net.{j}.bias']))
    return F.linear(x,state['net.4.weight'],state['net.4.bias']).squeeze(-1)
def neural_route(state,obs,c):
    fee=obs[...,1];w=obs[...,0]>.5
    cost=torch.where(w,torch.tensor(c,dtype=torch.float32)+fee,1.+fee)
    x=torch.stack((obs[...,0],fee,cost),-1)
    for j in (0,2):x=F.relu(F.linear(x,state[f'tile.{j}.weight'],state[f'tile.{j}.bias']))
    y=F.linear(x,state['tile.4.weight'],state['tile.4.bias']).squeeze(-1)
    out=-y.sum(-1)
    return (out[:,1]-out[:,0]).numpy().astype('f4')

def main(root,out):
    torch.set_num_threads(2)
    report=json.loads((out/'h47_results.json').read_text())
    assert hashfile(root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl')==SH
    rows=[json.loads(s) for s in (root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl').open()]
    meta=json.loads((root/'h40-heldout/h34-donor-training.json').read_text())['trained']
    nodes,ws=np.polynomial.legendre.leggauss(5)
    query=torch.zeros((2,25,4),dtype=torch.float32)
    for k in (0,1):
        query[k,:,0]=float(k);query[k,:,1]=.15
        query[k,:,2:]=torch.tensor([(u,v) for u in nodes for v in nodes],dtype=torch.float32)
    weights=np.array([u*v/4 for u in ws for v in ws],dtype='f8')
    donor={}
    with torch.inference_mode():
        for item in meta:
            name=f"{item['source']}-rep{item['replica']}"
            cp=root/'h40-heldout'/item['checkpoint']
            assert hashfile(cp)==item['checkpoint_sha256']
            t=torch.load(cp,weights_only=True,map_location='cpu')
            val=(neural_donor(t,query).numpy().astype('f8')@weights-.15).astype('f4')
            donor.setdefault(name,{})[item['law']]=val
    raw={k:(float(v[0][1])-float(v[0][0])+1.,float(v[1][1])-float(v[0][0])+1.) for k,v in donor.items()}
    assert raw=={k:tuple(v) for k,v in report['raw_genuine_donor_endpoint_costs'].items()}
    mu=(np.mean([v[0] for v in raw.values()],dtype='f8'),np.mean([v[1] for v in raw.values()],dtype='f8'))
    assert np.array_equal(np.asarray(mu),np.asarray(report['common_unlabelled_pooled_endpoint_costs']))
    arrays=np.load(out/'h47_all_float32_neural_route_margins.npz'); keys=sorted(raw)
    audit={'file_sha256':hashfile(out/'h47_all_float32_neural_route_margins.npz'), 'neural_checkpoints_replayed':[], 'all_scores_checked':0,
           'bit_disagreements':0,'decision_disagreements':0,'worst_abs_score_drift':0.0,'seed_control_counts':{}}
    assert audit['file_sha256']==report['all_raw_neural_route_margins_sha256']
    alphas=np.asarray([j/256 for j in range(1,256) if j%4])
    for seed in REC:
        rr=[r for r in rows if str(r['seed'])==seed]
        obs=torch.from_numpy(np.asarray([r['world_observation'] for r in rr],dtype='f4'))
        wet=obs[...,0]>.5;fee=obs[...,1]
        dw=wet[:,0].sum(-1).numpy()-wet[:,1].sum(-1).numpy()
        df=fee[:,0].numpy().sum(-1,dtype='f8')-fee[:,1].numpy().sum(-1,dtype='f8')
        truth=((df[None]+dw[None]*(.35+1.95*alphas[:,None]-1.))>0).astype('i1')
        for typ, path in [('h40',REC[seed]),('h45',CAND[seed])]:
            cp=root/path;state=torch.load(cp,weights_only=True,map_location='cpu')
            assert hashfile(cp)==report['receiver_seeds'][seed][f'{typ}_'+('original_frozen_receiver_sha256' if typ=='h40' else 'adjusted_receiver_sha256')]
            audit['neural_checkpoints_replayed'].append(path)
            ep=arrays[f'{typ}_{seed}_endpoint_margins'];assert ep.shape==(2,2,4,1600)
            off=arrays[f'{typ}_{seed}_offgrid_margins'] if typ=='h45' else None
            for k,name in enumerate(keys):
                p=raw[name]
                for cond in (0,1):
                    for law in (0,1):
                        cost=p[law] if cond==0 else mu[0]+(p[law]-p[0])*(mu[1]-mu[0])/(p[1]-p[0])
                        if cond:cost=float(np.float32(cost))
                        s=neural_route(state,obs,cost)
                        saved=ep[cond,law,k]
                        audit['all_scores_checked']+=len(s)
                        audit['bit_disagreements']+=int((s.view('u4')!=saved.view('u4')).sum())
                        audit['decision_disagreements']+=int(((s>0)!=(saved>0)).sum())
                        audit['worst_abs_score_drift']=max(audit['worst_abs_score_drift'],float(np.max(np.abs(s.astype('f8')-saved.astype('f8')))))
                    if typ=='h45':
                        for j,a in enumerate(alphas):
                            cost=float(np.float32((1-a)*p[0]+a*p[1]))
                            if cond: cost=float(np.float32(mu[0]+(cost-p[0])*(mu[1]-mu[0])/(p[1]-p[0])))
                            s=neural_route(state,obs,cost);saved=off[cond,k,j]
                            audit['all_scores_checked']+=len(s)
                            audit['bit_disagreements']+=int((s.view('u4')!=saved.view('u4')).sum())
                            audit['decision_disagreements']+=int(((s>0)!=(saved>0)).sum())
                            audit['worst_abs_score_drift']=max(audit['worst_abs_score_drift'],float(np.max(np.abs(s.astype('f8')-saved.astype('f8')))))
                print('audit',seed,typ,name,flush=True)
            # Reconstruct aggregate from raw neural score array and oracle independently.
            expected=np.asarray([r['oracle_actions'] for r in rr],dtype='i1').T
            for cond,cname in enumerate(('raw','aligned')):
                bad=(ep[cond]>0)!=expected[:,None,:]
                assert int(bad.sum())==report['receiver_seeds'][seed][typ][cname]['endpoint_action_errors']
                if typ=='h45':
                    bad_grid=(off[cond]>0)!=truth[None]
                    assert int(bad_grid.sum())==report['receiver_seeds'][seed][typ][cname]['offgrid_errors']
    audit['neural_audit_pass']=audit['bit_disagreements']==0 and audit['decision_disagreements']==0
    (out/'h47_independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('all_scores_checked','bit_disagreements','decision_disagreements','worst_abs_score_drift','neural_audit_pass')},indent=2))
    if not audit['neural_audit_pass']:raise AssertionError('independent CPU replay mismatch')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.out.resolve())
