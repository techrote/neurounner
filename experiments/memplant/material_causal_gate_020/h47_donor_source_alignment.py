#!/usr/bin/env python3
"""DG020 H47 exposed-only donor-channel alignment pilot (CPU).

Unsupervised profile-endpoint affine normalization from *genuine trained donor probes*:
  align_p(c)=mu_0+(c-c_p0)*(mu_1-mu_0)/(c_p1-c_p0).
No physical-law targets, oracle margins, world IDs or actions used to fit align_p.
The paired probes themselves identify low/high donor endpoints; this is a
TWO-CHECKPOINT, source-dependent calibration, not one-donor inheritance.
The law/world oracle is used for evaluation only; receiver receives cost scalar.
H40 worlds previously exposed. No checkpoint is retrained or changed.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from h45_independent_neural_replay import probe_donors, Receiver, REC, SELECTED, REC_HASH, sha

C0, C1 = .35, 2.30
ALPHAS = np.asarray([j/256 for j in range(1,256) if j%4],dtype='f8')
assert len(ALPHAS)==192

def make_costs(donors):
    # Baseline H42-H46 trained-donor route cost convention.
    raw={}
    for name,z in donors.items():
        baseline=float(z[0][0])
        raw[name]=(float(z[0][1])-baseline+1.,
                   float(z[1][1])-baseline+1.)
    keys=sorted(raw)
    mu0=float(np.mean([raw[k][0] for k in keys],dtype='f8'))
    mu1=float(np.mean([raw[k][1] for k in keys],dtype='f8'))
    return raw, (mu0,mu1)

def align_scalar(cost, pair, common):
    lo,hi=pair; a,b=common
    return float(np.float32(a+(float(cost)-lo)*(b-a)/(hi-lo)))

def score(model,obs,cost):
    wet=obs[...,0]>.5;fee=obs[...,1]
    d=torch.where(wet,torch.tensor(cost,dtype=torch.float32)+fee,1.+fee)
    with torch.inference_mode():
        logits=model(obs,d)
    margin=(logits[:,1]-logits[:,0]).numpy().astype('f4')
    return margin

def eval_model_set(root,seed,obs,rr,raw,common,weights_label,path,output_arrays):
    m=Receiver().eval(); m.load_state_dict(torch.load(root/path,weights_only=True,map_location='cpu'))
    assert sum(p.numel() for p in m.parameters())==209
    keys=sorted(raw)
    fee=obs[...,1];wet=obs[...,0]>.5
    dw=wet[:,0].sum(-1).numpy()-wet[:,1].sum(-1).numpy()
    df=fee[:,0].numpy().sum(-1,dtype='f8')-fee[:,1].numpy().sum(-1,dtype='f8')
    oracle_ep=np.asarray([r['oracle_actions'] for r in rr],dtype='i1')
    oracle_calc=((df[None]+dw[None]*(np.asarray([C0,C1])[:,None]-1.))>0).astype('i1').T
    assert np.array_equal(oracle_ep,oracle_calc)
    invariant=oracle_ep[:,0]==oracle_ep[:,1];divergent=~invariant
    assert int(invariant.sum())==960 and int(divergent.sum())==640
    endpoints=np.empty((2,2,4,1600),dtype='f4') # raw/aligned, cheap/exp, profile,world
    for k,name in enumerate(keys):
        for law in (0,1):
            c=raw[name][law]
            endpoints[0,law,k]=score(m,obs,c)
            endpoints[1,law,k]=score(m,obs,align_scalar(c,raw[name],common))
    expected=oracle_ep.T[:,None,:]
    outcomes={}
    for cidx,cname in enumerate(['raw','aligned']):
        pred=(endpoints[cidx]>0).astype('i1')
        correct=pred==expected
        # Both endpoint actions must be correct for each genuine paired donor profile.
        all_correct=correct.all(axis=0)
        inv_wrong=int((~correct[:,:,invariant]).sum())
        div_pair_wrong=int((~all_correct[:,divergent]).sum())
        sham=int((pred!=pred[:,:1,:]).sum())
        inv_flip=int((pred[0,:,invariant]!=pred[1,:,invariant]).sum())
        # Neutral and actual all-zero cost_tensors both law-invariant by construction.
        outcomes[cname]={
          'endpoint_action_errors':int((~correct).sum()),
          'endpoint_both_correct_divergent':int(all_correct[:,divergent].sum()),
          'endpoint_both_correct_invariant':int(all_correct[:,invariant].sum()),
          'divergent_pair_wrong':div_pair_wrong,
          'invariant_endpoint_action_errors':inv_wrong,
          'invariant_law_induced_flips':inv_flip,
          'same_law_donor_source_action_disagreements':sham,
          'min_abs_route_score':float(np.min(np.abs(endpoints[cidx]))),
        }
    output_arrays[f'{weights_label}_{seed}_endpoint_margins']=endpoints
    # H45 candidate off-grid only; H40 is endpoint baseline, not trained for synthetic alphas.
    if weights_label=='h45':
        off=np.empty((2,4,192,1600),dtype='f4')
        oracle=((df[None]+dw[None]*(C0+(C1-C0)*ALPHAS[:,None]-1.))>0).astype('i1')
        for k,name in enumerate(keys):
            c0,c1=raw[name]
            for j,a in enumerate(ALPHAS):
                c=float(np.float32((1-a)*c0+a*c1))
                off[0,k,j]=score(m,obs,c)
                off[1,k,j]=score(m,obs,align_scalar(c,raw[name],common))
        for cidx,cname in enumerate(['raw','aligned']):
            prediction=(off[cidx]>0).astype('i1')
            bad=prediction!=oracle[None]
            errs=int(bad.sum());inv_bad=int(bad[:,:,invariant].sum());div_bad=int(bad[:,:,divergent].sum())
            outcomes[cname]['offgrid_errors']=errs
            outcomes[cname]['offgrid_invariant_errors']=inv_bad
            outcomes[cname]['offgrid_divergent_errors']=div_bad
            outcomes[cname]['offgrid_same_law_profile_disagreements']=int((prediction!=prediction[:1]).sum())
            outcomes[cname]['offgrid_error_by_profile']={keys[k]:int(bad[k].sum()) for k in range(4)}
            outcomes[cname]['offgrid_first_10_errors']=[{
                 'source':keys[int(k)],'alpha_n_over_256':int(round(float(ALPHAS[j])*256)),
                 'seed':seed,'index':int(i),'family':rr[int(i)]['family'],
                 'bucket':rr[int(i)]['bucket'],'oracle':int(oracle[j,i]),
                 'pred':int(prediction[k,j,i]),'route_score':float(off[cidx,k,j,i])
               } for k,j,i in zip(*np.nonzero(bad))][:10]
        output_arrays[f'{weights_label}_{seed}_offgrid_margins']=off
    return outcomes

def main(root,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    assert sha(root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl')==REC_HASH
    rows=[json.loads(s) for s in (root/'h40-heldout/h40-unseen-genuine-neural-receipts.jsonl').open()]
    assert len(rows)==4800
    donors=probe_donors(root)
    raw,common=make_costs(donors)
    output_arrays={}
    report={
      'experiment':'DG020 H47 exposed-only donor-source affine alignment pilot',
      'scientific_gate':'NOT NEWLY EVALUATED: H40 historical restricted 30/30 PASS',
      'method':'per source two-endpoint affine alignment to unlabelled pooled trained-donor mean cost endpoints; no physical truth or receiver labels in calibration',
      'limitations':'uses paired low/high donor checkpoints per source; the probe is engineered material-specific; exposed H40 worlds; synthetic intermediate alpha is not a trained donor and labels are not an independently identifiable target',
      'h40_receipt_sha256':REC_HASH,
      'genuine_donor_checkpoint_sha256':{z['checkpoint']:z['checkpoint_sha256'] for z in json.loads((root/'h40-heldout/h34-donor-training.json').read_text())['trained']},
      'raw_genuine_donor_endpoint_costs':raw,
      'common_unlabelled_pooled_endpoint_costs':common,
      'profile_alignment_parameters':{k:{'scale':(common[1]-common[0])/(v[1]-v[0]),'offset':common[0]-v[0]*(common[1]-common[0])/(v[1]-v[0])} for k,v in raw.items()},
      'receiver_seeds':{},
    }
    for seed,(original,old_sha) in REC.items():
        assert sha(root/original)==old_sha
        rr=[r for r in rows if str(r['seed'])==seed];assert len(rr)==1600
        obs=torch.from_numpy(np.asarray([r['world_observation'] for r in rr],dtype='f4'))
        report['receiver_seeds'][seed]={
          'h40_original_frozen_receiver_sha256':old_sha,
          'h45_adjusted_receiver_sha256':sha(root/SELECTED[seed]),
          'h40':eval_model_set(root,seed,obs,rr,raw,common,'h40',original,output_arrays),
          'h45':eval_model_set(root,seed,obs,rr,raw,common,'h45',SELECTED[seed],output_arrays),
        }
        print('H47 seed',seed,'H40',report['receiver_seeds'][seed]['h40'],'H45',
              {k:{q:v for q,v in d.items() if not q.endswith('errors') or isinstance(v,int)} for k,d in report['receiver_seeds'][seed]['h45'].items()},flush=True)
    # Strict negative gate: all original end decisions and invariant selectivity retained for both receiver variants,
    # and all off-grid synthetic errors eliminated. Synthetic grid cannot prove genuine new physical law.
    report['h47_trainingonly_repair_pass']=all(
        (all(all(report['receiver_seeds'][s][typ]['aligned'][key]==0
                 for key in ['endpoint_action_errors','invariant_law_induced_flips','same_law_donor_source_action_disagreements'])
             for typ in ['h40','h45'])
         and report['receiver_seeds'][s]['h45']['aligned']['offgrid_errors']==0)
        for s in REC)
    np.savez_compressed(out/'h47_all_float32_neural_route_margins.npz',**output_arrays)
    report['all_raw_neural_route_margins_sha256']=sha(out/'h47_all_float32_neural_route_margins.npz')
    (out/'h47_results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('H47 training-only strict gate:', 'PASS' if report['h47_trainingonly_repair_pass'] else 'FAIL')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.out.resolve())
