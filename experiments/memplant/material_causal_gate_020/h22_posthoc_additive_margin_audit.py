#!/usr/bin/env python3
"""Read-only H22 post-hoc additive-law score-advantage audit.

Needs the *hash-verified original H22 compact capsule* extracted in cwd, or
--capsule-dir pointing to its extracted root. Tests only original worlds and
frozen genuine PyTorch donors/receivers. Never regenerates worlds, refits
checkpoints or changes the original 30-cell PASS verdict.
"""
import argparse
import collections
import hashlib
import tempfile
import zipfile
import stat
import json
from pathlib import Path
import sys

import numpy as np
import torch

PINNED_SOURCE_SHA='a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d'
PINNED_CAPSULE_SHA='47546540b96ae0800697e8a729c169e1191f111cec8c09f704c99a8843568667'
LAWS=(.35,2.30)
SEEDS=(26011639,26011640,26011641)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(condition,msg):
    if not condition:raise AssertionError(msg)

def describe(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),
                p99=float(np.percentile(a,99)),maximum=float(a.max()))

def audit(root,out):
    src=root/'source'
    manifest={}
    for line in (root/'MANIFEST.sha256').read_text().splitlines():
        h,path=line.split(None,1)
        manifest[path.strip(' *')]=h
    for path,h in manifest.items():
        check(digest(root/path)==h,f'original capsule manifest mismatch: {path}')
    check(digest(src/'h22_wet_coordinate_only_confirmation.py')==PINNED_SOURCE_SHA,'original source changed')
    sys.path.insert(0,str(src))
    from experiment import Donor,Receiver
    report=json.loads((src/'h22-evidence'/'report.json').read_text())
    torch.set_num_threads(2)
    probes=np.zeros((2,25,4),dtype='f4')
    nodes,weights=np.polynomial.legendre.leggauss(5)
    weights=np.asarray([u*v/4 for u in weights for v in weights],dtype='f8')
    for m in (0,1):
        probes[m,:,0]=m
        probes[m,:,1]=.15
        probes[m,:,2:]=np.asarray([(x,y) for x in nodes for y in nodes],dtype='f4')
    learned={}
    with torch.no_grad():
        for law in (0,1):
            values=[]
            for rep in (0,1):
                f=src/'h21-evidence'/f'fresh-donor-29002117-law{law}-rep{rep}.pt'
                check(digest(f)==report['authentic_frozen_donor_checkpoint_sha256'][f.name],'donor identity')
                model=Donor().eval()
                model.load_state_dict(torch.load(f,map_location='cpu',weights_only=True))
                values.append((model(torch.from_numpy(probes)).numpy().astype('f8')@weights-.15).astype('f4'))
            learned[law]=np.mean(values,axis=0,dtype='f4')
        check(abs(float(learned[0][0])-float(report['learned_material_values']['29002117']['0'][0]))<1e-6,'dry probe')
        check(abs(float(learned[0][1])-float(report['learned_material_values']['29002117']['0'][1]))<1e-6,'cheap wet probe')
        check(abs(float(learned[1][1])-float(report['learned_material_values']['29002117']['1'][1]))<1e-6,'expensive wet probe')
        receivers={}
        for seed in SEEDS:
            ck=src/'h19-evidence'/f'receiver-quadrature-{seed}.pt'
            check(digest(ck)==report['authentic_frozen_receiver_checkpoint_sha256'][str(seed)],'receiver identity')
            net=Receiver().eval()
            net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True))
            receivers[seed]=net
    groups=collections.defaultdict(lambda:dict(pairs=0,wrong=0,wrong_flips=0,
                   equal_wet=0,nonzero_margin_signal=0))
    all_abs_error=[]
    all_abs_slope_error=[]
    equal_abs_spurious=[]
    ideal_feature_action_disagreements=0
    ideal_feature_equal_spurious=[]
    negatives=[]
    observed_hashes=set()
    max_logit_drift=0.
    for filename in sorted((src/'h22-evidence').glob('receipts-*.jsonl')):
        rows=[json.loads(l) for l in filename.read_text().splitlines()]
        check(len(rows)==160,'cell should contain 160 receipts')
        seed=rows[0]['seed'];family=rows[0]['family'];bucket=rows[0]['bucket']
        check(seed in SEEDS,'unknown receiver seed')
        obs=np.asarray([r['observation'] for r in rows],dtype='f4')
        wet=obs[...,0]>.5;fee=obs[...,1]
        wet_counts=wet.sum(-1).astype('f8')
        wet_count_diff=wet_counts[:,0]-wet_counts[:,1]
        # Pair identity: original source records are hashed after float32 packing.
        for i,r in enumerate(rows):
            h=hashlib.sha256(obs[i].tobytes()).hexdigest()
            check(r['observation_sha256']==h and h not in observed_hashes,'receipt integrity/uniqueness')
            observed_hashes.add(h)
            check((r['seed'],r['family'],r['bucket'],r['index'])==(seed,family,bucket,i),'receipt indexing')
        ideal_cost=np.stack([(np.where(wet,v,1.)+fee).sum(-1,dtype='f8') for v in LAWS],axis=1)
        oracle=ideal_cost.argmin(-1)
        score=[]
        for law in (0,1):
            feature=(np.where(wet,learned[law][1],learned[0][0])+fee).astype('f4')
            with torch.no_grad():score.append(receivers[seed](torch.from_numpy(obs),torch.from_numpy(feature)).numpy())
        score=np.stack(score,axis=1)
        neural=score.argmax(-1)
        # Diagnostic ceiling: exact known material scalars, NOT another genuine donor.
        # This intervention is post-hoc only and never used for acceptance.
        ideal_feature_scores=[]
        for law in (0,1):
            feature=(np.where(wet,LAWS[law],1.)+fee).astype('f4')
            with torch.no_grad():
                ideal_feature_scores.append(receivers[seed](torch.from_numpy(obs),torch.from_numpy(feature)).numpy())
        ideal_feature_scores=np.stack(ideal_feature_scores,axis=1)
        ideal_feature_action_disagreements+=int(np.count_nonzero(ideal_feature_scores.argmax(-1)!=neural))
        check(np.array_equal(neural,np.asarray([r['actions'] for r in rows])),f'checkpoint action drift: {filename}')
        check(np.array_equal(oracle,np.asarray([r['oracle_actions'] for r in rows])),f'oracle drift: {filename}')
        max_logit_drift=max(max_logit_drift,float(np.max(np.abs(score-np.asarray([r['logits'] for r in rows])))))
        # Signed cost advantage: positive -> route 1, negative -> route 0.
        model_adv=score[:,:,1]-score[:,:,0]
        ideal_adv=ideal_cost[:,:,0]-ideal_cost[:,:,1]
        # Model scores are negative estimated route cost, so delta model score_1-score_0 = ideal cost_0-cost_1.
        check(np.array_equal(oracle,(ideal_adv>0).astype(int)),'tie/sign assumption wrong')
        err=model_adv-ideal_adv
        absolute=np.abs(err)
        slope_observed=model_adv[:,1]-model_adv[:,0]
        slope_expected=wet_count_diff*(LAWS[1]-LAWS[0])
        slope_error=slope_observed-slope_expected
        ideal_feature_adv=ideal_feature_scores[:,:,1]-ideal_feature_scores[:,:,0]
        ideal_feature_slope=ideal_feature_adv[:,1]-ideal_feature_adv[:,0]
        ideal_feature_equal_spurious.extend(np.abs(ideal_feature_slope[wet_count_diff==0]).tolist())
        all_abs_error.extend(absolute.ravel().tolist())
        all_abs_slope_error.extend(np.abs(slope_error).tolist())
        g=groups[str(seed)+'/'+family+'/'+bucket]
        g['pairs']+=len(rows)
        g['wrong']+=int(np.any(neural!=oracle,axis=1).sum())
        g['wrong_flips']+=int(np.logical_and(neural[:,0]!=neural[:,1],oracle[:,0]==oracle[:,1]).sum())
        same=wet_count_diff==0
        g['equal_wet']+=int(same.sum())
        g['nonzero_margin_signal']+=int(np.count_nonzero(np.abs(slope_observed)>0.0001))
        for i,r in enumerate(rows):
            if same[i]:
                equal_abs_spurious.append(abs(float(slope_observed[i])))
            if np.any(neural[i]!=oracle[i]):
                negatives.append(dict(seed=seed,family=family,bucket=bucket,index=i,
                    observation_sha256=r['observation_sha256'],wet_count_difference=int(wet_count_diff[i]),
                    oracle=oracle[i].tolist(),neural=neural[i].tolist(),
                    oracle_advantages=ideal_adv[i].tolist(),model_advantages=model_adv[i].tolist(),
                    margin_residuals=err[i].tolist(),law_delta_advantage=float(slope_observed[i]),
                    law_delta_expected=float(slope_expected[i]),law_slope_error=float(slope_error[i])))
    check(len(observed_hashes)==4800 and len(groups)==30,'cohort size changed')
    check(len(negatives)==3,'original H22 errors not preserved')
    check(max_logit_drift<5e-5,'neural logit mismatch')
    result=dict(status='POSTHOC_DIAGNOSTIC_ONLY',scientific_verdict_original='PASS_30_OF_30_UNCHANGED',
        diagnostic_verdict='FAIL_STRICT_ADDITIVE_ADVANTAGE_INVARIANCE',
        capsule_source_sha256=PINNED_SOURCE_SHA, original_manifest_sha256=digest(root/'MANIFEST.sha256'),
        receipt_pairs=len(observed_hashes),cells=len(groups),genuine_donors=4,genuine_receivers=3,
        original_invariant_wrong_action_flips=len(negatives),
        max_neural_logit_drift=max_logit_drift,
        overall_abs_route_advantage_error=describe(all_abs_error),
        overall_abs_law_response_error=describe(all_abs_slope_error),
        equal_wet_abs_spurious_law_response=describe(equal_abs_spurious),
        exact_cost_diagnostic=dict(action_disagreements_from_original=ideal_feature_action_disagreements,
            equal_wet_abs_spurious_law_response=describe(ideal_feature_equal_spurious),
            equal_wet_response_above_1e_4=sum(v>1e-4 for v in ideal_feature_equal_spurious)),
        equal_wet_response_above_1e_4=sum(v>1e-4 for v in equal_abs_spurious),
        exact_wrong_cases=negatives,per_cell=dict(sorted(groups.items())))
    if out:
        out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_cell'},indent=2))
    return 0

def main():
    p=argparse.ArgumentParser()
    s=p.add_mutually_exclusive_group(required=True)
    s.add_argument('--capsule',type=Path,help='Exact pinned original Git-backed H22 replay capsule ZIP')
    s.add_argument('--capsule-dir',type=Path,help='Already verified and extracted capsule root')
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    if a.capsule_dir:
        return audit(a.capsule_dir,a.out)
    check(digest(a.capsule)==PINNED_CAPSULE_SHA,'pinned capsule archive sha mismatch')
    with tempfile.TemporaryDirectory(prefix='dg020-h22-posthoc-') as scratch:
        root=Path(scratch)
        with zipfile.ZipFile(a.capsule) as z:
            names=set()
            for info in z.infolist():
                name=info.filename
                parts=Path(name).parts
                check(not name.startswith(('/', '\\')) and '\\' not in name and '..' not in parts and name not in names,'unsafe/duplicate capsule path')
                check(not stat.S_ISLNK((info.external_attr>>16)&0xffff),'symlink capsule entry')
                names.add(name)
            z.extractall(root)
        return audit(root,a.out)

if __name__=='__main__':raise SystemExit(main())
