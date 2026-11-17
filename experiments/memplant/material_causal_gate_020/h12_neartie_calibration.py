"""DG020 H12: preregistered receiver-only near-tie calibration and selectivity gate.

Same 2x12 two-law physics, frozen H6 trained donor replicas and same-size trained
Receiver. H11 is exposed historical FAIL, never acceptance. No oracle/family/law
identity, target delta, or labels enter model inference. CPU only, research only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

import h9_cross_seed_audit as base
import h10_fee_count_stress as evaluator
from experiment import Receiver

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'h12-evidence'
SEEDS = base.SEEDS
BUCKETS = base.BUCKETS
TRAIN_FAMILIES = ('central_plateau_train', 'bimodal_plateau_train')
EVAL_FAMILIES = ('left_skew_holdout', 'right_skew_holdout')
TRAIN_PER_CELL = 400
EVAL_PER_CELL = 160
UPDATES = 1400
THRESHOLDS = dict(divergent_both=.90, divergent_flip=.95,
                  invariant_both=.94, invariant_flip_max=.05,
                  each_law=.94, same_law_sham_max=.05,
                  actual_preinference_shuffled_both_max=.60,
                  raw_donor_mse_max=.02, illegal_max=0)
BETA = {'central_plateau_train':(1.7,1.7), 'bimodal_plateau_train':(.65,.65),
        'left_skew_holdout':(.85,2.9), 'right_skew_holdout':(2.9,.85)}


def sha(data): return hashlib.sha256(data).hexdigest()
def sha_file(path): return sha(Path(path).read_bytes())


def seed_for(seed, family, bucket, train=False):
    families = TRAIN_FAMILIES if train else EVAL_FAMILIES
    return seed + (183_000_000 if train else 197_000_000) + families.index(family)*277_013 + BUCKETS.index(bucket)*31_027


def fee_profile(rng, target, family):
    a,b=BETA[family]
    samples=rng.beta(a,b,size=12)
    centered=samples-samples.mean()
    span=max(0.,min(float(target),.30-float(target)))
    assert span>0, (family,target)
    return (target+.85*span*centered/max(abs(centered.min()),abs(centered.max()))).astype('f4')


def make_world(rng,family,bucket,train=False):
    rich=int(rng.integers(2)); poor=1-rich
    count=int(rng.integers(2,9))
    counts=[count,count]
    if bucket!='equal_wet_invariant':counts[rich]+=2
    wet=np.zeros((2,12),dtype='f4')
    for route in (0,1):wet[route,rng.permutation(12)[:counts[route]]]=1.
    if train:
        margin=float(rng.uniform(.015,.085) if rng.random()<.80 else rng.uniform(.085,.35))
    else:
        margin=float(rng.uniform(.015,.070))
    if bucket=='cheap_edge_divergent':delta=1.30-margin
    elif bucket=='expensive_edge_divergent':delta=-2.60+margin
    elif bucket=='wet_rich_invariant':delta=-2.60-margin
    elif bucket=='wet_poor_invariant':delta=1.30+margin
    else:delta=margin*(1 if int(rng.integers(2)) else -1)
    fees=np.zeros((2,12),dtype='f4')
    fees[rich]=fee_profile(rng,.15+delta/24.,family)
    fees[poor]=fee_profile(rng,.15-delta/24.,family)
    assert fees.min()>=0 and fees.max()<=.30
    texture=rng.uniform(-1,1,size=(2,12,2)).astype('f4')
    world=np.concatenate((wet[...,None],fees[...,None],texture),axis=-1)
    oa=[base.oracle(world,k) for k in (0,1)]
    if bucket.endswith('divergent'):assert (oa[0][0],oa[1][0])==(rich,poor)
    elif bucket=='wet_rich_invariant':assert oa[0][0]==oa[1][0]==rich
    elif bucket=='wet_poor_invariant':assert oa[0][0]==oa[1][0]==poor
    else:assert oa[0][0]==oa[1][0]
    return world


def generate(seed,family,bucket,train=False,n=EVAL_PER_CELL):
    rng=np.random.default_rng(seed_for(seed,family,bucket,train))
    return np.stack([make_world(rng,family,bucket,train) for _ in range(n)]).astype('f4')


def fit(seed, donors):
    """Train same Receiver(3->16->8->1) jointly on matched divergent/invariant pairs."""
    torch.manual_seed(seed+11_001)
    rng=np.random.default_rng(seed+27_001)
    worlds=np.concatenate([generate(seed,f,b,train=True,n=TRAIN_PER_CELL)
                           for f in TRAIN_FAMILIES for b in BUCKETS])
    assert len(worlds)==4000
    obs=torch.from_numpy(worlds)
    with torch.no_grad():
        pred=[[base.median_condition(obs,base.forecast(donors[seed][law][rep],obs))
               for rep in (0,1)] for law in (0,1)]
    costs=[torch.from_numpy(np.asarray([base.oracle(w,k)[2] for w in worlds],dtype='f4')) for k in (0,1)]
    targets=[torch.from_numpy((np.where(worlds[...,0]>.5,base.LAWS[k],1.)+worlds[...,1]).astype('f4')) for k in (0,1)]
    model=Receiver().train()
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0001)
    for step in range(UPDATES):
        ix=torch.from_numpy(rng.integers(0,len(obs),192))
        loss=torch.zeros((),dtype=torch.float32)
        for k in (0,1):
            truth_adv=costs[k][ix,1]-costs[k][ix,0]
            decision=(truth_adv<0).long()
            advantages=[]
            for rep in (0,1):
                cell=model.tile(torch.stack((obs[ix,...,0],obs[ix,...,1],pred[k][rep][ix]),-1)).squeeze(-1)
                logits=-cell.sum(-1)
                pred_adv=logits[:,0]-logits[:,1]
                advantages.append(pred_adv)
                # Near-tie regression receives stronger weight than H8's broad-margin fit.
                loss=loss+.25*(8.0*F.smooth_l1_loss(pred_adv,truth_adv,beta=.065)
                                 +.75*F.mse_loss(cell,targets[k][ix])
                                 +.10*F.cross_entropy(logits,decision))
            # Penalize route-score differences between independently trained same-law donors.
            loss=loss+2.0*F.mse_loss(advantages[0],advantages[1])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    model.eval()
    hashes={sha(w.tobytes()) for w in worlds}
    assert len(hashes)==len(worlds)
    return model,hashes


def previous_hashes():
    """Previously exposed acceptance cohorts must never be H12 acceptance."""
    names=('DG020-H8-exact-head-seed39.zip','DG020-H8-exact-head-seed40.zip',
           'DG020-H8-exact-head-seed41.zip','DG020-H9-CI-44051553537.zip',
           'DG020-H10-CI-44070049145.zip','DG020-H11-CI-44079138253.zip')
    known=set()
    for name in names:
        path=base.BASE/name
        assert path.exists(),('missing prior-cohort evidence',str(path))
        with zipfile.ZipFile(path) as z:
            for entry in z.namelist():
                if entry.endswith('.jsonl') and '/receipts-' in '/'+entry:
                    for line in z.read(entry).splitlines():
                        row=json.loads(line)
                        if 'observation_sha256' in row:known.add(row['observation_sha256'])
    for seed in SEEDS:
        for family in base.FAMILIES:
            for bucket in BUCKETS:
                worlds,_=base.fresh_worlds(seed,family,bucket)
                known.update(sha(x.tobytes()) for x in worlds)
    return known


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    _,donors,blinds,model_hashes=base.load_models()
    seen=set();old=previous_hashes();trained={};training_hashes=set()
    for seed in SEEDS:
        model,h=fit(seed,donors)
        assert not training_hashes.intersection(h) and not old.intersection(h)
        training_hashes.update(h);trained[seed]=model
        torch.save(model.state_dict(),OUT/f'receiver-neartie-{seed}.pt')
        (OUT/f'train-observation-hashes-{seed}.txt').write_text('\n'.join(sorted(h))+'\n')
        print('trained',seed,'worlds',len(h),flush=True)
    # The same fixed-size trained Receiver is evaluated with either learned donor;
    # h10 evaluator provides genuine pre-inference donor shuffles and raw receipts.
    evaluator.OUT=OUT
    evaluator.FAMILIES=EVAL_FAMILIES
    evaluator.THRESHOLDS=THRESHOLDS
    evaluator.seed_for=lambda seed,family,bucket:seed_for(seed,family,bucket,False)
    evaluator.generate=lambda seed,family,bucket:generate(seed,family,bucket,False,EVAL_PER_CELL)
    results={}
    for seed in SEEDS:
        results[str(seed)]={}
        for family in EVAL_FAMILIES:
            results[str(seed)][family]={}
            for bucket in BUCKETS:
                result=evaluator.eval_cell(seed,family,bucket,trained,donors,blinds,
                                           old.union(training_hashes),seen)
                results[str(seed)][family][bucket]=result
                print('H12',seed,family,bucket,result['verdict'],result['counts'],flush=True)
    cells=[x for s in results.values() for f in s.values() for x in f.values()]
    assert len(cells)==30 and len(seen)==4800
    report=dict(name='DG020 H12 receiver-only near-tie recalibration',
                status='PASS' if all(x['verdict']=='PASS' for x in cells) else 'FAIL',
                passed_cells=sum(x['verdict']=='PASS' for x in cells),total_cells=30,
                seeds=SEEDS,training_families=TRAIN_FAMILIES,heldout_families=EVAL_FAMILIES,
                buckets=BUCKETS,training_worlds=len(training_hashes),heldout_worlds=len(seen),
                training_updates=UPDATES,thresholds=THRESHOLDS,
                prior_hashes_checked=len(old),prior_acceptance_collisions=0,
                training_acceptance_collisions=0,model_sha256=model_hashes,
                receiver_checkpoint_sha256={str(k):sha_file(OUT/f'receiver-neartie-{k}.pt') for k in SEEDS},
                source_sha256=sha_file(__file__),torch_version=torch.__version__,
                numpy_version=np.__version__,results=results)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'sha256.json').write_text(json.dumps({p.name:sha_file(p) for p in sorted(OUT.iterdir())
                                               if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
    print('H12 SCIENTIFIC RESULT',report['status'],report['passed_cells'],'/30',flush=True)


if __name__=='__main__':main()
