"""DG020 H10: preregistered frozen receiver, changed wet-count and fee regimes.

Research only. No model training. Run from same directory as H9 source and with the
original H8 trained checkpoints staged by the existing H9 workflow. Never import
oracle costs or family/bucket metadata into inference.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import h9_cross_seed_audit as base

SEEDS = base.SEEDS
FAMILIES = ('lower_tail_fees', 'upper_tail_fees')
BUCKETS = base.BUCKETS
N = 160
LAWS = base.LAWS
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'h10-evidence'
THRESHOLDS = dict(divergent_both=.90, divergent_flip=.95,
                  invariant_both=.94, invariant_flip_max=.05,
                  each_law=.94, same_law_sham_max=.05,
                  actual_preinference_shuffled_both_max=.60,
                  raw_donor_mse_max=.02, illegal_max=0)


def seed_for(receiver_seed, family, bucket):
    return receiver_seed + 107_000_000 + FAMILIES.index(family)*170_011 + BUCKETS.index(bucket)*15_113


def material_pattern(rng, count):
    # Position placement is deliberately irrelevant to the new scientific axis.
    order = rng.permutation(12)
    wet = np.zeros(12, dtype='f4'); wet[order[:count]] = 1.0
    return wet


def make_fee(rng, target_mean, family):
    # Fix the target total before casting. Fee distributions differ in skew,
    # not merely within-route position permutation.
    if family == 'lower_tail_fees':
        samples = rng.beta(.55, 2.5, size=12)
    else:
        samples = rng.beta(2.5, .55, size=12)
    centered = samples - samples.mean()
    spread = max(0., min(float(target_mean), .30-float(target_mean)))
    if spread <= 0: return None
    maxabs = max(abs(float(centered.min())), abs(float(centered.max())))
    return (float(target_mean) + .85*spread*centered/maxabs).astype('f4')


def make_world(rng, family, bucket):
    rich = int(rng.integers(2)); poor = 1-rich
    count0 = int(rng.integers(2, 9))
    counts = [count0, count0]
    if bucket != 'equal_wet_invariant': counts[rich] += 2
    wet = np.stack([material_pattern(rng, n) for n in counts])
    edge = float(rng.uniform(.05, .20))
    if bucket == 'cheap_edge_divergent': fee_delta = 1.30-edge
    elif bucket == 'expensive_edge_divergent': fee_delta = -2.60+edge
    elif bucket == 'wet_rich_invariant': fee_delta = -2.60-edge
    elif bucket == 'wet_poor_invariant': fee_delta = 1.30+edge
    else: fee_delta = edge * (-1 if int(rng.integers(2)) else 1)
    fees = np.zeros((2,12), dtype='f4')
    fee_rich = .15+fee_delta/24
    fee_poor = .15-fee_delta/24
    fees[rich] = make_fee(rng,fee_rich,family)
    fees[poor] = make_fee(rng,fee_poor,family)
    assert np.isfinite(fees).all() and (fees>=0).all() and (fees<=.30).all()
    tex = rng.uniform(-1,1,(2,12,2)).astype('f4')
    obs = np.concatenate([wet[...,None], fees[...,None],tex],axis=-1)
    o = [base.oracle(obs,k) for k in (0,1)]
    a,b = o[0][0],o[1][0]
    if bucket.endswith('divergent'): assert (a,b)==(rich,poor),(bucket,a,b)
    elif bucket=='wet_rich_invariant': assert a==b==rich
    elif bucket=='wet_poor_invariant': assert a==b==poor
    else: assert a==b
    checked = {'cheap_edge_divergent':o[0][1], 'expensive_edge_divergent':o[1][1],
               'wet_rich_invariant':o[1][1], 'wet_poor_invariant':o[0][1],
               'equal_wet_invariant':o[0][1]}[bucket]
    assert .049 <= checked <= .201, (bucket,checked)
    return obs


def generate(seed, family, bucket):
    rng=np.random.default_rng(seed_for(seed,family,bucket))
    return np.stack([make_world(rng,family,bucket) for _ in range(N)]).astype('f4')


def eval_cell(receiver_seed,family,bucket,receivers,donors,blinds,old_hashes,seen):
    worlds=generate(receiver_seed,family,bucket)
    hashes=[hashlib.sha256(w.tobytes()).hexdigest() for w in worlds]
    assert len(set(hashes))==N and not (set(hashes)&seen) and not (set(hashes)&old_hashes)
    seen.update(hashes)
    x=torch.from_numpy(worlds)
    oracle_actions=np.array([[base.oracle(w,k)[0] for k in (0,1)] for w in worlds])
    oracle_costs=[[base.oracle(w,k)[2] for k in (0,1)] for w in worlds]
    gap=np.array([[base.oracle(w,k)[1] for k in (0,1)] for w in worlds])
    with torch.no_grad():
        raw=[[base.forecast(donors[receiver_seed][law][rep],x) for rep in (0,1)] for law in (0,1)]
        cond=[[base.median_condition(x,raw[law][rep]) for rep in (0,1)] for law in (0,1)]
        logits=[[receivers[receiver_seed](x,cond[law][rep]).numpy() for rep in (0,1)] for law in (0,1)]
        zero=receivers[receiver_seed](x,torch.zeros_like(cond[0][0])).argmax(-1).numpy()
        blind=blinds[receiver_seed](x,torch.zeros_like(cond[0][0])).argmax(-1).numpy()
    acts=np.stack([logits[law][0].argmax(-1) for law in (0,1)],axis=-1)
    shams=np.stack([logits[law][1].argmax(-1) for law in (0,1)],axis=-1)
    comparator=np.stack([cond[law][0].numpy().sum(-1).argmin(-1) for law in (0,1)],axis=-1)
    # Genuine pre-inference shuffle: each law slot randomly receives the trained
    # donor forecast belonging to law 0 or 1, then the *receiver is rerun*.
    rng=np.random.default_rng(seed_for(receiver_seed,family,bucket)+4040)
    selected=rng.integers(0,2,size=(N,2))
    mixed=[]
    with torch.no_grad():
        for slot in (0,1):
            chosen=torch.where(torch.tensor(selected[:,slot],dtype=torch.bool)[:,None,None],cond[1][0],cond[0][0])
            mixed.append(receivers[receiver_seed](x,chosen).argmax(-1).numpy())
    shuffled=np.stack(mixed,axis=-1)
    mse=[]
    for law in (0,1):
        tile=np.where(worlds[...,0]>.5,LAWS[law],1.)+worlds[...,1]
        for rep in (0,1): mse.extend(((raw[law][rep].numpy()-tile)**2).mean(axis=(1,2)).tolist())
    both=(acts==oracle_actions).all(-1)
    flips=acts[:,0]!=acts[:,1]
    sham=(acts!=shams).any(-1)
    surrogate_both=(comparator==oracle_actions).all(-1)
    shuffle_both=(shuffled==oracle_actions).all(-1)
    comparator_neural_disagree=(acts!=comparator).any(-1)
    counts=dict(n=N, both=int(both.sum()),flip=int(flips.sum()),law0=int((acts[:,0]==oracle_actions[:,0]).sum()),
                law1=int((acts[:,1]==oracle_actions[:,1]).sum()),sham=int(sham.sum()),
                shuffled_preinference_both=int(shuffle_both.sum()),comparator_both=int(surrogate_both.sum()),
                neural_comparator_disagreements=int(comparator_neural_disagree.sum()),
                zero_both=int(((zero==oracle_actions[:,0])&(zero==oracle_actions[:,1])).sum()),
                source_blind_both=int(((blind==oracle_actions[:,0])&(blind==oracle_actions[:,1])).sum()),
                illegal=int((~np.isin(acts,(0,1))).any(-1).sum()),max_donor_mse=max(mse))
    isdiv=bucket.endswith('divergent')
    p=bool(counts['both']/N >= (THRESHOLDS['divergent_both'] if isdiv else THRESHOLDS['invariant_both'])
           and (counts['flip']/N>=THRESHOLDS['divergent_flip'] if isdiv else counts['flip']/N<=THRESHOLDS['invariant_flip_max'])
           and min(counts['law0'],counts['law1'])/N>=THRESHOLDS['each_law']
           and counts['sham']/N<=THRESHOLDS['same_law_sham_max']
           and (not isdiv or counts['shuffled_preinference_both']/N<=THRESHOLDS['actual_preinference_shuffled_both_max'])
           and counts['max_donor_mse']<=THRESHOLDS['raw_donor_mse_max'] and counts['illegal']==0)
    filename=f'receipts-{receiver_seed}-{family}-{bucket}.jsonl'
    with (OUT/filename).open('w') as f:
        for i in range(N):
            row=dict(receiver_seed=receiver_seed,family=family,bucket=bucket,index=i,
                observation_sha256=hashes[i],observation=worlds[i].tolist(),legal_mask=[1,1],
                oracle_actions=oracle_actions[i].tolist(),oracle_route_costs=oracle_costs[i],oracle_gaps=gap[i].tolist(),
                actions=acts[i].tolist(),same_law_replica_actions=shams[i].tolist(),
                preinference_donor_selection=selected[i].tolist(),preinference_shuffled_actions=shuffled[i].tolist(),
                cost_comparator_actions=comparator[i].tolist(),zero_action=int(zero[i]),source_blind_action=int(blind[i]),
                logits=[logits[law][0][i].tolist() for law in (0,1)],
                raw_donor_forecasts=[raw[law][0][i].numpy().tolist() for law in (0,1)],
                conditioned_donor_forecasts=[cond[law][0][i].numpy().tolist() for law in (0,1)],
                same_law_replica_logits=[logits[law][1][i].tolist() for law in (0,1)],
                sham_raw_donor_forecasts=[raw[law][1][i].numpy().tolist() for law in (0,1)],
                both_correct=bool(both[i]),required_flip=bool(flips[i]),
                comparator_neural_disagreement=bool(comparator_neural_disagree[i]))
            f.write(json.dumps(row,separators=(',',':'))+'\n')
    return dict(verdict='PASS' if p else 'FAIL',counts=counts,
                failed_indices=[int(i) for i in range(N) if not both[i] or sham[i]][:40])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    receivers,donors,blinds,model_hashes=base.load_models()
    replay=base.baseline_replay(receivers,donors)
    assert replay['action_disagreement_count']==0,replay
    old_hashes=set()
    # The previous untouched H8 worlds are supplied by the same H9 CI fixture.
    for seed in SEEDS:
        with __import__('zipfile').ZipFile(base.BASE/f'DG020-H8-exact-head-seed{str(seed)[-2:]}.zip') as z:
            for path in z.namelist():
                if path.startswith('receipts-') and path.endswith('.jsonl'):
                    old_hashes.update(json.loads(s)['observation_sha256'] for s in z.read(path).splitlines())
    # H9 archived hashes are a second set; complete source/receipt archives kept.
    h9_archive=base.BASE/'DG020-H9-CI-44051553537.zip'
    if h9_archive.exists():
        with __import__('zipfile').ZipFile(h9_archive) as z:
            for path in z.namelist():
                if '/receipts-receiver' in path and path.endswith('.jsonl'):
                    old_hashes.update(json.loads(s)['observation_sha256'] for s in z.read(path).splitlines())
    for seed in SEEDS:
        for family in base.FAMILIES:
            for bucket in base.BUCKETS:
                x,_=base.fresh_worlds(seed,family,bucket)
                old_hashes.update(hashlib.sha256(w.tobytes()).hexdigest() for w in x)
    seen=set();results={}
    for seed in SEEDS:
        results[str(seed)]={}
        for family in FAMILIES:
            results[str(seed)][family]={}
            for bucket in BUCKETS:
                r=eval_cell(seed,family,bucket,receivers,donors,blinds,old_hashes,seen)
                results[str(seed)][family][bucket]=r
                print(seed,family,bucket,r['verdict'],r['counts'],flush=True)
    cells=[x for s in results.values() for f in s.values() for x in f.values()]
    summary=dict(experiment='DG020 H10 unseen fee-count regime frozen receiver challenge',
                 scope='fixed 2x12 two scalar laws; fresh wet count delta=2 and new fee distributions',
                 status='PASS' if all(c['verdict']=='PASS' for c in cells) else 'FAIL',
                 method='pinned CI if run under GitHub Actions; local is not exact-head',
                 preregistration='https://github.com/techrote/neurounner/pull/253#DG020-H10-prereg',
                 thresholds=THRESHOLDS,seeds=SEEDS,families=FAMILIES,buckets=BUCKETS,
                 unique_worlds=len(seen),prior_observation_hashes_checked=len(old_hashes),
                 total_cells=len(cells),passed_cells=sum(c['verdict']=='PASS' for c in cells),
                 h8_replay=replay,model_sha256=model_hashes,
                 torch_version=torch.__version__,numpy_version=np.__version__,results=results)
    (OUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    digests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*')) if p.name!='sha256.json'}
    (OUT/'sha256.json').write_text(json.dumps(digests,indent=2)+'\n')
    print('H10 RESULT',summary['status'],summary['passed_cells'],'/',summary['total_cells'],flush=True)

if __name__=='__main__':main()
