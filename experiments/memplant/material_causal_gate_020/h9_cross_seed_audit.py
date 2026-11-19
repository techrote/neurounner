"""DG020 H9 independent frozen cross-seed donor-only audit, prereg #6942182942.
Research-only local exploratory qualification; source requires torch CPU and numpy.
No CyberSand/DRIV3R changes, no model training, no new topology or laws.
"""
import hashlib, json, math, zipfile, sys
from pathlib import Path
import numpy as np
import torch
from experiment import Donor, Receiver

SEEDS = (26011639, 26011640, 26011641)
FAMILIES = ('woven_perimeter', 'modular_bands')
BUCKETS = ('cheap_edge_divergent','expensive_edge_divergent','wet_rich_invariant','wet_poor_invariant','equal_wet_invariant')
LAWS = (.35, 2.30)
PER_CELL = 160
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'h9-evidence'
BASE = Path('/mnt/data')
THRESHOLDS = {'divergent_both': .90,'divergent_flip': .95,'invariant_both': .94,'invariant_flip_max': .05,'each_law': .94,'same_law_replica_sham_max': .05,'foreign_same_law_swap_max': .05,'shuffled_both_max': .60,'donor_mse_max': .02}


def sha(x): return hashlib.sha256(x).hexdigest()
def hashfile(p): return sha(Path(p).read_bytes())

def oracle(obs,law):
    wet=obs[...,0]>.5
    tile=np.where(wet,LAWS[law],1.0)+obs[...,1]
    cost=tile.sum(axis=-1,dtype=np.float64)
    return int(np.argmin(cost)),float(abs(cost[0]-cost[1])),cost.tolist()

def place_wet(rng,n,family):
    if family=='woven_perimeter':
        order=np.array([0,11,3,8,6,5,1,10,4,7,2,9],dtype=int)
    elif family=='modular_bands':
        order=np.array([0,4,8,2,6,10,1,5,9,3,7,11],dtype=int)
    else: raise AssertionError(family)
    order=(order+int(rng.integers(12)))%12
    if int(rng.integers(2)): order=(11-order)%12
    a=np.zeros(12,dtype='f4'); a[order[:n]]=1.
    assert len(set(order))==12 and int(a.sum())==n
    return a

def make_world(rng,family,bucket):
    k=int(rng.integers(2,10))
    rich=int(rng.integers(2));poor=1-rich
    counts=(k,k) if bucket=='equal_wet_invariant' else ((k+1,k) if rich==0 else (k,k+1))
    wet=np.stack([place_wet(rng,n,family) for n in counts])
    edge=float(rng.uniform(.05,.20))
    if bucket=='cheap_edge_divergent': fee_delta=.65-edge
    elif bucket=='expensive_edge_divergent': fee_delta=-1.30+edge
    elif bucket=='wet_rich_invariant': fee_delta=-1.30-edge
    elif bucket=='wet_poor_invariant': fee_delta=.65+edge
    else: fee_delta=edge*(1 if int(rng.integers(2)) else -1)
    fees=rng.uniform(.12,.18,(2,12)).astype('f8')
    # Fee-difference placement is independent from donor, receiver, and oracle.
    offset=(fee_delta-(fees[rich].sum()-fees[poor].sum()))/24.
    fees[rich]+=offset;fees[poor]-=offset
    if fees.min()<0 or fees.max()>.30: return None
    tex=rng.uniform(-1,1,(2,12,2)).astype('f4')
    obs=np.concatenate([wet[...,None],fees.astype('f4')[...,None],tex],axis=-1)
    a,ga,ca=oracle(obs,0);b,gb,cb=oracle(obs,1)
    if bucket.endswith('divergent'):
        assert a==rich and b==poor, (bucket,a,b)
    elif bucket=='wet_rich_invariant': assert a==b==rich
    elif bucket=='wet_poor_invariant':assert a==b==poor
    else: assert a==b
    observed={'cheap_edge_divergent':ga,'expensive_edge_divergent':gb,'wet_rich_invariant':gb,'wet_poor_invariant':ga,'equal_wet_invariant':ga}[bucket]
    assert .049<=observed<=.201, (bucket,observed)
    return obs

def fresh_worlds(receiver_seed,family,bucket):
    fi=FAMILIES.index(family);bi=BUCKETS.index(bucket)
    rng=np.random.default_rng(receiver_seed+81_000_000+fi*170001+bi*15111)
    arr=[];attempts=0
    while len(arr)<PER_CELL and attempts<PER_CELL*100:
        x=make_world(rng,family,bucket);attempts+=1
        if x is not None: arr.append(x)
    assert len(arr)==PER_CELL,(receiver_seed,family,bucket,len(arr),attempts)
    return np.asarray(arr,dtype='f4'),attempts

def load_models():
    receivers={};donors={};blinds={};hashes={}
    for seed in SEEDS:
        short=str(seed)[-2:]
        rpath=ROOT/f'receiver-{short}.pt'
        hashes[str(rpath.relative_to(ROOT))]=hashfile(rpath)
        r=Receiver().eval();r.load_state_dict(torch.load(rpath,map_location='cpu',weights_only=True));receivers[seed]=r
        donors[seed]={}
        for law in (0,1):
            donors[seed][law]=[]
            for rep in (0,1):
                path=ROOT/'checkpoints'/str(seed)/f'donor-{law}-replica-{rep}.pt'
                hashes[str(path.relative_to(ROOT))]=hashfile(path)
                model=Donor().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));donors[seed][law].append(model)
        path=ROOT/'checkpoints'/str(seed)/'receiver-source-blind.pt'
        hashes[str(path.relative_to(ROOT))]=hashfile(path)
        blind=Receiver().eval();blind.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));blinds[seed]=blind
    return receivers,donors,blinds,hashes

def forecast(d, x):
    return d(x.reshape(-1,4)).reshape(len(x),2,12)

def median_condition(x,raw):
    wet=(x[...,0]>.5)
    no_fee=(raw-x[...,1]).reshape(len(x),-1)
    mask=wet.reshape(len(x),-1)
    assert mask.any(dim=1).all() and (~mask).any(dim=1).all()
    missing=torch.full_like(no_fee,float('nan'))
    medwet=torch.nanmedian(torch.where(mask,no_fee,missing),dim=1).values
    meddry=torch.nanmedian(torch.where(~mask,no_fee,missing),dim=1).values
    return torch.where(wet,medwet[:,None,None],meddry[:,None,None])+x[...,1]

def baseline_replay(receivers,donors):
    errors=[];total=0;largest=0.
    for seed in SEEDS:
        z=zipfile.ZipFile(BASE/f'DG020-H8-exact-head-seed{str(seed)[-2:]}.zip')
        for family in ('interleave_edges','shifted_nested'):
            for bucket in BUCKETS:
                rows=[json.loads(s) for s in z.read(f'receipts-{family}-{bucket}.jsonl').splitlines()]
                x=torch.tensor(np.asarray([r['observation'] for r in rows],dtype='f4'))
                with torch.no_grad():
                    for law in (0,1):
                        f=median_condition(x,forecast(donors[seed][law][0],x))
                        logits=receivers[seed](x,f).numpy()
                        saved=np.asarray([r['logits'][law] for r in rows],dtype='f4')
                        largest=max(largest,float(np.abs(logits-saved).max()))
                        diff=np.where(logits.argmax(axis=-1)!=saved.argmax(axis=-1))[0]
                        if len(diff):errors.append(dict(seed=seed,family=family,bucket=bucket,law=law,indices=diff.tolist()))
                        total+=len(rows)
    return dict(decisions_replayed=total, max_abs_logit_error=largest,action_disagreement_count=sum(len(x['indices']) for x in errors),examples=errors[:5])

def evaluate(receiving_seed,family,bucket,worlds,receivers,donors,blinds,source_seed,writer):
    n=len(worlds);x=torch.from_numpy(worlds)
    oracle_actions=np.asarray([[oracle(w,k)[0] for k in (0,1)] for w in worlds],dtype=int)
    oracle_gaps=np.asarray([[oracle(w,k)[1] for k in (0,1)] for w in worlds],dtype=float)
    oracle_costs=[[oracle(w,k)[2] for k in (0,1)] for w in worlds]
    own=receiving_seed
    with torch.no_grad():
        raw=[[forecast(donors[source_seed][law][rep],x) for rep in (0,1)] for law in (0,1)]
        conditioned=[[median_condition(x,raw[law][rep]) for rep in (0,1)] for law in (0,1)]
        logits=[[receivers[receiving_seed](x,conditioned[law][rep]).numpy() for rep in (0,1)] for law in (0,1)]
        ownraw=[forecast(donors[own][law][0],x) for law in (0,1)]
        ownlogits=[receivers[receiving_seed](x,median_condition(x,ownraw[law])).numpy() for law in (0,1)]
        z=receivers[receiving_seed](x,torch.zeros_like(raw[0][0])).argmax(dim=-1).numpy()
        blind=blinds[receiving_seed](x,torch.zeros_like(raw[0][0])).argmax(dim=-1).numpy()
    acts=np.stack([logits[k][0].argmax(axis=-1) for k in (0,1)],axis=-1)
    shams=np.stack([logits[k][1].argmax(axis=-1) for k in (0,1)],axis=-1)
    ownacts=np.stack([ownlogits[k].argmax(axis=-1) for k in (0,1)],axis=-1)
    rng=np.random.default_rng(124000+receiving_seed+source_seed*17+FAMILIES.index(family)*100+BUCKETS.index(bucket)*1000)
    picks=rng.integers(0,2,size=(n,2))
    shuffled=np.stack([acts[np.arange(n),picks[:,k]] for k in (0,1)],axis=-1)
    both=(acts==oracle_actions).all(axis=-1)
    changed=(acts[:,0]!=acts[:,1])
    sham=(shams!=acts).any(axis=-1)
    foreign=(acts!=ownacts).any(axis=-1)
    shuffleboth=(shuffled==oracle_actions).all(axis=-1)
    donor_error=[]
    for k in (0,1):
        oracle_tile=np.where(worlds[...,0]>.5,LAWS[k],1.)+worlds[...,1]
        for r in (0,1):donor_error.extend(np.mean((raw[k][r].numpy()-oracle_tile)**2,axis=(1,2)).tolist())
    max_mse=float(max(donor_error))
    m=dict(both=int(both.sum()),flip=int(changed.sum()),law0=int((acts[:,0]==oracle_actions[:,0]).sum()),law1=int((acts[:,1]==oracle_actions[:,1]).sum()),replica_sham=int(sham.sum()),foreign_source_same_law=int(foreign.sum()),shuffled_both=int(shuffleboth.sum()),zero_both=int(((z==oracle_actions[:,0])&(z==oracle_actions[:,1])).sum()),blind_both=int(((blind==oracle_actions[:,0])&(blind==oracle_actions[:,1])).sum()),illegal=int((~np.isin(acts,[0,1])).any(axis=-1).sum()),max_donor_mse=max_mse)
    diverge=bucket.endswith('divergent')
    if diverge:
        passed=(m['both']/n>=THRESHOLDS['divergent_both'] and m['flip']/n>=THRESHOLDS['divergent_flip'] and m['shuffled_both']/n<=THRESHOLDS['shuffled_both_max'] and m['zero_both']==0 and m['blind_both']==0)
    else:
        passed=(m['both']/n>=THRESHOLDS['invariant_both'] and m['flip']/n<=THRESHOLDS['invariant_flip_max'])
    passed=passed and min(m['law0'],m['law1'])/n>=THRESHOLDS['each_law'] and m['replica_sham']/n<=THRESHOLDS['same_law_replica_sham_max'] and m['foreign_source_same_law']/n<=THRESHOLDS['foreign_same_law_swap_max'] and m['illegal']==0 and max_mse<=THRESHOLDS['donor_mse_max']
    confidence=np.stack([torch.softmax(torch.from_numpy(logits[k][0]),-1).numpy()[np.arange(n),oracle_actions[:,k]] for k in (0,1)],axis=-1)
    margins=np.stack([logits[k][0][np.arange(n),oracle_actions[:,k]]-logits[k][0][np.arange(n),1-oracle_actions[:,k]] for k in (0,1)],axis=-1)
    for i in range(n):
        row=dict(receiver_seed=receiving_seed, donor_source_seed=source_seed, family=family,bucket=bucket,index=i,
            observation_sha256=sha(worlds[i].tobytes()),observation=worlds[i].tolist(), legal_mask=[1,1],
            oracle_actions=oracle_actions[i].tolist(),oracle_gaps=oracle_gaps[i].tolist(),oracle_route_costs=oracle_costs[i],
            actions=acts[i].tolist(),same_law_replica_actions=shams[i].tolist(),receiver_own_donor_actions=ownacts[i].tolist(),
            logits=[logits[k][0][i].tolist() for k in (0,1)],sham_logits=[logits[k][1][i].tolist() for k in (0,1)],
            correct_action_margins=margins[i].tolist(),correct_action_confidences=confidence[i].tolist(),
            raw_donor_forecasts=[raw[k][0][i].numpy().tolist() for k in (0,1)],
            raw_sham_forecasts=[raw[k][1][i].numpy().tolist() for k in (0,1)],
            conditioned_donor_forecasts=[conditioned[k][0][i].numpy().tolist() for k in (0,1)],
            conditioned_sham_forecasts=[conditioned[k][1][i].numpy().tolist() for k in (0,1)],
            shuffled_actions=shuffled[i].tolist(),zero_action=int(z[i]),source_blind_action=int(blind[i]),
            oracle_both_correct=bool(both[i]),law_swap_action_changed=bool(changed[i]),
            replica_sham_changed=bool(sham[i]),same_law_source_changed=bool(foreign[i]))
        writer.write(json.dumps(row,separators=(',',':'))+'\n')
    return dict(verdict='PASS' if passed else 'FAIL',n=n,counts=m,rate={k:round(v/n,6) for k,v in m.items() if k!='max_donor_mse'}, mean_oracle_gap=oracle_gaps.mean(axis=0).tolist(), mean_confidence=confidence.mean(axis=0).tolist(), min_correct_margin=margins.min(axis=0).tolist(), failures=[i for i in range(n) if not both[i] or sham[i] or foreign[i]][:30])

def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(exist_ok=True)
    receivers,donors,blinds,model_hashes=load_models()
    replay=baseline_replay(receivers,donors)
    assert replay['action_disagreement_count']==0, replay
    old_hashes=set()
    for seed in SEEDS:
        zip_path=BASE/f'DG020-H8-exact-head-seed{str(seed)[-2:]}.zip'
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.startswith('receipts-') and name.endswith('.jsonl'):
                    for line in z.read(name).splitlines():old_hashes.add(json.loads(line)['observation_sha256'])
    all_new=set();results={};attempts={};passed=True
    for receiver_seed in SEEDS:
        results[str(receiver_seed)]={}
        for family in FAMILIES:
            results[str(receiver_seed)][family]={}
            for bucket in BUCKETS:
                x,attempt=fresh_worlds(receiver_seed,family,bucket)
                hashes=[sha(obs.tobytes()) for obs in x]
                assert len(set(hashes))==len(hashes) and not (set(hashes)&all_new) and not (set(hashes)&old_hashes)
                all_new.update(hashes)
                attempts[f'{receiver_seed}/{family}/{bucket}']=attempt
                results[str(receiver_seed)][family][bucket]={}
                for donor_seed in SEEDS:
                    out=OUTPUT/f'receipts-receiver{receiver_seed}-donor{donor_seed}-{family}-{bucket}.jsonl'
                    with out.open('w') as writer:
                        result=evaluate(receiver_seed,family,bucket,x,receivers,donors,blinds,donor_seed,writer)
                    results[str(receiver_seed)][family][bucket][str(donor_seed)]=result
                    passed=passed and result['verdict']=='PASS'
                    print(f'{receiver_seed} <= {donor_seed} {family}/{bucket}: {result["verdict"]} both={result["counts"]["both"]}/160 flip={result["counts"]["flip"]}/160 sham={result["counts"]["replica_sham"]}/160 source={result["counts"]["foreign_source_same_law"]}/160',flush=True)
    # Technical audit of receipts, without re-tuning or changing thresholds.
    summary=dict(name='DG020 H9 preregistered frozen cross-seed donor transplant',scientific_verdict='PASS' if passed else 'FAIL',
        qualification='independent LOCAL replay on torch 2.10.0 CPU (not pinned CI)',
        preregistration='https://github.com/techrote/neurounner/pull/253#issuecomment-6942182942',
        baseline_h8_replay=replay, thresholds=THRESHOLDS,receiving_seeds=SEEDS, donor_seeds=SEEDS,
        families=FAMILIES,buckets=BUCKETS, pairs_per_cell=PER_CELL,total_cells=90,
        unique_new_worlds=len(all_new),h8_holdout_observation_count=len(old_hashes),h8_collisions=0,
        attempts=attempts,model_sha256=model_hashes,torch_version=torch.__version__,numpy_version=np.__version__,
        source_sha256=hashfile(__file__),results=results)
    (OUTPUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    m={p.name:hashfile(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!='sha256.json'}
    (OUTPUT/'sha256.json').write_text(json.dumps(m,indent=2)+'\n')
    print('FINAL',json.dumps(dict(scientific_verdict=summary['scientific_verdict'],baseline_replay=replay,passed_cells=sum(1 for a in results.values() for b in a.values() for c in b.values() for d in c.values() if d['verdict']=='PASS'),unique_new_worlds=len(all_new),num_receipts=90*160)),flush=True)

if __name__=='__main__':main()
