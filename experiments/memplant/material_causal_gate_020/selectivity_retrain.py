"""DG020 corrective causal selectivity: mixed invariant/divergent paired supervision.

Fixed 2x12 topology, 2 wet laws, 2 legal actions; independent learned donors.
Trainer-only local tile costs prevent global wet-route shortcut. No inference
law/family labels, oracle distances or labels. Research only; no game changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiment import Donor, Receiver, WET_COST, TILES, train_donor, oracle

SEEDS=(26011639,26011640,26011641)
FAMILIES=('checker_islands','mirrored_ribbons')
BUCKETS=('divergent','equal_wet_invariant','toll_invariant')
THRESHOLDS=dict(pairs_per_bucket=160, min_oracle_gap=.60,
    divergent_both=.90, divergent_flip=.95, each_law=.94,
    invariant_both=.94, invariant_flip_max=.05,
    sham_flip_max=.05, divergent_shuffle_both_max=.60,
    donor_mse_max=.02, illegal=0)

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha_bytes(Path(p).read_bytes())

def wet_route(rng,family,k):
    wet=np.zeros(TILES,dtype=np.float32)
    if family=='iid':
        ix=rng.choice(TILES,k,replace=False);wet[ix]=1
    elif family=='short_runs':
        ix=(int(rng.integers(TILES))+np.arange(k))%TILES;wet[ix]=1
    elif family=='checker_islands':
        order=np.array([0,2,4,6,8,10,1,3,5,7,9,11]);wet[(order[:k]+int(rng.integers(TILES)))%TILES]=1
        if rng.integers(2):wet=wet[::-1].copy()
    elif family=='mirrored_ribbons':
        n1=int(rng.integers(1,k)) if k>1 else 1
        ix=np.r_[np.arange(n1),np.arange(n1+1,n1+1+k-n1)]%TILES
        wet[np.roll(ix,int(rng.integers(TILES))%len(ix))]=1
        if rng.integers(2):wet=np.roll(wet,int(rng.integers(TILES)))
        if rng.integers(2):wet=wet[::-1].copy()
    else:raise ValueError(family)
    assert int(wet.sum())==k,(family,k,wet.sum())
    return wet

def make_candidate(rng,family,bucket):
    k=int(rng.integers(2,9))
    if bucket=='equal_wet_invariant':counts=(k,k)
    else:
        extra=int(rng.integers(1,4));counts=(k,min(11,k+extra))
    if rng.integers(2):counts=counts[::-1]
    wet=np.stack([wet_route(rng,family,i) for i in counts])
    fee=rng.uniform(0,.30,(2,TILES)).astype('f4')
    if bucket=='toll_invariant':
        # Bias the wetter path's fees upwards, with random per-cell offsets.
        rich=int(np.argmax(wet.sum(-1)))
        fee[rich]=rng.uniform(.15,.30,TILES)
        fee[1-rich]=rng.uniform(0,.15,TILES)
    elif bucket=='divergent':
        # Bidirectional fee offset; no donor label or family field in observations.
        rich=int(np.argmax(wet.sum(-1)))
        if rng.integers(2):
            fee[rich]=rng.uniform(0,.12,TILES);fee[1-rich]=rng.uniform(.18,.30,TILES)
    elif bucket=='equal_wet_invariant':
        favored=int(rng.integers(2));fee[favored]=rng.uniform(0,.11,TILES)
        fee[1-favored]=rng.uniform(.19,.30,TILES)
    texture=rng.uniform(-1,1,(2,TILES,2)).astype('f4')
    return np.concatenate([wet[...,None],fee[...,None],texture],axis=-1)

def sample(seed,family,bucket,n):
    rng=np.random.default_rng(seed);xs=[];attempts=0
    while len(xs)<n and attempts<n*2000:
        obs=make_candidate(rng,family,bucket);attempts+=1
        a,ga,_=oracle(obs,0);b,gb,_=oracle(obs,1)
        if min(ga,gb)<THRESHOLDS['min_oracle_gap']:continue
        if (a!=b)!=(bucket=='divergent'):continue
        if bucket=='equal_wet_invariant' and obs[0,:,0].sum()!=obs[1,:,0].sum():raise AssertionError
        if bucket=='toll_invariant' and obs[0,:,0].sum()==obs[1,:,0].sum():raise AssertionError
        xs.append(obs)
    assert len(xs)==n,(family,bucket,len(xs),attempts)
    return np.asarray(xs,dtype='f4'),attempts

def forecasts(donor,x):
    with torch.no_grad():return donor(torch.from_numpy(x).reshape(-1,4)).reshape(len(x),2,TILES)

def fit_receiver(seed,x,donors,steps,blind=False,shuffled=False):
    torch.manual_seed(seed);rng=np.random.default_rng(seed+81)
    r=Receiver();optim=torch.optim.AdamW(r.parameters(),lr=.0025,weight_decay=.0001)
    obs=torch.from_numpy(x);costs=[forecasts(donors[k][0],x) for k in (0,1)]
    labels=[torch.tensor([oracle(a,k)[0] for a in x],dtype=torch.long) for k in (0,1)]
    true_cost=[torch.from_numpy(np.where(x[...,0]>.5,WET_COST[k],1.)+x[...,1]).float() for k in (0,1)]
    for step in range(steps):
        ix=torch.from_numpy(rng.integers(len(x),size=128))
        batch_loss=0.
        for law in (0,1):
            source=0 if blind else (int(rng.integers(2)) if shuffled else law)
            donor=torch.zeros_like(costs[0][ix]) if blind else costs[source][ix]
            features=torch.stack((obs[ix,...,0],obs[ix,...,1],donor),-1)
            tile_pred=r.tile(features).squeeze(-1)
            logits=-tile_pred.sum(-1)
            # A cost-calibrated tile-to-route distillation objective, training-only oracle.
            # True costs never enter receiver inference, and input law is ONLY donor forecast.
            loss=F.cross_entropy(logits,labels[law][ix]) + 1.2*F.mse_loss(tile_pred,true_cost[law][ix])
            batch_loss+=loss*.5
        optim.zero_grad();batch_loss.backward();optim.step()
    return r.eval()

def outcome(seed,family,bucket,x,donors,receiver,blind,out):
    obs=torch.from_numpy(x);n=len(x)
    with torch.no_grad():
        costs=[forecasts(donors[k][0],x) for k in (0,1)]
        logit=[receiver(obs,c) for c in costs]
        sham=[receiver(obs,forecasts(donors[k][1],x)) for k in (0,1)]
        zero=receiver(obs,torch.zeros_like(costs[0])).argmax(-1)
        no_source=blind(obs,torch.zeros_like(costs[0])).argmax(-1)
        rng=np.random.default_rng(seed+90019)
        # Pairwise mismatched donor fingerprints are assigned independently of target law.
        shuf=[receiver(obs,torch.stack([costs[int(rng.integers(2))][i] for i in range(n)])) for _ in (0,1)]
        probs=[F.softmax(l,-1) for l in logit]
    actions=[v.argmax(-1).tolist() for v in logit]
    shams=[v.argmax(-1).tolist() for v in sham]
    shuffles=[v.argmax(-1).tolist() for v in shuf]
    keys=('both','flip','low_correct','high_correct','sham_flip','shuffled_both','zero_both','blind_both','illegal')
    counts={k:0 for k in keys};margin=[[],[]];confidence=[[],[]];failures=[]
    with open(out,'w') as f:
        for i,w in enumerate(x):
            targets=[oracle(w,k) for k in (0,1)]
            pair=[actions[k][i] for k in (0,1)]
            truth=[t[0] for t in targets];changed=pair[0]!=pair[1]
            good=pair==truth
            counts['both']+=int(good);counts['flip']+=int(changed)
            counts['low_correct']+=int(pair[0]==truth[0]);counts['high_correct']+=int(pair[1]==truth[1])
            counts['sham_flip']+=int(any(shams[k][i]!=pair[k] for k in (0,1)))
            counts['shuffled_both']+=int([shuffles[k][i] for k in (0,1)]==truth)
            counts['zero_both']+=int(zero[i]==truth[0]==truth[1]);counts['blind_both']+=int(no_source[i]==truth[0]==truth[1])
            counts['illegal']+=int(any(a not in (0,1) for a in pair))
            m=[float(logit[k][i,truth[k]]-logit[k][i,1-truth[k]]) for k in (0,1)]
            c=[float(probs[k][i,truth[k]]) for k in (0,1)]
            for k in (0,1):margin[k].append(m[k]);confidence[k].append(c[k])
            if not good and len(failures)<30:failures.append(dict(index=i,oracle=truth,pred=pair,margin=m,oracle_gap=[t[1] for t in targets]))
            receipt=dict(index=i,observation_sha256=sha_bytes(w.tobytes()),observation=w.tolist(),
                same_observation_both_laws=True,legal_mask=[1,1],host_action_names=['LEFT_ROUTE','RIGHT_ROUTE'],
                oracle_actions=truth,oracle_gaps=[t[1] for t in targets],oracle_costs=[t[2] for t in targets],
                actions=pair,logits=[l[i].tolist() for l in logit],correct_action_margins=m,confidence=c,
                donor_tile_forecasts=[costs[k][i].tolist() for k in (0,1)],
                sham_actions=[shams[k][i] for k in (0,1)],shuffled_actions=[shuffles[k][i] for k in (0,1)],
                zero_action=int(zero[i]),blind_action=int(no_source[i]),both_correct=good)
            f.write(json.dumps(receipt,separators=(',',':'))+'\n')
    metrics={k:round(v/n,5) for k,v in counts.items()}
    return dict(n=n,metrics=metrics,mean_margin=[float(np.mean(z)) for z in margin],
        mean_correct_confidence=[float(np.mean(z)) for z in confidence],failures=failures)

def verdict(report):
    for family in FAMILIES:
        for bucket in BUCKETS:
            m=report[family][bucket]['metrics']
            if m['illegal']!=0 or m['sham_flip']>THRESHOLDS['sham_flip_max']:return 'FAIL'
            if min(m['low_correct'],m['high_correct'])<THRESHOLDS['each_law']:return 'FAIL'
            if bucket=='divergent':
                if m['both']<THRESHOLDS['divergent_both'] or m['flip']<THRESHOLDS['divergent_flip'] or m['shuffled_both']>THRESHOLDS['divergent_shuffle_both_max']:return 'FAIL'
                if m['zero_both']>0 or m['blind_both']>0:return 'FAIL'
            else:
                if m['both']<THRESHOLDS['invariant_both'] or m['flip']>THRESHOLDS['invariant_flip_max']:return 'FAIL'
    return 'PASS'

def run(seed,out,train_each=600,eval_each=160,steps=700,donor_steps=400):
    torch.set_num_threads(2);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    donors={};mses={}
    for law in (0,1):
        donors[law]=[]
        for replica in (0,1):
            d,_=train_donor(seed+1000+law*200+replica*11,law,donor_steps)
            name=f'donor-{law}-replica-{replica}.pt';torch.save(d.state_dict(),out/name)
            donors[law].append(d)
            rng=np.random.default_rng(seed+6000+law*333+replica*22)
            xx=rng.uniform(-1,1,(4096,4)).astype('f4');xx[:,0]=rng.integers(0,2,4096);xx[:,1]=rng.uniform(0,.3,4096)
            with torch.no_grad():yy=d(torch.from_numpy(xx)).numpy()
            truth=np.where(xx[:,0]>.5,WET_COST[law],1.)+xx[:,1]
            mses[name]=float(np.mean((yy-truth)**2))
    training=[]
    for t,fam in enumerate(('iid','short_runs')):
        for b,bucket in enumerate(BUCKETS):
            xs,_=sample(seed+50000+t*13000+b*31,fam,bucket,train_each)
            training.append(xs)
    x=np.concatenate(training)
    receiver=fit_receiver(seed+7700,x,donors,steps)
    blind=fit_receiver(seed+7701,x,donors,steps,blind=True)
    torch.save(receiver.state_dict(),out/'receiver-cost-calibrated.pt')
    torch.save(blind.state_dict(),out/'receiver-source-blind.pt')
    report={}
    for j,fam in enumerate(FAMILIES):
        report[fam]={}
        for b,bucket in enumerate(BUCKETS):
            xs,attempts=sample(seed+19_000_000+j*100000+b*10000,fam,bucket,eval_each)
            rep=outcome(seed+j*100+b,fam,bucket,xs,donors,receiver,blind,out/f'receipts-{fam}-{bucket}.jsonl')
            rep['candidate_attempts']=attempts;report[fam][bucket]=rep
    status=verdict(report)
    if max(mses.values())>THRESHOLDS['donor_mse_max']:status='FAIL'
    summary=dict(experiment='DG020-selectivity-corrective-001',seed=seed,verdict=status,
        hypothesis='Changing only trained inherited wet-law donor changes action when oracle changes and preserves action when oracle invariant',
        preregistered_thresholds=THRESHOLDS,training_families=['iid','short_runs'],heldout_families=FAMILIES,
        training_buckets=list(BUCKETS),training_worlds=len(x),updates=steps,donor_steps=donor_steps,
        donor_mse=mses,source_sha256=sha_file(__file__),torch_version=torch.__version__,
        no_label_oracle_cost_family_at_inference=True,evaluation=report,
        scope='Narrow trained local-cost donor to additive receiver compositional transfer; not physics or memory transplantation')
    (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    checks={p.name:sha_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(dict(seed=seed,status=status,mse_max=max(mses.values()),metrics={f:{b:report[f][b]['metrics'] for b in BUCKETS} for f in FAMILIES})))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=SEEDS[0]);p.add_argument('--out',default='/tmp/dg020-selective')
    p.add_argument('--train-each',type=int,default=600);p.add_argument('--eval-each',type=int,default=160)
    p.add_argument('--updates',type=int,default=700);p.add_argument('--donor-steps',type=int,default=400)
    a=p.parse_args();run(a.seed,a.out,a.train_each,a.eval_each,a.updates,a.donor_steps)
