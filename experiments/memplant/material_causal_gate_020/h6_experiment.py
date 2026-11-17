"""DG020 H6 frozen-donor, retrained same-size receiver and sealed near-tie gate.
Preregistered at https://github.com/techrote/neurounner/pull/253#issuecomment-6938873924
Research only: never modifies game/runtime/provider code. Scientific FAIL is preserved.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiment import Donor, Receiver, TILES, WET_COST, oracle

SEEDS=(26011639,26011640,26011641)
TRAIN_FAMILIES=('iid','short_runs')
HELDOUT_FAMILIES=('split_islands','staggered_quads')
BUCKETS=('cheap_edge_divergent','expensive_edge_divergent','wet_rich_invariant','wet_poor_invariant','equal_wet_invariant')
THRESHOLDS=dict(n=160,divergent_both=.90,divergent_flip=.95,each_law=.94,invariant_both=.94,invariant_flip_max=.05,sham_flip_max=.05,shuffled_both_max=.60,donor_mse_max=.02,illegal=0)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def sha_file(path):return sha(Path(path).read_bytes())

def wet_route(rng,n,family):
    a=np.zeros(TILES,dtype='f4'); start=int(rng.integers(TILES)); flip=int(rng.integers(2))
    if family=='iid':order=rng.permutation(TILES)
    elif family=='short_runs':order=(start+np.arange(TILES))%TILES
    elif family=='split_islands':
        # Two bounded disjoint islands, interspersed with dry tiles; shift/reflect.
        k=(n+1)//2
        order=np.r_[np.arange(k), np.arange(6,6+n-k),np.setdiff1d(np.arange(TILES),np.r_[np.arange(k),np.arange(6,6+n-k)])]
        assert len(set(order[:n]))==n
        order=(order+start)%TILES
    elif family=='staggered_quads':
        order=(np.array([0,1,2,3,8,9,10,11,4,5,6,7])+start)%TILES
    else:raise ValueError(family)
    if flip:order=order[::-1] if family in ('iid','short_runs') else (TILES-1-order)%TILES
    a[order[:n]]=1
    assert int(a.sum())==n
    return a

def make_world(rng,family,bucket,near=True):
    k=int(rng.integers(2,10)); rich=int(rng.integers(2));poor=1-rich
    counts=[k,k] if bucket=='equal_wet_invariant' else ([k+1,k] if rich==0 else [k,k+1])
    wet=np.stack([wet_route(rng,n,family) for n in counts])
    gap=float(rng.uniform(.05,.20) if near else rng.uniform(.60,.95))
    if bucket=='cheap_edge_divergent':delta=.65-gap
    elif bucket=='expensive_edge_divergent':delta=-1.30+gap
    elif bucket=='wet_rich_invariant':delta=-1.30-gap
    elif bucket=='wet_poor_invariant':delta=.65+gap
    else:delta=gap*(1 if rng.integers(2) else -1)
    fees=rng.uniform(.12,.18,(2,TILES)).astype('f8')
    offset=(delta-(fees[rich].sum()-fees[poor].sum()))/(2*TILES)
    fees[rich]+=offset; fees[poor]-=offset
    # An unusual wide-gap requested combination can exceed fee support, so resample.
    if fees.min()<0 or fees.max()>.30:return None
    tex=rng.uniform(-1,1,(2,TILES,2)).astype('f4')
    x=np.concatenate((wet[...,None],fees.astype('f4')[...,None],tex),axis=-1)
    a,ga,_=oracle(x,0);b,gb,_=oracle(x,1)
    if bucket.endswith('divergent'):
        assert a==rich and b==poor,(bucket,a,b,rich,poor)
    elif bucket=='wet_rich_invariant':assert a==b==rich
    elif bucket=='wet_poor_invariant':assert a==b==poor
    else:assert a==b
    edge={'cheap_edge_divergent':ga,'expensive_edge_divergent':gb,'wet_rich_invariant':gb,'wet_poor_invariant':ga,'equal_wet_invariant':ga}[bucket]
    if not near and edge<.60:return None
    assert (0.05-.001<=edge<=.20+.001) if near else (.60-.001<=edge<=.95+.001),(bucket,edge)
    return x

def sample(seed,family,bucket,n,near=True):
    rng=np.random.default_rng(seed);ret=[];attempt=0
    while len(ret)<n and attempt<n*100:
        x=make_world(rng,family,bucket,near);attempt+=1
        if x is not None:ret.append(x)
    if len(ret)!=n:raise AssertionError((seed,family,bucket,len(ret),attempt))
    return np.stack(ret),attempt

def load_donors(seed,root):
    root=Path(root)/str(seed)
    donors=[]
    for law in (0,1):
        rr=[]
        for replica in (0,1):
            model=Donor().eval()
            model.load_state_dict(torch.load(root/f'donor-{law}-replica-{replica}.pt',map_location='cpu',weights_only=True))
            rr.append(model)
        donors.append(rr)
    blind=Receiver().eval()
    blind.load_state_dict(torch.load(root/'receiver-source-blind.pt',map_location='cpu',weights_only=True))
    return donors,blind

def donor_forecast(d,obs):
    with torch.no_grad():return d(obs.reshape(-1,4)).reshape(len(obs),2,TILES)

def fit(seed,worlds,donors,updates=1000):
    torch.manual_seed(seed+999)
    rng=np.random.default_rng(seed+101)
    obs=torch.from_numpy(worlds)
    fc=[donor_forecast(donors[k][0],obs) for k in (0,1)]
    targets=[torch.from_numpy(np.asarray([oracle(x,k)[2] for x in worlds],dtype='f4')) for k in (0,1)]
    truth=[torch.from_numpy((np.where(worlds[...,0]>.5,WET_COST[k],1.)+worlds[...,1]).astype('f4')) for k in (0,1)]
    model=Receiver().train()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    for step in range(updates):
        ids=torch.from_numpy(rng.integers(0,len(obs),192))
        loss=0.
        for law in (0,1):
            tile=model.tile(torch.stack((obs[ids,...,0],obs[ids,...,1],fc[law][ids]),-1)).squeeze(-1)
            logits=-tile.sum(-1)
            # Regression directly calibrates the route advantage, not just local cost.
            cost_diff=targets[law][ids,1]-targets[law][ids,0]
            score_diff=logits[:,0]-logits[:,1]
            labs=(cost_diff<0).long() # right route best for negative (cost_right-cost_left)
            labs=(cost_diff>0).long() # 0 if right cheaper, 1 if left cheaper -- incorrect for crossentropy!
            labs=torch.where(cost_diff>0,torch.zeros_like(labs),torch.ones_like(labs))
            loss+=.5*(3*F.smooth_l1_loss(score_diff,cost_diff,beta=.15)+.35*F.mse_loss(tile,truth[law][ids])+.15*F.cross_entropy(logits,labs))
        opt.zero_grad();loss.backward();opt.step()
    return model.eval()

def evaluate(seed,family,bucket,worlds,donors,receiver,blind,out):
    n=len(worlds);x=torch.from_numpy(worlds)
    with torch.no_grad():
        ff=[[donor_forecast(d,x) for d in replicas] for replicas in donors]
        logits=[receiver(x,ff[k][0]) for k in (0,1)]
        sham=[receiver(x,ff[k][1]).argmax(-1).numpy() for k in (0,1)]
        zero=receiver(x,torch.zeros_like(ff[0][0])).argmax(-1).numpy()
        blind_pred=blind(x,torch.zeros_like(ff[0][0])).argmax(-1).numpy()
        rng=np.random.default_rng(seed+90000+HELDOUT_FAMILIES.index(family)*79+BUCKETS.index(bucket)*1000)
        shuffled=[]
        for k in (0,1):
            picks=rng.integers(2,size=n)
            f=torch.stack([ff[int(picks[i])][0][i] for i in range(n)])
            shuffled.append(receiver(x,f).argmax(-1).numpy())
        probs=[F.softmax(logit,-1) for logit in logits]
    cnt={k:0 for k in ('both','flip','law0','law1','sham','shuffled_both','zero_both','blind_both','illegal')}
    margins=[[],[]];conf=[[],[]];fails=[];donor_mse=[]
    with open(out,'w') as w:
        for i,obs in enumerate(worlds):
            ora=[oracle(obs,k) for k in (0,1)];gt=[q[0] for q in ora]
            acts=[int(z[i].argmax()) for z in logits]
            ss=[int(sham[k][i]) for k in (0,1)];shu=[int(shuffled[k][i]) for k in (0,1)]
            margin=[float(z[i,gt[k]]-z[i,1-gt[k]]) for k,z in enumerate(logits)]
            confidence=[float(probs[k][i,gt[k]]) for k in (0,1)]
            ct=[np.where(obs[...,0]>.5,WET_COST[k],1.)+obs[...,1] for k in (0,1)]
            for k in (0,1):
                for replica in (0,1):donor_mse.append(float(np.mean((ff[k][replica][i].numpy()-ct[k])**2)))
            cnt['both']+=int(acts==gt);cnt['flip']+=int(acts[0]!=acts[1]);cnt['law0']+=int(acts[0]==gt[0]);cnt['law1']+=int(acts[1]==gt[1])
            cnt['sham']+=int(ss!=acts);cnt['shuffled_both']+=int(shu==gt)
            cnt['zero_both']+=int(int(zero[i])==gt[0]==gt[1]);cnt['blind_both']+=int(int(blind_pred[i])==gt[0]==gt[1]);cnt['illegal']+=int(any(a not in (0,1) for a in acts))
            for k in (0,1):margins[k].append(margin[k]);conf[k].append(confidence[k])
            if acts!=gt or ss!=acts:
                if len(fails)<40:fails.append(dict(index=i,oracle=gt,actions=acts,sham=ss,margin=margin,gaps=[q[1] for q in ora]))
            w.write(json.dumps(dict(seed=seed,family=family,bucket=bucket,index=i,observation_sha256=sha(obs.tobytes()),observation=obs.tolist(),legal_mask=[1,1],identical_observation_both_laws=True,oracle_actions=gt,oracle_gaps=[q[1] for q in ora],oracle_route_costs=[q[2] for q in ora],actions=acts,logits=[v[i].tolist() for v in logits],correct_margins=margin,correct_confidence=confidence,donor_forecasts=[ff[k][0][i].tolist() for k in (0,1)],sham_forecasts=[ff[k][1][i].tolist() for k in (0,1)],sham_actions=ss,shuffled_actions=shu,zero_action=int(zero[i]),blind_action=int(blind_pred[i])),separators=(',',':'))+'\n')
    metrics={k:round(v/n,6) for k,v in cnt.items()}
    divergent=bucket.endswith('divergent')
    passed=(metrics['illegal']==0 and metrics['sham']<=THRESHOLDS['sham_flip_max'] and min(metrics['law0'],metrics['law1'])>=THRESHOLDS['each_law'])
    if divergent:passed=passed and metrics['both']>=THRESHOLDS['divergent_both'] and metrics['flip']>=THRESHOLDS['divergent_flip'] and metrics['shuffled_both']<=THRESHOLDS['shuffled_both_max'] and metrics['zero_both']==0 and metrics['blind_both']==0
    else:passed=passed and metrics['both']>=THRESHOLDS['invariant_both'] and metrics['flip']<=THRESHOLDS['invariant_flip_max']
    return dict(verdict='PASS' if passed else 'FAIL',n=n,metrics=metrics,mean_true_action_margins=[float(np.mean(a)) for a in margins],mean_confidence=[float(np.mean(a)) for a in conf],max_donor_mse=max(donor_mse),failure_examples=fails)

def run(seed,checkpoint_root,out,train_each=320,eval_each=160,updates=1000):
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    donors,blind=load_donors(seed,checkpoint_root)
    samples=[];train_attempts={}
    for fi,family in enumerate(TRAIN_FAMILIES):
        for bi,bucket in enumerate(BUCKETS):
            near,n_attempts=sample(seed+1_000_000+fi*7777+bi*111,family,bucket,train_each,True)
            samples.append(near);train_attempts[f'{family}-{bucket}']=n_attempts
            # Moderate-margin anchor, same training samplers only.
            wide,n_attempts=sample(seed+1_900_000+fi*7777+bi*111,family,bucket,train_each//4,False)
            samples.append(wide);train_attempts[f'{family}-{bucket}-wide']=n_attempts
    x=np.concatenate(samples).astype('f4')
    receiver=fit(seed,x,donors,updates)
    torch.save(receiver.state_dict(),out/'receiver-route-difference.pt')
    results={};test_hashes=set();passed=True
    for fi,family in enumerate(HELDOUT_FAMILIES):
        results[family]={}
        for bi,bucket in enumerate(BUCKETS):
            worlds,attempt=sample(seed+37_000_000+fi*113000+bi*9920,family,bucket,eval_each,True)
            hashes=[sha(a.tobytes()) for a in worlds]
            assert not test_hashes.intersection(hashes);test_hashes.update(hashes)
            r=evaluate(seed,family,bucket,worlds,donors,receiver,blind,out/f'receipts-{family}-{bucket}.jsonl')
            r['attempts']=attempt;results[family][bucket]=r
            passed&=(r['verdict']=='PASS' and r['max_donor_mse']<=THRESHOLDS['donor_mse_max'])
    report=dict(name='DG020 H6 near-tie fixed-topology receiver route difference',seed=seed,scientific_verdict='PASS' if passed else 'FAIL',preregistration='https://github.com/techrote/neurounner/pull/253#issuecomment-6938873924',training_families=TRAIN_FAMILIES,heldout_families=HELDOUT_FAMILIES,training_worlds=len(x),updates=updates,train_each=train_each,eval_each=eval_each,thresholds=THRESHOLDS,train_attempts=train_attempts,model='frozen trained donors + retrained unchanged Receiver architecture',source_sha256=sha_file(__file__),torch=torch.__version__,numpy=np.__version__,results=results)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    sums={p.name:sha_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(sums,indent=2)+'\n')
    print(json.dumps(dict(seed=seed,verdict=report['scientific_verdict'],results={f:{b:r['metrics']|{'verdict':r['verdict']} for b,r in by.items()} for f,by in results.items()})))
    return passed

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--checkpoint-root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--train-each',type=int,default=320);ap.add_argument('--eval-each',type=int,default=160);ap.add_argument('--updates',type=int,default=1000)
    a=ap.parse_args();run(a.seed,a.checkpoint_root,a.out,a.train_each,a.eval_each,a.updates)
