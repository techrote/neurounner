"""DG020 H4: preregistered frozen-weight near-boundary invariant falsification.
https://github.com/techrote/neurounner/pull/253#issuecomment-6936727739
No retraining, topology or law expansion. Source-pinned research only.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import softmax
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_h3_audit import load, SEEDS, oracle, TILES

FAMILIES=("wet_rich_near_invariant","wet_poor_near_invariant")
N=160
REQUIRED=dict(both=.94, each_law=.94, flip_max=.05, sham_max=.05, donor_mse_max=.02)
def sha(b): return hashlib.sha256(b).hexdigest()
def wet(rng,k):
    v=np.zeros(TILES,dtype="f4")
    v[(int(rng.integers(TILES))+int(rng.choice([5,7]))*np.arange(k))%TILES]=1.
    if rng.integers(2): v=v[::-1].copy()
    return v
def candidate(rng,family):
    k=int(rng.integers(2,9))
    rich=int(rng.integers(2));poor=1-rich
    w=np.empty((2,TILES),dtype="f4")
    w[rich]=wet(rng,k+1);w[poor]=wet(rng,k)
    fee=np.empty((2,TILES),dtype="f4")
    if family==FAMILIES[0]:
        fee[rich]=rng.uniform(.045,.080,TILES)
        fee[poor]=rng.uniform(.220,.270,TILES)
    else:
        fee[rich]=rng.uniform(.190,.250,TILES)
        fee[poor]=rng.uniform(.075,.120,TILES)
    texture=rng.uniform(-1,1,(2,TILES,2)).astype("f4")
    return np.concatenate((w[...,None],fee[...,None],texture),-1),rich,poor
def sample(seed,family):
    rng=np.random.default_rng(seed+43001107+(0 if family==FAMILIES[0] else 84109))
    arr=[];attempts=0;seen=set()
    while len(arr)<N and attempts<500000:
        x,rich,poor=candidate(rng,family);attempts+=1
        a,ga,_=oracle(x,0); b,gb,_=oracle(x,1)
        gap=gb if family==FAMILIES[0] else ga
        aim=rich if family==FAMILIES[0] else poor
        if a!=aim or b!=aim or not (.60<=gap<=.90) or min(ga,gb)<.60: continue
        digest=sha(x.tobytes())
        if digest in seen: continue
        seen.add(digest);arr.append(x)
    assert len(arr)==N,(seed,family,len(arr),attempts)
    return np.asarray(arr,dtype="f4"),attempts
def evaluate(seed,family,donors,receiver,blind,out):
    x,attempts=sample(seed,family); n=len(x); z=torch.from_numpy(x)
    with torch.no_grad():
        f=[[m(z.reshape(-1,4)).reshape(n,2,TILES) for m in replicas] for replicas in donors]
        logit=[receiver(z,f[k][0]) for k in (0,1)]
        sham=[receiver(z,f[k][1]).argmax(-1) for k in (0,1)]
        zero=receiver(z,torch.zeros_like(f[0][0])).argmax(-1)
        no_source=blind(z,torch.zeros_like(f[0][0])).argmax(-1)
        rng=np.random.default_rng(seed+43008259+(0 if family==FAMILIES[0] else 712))
        shuffled=[]
        for _ in (0,1):
            assign=rng.integers(2,size=n)
            alt=torch.stack([f[int(assign[i])][0][i] for i in range(n)])
            shuffled.append(receiver(z,alt).argmax(-1))
        probs=[softmax(v,-1) for v in logit]
    keys=("both","flip","sham_flip","low_correct","high_correct","shuffled_both","zero_both","blind_both","illegal")
    count={k:0 for k in keys};margins=[[],[]];conf=[[],[]];gaps=[[],[]];err=[];fail=[]
    with (out/f"receipts-{seed}-{family}.jsonl").open("w") as fd:
        for i,w in enumerate(x):
            targets=[oracle(w,k) for k in (0,1)]
            y=[t[0] for t in targets]
            a=[int(logit[k][i].argmax()) for k in (0,1)]
            sh=[int(sham[k][i]) for k in (0,1)]
            mixed=[int(shuffled[k][i]) for k in (0,1)]
            margin=[float(logit[k][i,y[k]]-logit[k][i,1-y[k]]) for k in (0,1)]
            confidence=[float(probs[k][i,y[k]]) for k in (0,1)]
            good=a==y
            count["both"]+=int(good)
            count["flip"]+=int(a[0]!=a[1])
            count["sham_flip"]+=int(sh!=a)
            count["low_correct"]+=int(a[0]==y[0])
            count["high_correct"]+=int(a[1]==y[1])
            count["shuffled_both"]+=int(mixed==y)
            count["zero_both"]+=int(int(zero[i])==y[0]==y[1])
            count["blind_both"]+=int(int(no_source[i])==y[0]==y[1])
            count["illegal"]+=int(any(k not in (0,1) for k in a))
            for k in (0,1):
                margins[k].append(margin[k]);conf[k].append(confidence[k]);gaps[k].append(targets[k][1])
                true=np.where(w[...,0]>.5,(.35,2.30)[k],1.)+w[...,1]
                for j in (0,1):err.append(float(np.mean((f[k][j][i].numpy()-true)**2)))
            receipt=dict(i=i,obs_sha256=sha(w.tobytes()),observation=w.tolist(),
                legal_mask=[1,1],oracle_actions=y,oracle_costs=[t[2] for t in targets],
                oracle_gaps=[t[1] for t in targets],actions=a,logits=[v[i].tolist() for v in logit],
                correct_margins=margin,correct_confidence=confidence,
                donor_forecasts=[f[k][0][i].tolist() for k in (0,1)],
                sham_actions=sh,shuffled_actions=mixed,zero_action=int(zero[i]),
                blind_action=int(no_source[i]),both_correct=good,same_observation_both_laws=True)
            fd.write(json.dumps(receipt,separators=(",",":"))+"\n")
            if not good or a[0]!=a[1] or sh!=a:
                fail.append({k:receipt[k] for k in ("i","obs_sha256","oracle_actions","oracle_gaps",
                    "actions","correct_margins","sham_actions")})
    metrics={k:v/n for k,v in count.items()}
    mse=float(np.mean(err))
    verdict=("PASS" if metrics["both"]>=REQUIRED["both"] and
        min(metrics["low_correct"],metrics["high_correct"])>=REQUIRED["each_law"] and
        metrics["flip"]<=REQUIRED["flip_max"] and metrics["sham_flip"]<=REQUIRED["sham_max"] and
        mse<=REQUIRED["donor_mse_max"] and metrics["illegal"]==0 else "FAIL")
    return dict(verdict=verdict,n=n,attempts=attempts,metrics=metrics,donor_mse=mse,
        mean_oracle_gap=[float(np.mean(v)) for v in gaps],
        mean_correct_margin=[float(np.mean(v)) for v in margins],
        mean_correct_confidence=[float(np.mean(v)) for v in conf],failures=fail)
def main(root,out):
    torch.set_num_threads(2)
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=dict(experiment="DG020-H4",preregistration="issuecomment-6936727739",
        frozen=True,seeds=SEEDS,families=FAMILIES,thresholds=REQUIRED,checks={})
    for seed in SEEDS:
        donors,receiver,blind,prior,hashes=load(root,seed)
        sr=dict(checkpoint_sha256=hashes,prior_verdict=prior["verdict"],families={})
        for family in FAMILIES:sr["families"][family]=evaluate(seed,family,donors,receiver,blind,out)
        result["checks"][str(seed)]=sr
    result["verdict"]="PASS" if all(a["verdict"]=="PASS" for s in result["checks"].values() for a in s["families"].values()) else "FAIL"
    (out/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    manifest={p.name:sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file() and p.name!="sha256.json"}
    (out/"sha256.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(result["verdict"])
    if result["verdict"]!="PASS":sys.exit(3)
if __name__=="__main__":
    if len(sys.argv)!=3:sys.exit("usage: near_boundary_h4_audit.py INPUT_CHECKPOINT_DIR OUTPUT_DIR")
    main(*sys.argv[1:])
