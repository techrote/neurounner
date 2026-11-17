"""DG020 H3: frozen corrected donor/receiver selectivity falsification.
Preregistered: PR #253 issuecomment-6935553280. Research only, no training.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import softmax
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import Donor, Receiver, oracle, TILES

SEEDS = (26011639, 26011640, 26011641)
FAMILIES = ("inverted_toll_invariant", "fee_edge_divergent")
N = 160
REQUIRED = dict(gap=.60, both_divergent=.90, flip_divergent=.95,
                each_law=.94, both_invariant=.94, flip_invariant_max=.05,
                sham_max=.05, shuffled_max=.60, donor_mse_max=.02)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def wet(rng, n):
    # 5/7-stride arrangements absent from original training/test families.
    a=np.zeros(TILES,dtype="f4")
    a[(int(rng.integers(TILES)) + int(rng.choice([5,7]))*np.arange(n))%TILES]=1.
    return a[::-1].copy() if rng.integers(2) else a
def make(rng, family):
    k=int(rng.integers(2,9))
    counts=(k+1,k) if rng.integers(2) else (k,k+1)
    w=np.stack([wet(rng,c) for c in counts])
    rich=int(np.argmax(w.sum(-1)))
    fee=np.empty((2,TILES),dtype="f4")
    if family==FAMILIES[0]:
        fee[rich]=rng.uniform(0,.025,TILES)
        fee[1-rich]=rng.uniform(.270,.300,TILES)
    else:
        fee[rich]=rng.uniform(.140,.156,TILES)
        fee[1-rich]=rng.uniform(.140,.154,TILES)
    tex=rng.uniform(-1,1,(2,TILES,2)).astype("f4")
    return np.concatenate((w[...,None],fee[...,None],tex),axis=-1)
def sample(seed, family):
    rng=np.random.default_rng(seed+32000000+(0 if family==FAMILIES[0] else 773797))
    data=[];attempts=0
    while len(data)<N and attempts<500000:
        x=make(rng,family);attempts+=1
        a,ga,_=oracle(x,0);b,gb,_=oracle(x,1)
        if min(ga,gb)<REQUIRED["gap"]-1e-6: continue
        if family==FAMILIES[0] and (a!=b or a!=int(np.argmax(x[...,0].sum(-1)))):continue
        if family==FAMILIES[1] and a==b:continue
        assert np.min(x[...,1])>=0 and np.max(x[...,1])<=.3000001
        data.append(x)
    assert len(data)==N,(seed,family,len(data),attempts)
    return np.stack(data), attempts
def load(root, seed):
    base=root/str(seed)
    hashes=json.loads((base/"sha256.json").read_text())
    assert len(hashes)>=13
    for p,sha in hashes.items(): assert digest(base/p)==sha,p
    old=json.loads((base/"report.json").read_text())
    assert old["seed"]==seed and old["verdict"]=="PASS"
    def load_model(name,typ):
        m=typ().eval()
        m.load_state_dict(torch.load(base/name,weights_only=True,map_location="cpu"))
        return m
    donors=[[load_model(f"donor-{k}-replica-{j}.pt",Donor) for j in (0,1)] for k in (0,1)]
    return donors,load_model("receiver-cost-calibrated.pt",Receiver),load_model("receiver-source-blind.pt",Receiver),old,hashes
def run_family(seed,family,donors,receiver,blind,out):
    x,attempts=sample(seed,family); z=torch.from_numpy(x);n=len(x)
    with torch.no_grad():
        f=[[d(z.reshape(-1,4)).reshape(n,2,TILES) for d in layer] for layer in donors]
        v=[receiver(z,f[k][0]) for k in (0,1)]
        sh=[receiver(z,f[k][1]).argmax(-1) for k in (0,1)]
        zero=receiver(z,torch.zeros_like(f[0][0])).argmax(-1)
        bs=blind(z,torch.zeros_like(f[0][0])).argmax(-1)
        rng=np.random.default_rng(seed+35000000+(0 if family==FAMILIES[0] else 45671))
        shuffled=[]
        for _ in (0,1):
            assign=rng.integers(2,size=n)
            alt=torch.stack([f[int(assign[j])][0][j] for j in range(n)])
            shuffled.append(receiver(z,alt).argmax(-1))
        probs=[softmax(a,-1) for a in v]
    metrics={k:0 for k in ("both","flip","low_correct","high_correct","sham_flip","shuffled_both","zero_both","blind_both","illegal")}
    failures=[];margin=[[],[]];confidence=[[],[]];sqerr=[]
    with (out/f"receipts-{seed}-{family}.jsonl").open("w") as handle:
        for i,w in enumerate(x):
            oracle_both=[oracle(w,k) for k in (0,1)]
            target=[a for a,g,cost in oracle_both]
            actions=[int(v[k][i].argmax()) for k in (0,1)]
            shams=[int(sh[k][i]) for k in (0,1)]
            shuffled_actions=[int(shuffled[k][i]) for k in (0,1)]
            margins=[float(v[k][i,target[k]]-v[k][i,1-target[k]]) for k in (0,1)]
            conf=[float(probs[k][i,target[k]]) for k in (0,1)]
            for k in (0,1):
                margin[k].append(margins[k]);confidence[k].append(conf[k])
                true=np.where(w[...,0]>.5,(.35,2.30)[k],1.)+w[...,1]
                for rep in (0,1):sqerr.append(float(np.mean((f[k][rep][i].numpy()-true)**2)))
            both=actions==target
            metrics["both"]+=int(both)
            metrics["flip"]+=int(actions[0]!=actions[1])
            metrics["low_correct"]+=int(actions[0]==target[0])
            metrics["high_correct"]+=int(actions[1]==target[1])
            metrics["sham_flip"]+=int(shams!=actions)
            metrics["shuffled_both"]+=int(shuffled_actions==target)
            metrics["zero_both"]+=int(int(zero[i])==target[0]==target[1])
            metrics["blind_both"]+=int(int(bs[i])==target[0]==target[1])
            metrics["illegal"]+=int(any(a not in (0,1) for a in actions))
            receipt=dict(i=i,observation_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                observation=w.tolist(),legal_mask=[1,1],oracle_actions=target,
                oracle_gaps=[a[1] for a in oracle_both],oracle_costs=[a[2] for a in oracle_both],
                actions=actions,logits=[v[k][i].tolist() for k in (0,1)],
                correct_margins=margins,correct_confidence=conf,
                donor_forecasts=[f[k][0][i].tolist() for k in (0,1)],
                sham_donor_forecasts=[f[k][1][i].tolist() for k in (0,1)],
                sham_actions=shams,shuffled_actions=shuffled_actions,
                zero_action=int(zero[i]),blind_action=int(bs[i]),both_correct=both)
            handle.write(json.dumps(receipt,separators=(",",":"))+"\n")
            if not both:failures.append(dict(i=i,correct=target,actions=actions,margins=margins))
    m={k:v/n for k,v in metrics.items()}
    passing=(m["illegal"]==0 and m["sham_flip"]<=REQUIRED["sham_max"] and
             min(m["low_correct"],m["high_correct"])>=REQUIRED["each_law"] and
             float(np.mean(sqerr))<=REQUIRED["donor_mse_max"])
    if family==FAMILIES[0]:
        passing &= m["both"]>=REQUIRED["both_invariant"] and m["flip"]<=REQUIRED["flip_invariant_max"]
    else:
        passing &= (m["both"]>=REQUIRED["both_divergent"] and m["flip"]>=REQUIRED["flip_divergent"] and
                    m["shuffled_both"]<=REQUIRED["shuffled_max"] and
                    m["zero_both"]==m["blind_both"]==0)
    return dict(status="PASS" if passing else "FAIL",n=n,attempts=attempts,
        metrics=m,donor_tile_mse=float(np.mean(sqerr)),
        mean_oracle_margin=[float(np.mean(a)) for a in margin],
        mean_oracle_confidence=[float(np.mean(a)) for a in confidence],
        failure_count=len(failures),failures=failures)
def main():
    torch.set_num_threads(2)
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path("runtime/memplant/dg020-h3-input")
    out=Path(sys.argv[2]) if len(sys.argv)>2 else Path("runtime/memplant/dg020-h3")
    out.mkdir(parents=True,exist_ok=True)
    result={}
    for seed in SEEDS:
        donors,receiver,blind,prior,checks=load(root,seed)
        reports={f:run_family(seed,f,donors,receiver,blind,out) for f in FAMILIES}
        result[str(seed)]=dict(input_source_sha256=prior["source_sha256"],
            checkpoint_sha256={k:v for k,v in checks.items() if k.endswith(".pt")},
            reports=reports)
        print(seed,{f:reports[f]["status"] for f in FAMILIES},flush=True)
    verdict="PASS" if all(result[str(s)]["reports"][f]["status"]=="PASS" for s in SEEDS for f in FAMILIES) else "FAIL"
    report=dict(hypothesis="Frozen donor wet-law swap changes correct action only when it matters",
        preregistration="https://github.com/techrote/neurounner/pull/253#issuecomment-6935553280",
        no_retraining=True,source_sha256=digest(__file__),requirements=REQUIRED,
        verdict=verdict,results=result)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"sha256.json").write_text(json.dumps({p.name:digest(p) for p in sorted(out.glob("*")) if p.is_file()},indent=2)+"\n")
    print("SCIENTIFIC GATE",verdict,flush=True)
if __name__=="__main__":main()
