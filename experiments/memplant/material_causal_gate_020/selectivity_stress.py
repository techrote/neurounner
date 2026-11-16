"""DG020 frozen material-law selectivity stress; preregistered on PR #253.

Uses ORIGINAL trained models without fine-tuning or checkpoint selection.
Reports a scientific FAIL rather than silently dropping nondivergent worlds.
No code from CyberSand and no game/runtime modifications.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiment import Donor, Receiver, oracle, WET_COST, DRY_COST, TILES

FAMILIES = ("offset_differential", "equal_wet_fee", "toll_dominated")
FROZEN_SOURCE = "5e242429d4e6eed5c44a5d25669f7304971dab4dd48cf2f33635d60ade6322bf"
FROZEN_HEAD = "3d1571fde6865d65891242ef03fd0b35a25e55a5"
SEEDS = (26011636, 26011637, 26011638)
MIN_GAP = .60
N = 160
LIMITS = dict(diff_both=.90, diff_flip=.95, diff_each=.94,
              diff_sham=.05, diff_shuffle=.60, same_both=.94,
              same_flip=.05, same_sham=.05, mse=.02)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return sha(Path(path).read_bytes())


def islands(rng, k):
    """Two separated rotated/reflected islands rather than training short-runs."""
    n1 = int(rng.integers(1, k))
    gap = int(rng.integers(1, 12-k))
    w = np.zeros(TILES, np.float32)
    w[:n1] = 1
    w[n1+gap:k+gap] = 1
    w = np.roll(w, int(rng.integers(TILES)))
    if rng.integers(2):
        w = w[::-1].copy()
    assert w.sum() == k
    return w


def candidate(rng, family):
    wet = np.zeros((2,TILES),np.float32)
    if family == "equal_wet_fee":
        k = int(rng.integers(2,10))
        wet[:] = (islands(rng,k), islands(rng,k))
        favored = int(rng.integers(2))
        fee = np.empty((2,TILES),np.float32)
        fee[favored] = rng.uniform(0,.105,TILES)
        fee[1-favored] = rng.uniform(.195,.30,TILES)
    elif family == "toll_dominated":
        extra = int(rng.integers(1,3))
        k = int(rng.integers(2,10-extra))
        rich = int(rng.integers(2))
        wet[rich] = islands(rng,k+extra)
        wet[1-rich] = islands(rng,k)
        fee = np.empty((2,TILES),np.float32)
        fee[rich] = rng.uniform(.205,.30,TILES)
        fee[1-rich] = rng.uniform(0,.095,TILES)
    elif family == "offset_differential":
        extra = int(rng.integers(1,4))
        k = int(rng.integers(2,10-extra))
        rich = int(rng.integers(2))
        wet[rich] = islands(rng,k+extra)
        wet[1-rich] = islands(rng,k)
        fee = rng.uniform(0,.30,(2,TILES)).astype(np.float32)
    else:
        raise ValueError(family)
    texture = rng.uniform(-1,1,(2,TILES,2)).astype(np.float32)
    observation = np.concatenate((wet[...,None],fee[...,None],texture),-1)
    return observation, int(rng.integers(1,5))


def worlds(seed, family, n=N):
    rng = np.random.default_rng(seed+87_000_000+800_000*FAMILIES.index(family))
    accepted=[]
    attempts=0
    divergent=family=="offset_differential"
    while len(accepted)<n and attempts<n*1000:
        x,phase=candidate(rng,family)
        attempts+=1
        a,ga,_=oracle(x,0)
        b,gb,_=oracle(x,1)
        if (a!=b)!=divergent or min(ga,gb)<MIN_GAP:
            continue
        if family=="equal_wet_fee":
            assert x[0,:,0].sum()==x[1,:,0].sum()
        if family=="toll_dominated":
            rich=int(np.argmax(x[:,:,0].sum(-1)))
            assert x[rich,:,1].mean()>x[1-rich,:,1].mean()
        accepted.append((x,phase,attempts))
    assert len(accepted)==n, (family,len(accepted),attempts)
    return accepted,attempts


def load_frozen(folder):
    folder=Path(folder)
    assert (folder/"source.sha256").read_text().split()[0]==FROZEN_SOURCE
    original=json.loads((folder/"report.json").read_text())
    assert original["source_sha256"]==FROZEN_SOURCE and original["seed_status"]=="PASS"
    hashes=json.loads((folder/"sha256.json").read_text())
    for name,expected in hashes.items():
        assert file_hash(folder/name)==expected,name
    donors=[[],[]]
    for law in (0,1):
        for rep in (0,1):
            m=Donor().eval()
            m.load_state_dict(torch.load(folder/f"donor-wet-{law}-replica-{rep}.pt",
                                         map_location="cpu",weights_only=True),strict=True)
            donors[law].append(m)
    rec=Receiver().eval()
    rec.load_state_dict(torch.load(folder/"receiver-conditioned.pt",
                                   map_location="cpu",weights_only=True),strict=True)
    blind=Receiver().eval()
    blind.load_state_dict(torch.load(folder/"receiver-law-blind.pt",
                                     map_location="cpu",weights_only=True),strict=True)
    return donors,rec,blind,{k:file_hash(folder/k) for k in hashes}


def verify_replay(folder, models):
    donors,rec,_,_=models
    for fam in ("edge_bands","alternating"):
        prior=[json.loads(s) for s in (Path(folder)/f"receipts-{fam}.jsonl").read_text().splitlines()]
        obs=torch.from_numpy(np.asarray([r["observation"] for r in prior],np.float32))
        with torch.no_grad():
            for law in (0,1):
                forecast=donors[law][0](obs.reshape(-1,4)).reshape(-1,2,TILES)
                old=np.asarray([r["local_donor_predictions"][law] for r in prior],np.float32)
                assert np.allclose(forecast.numpy(),old,rtol=1e-5,atol=2e-5)
                predictions=rec(obs,forecast).argmax(-1).tolist()
                assert predictions==[r["receiver_actions"][law] for r in prior]
    return 320


def test_family(seed,family,samples,models,out):
    donors,rec,blind,_=models
    x=torch.from_numpy(np.stack([w[0] for w in samples]).astype(np.float32))
    rng=np.random.default_rng(seed+94_000_000+FAMILIES.index(family)*10_000)
    with torch.no_grad():
        cost=[donors[k][0](x.reshape(-1,4)).reshape(-1,2,TILES) for k in (0,1)]
        sham=[donors[k][1](x.reshape(-1,4)).reshape(-1,2,TILES) for k in (0,1)]
        logits=[rec(x,c) for c in cost]
        shamlogits=[rec(x,c) for c in sham]
        zero=torch.zeros_like(cost[0])
        zero_logits=rec(x,zero)
        blind_logits=blind(x,zero)
        shuffled=[torch.stack([cost[int(rng.integers(2))][i]
                    for i in range(len(samples))]) for _ in (0,1)]
        shuffled_logits=[rec(x,c) for c in shuffled]
    actions=[z.argmax(-1).tolist() for z in logits]
    sham_actions=[z.argmax(-1).tolist() for z in shamlogits]
    shuffled_actions=[z.argmax(-1).tolist() for z in shuffled_logits]
    zactions=zero_logits.argmax(-1).tolist()
    bactions=blind_logits.argmax(-1).tolist()
    keys=("both","flip","correct_low","correct_high","sham_flip",
          "sham_both","shuffled_both","zero_both","blind_both","illegal","identity")
    totals={key:0 for key in keys}
    tile_sq_errors=[[],[]]
    margins=[[],[]]
    conf=[[],[]]
    left_shifts=[]
    failures=[]
    with open(out,"w",encoding="utf8") as stream:
        for i,(obs,phase,attempt) in enumerate(samples):
            a,ga,ca=oracle(obs,0)
            b,gb,cb=oracle(obs,1)
            assert (a!=b)==(family=="offset_differential")
            assert min(ga,gb)>=MIN_GAP
            acts=[actions[0][i],actions[1][i]]
            sh=[sham_actions[0][i],sham_actions[1][i]]
            shuffled=[shuffled_actions[0][i],shuffled_actions[1][i]]
            both=int(acts==[a,b])
            totals["both"]+=both
            totals["flip"]+=int(acts[0]!=acts[1])
            totals["correct_low"]+=int(acts[0]==a)
            totals["correct_high"]+=int(acts[1]==b)
            totals["sham_flip"]+=int(sh!=acts)
            totals["sham_both"]+=int(sh==[a,b])
            totals["shuffled_both"]+=int(shuffled==[a,b])
            totals["zero_both"]+=int(zactions[i]==a==b)
            totals["blind_both"]+=int(bactions[i]==a==b)
            totals["illegal"]+=int(any(k not in (0,1) for k in acts))
            totals["identity"]+=1
            for law in (0,1):
                actual=np.where(obs[...,0]>.5,WET_COST[law],DRY_COST)+obs[...,1]
                tile_sq_errors[law].extend(((cost[law][i].numpy()-actual)**2).reshape(-1).tolist())
                target=(a,b)[law]
                margins[law].append(float(logits[law][i,target]-logits[law][i,1-target]))
                conf[law].append(float(F.softmax(logits[law][i],-1)[target]))
            left_shifts.append(float((logits[1][i,0]-logits[1][i,1])-
                                     (logits[0][i,0]-logits[0][i,1])))
            if not both:
                failures.append(dict(index=i,oracle=[a,b],receiver=acts,gaps=[ga,gb]))
            stream.write(json.dumps(dict(index=i,family=family,phase=phase,attempt=attempt,
                 host_prefix=["MOVE_FORWARD"]*phase,legal_mask=[1,1],
                 action_names=["LEFT_ROUTE","RIGHT_ROUTE"],
                 shared_observation_sha256=sha(obs.tobytes()), observation=obs.tolist(),
                 oracle_actions=[a,b],oracle_margins=[ga,gb],oracle_route_costs=[ca,cb],
                 receiver_actions=acts,donor_predictions=[cost[k][i].tolist() for k in (0,1)],
                 donor_prediction_sha256=[sha(cost[k][i].numpy().tobytes()) for k in (0,1)],
                 logits=[v[i].tolist() for v in logits],
                 correct_action_logit_margins=[margins[k][-1] for k in (0,1)],
                 oracle_action_confidence=[conf[k][-1] for k in (0,1)],
                 sham_actions=sh,shuffled_actions=shuffled,
                 zero_action=zactions[i],blind_action=bactions[i],
                 both_correct=bool(both)),separators=(",",":"))+"\n")
    n=len(samples)
    m={k:round(v/n,6) for k,v in totals.items()}
    dmse=[float(np.mean(v)) for v in tile_sq_errors]
    if family=="offset_differential":
        pass_gate=(m["both"]>=LIMITS["diff_both"] and m["flip"]>=LIMITS["diff_flip"]
                   and m["correct_low"]>=LIMITS["diff_each"]
                   and m["correct_high"]>=LIMITS["diff_each"]
                   and m["sham_flip"]<=LIMITS["diff_sham"]
                   and m["shuffled_both"]<=LIMITS["diff_shuffle"]
                   and m["blind_both"]==0 and m["zero_both"]==0)
    else:
        pass_gate=(m["both"]>=LIMITS["same_both"] and m["flip"]<=LIMITS["same_flip"]
                   and m["sham_flip"]<=LIMITS["same_sham"])
    pass_gate=bool(pass_gate and all(e<=LIMITS["mse"] for e in dmse)
                   and m["illegal"]==0 and m["identity"]==1)
    return dict(n=n,metrics=m,donor_mse=dmse,
        mean_correct_margin=[float(np.mean(v)) for v in margins],
        mean_oracle_confidence=[float(np.mean(v)) for v in conf],
        mean_left_margin_shift=float(np.mean(left_shifts)),
        mean_absolute_left_margin_shift=float(np.mean(np.abs(left_shifts))),
        failures=failures,verdict="PASS" if pass_gate else "FAIL")


def run(seed,frozen,out,n=N):
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    models=load_frozen(frozen)
    replayed=verify_replay(frozen,models)
    by_family={}
    for family in FAMILIES:
        cases,attempts=worlds(seed,family,n)
        result=test_family(seed,family,cases,models,out/f"receipts-{family}.jsonl")
        result["candidates_examined"]=attempts
        by_family[family]=result
        print(json.dumps(dict(seed=seed,family=family,verdict=result["verdict"],
                              metrics=result["metrics"],donor_mse=result["donor_mse"])))
    verdict="PASS" if n>=N and all(x["verdict"]=="PASS" for x in by_family.values()) else "FAIL"
    report=dict(experiment="DG020-frozen-selectivity-stress",seed=seed,
        frozen_head=FROZEN_HEAD,frozen_source_sha256=FROZEN_SOURCE,
        audit_source_sha256=file_hash(__file__),original_receipts_replayed=replayed,
        frozen_checkpoint_sha256=models[3],trained_models_modified=False,
        fixed_topology="identical two 12-tile routes, fixed LEFT/RIGHT legal mask",
        preregistration="https://github.com/techrote/neurounner/pull/253#issuecomment-6931232665",
        thresholds=LIMITS,per_family=by_family,verdict=verdict,
        scope="synthetic material-only selectivity, not KV transfer or planning")
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    checks={p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="sha256.json"}
    (out/"sha256.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(dict(seed=seed,stress_verdict=verdict,report_sha256=file_hash(out/"report.json"))))


def selftest():
    for fam in FAMILIES:
        case,_=worlds(SEEDS[0],fam,N)
        assert len(case)==N
        for x,phase,_ in case:
            assert x.shape==(2,12,4) and 1<=phase<=4
            a,ga,_=oracle(x,0);b,gb,_=oracle(x,1)
            assert (a!=b)==(fam=="offset_differential") and min(ga,gb)>=MIN_GAP
    print("SELECTIVITY SELFTEST PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--frozen")
    p.add_argument("--out")
    p.add_argument("--pairs",type=int,default=N)
    args=p.parse_args()
    if args.selftest:
        selftest()
    else:
        if args.seed is None or not args.frozen or not args.out:
            p.error("--seed, --frozen, and --out required")
        run(args.seed,args.frozen,args.out,args.pairs)
