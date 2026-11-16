"""P4: independent original W11 behaviour—small CPU-trained future-event rankers.

Evidence: original Sprocket W11 Actions 43813501513, artifact 14201985226,
source c11211d6013de8dc6393c2ce1dbcc70eb344408b, ZIP SHA256
fbe090d3df4397b1a8c954d5f4b0503c76d003783d1b710ae8d1df81e839a3fe.
The frozen 45 episode rows below are first32/future96 event-derived features from
five actual PyTorch auxiliary-head policies in synthetic W0, independently audited
against 153 ZIP CRC / 152 member hashes, legal receipts, and original PNGs.
Remaining 27 original model-free controls are in the original ZIP, not trained on.
NO new model forwards, NO real native Godot material, NO human aesthetic votes.
"""
from __future__ import annotations
import hashlib, json, math, os, pathlib
from itertools import combinations

SOURCE_RUN=43813501513
ORIGINAL_ARTIFACT=14201985226
SOURCE_COMMIT="c11211d6013de8dc6393c2ce1dbcc70eb344408b"
ORIGINAL_ZIP_SHA="fbe090d3df4397b1a8c954d5f4b0503c76d003783d1b710ae8d1df81e839a3fe"
COMPACT_SHA="67e384be9fae8bec080ec22827107d9a8625857ce76939d2e9fa34382c153448"
# family(A/G/P), seed, controller(9/g/3/7/4), seven FIRST-32 features,
# FUTURE-96 new/branch/retrace, exact total turns, actual terminal STOP flag.
DATA="""A 1901 g 32 0 0 1 31 4 190 92 1 0 125 1
A 1913 g 32 0 0 2 30 10 188 80 3 0 125 1
A 1931 g 32 0 0 0 32 0 191 85 1 0 125 1
A 1901 3 29 4 0 2 25 0 190 56 9 12 125 1
A 1913 3 27 3 0 3 23 0 189 85 13 2 125 1
A 1931 3 29 3 0 2 26 9 190 84 7 0 125 1
A 1901 7 30 7 0 2 22 0 190 76 15 2 125 1
A 1913 7 28 6 0 2 21 0 189 71 12 10 125 1
A 1931 7 32 4 0 3 25 0 191 49 4 22 125 1
A 1901 4 30 8 0 3 15 2 188 82 15 4 125 1
A 1913 4 30 7 1 6 14 3 189 75 19 5 128 0
A 1931 4 31 6 0 5 15 10 185 64 12 6 125 1
A 1901 9 32 1 0 0 30 12 187 96 0 0 128 0
A 1913 9 32 0 0 0 32 9 186 96 3 0 128 0
A 1931 9 32 0 0 0 32 0 191 96 4 0 128 0
G 1901 g 24 0 1 14 2 0 184 57 0 13 125 1
G 1913 g 24 0 1 14 2 0 184 53 0 17 126 1
G 1931 g 24 0 1 14 2 2 184 53 0 17 126 1
G 1901 3 30 2 0 17 0 0 188 15 2 36 125 1
G 1913 3 17 4 7 11 1 0 177 38 3 28 126 1
G 1931 3 26 3 0 14 2 0 184 19 1 35 125 1
G 1901 7 28 4 0 12 1 12 186 65 10 9 125 1
G 1913 7 17 5 7 10 1 0 177 56 10 15 125 1
G 1931 7 28 5 1 8 7 0 187 65 13 10 125 1
G 1901 4 24 5 2 8 3 10 184 67 11 10 125 1
G 1913 4 24 8 3 7 4 0 181 72 14 4 125 1
G 1931 4 27 4 3 10 7 0 187 64 16 15 128 0
G 1901 9 32 8 0 0 20 2 190 71 27 8 128 0
G 1913 9 32 8 0 0 20 12 185 50 16 19 128 0
G 1931 9 32 8 0 0 20 8 189 32 9 31 128 0
P 1901 g 32 0 0 1 30 12 187 76 1 9 125 1
P 1913 g 32 0 0 1 29 11 185 89 0 0 125 1
P 1931 g 32 0 0 0 32 0 191 86 5 0 125 1
P 1901 3 32 3 0 1 28 0 191 65 5 14 125 1
P 1913 3 30 5 0 1 23 0 190 70 8 11 125 1
P 1931 3 29 5 0 1 23 8 189 73 8 4 125 1
P 1901 7 30 5 0 5 17 9 187 82 10 1 125 1
P 1913 7 29 8 0 1 19 0 187 73 11 7 125 1
P 1931 7 29 7 1 2 20 3 188 72 15 6 125 1
P 1901 4 29 7 1 6 10 0 186 74 14 6 126 1
P 1913 4 28 7 2 3 13 0 185 63 9 14 125 1
P 1931 4 31 9 0 5 12 0 188 54 13 20 126 1
P 1901 9 32 1 0 1 28 12 187 96 0 0 128 0
P 1913 9 32 0 0 0 32 9 186 96 0 0 128 0
P 1931 9 32 0 0 0 32 0 191 96 5 0 128 0
"""
FEATURES=("new","branch","retrace","right","straight","obstacle","legal")
SCALE=(32.,32.,32.,32.,32.,32.,224.)
FAMILIES={"A":"architecture","G":"greebles","P":"pipework"}
MODES={"9":"W9-greedy","g":"W10-greedy","3":"W10-T0.35","7":"W10-T0.70","4":"W10-T1.40"}
SEEDS=(1901,1913,1931)
AXES={"future_branch":2,"future_new":3}

def records(source=DATA):
    if hashlib.sha256(source.encode()).hexdigest()!=COMPACT_SHA:
        raise ValueError("frozen source-derived W11 45-row byte identity changed")
    rows=[];ids=set()
    for line in source.strip().splitlines():
        q=line.split()
        if len(q)!=15 or q[0] not in FAMILIES or q[2] not in MODES:
            raise ValueError("missing or malformed source row")
        f,seed,mode=q[:3];numbers=[int(z) for z in q[3:]]
        if seed not in [str(z) for z in SEEDS] or any(z<0 for z in numbers):
            raise ValueError("bad source seed or count")
        prefix=numbers[:7];fnew,fbranch,fretrace,turns,stop=numbers[7:]
        if not 32<=turns<=128 or turns<32+fbranch+fretrace or fnew>turns-32 or prefix[0]>32 or prefix[-1]>224 or stop not in (0,1):
            raise ValueError("impossible original event counts")
        key=(f,int(seed),mode)
        if key in ids:raise ValueError("duplicate model/World source")
        ids.add(key)
        rows.append({"family":f,"seed":int(seed),"mode":mode,"prefix":prefix,
           "future_new":fnew,"future_branch":fbranch,"future_retrace":fretrace,
           "turns":turns,"stopped":bool(stop)})
    if len(rows)!=45 or ids!={(f,s,m) for f in FAMILIES for s in SEEDS for m in MODES}:
        raise ValueError("incomplete original five-policy/three-family/three-seed cohort")
    return rows

def pairs(rows,axis):
    margin=AXES[axis];groups={}
    for r in rows:groups.setdefault((r["family"],r["seed"]),[]).append(r)
    out=[];ties=0
    for world,rr in sorted(groups.items()):
        for a,b in combinations(sorted(rr,key=lambda r:r["mode"]),2):
            d=a[axis]-b[axis]
            if abs(d)<margin:ties+=1;continue
            out.append((world,a,b,int(d>0)))
    return out,ties

def vector(row):return [a/d for a,d in zip(row["prefix"],SCALE)]

def fit(training,steps=240,lr=.04):
    if not training:raise ValueError("no training source pairs")
    examples=[]
    for _,a,b,y in training:
        d=[x-z for x,z in zip(vector(a),vector(b))]
        examples.extend(((d,y),([-v for v in d],1-y)))
    w=[0.]*7;m=[0.]*7;v=[0.]*7;losses=[]
    for t in range(1,steps+1):
        grad=[.01*x for x in w];loss=0.
        for d,y in examples:
            z=max(-25.,min(25.,sum(x*a for x,a in zip(w,d))))
            p=1/(1+math.exp(-z))
            loss-=y*math.log(max(p,1e-12))+(1-y)*math.log(max(1-p,1e-12))
            for i in range(7):grad[i]+=(p-y)*d[i]/len(examples)
        for i,g in enumerate(grad):
            m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
            w[i]-=lr*(m[i]/(1-.9**t))/(math.sqrt(v[i]/(1-.999**t))+1e-8)
        if t in (1,40,120,steps):losses.append([t,round(loss/len(examples),6)])
    return {"weights":w,"epochs":steps,"Adam_updates":steps,"balanced_training_pairs":len(examples),
       "features":FEATURES,"prefix_limit":32,"loss":losses,
       "target":"archived future W0 events, weak numeric labels; no human preference votes"}

def assess(model,test,axis):
    correct=ties=base_correct=base_eval=0
    j=0 if axis=="future_new" else 1
    for _,a,b,y in test:
        z=sum(w*(x-z) for w,x,z in zip(model["weights"],vector(a),vector(b)))
        if abs(z)<1e-9:ties+=1
        else:correct+=int((z>0)==bool(y))
        d=a["prefix"][j]-b["prefix"][j]
        if d:base_eval+=1;base_correct+=int((d>0)==bool(y))
    return {"pairs":len(test),"ranker_correct_non_ties":correct,"ranker_ties":ties,
       "trivial_prefix_correct":base_correct,"trivial_prefix_evaluable":base_eval,
       "trivial_prefix_ties":len(test)-base_eval}

def run(destination):
    rows=records();out=pathlib.Path(destination);out.mkdir(parents=True,exist_ok=True)
    report={"schema":"driv3r-preference-P4-W11/v1","producer_run":SOURCE_RUN,
      "original_artifact_id":ORIGINAL_ARTIFACT,"original_zip_sha256":ORIGINAL_ZIP_SHA,
      "source_commit":SOURCE_COMMIT,"derived_source_sha256":COMPACT_SHA,
      "genuine_pinned_PyTorch_head_episodes":45,"original_W11_head_forwards":5663,
      "original_W11_worlds":9,"original_W11_episode_count_including_controls":72,
      "synthetic_Python_W0":True,"new_model_forwards":0,"native_Godot_receipts":0,
      "human_preference_ratings":0,"objective":"future-event prediction, NOT artistic quality",
      "axes":{}}
    for axis in AXES:
        pp,margin_ties=pairs(rows,axis)
        folds=[]
        for seed in SEEDS:
            train=[p for p in pp if p[0][1]!=seed]
            test=[p for p in pp if p[0][1]==seed]
            folds.append({"unseen_seed":seed,"train_pairs":len(train),
                **assess(fit(train),test,axis)})
        head=fit(pp)
        (out/("P4-"+axis+"-head.json")).write_text(json.dumps(head,indent=2,sort_keys=True)+"\n")
        report["axes"][axis]={"matched_pairs":len(pp),"abstained_margin_ties":margin_ties,
            "heldout_seed_folds":folds,"exploratory_train_fit":assess(head,pp,axis),
            "actual_cpu_Adam_updates_including_three_folds":960}
    index={(r["family"],r["seed"],r["mode"]):r for r in rows}
    interventions=[]
    for f in FAMILIES:
        for s in SEEDS:
            g=index[f,s,"g"];t=index[f,s,"7"]
            interventions.append({"family":FAMILIES[f],"seed":s,
                "W10_greedy_future_branch":g["future_branch"],
                "W10_T07_future_branch":t["future_branch"],
                "W10_greedy_future_new":g["future_new"],
                "W10_T07_future_new":t["future_new"],
                "W10_greedy_stop":g["stopped"],"W10_T07_stop":t["stopped"]})
    report["matched_greedy_vs_sampled"]={"worlds":interventions,
         "T07_more_future_branches":sum(x["W10_T07_future_branch"]>x["W10_greedy_future_branch"] for x in interventions),
         "T07_more_future_new":sum(x["W10_T07_future_new"]>x["W10_greedy_future_new"] for x in interventions),
         "T07_less_future_new":sum(x["W10_T07_future_new"]<x["W10_greedy_future_new"] for x in interventions)}
    (out/"P4-W11-falsification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"SHA256SUMS.txt").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.name+"\n" for p in sorted(out.glob("*.json"))))
    return report

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);args=p.parse_args()
    print(json.dumps(run(args.output),indent=2))
