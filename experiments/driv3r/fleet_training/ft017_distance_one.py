"""FT-017: distance-one hard-negative STOP calibration on source-pinned tiny policies.

External synthetic game-policy host, not native CyberSand, W0, or pretrained SLM.
Matched Adam exposure: one arm sees far negatives, the other sees legally reached
distance-one predecessors; both see identical reached goals. Compare to the
immutable original gate, frozen backbone and non-learned exact-goal rule.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import fleet_bench as base
import ft004_shift as shift
import ft010_goal_reachability as goals
import ft014_stop_sidecar as side
import ft015_ood_stop as prior
import ft016_endings as endings

ID = "FT-017-20261116-distance-one-hard-negative"
SCHEMA = "cybersand.fleet-training.ft017-request/1"
HEAD_016 = "ad11cb0b5b1457503c888a227092cd6ce8e48ca5"
BLOB_016 = "3a4f3759fd479d15de63aa3c1fcb1a60af5171e9"
SEEDS, COHORTS, SCENARIOS = side.SEEDS, side.COHORTS, prior.SCENARIOS
ARMS = ("frozen_baseline", "distance_feature_gate", "plain_continuation",
        "distance_one_negatives", "exact_goal_rule")
TRAIN_ARMS = ("plain_continuation", "distance_one_negatives")
ROUTES = {"TRAIN":6, "VALIDATE":4, "HELDOUT":8}
SEARCH, AUTO, UPDATES = 64, 6, 600
MAX_SECONDS, MAX_RSS_KIB = 360, 900000

def expected_request():
    r=json.loads((Path(__file__).parent/"request_ft016.json").read_text())
    r.pop("ft015_source_head")
    r.pop("ft015_git_blob")
    r.pop("worlds_per_scenario")
    r.pop("legal_scaffold_probes_per_scenario")
    r.pop("optimizer_updates")
    r.update(schema=SCHEMA, request_id=ID, ft016_source_head=HEAD_016,
             ft016_git_blob=BLOB_016, arms=list(ARMS),
             route_targets_per_scenario=ROUTES, max_route_search_per_split=SEARCH,
             autonomous_worlds_per_scenario=AUTO, adam_updates_per_new_sidecar=UPDATES)
    return r

def check_request(r):
    if type(r) is not dict or r!=expected_request():
        raise ValueError("FT-017 source/checkpoint/split/budget drift")
    return r

def lease(seed, cohort, niche, arm):
    if seed not in SEEDS or cohort not in COHORTS or niche not in COHORTS[cohort] or arm not in TRAIN_ARMS:
        raise ValueError("Unowned FT-017 model lease")
    return f"{ID}/{seed}/{cohort}/{niche}/{arm}"

def world_seed(pop, split, seed, niche, scenario, i):
    if split not in (*ROUTES,"AUTONOMOUS") or seed not in SEEDS or niche not in base.NICHES or scenario not in SCENARIOS or not isinstance(i,int) or not 0<=i<SEARCH:
        raise ValueError("Invalid research world lease")
    return pop.stable_seed(ID,split,seed,niche,scenario,i)

def budget(t0):
    if time.monotonic()-t0>MAX_SECONDS:
        raise TimeoutError("FT-017 exceeded job wall budget")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>MAX_RSS_KIB:
        raise MemoryError("FT-017 exceeded RSS ceiling")

def route_states(pop,sid,scenario):
    w=prior.make_world(pop,sid,scenario)
    path=goals.legal_route(pop,w)
    if not path: return None
    rows=[]
    for action in path:
        d=side.distance(w)
        if d==0 or w.switched or action not in w.legal():
            raise AssertionError("Illegal pre-goal teacher route")
        rows.append((side.features(w,True).copy(),0,d))
        w.step(action)
    if side.distance(w)!=0 or w.switched or "STOP" not in w.legal():
        raise AssertionError("Independent route failed to reach active goal")
    rows.append((side.features(w,True).copy(),1,0))
    return rows

def make_routes(pop,split,seed,niche,scenario,used):
    found=[]
    for i in range(SEARCH):
        sid=world_seed(pop,split,seed,niche,scenario,i)
        if sid in used: raise AssertionError("Duplicate world seed across splits")
        used.add(sid)
        rows=route_states(pop,sid,scenario)
        if rows is None or not any(d==1 for x,y,d in rows if y==0) or not any(d>=2 for x,y,d in rows if y==0):
            continue
        found.append((sid,rows))
        if len(found)==ROUTES[split]: break
    if len(found)!=ROUTES[split]:
        raise AssertionError(f"Insufficient legal {split} routes {niche}/{scenario}: {len(found)}")
    return found

def train_data(routes,arm):
    if arm not in TRAIN_ARMS: raise ValueError("Unknown optimizer arm")
    samples=[]
    for sid,rows in routes:
        far=[x for x,y,d in rows if y==0 and d>=2]
        near=[x for x,y,d in rows if y==0 and d==1]
        positives=[x for x,y,d in rows if y==1 and d==0]
        if not far or not near or len(positives)!=1:
            raise AssertionError("Unpaired reached-goal case")
        samples.extend(((positives[0],1), (far[0],0),
                        (far[-1] if arm==TRAIN_ARMS[0] else near[-1],0)))
    X=np.stack([x for x,y in samples])
    y=np.asarray([y for x,y in samples],dtype=np.float64)
    if X.shape!=(len(routes)*3,21): raise AssertionError("Non-matched batch size")
    return X,y

def all_data(routes):
    rows=[row for sid,route in routes for row in route]
    X=np.stack([x for x,y,d in rows])
    y=np.asarray([y for x,y,d in rows])
    d=np.asarray([d for x,y,d in rows])
    if not np.array_equal(y,(d==0).astype(int)): raise AssertionError("Corrupt goal labels")
    return X,y,d

def calibrate(gate,X,y):
    p=gate.probability(X)
    negatives=p[y==0]
    positives=p[y==1]
    if not len(negatives) or not len(positives) or not np.isfinite(p).all():
        raise ValueError("Empty or non-finite validation labels")
    c=np.unique(np.concatenate(([0.,1.],p,np.nextafter(p,1))))
    best=None
    for t in c:
        fp=float(np.mean(negatives>=t))
        if fp>0.01+1e-12: continue
        recall=float(np.mean(positives>=t))
        key=(recall,-fp,-float(t))
        if best is None or key>best[0]: best=(key,float(t))
    gate.threshold=best[1]
    return {"threshold":float(gate.threshold),"validation_recall":best[0][0],
            "validation_fpr":-best[0][1],"positives":len(positives),
            "negatives":len(negatives)}

def heldout(gate,X,y,d):
    stopped=gate.probability(X)>=gate.threshold
    return {"goal_tp":int(np.sum(stopped & (y==1))),"goal_total":int(np.sum(y==1)),
            "distance_one_fp":int(np.sum(stopped & (d==1))),
            "distance_one_total":int(np.sum(d==1)),
            "far_fp":int(np.sum(stopped & (d>=2))),
            "far_total":int(np.sum(d>=2))}

def episode(pop,frozen,gates,arm,sid,niche,scenario,trace):
    # Evaluate each new STOP model through exactly the same frozen spatial
    # policy and host legal menu as the original distance gate.
    if arm in TRAIN_ARMS:
        chosen={**gates,"distance_feature_gate":gates[arm]}
        effective="distance_feature_gate"
    else:
        chosen,effective=gates,arm
    ep,w=endings.episode(pop,frozen,chosen,sid,niche,scenario,effective,trace)
    ep["arm"]=arm
    return ep,w

def run(request,upstream,ft003,ft014_archive,output,seed,cohort):
    check_request(json.loads(request.read_text()))
    if seed not in SEEDS or cohort not in COHORTS: raise ValueError("Foreign job lease")
    pop,source=shift.load_source(upstream)
    if len(pop.FEATURES)!=20 or pop.FEATURES[3]!="stop":
        raise AssertionError("Pinned host observation mismatch")
    output.mkdir(parents=True,exist_ok=False)
    t0=time.monotonic()
    report={"schema":"cybersand.fleet-training.ft017-evidence/1",
            "request_id":ID,"lease":f"{ID}/{seed}/{cohort}",
            "seed":seed,"cohort":cohort,"status":"starting",
            "classification":"REAL_NUMPY_ADAM_TINY_STOP_MODEL_EXTERNAL_SYNTHETIC_NOT_NATIVE_NOT_W0_NOT_SLM",
            "source_commit":shift.SOURCE_COMMIT,"source_blob":shift.SOURCE_BLOB,
            "source_sha256":base.sha_file(source),
            "ft016_source_head":HEAD_016,"ft016_git_blob":BLOB_016,
            "ft014_artifact_id":prior.ARTIFACTS[f"{seed}/{cohort}"],
            "ft003_zip_sha256":shift.FT003_SHA,
            "executor_sha256":base.sha_file(Path(__file__)),
            "request_sha256":base.sha_file(request),
            "optimizer_updates":0,"native_material_edits":0,
            "resident_training_processes_per_runner":1,
            "workers":[],"groups":[],"failures":[]}
    try:
        original,custody=prior.archive_audit(ft014_archive,seed,cohort)
        report["ft014_custody"]=custody
        report["ft003_custody"]=shift.archive_check(
            ft003,source,request.parent/"request_dose.json")
        used=set()
        historical=json.loads((ft014_archive/"run"/"report.json").read_text())["workers"]
        for niche in COHORTS[cohort]:
            checkpoint=ft003/f"dose_48/{seed}/models/{niche}.json"
            frozen=shift.load_policy(pop,checkpoint)
            frozen_hash=base.sha_file(checkpoint)
            matching=[w for w in historical if w["niche"]==niche]
            if not matching or any(w["backbone_checkpoint_sha256"]!=frozen_hash for w in matching):
                raise AssertionError("Frozen policy lineage mismatch")
            routes={split:[] for split in ROUTES}
            for scenario in SCENARIOS:
                for split in ROUTES:
                    routes[split].extend(make_routes(pop,split,seed,niche,scenario,used))
                    budget(t0)
            Xv,yv,dv=all_data(routes["VALIDATE"])
            Xt,yt,dt=all_data(routes["HELDOUT"])
            basegate=original[(niche,"distance_feature_gate")]
            backbone_sha=custody["model_sha256"][f"{niche}/distance_feature_gate"]
            gates={"distance_feature_gate":basegate,
                   "legacy_feature_gate":original[(niche,"legacy_feature_gate")]}
            report["workers"].append({"lease":f"{ID}/{seed}/{cohort}/{niche}/frozen",
                "arm":"distance_feature_gate","niche":niche,
                "backbone_sha256":frozen_hash,"original_gate_sha256":backbone_sha,
                "optimizer_updates":0,"heldout":heldout(basegate,Xt,yt,dt)})
            for arm in TRAIN_ARMS:
                X,y=train_data(routes["TRAIN"],arm)
                model=side.StopGate(True)
                # Weight continuation, new bounded optimizer moments; original is immutable.
                model.w=basegate.w.copy()
                for j in range(UPDATES):
                    model.teach(X,y)
                    if j%100==0: budget(t0)
                if model.steps!=UPDATES or not np.isfinite(model.w).all():
                    raise AssertionError("Optimizer progress missing")
                validation=calibrate(model,Xv,yv)
                score=heldout(model,Xt,yt,dt)
                payload=model.serialize()
                payload["lineage_sha256"]=backbone_sha
                payload["curriculum"]=arm
                dest=output/"models"/niche/(arm+".json")
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(json.dumps(payload,sort_keys=True)+"\n")
                gates[arm]=model
                report["optimizer_updates"]+=model.steps
                report["workers"].append({"lease":lease(seed,cohort,niche,arm),
                    "arm":arm,"niche":niche,"backbone_sha256":frozen_hash,
                    "original_gate_sha256":backbone_sha,
                    "trained_checkpoint_sha256":base.sha_file(dest),
                    "train_rows":len(y),"train_positive":int(y.sum()),
                    "optimizer_updates":model.steps,"validation":validation,
                    "heldout":score})
            for scenario in SCENARIOS:
                cases={arm:[] for arm in ARMS}
                for i in range(AUTO):
                    sid=world_seed(pop,"AUTONOMOUS",seed,niche,scenario,i)
                    if sid in used: raise AssertionError("Duplicate autonomous lease")
                    used.add(sid)
                    for arm in ARMS:
                        ep,w=episode(pop,frozen,gates,arm,sid,niche,scenario,i==0)
                        if i==0:
                            prior.audit_trace(pop,sid,scenario,ep["receipts"],w)
                            directory=output/"gallery"/niche/scenario
                            directory.mkdir(parents=True,exist_ok=True)
                            (directory/(arm+".svg")).write_text(pop.draw_svg(w))
                            (directory/(arm+".json")).write_text(json.dumps(ep,sort_keys=True)+"\n")
                        cases[arm].append(ep)
                    budget(t0)
                directory=output/"episodes"/niche
                directory.mkdir(parents=True,exist_ok=True)
                (directory/(scenario+".json")).write_text(json.dumps(cases,sort_keys=True)+"\n")
                report["groups"].append({"niche":niche,"scenario":scenario,
                   "by_arm":{arm:endings.summarize(cases[arm]) for arm in ARMS}})
            if base.sha_file(checkpoint)!=frozen_hash:
                raise AssertionError("Frozen base weights mutated")
        leases=[w["lease"] for w in report["workers"]]
        if len(leases)!=len(set(leases)): raise AssertionError("Duplicate model lease")
        metrics=("goal_tp","goal_total","distance_one_fp","distance_one_total","far_fp","far_total")
        report["aggregate"]={"trained_checkpoint_count":sum(
            w["arm"] in TRAIN_ARMS for w in report["workers"]),
            "optimizer_updates":report["optimizer_updates"],
            "heldout":{arm:{k:sum(w["heldout"][k] for w in report["workers"] if w["arm"]==arm)
                for k in metrics} for arm in ("distance_feature_gate",*TRAIN_ARMS)},
            "by_arm":{arm:{
                "voluntary_goal_stop":sum(g["by_arm"][arm]["endings"]["voluntary_goal_stop"] for g in report["groups"]),
                "voluntary_nongoal_stop":sum(g["by_arm"][arm]["endings"]["voluntary_nongoal_stop"] for g in report["groups"]),
                "forced_nongoal_stop":sum(g["by_arm"][arm]["endings"]["forced_only_stop_nongoal"] for g in report["groups"]),
                "mean_new_ink":round(sum(g["by_arm"][arm]["mean_ink"] for g in report["groups"])/len(report["groups"]),5),
                "mean_total_utility":round(sum(g["by_arm"][arm]["mean_total_shaped_utility"] for g in report["groups"])/len(report["groups"]),5)}
                for arm in ARMS},
            "unique_seed_leases":len(used),
            "autonomous_episodes":len(report["groups"])*AUTO*len(ARMS),
            "elapsed_seconds":round(time.monotonic()-t0,5),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        report["status"]="completed"
    except BaseException as exc:
        report["status"]="failed"
        report["failures"].append(type(exc).__name__+": "+str(exc))
        raise
    finally:
        (output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        (output/"SHA256SUMS").write_text("".join(
            base.sha_file(f)+"  "+str(f.relative_to(output))+"\n"
            for f in sorted(output.rglob("*")) if f.is_file() and f.name!="SHA256SUMS"))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--request",type=Path,required=True)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--ft003",type=Path,required=True)
    p.add_argument("--ft014",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--cohort",choices=COHORTS,required=True)
    a=p.parse_args()
    r=run(a.request,a.upstream,a.ft003,a.ft014,a.output,a.seed,a.cohort)
    print(json.dumps({"lease":r["lease"],"status":r["status"],"aggregate":r["aggregate"]},sort_keys=True))
