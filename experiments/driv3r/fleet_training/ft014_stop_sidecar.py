"""FT-014: independently learned STOP sidecar, matched legacy-feature ablation.

This is external synthetic CyberSand-like research only. The FT-003 24-unit spatial
policies are source-pinned and frozen. Two genuinely optimized logistic sidecars
differ ONLY in whether the host supplies current-goal Manhattan distance.
No edits to native game code, authoring tools or production providers.
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
import ft010_goal_reachability as goal

ID = "FT-014-20261116-goal-distance-stop-sidecar"
SCHEMA = "cybersand.fleet-training.stop-sidecar-request/1"
ARMS = ("frozen_baseline", "legacy_feature_gate", "distance_feature_gate")
GATE_ARMS = ARMS[1:]
SEEDS = (9401, 9403)
COHORTS = {
    "navigation": ("navigator", "recoverer", "material_cautious"),
    "art": ("branch_artist", "texture_worm", "chaotic_brush"),
}
SCENARIOS = ("long60", "dense60", "shifted_guard60")
SPLITS = ("TRAIN", "VALIDATION", "HELDOUT", "MINE", "AUTONOMOUS")
ROUTE_COUNTS = {"TRAIN": 6, "VALIDATION": 4, "HELDOUT": 6}
MINE_WORLDS = 3
AUTO_WORLDS = 6
POOL = 48
STEPS = 600
LR = 0.075
L2 = 0.0001
MAX_SECONDS = 300
MAX_RSS_KIB = 900000
MAX_PARALLEL = 4


def checked_request(r):
    want = {
        "schema": SCHEMA, "request_id": ID,
        "source_commit": shift.SOURCE_COMMIT,
        "source_blob": shift.SOURCE_BLOB,
        "ft003_run_id": 43797061143,
        "ft003_artifact_id": 14194214028,
        "ft003_zip_sha256": shift.FT003_SHA,
        "seeds": list(SEEDS),
        "cohorts": {k: list(v) for k, v in COHORTS.items()},
        "scenarios": list(SCENARIOS),
        "arms": list(ARMS),
        "route_examples_per_scenario": ROUTE_COUNTS,
        "onpolicy_mining_worlds_per_scenario": MINE_WORLDS,
        "autonomous_worlds_per_scenario": AUTO_WORLDS,
        "optimizer_updates_per_sidecar": STEPS,
        "adam_learning_rate": LR, "l2": L2,
        "validation_false_positive_cap": 0.01,
        "max_seconds_per_job": MAX_SECONDS,
        "max_rss_kib_per_job": MAX_RSS_KIB,
        "max_parallel_runner_jobs": MAX_PARALLEL,
        "native_material_edits": 0,
        "pretrained_slm_optimizer_steps": 0,
    }
    if type(r) is not dict or r != want:
        raise ValueError("FT-014 immutable source/checkpoint/split/budget mismatch")
    return r


def budget(t0):
    if time.monotonic()-t0 > MAX_SECONDS:
        raise TimeoutError("FT-014 job wall deadline")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_KIB:
        raise MemoryError("FT-014 RSS cap")


def lease(seed, niche, arm):
    if seed not in SEEDS or niche not in base.NICHES or arm not in GATE_ARMS:
        raise ValueError("Invalid FT-014 model lease")
    return f"{ID}/{seed}/{niche}/{arm}"


def world_seed(pop, split, seed, niche, scenario, i):
    if (split not in SPLITS or seed not in SEEDS or niche not in base.NICHES
            or scenario not in SCENARIOS or type(i) is not int or not 0 <= i < POOL):
        raise ValueError("Invalid FT-014 world lease")
    return pop.stable_seed(ID, split, seed, niche, scenario, i)


def distance(w):
    return abs(w.cursor[0]-w.goal[0]) + abs(w.cursor[1]-w.goal[1])


def features(w, geometry):
    """Only an externally readable, host-legal STOP observation is accessed.

    The pre-existing 20 features omit absolute goal distance. Geometry variant
    adds that scalar as a proposed external model interface, NOT a game edit.
    """
    if "STOP" not in w.legal():
        raise AssertionError("No host-legal STOP observation")
    x = w.feature("STOP")
    if x.shape != (20,) or not np.isfinite(x).all():
        raise AssertionError("Pinned host observation drift")
    if not geometry:
        return x.copy()
    return np.concatenate((x, np.array([distance(w)/3.], dtype=np.float64)))


class StopGate:
    """Separate Adam-trained binary STOP head; base 24-unit action MLP stays frozen."""
    def __init__(self, geometry):
        self.geometry = bool(geometry)
        self.w = np.zeros(21 if geometry else 20, dtype=np.float64)
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        self.steps = 0
        self.threshold = 0.5

    def probability(self, x):
        z = np.clip(np.asarray(x, dtype=np.float64) @ self.w, -35, 35)
        return 1/(1+np.exp(-z))

    def teach(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.w) or y.shape != (len(X),):
            raise ValueError("Malformed real sidecar training batch")
        if len(X) < 2 or not (np.any(y==0) and np.any(y==1)):
            raise ValueError("Training batch missing a genuine class")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("Nonfinite training batch")
        weight = np.where(y==1, 0.5/np.mean(y==1), 0.5/np.mean(y==0))
        predicted = self.probability(X)
        grad = (X.T @ (weight*(predicted-y)))/len(X)
        ridge = self.w.copy()
        ridge[0] = 0
        grad += L2*ridge
        self.steps += 1
        self.m = .9*self.m + .1*grad
        self.v = .999*self.v + .001*grad*grad
        mhat = self.m/(1-.9**self.steps)
        vhat = self.v/(1-.999**self.steps)
        self.w -= LR*mhat/(np.sqrt(vhat)+1e-8)

    def fit(self, cases):
        X = np.stack([features(w, self.geometry) for w,y in cases])
        y = np.asarray([int(y) for w,y in cases],dtype=np.float64)
        for _ in range(STEPS):
            self.teach(X,y)
        return {"adam_updates":self.steps, "examples":len(y),
                "terminal":int(y.sum()), "nonterminal":int((1-y).sum()),
                "balanced_bce":round(float(
                    -.5*np.log(np.maximum(self.probability(X[y==1]),1e-12)).mean()
                    -.5*np.log(np.maximum(1-self.probability(X[y==0]),1e-12)).mean()
                ),6)}

    def serialize(self):
        if self.steps != STEPS or not np.isfinite(self.w).all():
            raise AssertionError("Sidecar is not genuinely trained")
        return {
            "format":"cybersand-research-stop-sidecar/1",
            "base_features":["bias","paint","retrace","stop","goal_progress","onward",
                 "blocked","junction","turn","frontier","material_edge",
                 "paint_density","early","protected_near","goal_switched",
                 "direction_x","direction_y","guidance","motif","recovery_need"],
            "added_feature":"manhattan_goal_distance_div_3" if self.geometry else None,
            "w":self.w.tolist(),
            "adam_updates":self.steps,"threshold":float(self.threshold),
            "optimizer":"full_batch_adam_class_balanced_lr_0.075",
        }


def route_cases(pop, seed, niche, scenario, split, used):
    wanted = ROUTE_COUNTS[split]
    found = []
    for i in range(POOL):
        sid = world_seed(pop,split,seed,niche,scenario,i)
        if sid in used:
            raise AssertionError("Cross-split world collision")
        used.add(sid)
        route = goal.legal_route(pop,shift.world_for(pop,sid,scenario))
        if route is None:
            continue
        if not route:
            raise AssertionError("Route unexpectedly starts at goal")
        found.append((sid,tuple(route)))
        if len(found)==wanted:
            break
    if len(found) < wanted:
        raise AssertionError(f"Insufficient legal goal examples: {split}/{scenario}")
    return found


def sample_route(pop, sid, route, scenario):
    w = shift.world_for(pop,sid,scenario)
    states = []
    receipts = []
    for move in route:
        if move not in w.legal() or distance(w)==0 or w.switched:
            raise AssertionError("Teacher route violated host legal contract")
        states.append((w,0))
        # Preserve pre-step data without allowing later mutations.
        state = w
        nxt = shift.world_for(pop,sid,scenario)
        for rec in receipts:
            nxt.step(rec["choice"])
        if shift.state_digest(nxt)!=shift.state_digest(state):
            raise AssertionError("Independent scaffold reconstruction mismatch")
        before=shift.state_digest(w)
        w.step(move)
        receipts.append({"menu":nxt.legal(),"choice":move,"before":before,
                         "after":shift.state_digest(w),"turn":w.steps})
    if distance(w)!=0 or w.switched or "STOP" not in w.legal():
        raise AssertionError("Route did not reach active goal legally")
    # States must be snapshots: replay from root for each predecessor.
    captured=[]
    replay=shift.world_for(pop,sid,scenario)
    for i,move in enumerate(route):
        captured.append((replay,0))
        current=shift.world_for(pop,sid,scenario)
        for prior in route[:i]:
            current.step(prior)
        captured[-1]=(current,0)
        replay.step(move)
    captured.append((w,1))
    shift.audit_receipts(pop,receipts,"navigator",sid,scenario)
    return captured, receipts


def mine(pop, frozen, seed, niche, scenario, used, t0):
    examples, receipts = [], []
    for i in range(MINE_WORLDS):
        sid = world_seed(pop,"MINE",seed,niche,scenario,i)
        if sid in used: raise AssertionError("MINE world collision")
        used.add(sid)
        w=shift.world_for(pop,sid,scenario)
        trace=[]
        while not w.stopped and w.steps<w.max_steps:
            menu=w.legal()
            X=np.stack([w.feature(a) for a in menu])
            logits=frozen.logits(X)
            if not np.isfinite(logits).all(): raise AssertionError("Nonfinite frozen MLP logits")
            choice=menu[int(np.argmax(logits))]
            # All observed states are genuine on-policy. No STOP masking.
            # Preserve immutable copy via deterministic replay.
            clone=shift.world_for(pop,sid,scenario)
            for prior in trace:
                clone.step(prior["choice"])
            if shift.state_digest(clone)!=shift.state_digest(w):
                raise AssertionError("MINE divergence")
            examples.append((clone,int(distance(clone)==0)))
            before=shift.state_digest(w)
            w.step(choice)
            trace.append({"menu":menu,"choice":choice,"before":before,
                          "after":shift.state_digest(w),"turn":w.steps})
            budget(t0)
        if i==0:
            shift.audit_receipts(pop,trace,niche,sid,scenario)
            receipts=trace
    return examples,receipts


def choose_threshold(gate, validation):
    """Calibrate strictly on a disjoint validation split, never heldout."""
    y=np.asarray([int(z) for w,z in validation])
    p=np.asarray([gate.probability(features(w,gate.geometry)) for w,z in validation])
    negatives=p[y==0]
    positives=p[y==1]
    if not len(negatives) or not len(positives):
        raise AssertionError("Empty validation class")
    # Strict threshold; select highest recall subject to <=1% false positives.
    candidates=np.unique(np.concatenate(([0.,1.],p,np.nextafter(p,1))))
    best=(float("-inf"),float("-inf"),1.0)
    for threshold in candidates:
        fp=float(np.mean(negatives>=threshold))
        if fp>0.01+1e-12: continue
        recall=float(np.mean(positives>=threshold))
        key=(recall,-fp,-float(threshold))
        if key>(best[0],best[1],-best[2]):
            best=(recall,-fp,float(threshold))
    gate.threshold=best[2]
    return {"threshold":round(float(gate.threshold),10),
            "false_positive_rate":round(float(np.mean(negatives>=gate.threshold)),6),
            "recall":round(float(np.mean(positives>=gate.threshold)),6),
            "validation_positive":int(y.sum()),"validation_negative":int((1-y).sum())}


def score_gate(gate,cases):
    y=np.asarray([int(z) for w,z in cases])
    p=np.asarray([gate.probability(features(w,gate.geometry)) for w,z in cases])
    choice=p>=gate.threshold
    if not np.isfinite(p).all(): raise AssertionError("Nonfinite heldout head scores")
    tp=int(np.sum(choice & (y==1)))
    fp=int(np.sum(choice & (y==0)))
    return {"positive":int(y.sum()),"negative":int((1-y).sum()),"tp":tp,"fp":fp,
            "fn":int(np.sum(~choice & (y==1))),
            "tn":int(np.sum(~choice & (y==0))),
            "recall":round(tp/max(1,np.sum(y==1)),5),
            "false_positive_rate":round(fp/max(1,np.sum(y==0)),5),
            "probability_positive_mean":round(float(np.mean(p[y==1])),6),
            "probability_negative_mean":round(float(np.mean(p[y==0])),6)}


def autonomous(pop, frozen, gate, arm, niche, sid, scenario, trace=False):
    if arm not in ARMS: raise ValueError("Unknown inference arm")
    w=shift.world_for(pop,sid,scenario)
    reward=0.; retraces=0; forwards=0; gate_forwards=0; receipts=[]
    stop_at_goal=0; premature_stop=0; goal_visits=0
    while not w.stopped and w.steps<w.max_steps:
        menu=w.legal()
        X=np.stack([w.feature(a) for a in menu])
        logits=frozen.logits(X)
        if not np.isfinite(logits).all(): raise AssertionError("Frozen base model nonfinite")
        forwards+=1
        # Only the separate STOP decision is learned by the new head.
        # Spatial movement scores remain the authentic frozen policy logits.
        idx=int(np.argmax(logits))
        if gate is not None and len(menu)>1:
            score=gate.probability(features(w,gate.geometry))
            gate_forwards+=1
            if score>=gate.threshold: idx=menu.index("STOP")
            else: idx=max((i for i,a in enumerate(menu) if a!="STOP"),
                          key=lambda i:float(logits[i]))
        selected=menu[idx]
        if distance(w)==0:
            goal_visits+=1
            if selected=="STOP": stop_at_goal+=1
        elif selected=="STOP":
            premature_stop+=1
        before=shift.state_digest(w)
        reward+=pop.utility(niche,X[idx],w)
        w.step(selected)
        retraces+=int(selected=="RETRACE")
        if w.painted & (w.protected | w.obstacles): raise AssertionError("Protected paint")
        if trace:
            receipts.append({"turn":w.steps,"menu":menu,"choice":selected,
                             "before":before,"after":shift.state_digest(w)})
    ep={"seed":sid,"scenario":scenario,"niche":niche,"arm":arm,
        "reward":round(reward,5),"reward_per_decision":round(reward/max(1,w.steps),5),
        "decisions":w.steps,"new_ink":len(w.painted)-1,"retraces":retraces,
        "early_stop":bool(w.stopped and w.steps<w.max_steps),
        "goal_visits":goal_visits,"stop_at_goal":stop_at_goal,
        "premature_stop":premature_stop,"model_forwards":forwards,
        "gate_forwards":gate_forwards,"protected_violations":0,
        "receipts":receipts if trace else None}
    return ep,w


def run(request, upstream, frozen_dir, output, seed, cohort):
    checked_request(json.loads(request.read_text()))
    if seed not in SEEDS or cohort not in COHORTS:
        raise ValueError("Unknown seed/cohort job")
    pop,source=shift.load_source(upstream)
    if len(pop.FEATURES)!=20 or pop.FEATURES[3]!="stop":
        raise AssertionError("Pinned host feature contract drift")
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report={
        "schema":"cybersand.fleet-training.stop-sidecar-evidence/1",
        "request_id":ID,"seed":seed,"cohort":cohort,"status":"starting",
        "classification":"REAL_NUMPY_ADAM_EXTERNAL_SYNTHETIC_NOT_NATIVE_NOT_W0_NOT_PRETRAINED_SLM",
        "source_commit":shift.SOURCE_COMMIT,"source_blob":shift.SOURCE_BLOB,
        "source_sha256":base.sha_file(source),"executor_sha256":base.sha_file(Path(__file__)),
        "request_sha256":base.sha_file(request),
        "ft003_original_zip_sha256":shift.FT003_SHA,"native_material_edits":0,
        "resident_training_processes_per_runner":1,"pretrained_slm_optimizer_steps":0,
        "workers":[],"groups":[],"failures":[]}
    try:
        report["original_ft003_audit"]=shift.archive_check(
            frozen_dir,source,request.parent/"request_dose.json")
        report["status"]="running"
        used=set()
        for niche in COHORTS[cohort]:
            source_ckpt=frozen_dir/f"dose_48/{seed}/models/{niche}.json"
            frozen=shift.load_policy(pop,source_ckpt)
            original_hash=base.sha_file(source_ckpt)
            train=[];validation=[];heldout=[];sample_receipts=[]
            for scenario in SCENARIOS:
                for split,dest in (("TRAIN",train),("VALIDATION",validation),("HELDOUT",heldout)):
                    for sid,route in route_cases(pop,seed,niche,scenario,split,used):
                        observations,receipt=sample_route(pop,sid,route,scenario)
                        dest.extend(observations)
                        if split=="HELDOUT" and not sample_receipts:
                            sample_receipts=[{"seed":sid,"scenario":scenario,
                                              "receipts":receipt}]
                        budget(start)
                mined,first_receipts=mine(pop,frozen,seed,niche,scenario,used,start)
                train.extend(mined)
                d=output/"mining"/niche
                d.mkdir(parents=True,exist_ok=True)
                (d/(scenario+".json")).write_text(json.dumps({
                    "classification":"ORIGINAL_FROZEN_ONPOLICY_SYNTHETIC_ACTIONS",
                    "receipts":first_receipts},sort_keys=True)+"\n")
            gates={}
            for arm in GATE_ARMS:
                model=StopGate(geometry=(arm==GATE_ARMS[1]))
                training=model.fit(train)
                calibration=choose_threshold(model,validation)
                evaluation=score_gate(model,heldout)
                path=output/"models"/niche/(arm+".json")
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(model.serialize(),sort_keys=True)+"\n")
                gates[arm]=model
                report["workers"].append({
                    "lease_id":lease(seed,niche,arm),"state":"completed","niche":niche,
                    "arm":arm,"model":"binary-logistic-stop-gate",
                    "backbone":"frozen-ft003-24-unit-mlp",
                    "backbone_checkpoint_sha256":original_hash,
                    "trained_sidecar_sha256":base.sha_file(path),
                    "training":training,"calibration":calibration,"fresh_heldout":evaluation})
            for scenario in SCENARIOS:
                sidlist=[world_seed(pop,"AUTONOMOUS",seed,niche,scenario,i)
                         for i in range(AUTO_WORLDS)]
                if any(s in used for s in sidlist):
                    raise AssertionError("Autonomous split collided")
                used.update(sidlist)
                group={"niche":niche,"scenario":scenario,"world_seeds":sidlist,
                       "autonomous":{}}
                for arm in ARMS:
                    episodes=[]
                    for i,sid in enumerate(sidlist):
                        gate=gates.get(arm)
                        ep,w=autonomous(pop,frozen,gate,arm,niche,sid,scenario,i==0)
                        if i==0:
                            shift.audit_receipts(pop,ep["receipts"],niche,sid,scenario)
                            p=output/"gallery"/niche/scenario
                            p.mkdir(parents=True,exist_ok=True)
                            (p/(arm+".svg")).write_text(pop.draw_svg(w))
                            (p/(arm+".json")).write_text(json.dumps(ep,sort_keys=True)+"\n")
                        episodes.append(ep)
                        budget(start)
                    aggregate=shift.summarize(episodes)
                    for key in ("stop_at_goal","premature_stop","goal_visits","gate_forwards"):
                        aggregate[key]=sum(x[key] for x in episodes)
                    group["autonomous"][arm]=aggregate
                report["groups"].append(group)
            (output/"evidence"/(niche+".json")).parent.mkdir(parents=True,exist_ok=True)
            (output/"evidence"/(niche+".json")).write_text(json.dumps({
                "heldout_original_legal_goal_route":sample_receipts,
                "train_states":len(train),"validation_states":len(validation),
                "heldout_states":len(heldout),
                "train_labels_sha256":hashlib.sha256(
                    json.dumps([int(y) for w,y in train]).encode()).hexdigest(),
                "validation_labels_sha256":hashlib.sha256(
                    json.dumps([int(y) for w,y in validation]).encode()).hexdigest(),
                "heldout_labels_sha256":hashlib.sha256(
                    json.dumps([int(y) for w,y in heldout]).encode()).hexdigest()
            },sort_keys=True)+"\n")
            if base.sha_file(source_ckpt)!=original_hash:
                raise AssertionError("Frozen backbone modified")
        if len(set(x["lease_id"] for x in report["workers"]))!=len(report["workers"]):
            raise AssertionError("Duplicate model lease")
        report["custody_sha256"]=hashlib.sha256(json.dumps(sorted(used)).encode()).hexdigest()
        report["aggregate"]={
            "trained_sidecars":len(report["workers"]),
            "real_adam_updates":sum(w["training"]["adam_updates"] for w in report["workers"]),
            "validation_goal_recall":{a:round(sum(w["calibration"]["recall"]
                for w in report["workers"] if w["arm"]==a)/len(COHORTS[cohort]),5)
                for a in GATE_ARMS},
            "heldout_goal_tp":{a:sum(w["fresh_heldout"]["tp"] for w in report["workers"]
                if w["arm"]==a) for a in GATE_ARMS},
            "heldout_nongoal_fp":{a:sum(w["fresh_heldout"]["fp"] for w in report["workers"]
                if w["arm"]==a) for a in GATE_ARMS},
            "heldout_nongoal_total":sum(w["fresh_heldout"]["negative"]
                for w in report["workers"] if w["arm"]==GATE_ARMS[0]),
            "autonomous_episodes":len(report["groups"])*len(ARMS)*AUTO_WORLDS,
            "autonomous_mean_reward_per_decision":{a:round(sum(
                g["autonomous"][a]["reward_per_decision"] for g in report["groups"]
            )/len(report["groups"]),6) for a in ARMS},
            "autonomous_early_stop":{a:sum(g["autonomous"][a]["early_stop"]*AUTO_WORLDS
                for g in report["groups"]) for a in ARMS},
            "autonomous_retrace_mean":{a:round(sum(g["autonomous"][a]["retraces"]
                for g in report["groups"])/len(report["groups"]),5) for a in ARMS},
            "elapsed_s":round(time.monotonic()-start,5),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        report["status"]="completed"
    except BaseException as exc:
        report["status"]="failed"
        report["failures"].append(type(exc).__name__+": "+str(exc))
        raise
    finally:
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        (output/"SHA256SUMS").write_text("".join(
            base.sha_file(p)+"  "+str(p.relative_to(output))+"\n"
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name!="SHA256SUMS"))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--upstream",type=Path,required=True)
    parser.add_argument("--ft003",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--cohort",choices=COHORTS,required=True)
    args=parser.parse_args()
    report=run(args.request,args.upstream,args.ft003,args.output,args.seed,args.cohort)
    print(json.dumps({"status":report["status"],"seed":args.seed,
                      "cohort":args.cohort,"aggregate":report["aggregate"]},sort_keys=True))
