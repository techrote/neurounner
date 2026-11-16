"""FT-016: frozen policy STOP custody, forced/voluntary endings and fixed-horizon utility.

External 22x22 synthetic game host ONLY. Models are immutable FT-003 / FT-014
Actions artifacts; there is no native/W0 or pretrained-SLM inference claim.
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

ID = "FT-016-20261116-forced-vs-voluntary-horizon-utility"
SCHEMA = "cybersand.fleet-training.ft016-request/1"
HEAD_015 = "5202f329ae2b5802b2ec74e8039f37f3d7a445ac"
BLOB_015 = "e3e232ddc11fa7e4d4b0113e741e7008ab9394a7"
SEEDS = prior.SEEDS
COHORTS = prior.COHORTS
SCENARIOS = prior.SCENARIOS
ARMS = prior.ARMS
WORLDS = 12
PROBES = 8
MAX_SECONDS = 360
MAX_RSS_KIB = 900000


def check_request(req):
    want = {
        "schema": SCHEMA, "request_id": ID, "ft015_source_head": HEAD_015,
        "ft015_git_blob": BLOB_015,
        "ft014_source_head": prior.HEAD_014,
        "ft014_actions_run": prior.RUN_014, "ft014_artifact_ids": prior.ARTIFACTS,
        "ft003_artifact_id": 14194214028, "ft003_zip_sha256": shift.FT003_SHA,
        "source_commit": shift.SOURCE_COMMIT, "source_blob": shift.SOURCE_BLOB,
        "seeds": list(SEEDS), "cohorts": {k:list(v) for k,v in COHORTS.items()},
        "scenarios": list(SCENARIOS), "arms": list(ARMS),
        "worlds_per_scenario": WORLDS, "legal_scaffold_probes_per_scenario": PROBES,
        "max_seconds_per_job": MAX_SECONDS, "max_rss_kib_per_job": MAX_RSS_KIB,
        "max_parallel_runner_jobs": 4, "optimizer_updates": 0,
        "native_material_edits": 0, "pretrained_slm_optimizer_steps": 0,
    }
    if type(req) is not dict or req != want:
        raise ValueError("FT-016 immutable source, split, artifacts or budget drift")
    return req


def lease(seed, cohort):
    if seed not in SEEDS or cohort not in COHORTS:
        raise ValueError("Unowned FT-016 seed/cohort")
    return f"{ID}/{seed}/{cohort}"


def world_seed(pop, seed, niche, scenario, index, split):
    if (seed not in SEEDS or niche not in base.NICHES or
        scenario not in SCENARIOS or split not in ("AUTONOMOUS","SCAFFOLD") or
        index < 0 or index >= (WORLDS if split=="AUTONOMOUS" else PROBES)):
        raise ValueError("Invalid FT-016 world lease")
    new = pop.stable_seed(ID,split,seed,niche,scenario,index)
    if split=="AUTONOMOUS":
        old={prior.world_seed(pop,s,niche,scenario,i)
             for s in SEEDS for i in range(prior.WORLDS)}
        if new in old: raise AssertionError("FT-015 OOD seed reuse")
    return new


def budget(start):
    if time.monotonic()-start > MAX_SECONDS:
        raise TimeoutError("FT-016 job wall budget exceeded")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_KIB:
        raise MemoryError("FT-016 process memory budget exceeded")


def ending_kind(action, menu, at_goal, horizon_exhausted=False):
    """STOP is compulsory only if the host supplies no alternative legal action."""
    if horizon_exhausted and action != "STOP":
        return "horizon_exhausted"
    if action != "STOP":
        return "continuing"
    if menu == ["STOP"]:
        return "forced_only_stop_at_goal" if at_goal else "forced_only_stop_nongoal"
    if "STOP" not in menu or len(menu) < 2:
        raise AssertionError("Inconsistent host action menu")
    return "voluntary_goal_stop" if at_goal else "voluntary_nongoal_stop"


def gate_decisions(pop, frozen, gates, world, niche):
    return {arm: prior.choose(pop,frozen,gates,arm,world,niche)[0] for arm in ARMS}


def episode(pop, frozen, gates, seed, niche, scenario, arm, trace=False):
    w=prior.make_world(pop,seed,scenario)
    sum_utility=0.
    turns=[]
    retraces=goal_visits=post_switch_steps=0
    ending="horizon_exhausted"
    post_switch_voluntary_nongoal=0
    while not w.stopped and w.steps<w.max_steps:
        act,feature,menu=prior.choose(pop,frozen,gates,arm,w,niche)
        at_goal=(side.distance(w)==0)
        goal_visits+=int(at_goal)
        post_switch_steps+=int(w.switched)
        if act=="STOP":
            ending=ending_kind(act,menu,at_goal)
            post_switch_voluntary_nongoal+=int(
                w.switched and ending=="voluntary_nongoal_stop")
        elif w.steps+1==w.max_steps:
            ending="horizon_exhausted"
        was_switched=bool(w.switched)
        before=shift.state_digest(w)
        sum_utility+=float(pop.utility(niche,feature,w))
        w.step(act)
        retraces+=int(act=="RETRACE")
        if w.painted & (w.protected | w.obstacles):
            raise AssertionError("Illegal protected/obstacle paint")
        if trace:
            turns.append({"turn":w.steps,"menu":menu,"choice":act,
                          "before":before,"after":shift.state_digest(w),
                          "at_goal_before":at_goal,"after_switch_before":was_switched})
    if ending=="continuing":
        raise AssertionError("Unclassified episode end")
    result={
        "world_seed":seed,"niche":niche,"scenario":scenario,"arm":arm,
        "horizon":w.max_steps,"decisions":w.steps,
        "total_shaped_utility":round(sum_utility,7),
        "utility_per_budget_step":round(sum_utility/w.max_steps,8),
        "utility_per_decision":round(sum_utility/max(1,w.steps),8),
        "new_ink":len(w.painted)-1,"ink_per_budget_step":round(
            (len(w.painted)-1)/w.max_steps,8),
        "retraces":retraces,"goal_visits":goal_visits,
        "switch_occurred":bool(w.switched),"post_switch_decisions":post_switch_steps,
        "post_switch_voluntary_nongoal_stop":post_switch_voluntary_nongoal,
        "ending":ending,"voluntary_goal_success":int(ending=="voluntary_goal_stop"),
        "voluntary_nongoal_failure":int(ending=="voluntary_nongoal_stop"),
        "host_forced_stop":int(ending.startswith("forced_only_")),
        "horizon_exhausted":int(ending=="horizon_exhausted"),
    }
    if trace: result["receipts"]=turns
    return result,w


def empty_probe():
    return {arm:{"total":0,"distance_0":0,"distance_1":0,"distance_2":0,
                 "distance_3":0,"distance_4plus":0,
                 "stop_at_goal":0,"false_stop_nongoal":0,
                 "false_stop_1":0,"false_stop_2":0,"false_stop_3":0,
                 "false_stop_4plus":0} for arm in ARMS}


def probe_state(pop, frozen, gates, world, niche, score):
    d=side.distance(world)
    bin_name="distance_"+str(d) if d<=3 else "distance_4plus"
    actions=gate_decisions(pop,frozen,gates,world,niche)
    for arm,action in actions.items():
        row=score[arm]
        row["total"]+=1
        row[bin_name]+=1
        if action=="STOP":
            if d==0: row["stop_at_goal"]+=1
            else:
                row["false_stop_nongoal"]+=1
                key="false_stop_"+str(d) if d<=3 else "false_stop_4plus"
                row[key]+=1


def legal_scaffold_probes(pop,frozen,gates,seed,niche,scenario):
    """Host-legal BFS route. This is NOT an autonomous policy trajectory."""
    result=empty_probe()
    reached=failed=receipts=0
    example=[]
    for i in range(PROBES):
        sid=world_seed(pop,seed,niche,scenario,i,"SCAFFOLD")
        w=prior.make_world(pop,sid,scenario)
        route=goals.legal_route(pop,w)
        if route is None or not route:
            failed+=1
            continue
        visited=[]
        for action in route:
            if action not in w.legal() or "STOP" not in w.legal():
                raise AssertionError("Illegal independent BFS scaffold")
            probe_state(pop,frozen,gates,w,niche,result)
            before=shift.state_digest(w)
            w.step(action)
            visited.append({"choice":action,"before":before,
                            "after":shift.state_digest(w)})
            receipts+=1
        if side.distance(w)!=0 or w.switched:
            raise AssertionError("Scaffold did not reach active pre-switch goal")
        probe_state(pop,frozen,gates,w,niche,result)
        reached+=1
        if not example:
            example=[{"seed":sid,"scenario":scenario,"actions":visited}]
    return {"legally_reached_goal_states":reached,"unreachable_worlds":failed,
            "scaffold_transitions":receipts,"arms":result,
            "first_route":example}


def mean(xs,key):
    return round(sum(x[key] for x in xs)/len(xs),7)


def summarize(episodes):
    first=episodes[0]
    endings={key:sum(x["ending"]==key for x in episodes) for key in (
        "voluntary_goal_stop","voluntary_nongoal_stop",
        "forced_only_stop_at_goal","forced_only_stop_nongoal",
        "horizon_exhausted")}
    if sum(endings.values())!=len(episodes):
        raise AssertionError("Missing/overlapping ending classification")
    return {"episodes":len(episodes),"endings":endings,
        "mean_total_shaped_utility":mean(episodes,"total_shaped_utility"),
        "mean_utility_per_budget_step":mean(episodes,"utility_per_budget_step"),
        "mean_utility_per_decision":mean(episodes,"utility_per_decision"),
        "mean_ink":mean(episodes,"new_ink"),
        "mean_retraces":mean(episodes,"retraces"),
        "mean_decisions":mean(episodes,"decisions"),
        "goal_visits":sum(x["goal_visits"] for x in episodes),
        "switch_episodes":sum(x["switch_occurred"] for x in episodes),
        "post_switch_decisions":sum(x["post_switch_decisions"] for x in episodes),
        "post_switch_voluntary_nongoal_stops":sum(
            x["post_switch_voluntary_nongoal_stop"] for x in episodes),
        "fixed_horizon_steps":len(episodes)*first["horizon"]}


def paired(cases):
    control=cases["frozen_baseline"]
    comparisons={}
    for arm in ARMS[1:]:
        other=cases[arm]
        for x,y in zip(control,other):
            if x["world_seed"]!=y["world_seed"]:
                raise AssertionError("Unpaired evaluation worlds")
        comparisons[arm]={
            "total_utility_delta_mean":round(sum(
                b["total_shaped_utility"]-a["total_shaped_utility"]
                for a,b in zip(control,other))/len(control),7),
            "ink_delta_mean":round(sum(
                b["new_ink"]-a["new_ink"] for a,b in zip(control,other))/len(control),7),
            "wins_total_utility":sum(b["total_shaped_utility"]>a["total_shaped_utility"]
                                     for a,b in zip(control,other)),
            "losses_total_utility":sum(b["total_shaped_utility"]<a["total_shaped_utility"]
                                       for a,b in zip(control,other)),
            "ties_total_utility":sum(b["total_shaped_utility"]==a["total_shaped_utility"]
                                     for a,b in zip(control,other))}
    return comparisons


def run(reqfile, upstream, ft003, ft014_archive, output, seed, cohort):
    check_request(json.loads(reqfile.read_text()))
    active=lease(seed,cohort)
    pop,source=shift.load_source(upstream)
    if len(pop.FEATURES)!=20 or pop.FEATURES[3]!="stop":
        raise AssertionError("Pinned 20-feature STOP contract drift")
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report={"schema":"cybersand.fleet-training.ft016-evidence/1",
            "request_id":ID,"lease":active,"seed":seed,"cohort":cohort,
            "status":"starting","classification":
            "FROZEN_TINY_MLP_SIDECARS_EXTERNAL_SYNTHETIC_NOT_NATIVE_NOT_W0_NOT_SLM",
            "source_commit":shift.SOURCE_COMMIT,"source_blob":shift.SOURCE_BLOB,
            "source_sha256":base.sha_file(source),
            "ft015_source_head":HEAD_015,"ft015_sha256":base.sha_file(Path(prior.__file__)),
            "executor_sha256":base.sha_file(Path(__file__)),
            "request_sha256":base.sha_file(reqfile),
            "ft014_original_artifact_id":prior.ARTIFACTS[f"{seed}/{cohort}"],
            "ft003_original_zip_sha256":shift.FT003_SHA,
            "optimizer_updates":0,"native_material_edits":0,
            "resident_inference_processes_per_job":1,"groups":[],"failures":[]}
    try:
        models,custody=prior.archive_audit(ft014_archive,seed,cohort)
        report["ft014_custody"]=custody
        report["ft003_custody"]=shift.archive_check(
            ft003,source,reqfile.parent/"request_dose.json")
        seen=set()
        for niche in COHORTS[cohort]:
            checkpoint=ft003/f"dose_48/{seed}/models/{niche}.json"
            frozen=shift.load_policy(pop,checkpoint)
            backbone_hash=base.sha_file(checkpoint)
            ft014_records=json.loads(
                (ft014_archive/"run"/"report.json").read_text())["workers"]
            historical=next(x["backbone_checkpoint_sha256"] for x in ft014_records
                            if x["niche"]==niche)
            if backbone_hash!=historical:
                raise ValueError("Frozen FT-003 backbone custody mismatch")
            gates={arm:models[(niche,arm)] for arm in side.GATE_ARMS}
            for scenario in SCENARIOS:
                cases={arm:[] for arm in ARMS}
                for i in range(WORLDS):
                    sid=world_seed(pop,seed,niche,scenario,i,"AUTONOMOUS")
                    if sid in seen: raise AssertionError("Duplicate world lease")
                    seen.add(sid)
                    for arm in ARMS:
                        res,w=episode(pop,frozen,gates,sid,niche,scenario,arm,i==0)
                        if i==0:
                            prior.audit_trace(pop,sid,scenario,res["receipts"],w)
                            folder=output/"gallery"/niche/scenario
                            folder.mkdir(parents=True,exist_ok=True)
                            (folder/(arm+".svg")).write_text(pop.draw_svg(w))
                            (folder/(arm+".json")).write_text(
                                json.dumps(res,sort_keys=True)+"\n")
                        cases[arm].append(res)
                    budget(start)
                probes=legal_scaffold_probes(pop,frozen,gates,seed,niche,scenario)
                (output/"episodes"/niche).mkdir(parents=True,exist_ok=True)
                (output/"episodes"/niche/(scenario+".json")).write_text(
                    json.dumps(cases,sort_keys=True)+"\n")
                (output/"probes"/niche).mkdir(parents=True,exist_ok=True)
                (output/"probes"/niche/(scenario+".json")).write_text(
                    json.dumps(probes,sort_keys=True)+"\n")
                report["groups"].append({
                    "niche":niche,"scenario":scenario,
                    "backbone_checkpoint_sha256":backbone_hash,
                    "sidecar_sha256":{arm:custody["model_sha256"][f"{niche}/{arm}"]
                                       for arm in side.GATE_ARMS},
                    "autonomous":{arm:summarize(cases[arm]) for arm in ARMS},
                    "paired_vs_frozen":paired(cases),
                    "legal_probe":{k:v for k,v in probes.items() if k!="first_route"}})
                budget(start)
            if base.sha_file(checkpoint)!=backbone_hash:
                raise AssertionError("Frozen policy checkpoint modified")
        if len(seen)!=len(COHORTS[cohort])*len(SCENARIOS)*WORLDS:
            raise AssertionError("Missing unique autonomous world leases")
        report["aggregate"]={
            "unique_autonomous_worlds":len(seen),
            "autonomous_episodes":len(seen)*len(ARMS),
            "legally_reached_probe_goals":sum(
                g["legal_probe"]["legally_reached_goal_states"] for g in report["groups"]),
            "scaffold_transitions":sum(
                g["legal_probe"]["scaffold_transitions"] for g in report["groups"]),
            "by_arm":{arm:{
                "ending_counts":{kind:sum(g["autonomous"][arm]["endings"][kind]
                    for g in report["groups"])
                    for kind in report["groups"][0]["autonomous"][arm]["endings"]},
                "mean_total_utility":round(sum(g["autonomous"][arm]["mean_total_shaped_utility"]
                    for g in report["groups"])/len(report["groups"]),7),
                "mean_reward_per_decision":round(sum(g["autonomous"][arm]["mean_utility_per_decision"]
                    for g in report["groups"])/len(report["groups"]),7),
                "mean_ink":round(sum(g["autonomous"][arm]["mean_ink"]
                    for g in report["groups"])/len(report["groups"]),7),
                "switch_episodes":sum(g["autonomous"][arm]["switch_episodes"]
                    for g in report["groups"])}
                    for arm in ARMS},
            "elapsed_seconds":round(time.monotonic()-start,5),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        report["unique_worlds_sha256"]=hashlib.sha256(
            json.dumps(sorted(seen)).encode()).hexdigest()
        report["status"]="completed"
    except BaseException as exc:
        report["status"]="failed"
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        (output/"SHA256SUMS").write_text("".join(
            base.sha_file(p)+"  "+str(p.relative_to(output))+"\n"
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name!="SHA256SUMS"))
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
    result=run(a.request,a.upstream,a.ft003,a.ft014,a.output,a.seed,a.cohort)
    print(json.dumps({"lease":result["lease"],"status":result["status"],
                      "aggregate":result["aggregate"]},sort_keys=True))
