"""FT-018: frozen-policy two-objective STOP, goal replacement and legal replay.
Synthetic 22x22 host only, not native CyberSand, W0 or a pretrained SLM.
"""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import resource
import time
import numpy as np
import fleet_bench as base
import ft004_shift as shift
import ft010_goal_reachability as routes
import ft014_stop_sidecar as side
import ft015_ood_stop as prior

ID="FT-018-20261116-two-stage-goal-switch"
SCHEMA="cybersand.fleet-training.ft018-request/1"
HEAD_017="b34cc2a08acad0d46337069c1ff31b52c60eb2b6"
BLOB_017="1c240d933531be07a968632cd6160a705da153bd"
ARMS=("frozen_baseline","distance_feature_gate","exact_current_goal","phase_gate","phase_exact")
SEEDS, COHORTS, SCENARIOS=side.SEEDS,side.COHORTS,prior.SCENARIOS
WORLDS,CANDIDATES,PAIRS,HORIZON=8,96,2,80
MAX_SECONDS,MAX_RSS_KIB=360,900000

def expected_request():
    return dict(schema=SCHEMA,request_id=ID,source_commit=shift.SOURCE_COMMIT,
        source_blob=shift.SOURCE_BLOB,ft017_source_head=HEAD_017,ft017_git_blob=BLOB_017,
        ft014_source_head=prior.HEAD_014,ft014_artifact_ids=prior.ARTIFACTS,
        ft003_artifact_id=14194214028,ft003_zip_sha256=shift.FT003_SHA,
        seeds=list(SEEDS),cohorts={k:list(v) for k,v in COHORTS.items()},
        scenarios=list(SCENARIOS),arms=list(ARMS),worlds_per_scenario=WORLDS,
        teacher_candidate_worlds=CANDIDATES,teacher_target_pairs_per_scenario=PAIRS,
        evaluation_horizon=HORIZON,max_seconds_per_job=MAX_SECONDS,
        max_rss_kib_per_job=MAX_RSS_KIB,max_parallel_runner_jobs=4,
        optimizer_updates=0,pretrained_slm_optimizer_steps=0,native_material_edits=0)

def check_request(r):
    if type(r) is not dict or r!=expected_request():
        raise ValueError("FT-018 source/checkpoint/contract/budget drift")
    return r

def lease(seed,cohort):
    if seed not in SEEDS or cohort not in COHORTS: raise ValueError("Foreign wrangler lease")
    return f"{ID}/{seed}/{cohort}"

def world_seed(pop,split,seed,niche,scenario,i):
    if (split not in ("TEACHER","AUTONOMOUS") or seed not in SEEDS or niche not in base.NICHES
        or scenario not in SCENARIOS or type(i) is not int
        or not 0<=i<(CANDIDATES if split=="TEACHER" else WORLDS)):
        raise ValueError("Foreign world lease")
    return pop.stable_seed(ID,split,seed,niche,scenario,i)

def make_world(pop,sid,scenario):
    w=prior.make_world(pop,sid,scenario)
    if w.steps or w.stopped or w.switched: raise AssertionError("Not a fresh world")
    w.max_steps=HORIZON
    if w.painted & (w.protected|w.obstacles): raise AssertionError("Illegal starting paint")
    return w

def budget(start):
    if time.monotonic()-start>MAX_SECONDS: raise TimeoutError("FT-018 wall budget")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>MAX_RSS_KIB:
        raise MemoryError("FT-018 memory cap")

def second_route(pop,w):
    """BFS across unpainted cells plus previously authored RETRACE ancestors."""
    if not w.switched or w.stopped or w.goal!=w.second_goal:
        raise ValueError("Goal has not been replaced by host")
    queue=deque([(w.cursor,())])
    seen={w.cursor}
    while queue:
        pos,path=queue.popleft()
        if pos==w.goal: return path if w.steps+len(path)<w.max_steps else None
        if w.steps+len(path)>=w.max_steps-1: continue
        neighbours=[]
        if pos in w.parent and pos in w.painted:
            neighbours.append(("RETRACE",w.parent[pos]))
        for a,(dx,dy) in pop.DIRECTIONS.items():
            nxt=(pos[0]+dx,pos[1]+dy)
            if w.open(nxt): neighbours.append((a,nxt))
        for action,nxt in neighbours:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt,path+(action,)))
    return None

def receipt(pop,w,a):
    menu=w.legal()
    if a=="STOP" or a not in menu: raise AssertionError("Teacher illegal action")
    before=shift.state_digest(w)
    w.step(a)
    if w.painted & (w.protected|w.obstacles): raise AssertionError("Protected cell painted")
    return dict(turn=w.steps,choice=a,menu=menu,before=before,after=shift.state_digest(w))

def teacher_pair(pop,sid,scenario):
    w=make_world(pop,sid,scenario)
    first=routes.legal_route(pop,w)
    if not first: return None
    receipts=[receipt(pop,w,a) for a in first]
    if w.switched or side.distance(w)!=0 or len(w.legal())==1:
        raise AssertionError("Initial goal not legally reached before switch")
    first_state=dict(turn=w.steps,goal=w.goal,hash=shift.state_digest(w))
    while not w.switched:
        moves=[]
        for a,(dx,dy) in pop.DIRECTIONS.items():
            if a in w.legal():
                dest=(w.cursor[0]+dx,w.cursor[1]+dy)
                if dest!=w.second_goal:
                    d=abs(dest[0]-w.second_goal[0])+abs(dest[1]-w.second_goal[1])
                    moves.append((d,a))
        if moves:
            a=sorted(moves)[0][1]
        elif "RETRACE" in w.legal(): a="RETRACE"
        else: return None
        receipts.append(receipt(pop,w,a))
    if w.steps!=HORIZON//2 or w.goal!=w.second_goal:
        raise AssertionError("Source goal replacement contract drift")
    if w.goal in w.painted: return None
    second=second_route(pop,w)
    if second is None: return None
    for a in second: receipts.append(receipt(pop,w,a))
    if side.distance(w)!=0 or not w.switched or len(w.legal())==1:
        return None
    second_state=dict(turn=w.steps,goal=w.goal,hash=shift.state_digest(w))
    replay=make_world(pop,sid,scenario)
    for r in receipts:
        if r["before"]!=shift.state_digest(replay) or r["menu"]!=replay.legal():
            raise AssertionError("Illegal teacher pre-state")
        replay.step(r["choice"])
        if r["after"]!=shift.state_digest(replay):
            raise AssertionError("Invalid teacher replay receipt")
    if shift.state_digest(replay)!=shift.state_digest(w):
        raise AssertionError("Teacher endpoint differs")
    return dict(seed=sid,first=first_state,second=second_state,
                first_route=list(first),receipts=receipts,
                retraces=sum(r["choice"]=="RETRACE" for r in receipts)),w

def choose(pop,frozen,gate,arm,w):
    if arm not in ARMS: raise ValueError("Foreign STOP controller")
    menu=w.legal()
    X=np.stack([w.feature(a) for a in menu])
    scores=frozen.logits(X)
    if not np.isfinite(scores).all(): raise ValueError("Nonfinite model logit")
    if arm=="frozen_baseline" or len(menu)==1:
        j=int(np.argmax(scores))
    else:
        at_goal=side.distance(w)==0
        learned=bool(gate.probability(side.features(w,True))>=gate.threshold)
        should_stop={"distance_feature_gate":learned,"exact_current_goal":at_goal,
            "phase_gate":w.switched and learned,"phase_exact":w.switched and at_goal}[arm]
        if should_stop: j=menu.index("STOP")
        else:
            idx=[i for i,a in enumerate(menu) if a!="STOP"]
            j=max(idx,key=lambda i:float(scores[i])) if idx else menu.index("STOP")
    return menu[j],X[j],menu

def episode(pop,frozen,gate,arm,sid,niche,scenario,trace=False):
    w=make_world(pop,sid,scenario)
    reward=0.; retraces=first_visits=final_visits=post_switch=0
    completion=premature=forced=0; receipts=[]; ending="horizon_exhausted"
    while not w.stopped and w.steps<w.max_steps:
        a,x,menu=choose(pop,frozen,gate,arm,w)
        phase=bool(w.switched); at_goal=side.distance(w)==0
        first_visits+=int(at_goal and not phase)
        final_visits+=int(at_goal and phase)
        post_switch+=int(phase)
        if a=="STOP":
            forced=int(len(menu)==1)
            completion=int(not forced and phase and at_goal)
            premature=int(not forced and not (phase and at_goal))
            ending=("host_forced" if forced else "completed_final_goal" if completion
                    else "premature_first_stage" if not phase else "premature_final_stage")
        before=shift.state_digest(w)
        reward+=float(pop.utility(niche,x,w))
        w.step(a)
        retraces+=int(a=="RETRACE")
        if w.painted & (w.protected|w.obstacles):
            raise AssertionError("Policy painted host-protected material")
        if trace:
            receipts.append(dict(turn=w.steps,menu=menu,choice=a,before=before,
                after=shift.state_digest(w),switched_before=phase,at_goal_before=at_goal))
    ep=dict(world_seed=sid,arm=arm,niche=niche,scenario=scenario,
        decisions=w.steps,new_ink=len(w.painted)-1,retraces=retraces,
        first_goal_visits=first_visits,final_goal_visits=final_visits,
        post_switch_decisions=post_switch,completed_final_goal=completion,
        premature_voluntary_stop=premature,host_forced_stop=forced,
        ending=ending,total_utility=round(reward,7))
    if trace:
        ep["receipts"]=receipts
        replay=make_world(pop,sid,scenario)
        for r in receipts:
            if r["before"]!=shift.state_digest(replay) or r["menu"]!=replay.legal():
                raise AssertionError("Autonomous replay pre-state drift")
            replay.step(r["choice"])
            if r["after"]!=shift.state_digest(replay):
                raise AssertionError("Autonomous replay post-state drift")
        if shift.state_digest(replay)!=shift.state_digest(w):
            raise AssertionError("Autonomous replay differs")
    return ep,w

def summary(cases):
    return dict(episodes=len(cases),
        final_goal_completions=sum(e["completed_final_goal"] for e in cases),
        premature_voluntary_stops=sum(e["premature_voluntary_stop"] for e in cases),
        host_forced_stops=sum(e["host_forced_stop"] for e in cases),
        first_goal_visits=sum(e["first_goal_visits"] for e in cases),
        final_goal_visits=sum(e["final_goal_visits"] for e in cases),
        switch_episodes=sum(e["post_switch_decisions"]>0 for e in cases),
        mean_ink=round(sum(e["new_ink"] for e in cases)/len(cases),6),
        mean_decisions=round(sum(e["decisions"] for e in cases)/len(cases),6),
        mean_retraces=round(sum(e["retraces"] for e in cases)/len(cases),6),
        mean_total_shaped_utility=round(sum(e["total_utility"] for e in cases)/len(cases),6))

def run(request,upstream,ft003,ft014_archive,output,seed,cohort):
    check_request(json.loads(request.read_text()))
    job=lease(seed,cohort)
    pop,source=shift.load_source(upstream)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(schema="cybersand.fleet-training.ft018-evidence/1",request_id=ID,
        lease=job,seed=seed,cohort=cohort,status="starting",
        classification="EXTERNAL_SYNTHETIC_FROZEN_TINY_CPU_GAME_POLICY_NOT_NATIVE_NOT_W0_NOT_SLM",
        source_commit=shift.SOURCE_COMMIT,source_blob=shift.SOURCE_BLOB,
        source_sha256=base.sha_file(source),executor_sha256=base.sha_file(Path(__file__)),
        request_sha256=base.sha_file(request),ft003_zip_sha256=shift.FT003_SHA,
        ft014_artifact_id=prior.ARTIFACTS[f"{seed}/{cohort}"],optimizer_updates=0,
        native_material_edits=0,resident_inference_processes_per_runner=1,
        models=[],groups=[],failures=[])
    try:
        models,custody=prior.archive_audit(ft014_archive,seed,cohort)
        report["ft014_custody"]=custody
        report["ft003_custody"]=shift.archive_check(
            ft003,source,request.parent/"request_dose.json")
        seen=set()
        for niche in COHORTS[cohort]:
            backbone=ft003/f"dose_48/{seed}/models/{niche}.json"
            frozen=shift.load_policy(pop,backbone)
            backbone_sha=base.sha_file(backbone)
            prior_workers=json.loads((ft014_archive/"run"/"report.json").read_text())["workers"]
            if not any(x["niche"]==niche and x["backbone_checkpoint_sha256"]==backbone_sha
                       for x in prior_workers):
                raise ValueError("Frozen backbone not matched to original sidecar")
            gate=models[(niche,"distance_feature_gate")]
            report["models"].append(dict(lease=job+"/"+niche,niche=niche,
                backbone_sha256=backbone_sha,
                gate_sha256=custody["model_sha256"][f"{niche}/distance_feature_gate"]))
            for scenario in SCENARIOS:
                pairs=[]; tried=0
                for i in range(CANDIDATES):
                    sid=world_seed(pop,"TEACHER",seed,niche,scenario,i)
                    if sid in seen: raise AssertionError("Duplicate teacher dispatch")
                    seen.add(sid); tried+=1
                    result=teacher_pair(pop,sid,scenario)
                    if result is not None:
                        record,w=result
                        at_first=make_world(pop,sid,scenario)
                        for a in record["first_route"]: at_first.step(a)
                        if at_first.switched or side.distance(at_first)!=0:
                            raise AssertionError("First-stage teacher state drift")
                        record["probes"]={}
                        for arm in ARMS:
                            a,_,menu=choose(pop,frozen,gate,arm,at_first)
                            b,_,terminal_menu=choose(pop,frozen,gate,arm,w)
                            record["probes"][arm]=dict(
                                first_stop=int(a=="STOP" and len(menu)>1),
                                final_stop=int(b=="STOP" and len(terminal_menu)>1))
                        pairs.append(record)
                        if len(pairs)>=PAIRS:
                            directory=output/"gallery"/niche/scenario
                            directory.mkdir(parents=True,exist_ok=True)
                            (directory/"teacher.svg").write_text(pop.draw_svg(w))
                            break
                    if i%12==0: budget(started)
                if not pairs:
                    report["failures"].append(f"NO_LEGAL_TWO_STAGE_REACHABILITY:{niche}/{scenario}")
                cases={arm:[] for arm in ARMS}
                for i in range(WORLDS):
                    sid=world_seed(pop,"AUTONOMOUS",seed,niche,scenario,i)
                    if sid in seen: raise AssertionError("Duplicate inference world lease")
                    seen.add(sid)
                    for arm in ARMS:
                        ep,w=episode(pop,frozen,gate,arm,sid,niche,scenario,i==0)
                        cases[arm].append(ep)
                        if i==0:
                            directory=output/"gallery"/niche/scenario
                            directory.mkdir(parents=True,exist_ok=True)
                            (directory/(arm+".svg")).write_text(pop.draw_svg(w))
                            (directory/(arm+".json")).write_text(json.dumps(ep,sort_keys=True)+"\n")
                    budget(started)
                d=output/"episodes"/niche
                d.mkdir(parents=True,exist_ok=True)
                (d/(scenario+".json")).write_text(json.dumps(cases,sort_keys=True)+"\n")
                (d/(scenario+"-teacher.json")).write_text(
                    json.dumps(dict(attempted=tried,pairs=pairs),sort_keys=True)+"\n")
                report["groups"].append(dict(niche=niche,scenario=scenario,
                    candidate_worlds_tried=tried,legally_reached_both=len(pairs),
                    teacher_probes={arm:dict(
                        first_stop=sum(p["probes"][arm]["first_stop"] for p in pairs),
                        final_stop=sum(p["probes"][arm]["final_stop"] for p in pairs))
                        for arm in ARMS},
                    autonomous={arm:summary(cases[arm]) for arm in ARMS}))
        report["aggregate"]=dict(unique_world_leases=len(seen),
            model_niches=len(report["models"]),
            legal_two_goal_teacher_pairs=sum(g["legally_reached_both"] for g in report["groups"]),
            unreachable_groups=sum(not g["legally_reached_both"] for g in report["groups"]),
            autonomous_episodes=len(report["groups"])*WORLDS*len(ARMS),
            by_arm={arm:dict(
                teacher_first_stop=sum(g["teacher_probes"][arm]["first_stop"] for g in report["groups"]),
                teacher_final_stop=sum(g["teacher_probes"][arm]["final_stop"] for g in report["groups"]),
                autonomous_final_completion=sum(g["autonomous"][arm]["final_goal_completions"] for g in report["groups"]),
                autonomous_premature_stop=sum(g["autonomous"][arm]["premature_voluntary_stops"] for g in report["groups"]),
                mean_ink=round(sum(g["autonomous"][arm]["mean_ink"] for g in report["groups"])/len(report["groups"]),6))
                for arm in ARMS},elapsed_seconds=round(time.monotonic()-started,5),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if base.sha_file(source)!=report["source_sha256"]:
            raise AssertionError("Source mutated")
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
    for key in ("request","upstream","ft003","ft014","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--cohort",required=True)
    a=p.parse_args()
    r=run(a.request,a.upstream,a.ft003,a.ft014,a.output,a.seed,a.cohort)
    print(json.dumps(dict(lease=r["lease"],status=r["status"],aggregate=r["aggregate"]),sort_keys=True))
