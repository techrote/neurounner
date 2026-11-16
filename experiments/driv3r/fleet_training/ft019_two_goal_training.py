"""FT-019: real, matched two-goal movement-policy optimizer continuations.

External synthetic CyberSand-like 22x22 host. No game, W0, native PAINT,
production provider, paid compute or pretrained SLM changes. Distinct frozen
FT-003 parents, split-separated legal teachers and held-out autonomous worlds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import fleet_bench as base
import ft004_shift as shift
import ft014_stop_sidecar as side
import ft018_two_stage as previous

ID = "FT-019-20261116-two-goal-navigation-optimizer"
SCHEMA = "cybersand.fleet-training.ft019-request/1"
HEAD_018 = "66252eeabc2d13b135d36f8fbe5a5d157294abe7"
BLOB_018 = "0a1f5970d82d59cc918b3f073065c0e511ca138e"
SEEDS, COHORTS = side.SEEDS, side.COHORTS
SCENARIOS = previous.SCENARIOS
ARMS = ("frozen", "legal_route", "recovery_rehearsal")
TRAIN = ("legal_route", "recovery_rehearsal")
CANDIDATES, PAIRS, RECOVERY_WORLDS, TEST_WORLDS = 96, 2, 8, 10
UPDATES = 600
MAX_SECONDS, MAX_RSS_KIB = 450, 900000


def expected_request():
    return dict(schema=SCHEMA,request_id=ID,
        source_commit=shift.SOURCE_COMMIT, source_blob=shift.SOURCE_BLOB,
        ft018_source_head=HEAD_018,ft018_git_blob=BLOB_018,
        ft003_artifact_id=14194214028,ft003_zip_sha256=shift.FT003_SHA,
        seeds=list(SEEDS),cohorts={k:list(v) for k,v in COHORTS.items()},
        scenarios=list(SCENARIOS),arms=list(ARMS),
        teacher_candidates_per_group=CANDIDATES,teacher_pairs_per_group=PAIRS,
        recovery_mining_worlds_per_group=RECOVERY_WORLDS,
        heldout_worlds_per_group=TEST_WORLDS,
        adam_updates_per_continuation=UPDATES,
        max_seconds_per_job=MAX_SECONDS,max_rss_kib_per_job=MAX_RSS_KIB,
        max_parallel_runner_jobs=4,native_material_edits=0,
        pretrained_slm_optimizer_steps=0)


def check_request(obj):
    if type(obj) is not dict or obj != expected_request():
        raise ValueError("FT019 model/source/split/budget/request drift")
    return obj


def lease(seed,cohort,niche,arm):
    if seed not in SEEDS or cohort not in COHORTS or niche not in COHORTS[cohort] or arm not in ARMS:
        raise ValueError("Foreign model lease")
    return f"{ID}/{seed}/{cohort}/{niche}/{arm}"


def world_seed(pop,domain,seed,niche,scenario,i):
    ranges={"TEACHER":CANDIDATES,"RECOVERY":RECOVERY_WORLDS,"HELDOUT":TEST_WORLDS}
    if (domain not in ranges or seed not in SEEDS or niche not in base.NICHES
        or scenario not in SCENARIOS or type(i) is not int or not 0<=i<ranges[domain]):
        raise ValueError("Invalid unique world lease")
    return pop.stable_seed(ID,domain,seed,niche,scenario,i)


def budget(t0):
    if time.monotonic()-t0>MAX_SECONDS:
        raise TimeoutError("FT019 elapsed time ceiling")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>MAX_RSS_KIB:
        raise MemoryError("FT019 RSS ceiling")


def action_features(world):
    menu=[a for a in world.legal() if a!="STOP"]
    if not menu:
        return menu,None
    X=np.stack([world.feature(a) for a in menu])
    if X.shape[1]!=20 or not np.isfinite(X).all():
        raise AssertionError("Host action-feature contract drift")
    return menu,X


def choose(world,policy):
    menu=world.legal()
    if len(menu)==1:
        return menu[0],False
    if world.switched and world.cursor==world.goal:
        return "STOP",True
    moves,X=action_features(world)
    if not moves: return "STOP",False
    logits=policy.logits(X)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite neural movement logits")
    return moves[int(np.argmax(logits))],False


def example(world,action):
    moves,X=action_features(world)
    if action not in moves:
        raise AssertionError("Teacher selected action outside legal menu")
    if len(moves)<2:
        return None
    y=np.zeros(len(moves),dtype=np.float64)
    y[moves.index(action)]=1
    return X,y


def legal_teacher_samples(pop,seed,niche,scenario,seen):
    """Reconstruct real legal actions; exclude arbitrary between-goal waiting.

    The first-goal and second-goal segments use distinct host goals. Host
    replacements occur on the original legal turn count, never by teleporting.
    """
    out=[];cases=[];tries=0
    for i in range(CANDIDATES):
        sid=world_seed(pop,"TEACHER",seed,niche,scenario,i)
        if sid in seen: raise AssertionError("Duplicate training world dispatch")
        seen.add(sid);tries+=1
        result=previous.teacher_pair(pop,sid,scenario)
        if result is None:continue
        record,_=result
        w=previous.make_world(pop,sid,scenario)
        first_len=len(record["first_route"])
        saved=0;retrace=0
        for receipt in record["receipts"]:
            if (receipt["before"]!=shift.state_digest(w)
                or receipt["menu"]!=w.legal()):
                raise AssertionError("Teacher replay pre-state drift")
            if w.steps<first_len or w.switched:
                e=example(w,receipt["choice"])
                if e is not None:out.append(e);saved+=1
                retrace+=int(receipt["choice"]=="RETRACE")
            w.step(receipt["choice"])
            if receipt["after"]!=shift.state_digest(w):
                raise AssertionError("Teacher replay post-state drift")
        if w.cursor!=w.goal or not w.switched:
            raise AssertionError("Teacher did not reach second active goal")
        cases.append(dict(seed=sid,first_steps=first_len,
            terminal_turn=w.steps,samples=saved,retraces=retrace,
            state_sha256=shift.state_digest(w)))
        if len(cases)==PAIRS:break
    return out,cases,tries


def recovery_samples(pop,frozen,seed,niche,scenario,seen):
    """On-policy states sampled AFTER an actual legal host goal replacement."""
    examples=[];records=[]
    for i in range(RECOVERY_WORLDS):
        sid=world_seed(pop,"RECOVERY",seed,niche,scenario,i)
        if sid in seen:raise AssertionError("Duplicate recovery world lease")
        seen.add(sid)
        w=previous.make_world(pop,sid,scenario)
        while w.steps<w.max_steps//2 and not w.stopped:
            a,_=choose(w,frozen)
            w.step(a)
        if not w.switched or w.stopped:continue
        for step in range(4):
            path=previous.second_route(pop,w)
            if path:
                e=example(w,path[0])
                if e is not None:
                    examples.append(e)
                    records.append(dict(world_seed=sid,turn=w.steps,
                        correction=path[0],goal_distance=side.distance(w),
                        state_sha256=shift.state_digest(w)))
            if w.stopped or w.steps>=w.max_steps-1:break
            a,_=choose(w,frozen)
            if a=="STOP":break
            w.step(a)
    return examples,records


def fit(pop,checkpoint,arm,teacher,recovery,seed,niche):
    if arm not in TRAIN:raise ValueError("Not a training arm")
    p=shift.load_policy(pop,checkpoint)
    if not teacher:raise ValueError("No valid legal-route training examples")
    if arm=="recovery_rehearsal" and not recovery:
        raise ValueError("No legally recoverable on-policy states; preserve negative")
    rng=np.random.default_rng(pop.stable_seed(ID,"MATCHED_ORDER",seed,niche))
    losses=[]
    for step in range(UPDATES):
        cases=recovery if (arm=="recovery_rehearsal" and step%3==0) else teacher
        X,y=cases[int(rng.integers(0,2**32)) % len(cases)]
        losses.append(float(p.teach(X,y)))
    if p.updates!=UPDATES or not np.isfinite(np.asarray(losses)).all():
        raise AssertionError("Actual Adam update count/gradient failure")
    return p,dict(adam_updates=p.updates,teacher_examples=len(teacher),
        recovery_examples=len(recovery),mean_training_loss=round(float(np.mean(losses)),6))


def episode(pop,policy,sid,niche,scenario,trace=False):
    w=previous.make_world(pop,sid,scenario)
    receipts=[];retrace=0;visits=0;premature=0;forced=0;complete=0;forwards=0
    ending="horizon"
    while not w.stopped and w.steps<w.max_steps:
        menu=w.legal()
        a,at_final=choose(w,policy)
        if a not in menu:raise AssertionError("Model action outside host menu")
        if len(menu)>1 and not at_final:forwards+=1
        visits+=int(w.switched and w.cursor==w.goal)
        before=shift.state_digest(w)
        if a=="STOP":
            forced=int(len(menu)==1)
            complete=int(not forced and at_final)
            premature=int(not forced and not at_final)
            ending="completed" if complete else "host_forced" if forced else "premature"
        w.step(a)
        retrace+=int(a=="RETRACE")
        if w.painted & (w.protected|w.obstacles):
            raise AssertionError("Host-protected material edited")
        if trace:
            receipts.append(dict(turn=w.steps,before=before,after=shift.state_digest(w),
                menu=menu,choice=a))
    e=dict(world_seed=sid,scenario=scenario,niche=niche,
        decisions=w.steps,new_ink=len(w.painted)-1,retraces=retrace,
        goal_visits=visits,completed_final_goal=complete,
        premature_stop=premature,host_forced_stop=forced,
        post_switch_steps=sum(int(x["turn"]>w.max_steps//2) for x in receipts) if trace else max(0,w.steps-w.max_steps//2),
        neural_decisions=forwards,ending=ending)
    if trace:
        e["receipts"]=receipts
        replay=previous.make_world(pop,sid,scenario)
        for r in receipts:
            if r["before"]!=shift.state_digest(replay) or r["menu"]!=replay.legal():
                raise AssertionError("Autonomous replay pre-state mismatch")
            replay.step(r["choice"])
            if r["after"]!=shift.state_digest(replay):
                raise AssertionError("Autonomous replay state mismatch")
        if shift.state_digest(replay)!=shift.state_digest(w):
            raise AssertionError("Autonomous replay terminal mismatch")
    return e,w


def summary(cases):
    return dict(episodes=len(cases),
        final_goal_completions=sum(x["completed_final_goal"] for x in cases),
        host_forced=sum(x["host_forced_stop"] for x in cases),
        premature=sum(x["premature_stop"] for x in cases),
        final_goal_visits=sum(x["goal_visits"] for x in cases),
        mean_ink=round(sum(x["new_ink"] for x in cases)/len(cases),5),
        mean_retraces=round(sum(x["retraces"] for x in cases)/len(cases),5),
        mean_decisions=round(sum(x["decisions"] for x in cases)/len(cases),5),
        neural_decisions=sum(x["neural_decisions"] for x in cases))


def run(request,upstream,ft003,output,seed,cohort):
    check_request(json.loads(request.read_text()))
    if seed not in SEEDS or cohort not in COHORTS:raise ValueError("Invalid job lease")
    pop,source=shift.load_source(upstream)
    output.mkdir(parents=True,exist_ok=False)
    t0=time.monotonic()
    report=dict(schema="cybersand.fleet-training.ft019-evidence/1",
        request_id=ID,job_lease=f"{ID}/{seed}/{cohort}",
        seed=seed,cohort=cohort,status="starting",failures=[],
        classification="REAL_ADAM_EXTERNAL_SYNTHETIC_22X22_TINY_GAME_POLICY_NOT_NATIVE_NOT_SLM",
        source_commit=shift.SOURCE_COMMIT,source_blob=shift.SOURCE_BLOB,
        source_sha256=base.sha_file(source),
        executor_sha256=base.sha_file(Path(__file__)),
        request_sha256=base.sha_file(request),
        ft003_zip_sha256=shift.FT003_SHA,
        resident_training_processes_per_runner=1,models=[],groups=[])
    try:
        report["upstream_custody"]=shift.archive_check(
            ft003,source,request.parent/"request_dose.json")
        seen=set()
        for niche in COHORTS[cohort]:
            parent=ft003/f"dose_48/{seed}/models/{niche}.json"
            parent_sha=base.sha_file(parent)
            frozen=shift.load_policy(pop,parent)
            teacher=[];recovery=[];sources=[]
            for scenario in SCENARIOS:
                samples,pairs,tries=legal_teacher_samples(pop,seed,niche,scenario,seen)
                teacher.extend(samples)
                extra,mine=recovery_samples(pop,frozen,seed,niche,scenario,seen)
                recovery.extend(extra)
                sources.append(dict(scenario=scenario,candidates_tried=tries,
                    legal_two_goal_pairs=len(pairs),legal_route_examples=len(samples),
                    recovery_examples=len(extra),teacher_endpoints=pairs,
                    recovered_states=mine))
                budget(t0)
            if not all(x["legal_two_goal_pairs"]>0 for x in sources):
                raise RuntimeError("One or more scenarios yielded zero legal two-goal teachers")
            models={"frozen":frozen}
            for arm in TRAIN:
                p,metrics=fit(pop,parent,arm,teacher,recovery,seed,niche)
                models[arm]=p
                path=output/"models"/niche
                path.mkdir(parents=True,exist_ok=True)
                dest=path/(arm+".json")
                dest.write_text(json.dumps(p.serialized(),sort_keys=True)+"\n")
                report["models"].append(dict(lease=lease(seed,cohort,niche,arm),
                    parent_checkpoint_sha256=parent_sha,
                    checkpoint_sha256=base.sha_file(dest),**metrics))
            report["models"].append(dict(lease=lease(seed,cohort,niche,"frozen"),
                parent_checkpoint_sha256=parent_sha,adam_updates=0))
            for scenario in SCENARIOS:
                cases={arm:[] for arm in ARMS}
                for i in range(TEST_WORLDS):
                    sid=world_seed(pop,"HELDOUT",seed,niche,scenario,i)
                    if sid in seen:raise AssertionError("Training/test world collision")
                    seen.add(sid)
                    for arm in ARMS:
                        e,w=episode(pop,models[arm],sid,niche,scenario,trace=i==0)
                        cases[arm].append(e)
                        if i==0:
                            gallery=output/"gallery"/niche/scenario
                            gallery.mkdir(parents=True,exist_ok=True)
                            (gallery/(arm+".svg")).write_text(pop.draw_svg(w))
                            (gallery/(arm+".json")).write_text(json.dumps(e,sort_keys=True)+"\n")
                    budget(t0)
                ep_dir=output/"episodes"/niche
                ep_dir.mkdir(parents=True,exist_ok=True)
                (ep_dir/(scenario+".json")).write_text(json.dumps(cases,sort_keys=True)+"\n")
                report["groups"].append(dict(niche=niche,scenario=scenario,
                    autonomous={arm:summary(cases[arm]) for arm in ARMS}))
            report.setdefault("training_sources",{})[niche]=sources
        report["aggregate"]=dict(
            model_experiments=len(report["models"]),
            new_optimizer_updates=sum(x["adam_updates"] for x in report["models"]),
            original_heldout_episodes=len(report["groups"])*TEST_WORLDS*len(ARMS),
            unique_world_leases=len(seen),
            recovered_onpolicy_examples=sum(m["recovery_examples"] for m in report["models"] if "recovery_examples" in m and m["lease"].endswith("recovery_rehearsal")),
            by_arm={arm:dict(
                final_goal_completions=sum(g["autonomous"][arm]["final_goal_completions"] for g in report["groups"]),
                host_forced=sum(g["autonomous"][arm]["host_forced"] for g in report["groups"]),
                mean_ink=round(sum(g["autonomous"][arm]["mean_ink"] for g in report["groups"])/len(report["groups"]),5),
                mean_retraces=round(sum(g["autonomous"][arm]["mean_retraces"] for g in report["groups"])/len(report["groups"]),5))
                for arm in ARMS},
            elapsed_seconds=round(time.monotonic()-t0,5),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if base.sha_file(source)!=report["source_sha256"]:raise AssertionError("Pinned source mutated")
        report["status"]="completed"
    except BaseException as exc:
        report["status"]="failed"
        report["failures"].append(type(exc).__name__+": "+str(exc))
        raise
    finally:
        (output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        (output/"SHA256SUMS").write_text("".join(
            base.sha_file(p)+"  "+str(p.relative_to(output))+"\n"
            for p in sorted(output.rglob("*")) if p.is_file() and p.name!="SHA256SUMS"))
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for k in ("request","upstream","ft003","output"):
        ap.add_argument("--"+k,required=True,type=Path)
    ap.add_argument("--seed",required=True,type=int)
    ap.add_argument("--cohort",required=True)
    a=ap.parse_args()
    r=run(a.request,a.upstream,a.ft003,a.output,a.seed,a.cohort)
    print(json.dumps(dict(lease=r["job_lease"],status=r["status"],
        aggregate=r["aggregate"]),sort_keys=True))
