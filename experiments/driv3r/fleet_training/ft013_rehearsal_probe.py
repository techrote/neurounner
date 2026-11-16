"""FT-013: paired sparse negative vs reached-goal rehearsal, external synthetic game.

No CyberSand runtime edits. Genuine frozen FT-003 checkpoint continuation,
full host-legal action menus, on-policy mining without STOP masking/teleportation,
matched Adam exposure and independent fresh-goal/autonomous evaluations.
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
import ft011_goal_stop as previous

ID = "FT-013-20261116-stop-rehearsal-retention"
SCHEMA = "cybersand.fleet-training.stop-rehearsal-request/1"
ARMS = ("goal_only_control", "sparse_hardnegative", "sparse_negative_plus_goal_rehearsal")
SEEDS = (9401, 9403)
SCENARIOS = previous.SCENARIOS
TRAIN_GOALS = 6
FRESH_GOALS = 6
AUTONOMOUS_WORLDS = 4
MINE_WORLDS = 4
MINE_ROUNDS = 2
MAX_MINED_PER_ROUND = 12
NEGATIVE_REPEATS = 2
STOP_NEGATIVE_LOGIT = 1.6
SPARSE_STRIDE = 2
REHEARSAL_PER_ROUND = 6
GOAL_REPEATS = 3
TEMPERATURE = previous.TEMPERATURE
POOL = 48
WALL_SECONDS = 300
RSS_KIB = 900000


def checked_request(r):
    expected = {
        "schema": SCHEMA, "request_id": ID,
        "source_commit": shift.SOURCE_COMMIT, "source_blob": shift.SOURCE_BLOB,
        "ft003_run_id": 43797061143, "ft003_artifact_id": 14194214028,
        "ft003_zip_sha256": shift.FT003_SHA,
        "seeds": list(SEEDS), "scenarios": list(SCENARIOS), "arms": list(ARMS),
        "train_reached_goals_per_group": TRAIN_GOALS,
        "fresh_heldout_goals_per_group": FRESH_GOALS,
        "autonomous_worlds_per_group": AUTONOMOUS_WORLDS,
        "onpolicy_worlds_per_arm_round": MINE_WORLDS,
        "onpolicy_rounds": MINE_ROUNDS,
        "max_hard_negatives_per_round": MAX_MINED_PER_ROUND,
        "negative_repeats": NEGATIVE_REPEATS,
        "negative_stop_logit_penalty": STOP_NEGATIVE_LOGIT,
        "sparse_negative_stride": SPARSE_STRIDE,
        "goal_rehearsal_examples_per_round": REHEARSAL_PER_ROUND,
        "goal_repeats": GOAL_REPEATS, "temperature": TEMPERATURE,
        "max_seconds_per_seed": WALL_SECONDS,
        "max_rss_kib_per_seed": RSS_KIB, "max_parallel_runner_jobs": 2,
        "native_material_edits": 0, "pretrained_slm_optimizer_steps": 0
    }
    if type(r) is not dict or r != expected:
        raise ValueError("FT-013 immutable input/model/custody/budget mismatch")
    return r


def budget(start):
    if time.monotonic()-start > WALL_SECONDS:
        raise TimeoutError("FT-013 training/evaluation wall budget exceeded")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > RSS_KIB:
        raise MemoryError("FT-013 memory budget exceeded")


def lease(seed, niche, arm):
    if seed not in SEEDS or niche not in base.NICHES or arm not in ARMS:
        raise ValueError("Unknown optimizer lease")
    return f"{ID}/{seed}/{niche}/{arm}"


def seed_for(pop, split, seed, niche, scenario, index):
    if (split not in ("TRAIN", "MINE", "FRESH_GOAL", "AUTONOMOUS")
        or seed not in SEEDS or niche not in base.NICHES
        or scenario not in SCENARIOS or type(index) is not int
        or not 0 <= index < POOL):
        raise ValueError("Invalid world/held-out lease")
    if split == "TRAIN":
        return previous.world_seed(pop, "TRAIN", seed, niche, scenario, index)
    return pop.stable_seed(ID, split, seed, niche, scenario, index)


def fresh_cases(pop, seed, niche, scenario, count, used):
    found = []
    for i in range(POOL):
        sid = seed_for(pop, "FRESH_GOAL", seed, niche, scenario, i)
        if sid in used:
            raise AssertionError("Leaked/colliding FT-012 held-out world")
        used.add(sid)
        route = goal.legal_route(pop, shift.world_for(pop, sid, scenario))
        if route is not None:
            found.append((sid, tuple(route)))
            if len(found) == count:
                break
    if len(found) != count:
        raise AssertionError("Fresh legally reached goals unavailable")
    return found


def teach_goal_routes(pop, policies, niche, scenario, seed, used, start):
    found = previous.goal_cases(pop, "TRAIN", seed, niche, scenario, TRAIN_GOALS, used)
    before = {a: p.updates for a,p in policies.items()}
    first = None
    samples = []
    for sid, route in found:
        w = shift.world_for(pop, sid, scenario)
        receipts = []
        for action in (*route, "STOP"):
            menu = w.legal()
            at_goal = w.cursor == w.goal
            if action not in menu or at_goal != (action=="STOP") or w.switched:
                raise AssertionError("Goal teacher broke host state or legal menu")
            for _ in range(GOAL_REPEATS if at_goal else 1):
                X, target = previous.teacher(pop, niche, w, menu,
                                             previous.ARMS[1], at_goal)
                for p in policies.values():
                    p.teach(X, target)
            digest = shift.state_digest(w)
            if at_goal:
                X = np.stack([w.feature(a) for a in menu])
                samples.append({"seed":sid,"state_sha256":digest,"X":X,
                                "utilities":np.array([pop.utility(niche,x,w) for x in X],
                                                     dtype=np.float64),
                                "stop_idx":menu.index("STOP"),"menu":list(menu)})
            w.step(action)
            receipts.append({"turn":w.steps, "menu":menu, "choice":action,
                             "before":digest, "after":shift.state_digest(w)})
            if w.painted & (w.obstacles | w.protected):
                raise AssertionError("Illegal material paint")
            budget(start)
        if w.cursor != w.goal or not w.stopped:
            raise AssertionError("Goal route did not stop at reached goal")
        if first is None:
            first = {"seed":sid,"route":route,"receipts":receipts,
                     "classification":"LEGAL_TEACHER_NOT_AUTONOMOUS"}
    deltas = {a:p.updates-before[a] for a,p in policies.items()}
    if min(deltas.values()) <= 0 or len(set(deltas.values())) != 1:
        raise AssertionError("Goal optimizer exposure unmatched")
    return {"reached_goals":len(found), "matched_adam_updates":next(iter(deltas.values())),
            "first_legal_teacher":first, "rehearsal_samples":samples}


def mine_round(pop, policies, niche, scenario, seed, round_id, used, output, rehearsal, start):
    """Collect genuine model top-1 trajectories, including their own premature STOP.

    No STOP masking, injected movement, teacher paths or future-state teleporting.
    Both arms train against identical selected pre-update states for causal matching.
    """
    candidates, records = [], []
    for arm, p in policies.items():
        for i in range(MINE_WORLDS):
            sid = seed_for(pop, "MINE", seed, niche, scenario,
                           round_id*MINE_WORLDS+i)
            # Same mining-world start for both arms is deliberate pairing;
            # each arm runs a separate on-policy trajectory of its own.
            used.add(sid)
            w = shift.world_for(pop, sid, scenario)
            receipts, early_stop = [], False
            while not w.stopped and w.steps < w.max_steps:
                menu = w.legal()
                X = np.stack([w.feature(a) for a in menu])
                logits = p.logits(X)
                if not np.isfinite(logits).all():
                    raise AssertionError("Nonfinite real model logits")
                choice = menu[int(np.argmax(logits))]
                at_goal = w.cursor == w.goal
                if "STOP" in menu and len(menu) > 1 and not at_goal:
                    stop_idx = menu.index("STOP")
                    margin = float(logits[stop_idx] - max(
                        logits[j] for j in range(len(menu)) if j != stop_idx))
                    utilities = np.array([pop.utility(niche,x,w) for x in X],
                                         dtype=np.float64)
                    dist = abs(w.cursor[0]-w.goal[0])+abs(w.cursor[1]-w.goal[1])
                    candidates.append({
                        "state":shift.state_digest(w), "X":X, "utilities":utilities,
                        "stop_idx":stop_idx, "margin":margin, "distance":dist,
                        "origin_arm":arm, "seed":sid, "step":w.steps,
                        "stop_chosen":choice=="STOP", "menu":list(menu)})
                before = shift.state_digest(w)
                w.step(choice)
                receipts.append({"turn":w.steps,"menu":menu,"choice":choice,
                                 "before":before,"after":shift.state_digest(w)})
                if w.painted & (w.obstacles | w.protected):
                    raise AssertionError("On-policy illegal paint")
                budget(start)
            if w.stopped and w.steps < w.max_steps:
                early_stop = True
            records.append({"arm":arm, "seed":sid,"decisions":w.steps,
                            "early_stop":early_stop,
                            "saw_goal":bool(w.cursor==w.goal),
                            "model_forwards":len(receipts)})
            if i == 0:
                shift.audit_receipts(pop,receipts,niche,sid,scenario)
                d=output/"mining"/niche/scenario/str(round_id)
                d.mkdir(parents=True,exist_ok=True)
                (d/(arm+".svg")).write_text(pop.draw_svg(w))
                (d/(arm+".json")).write_text(json.dumps({
                    "classification":"REAL_ONPOLICY_SYNTHETIC_MODEL_TRAJECTORY",
                    "arm":arm,"seed":sid,"receipts":receipts},sort_keys=True)+"\n")
    # Rank high-STOP-confidence and near-goal legal nonterminal states.
    dedup = {}
    for c in candidates:
        key = c["state"]
        score = c["margin"] + (0.5 if c["distance"] <= 2 else 0.)
        if key not in dedup or score > dedup[key][0]:
            dedup[key]=(score,c)
    selected = [p[1] for p in sorted(dedup.values(),
                key=lambda p:(-p[0],p[1]["state"]))[:MAX_MINED_PER_ROUND]]
    if not selected:
        raise AssertionError("No on-policy hard-negative states mined")
    before={a:p.updates for a,p in policies.items()}
    for selected_index, c in enumerate(selected):
        if "STOP" not in c["menu"] or c["distance"] == 0:
            raise AssertionError("Hard-negative lacks legal nonterminal STOP")
        for _ in range(NEGATIVE_REPEATS):
            for arm,p in policies.items():
                vals=c["utilities"].copy()
                if arm!=ARMS[0] and selected_index % SPARSE_STRIDE == 0:
                    vals[c["stop_idx"]]-=STOP_NEGATIVE_LOGIT
                target=pop.softmax(vals/TEMPERATURE)
                if not np.isfinite(target).all() or target.sum() <= 0:
                    raise AssertionError("Invalid hard-negative target")
                p.teach(c["X"],target)
    if len(rehearsal)<REHEARSAL_PER_ROUND:
        raise AssertionError("Missing genuine reached-goal rehearsal samples")
    for sample in rehearsal[:REHEARSAL_PER_ROUND]:
        if sample["menu"][sample["stop_idx"]]!="STOP":
            raise AssertionError("Rehearsal sample is not host-legal STOP")
        for arm,p in policies.items():
            vals=sample["utilities"].copy()
            if arm==ARMS[2]:
                vals[sample["stop_idx"]]+=previous.STOP_BONUS
            target=pop.softmax(vals/TEMPERATURE)
            p.teach(sample["X"],target)
    deltas={a:p.updates-before[a] for a,p in policies.items()}
    if min(deltas.values())<=0 or len(set(deltas.values()))!=1:
        raise AssertionError("Hard-negative optimizer update asymmetry")
    budget(start)
    return {
        "round":round_id, "onpolicy_trajectories":records,
        "model_forwards":sum(r["model_forwards"] for r in records),
        "candidate_nonterminal_states":len(candidates),
        "unique_nonterminal_states":len(dedup),
        "selected_hard_negatives":len(selected),
        "sparse_penalized_hard_negatives":sum(i % SPARSE_STRIDE == 0 for i in range(len(selected))),
        "rehearsed_reached_goals":REHEARSAL_PER_ROUND,
        "selected_predicted_stop_top1":sum(c["stop_chosen"] for c in selected),
        "selected_near_goal":sum(c["distance"] <= 2 for c in selected),
        "matched_adam_updates":next(iter(deltas.values())),
        "selected_state_sha256":hashlib.sha256(
            json.dumps([c["state"] for c in selected]).encode()).hexdigest(),
        "selected_evidence":[{"state":c["state"],"seed":c["seed"],
            "origin_arm":c["origin_arm"],"step":c["step"],
            "stop_margin":round(c["margin"],7),"goal_distance":c["distance"],
            "stop_chosen":c["stop_chosen"]} for c in selected]}



def probe_stage(pop, policies, niche, scenario, cases, stage, start):
    """Repeated score-only fresh-goal probes reveal within-model STOP forgetting.

    Same independent worlds, model and host-legal menu before and after each
    on-policy optimizer round. NEVER use these probe states for optimizer work.
    """
    rows=[]
    for sid, route in cases:
        w=shift.world_for(pop,sid,scenario)
        initial={a:previous.stop_scores(pop,p,w) for a,p in policies.items()}
        for action in route:
            if action not in w.legal() or w.switched or w.cursor == w.goal:
                raise AssertionError("Probe path escaped live legal host")
            w.step(action)
        if w.switched or w.cursor != w.goal:
            raise AssertionError("Probe did not really reach live goal")
        reached={a:previous.stop_scores(pop,p,w) for a,p in policies.items()}
        rows.append({"seed":sid,"initial":initial,"goal":reached,
                     "goal_state_sha256":shift.state_digest(w)})
        budget(start)
    return {"stage":stage,"niche":niche,"scenario":scenario,
            "fresh_goal_count":len(cases),
            "model_forward_batches":len(cases)*len(ARMS)*2,
            "worlds_sha256":hashlib.sha256(
                json.dumps([x["seed"] for x in rows]).encode()).hexdigest(),
            "goal_top1":{a:sum(r["goal"][a]["stop_top1"] for r in rows) for a in ARMS},
            "initial_top1":{a:sum(r["initial"][a]["stop_top1"] for r in rows) for a in ARMS},
            "goal_margin_mean":{a:round(sum(r["goal"][a]["stop_margin"] for r in rows)/len(rows),7)
                                for a in ARMS},
            "initial_margin_mean":{a:round(sum(r["initial"][a]["stop_margin"] for r in rows)/len(rows),7)
                                   for a in ARMS},
            "scores":rows}

def run(request, upstream, frozen, output, seed):
    checked_request(json.loads(request.read_text()))
    if seed not in SEEDS:
        raise ValueError("Unknown seed job")
    pop, source = shift.load_source(upstream)
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report={"schema":"cybersand.fleet-training.stop-rehearsal-evidence/1",
        "request_id":ID,"seed":seed,"status":"starting",
        "classification":"REAL_NUMPY_ADAM_AND_ONPOLICY_EXTERNAL_SYNTHETIC_NOT_NATIVE_NOT_SLM",
        "source_commit":shift.SOURCE_COMMIT,"source_blob":shift.SOURCE_BLOB,
        "source_sha256":base.sha_file(source),
        "executor_sha256":base.sha_file(Path(__file__)),
        "request_sha256":base.sha_file(request),
        "ft003_original_zip_sha256":shift.FT003_SHA,
        "resident_training_processes_per_runner":1,"native_material_edits":0,
        "pretrained_slm_optimizer_steps":0,
        "workers":[],"groups":[],"mining":[],"probes":[],"failures":[]}
    try:
        report["original_ft003_audit"]=shift.archive_check(
            frozen,source,request.parent/"request_dose.json")
        report["status"]="running"
        used=set()
        for niche in base.NICHES:
            source_ckpt=frozen/f"dose_48/{seed}/models/{niche}.json"
            policies={a:shift.load_policy(pop,source_ckpt) for a in ARMS}
            initial={a:p.updates for a,p in policies.items()}
            heldout_by_scenario={}
            for scenario in SCENARIOS:
                cases=fresh_cases(pop,seed,niche,scenario,FRESH_GOALS,used)
                heldout_by_scenario[scenario]=cases
                positive=teach_goal_routes(pop,policies,niche,scenario,seed,used,start)
                rehearsal=positive.pop("rehearsal_samples")
                report["probes"].append(probe_stage(pop,policies,niche,scenario,cases,
                                                    "after_goal_training",start))
                d=output/"training"/niche/scenario
                d.mkdir(parents=True,exist_ok=True)
                (d/"original_teacher.json").write_text(
                    json.dumps(positive.pop("first_legal_teacher"),sort_keys=True)+"\n")
                rounds=[]
                for r in range(MINE_ROUNDS):
                    rounds.append(mine_round(pop,policies,niche,scenario,seed,r,used,
                                             output,rehearsal,start))
                    report["probes"].append(probe_stage(pop,policies,niche,scenario,cases,
                                                       f"after_mining_round_{r}",start))
                report["mining"].append({"niche":niche,"scenario":scenario,
                                        "positive":positive,"rounds":rounds})
            for arm,p in policies.items():
                d=output/"models"/arm
                d.mkdir(parents=True,exist_ok=True)
                dest=d/(niche+".json")
                dest.write_text(json.dumps(p.serialized(),sort_keys=True)+"\n")
                if base.sha_file(dest)==base.sha_file(source_ckpt):
                    raise AssertionError("Adam did not change checkpoint")
                report["workers"].append({"lease_id":lease(seed,niche,arm),
                    "state":"completed","niche":niche,"arm":arm,
                    "model":"population-mlp-24",
                    "source_checkpoint_sha256":base.sha_file(source_ckpt),
                    "trained_checkpoint_sha256":base.sha_file(dest),
                    "actual_adam_updates":p.updates-initial[arm]})
            for scenario in SCENARIOS:
                cases=heldout_by_scenario[scenario]
                fresh=previous.heldout_group(pop,policies,niche,scenario,cases,output,start)
                autos=[seed_for(pop,"AUTONOMOUS",seed,niche,scenario,i)
                       for i in range(AUTONOMOUS_WORLDS)]
                if any(s in used for s in autos):
                    raise AssertionError("Autonomy split reused another domain")
                used.update(autos)
                auto=previous.autonomy(pop,policies,niche,scenario,autos,output,start)
                report["groups"].append({
                    "niche":niche,"scenario":scenario,
                    "fresh_heldout_goal_scores":fresh,
                    "autonomous_world_seeds":autos,"autonomous":auto,
                    "goal_stop_top1":{a:sum(r["at_reached_goal"][a]["stop_top1"]
                                         for r in fresh) for a in ARMS},
                    "initial_stop_top1":{a:sum(r["initial"][a]["stop_top1"]
                                           for r in fresh) for a in ARMS}})
        report["seed_custody_sha256"]=hashlib.sha256(
            json.dumps(sorted(used)).encode()).hexdigest()
        report["aggregate"]={
            "trained_checkpoints":len(report["workers"]),
            "phase_probe_panels":len(report["probes"]),
            "goal_stop_by_phase":{stage:{a:sum(p["goal_top1"][a] for p in report["probes"]
                if p["stage"]==stage) for a in ARMS}
                for stage in ("after_goal_training","after_mining_round_0","after_mining_round_1")},
            "mean_goal_margin_by_phase":{stage:{a:round(sum(p["goal_margin_mean"][a] for p in report["probes"]
                if p["stage"]==stage)/max(1,sum(p["stage"]==stage for p in report["probes"])),7)
                for a in ARMS} for stage in ("after_goal_training","after_mining_round_0","after_mining_round_1")},
            "reward_improved_by_arm":{a:sum(g["autonomous"][a]["reward_per_decision"]
                >g["autonomous"][ARMS[0]]["reward_per_decision"] for g in report["groups"])
                for a in ARMS[1:]},
            "reward_regressed_by_arm":{a:sum(g["autonomous"][a]["reward_per_decision"]
                <g["autonomous"][ARMS[0]]["reward_per_decision"] for g in report["groups"])
                for a in ARMS[1:]},
            "adam_updates":sum(w["actual_adam_updates"] for w in report["workers"]),
            "mining_model_forwards":sum(r["model_forwards"] for g in report["mining"]
                                      for r in g["rounds"]),
            "unique_mined_hard_negative_samples":sum(r["selected_hard_negatives"]
                 for g in report["mining"] for r in g["rounds"]),
            "fresh_reached_goals":len(report["groups"])*FRESH_GOALS,
            "goal_stop_top1":{a:sum(g["goal_stop_top1"][a] for g in report["groups"])
                              for a in ARMS},
            "initial_stop_top1":{a:sum(g["initial_stop_top1"][a] for g in report["groups"])
                                 for a in ARMS},
            "autonomous_episodes":len(report["groups"])*len(ARMS)*AUTONOMOUS_WORLDS,
            "reward_improved_groups":sum(g["autonomous"][ARMS[1]]["reward_per_decision"]
                > g["autonomous"][ARMS[0]]["reward_per_decision"] for g in report["groups"]),
            "reward_regressed_groups":sum(g["autonomous"][ARMS[1]]["reward_per_decision"]
                < g["autonomous"][ARMS[0]]["reward_per_decision"] for g in report["groups"]),
            "early_stop_by_arm":{a:round(sum(g["autonomous"][a]["early_stop"]
                  for g in report["groups"])/len(report["groups"]),5) for a in ARMS},
            "mean_decisions_by_arm":{a:round(sum(g["autonomous"][a]["decisions"]
                  for g in report["groups"])/len(report["groups"]),5) for a in ARMS},
            "mean_ink_by_arm":{a:round(sum(g["autonomous"][a]["new_ink"]
                  for g in report["groups"])/len(report["groups"]),5) for a in ARMS},
            "mean_retraces_by_arm":{a:round(sum(g["autonomous"][a]["retraces"]
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
            base.sha_file(f)+"  "+str(f.relative_to(output))+"\n"
            for f in sorted(output.rglob("*"))
            if f.is_file() and f.name!="SHA256SUMS"))
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--request",type=Path,required=True)
    ap.add_argument("--upstream",type=Path,required=True)
    ap.add_argument("--ft003",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args()
    result=run(args.request,args.upstream,args.ft003,args.output,args.seed)
    print(json.dumps({"status":result["status"],"seed":args.seed,
                      "aggregate":result["aggregate"]},sort_keys=True))
