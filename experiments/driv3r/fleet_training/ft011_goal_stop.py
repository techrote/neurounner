"""FT-011: matched legal-goal STOP curriculum for 24-unit game-policy models.

External CyberSand-like synthetic host (NOT native PAINT, W0 or pretrained SLM).
Train on legal paths reaching live goals, retain the full legal action menu,
evaluate separately on fresh reached-goal states and autonomous drawings.
No game-side changes and no training of WORM-001's owned checkpoints.
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
import ft009_situational_stop as old
import ft010_goal_reachability as goal

ID = "FT-011-20261116-reached-goal-stop-adam"
SCHEMA = "cybersand.fleet-training.reached-goal-stop-request/1"
SEEDS = (9401, 9403)
ARMS = ("plain_goal_exposure", "goal_stop_bonus")
SCENARIOS = ("long60", "dense60", "shifted_guard60")
TRAIN_PER_GROUP = 6
EVAL_PER_GROUP = 6
AUTO_PER_GROUP = 4
POOL = 48
GOAL_REPEATS = 3
STOP_BONUS = 5.0
TEMPERATURE = 0.42
WALL_SECONDS = 300
RSS_KIB = 900000


def checked_request(doc):
    expected = {
        "schema": SCHEMA, "request_id": ID,
        "source_commit": shift.SOURCE_COMMIT, "source_blob": shift.SOURCE_BLOB,
        "ft003_run_id": 43797061143,
        "ft003_artifact_id": 14194214028,
        "ft003_zip_sha256": shift.FT003_SHA,
        "ft010_request_id": goal.ID,
        "seeds": list(SEEDS), "scenarios": list(SCENARIOS),
        "arms": list(ARMS), "train_reached_goals_per_group": TRAIN_PER_GROUP,
        "fresh_heldout_goals_per_group": EVAL_PER_GROUP,
        "autonomous_worlds_per_group": AUTO_PER_GROUP,
        "candidate_pool_per_group": POOL,
        "repeated_terminal_samples": GOAL_REPEATS,
        "goal_stop_logit_teacher_bonus": STOP_BONUS,
        "target_temperature": TEMPERATURE,
        "max_seconds_per_seed": WALL_SECONDS,
        "max_rss_kib_per_seed": RSS_KIB,
        "max_parallel_runner_jobs": 2,
        "native_material_edits": 0,
        "pretrained_slm_optimizer_steps": 0
    }
    if type(doc) is not dict or doc != expected:
        raise ValueError("FT-011 immutable source/checkpoint/custody/budget mismatch")
    return doc


def lease(seed, niche, arm):
    if seed not in SEEDS or niche not in base.NICHES or arm not in ARMS:
        raise ValueError("Unknown optimizer lease")
    return f"{ID}/{seed}/{niche}/{arm}"


def world_seed(pop, split, seed, niche, scenario, i):
    if (seed not in SEEDS or niche not in base.NICHES
            or scenario not in SCENARIOS or type(i) is not int
            or not 0 <= i < POOL):
        raise ValueError("Invalid world seed domain")
    if split == "TRAIN":
        # FT-010's training-candidate split was source-qualified and separate
        # from its held-out-candidate split. Never consume the latter here.
        return goal.seed_for(pop, "TRAIN_CANDIDATES", seed, niche, scenario, i)
    if split not in ("FRESH_HELDOUT_GOAL", "AUTONOMOUS"):
        raise ValueError("Unregistered or exposed held-out split")
    return pop.stable_seed(ID, split, seed, niche, scenario, i)


def check_budget(t0):
    if time.monotonic() - t0 > WALL_SECONDS:
        raise TimeoutError("FT-011 wall deadline exceeded")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > RSS_KIB:
        raise MemoryError("FT-011 RSS cap exceeded")


def goal_cases(pop, split, seed, niche, scenario, wanted, used):
    """Choose only legally reachable independent worlds, never teleport."""
    found = []
    for i in range(POOL):
        sid = world_seed(pop, split, seed, niche, scenario, i)
        if sid in used:
            raise AssertionError("Train/heldout/autonomy seed collision")
        used.add(sid)
        w = shift.world_for(pop, sid, scenario)
        route = goal.legal_route(pop, w)
        if route is None:
            continue
        if len(route) == 0 or len(route) >= w.max_steps // 2 - 1:
            raise AssertionError("Improper reached-goal custody route")
        found.append((sid, tuple(route)))
        if len(found) == wanted:
            break
    if len(found) < wanted:
        raise AssertionError(f"Insufficient real legal goals {split}/{seed}/{niche}/{scenario}: {len(found)}/{wanted}")
    return found


def teacher(pop, niche, w, menu, arm, at_goal):
    if arm not in ARMS or menu != w.legal() or "STOP" not in menu:
        raise ValueError("Teacher must use complete genuine host-legal menu")
    X = np.stack([w.feature(a) for a in menu])
    vals = np.asarray([pop.utility(niche, x, w) for x in X], dtype=np.float64)
    if arm == ARMS[1] and at_goal:
        vals[menu.index("STOP")] += STOP_BONUS
    target = pop.softmax(vals / TEMPERATURE)
    if not np.isfinite(target).all() or target.shape != (len(menu),):
        raise AssertionError("Nonfinite/incorrect teacher targets")
    return X, target


def train_group(pop, policies, niche, cases, out, began):
    starts = {arm: p.updates for arm, p in policies.items()}
    state_hashes, first_route_receipts = [], None
    terminal_count, nonterminal_count = 0, 0
    for sample_id, (sid, route) in enumerate(cases):
        w = shift.world_for(pop, sid, out["scenario"])
        receipts = []
        for action in (*route, "STOP"):
            menu = w.legal()
            at_goal = w.cursor == w.goal
            if action not in menu or at_goal != (action == "STOP") or w.switched:
                raise AssertionError("Scaffold left legal current-goal window")
            state_hashes.append(shift.state_digest(w))
            if len(menu) > 1:
                repeats = GOAL_REPEATS if at_goal else 1
                for _ in range(repeats):
                    for arm, p in policies.items():
                        X, target = teacher(pop, niche, w, menu, arm, at_goal)
                        p.teach(X, target)
                terminal_count += int(at_goal) * repeats
                nonterminal_count += int(not at_goal)
            before = shift.state_digest(w)
            w.step(action)
            receipts.append({"menu": menu, "action": action, "before": before,
                             "after": shift.state_digest(w), "turn": w.steps})
            if w.painted & (w.protected | w.obstacles):
                raise AssertionError("Protected/obstacle material painted")
            check_budget(began)
        if not w.stopped or w.cursor != w.goal or w.switched:
            raise AssertionError("Teacher route did not terminate legally at goal")
        if sample_id == 0:
            first_route_receipts = {"seed": sid, "scenario": out["scenario"],
                                    "path": list(route), "receipts": receipts,
                                    "classification": "LEGAL_SYNTHETIC_TEACHER_NOT_MODEL"}
    deltas = {a: p.updates - starts[a] for a,p in policies.items()}
    if min(deltas.values()) <= 0 or len(set(deltas.values())) != 1:
        raise AssertionError("Unmatched/absent real Adam steps")
    return {"train_reached": len(cases), "terminal_examples": terminal_count,
            "nonterminal_examples": nonterminal_count,
            "matched_updates_per_arm": next(iter(deltas.values())),
            "teacher_state_sha256": hashlib.sha256(
                json.dumps(state_hashes, sort_keys=True).encode()).hexdigest(),
            "sample": first_route_receipts}


def stop_scores(pop, policy, w):
    menu = w.legal()
    if "STOP" not in menu or len(menu) <= 1:
        raise AssertionError("No nontrivial host-legal STOP ranking opportunity")
    X = np.stack([w.feature(a) for a in menu])
    scores = policy.logits(X)
    if not np.isfinite(scores).all() or scores.shape != (len(menu),):
        raise AssertionError("Invalid actual trained-model forward")
    idx = menu.index("STOP")
    other = max(scores[j] for j in range(len(menu)) if j != idx)
    margin = float(scores[idx] - other)
    return {"stop_margin": round(margin, 7),
            "stop_top1": bool(int(np.argmax(scores)) == idx),
            "full_menu": list(menu)}


def heldout_group(pop, policies, niche, scenario, cases, output, began):
    rows = []
    for index, (sid, route) in enumerate(cases):
        w = shift.world_for(pop, sid, scenario)
        before = {a: stop_scores(pop, p, w) for a,p in policies.items()}
        receipts = []
        for action in route:
            menu = w.legal()
            if action not in menu or w.cursor == w.goal or w.switched:
                raise AssertionError("Held-out route action not legal or live")
            digest = shift.state_digest(w)
            w.step(action)
            receipts.append({"menu": menu, "action": action, "before": digest,
                             "after": shift.state_digest(w), "turn": w.steps})
        if w.cursor != w.goal or w.switched:
            raise AssertionError("Held-out goal was not legally reached")
        goal_state_hash = shift.state_digest(w)
        reached = {a: stop_scores(pop, p, w) for a,p in policies.items()}
        stop_menu = w.legal()
        w.step("STOP")
        if not w.stopped or w.painted & (w.obstacles | w.protected):
            raise AssertionError("Protected paint or invalid legal goal STOP")
        receipts.append({"menu": stop_menu, "action": "STOP",
                         "before": goal_state_hash, "after": shift.state_digest(w),
                         "turn": w.steps})
        rows.append({"seed": sid, "steps_to_goal": len(route),
                     "initial": before, "at_reached_goal": reached})
        if index == 0:
            d = output / "heldout" / niche / scenario
            d.mkdir(parents=True, exist_ok=True)
            (d / "original_route.svg").write_text(pop.draw_svg(w))
            (d / "original_route.json").write_text(json.dumps({
                "classification": "LEGAL_SYNTHETIC_HELDOUT_SCAFFOLD_NOT_AUTONOMOUS",
                "seed":sid, "route": list(route), "receipts":receipts,
                "initial_model_scores":before, "goal_model_scores":reached
            }, sort_keys=True)+"\n")
        check_budget(began)
    return rows


def autonomy(pop, policies, niche, scenario, seeds, output, began):
    result = {}
    for arm, policy in policies.items():
        episodes = []
        for i, sid in enumerate(seeds):
            ep, world = shift.execute(pop, policy, niche, sid, scenario,
                                      save_trace=(i == 0))
            if ep["model_forwards"] != ep["decisions"] or ep["protected_violations"]:
                raise AssertionError("Invalid real-model legal-autonomy receipts")
            if i == 0:
                shift.audit_receipts(pop, ep["receipts"], niche, sid, scenario)
                d = output / "gallery" / niche / scenario
                d.mkdir(parents=True, exist_ok=True)
                (d / (arm+".svg")).write_text(pop.draw_svg(world))
                (d / (arm+".json")).write_text(json.dumps(ep,sort_keys=True)+"\n")
            episodes.append(ep)
            check_budget(began)
        result[arm] = shift.summarize(episodes)
    return result


def run(request, upstream, frozen, output, seed):
    checked_request(json.loads(request.read_text()))
    if seed not in SEEDS:
        raise ValueError("Unknown seed job lease")
    pop, source = shift.load_source(upstream)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = {
        "schema":"cybersand.fleet-training.reached-goal-stop-evidence/1",
        "request_id":ID, "seed":seed, "status":"starting",
        "classification":"REAL_NUMPY_ADAM_EXTERNAL_SYNTHETIC_NOT_NATIVE_NOT_W0_NOT_SLM",
        "source_commit":shift.SOURCE_COMMIT, "source_blob":shift.SOURCE_BLOB,
        "source_sha256":base.sha_file(source),
        "executor_sha256":base.sha_file(Path(__file__)),
        "request_sha256":base.sha_file(request),
        "ft003_original_zip_sha256":shift.FT003_SHA,
        "native_material_edits":0, "pretrained_slm_optimizer_steps":0,
        "resident_model_processes_per_runner":1,
        "workers":[], "groups":[], "failures":[]}
    try:
        report["original_ft003_audit"] = shift.archive_check(
            frozen, source, request.parent/"request_dose.json")
        report["status"] = "running"
        used = set()
        for niche in base.NICHES:
            source_ckpt = frozen / f"dose_48/{seed}/models/{niche}.json"
            policies = {arm:shift.load_policy(pop, source_ckpt) for arm in ARMS}
            train_exposure, heldouts_by_scenario = [], {}
            for scenario in SCENARIOS:
                training = goal_cases(pop,"TRAIN",seed,niche,scenario,
                                      TRAIN_PER_GROUP,used)
                heldout = goal_cases(pop,"FRESH_HELDOUT_GOAL",seed,niche,scenario,
                                     EVAL_PER_GROUP,used)
                a_seeds = [world_seed(pop,"AUTONOMOUS",seed,niche,scenario,i)
                           for i in range(AUTO_PER_GROUP)]
                if any(s in used for s in a_seeds):
                    raise AssertionError("Autonomous seed domain reuse")
                used.update(a_seeds)
                details = train_group(pop,policies,niche,training,{"scenario":scenario},began)
                sample = details.pop("sample")
                train_exposure.append({"scenario":scenario,**details})
                heldouts_by_scenario[scenario] = heldout
                if sample is not None:
                    d = output/"training"/niche/scenario
                    d.mkdir(parents=True,exist_ok=True)
                    (d/"original_teacher.json").write_text(json.dumps(
                        sample,sort_keys=True)+"\n")
            for arm, policy in policies.items():
                destination = output/"models"/arm/(niche+".json")
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_text(json.dumps(policy.serialized(),sort_keys=True)+"\n")
                if base.sha_file(destination)==base.sha_file(source_ckpt):
                    raise AssertionError("Optimizer produced unchanged checkpoint")
                report["workers"].append({
                    "lease_id":lease(seed,niche,arm), "state":"completed",
                    "model":"population-mlp-24", "niche":niche, "arm":arm,
                    "original_checkpoint_sha256":base.sha_file(source_ckpt),
                    "new_checkpoint_sha256":base.sha_file(destination),
                    "actual_adam_updates":policy.updates -
                        shift.load_policy(pop,source_ckpt).updates,
                    "teacher_goal_exposures":sum(r["terminal_examples"]
                                                  for r in train_exposure),
                    "teacher_nonterminal_exposures":sum(r["nonterminal_examples"]
                                                      for r in train_exposure)})
            for scenario in SCENARIOS:
                h = heldouts_by_scenario[scenario]
                test = heldout_group(pop,policies,niche,scenario,h,output,began)
                autos = [world_seed(pop,"AUTONOMOUS",seed,niche,scenario,i)
                         for i in range(AUTO_PER_GROUP)]
                auto = autonomy(pop,policies,niche,scenario,autos,output,began)
                report["groups"].append({"niche":niche,"scenario":scenario,
                    "fresh_heldout_reached_goal_scores":test,
                    "autonomous_world_seeds":autos, "autonomous":auto,
                    "goal_stop_top1":{arm:sum(r["at_reached_goal"][arm]["stop_top1"]
                                           for r in test) for arm in ARMS},
                    "initial_stop_top1":{arm:sum(r["initial"][arm]["stop_top1"]
                                             for r in test) for arm in ARMS},
                    "mean_goal_stop_margin":{arm:round(
                        sum(r["at_reached_goal"][arm]["stop_margin"]
                            for r in test)/len(test),6) for arm in ARMS}})
        report["custody_sha256"] = hashlib.sha256(
            json.dumps(sorted(used)).encode()).hexdigest()
        goal_counts = {a:sum(g["goal_stop_top1"][a] for g in report["groups"])
                       for a in ARMS}
        initial_counts = {a:sum(g["initial_stop_top1"][a] for g in report["groups"])
                          for a in ARMS}
        rewards = [(g["autonomous"][ARMS[1]]["reward_per_decision"] -
                    g["autonomous"][ARMS[0]]["reward_per_decision"])
                   for g in report["groups"]]
        report["aggregate"] = {
            "trained_checkpoints":len(report["workers"]),
            "actual_adam_updates":sum(w["actual_adam_updates"] for w in report["workers"]),
            "fresh_reached_goal_states":len(report["groups"])*EVAL_PER_GROUP,
            "goal_stop_top1_by_arm":goal_counts,
            "premature_stop_top1_at_world_start_by_arm":initial_counts,
            "reward_improved_groups":sum(x>0 for x in rewards),
            "reward_regressed_groups":sum(x<0 for x in rewards),
            "mean_autonomous_reward_delta":round(sum(rewards)/len(rewards),6),
            "autonomous_episodes":len(report["groups"])*len(ARMS)*AUTO_PER_GROUP,
            "original_autonomous_gallery_svgs":len(list((output/"gallery").rglob("*.svg"))),
            "elapsed_s":round(time.monotonic()-began,5),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        report["status"]="completed"
    except BaseException as exc:
        report["status"]="failed"
        report["failures"].append(type(exc).__name__+": "+str(exc))
        raise
    finally:
        (output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        (output/"SHA256SUMS").write_text("".join(
            base.sha_file(p)+"  "+str(p.relative_to(output))+"\n"
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name!="SHA256SUMS"))
    return report


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--request",type=Path,required=True)
    ap.add_argument("--upstream",type=Path,required=True)
    ap.add_argument("--ft003",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args()
    result=run(a.request,a.upstream,a.ft003,a.output,a.seed)
    print(json.dumps({"status":result["status"],"seed":a.seed,
                      "aggregate":result["aggregate"]},sort_keys=True))
