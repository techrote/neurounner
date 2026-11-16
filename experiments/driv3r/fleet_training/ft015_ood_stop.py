"""FT-015: independent out-of-distribution audit of immutable FT-014 STOP gates.

CPU-only external synthetic host. No optimizer, native game changes, W0 or SLM claim.
Model artifacts are immutable originals from FT-003 and FT-014, never Git binaries.
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
import ft014_stop_sidecar as ft

ID = "FT-015-20261116-switch-guard-long120"
SCHEMA = "cybersand.fleet-training.ft015-request/1"
HEAD_014 = "2272919b37cc976a2bbdc070ce75bbcec039414e"
RUN_014 = 43884145188
ARTIFACTS = {
    "9401/art": 14232543956, "9401/navigation": 14233139381,
    "9403/art": 14233259126, "9403/navigation": 14232149598,
}
SCENARIOS = ("long120", "dense120", "guard120", "shifted_guard60")
ARMS = ("frozen_baseline", "legacy_feature_gate",
        "distance_feature_gate", "exact_goal_rule")
SEEDS = ft.SEEDS
COHORTS = ft.COHORTS
WORLDS = 8
MAX_SECONDS = 360
MAX_RSS_KIB = 900000


def checked_request(r):
    want = {
        "schema": SCHEMA, "request_id": ID, "ft014_source_head": HEAD_014,
        "ft014_actions_run": RUN_014, "ft014_artifact_ids": ARTIFACTS,
        "source_commit": shift.SOURCE_COMMIT, "source_blob": shift.SOURCE_BLOB,
        "ft003_artifact_id": 14194214028, "ft003_zip_sha256": shift.FT003_SHA,
        "seeds": list(SEEDS), "cohorts": {k:list(v) for k,v in COHORTS.items()},
        "scenarios": list(SCENARIOS), "arms": list(ARMS),
        "autonomous_worlds_per_scenario": WORLDS,
        "optimizer_updates": 0, "max_seconds_per_job": MAX_SECONDS,
        "max_rss_kib_per_job": MAX_RSS_KIB, "max_parallel_runner_jobs": 4,
        "native_material_edits": 0, "pretrained_slm_optimizer_steps": 0
    }
    if type(r) is not dict or r != want:
        raise ValueError("FT-015 source, artifact, OOD split or budget drift")
    return r


def budget(start):
    if time.monotonic() - start > MAX_SECONDS:
        raise TimeoutError("FT-015 exceeded bounded job wall")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_KIB:
        raise MemoryError("FT-015 exceeded memory limit")


def job_lease(seed, cohort):
    if seed not in SEEDS or cohort not in COHORTS:
        raise ValueError("Unknown seed/cohort lease")
    return f"{ID}/{seed}/{cohort}"


def world_seed(pop, seed, niche, scenario, index):
    if seed not in SEEDS or niche not in base.NICHES or scenario not in SCENARIOS or not 0 <= index < WORLDS:
        raise ValueError("Unknown FT-015 independent world lease")
    sid = pop.stable_seed(ID, "UNSEEN-OOD", seed, niche, scenario, index)
    # FT-014 used a different namespace, not just another suffix on a known seed.
    for split in ft.SPLITS:
        for i in range(max(WORLDS, ft.POOL)):
            if sid == ft.world_seed(pop, split, seed, niche,
                                   "shifted_guard60" if scenario=="shifted_guard60" else "long60", i):
                raise AssertionError("FT-014 corpus seed reused")
    return sid


def make_world(pop, seed, scenario):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown OOD situation")
    if scenario == "shifted_guard60":
        return shift.world_for(pop, seed, scenario)
    source = {"long120":"long60", "dense120":"dense60", "guard120":"shifted_guard60"}[scenario]
    w = shift.world_for(pop, seed, source)
    w.max_steps = 120
    if scenario == "guard120":
        # Out-of-family guarded materials; do not touch the root/goal cells.
        guard = {(x,y) for x in (5,6,14,15) for y in range(5,17)}
        guard -= {w.cursor, w.goal, w.second_goal}
        w.protected |= guard
        w.obstacles -= guard
        if w.painted & (w.protected | w.obstacles):
            raise AssertionError("OOD guard overlays authored cells")
    return w


def read_gate(path, geometry, expected_hash):
    if base.sha_file(path) != expected_hash:
        raise ValueError("FT-014 trained checkpoint digest mismatch")
    obj = json.loads(path.read_text())
    want_feature = "manhattan_goal_distance_div_3" if geometry else None
    length = 21 if geometry else 20
    if (obj.get("format") != "cybersand-research-stop-sidecar/1" or
        obj.get("added_feature") != want_feature or
        obj.get("adam_updates") != ft.STEPS or
        not isinstance(obj.get("threshold"), (int,float)) or
        not 0 <= obj["threshold"] <= 1 or
        len(obj.get("w",[])) != length):
        raise ValueError("Malformed or untrained FT-014 STOP sidecar")
    w = np.asarray(obj["w"], dtype=np.float64)
    if not np.isfinite(w).all():
        raise ValueError("Nonfinite sidecar model")
    gate = ft.StopGate(geometry)
    gate.w = w
    gate.steps = obj["adam_updates"]
    gate.threshold = float(obj["threshold"])
    return gate


def archive_audit(root, seed, cohort):
    if (root/"exact-head.txt").read_text().strip() != HEAD_014:
        raise ValueError("FT-014 artifact not from exact qualified source HEAD")
    home = root/"run"
    sums = {}
    for line in (home/"SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ",1)
        file = Path(rel)
        if file.is_absolute() or ".." in file.parts or rel in sums:
            raise ValueError("Unsafe/duplicate FT-014 archive member")
        sums[rel] = digest
        if not (home/file).is_file() or base.sha_file(home/file) != digest:
            raise ValueError("FT-014 artifact SHA mismatch: "+rel)
    report = json.loads((home/"report.json").read_text())
    if (report.get("status") != "completed" or report.get("seed") != seed
            or report.get("cohort") != cohort or
            report.get("executor_sha256") != base.sha_file(Path(ft.__file__)) or
            report.get("request_sha256") != base.sha_file(Path(ft.__file__).parent/"request_ft014.json") or
            report.get("source_commit") != shift.SOURCE_COMMIT or
            report.get("source_blob") != shift.SOURCE_BLOB or
            report.get("ft003_original_zip_sha256") != shift.FT003_SHA):
        raise ValueError("FT-014 report provenance/custody mismatch")
    entries = report.get("workers", [])
    if len(entries) != 2*len(COHORTS[cohort]):
        raise ValueError("FT-014 missing trained sidecar records")
    models = {}
    for worker in entries:
        niche, arm = worker["niche"], worker["arm"]
        if niche not in COHORTS[cohort] or arm not in ft.GATE_ARMS:
            raise ValueError("Foreign sidecar in pinned cohort")
        key = (niche,arm)
        if key in models or worker.get("lease_id") != ft.lease(seed,niche,arm):
            raise ValueError("Duplicate/invalid model lease")
        path = home/"models"/niche/(arm+".json")
        if worker.get("trained_sidecar_sha256") != sums.get(str(path.relative_to(home))):
            raise ValueError("Checkpoint omitted from FT-014 signed ledger")
        models[key] = read_gate(path, arm=="distance_feature_gate",
                                worker["trained_sidecar_sha256"])
    if len(models) != len(entries):
        raise AssertionError("Nonunique sidecar models")
    return models, {"members":len(sums), "report_sha256":sums["report.json"],
                    "model_sha256":{f"{n}/{a}":base.sha_file(home/"models"/n/(a+".json"))
                                    for n,a in models}}


def choose(pop, policy, gates, arm, w, niche):
    menu = w.legal()
    X = np.stack([w.feature(a) for a in menu])
    scores = policy.logits(X)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite frozen-model logits")
    if arm == "frozen_baseline" or len(menu)==1:
        idx = int(np.argmax(scores))
    else:
        if arm == "exact_goal_rule":
            stop = ft.distance(w)==0
        else:
            gate = gates[arm]
            stop = gate.probability(ft.features(w,gate.geometry)) >= gate.threshold
        if stop:
            idx = menu.index("STOP")
        else:
            choices = [i for i,a in enumerate(menu) if a!="STOP"]
            idx = max(choices,key=lambda i:float(scores[i])) if choices else menu.index("STOP")
    return menu[idx], X[idx], menu


def run_episode(pop, policy, gates, arm, niche, sid, scenario, full_trace=False):
    w = make_world(pop,sid,scenario)
    total=0.; retraces=0; traces=[]; switched_steps=0
    stop_at_goal=0; stop_non_goal=0; reached=0; switch_stop=0
    while not w.stopped and w.steps < w.max_steps:
        action, feature, menu = choose(pop, policy, gates, arm, w, niche)
        if ft.distance(w)==0:
            reached+=1
            stop_at_goal += int(action=="STOP")
        else:
            stop_non_goal += int(action=="STOP")
        if w.switched:
            switched_steps += 1
            switch_stop += int(action=="STOP")
        before = shift.state_digest(w)
        total += pop.utility(niche, feature, w)
        w.step(action)
        retraces += int(action=="RETRACE")
        if w.painted & (w.protected | w.obstacles):
            raise AssertionError("Illegal protected/obstacle painting")
        if full_trace:
            traces.append({"turn":w.steps, "menu":menu, "choice":action,
                           "before":before, "after":shift.state_digest(w)})
    result = {"seed":sid,"scenario":scenario,"niche":niche,"arm":arm,
              "decisions":w.steps,"reward_per_decision":round(total/max(1,w.steps),6),
              "new_ink":len(w.painted)-1,"retraces":retraces,
              "goal_visits":reached,"stop_at_goal":stop_at_goal,
              "non_goal_stop":stop_non_goal,"switch_steps":switched_steps,
              "stop_after_switch":switch_stop,
              "switch_occurred":bool(w.switched),
              "early_stop":bool(w.stopped and w.steps<w.max_steps)}
    if full_trace:
        result["receipts"]=traces
    return result,w


def audit_trace(pop, sid, scenario, receipts, final):
    w = make_world(pop,sid,scenario)
    for i, r in enumerate(receipts):
        if r["turn"]!=i+1 or r["menu"]!=w.legal() or r["choice"] not in w.legal():
            raise AssertionError("Archived receipt rejected by legal action menu")
        if r["before"]!=shift.state_digest(w):
            raise AssertionError("Archived pre-action hash mismatch")
        w.step(r["choice"])
        if r["after"]!=shift.state_digest(w):
            raise AssertionError("Archived post-action hash mismatch")
    if shift.state_digest(w)!=shift.state_digest(final):
        raise AssertionError("Exact independent state replay failed")


def score_reached_goal(pop, gates, seed, niche, scenario):
    """Independent legal BFS teacher, never claim these are model-selected routes."""
    positives=negatives=0
    observations={a:{"tp":0,"fp":0} for a in ("legacy_feature_gate",
                   "distance_feature_gate","exact_goal_rule")}
    for i in range(WORLDS):
        sid=world_seed(pop,seed,niche,scenario,i)
        root=make_world(pop,sid,scenario)
        route=goals.legal_route(pop,root)
        if route is None or not route:
            continue
        for action in route:
            if action not in root.legal():
                raise AssertionError("Scaffold chose illegal transition")
            # Genuine pre-goal negatives on the same legal path.
            for arm in observations:
                if arm=="exact_goal_rule":
                    stop=ft.distance(root)==0
                else:
                    gate=gates[arm]
                    stop=gate.probability(ft.features(root,gate.geometry))>=gate.threshold
                observations[arm]["fp"]+=int(stop)
            negatives+=1
            root.step(action)
        if ft.distance(root)!=0 or root.switched or "STOP" not in root.legal():
            raise AssertionError("Goal scaffold failed to arrive before switch")
        positives+=1
        for arm in observations:
            if arm=="exact_goal_rule":
                stop=ft.distance(root)==0
            else:
                gate=gates[arm]
                stop=gate.probability(ft.features(root,gate.geometry))>=gate.threshold
            observations[arm]["tp"]+=int(stop)
    return {"positive":positives,"negative":negatives,"arms":observations}


def run(request, upstream, ft003, ft014_archive, out, seed, cohort):
    checked_request(json.loads(request.read_text()))
    job_lease(seed,cohort)
    pop,source=shift.load_source(upstream)
    if len(pop.FEATURES)!=20 or "stop" not in pop.FEATURES:
        raise AssertionError("Synthetic host observation/action contract drift")
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report={"schema":"cybersand.fleet-training.ft015-evidence/1",
            "request_id":ID,"lease":job_lease(seed,cohort),"seed":seed,
            "cohort":cohort,"status":"starting", "classification":
            "FROZEN_ORIGINAL_SIDECAR_CPU_SYNTHETIC_OOD_NOT_NATIVE_NOT_W0_NOT_SLM",
            "ft014_artifact_id":ARTIFACTS[f"{seed}/{cohort}"],
            "ft014_source_head":HEAD_014, "ft003_zip_sha256":shift.FT003_SHA,
            "executor_sha256":base.sha_file(Path(__file__)),
            "request_sha256":base.sha_file(request),
            "original_source_sha256":base.sha_file(source),
            "optimizer_updates":0,"native_material_edits":0,
            "resident_inference_processes_per_runner":1,
            "models":[], "groups":[], "failures":[]}
    try:
        models, custody=archive_audit(ft014_archive,seed,cohort)
        report["ft014_custody"]=custody
        report["ft003_custody"]=shift.archive_check(
            ft003,source,request.parent/"request_dose.json")
        report["status"]="running"
        seen=set()
        for niche in COHORTS[cohort]:
            backbone=ft003/f"dose_48/{seed}/models/{niche}.json"
            frozen=shift.load_policy(pop,backbone)
            recorded=next(w["backbone_checkpoint_sha256"] for w in
                          json.loads((ft014_archive/"run"/"report.json").read_text())["workers"]
                          if w["niche"]==niche)
            if base.sha_file(backbone)!=recorded:
                raise ValueError("Frozen FT-003 backbone mismatch against FT-014")
            report["models"].append({"niche":niche,
                    "backbone_sha256":recorded,
                    "sidecars":{a:custody["model_sha256"][f"{niche}/{a}"]
                                for a in ft.GATE_ARMS}})
            gates={a:models[(niche,a)] for a in ft.GATE_ARMS}
            for scenario in SCENARIOS:
                cases={a:[] for a in ARMS}
                for i in range(WORLDS):
                    sid=world_seed(pop,seed,niche,scenario,i)
                    if sid in seen: raise AssertionError("Duplicate world dispatch")
                    seen.add(sid)
                    for arm in ARMS:
                        episode,w=run_episode(pop,frozen,gates,arm,niche,sid,scenario,i==0)
                        if i==0:
                            audit_trace(pop,sid,scenario,episode["receipts"],w)
                            folder=out/"gallery"/niche/scenario
                            folder.mkdir(parents=True,exist_ok=True)
                            (folder/(arm+".svg")).write_text(pop.draw_svg(w))
                            (folder/(arm+".json")).write_text(json.dumps(
                                episode,sort_keys=True)+"\n")
                        else:
                            episode["receipts"]=None
                        cases[arm].append(episode)
                    budget(start)
                summary={arm:{
                    "episodes":len(v),
                    "mean_reward_per_decision":round(sum(e["reward_per_decision"] for e in v)/len(v),6),
                    "mean_decisions":round(sum(e["decisions"] for e in v)/len(v),4),
                    "mean_ink":round(sum(e["new_ink"] for e in v)/len(v),4),
                    "mean_retraces":round(sum(e["retraces"] for e in v)/len(v),4),
                    "goal_stops":sum(e["stop_at_goal"] for e in v),
                    "nongoal_stops":sum(e["non_goal_stop"] for e in v),
                    "switch_episodes":sum(e["switch_occurred"] for e in v),
                    "post_switch_decisions":sum(e["switch_steps"] for e in v),
                    "post_switch_stops":sum(e["stop_after_switch"] for e in v)
                } for arm,v in cases.items()}
                legal=score_reached_goal(pop,gates,seed,niche,scenario)
                report["groups"].append({"niche":niche,"scenario":scenario,
                      "autonomous":summary,"legal_goal_probe":legal})
                budget(start)
        if len(seen)!=len(COHORTS[cohort])*len(SCENARIOS)*WORLDS:
            raise AssertionError("Some independent worlds were never dispatched")
        report["aggregate"]={
            "unique_ood_worlds":len(seen),
            "autonomous_episodes":len(seen)*len(ARMS),
            "legal_goal_positive":sum(g["legal_goal_probe"]["positive"] for g in report["groups"]),
            "legal_goal_negative":sum(g["legal_goal_probe"]["negative"] for g in report["groups"]),
            "elapsed_seconds":round(time.monotonic()-start,5),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        report["world_lease_sha256"]=hashlib.sha256(
            json.dumps(sorted(seen)).encode()).hexdigest()
        report["status"]="completed"
    except BaseException as err:
        report["status"]="failed"
        report["failures"].append(f"{type(err).__name__}: {err}")
        raise
    finally:
        (out/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        (out/"SHA256SUMS").write_text("".join(
            base.sha_file(file)+"  "+str(file.relative_to(out))+"\n"
            for file in sorted(out.rglob("*"))
            if file.is_file() and file.name!="SHA256SUMS"))
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--request",type=Path,required=True)
    ap.add_argument("--upstream",type=Path,required=True)
    ap.add_argument("--ft003",type=Path,required=True)
    ap.add_argument("--ft014",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--cohort",choices=COHORTS,required=True)
    opts=ap.parse_args()
    result=run(opts.request,opts.upstream,opts.ft003,opts.ft014,
               opts.output,opts.seed,opts.cohort)
    print(json.dumps({"status":result["status"],"aggregate":result["aggregate"],
                      "lease":result["lease"]},sort_keys=True))
