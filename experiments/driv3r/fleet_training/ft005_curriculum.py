"""FT-005: real bounded Adam continuation from FT-003 frozen tiny policies.

Two independent CI seed jobs; each trains six synthetic game policies on longer,
denser, and shifted-protection worlds. Matched frozen vs continued tests use a
NEW disjoint evaluation seed domain. These are NOT SLMs, native PAINT, or W0.
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
import ft004_shift as transfer

SCHEMA = "cybersand.fleet-training.shift-curriculum-request/1"
ID = "FT-005-20261116-long-shift-continuation"
TRAIN_SCENARIOS = ("long60", "dense60", "shifted_guard60")
EVAL_SCENARIOS = ("fresh30", "long60", "dense60", "shifted_guard60")
SEEDS = (9401, 9403)
TRAIN_WORLDS_PER_SCENARIO = 8
TEST_WORLDS = 8
CAP_SECONDS = 360
CAP_RSS_KIB = 900000
TARGET_TEMPERATURE = .42

def checked_request(r):
    expected = {
        "schema": SCHEMA, "request_id": ID,
        "population_source_commit": transfer.SOURCE_COMMIT,
        "population_source_blob": transfer.SOURCE_BLOB,
        "ft003_run_id": 43797061143, "ft003_artifact_id": 14194214028,
        "ft003_zip_sha256": transfer.FT003_SHA,
        "train_seeds": list(SEEDS),
        "frozen_dose": 48,
        "training_scenarios": list(TRAIN_SCENARIOS),
        "test_scenarios": list(EVAL_SCENARIOS),
        "training_worlds_per_scenario": TRAIN_WORLDS_PER_SCENARIO,
        "test_worlds_per_scenario": TEST_WORLDS,
        "max_seconds_per_seed": CAP_SECONDS,
        "max_rss_kib_per_seed": CAP_RSS_KIB,
        "max_parallel_runners": 2,
        "native_material_edits": 0,
        "pretrained_slm_optimizer_steps": 0
    }
    if type(r) is not dict or r != expected:
        raise ValueError("FT-005 source, custody, request or resource bounds differ")
    return r

def lease_id(seed, niche):
    if seed not in SEEDS or niche not in base.NICHES:
        raise ValueError("Unknown seed or niche")
    return f"{ID}/{seed}/{niche}"

def domain_seed(pop, domain, training_seed, niche, scenario, index):
    if domain not in ("TRAIN","TEST") or training_seed not in SEEDS or niche not in base.NICHES:
        raise ValueError("Invalid world-seed domain")
    max_i = TRAIN_WORLDS_PER_SCENARIO if domain == "TRAIN" else TEST_WORLDS
    valid = TRAIN_SCENARIOS if domain == "TRAIN" else EVAL_SCENARIOS
    if scenario not in valid or type(index) is not int or not 0 <= index < max_i:
        raise ValueError("Unbounded world seed")
    seed = pop.stable_seed(ID, domain, training_seed, niche, scenario, index)
    upstream = {pop.stable_seed(training_seed, split, j)
                for split, count in (("train",48),("selection",4),("test",12))
                for j in range(count)}
    original_ood = {transfer.heldout_world_seed(pop, training_seed, j)
                    for j in range(transfer.WORLDS)}
    if seed in upstream or seed in original_ood:
        raise ValueError("New source seed collides with prior custody")
    return seed

def check_seed_custody(pop):
    seen = set()
    for source_seed in SEEDS:
        for niche in base.NICHES:
            for domain, scenarios, n in (
                ("TRAIN",TRAIN_SCENARIOS,TRAIN_WORLDS_PER_SCENARIO),
                ("TEST",EVAL_SCENARIOS,TEST_WORLDS)):
                for scenario in scenarios:
                    for i in range(n):
                        v = domain_seed(pop,domain,source_seed,niche,scenario,i)
                        if v in seen:
                            raise AssertionError("Train/test/peer world seed overlap")
                        seen.add(v)
    return hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest()

def train_continuation(pop, policy, niche, seed, began):
    receipts = []
    optimizer_before = policy.updates
    for scenario in TRAIN_SCENARIOS:
        for i in range(TRAIN_WORLDS_PER_SCENARIO):
            wseed = domain_seed(pop,"TRAIN",seed,niche,scenario,i)
            w = transfer.world_for(pop,wseed,scenario)
            rng = np.random.default_rng(pop.stable_seed(ID,"optimizer",seed,niche,scenario,i))
            start = policy.updates
            while not w.stopped and w.steps < w.max_steps:
                menu = w.legal()
                X = np.stack([w.feature(a) for a in menu])
                target = pop.softmax(np.array(
                    [pop.utility(niche,x,w) for x in X]) / TARGET_TEMPERATURE)
                policy.teach(X,target)
                choice = (int(rng.choice(len(menu),p=target)) if rng.random() < .25
                          else policy.choice(X,rng))
                action = menu[choice]
                w.step(action)
                if w.painted & (w.protected | w.obstacles):
                    raise AssertionError("Training wrote protected/obstacle material")
                if time.monotonic() - began > CAP_SECONDS:
                    raise TimeoutError("FT-005 per-seed optimizer wall budget")
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > CAP_RSS_KIB:
                    raise MemoryError("FT-005 per-seed worker RSS cap")
            receipts.append({"scenario":scenario,"train_world_seed":wseed,
                             "decisions":w.steps,
                             "optimizer_updates":policy.updates-start,
                             "synthetic_new_ink":len(w.painted)-1,
                             "early_stop":bool(w.stopped and w.steps<w.max_steps)})
    if policy.updates <= optimizer_before:
        raise AssertionError("No genuine Adam parameter updates")
    return receipts

def matched_evaluate(pop, baseline, adapted, niche, seed, output):
    rows=[]
    for scenario in EVAL_SCENARIOS:
        world_seeds=[domain_seed(pop,"TEST",seed,niche,scenario,i) for i in range(TEST_WORLDS)]
        arms={}
        for arm, policy in (("frozen48",baseline),("shift_continued",adapted)):
            episodes=[]
            for i,wseed in enumerate(world_seeds):
                item,world=transfer.execute(pop,policy,niche,wseed,scenario,save_trace=(i==0))
                if item["model_forwards"] != item["decisions"] or item["protected_violations"]:
                    raise AssertionError("Model forward count or host material violation")
                if i==0:
                    transfer.audit_receipts(pop,item["receipts"],niche,wseed,scenario)
                    dst=output/"gallery"/niche/scenario
                    dst.mkdir(parents=True,exist_ok=True)
                    (dst/f"{arm}.svg").write_text(pop.draw_svg(world))
                    (dst/f"{arm}.json").write_text(json.dumps(item,sort_keys=True)+"\n")
                episodes.append(item)
            arms[arm]=transfer.summarize(episodes)
        rows.append({"niche":niche,"scenario":scenario,"test_world_seeds":world_seeds,
                     "frozen48":arms["frozen48"],"shift_continued":arms["shift_continued"],
                     "reward_per_decision_delta":round(
                         arms["shift_continued"]["reward_per_decision"]-
                         arms["frozen48"]["reward_per_decision"],5),
                     "new_ink_delta":round(arms["shift_continued"]["new_ink"]-
                         arms["frozen48"]["new_ink"],5)})
    return rows

def run(request, upstream, ft003, output, seed):
    checked_request(json.loads(request.read_text()))
    if seed not in SEEDS:
        raise ValueError("Unapproved seed lease")
    pop, original_source=transfer.load_source(upstream)
    source_request=Path(__file__).parent/"request_dose.json"
    if not source_request.is_file():
        raise FileNotFoundError("Source pinned dose request missing")
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    report={
        "schema":"cybersand.fleet-training.shift-curriculum-evidence/1",
        "request_id":ID,"training_seed":seed,
        "producer_source_commit":transfer.SOURCE_COMMIT,
        "producer_source_blob":transfer.SOURCE_BLOB,
        "producer_source_sha256":base.sha_file(original_source),
        "executor_sha256":base.sha_file(Path(__file__)),
        "request_sha256":base.sha_file(request),
        "ft003_archive_sha256":transfer.FT003_SHA,
        "classification":"REAL_ADAM_CONTINUATION_SYNTHETIC_NOT_SLM_NATIVE_OR_W0",
        "native_edits":0,"pretrained_slm_optimizer_steps":0,
        "concurrent_residents_per_seed_job":1,
        "workers":[],"evaluations":[],"failures":[],"status":"starting"
    }
    try:
        report["original_audit"]=transfer.archive_check(ft003,original_source,source_request)
        report["all_new_seed_domains_sha256"]=check_seed_custody(pop)
        report["status"]="running"
        for niche in base.NICHES:
            source=ft003/f"dose_48/{seed}/models/{niche}.json"
            original_model_sha=base.sha_file(source)
            baseline=transfer.load_policy(pop,source)
            adapted=transfer.load_policy(pop,source)
            lease=lease_id(seed,niche)
            train= train_continuation(pop,adapted,niche,seed,began)
            destination=output/"models"/f"{niche}.json"
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(adapted.serialized(),sort_keys=True)+"\n")
            if original_model_sha==base.sha_file(destination):
                raise AssertionError("Fine-tuned weights did not change")
            report["workers"].append({
                "lease_id":lease,"status":"completed","model":"population-mlp-24",
                "original_checkpoint_sha256":original_model_sha,
                "adapted_checkpoint_sha256":base.sha_file(destination),
                "original_optimizer_steps":json.loads(source.read_text())["optimizer_steps"],
                "new_optimizer_updates":adapted.updates,
                "train_receipts":train,
                "train_synthetic_ink":sum(x["synthetic_new_ink"] for x in train),
                "train_early_stops":sum(x["early_stop"] for x in train)})
            report["evaluations"].extend(
                matched_evaluate(pop,baseline,adapted,niche,seed,output))
        rows=report["evaluations"]
        report["aggregate"]={
            "paired_niche_scenarios":len(rows),
            "wins":sum(x["reward_per_decision_delta"]>0 for x in rows),
            "regressions":sum(x["reward_per_decision_delta"]<0 for x in rows),
            "ties":sum(x["reward_per_decision_delta"]==0 for x in rows),
            "total_new_optimizer_updates":sum(x["new_optimizer_updates"] for x in report["workers"]),
            "mean_reward_per_decision_delta":round(
                sum(x["reward_per_decision_delta"] for x in rows)/len(rows),5),
            "mean_new_ink_delta":round(sum(x["new_ink_delta"] for x in rows)/len(rows),5),
            "synthetic_evaluation_episodes":len(rows)*2*TEST_WORLDS,
            "wall_s":round(time.monotonic()-began,5),
            "max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "original_svg_gallery_count":len(list((output/"gallery").rglob("*.svg"))),
            "real_optimizer":True,"native_godot":False
        }
        if report["aggregate"]["total_new_optimizer_updates"]<=0:
            raise AssertionError("Missing genuine optimizer receipts")
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

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--request",type=Path,required=True)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--ft003",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    a=p.parse_args()
    r=run(a.request,a.upstream,a.ft003,a.output,a.seed)
    print(json.dumps({"status":r["status"],"seed":a.seed,
                      "aggregate":r["aggregate"]},sort_keys=True))

if __name__=="__main__":
    main()
