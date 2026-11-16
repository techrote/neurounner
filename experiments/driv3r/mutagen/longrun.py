#!/usr/bin/env python3
"""MUTAGEN M1: longer, matched, REAL d1 /v1/systemone worm-steering episodes.

Each experimental arm runs against identical W0 world seeds, observation/choice
contracts and source-replayed legal actions. The original choices and probability
tables are preserved. PNGs are Python W0 source-faithful, NOT Godot/Workbench.
No training, native PAINT, model-parameter inference shortcut or safety edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from experiments.driv3r.backend import PINS, sha256_file
from experiments.driv3r.mutagen import infer

DEFAULT_SEEDS = (570, 2111)
W0_SHA = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
DONOR_SHA = "073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c"
BUDGETS = {"stock":128, "mlp_scoop":128, "attn_gouge":128,
           "conv_scrape_10":64, "conv_scrape_25":64}
FAMILIES = ("greebles", "pipework", "architecture")


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=True).encode()
    ).hexdigest()


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1) * p
    lo = int(position)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def actions_summary(records, events, budget):
    choices = [r["original_selected_action"] for r in records if r.get("status") == "ok"]
    duration = [r["latency_seconds"] for r in records if r.get("status") == "ok"]
    counts = Counter(choices)
    maximum_streak = 0
    last = None
    streak = 0
    for choice in choices:
        streak = streak+1 if choice == last else 1
        maximum_streak = max(maximum_streak, streak)
        last = choice
    return {
        "original_inference_calls": len(records),
        "valid_typed_decisions": len(choices),
        "decision_budget": budget,
        "action_histogram": {key:counts.get(key,0) for key in infer.ACTION_DESCRIPTIONS},
        "max_identical_action_streak": maximum_streak,
        "model_latency_p50_seconds": percentile(duration,.5),
        "model_latency_p95_seconds": percentile(duration,.95),
        "mean_model_latency_seconds": statistics.mean(duration) if duration else None,
        "stop_selected": "STOP" in choices,
        "raw_choices": choices,
        "host_applied_choices":[e["applied"] for e in events],
    }


def replay_partial(worm, seed, family, receipts):
    """Store a true partial W0 source replay, never as a completed model episode."""
    choices = [r["original_selected_action"] for r in receipts
               if r.get("status")=="ok" and r.get("original_selected_action") in infer.ACTION_DESCRIPTIONS]
    if not choices:
        return None
    return worm.run(seed, mode="replay", family=family,
                    steps=len(choices), raw_actions=choices)


def main(args):
    if args.arm not in BUDGETS or args.steps != BUDGETS[args.arm]:
        raise ValueError("Unknown or changed M1 intervention/budget")
    if tuple(args.seeds) != DEFAULT_SEEDS:
        raise ValueError("Different world-seed cohort is not the committed experiment")
    if args.output.exists():
        raise ValueError("Refusing to overwrite source-custody evidence")
    args.output.mkdir(parents=True)
    run_start=time.monotonic()
    runtime=json.loads(args.runtime_manifest.read_text())
    binary=Path(runtime["binary"])
    if sha256_file(binary)!=PINS["runtime"]["binary_sha256"]:
        raise ValueError("llama.cpp binary is not the frozen qualified System One runtime")
    if sha256_file(args.donor)!=DONOR_SHA or args.donor.stat().st_size!=PINS["model"]["bytes"]:
        raise ValueError("Original stock model differs from previous native/Q8 source pin")
    if args.arm == "stock":
        if sha256_file(args.model)!=DONOR_SHA:
            raise ValueError("Stock arm must use original donor bytes")
    elif sha256_file(args.model)==DONOR_SHA:
        raise ValueError("Mutant file was not changed")
    worm=infer.load_worm(args.w0)
    if tuple(worm.FAMILIES)!=FAMILIES:
        raise ValueError("Pinned W0 theme contracts changed")
    identity={
        "schema":"driv3r-d1-mutagen-longrun/1",
        "experiment":"mutagen-m1-two-worlds-graded-conv",
        "arm":args.arm,
        "donor_sha256":DONOR_SHA,
        "model_sha256":sha256_file(args.model),
        "source_sha":os.environ.get("GITHUB_SHA"),
        "w0_commit":W0_SHA,
        "seeds":list(DEFAULT_SEEDS),
        "families":list(FAMILIES),
        "budget_per_episode":args.steps,
        "runtime_binary_sha256":PINS["runtime"]["binary_sha256"],
        "real_model_inference":True,
        "native_workbench":False,
        "training":False,
        "original_artifact_preserved":True
    }
    infer.json_write(args.output/"identity.json",identity)
    episode_root=args.output/"episodes"
    receipt_root=args.output/"raw-choices"
    episode_root.mkdir()
    receipt_root.mkdir()
    records=[]
    error_count=0
    with infer.D1Server(binary,args.model,args.output/"model-process") as server:
        for seed in DEFAULT_SEEDS:
            server.world_seed=seed
            for family in FAMILIES:
                t0=time.monotonic()
                choices=[]
                filename=f"{args.arm}-{family}-seed{seed}"
                initial_observation = worm.run(seed, "replay", family,
                                               steps=1, raw_actions=["STOP"])["events"][0]["observation"]
                note={
                    "arm":args.arm, "seed":seed, "family":family,
                    "initial_observation_sha256":canonical_sha(initial_observation),
                    "initial_legal_actions":initial_observation["legal"],
                    "budget":args.steps, "original_full_model_inference":True,
                    "done":False,
                }
                try:
                    record=worm.run(seed,mode="callback",family=family,steps=args.steps,
                                    selector=lambda obs:server.choose(obs,choices))
                    record["mode"]="d1-m1-"+args.arm
                    record["engine"]="W0-source-faithful-real-d1-System-One-not-native"
                    worm.write_result(record,episode_root)
                    note["status"]="COMPLETE" # includes legitimate early STOP, not necessarily full budget
                    note["done"]=True
                    note["steps_executed"]=record["steps_executed"]
                    note["trajectory_metrics"]=record["metrics"]
                    note["png_sha256"]=sha256_file(
                         episode_root/f"{family}-d1-m1-{args.arm}-{seed}.png")
                    note["event_sha256"]=canonical_sha(record["events"])
                    note["first_choice"]=choices[0].get("original_selected_action") if choices else None
                    events=record["events"]
                except Exception as exc:
                    error_count+=1
                    note["status"]="FAILED_OR_PARTIAL"
                    note["failure"]={"type":type(exc).__name__,"error":str(exc),
                                      "traceback":traceback.format_exc()[-4500:]}
                    events=[]
                    try:
                        partial=replay_partial(worm,seed,family,choices)
                        if partial is not None:
                            partial["mode"]="d1-m1-PARTIAL-"+args.arm
                            partial["engine"]="replayed-partial-actual-model-choices-not-native"
                            worm.write_result(partial,episode_root)
                            events=partial["events"]
                            note["partial_replayed_choices"]=len(events)
                    except Exception as again:
                        note["partial_replay_error"]=str(again)
                finally:
                    infer.json_write(receipt_root/(filename+".json"),choices)
                    note["decisions"]=actions_summary(choices,events,args.steps)
                    note["wall_seconds"]=time.monotonic()-t0
                    records.append(note)
                    infer.json_write(args.output/"episode-progress.json",records)
                    print(json.dumps({
                        "event":"d1-longrun-episode",
                        "arm":args.arm,"seed":seed,"family":family,
                        "status":note["status"],
                        "decisions":note["decisions"]["valid_typed_decisions"],
                        "stop":note["decisions"]["stop_selected"],
                        "unique_cells":note.get("trajectory_metrics",{}).get("unique_trail_cells"),
                        "seconds":note["wall_seconds"]
                    }),flush=True)
    summaries=[x["decisions"] for x in records]
    latencies=[r["latency_seconds"]
               for p in sorted(receipt_root.glob("*.json"))
               for r in json.loads(p.read_text()) if r.get("status")=="ok"]
    result={
        **identity,
        "status":"COMPLETE" if error_count==0 and len(records)==len(DEFAULT_SEEDS)*len(FAMILIES)
                 else "PARTIAL_OR_FAILED",
        "failure_episodes":error_count,
        "episodes":records,
        "completed_episodes":sum(x["done"] for x in records),
        "total_raw_valid_decisions":sum(x["valid_typed_decisions"] for x in summaries),
        "overall_action_histogram":{
            action:sum(x["action_histogram"][action] for x in summaries)
            for action in infer.ACTION_DESCRIPTIONS
        },
        "p50_model_seconds":percentile(latencies,.5),
        "p95_model_seconds":percentile(latencies,.95),
        "total_wall_seconds":time.monotonic()-run_start,
        "state_identity":"same original W0 initial seed/family/legal grid per arm; trajectories diverge after choices",
        "visual_identity":"original 384x384 W0 PNGs not Godot native material"
    }
    infer.json_write(args.output/"summary.json",result)
    print(json.dumps({
        "event":"d1-longrun-finished","arm":args.arm,"status":result["status"],
        "episodes":result["completed_episodes"],"decisions":result["total_raw_valid_decisions"],
        "p50_s":result["p50_model_seconds"],"p95_s":result["p95_model_seconds"]
    }),flush=True)
    if error_count:
        raise RuntimeError(f"{error_count} genuine model/W0 episodes failed; original results preserved")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arm",required=True)
    p.add_argument("--steps",type=int,required=True)
    p.add_argument("--seeds",type=int,nargs=2,default=list(DEFAULT_SEEDS))
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--donor",type=Path,required=True)
    p.add_argument("--runtime-manifest",type=Path,required=True)
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
