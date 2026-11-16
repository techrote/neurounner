#!/usr/bin/env python3
"""W11: true 1-versus-2 resident full-LFM checkpoint-only WORM inference benchmark.

One Python *process* per loaded pinned full LFM2.5-VL + W10 LoRA.
The supervisor starts one, waits for it to finish, then starts two *simultaneous*
and independently loaded model processes behind a ready/go barrier.
Every original W0 PNG comes from live model logits -> legal host sampling ->
host-applied W0 actions, NEVER from model-owned world mutation.
No optimizer, no sealed FINAL, no native execution, no claimed 128-tick proof.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from statistics import median

import worm_train_w7_v1 as w7
import worm_w10_scored_crn_v1 as w10score
from worm_w7_sensitivity_eval_v1 import score_next_token
from worm_w8_scored_crn_v1 import scored_policy
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint, load_base

VERSION = "lfm-worm-w13-128-turn-real-four-resident-synthetic-v1"
MODEL = "LiquidAI/LFM2.5-VL-450M"
REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"
W0_SHA = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
SUPPORT_SHA = "cc27b3206cb61f98432fa95b799c9b9ff508c2bc"
PRODUCER_SHA = "1e57a4df8d344b209a14df886dd898211dcc4918"
PRODUCER_RUN = 43754365984
TURNS = 128
GLOBAL_DEADLINE = 1750
STOP = False


def handle_stop(_sig, _frame):
    global STOP
    STOP = True


def percent(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered)-1, int((len(ordered)-1)*p))], 4)


def vm_rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def emit_failure(directory, message):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"failure.json", {
        "status": "failed", "reason": str(message), "schema": VERSION,
        "native_execution": False, "optimizer_updates": 0,
        "sealed_FINAL_opened": False,
    })


def worker(args):
    out = args.output
    if out.exists():
        raise RuntimeError("fresh immutable worker lease required")
    out.mkdir(parents=True)
    start = time.monotonic()
    write_json(out/"lease.json", {
        "schema": VERSION, "status": "starting", "lease_id": args.lease,
        "phase": args.phase, "runner_pid": os.getpid(), "parent_pid": os.getppid(),
        "seed": args.seed, "family": args.family, "turn_budget": TURNS,
        "model": MODEL, "revision": REVISION, "producer_source": PRODUCER_SHA,
        "producer_run": PRODUCER_RUN, "optimizer_updates": 0,
        "host_authority": "pinned-W0", "native_execution": False,
        "sealed_FINAL_opened": False, "128_turn_host_only_synthetic": True,
    })
    worm = w7.worm_source(args.worm_root)
    support = w7.support_source(args.support_root, worm)
    selected = profile("lfm25_vl", require_supported=True)
    if selected["model_id"] != MODEL or selected["revision"] != REVISION:
        raise ValueError("full official model pin changed")
    configure_cpu(1)
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    receipt = w10score.verify_original(args.producer_root, args.model, selected)
    base, processor = load_base(args.model, selected)
    checkpoint = args.producer_root/"checkpoint-000045"/"adapter"
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True,
                                       local_files_only=True)
    adapter = adapter_for(selected)
    adapter.set_training_mode(model)
    contract = adapter.validate_trainable(model, selected)
    if fingerprint(model, False) != receipt["frozen_base_sha256"]:
        raise ValueError("frozen full-VL base fingerprint mismatch")
    stored = load_file(str(checkpoint/"adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model)
    if stored.keys() != loaded.keys() or any(
        not torch.equal(stored[k], loaded[k].detach().cpu()) for k in stored
    ):
        raise ValueError("original W10 rank8 adapter mismatch")
    model.eval()
    adapter_sha = sha256_file(checkpoint/"adapter_model.safetensors")
    if contract["language_lora_modules"] != 92 or contract["vision_training"] or contract["connector_training"]:
        raise ValueError("qualified language-only LoRA / vision freeze changed")
    write_json(out/"model-readback.json", {
        "model_class": "Lfm2VlForConditionalGeneration", "processor": "Lfm2VlProcessor",
        "model_revision": REVISION, "model_download_receipt_sha256": sha256_file(args.model/"download_receipt.json"),
        "adapter_sha256": adapter_sha, "adapter_tensors": len(stored),
        "contract": contract, "original_adamw_steps": 45, "new_optimizer_steps": 0,
    })
    write_json(out/"ready.json", {"pid": os.getpid(), "lease_id": args.lease,
              "ready_monotonic_ns": time.monotonic_ns(), "resident_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "adapter_sha256": adapter_sha, "seed": args.seed})
    while not args.go_file.is_file():
        if STOP or time.monotonic()-start > 650:
            raise TimeoutError("worker never received go barrier")
        time.sleep(0.1)
    if STOP:
        raise RuntimeError("Stop requested")
    times = []
    decisions = []
    raw_probe = []
    def choose(obs):
        if STOP or time.monotonic()-start > 900:
            raise TimeoutError("bounded model decision deadline or Stop")
        tick = len(decisions)
        with torch.inference_mode():
            if tick == 0:
                raw_probe.append(w7.generate(model, processor, selected,
                    w7.make_prompt(obs), obs["legal"], max_tokens=10))
            t = time.monotonic()
            scores = score_next_token(model, processor, selected, w7.make_prompt(obs))
            times.append(time.monotonic()-t)
        choice, evidence = scored_policy(scores, obs["legal"], "learned", args.seed, tick)
        decisions.append({"tick": tick, "raw_choice_selected_by_host_sampler": choice,
                          "model_six_code_scores": scores, "legal": obs["legal"],
                          "sampling_receipt": evidence,
                          "raw_unmodified_greedy_probe": raw_probe[0] if tick == 0 else None})
        # Verified per-forward disk progress, independently sampled by supervisor.
        with (out / "forward-pulse.jsonl").open("a", encoding="utf-8") as pulse:
            pulse.write(json.dumps({
                "pid": os.getpid(), "lease_id": args.lease, "tick": tick,
                "finished_utc_epoch_ns": time.time_ns(), "real_score_forward": True,
                "model_scored_choice": choice, "host_legal": choice in obs["legal"],
            }, sort_keys=True, separators=(",", ":")) + "\n")
        return choice
    actual = support.rollout(args.seed, args.family, TURNS, choose)
    if len(actual["events"]) != len(decisions) or len(decisions) != TURNS:
        raise ValueError("W0 source-applied event count mismatch")
    for event, d in zip(actual["events"], decisions):
        if event["raw_choice"] != d["raw_choice_selected_by_host_sampler"]:
            raise ValueError("model-choice / host-event receipt mismatch")
    png = out/"original-w0.png"
    png.write_bytes(worm.png_bytes(actual))
    write_json(out/"original-w0.json", actual)
    write_json(out/"model-decisions.json", decisions)
    # Independent, seed-matched model-free controls; only the learned arm runs LFM.
    controls = {}
    for baseline in ("random", "wall", "fidget"):
        control = worm.run(seed=args.seed, mode=baseline, family=args.family, steps=TURNS)
        control_png = out / ("control-" + baseline + ".png")
        control_png.write_bytes(worm.png_bytes(control))
        write_json(out / ("control-" + baseline + ".json"), control)
        controls[baseline] = {
            "policy": baseline, "model_forwards": 0, "native_execution": False,
            "steps_executed": len(control["events"]),
            "metrics": support.statistics(control),
            "png_sha256": sha256_file(control_png),
        }
    write_json(out/"completed.json", {
        "schema": VERSION, "status": "completed", "lease_id": args.lease,
        "pid": os.getpid(), "parent_pid": os.getppid(), "runner": os.getenv("RUNNER_NAME"),
        "run_id": os.getenv("GITHUB_RUN_ID"), "job": os.getenv("GITHUB_JOB"),
        "phase": args.phase, "seed": args.seed, "family": args.family,
        "model": MODEL, "revision": REVISION, "adapter_sha256": adapter_sha,
        "model_score_forwards": len(times), "genuine_model_forward": True,
        "host_sampling": "source-fixed-W8-70percent-model-30percent-flat-T2.5",
        "host_applied_steps": len(actual["events"]),
        "raw_greedy_probe": raw_probe[0],
        "metrics": support.statistics(actual), "png_sha256": sha256_file(png),
        "original_matched_model_free_controls": controls,
        "synthetic_W0_128_turn_success_only": True,
        "score_latency_seconds": [round(x, 4) for x in times],
        "p50_score_seconds": round(median(times), 4),
        "p95_score_seconds": percent(times, 0.95),
        "wall_seconds": round(time.monotonic()-start, 3),
        "cpu_seconds": round(time.process_time(), 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "original_optimizer_steps": 45, "new_optimizer_steps": 0,
        "native_execution": False, "sealed_FINAL_opened": False,
        "128_turn_host_only_synthetic": True,
    })
    print(canonical({"event": "resident-worm-completed", "lease": args.lease,
                     "pid": os.getpid(), "turns": len(decisions)}), flush=True)


def supervisor(args):
    out = args.output
    if out.exists():
        raise RuntimeError("immutable supervisor lease already exists")
    out.mkdir(parents=True)
    deadline = time.monotonic()+GLOBAL_DEADLINE
    phases = [("one", [(4101, "greebles")]),
              ("two", [(4101, "greebles"), (4119, "pipework")])]
    reports = []
    for phase, roster in phases:
        if STOP or time.monotonic() > deadline-240:
            raise TimeoutError("supervisor global deadline")
        phase_dir = out/phase
        phase_dir.mkdir()
        go_file = phase_dir/"go"
        procs = []
        for idx, (seed, family) in enumerate(roster):
            lease = f"lfm-w13-{os.getenv('GITHUB_RUN_ID','local')}-{phase}-{idx}-{seed}"
            dest = phase_dir/lease
            cmd = [sys.executable, __file__, "--worker", "--model", str(args.model),
                   "--worm-root", str(args.worm_root), "--support-root", str(args.support_root),
                   "--producer-root", str(args.producer_root), "--output", str(dest),
                   "--go-file", str(go_file), "--phase", phase,
                   "--lease", lease, "--seed", str(seed), "--family", family]
            stdout = (phase_dir/(lease+".log")).open("w")
            procs.append((subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT), dest, stdout))
        try:
            while True:
                if STOP or time.monotonic() > deadline-90:
                    raise TimeoutError("supervisor Stop or deadline before barrier")
                if all((dest/"ready.json").is_file() for _,dest,_ in procs):
                    if any(p.poll() is not None for p,_,_ in procs):
                        raise RuntimeError("worker exited before simultaneous barrier")
                    break
                if any(p.poll() is not None for p,_,_ in procs):
                    raise RuntimeError("worker exited before model readback")
                time.sleep(0.3)
            barrier_time = time.time()
            go_file.write_text(str(barrier_time))
            samples = []
            while any(p.poll() is None for p,_,_ in procs):
                if STOP or time.monotonic() > deadline-35:
                    raise TimeoutError("supervisor Stop or inference deadline")
                samples.append({"monotonic_ns": time.monotonic_ns(),
                       "rss_kib": {str(p.pid): vm_rss_kib(p.pid) for p,_,_ in procs},
                       "alive_pids": [p.pid for p,_,_ in procs if p.poll() is None]})
                time.sleep(0.3)
            if any(p.returncode != 0 for p,_,_ in procs):
                raise RuntimeError("one or more resident inference workers failed")
            completed = [json.loads((dest/"completed.json").read_text()) for _,dest,_ in procs]
            if len(completed) != len(roster) or len({r["pid"] for r in completed}) != len(roster):
                raise ValueError("resident PID cardinality does not prove real processes")
            if len(roster) > 1 and not any(len(s["alive_pids"]) == len(roster) for s in samples):
                raise ValueError("no recorded simultaneous live model processes")
            phase_summary = {
                "phase": phase, "resident_models": len(roster),
                "simultaneously_loaded_before_go": True,
                "distinct_pids": [r["pid"] for r in completed],
                "model_hashes": [r["adapter_sha256"] for r in completed],
                "lease_ids": [r["lease_id"] for r in completed],
                "seeds": [r["seed"] for r in completed],
                "all_real_inference": True,
                "model_score_forwards": sum(r["model_score_forwards"] for r in completed),
                "p50_score_seconds": round(median(x for r in completed for x in r["score_latency_seconds"]),4),
                "p95_score_seconds": percent([x for r in completed for x in r["score_latency_seconds"]],0.95),
                "peak_sampled_aggregate_rss_kib": max((sum(v or 0 for v in s["rss_kib"].values()) for s in samples),default=0),
                "peak_worker_rss_kib": max(r["peak_rss_kib"] for r in completed),
                "barrier_wall_time": barrier_time,
                "end_wall_time": time.time(),
                "elapsed_after_barrier_seconds": round(time.time()-barrier_time,3),
                "worker_results": completed,
            }
            write_json(phase_dir/"sampling-observations.json", samples)
            write_json(phase_dir/"summary.json", phase_summary)
            reports.append(phase_summary)
            write_json(out/"rolling-evidence.json", {"phases":reports, "new_optimizer_steps":0})
        finally:
            for p,_,_ in procs:
                if p.poll() is None:
                    p.terminate()
            for p,_,f in procs:
                try:
                    p.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait()
                f.close()
    one,two=reports
    # A capacity observation, not an automatic four-worm allocation.
    throughput1=one["model_score_forwards"]/one["elapsed_after_barrier_seconds"]
    throughput2=two["model_score_forwards"]/two["elapsed_after_barrier_seconds"]
    report = {"schema":VERSION, "status":"completed",
        "runner_id":os.getenv("RUNNER_NAME"), "run_id":os.getenv("GITHUB_RUN_ID"),
        "job":os.getenv("GITHUB_JOB"),"cpu_affinity_count":len(os.sched_getaffinity(0)),
        "model":MODEL,"revision":REVISION,"producer_source":PRODUCER_SHA,
        "unique_loaded_pids_total":sum(len(r["distinct_pids"]) for r in reports),
        "peak_simultaneous_resident_inference_models":2,
        "four_simultaneous_worms_proven":False,
        "two_x_two_fleet_authorized":False,
        "throughput_1_forward_per_second":round(throughput1,5),
        "throughput_2_aggregate_forwards_per_second":round(throughput2,5),
        "aggregate_speedup_2_over_1":round(throughput2/throughput1,3),
        "peak_two_model_aggregate_rss_kib":two["peak_sampled_aggregate_rss_kib"],
        "phases":reports, "optimizer_updates":0, "native_execution":False,
        "sealed_FINAL_opened":False,"source_truth":"synthetic_W0_original_PNG",
        "target_128_turns_deferred_until_host_safety_verified":True}
    write_json(out/"summary.json",report)
    print(canonical({"event":"lfm-w13-resident-comparison","speedup":report["aggregate_speedup_2_over_1"],
        "peak_rss_kib":report["peak_two_model_aggregate_rss_kib"]}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("model","worm_root","support_root","producer_root","output"):
        parser.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--go-file",type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--lease")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--family")
    args = parser.parse_args()
    try:
        if args.worker:
            if not all((args.go_file,args.phase,args.lease,args.seed,args.family)):
                raise ValueError("incomplete worker identity")
            worker(args)
        else:
            supervisor(args)
    except Exception as exc:
        emit_failure(args.output, str(exc))
        raise


if __name__=="__main__":
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    main()
