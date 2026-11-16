#!/usr/bin/env python3
"""W14: pinned full multimodal W10 LoRA live family/STOP diagnostic.
One model, original W0 host, real logits, original seed-matched PNGs.
No optimizer, native application, FINAL access or production change.
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
import math
from worm_w8_scored_crn_v1 import common_u
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint, load_base

VERSION = "lfm-worm-w14-family-STOP-unmasked-synthetic-24-v1"
MODEL = "LiquidAI/LFM2.5-VL-450M"
REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"
W0_SHA = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
SUPPORT_SHA = "cc27b3206cb61f98432fa95b799c9b9ff508c2bc"
PRODUCER_SHA = "1e57a4df8d344b209a14df886dd898211dcc4918"
PRODUCER_RUN = 43754365984
TURNS = 24
GLOBAL_DEADLINE = 1050
STOP = False



def scored_policy(scores, legal, mode, seed, tick):
    """Host legal sampler: retain optional STOP, unlike W13's suppressed STOP."""
    if mode not in ("learned", "flat") or len(legal)!=len(set(legal)) or any(a not in w7.ACTIONS for a in legal):
        raise ValueError("invalid sampler/menu")
    allowed=[a for a in w7.ACTIONS if a in legal]
    if not allowed:
        raise ValueError("empty host menu")
    mix=.7 if mode=="learned" else 0.
    values=[scores["candidate_logits"][a]/2.5 for a in allowed] if mix else [0.]*len(allowed)
    peak=max(values)
    exp=[math.exp(v-peak) for v in values]
    probs=[mix*x/sum(exp)+(1-mix)/len(allowed) for x in exp]
    u=common_u(seed,tick)
    cumulative=0.
    choice=allowed[-1]
    for a,p in zip(allowed,probs):
        cumulative+=p
        if u<cumulative:
            choice=a
            break
    return choice,{"sampled_legal":allowed,"legal":list(legal),"sample_probs":dict(zip(allowed,probs)),
       "host_u":u,"choice":choice,"sampler":mode,"temperature":2.5,"model_fraction":mix,
       "STOP_deferred_unless_forced":False}

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
        "sealed_FINAL_opened": False, "24_turn_host_only_synthetic": True,
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
    if len(actual["events"]) != len(decisions) or not 1 <= len(decisions) <= TURNS:
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
        "host_sampling": "W14-host-legal-STOP-included-70percent-model-30percent-flat-T2.5",
        "host_applied_steps": len(actual["events"]),
        "raw_greedy_probe": raw_probe[0],
        "metrics": support.statistics(actual), "png_sha256": sha256_file(png),
        "original_matched_model_free_controls": controls,
        "synthetic_W0_24_turn_budget_only": True,
        "score_latency_seconds": [round(x, 4) for x in times],
        "p50_score_seconds": round(median(times), 4),
        "p95_score_seconds": percent(times, 0.95),
        "wall_seconds": round(time.monotonic()-start, 3),
        "cpu_seconds": round(time.process_time(), 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "original_optimizer_steps": 45, "new_optimizer_steps": 0,
        "native_execution": False, "sealed_FINAL_opened": False,
        "24_turn_host_only_synthetic": True,
    })
    print(canonical({"event": "resident-worm-completed", "lease": args.lease,
                     "pid": os.getpid(), "turns": len(decisions)}), flush=True)



def main():
    p=argparse.ArgumentParser()
    for key in ("model","worm_root","support_root","producer_root","output","go_file"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    for key in ("phase","lease","family"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--seed",type=int,required=True)
    args=p.parse_args()
    if args.seed not in (6211,6217) or args.family not in ("greebles","pipework","architecture"):
        raise ValueError("W14 requested outside frozen world/family roster")
    try:
        worker(args)
    except BaseException as exc:
        emit_failure(args.output,str(exc))
        raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,handle_stop)
    signal.signal(signal.SIGINT,handle_stop)
    main()
