#!/usr/bin/env python3
"""Untuned SmolLM2-135M 1-vs-2 REAL resident inference benchmark, W0 synthetic only.

Two identical seed/family episodes run serially, then simultaneously. The only
model action is next-token digit logit scoring; W0 owns legality and drawing.
No optimizer, adapter, native Godot, arbitrary material writes or paid endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
W0_SOURCE = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
W0_BLOB = "8979b6f0296e81051194bacc907992d38e006605"
SCHEMA = "driv3r.smol135-residency/1"
SEEDS = (4101, 4119)
FAMILIES = ("greebles", "pipework")
TURNS = 8
STOP = False

def stopping(_sig, _frame):
    global STOP
    STOP = True

def ensure(ok, reason):
    if not ok:
        raise ValueError(reason)

def write(path, data):
    ensure(not path.exists(), "immutable evidence file already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")

def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_w0(path):
    spec = importlib.util.spec_from_file_location("pinned_w0_host", path)
    ensure(spec is not None and spec.loader is not None, "missing W0 Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ensure(module.ACTIONS == ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP"),
           "W0 legal action vocabulary changed")
    return module

def rss_kib(pid):
    try:
        return next(int(s.split()[1]) for s in Path("/proc/%d/status" % pid).read_text().splitlines()
                    if s.startswith("VmRSS:"))
    except (OSError, StopIteration, ValueError):
        return None

def identity(args):
    return {"schema": SCHEMA, "lease": args.lease, "phase": args.phase,
            "seed": args.seed, "family": args.family, "turn_budget": TURNS,
            "model": MODEL, "revision": REV, "w0_source": W0_SOURCE,
            "source_sha": os.getenv("GITHUB_SHA"), "run_id": os.getenv("GITHUB_RUN_ID"),
            "job_id": os.getenv("GITHUB_JOB"), "native_execution": False,
            "synthetic_W0": True, "optimizer_steps": 0, "adapter": None,
            "model_selection": "host_legal_single_digit_next_token_logits"}

def worker(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    meta = {**identity(args), "pid": os.getpid()}
    write(out/"identity.json", meta)
    w0 = load_w0(args.w0)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                             trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                 trust_remote_code=False, dtype=torch.float32)
    model.eval()
    ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(len(w0.ACTIONS))]
    ensure(all(len(v) == 1 for v in ids) and len({v[0] for v in ids}) == len(ids),
           "six distinct single-digit action IDs required")
    write(out/"model-ready.json", {"pid": os.getpid(), "model_class": type(model).__name__,
          "model_config": hash_file(args.model_dir/"config.json"),
          "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "model_forward_count": 0, "training": False, "w0_action_ids": [v[0] for v in ids]})
    t0 = time.monotonic()
    while not args.go.is_file():
        if STOP or time.monotonic()-t0 > 150:
            raise TimeoutError("model ready but supervisor did not release GO")
        time.sleep(.1)
    choices, fwd = [], []
    def choose(obs):
        ensure(not STOP, "Stop")
        tick = obs["tick"]
        # Explicit host-only early-STOP hold; raw model preference still retained.
        legal = list(obs["legal"])
        offered = [a for a in legal if a != "STOP"] if tick < TURNS-1 else legal
        ensure(offered, "no host-offered actions")
        prompt = ("You are a slightly deranged, curious line-steering worm. "
                  "Choose the next action by one digit only.\n"
                  + "Actions: " + ", ".join("%d=%s" % (i, x)
                  for i, x in enumerate(w0.ACTIONS)) + "\n"
                  + "State: " + json.dumps(obs, sort_keys=True, separators=(",", ":"))
                  + "\nDigit:")
        encoded = tokenizer(prompt, return_tensors="pt")
        before = time.monotonic_ns()
        with torch.inference_mode():
            logits = model(**encoded).logits[0, -1, :].detach().float().cpu()
        after = time.monotonic_ns()
        scored = {a: float(logits[ids[i][0]].item()) for i, a in enumerate(w0.ACTIONS)}
        ensure(all(abs(v) < 1e6 for v in scored.values()), "nonfinite model score")
        raw = max(w0.ACTIONS, key=lambda a: (scored[a], -w0.ACTIONS.index(a)))
        chosen = max(offered, key=lambda a: (scored[a], -w0.ACTIONS.index(a)))
        choices.append({"tick": tick, "observation": obs, "raw_unmasked_argmax": raw,
             "host_legal_offered": offered, "selected": chosen,
             "digit_logits": scored, "forward_start_ns": before, "forward_end_ns": after,
             "host_STOP_deferred_before_final_tick": tick < TURNS-1,
             "raw_text_generated": False, "selection_is_scored_not_raw_completion": True})
        fwd.append((before, after))
        return chosen
    record = w0.run(args.seed, mode="callback", family=args.family, steps=TURNS, selector=choose)
    ensure(len(record["events"]) == len(choices) >= 1, "model/host decision mismatch")
    ensure(all(a["raw_choice"] == c["selected"] and a["raw_legal"]
               for a, c in zip(record["events"], choices)), "host legality / model choice mismatch")
    # Independently replay original selected actions with the same pinned W0 host.
    replay = w0.run(args.seed, mode="replay", family=args.family, steps=TURNS,
                    raw_actions=[c["selected"] for c in choices])
    ensure(record["trajectory"] == replay["trajectory"] and
           w0.png_bytes(record) == w0.png_bytes(replay), "original W0 replay mismatch")
    (out/"original-w0.png").write_bytes(w0.png_bytes(record))
    write(out/"original-w0.json", record)
    (out/"original-model-scores.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in choices))
    write(out/"completed.json", {**meta, "status": "completed",
          "forward_count": len(fwd), "host_decisions": len(record["events"]),
          "unique_trail_cells": record["metrics"]["unique_trail_cells"],
          "recoveries": record["metrics"]["retraces"],
          "raw_unmasked_first": choices[0]["raw_unmasked_argmax"],
          "new_optimizer_steps": 0, "forward_interval_ns": [fwd[0][0], fwd[-1][1]],
          "forward_seconds": round(sum(b-a for a,b in fwd)/1e9, 5),
          "wall_seconds": round(time.monotonic()-start, 5),
          "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "original_png_sha256": hash_file(out/"original-w0.png")})
    print(json.dumps({"event":"model-completed","seed":args.seed,
                      "pid":os.getpid(),"model_forwards":len(fwd)}),flush=True)

def stop_group(processes):
    for p, _, handle in processes:
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    for p, _, handle in processes:
        try: p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
        handle.close()

def phase(args, label, concurrent):
    directory = args.out/label
    directory.mkdir(parents=True, exist_ok=False)
    go = directory/"go"
    processes, reports, samples = [], [], []
    start = time.monotonic()
    def spawn(i):
        lease = "%s-%s-%d" % (args.lease, label, i)
        dest = directory/lease
        cmd = [sys.executable, os.environ.get("DRIV3R_RESIDENCY_ENTRYPOINT", __file__), "--worker", "--model-dir", str(args.model_dir),
          "--w0", str(args.w0), "--out", str(dest), "--go", str(go), "--lease", lease,
          "--phase", label, "--seed", str(SEEDS[i]), "--family", FAMILIES[i]]
        log = (directory/(lease+".log")).open("w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((p,dest,log))
    try:
        spawn(0)
        if concurrent:
            spawn(1)
        next_idx = 2 if concurrent else 1
        while processes:
            if STOP or time.monotonic()-start > 650:
                raise TimeoutError("bounded resident phase")
            # Workers count as loaded only after each writes model-ready.json.
            if not go.exists() and all((d/"model-ready.json").is_file() for _,d,_ in processes):
                go.write_text("go\n")
            if go.exists():
                samples.append({"at_ns":time.monotonic_ns(),
                    "alive": [p.pid for p,_,_ in processes if p.poll() is None],
                    "rss_kib": {str(p.pid):rss_kib(p.pid) for p,_,_ in processes}})
            for p,d,f in list(processes):
                if p.poll() is None:
                    continue
                if p.returncode != 0 or not (d/"completed.json").is_file():
                    raise RuntimeError("inference worker failed; see original log "+str(d))
                f.close()
                reports.append(json.loads((d/"completed.json").read_text()))
                processes.remove((p,d,f))
                if not concurrent and next_idx < 2:
                    go.unlink(missing_ok=True)
                    spawn(next_idx)
                    next_idx += 1
            time.sleep(.2)
        ensure(len(reports) == 2, "phase did not run two real model episodes")
        if concurrent:
            intersection = min(x["forward_interval_ns"][1] for x in reports) - max(
                x["forward_interval_ns"][0] for x in reports)
            ensure(intersection > 0, "no overlap between real inference intervals")
            ensure(any(len(s["alive"]) == 2 and all(v is not None for v in s["rss_kib"].values())
                       for s in samples), "no two process RSS samples during inference")
        elapsed = time.monotonic()-start
        result = {"phase":label, "mode":"two_resident" if concurrent else "serial",
                  "wall_seconds":round(elapsed,5), "real_model_forwards":sum(x["forward_count"] for x in reports),
                  "peak_sampled_rss_kib":max((sum(v or 0 for v in s["rss_kib"].values())
                                             for s in samples),default=0),
                  "two_real_resident_pids_observed":concurrent,
                  "real_model_forward_overlap_ns":max(0,intersection) if concurrent else 0,
                  "workers":reports}
        write(directory/"result.json", result)
        write(directory/"rss-samples.json", samples)
        return result
    finally:
        stop_group(processes)

def supervisor(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ensure(os.getenv("GITHUB_SHA"), "requires GitHub source-pinned run")
    request=json.loads(args.request.read_text())
    ensure(request=={"schema":SCHEMA,"model":MODEL,"model_revision":REV,
        "w0_source":W0_SOURCE,"seeds":list(SEEDS),"families":list(FAMILIES),
        "turns":TURNS,"training":False,"native":False,"runner_max":1,
        "models_per_runner_max":2,"attempt":1}, "immutable request changed")
    write(args.out/"source.json", {**request, "github_sha":os.environ["GITHUB_SHA"],
          "run_id":os.getenv("GITHUB_RUN_ID"), "lease":args.lease,
          "model_config_sha256":hash_file(args.model_dir/"config.json"),
          "w0_sha256":hash_file(args.w0)})
    try:
        one=phase(args,"serial",False)
        two=phase(args,"dual",True)
        ensure(one["real_model_forwards"]==two["real_model_forwards"],"unequal workloads")
        report={"schema":SCHEMA,"status":"completed","model":MODEL,"revision":REV,
            "native_execution":False,"training":False,"two_resident_inference_measured":True,
            "four_resident_driv3r_worms_proven":False,
            "serial":one,"dual":two,
            "speedup_aggregate":round(one["wall_seconds"]/two["wall_seconds"],4),
            "peak_dual_sampled_rss_kib":two["peak_sampled_rss_kib"],
            "total_real_model_forwards":one["real_model_forwards"]+two["real_model_forwards"]}
        write(args.out/"summary.json",report)
        print(json.dumps({"event":"smol-real-residency","speedup":report["speedup_aggregate"],
          "dual_rss_kib":report["peak_dual_sampled_rss_kib"]}),flush=True)
    except BaseException as exc:
        write(args.out/"failure.json",{"error":str(exc),"type":type(exc).__name__,
              "no_success_claim":True,"native_execution":False})
        raise
    finally:
        checks={str(p.relative_to(args.out)):hash_file(p) for p in sorted(args.out.rglob("*"))
                if p.is_file() and p.name!="SHA256SUMS.json"}
        write(args.out/"SHA256SUMS.json",checks)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",action="store_true")
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--request",type=Path)
    p.add_argument("--lease",required=True)
    p.add_argument("--phase")
    p.add_argument("--seed",type=int)
    p.add_argument("--family")
    p.add_argument("--go",type=Path)
    a=p.parse_args()
    if a.worker:
        ensure(a.phase and a.seed is not None and a.family and a.go is not None,
               "incomplete worker identity")
        worker(a)
    else:
        ensure(a.request is not None, "missing request")
        supervisor(a)

if __name__=="__main__":
    signal.signal(signal.SIGTERM,stopping)
    signal.signal(signal.SIGINT,stopping)
    main()
