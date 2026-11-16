#!/usr/bin/env python3
"""W12: two real, independently loaded full-LFM W10 steering worms on one VM.

Run on two distinct GitHub Actions matrix VMs for 2x2 physical residency.
The W11 worker verifies complete pinned LFM + adapter bytes and performs actual
local-observation -> legal score sample -> engine-owned W0 apply -> original PNG.
No shared model process; no optimization; cross-VM overlap judged from UTC receipts.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import canonical, sha256_file, write_json
import worm_w13_worker_v1 as w11

VERSION="lfm-worm-w13-two-vm-128-turn-four-resident-synthetic-v1"
ROSTER={
    "alpha":((6111,"greebles"),(6127,"pipework")),
    "beta": ((6143,"architecture"),(6167,"greebles")),
}
DEADLINE_SECONDS=1700
STOP=False

def stop(_sig,_frame):
    global STOP
    STOP=True

def rss(pid):
    return w11.vm_rss_kib(pid)

def main():
    p=argparse.ArgumentParser()
    for key in ("model","worm_root","support_root","producer_root","output"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    p.add_argument("--shard",choices=tuple(ROSTER),required=True)
    args=p.parse_args()
    out=args.output
    if out.exists():
        raise ValueError("immutable fleet shard output lease already exists")
    out.mkdir(parents=True)
    start=time.monotonic()
    go_file=out/"go"
    processes=[]
    try:
        for idx,(seed,family) in enumerate(ROSTER[args.shard]):
            lease=f"lfm-w13-{os.environ.get('GITHUB_RUN_ID','local')}-{args.shard}-{idx}-{seed}"
            dest=out/lease
            cmd=[sys.executable, w11.__file__, "--worker",
                "--model",str(args.model),"--worm-root",str(args.worm_root),
                "--support-root",str(args.support_root),
                "--producer-root",str(args.producer_root),
                "--output",str(dest),"--go-file",str(go_file),
                "--phase","fleet-"+args.shard,"--lease",lease,
                "--seed",str(seed),"--family",family]
            fd=(out/(lease+".log")).open("w")
            processes.append((subprocess.Popen(cmd,stdout=fd,stderr=subprocess.STDOUT),dest,fd))
        while not all((path/"ready.json").is_file() for _,path,_ in processes):
            if STOP or time.monotonic()-start > DEADLINE_SECONDS-200:
                raise TimeoutError("fleet resident readiness/Stop deadline")
            if any(proc.poll() is not None for proc,_,_ in processes):
                raise RuntimeError("fleet model worker failed before loading full model")
            time.sleep(.2)
        if any(proc.poll() is not None for proc,_,_ in processes):
            raise RuntimeError("fleet model worker failed at ready barrier")
        start_epoch_ns=time.time_ns()
        go_file.write_text(str(start_epoch_ns))
        samples=[]
        while any(proc.poll() is None for proc,_,_ in processes):
            if STOP or time.monotonic()-start > DEADLINE_SECONDS-30:
                raise TimeoutError("fleet inference deadline/Stop")
            samples.append({"epoch_ns":time.time_ns(),
                 "alive": [proc.pid for proc,_,_ in processes if proc.poll() is None],
                 "rss_kib":{str(proc.pid):rss(proc.pid) for proc,_,_ in processes},
                 "forward_pulse_bytes":{str(proc.pid):((path/"forward-pulse.jsonl").stat().st_size
                     if (path/"forward-pulse.jsonl").exists() else 0)
                     for proc,path,_ in processes}})
            time.sleep(.25)
        end_epoch_ns=time.time_ns()
        if any(proc.returncode for proc,_,_ in processes):
            raise RuntimeError("fleet model worker failed during inference")
        rows=[json.loads((path/"completed.json").read_text()) for _,path,_ in processes]
        if len(rows)!=2 or len({row["pid"] for row in rows})!=2 or not any(len(s["alive"])==2 for s in samples):
            raise ValueError("both genuine resident worker PIDs not observed in overlap")
        # Both resident processes must advance genuine on-disk forward receipts.
        for proc,_,_ in processes:
            sequence=[s["forward_pulse_bytes"][str(proc.pid)] for s in samples
                      if proc.pid in s["alive"]]
            if not sequence or max(sequence) <= min(sequence):
                raise ValueError("no sampled progress by genuinely loaded model")
        if any(row["model_score_forwards"]!=w11.TURNS or row["host_applied_steps"]!=w11.TURNS for row in rows):
            raise ValueError("missing 128 real model forwards or W0 actions")
        if len({row["adapter_sha256"] for row in rows})!=1:
            raise ValueError("model adapter hash mismatch")
        from PIL import Image,ImageDraw
        contact=Image.new("RGB",(768,420),(24,24,24))
        draw=ImageDraw.Draw(contact)
        for idx,((_,path,_),row) in enumerate(zip(processes,rows)):
            x=idx*384
            draw.text((x+6,8),f"{args.shard}: {row['family']} seed {row['seed']} PID {row['pid']}",fill=(245,245,245))
            with Image.open(path/"original-w0.png") as img:
                if img.mode!="RGB" or img.size!=(384,384):
                    raise ValueError("original W0 image geometry changed")
                contact.paste(img,(x,36))
        contact_file=out/"original-png-contact.png"
        contact.save(contact_file)
        report={
            "schema":VERSION, "status":"completed", "shard":args.shard,
            "run_id":os.getenv("GITHUB_RUN_ID"),"job_id":os.getenv("GITHUB_JOB"),
            "runner_name":os.getenv("RUNNER_NAME"),"cpu_affinity_count":len(os.sched_getaffinity(0)),
            "resident_models":2,"simultaneous_loaded_models_verified":True,
            "model":"LiquidAI/LFM2.5-VL-450M",
            "model_revision":w11.REVISION,"original_W10_source":w11.PRODUCER_SHA,
            "original_W10_run":w11.PRODUCER_RUN,
            "adapter_sha256":rows[0]["adapter_sha256"],
            "distinct_loaded_pids":[x["pid"] for x in rows],
            "distinct_lease_ids":[x["lease_id"] for x in rows],
            "independent_seeds":[x["seed"] for x in rows],
            "start_epoch_ns":start_epoch_ns,"end_epoch_ns":end_epoch_ns,
            "actual_inference_model_forwards":sum(x["model_score_forwards"] for x in rows),
            "actual_W0_applied_decisions":sum(x["host_applied_steps"] for x in rows),
            "original_png_contact_sha256":sha256_file(contact_file),
            "peak_aggregate_resident_rss_kib":max(sum(v or 0 for v in s["rss_kib"].values()) for s in samples),
            "wall_after_barrier_seconds":round((end_epoch_ns-start_epoch_ns)/1e9,3),
            "workers":rows,"new_optimizer_updates":0,"native_execution":False,
            "sealed_FINAL_opened":False,"synthetic_W0_128_turn_qualified_only":True,
            "two_VMs_four_simultaneous_proof_requires_other_shard_receipt":True
        }
        write_json(out/"resource-samples.json",samples)
        write_json(out/"summary.json",report)
        print(canonical({"event":"two-real-resident-LFM-worms-complete",
            "shard":args.shard,"pids":report["distinct_loaded_pids"],
            "model_forwards":report["actual_inference_model_forwards"]}),flush=True)
    except BaseException as exc:
        w11.emit_failure(out,str(exc))
        raise
    finally:
        for proc,_,_ in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc,_,fd in processes:
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
            fd.close()

if __name__=="__main__":
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    main()
