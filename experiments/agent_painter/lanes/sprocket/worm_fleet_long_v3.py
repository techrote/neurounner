#!/usr/bin/env python3
"""SPROCKET WORM F3: four independent long-horizon trained root-Instruct processes across two VMs.

Must never dispatch until independently qualified F1 serial-vs-two-resident
original model receipt is SUCCESS. This source does not start training; models
select via genuine six-action forward logits, host legal/seed and pinned W0.
Four-way PASS additionally requires overlapping original process timestamps.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import worm_fleet_capacity_v1 as f1

STEPS=128
SEEDS=(8111,8123,8147,8161)
FAMILIES=("greebles","pipework","architecture","greebles")
TRAINED_SOURCE=f1.TRAIN_SOURCE
F1_HEAD="9b4627d8f3602627291d5939f4787b5976b6c195"
F2_HEAD="b44038a2e26850bf5f97457243f1b2f9c018d2ca"
F2_RUN=43766586563

def read_f1(path):
    root=Path(path)
    bench=root/"results"/"benchmark.json"
    if not bench.is_file():
        raise ValueError("F1 benchmark not complete; four-way inference NOT admitted")
    obj=json.loads(bench.read_text())
    if not (obj.get("qualified_two_resident") is True and
            obj.get("status")=="qualified_two_resident" and
            obj.get("source_sha")==F1_HEAD and
            obj.get("model")==f1.MODEL_ID and obj.get("revision")==f1.REVISION and
            obj.get("producer_failed_exit137_preserved") is True and
            obj.get("sealed_final_opened") is False and
            obj.get("max_approved_decisions_now")==8 and
            obj.get("dual",{}).get("maximum_simultaneous_live_processes")==2 and
            obj.get("serial",{}).get("real_forwards")==obj.get("dual",{}).get("real_forwards")):
        raise ValueError("F1 genuine 1-vs-2 checkpoint/throughput qualification unavailable")
    # Independently check every original F1 artifact member, not just a summary flag.
    sha=root/"SHA256SUMS.txt"
    if not sha.is_file(): raise ValueError("F1 original SHA256SUMS missing")
    entries=sha.read_text().splitlines()
    if len(entries)<15: raise ValueError("F1 original artifact incomplete")
    for line in entries:
        digest,name=line.split("  ",1)
        f=root/name.removeprefix("./")
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
            raise ValueError("F1 original artifact hash mismatch: "+name)
    for stage in ("one_resident_serial","two_resident_concurrent"):
        for index in range(2):
            lease="%s-%s-%d"%(obj["lease"],stage,index)
            c=json.loads((root/"results"/stage/lease/"complete.json").read_text())
            if (c["model_id"]!=f1.MODEL_ID or c["revision"]!=f1.REVISION or
                c["real_trained_six_score_forwards"]<1 or c["optimizer_steps"]!=0 or
                c["source_faithful_W0_not_native"] is not True or
                c["checkpoint_inventory_sha256"] is None):
                raise ValueError("F1 worker/model receipt mismatch")
    return obj

def read_f2(path):
    """Admit F3 only from the immutable original F2 4-forward overlap artifact."""
    root=Path(path)
    a=root/"four-way-audit.json"
    if not a.is_file():
        raise ValueError("F2 original four-model audit missing; long128 NOT admitted")
    obj=json.loads(a.read_text())
    rows=obj.get("original_worm_processes", [])
    if not (obj.get("source_sha")==F2_HEAD and
            obj.get("schema")=="cybersand.worm.sprocket-4-live-evidence/1" and
            obj.get("four_actually_simultaneous") is True and
            obj.get("actual_four_MODEL_INFERENCE_interval_overlap_seconds",0)>1 and
            obj.get("decision_bound")==24 and
            obj.get("optimizer_steps_this_run")==0 and
            obj.get("producer_failure_preserved") is True and
            obj.get("not_native") is True and len(rows)==4):
        raise ValueError("F2 original qualified four-live model evidence mismatch")
    if (len({(slot,w["pid"],w["lease"],w["seed"]) for slot,w in rows})!=4 or
            len({w["lease"] for _,w in rows})!=4 or
            len({w["seed"] for _,w in rows})!=4 or
            len({w["checkpoint_inventory_sha256"] for _,w in rows})!=1 or
            any(w["real_trained_six_score_forwards"]!=24 for _,w in rows)):
        raise ValueError("F2 PID/seed/checkpoint/live-forward audit mismatch")
    for slot in (0,1):
        folder=root/("fleet-"+str(slot))
        sha=folder/"SHA256SUMS.txt"
        if not sha.is_file():
            raise ValueError("F2 original worker SHA256SUMS missing")
        lines=sha.read_text().splitlines()
        if len(lines)<25:
            raise ValueError("F2 original worker artifact incomplete")
        for line in lines:
            digest,name=line.split("  ",1)
            file=folder/name.removeprefix("./")
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
                raise ValueError("F2 original immutable member mismatch: "+name)
        if len(list(folder.rglob("*.png")))!=8:
            raise ValueError("F2 original W0 PNG evidence incomplete")
    return obj

def worker(args):
    # Reuses exact F1 verified model/checkpoint loader, W0 application,
    # original PNG controls and no-optimizer safety, increasing only decision cap.
    f1.STAGE_STEPS=STEPS
    f1.SEEDS=SEEDS
    f1.FAMILIES=tuple(dict.fromkeys(FAMILIES))
    f1.worker(args)

def run_fleet(args):
    if args.slot not in (0,1) or args.steps!=STEPS:
        raise ValueError("only post-F2 approved two-VM, four-worm 128-turn F3 inference")
    f1receipt=read_f1(args.f1)
    f2receipt=read_f2(args.f2)
    args.output.mkdir(parents=True,exist_ok=False)
    base=args.slot*2
    stop_at=time.monotonic()+args.deadline
    limit=min(12*1024**2,int(.7*f1.memory_limit_kib()))
    stage={"schema":"cybersand.worm.sprocket-fleet-f3/1",
           "parent_F1_source":F1_HEAD,
           "F1_run_id":f1receipt["run_id"],"F1_qualified":True,
           "F2_run_id":F2_RUN,"F2_source":F2_HEAD,"F2_qualified":True,
           "F2_overlap_seconds":f2receipt["actual_four_MODEL_INFERENCE_interval_overlap_seconds"],
           "run_id":os.environ.get("GITHUB_RUN_ID"),"source_sha":os.environ.get("GITHUB_SHA"),
           "host_job":os.environ.get("GITHUB_JOB"),"slot":args.slot,
           "model_id":f1.MODEL_ID,"model_revision":f1.REVISION,
           "trained_checkpoint":"W6-step24-original-failed-producer",
           "train_source":TRAINED_SOURCE,"w0_commit":f1.W0_COMMIT,
           "native":False,"optimizer_steps":0,"sealed_final_opened":False,
           "decision_bound":STEPS,"future_possible_bound_after_safety":128,
           "memory_limit_kib":limit,"start_utc":time.time(),
           "active":0,"queued":2,"completed":0,"failed":0,"cancelled":0}
    f1.save(args.output/"lease-and-counts.json",stage)
    children=[]
    def spawn(index):
        seed=SEEDS[index]
        family=FAMILIES[index]
        lease="%s-%s-f3-%d-%d"%(args.lease,args.slot,index,seed)
        dest=args.output/lease
        log=open(args.output/(lease+".log"),"w")
        cmd=[sys.executable,__file__,"--mode","worker",
             "--model",str(args.model),"--w0",str(args.w0),
             "--checkpoint",str(args.checkpoint),"--output",str(dest),
             "--lease",lease,"--family",family,"--seed",str(seed),
             "--steps",str(args.steps)]
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append({"p":p,"pid":p.pid,"seed":seed,"family":family,
                         "lease":lease,"path":dest,"log":log,"start_utc":time.time()})
    def cancellation(sig,frame):
        raise InterruptedError("host timeout/cancellation signal %d"%sig)
    signal.signal(signal.SIGTERM,cancellation)
    signal.signal(signal.SIGINT,cancellation)
    try:
        spawn(base);spawn(base+1)
        stage["queued"]=0
        stage["active"]=2
        f1.save(args.output/"lease-and-counts.json",stage)
        max_live=0
        peak_rss=0
        while True:
            live=[x for x in children if x["p"].poll() is None]
            max_live=max(max_live,len(live))
            rss=sum(f1.proc_rss(x["pid"]) for x in live)
            peak_rss=max(peak_rss,rss)
            if rss>limit:
                raise MemoryError("two resident models exceeded 70% memory admission")
            if time.monotonic()>stop_at:
                raise TimeoutError("two real model inference process timeout")
            if not live:break
            stage["active"]=len(live)
            stage["completed"]=sum(x["p"].poll()==0 and (x["path"]/"complete.json").is_file()
                                   for x in children)
            stage["failed"]=sum(x["p"].poll() not in (None,0) for x in children)
            f1.save(args.output/"lease-and-counts.json",stage)
            time.sleep(1)
        stage["active"]=0
        stage["completed"]=sum(x["p"].returncode==0 and (x["path"]/"complete.json").is_file()
                                for x in children)
        stage["failed"]=2-stage["completed"]
        stage["maximum_actually_live_pids"]=max_live
        stage["peak_two_pids_rss_kib"]=peak_rss
        stage["end_utc"]=time.time()
        stage["workers"]=[]
        for x in children:
            x["log"].close()
            if x["p"].returncode!=0:
                raise RuntimeError("live inference worker failed: "+x["lease"])
            c=json.loads((x["path"]/"complete.json").read_text())
            if (c["pid"]!=x["pid"] or c["seed"]!=x["seed"] or c["lease"]!=x["lease"] or
                c["real_trained_six_score_forwards"]<96 or c["optimizer_steps"]!=0 or
                c["source_faithful_W0_not_native"] is not True):
                raise ValueError("worker identity/receipt contract violated")
            stage["workers"].append({k:c[k] for k in
                    ("pid","seed","family","lease","start_utc","end_utc",
                     "inference_start_utc","inference_end_utc",
                     "real_trained_six_score_forwards","model_download_receipt_sha256",
                     "checkpoint_inventory_sha256","peak_rss_kib","wall_seconds")})
        if max_live!=2 or stage["completed"]!=2:
            raise ValueError("never demonstrated two actually live per VM")
        stage["status"]="two_original_model_processes_complete"
        f1.save(args.output/"fleet.json",stage)
        print(json.dumps({"slot":args.slot,"live":max_live,"complete":2,
             "rss_kib":peak_rss}),flush=True)
    except BaseException as exc:
        stage["status"]="FAILED_OR_CANCELLED"
        stage["error"]=str(exc)
        stage["cancelled"]=sum(x["p"].poll() is None for x in children)
        f1.save(args.output/"error-ledger.json",{
            "error":str(exc),"type":type(exc).__name__,
            "trace":traceback.format_exc(),"stage":stage})
        for x in children:
            if x["p"].poll() is None:
                try:os.killpg(x["pid"],signal.SIGTERM)
                except ProcessLookupError:pass
        for x in children:
            try:x["p"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(x["pid"],signal.SIGKILL)
                except ProcessLookupError:pass
            x["log"].close()
        raise
    finally:
        checks={}
        for file in args.output.rglob("*"):
            if file.is_file() and file.name!="SHA256SUMS.json":
                checks[str(file.relative_to(args.output))]=f1.digest(file)
        f1.save(args.output/"SHA256SUMS.json",checks)

def audit(args):
    roots=[Path(args.artifacts)/("fleet-"+str(i))/"results" for i in (0,1)]
    workers=[]
    results=[]
    for slot,root in enumerate(roots):
        obj=json.loads((root/"fleet.json").read_text())
        if (obj["slot"]!=slot or obj["status"]!="two_original_model_processes_complete"
            or obj["completed"]!=2 or obj["maximum_actually_live_pids"]!=2
            or obj["source_sha"]!=args.expected_sha or obj.get("F2_qualified") is not True):
            raise ValueError("F3 source/job/2-resident qualification mismatch")
        results.append(obj)
        workers.extend((slot,w) for w in obj["workers"])
    # Independently verify original per-worker PNGs, traces and artifact hashes.
    for slot,root in enumerate(roots):
        parent=root.parent
        sha=parent/"SHA256SUMS.txt"
        if not sha.is_file():
            raise ValueError("F3 per-runner original SHA256SUMS missing")
        for line in sha.read_text().splitlines():
            digest,name=line.split("  ",1)
            f=parent/name.removeprefix("./")
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
                raise ValueError("F3 per-runner original SHA mismatch: "+name)
        if len(list(root.rglob("*.png")))!=8:
            raise ValueError("F3 original image count mismatch on slot "+str(slot))
    if any(w["real_trained_six_score_forwards"]<96 for _,w in workers):
        raise ValueError("F3 did not show >=96 actual model-forward decisions per worm")
    if len(workers)!=4 or len({w["lease"] for _,w in workers})!=4 or len({w["seed"] for _,w in workers})!=4:
        raise ValueError("not four unique inference worm leases/seeds")
    if len({(slot,w["pid"]) for slot,w in workers})!=4:
        raise ValueError("not four distinct job-qualified PIDs")
    if len({w["model_download_receipt_sha256"] for _,w in workers})!=1:
        raise ValueError("four worms not using the same pinned model download identity")
    if len({w["checkpoint_inventory_sha256"] for _,w in workers})!=1:
        raise ValueError("four worms not on the same original checkpoint")
    overlap_start=max(w["inference_start_utc"] for _,w in workers)
    overlap_end=min(w["inference_end_utc"] for _,w in workers)
    complete=overlap_end>overlap_start+1
    result={"schema":"cybersand.worm.sprocket-4-live-long128-evidence/1",
            "source_sha":args.expected_sha,"four_unique_job_pid_model_seed_leases":True,
            "actual_four_MODEL_INFERENCE_interval_overlap_seconds":max(0,overlap_end-overlap_start),
            "four_actually_simultaneous":complete,"not_native":True,
            "optimizer_steps_this_run":0,"decision_bound":STEPS,
            "original_worm_processes":workers,"producer_failure_preserved":True}
    f1.save(args.output,result)
    if not complete:
        raise RuntimeError("4 concurrent inference not demonstrable: no overlap")
    print(json.dumps({"four_simultaneous":True,"overlap":overlap_end-overlap_start}))
    
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=("worker","fleet","audit"))
    p.add_argument("--model",type=Path)
    p.add_argument("--w0",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--f1",type=Path)
    p.add_argument("--f2",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--lease",default="not-set")
    p.add_argument("--slot",type=int,default=-1)
    p.add_argument("--family",default="greebles")
    p.add_argument("--seed",type=int,default=7013)
    p.add_argument("--steps",type=int,default=STEPS)
    p.add_argument("--deadline",type=int,default=3400)
    p.add_argument("--artifacts",type=Path)
    p.add_argument("--expected-sha")
    args=p.parse_args()
    if args.mode=="worker":worker(args)
    elif args.mode=="fleet":run_fleet(args)
    else:audit(args)
