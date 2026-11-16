#!/usr/bin/env python3
"""WORM-001 SPROCKET F1: genuine root-Instruct W6 inference residency benchmark.

One process per independently resident model. Compare two serial worm processes
with two simultaneously resident worm processes on the SAME hosted CPU. This is
checkpoint-only inference, NOT AdamW, not native Godot, and not four-way fleet
admission. Never reinterpret a score-sampled action as a raw EOS5 completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import worm_score_v6_e3_eval as e3

MODEL_ID="kandivault/sprocket-500m"
REVISION="eb157d4ddddc4619e661d4a2426bb2dd000fc668"
W0_COMMIT="4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
TRAIN_SOURCE="fbeece84ff2609e564bb43b7f40ecde1d80ef187"
PRODUCER_RUN=43743251862
STAGE_STEPS=8
SEEDS=(6601,6619)
FAMILIES=("greebles","pipework")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".part")
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    os.replace(temp,path)

def proc_rss(pid):
    try:
        for line in Path("/proc/%d/status"%pid).read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError,ProcessLookupError):
        pass
    return 0

def memory_limit_kib():
    # cgroup-v2 is the real memory ceiling in containerized runners.
    candidates=[]
    for c in ("/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw=Path(c).read_text().strip()
            if raw!="max":
                val=int(raw)//1024
                if 0<val<1<<45:
                    candidates.append(val)
        except (OSError,ValueError):
            pass
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            candidates.append(int(line.split()[1]))
    return min(candidates) if candidates else 0

def admission(w0_path,checkpoint,model_path):
    from common import profile,sha256_file,verify_checkpoint
    from numerics import configure_cpu
    expected=profile("sprocket",require_supported=True)
    numerical,_=configure_cpu(1)
    receipt=Path(model_path)/"download_receipt.json"
    if not receipt.is_file():
        raise ValueError("no original pinned model download receipt")
    if expected["model_id"]!=MODEL_ID or expected["revision"]!=REVISION or expected.get("subfolder","")!="":
        raise ValueError("wrong root-Instruct model identity")
    if not Path(w0_path).is_file():
        raise ValueError("missing pinned W0")
    cfg=verify_checkpoint(checkpoint)["config"]
    required=(cfg.get("schema")=="cybersand.worm.sprocket-family-counterfactual-v6"
       and cfg.get("source_sha")==TRAIN_SOURCE and cfg.get("model")==expected
       and cfg.get("worm_common_git_commit")==W0_COMMIT
       and cfg.get("worm_common_sha256")==sha256_file(w0_path)
       and cfg.get("worm_support_git_commit")==e3.objective.SUPPORT_COMMIT
       and cfg.get("synthetic_not_native") is True and cfg.get("final_opened") is False
       and cfg.get("rank")==8 and cfg.get("threads")==1
       and cfg.get("numerics")==numerical and Path(checkpoint).name=="checkpoint-000024")
    if not required: raise ValueError("original W6 checkpoint admission failed")
    return {"model_id":MODEL_ID,"revision":REVISION,
            "model_download_receipt_sha256":digest(receipt),
            "checkpoint_inventory_sha256":digest(Path(checkpoint)/"inventory.json"),
            "w0_sha256":digest(w0_path),"producer_run":PRODUCER_RUN,
            "producer_conclusion":"failure_exit_137_checkpoint_only",
            "training_source":TRAIN_SOURCE}

def worker(args):
    if args.steps!=STAGE_STEPS or args.family not in FAMILIES or args.seed not in SEEDS:
        raise ValueError("unqualified F1 seed/family/step selection")
    if not args.lease or "/" in args.lease or len(args.lease)>160:
        raise ValueError("unsafe lease")
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    start_utc=time.time()
    start_mono=time.monotonic()
    start_cpu=time.process_time()
    identity={"lease":args.lease,"pid":os.getpid(),"ppid":os.getppid(),
       "start_utc":start_utc,"family":args.family,"seed":args.seed,
       "steps_bound":args.steps,"stage":"WORM-SPROCKET-F1-capacity-inference",
       "native":False,"optimizer_steps":0}
    save(out/"identity.json",identity)
    try:
        pin=admission(args.w0,args.checkpoint,args.model)
        w0=e3.contract.load_w0(args.w0)
        from peft import PeftModel
        import torch
        base,tokenizer=e3.shared.load_base(args.model,e3.profile("sprocket",require_supported=True))
        adapter=e3.adapter_for(e3.profile("sprocket",require_supported=True))
        tokens,mapping=e3.action_tokens(tokenizer)
        tuned=PeftModel.from_pretrained(base,args.checkpoint/"adapter",
                                        is_trainable=False,local_files_only=True)
        e3.BUDGET=args.steps
        e3.STOP_DEFER_BEFORE=args.steps-2
        inference_start_utc=time.time()
        record,receipts=e3.rollout(w0,tuned,tokenizer,adapter,
                                   args.family,args.seed,"trained",tokens)
        inference_end_utc=time.time()
        if not receipts or len(receipts)!=record["steps_executed"]:
            raise ValueError("no genuine model decisions")
        if any(not x["model_forward"] or x["actual_W0_applied"]!=x["host_selected"] or
               not x["selected_host_legal"] or not x["trained_adapter_active"]
               for x in receipts):
            raise ValueError("model/host selected-action receipt mismatch")
        record["mode"]="sprocket-f1-"+args.lease
        record["engine"]="pinned-W0-Python-synthetic-host-not-native"
        w0.write_result(record,out/"original-png-and-json")
        controls={}
        for mode in ("random","wall","fidget"):
            control=w0.run(args.seed,mode,args.family,args.steps)
            w0.write_result(control,out/"original-png-and-json")
            controls[mode]={
                "steps":control["steps_executed"],
                "spatial":e3.spatial(control),
                "metrics":control["metrics"]}
        complete={**identity,**pin,"end_utc":time.time(),
           "wall_seconds":time.monotonic()-start_mono,
           "inference_start_utc":inference_start_utc,
           "inference_end_utc":inference_end_utc,
           "cpu_seconds":time.process_time()-start_cpu,
           "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "trained":{"steps":len(receipts),"invalid":sum(not r["selected_host_legal"] for r in receipts),
                      "early_stop":len(receipts)<args.steps,
                      "spatial":e3.spatial(record),"metrics":record["metrics"],
                      "actions":[r["host_selected"] for r in receipts],
                      "differs_same_observation_flat":sum(r["differs_from_flat"] for r in receipts)},
           "controls":controls,"action_token_mapping":mapping,
           "source_faithful_W0_not_native":True,"raw_eos5_completion_not_measured":True,
           "real_trained_six_score_forwards":len(receipts),"optimizer_steps":0}
        save(out/"complete.json",complete)
        print(json.dumps({"lease":args.lease,"pid":os.getpid(),
                          "real_model_forwards":len(receipts),
                          "wall_seconds":complete["wall_seconds"],
                          "rss_kib":complete["peak_rss_kib"]}),flush=True)
    except BaseException as exc:
        save(out/"error-ledger.json",{"lease":args.lease,"pid":os.getpid(),
             "type":type(exc).__name__,"error":str(exc),"trace":traceback.format_exc(),
             "end_utc":time.time(),"not_success":True})
        raise

def run_group(args,stage,concurrent):
    root=args.output/stage
    root.mkdir(parents=True,exist_ok=False)
    pairs=list(zip(SEEDS,FAMILIES))
    pending=[]
    report=[]
    start=time.monotonic()
    peak_combined=0
    max_active=0
    def spawn(idx):
        seed,family=pairs[idx]
        lease="%s-%s-%d"%(args.lease,stage,idx)
        destination=root/lease
        cmd=[sys.executable,__file__,"--mode","worker",
             "--model",str(args.model),"--w0",str(args.w0),
             "--checkpoint",str(args.checkpoint),"--output",str(destination),
             "--lease",lease,"--family",family,"--seed",str(seed),
             "--steps",str(args.steps)]
        log=open(root/(lease+".log"),"w")
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                                  start_new_session=True)
        row={"lease":lease,"seed":seed,"family":family,"pid":process.pid,
             "start_utc":time.time(),"status":"active","log":log,
             "process":process,"path":destination}
        pending.append(row)
        return row
    try:
        if concurrent:
            spawn(0);spawn(1)
        else:
            spawn(0)
        next_idx=1 if not concurrent else 2
        while pending:
            now=time.monotonic()
            if now-start>args.deadline:
                raise TimeoutError("%s wall deadline"%stage)
            active=[p for p in pending if p["process"].poll() is None]
            max_active=max(max_active,len(active))
            total_rss=sum(proc_rss(p["pid"]) for p in active)
            peak_combined=max(peak_combined,total_rss)
            # Cancel before runner OOM, protect one full host safety margin.
            if args.rss_limit_kib and total_rss>args.rss_limit_kib:
                raise MemoryError("%s combined RSS %s exceeded %s"%
                                  (stage,total_rss,args.rss_limit_kib))
            for p in list(pending):
                code=p["process"].poll()
                if code is None: continue
                p["log"].close()
                p["end_utc"]=time.time()
                p["exit_code"]=code
                p["status"]="completed" if code==0 and (p["path"]/"complete.json").is_file() else "failed"
                report.append({k:p[k] for k in ("lease","seed","family","pid",
                                                "start_utc","end_utc","exit_code","status")})
                pending.remove(p)
                if code: raise RuntimeError("worker %s exited %s"%(p["lease"],code))
                if not concurrent and next_idx<len(pairs):
                    spawn(next_idx)
                    next_idx+=1
            time.sleep(.5)
        end=time.monotonic()
        if len(report)!=2 or any(x["status"]!="completed" for x in report):
            raise AssertionError("missing residency worker")
        return {"stage":stage,"wall_seconds":end-start,
                "peak_combined_rss_kib_sampled":peak_combined,
                "maximum_simultaneous_live_processes":max_active,
                "workers":report,
                "real_forwards":sum(json.loads((root/p["lease"]/"complete.json").read_text())
                                    ["real_trained_six_score_forwards"] for p in report)}
    except BaseException:
        for p in pending:
            p["status"]="cancelled"
            try:os.killpg(p["process"].pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for p in pending:
            try:p["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(p["process"].pid,signal.SIGKILL)
                except ProcessLookupError:pass
            p["log"].close()
        raise

def benchmark(args):
    def cancel(sig,frame):
        raise InterruptedError("host cancellation/timeout signal %d"%sig)
    signal.signal(signal.SIGTERM,cancel)
    signal.signal(signal.SIGINT,cancel)
    if args.steps!=STAGE_STEPS or args.deadline<=0:
        raise ValueError("F1 is only 8-step original-image residency preflight")
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.time()
    mem=memory_limit_kib()
    # Cap at 70% of actual memory, with a further hard ceiling of 12GiB.
    args.rss_limit_kib=min(12*1024**2,int(.7*mem)) if mem else 8*1024**2
    ledger={"schema":"sprocket-worm-f1-residency/1","source_sha":os.environ.get("GITHUB_SHA"),
            "run_id":os.environ.get("GITHUB_RUN_ID"),"lease":args.lease,
            "max_optimizers":0,"model":MODEL_ID,"revision":REVISION,
            "checkpoint":str(args.checkpoint),"w0_commit":W0_COMMIT,
            "producer_failed_exit137_preserved":True,
            "sealed_final_opened":False,"native":False,"bound_future":128,
            "max_approved_decisions_now":STAGE_STEPS,
            "memory_total_kib":mem,"rss_safety_ceiling_kib":args.rss_limit_kib,
            "start_utc":start,"status":"in_progress"}
    save(args.output/"lease.json",ledger)
    try:
        # One actual process at a time, then two actual simultaneously resident
        # model instances, same seeds/observations/checkpoint/CPU.
        serial=run_group(args,"one_resident_serial",False)
        save(args.output/"serial.json",serial)
        dual=run_group(args,"two_resident_concurrent",True)
        if dual["maximum_simultaneous_live_processes"]!=2:
            raise ValueError("never observed two simultaneous model processes")
        if serial["real_forwards"]!=dual["real_forwards"]:
            raise ValueError("different decision budget")
        if dual["peak_combined_rss_kib_sampled"]>args.rss_limit_kib:
            raise ValueError("dual inference exceeds safety memory limit")
        throughput_gain=serial["wall_seconds"]/dual["wall_seconds"]
        qualified=throughput_gain>=.70
        result={**ledger,"status":"qualified_two_resident" if qualified else "two_resident_too_slow",
                "end_utc":time.time(),"serial":serial,"dual":dual,
                "throughput_gain":throughput_gain,
                "qualified_two_resident":qualified,
                "decision":"2x2 possible only after this receipt and separately qualified four-live concurrency"
                  if qualified else "use isolated single-model runners; do not assume 2x2",
                "original_pngs_each_worker":4}
        save(args.output/"benchmark.json",result)
        print(json.dumps({"qualified_two_resident":qualified,
                          "real_forwards_each_stage":serial["real_forwards"],
                          "serial_wall":serial["wall_seconds"],
                          "dual_wall":dual["wall_seconds"],
                          "dual_sampled_rss_kib":dual["peak_combined_rss_kib_sampled"],
                          "throughput_gain":throughput_gain}),flush=True)
        if not qualified: raise RuntimeError("dual model residency fails throughput qualification")
    except BaseException as exc:
        save(args.output/"error-ledger.json",{"status":"FAILED",
             "type":type(exc).__name__,"error":str(exc),"trace":traceback.format_exc(),
             "end_utc":time.time(),"no_four_way_admission":True})
        raise
    finally:
        checks={}
        for f in sorted(args.output.rglob("*")):
            if f.is_file() and f.name!="SHA256SUMS.json":
                checks[str(f.relative_to(args.output))]=digest(f)
        save(args.output/"SHA256SUMS.json",checks)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("bench","worker"),required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--lease",required=True)
    p.add_argument("--family",default="greebles")
    p.add_argument("--seed",type=int,default=6601)
    p.add_argument("--steps",type=int,default=STAGE_STEPS)
    p.add_argument("--deadline",type=int,default=2100)
    args=p.parse_args()
    if args.mode=="worker":worker(args)
    else:benchmark(args)
