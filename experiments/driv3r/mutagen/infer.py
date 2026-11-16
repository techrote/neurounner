#!/usr/bin/env python3
"""Source-faithful W0 worm + real resident d1 System One inference for d1-MUTAGEN.

Runs the unchanged, pinned legal-action W0 host in callback mode. Every steering
choice is an original /v1/systemone typed decision from stock or mutated d1 GGUF.
The PNGs are original Python W0 renderings, NOT native Godot PAINT or screenshots.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import statistics
import subprocess
import tempfile
import time
import traceback

from experiments.driv3r.backend import (
    _child_environment, parse_choice_response, sha256_file, validate_questions,
)
from experiments.agent_painter.process import stop_group

SEED = 570
ACTION_DESCRIPTIONS = {
    "STRAIGHT":"Continue straight ahead for one legal cell, extending your odd ink trail.",
    "LEFT":"Turn left one quarter-turn and extend the trail by one legal cell.",
    "RIGHT":"Turn right one quarter-turn and extend the trail by one legal cell.",
    "BRANCH":"Make a host-approved branch from the current position, choosing an open arm.",
    "RETRACE":"Lift the brush and return to a previous position without erasing ink.",
    "STOP":"Finish the worm's drawing if no interesting direction remains.",
}


def json_write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n")


class D1Server:
    def __init__(self, binary:Path, model:Path, root:Path):
        self.binary, self.model, self.root = binary, model, root
        self.proc=None
        self.port=None
        self.key=None
        self.temporary=None
        self.log=None
        self.startup={}

    def __enter__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="d1-rusty-spoon-key-")
        secret=Path(self.temporary.name)/"key"
        self.key=secrets.token_hex(32)
        secret.write_text(self.key+"\n")
        secret.chmod(0o600)
        with socket.socket() as s:
            s.bind(("127.0.0.1",0))
            self.port=s.getsockname()[1]
        cmd=[str(self.binary),"-m",str(self.model),"--host","127.0.0.1",
             "--port",str(self.port),"--api-key-file",str(secret),
             "--alias","d1-m0-surgery-worm","-t","2","-tb","2",
             "-ngl","0","-np","1","-c","4096","-b","2048","-ub","2048",
             "--no-mmproj","--timeout","120","--metrics"]
        self.root.mkdir(parents=True,exist_ok=True)
        self.log=(self.root/"llama-server.log").open("xb")
        t0=time.monotonic()
        self.proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,
             stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True,
             cwd=self.binary.parent,env=_child_environment(self.binary.parent,2))
        self.startup={"pid":self.proc.pid,"model_sha256":sha256_file(self.model),
                       "port":self.port,"status":"starting"}
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"Resident d1 exited early code={self.proc.returncode}")
                try:
                    status, response=self.request("GET","/health",None,2)
                    if status==200:
                        status,models=self.request("GET","/v1/models",None,2)
                        if status==200 and any(v.get("id")=="d1-m0-surgery-worm"
                                               for v in models.get("data",[])):
                            self.startup["status"]="ready"
                            break
                except (OSError,TimeoutError,http.client.HTTPException,ValueError):
                    pass
                time.sleep(.12)
            else:
                raise TimeoutError("Pinned mutant d1 server readiness deadline")
            self.startup["start_seconds"]=time.monotonic()-t0
            json_write(self.root/"startup.json",self.startup)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def request(self,method,path,data,deadline):
        conn=http.client.HTTPConnection("127.0.0.1",self.port,timeout=deadline)
        try:
            headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"}
            conn.request(method,path,body=data,headers=headers)
            reply=conn.getresponse()
            content=reply.read(512_001)
            if len(content)>512_000:
                raise RuntimeError("Unbounded model response")
            return reply.status,json.loads(content)
        finally:
            conn.close()

    def choose(self, observation, records):
        legal=observation["legal"]
        criteria={a:ACTION_DESCRIPTIONS[a] for a in legal}
        state={"identity":"Small whimsical WORM-001 material trail creature",
               "objective":"Make a lively, slightly deranged but sometimes purposeful line.",
               "seed":getattr(self, "world_seed", SEED),"x":observation["x"],"y":observation["y"],
               "heading":observation["heading"],
               "local_5x5":observation["local_5x5"],
               "last_actions":observation["last_actions"],
               "family":observation["family"],"tick":observation["tick"],
               "legal_actions":legal}
        question={"worm_move":{"type":"choice",
               "instructions":"Select one legal action only. Continue a curious trail with occasional turns, hesitations and branches. You have no destination; STOP is allowed. Never invent an unavailable action.",
               "criteria":criteria}}
        text_state=json.dumps(state,sort_keys=True,separators=(",",":"))
        payload=validate_questions(text_state,question)
        t0=time.monotonic()
        status,raw=self.request("POST","/v1/systemone",payload,95)
        latency=time.monotonic()-t0
        receipt={"observation":observation,"question":question,
                 "state":state,"raw_response":raw,
                 "http_status":status,"latency_seconds":latency}
        if status!=200:
            receipt["status"]="http_failure"
            records.append(receipt)
            raise RuntimeError("d1 typed directional request refused")
        try:
            parsed=parse_choice_response(raw,"worm_move",list(criteria))
            receipt.update({"status":"ok","parsed":parsed,
                            "original_selected_action":parsed["candidate_id"]})
            records.append(receipt)
            return parsed["candidate_id"]
        except BaseException as exc:
            receipt["status"]="original_response_invalid"
            receipt["error"]=str(exc)
            records.append(receipt)
            raise

    def __exit__(self,*_):
        if self.proc is not None:
            stop_group(self.proc)
        if self.log is not None:
            self.log.close()
        if self.temporary is not None:
            self.temporary.cleanup()
        self.key=None
        self.startup.update({"exited":self.proc.returncode if self.proc else None,
                             "process_reaped":self.proc is None or self.proc.poll() is not None})
        if self.root.exists():
            json_write(self.root/"shutdown.json",self.startup)


def load_worm(path):
    spec=importlib.util.spec_from_file_location("frozen_d1_m0_w0",path)
    if not spec or not spec.loader:
        raise RuntimeError("Pinned W0 cannot be imported")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.ACTIONS)!=tuple(ACTION_DESCRIPTIONS):
        raise ValueError("W0 legal actions diverged")
    return module


def arm(runtime, donor_or_mutant, label, worm, out, steps, families):
    from experiments.driv3r.backend import PINS
    binary=Path(runtime["binary"])
    assert sha256_file(binary)==PINS["runtime"]["binary_sha256"]
    records=[]
    results=[]
    last_start=time.monotonic()
    with D1Server(binary,donor_or_mutant,out/"model-process") as server:
        for family in families:
            calls=[]
            try:
                record=worm.run(SEED,mode="callback",family=family,
                                steps=steps,selector=lambda obs:server.choose(obs,calls))
                rendered=worm.png_bytes(record)
                name=f"{label}-{family}-{SEED}"
                (out/(name+".png")).write_bytes(rendered)
                json_write(out/(name+".json"),record)
                result={"name":name,"family":family,
                        "actual_model_decisions":len(calls),
                        "metrics":record["metrics"],
                        "raw_choices":[r.get("original_selected_action") for r in calls],
                        "applied_choices":[e["applied"] for e in record["events"]],
                        "png_sha256":hashlib.sha256(rendered).hexdigest(),
                        "completed":True}
            except BaseException as err:
                result={"name":f"{label}-{family}-{SEED}","family":family,
                        "completed":False,"error":f"{type(err).__name__}: {err}",
                        "traceback":traceback.format_exc()[-4000:],
                        "actual_model_decisions":len(calls)}
            json_write(out/f"{label}-{family}-{SEED}-model-choices.json",calls)
            results.append(result)
            print(json.dumps({"event":"worm-directional-result","label":label,
                              "family":family,"completed":result["completed"],
                              "decisions":result["actual_model_decisions"]}),flush=True)
    latencies=[]
    for family in families:
        p=out/f"{label}-{family}-{SEED}-model-choices.json"
        latencies.extend(r["latency_seconds"] for r in json.loads(p.read_text())
                         if r.get("status")=="ok")
    summary={"schema":"driv3r-d1-m0-inference/1","model_sha256":sha256_file(donor_or_mutant),
             "label":label,"not_native":True,
             "real_gguf_model_inference":True,"training":False,"seed":SEED,
             "families":list(families),"budget_per_family":steps,
             "experiments":results,"total_legal_original_choices":len(latencies),
             "mean_inference_seconds":statistics.mean(latencies) if latencies else None,
             "p50_inference_seconds":statistics.median(latencies) if latencies else None,
             "wall_seconds":time.monotonic()-last_start}
    json_write(out/"summary.json",summary)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runtime-manifest",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--mode",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise RuntimeError("No overwriting an immutable inference evidence root")
    args.output.mkdir(parents=True)
    runtime=json.loads(args.runtime_manifest.read_text())
    worm=load_worm(args.w0)
    from experiments.driv3r.backend import PINS
    if sha256_file(args.original)!=PINS["model"]["sha256"]:
        raise ValueError("Original reference Q8 bytes no longer match pinned DRIV3R donor")
    if sha256_file(args.model)==PINS["model"]["sha256"]:
        raise ValueError("Mutation had no effect; refusing false model result")
    identity={"mode":args.mode,"original_sha256":PINS["model"]["sha256"],
              "mutant_sha256":sha256_file(args.model),
              "original":str(args.original),"mutant":str(args.model),
              "source_sha":os.environ.get("GITHUB_SHA"),
              "evidence_status":"inference_not_yet_run"}
    json_write(args.output/"run-identity.json",identity)
    baseline=arm(runtime,args.original,"stock",worm,args.output/"stock",
                 steps=4,families=("greebles",))
    mutant=arm(runtime,args.model,args.mode,worm,args.output/args.mode,
               steps=8,families=tuple(worm.FAMILIES))
    result={"schema":"driv3r-d1-mutagen-m0/1","mode":args.mode,
            "baseline":baseline,"mutant":mutant,
            "status":"both_executed" if baseline["total_legal_original_choices"] and
                 mutant["total_legal_original_choices"] else "partial_or_failed",
            "actual_native_workbench":False,"sealed_final_opened":False}
    json_write(args.output/"comparison.json",result)
    print(json.dumps({"event":"mutagen-directional-inference-completed",
                      "mode":args.mode,"status":result["status"],
                      "stock_choices":baseline["total_legal_original_choices"],
                      "mutant_choices":mutant["total_legal_original_choices"]}),flush=True)


if __name__=="__main__":
    main()
