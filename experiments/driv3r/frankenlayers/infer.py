#!/usr/bin/env python3
"""Real d1 Q8 System One inference through deliberately mangled layer flows.

No native Workbench writes. Same immutable Q8 weights, same W0 legal-action host,
same custom llama.cpp binary; only CYBERSAND_D1_FRANKEN_FLOW differs.
"""
from __future__ import annotations
import argparse, hashlib, http.client, importlib.util, json, os
from pathlib import Path
import secrets, socket, statistics, subprocess, tempfile, time

from experiments.agent_painter.process import stop_group
from experiments.driv3r.backend import _child_environment, parse_choice_response, sha256_file, validate_questions
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS

SEED=570
FAMILIES=("greebles","pipework","architecture")
STEPS=16
ACTION_DESCRIPTIONS={
    "STRAIGHT":"Continue straight one legal cell.",
    "LEFT":"Turn left and extend one legal cell.",
    "RIGHT":"Turn right and extend one legal cell.",
    "BRANCH":"Make one host-approved branch into an open arm.",
    "RETRACE":"Lift brush and return to a prior position without erasing.",
    "STOP":"End this worm drawing now.",
}

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")

def percentile(values,p):
    values=sorted(values)
    if not values:return None
    x=(len(values)-1)*p
    lo=int(x); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)

class Server:
    def __init__(self,binary:Path,model:Path,mode:str,root:Path):
        self.binary=binary.resolve(); self.model=model.resolve(); self.mode=mode; self.root=root
        self.proc=None; self.log=None; self.tmp=None; self.key=None; self.port=None; self.startup=None
    def __enter__(self):
        self.root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix="franken-d1-key-")
        keyfile=Path(self.tmp.name)/"key"; self.key=secrets.token_hex(32)
        keyfile.write_text(self.key+"\n"); keyfile.chmod(0o600)
        with socket.socket() as s:
            s.bind(("127.0.0.1",0)); self.port=s.getsockname()[1]
        cmd=[str(self.binary),"-m",str(self.model),"--host","127.0.0.1",
             "--port",str(self.port),"--api-key-file",str(keyfile),
             "--alias","cybersand-franken-d1","-t","2","-tb","2","-ngl","0","-np","1",
             "-c","4096","-b","2048","-ub","2048","--no-mmproj","--timeout","120","--metrics"]
        env=_child_environment(self.binary.parent,2)
        env["CYBERSAND_D1_FRANKEN_FLOW"]=self.mode
        self.log=(self.root/"server.log").open("xb")
        t0=time.monotonic()
        self.proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT,
                                   start_new_session=True,cwd=self.binary.parent,env=env)
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited {self.proc.returncode}")
                try:
                    status,body=self.request("GET","/health",None,2)
                    if status==200:
                        status,models=self.request("GET","/v1/models",None,2)
                        if status==200:
                            break
                except Exception:
                    pass
                time.sleep(.15)
            else: raise TimeoutError("Franken d1 startup timeout")
            self.startup={"mode":self.mode,"pid":self.proc.pid,"seconds":time.monotonic()-t0,
                          "binary_sha256":sha256_file(self.binary),"model_sha256":sha256_file(self.model)}
            write_json(self.root/"startup.json",self.startup)
            return self
        except BaseException:
            self.__exit__(None,None,None); raise
    def request(self,method,path,data,timeout):
        conn=http.client.HTTPConnection("127.0.0.1",self.port,timeout=timeout)
        try:
            headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"}
            conn.request(method,path,body=data,headers=headers)
            r=conn.getresponse(); raw=r.read(512001)
            if len(raw)>512000: raise RuntimeError("model response too large")
            return r.status,json.loads(raw)
        finally: conn.close()
    def choose(self,obs,records):
        legal=obs["legal"]; criteria={a:ACTION_DESCRIPTIONS[a] for a in legal}
        state={"identity":"small slightly deranged CyberSand drawing worm",
               "objective":"make an odd continuing line, not optimal pathfinding",
               "seed":SEED,"family":obs["family"],"tick":obs["tick"],
               "x":obs["x"],"y":obs["y"],"heading":obs["heading"],
               "local_5x5":obs["local_5x5"],"last_actions":obs["last_actions"],
               "legal_actions":legal}
        q={"worm_move":{"type":"choice",
           "instructions":"Choose exactly one legal action. Be a curious drawing worm. Weird is allowed. Never invent actions.",
           "criteria":criteria}}
        payload=validate_questions(json.dumps(state,sort_keys=True,separators=(",",":")),q)
        t0=time.monotonic(); status,raw=self.request("POST","/v1/systemone",payload,95); dt=time.monotonic()-t0
        rec={"observation":obs,"state":state,"question":q,"http_status":status,
             "latency_seconds":dt,"raw_response":raw}
        if status!=200:
            rec["status"]="http_failure"; records.append(rec); raise RuntimeError("systemone HTTP failure")
        parsed=parse_choice_response(raw,"worm_move",list(criteria))
        usage=raw.get("usage",{})
        rec.update({"status":"ok","parsed":parsed,"choice":parsed["candidate_id"],
                    "input_tokens":usage.get("input_tokens"),
                    "input_tokens_per_second":usage.get("input_tokens")/dt if usage.get("input_tokens") else None})
        records.append(rec)
        return parsed["candidate_id"]
    def __exit__(self,*_):
        if self.proc is not None: stop_group(self.proc)
        if self.log is not None:self.log.close()
        if self.tmp is not None:self.tmp.cleanup()

def load_w0(path):
    spec=importlib.util.spec_from_file_location("franken_w0",path)
    if not spec or not spec.loader: raise RuntimeError("W0 import failed")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    if tuple(m.ACTIONS)!=tuple(ACTION_DESCRIPTIONS): raise ValueError("W0 action contract changed")
    return m

def run_arm(binary,model,w0,mode,out):
    calls=[]; experiments=[]
    t0=time.monotonic()
    with Server(binary,model,mode,out/"process") as server:
        for family in FAMILIES:
            family_calls=[]
            record=w0.run(SEED,mode="callback",family=family,steps=STEPS,
                          selector=lambda obs:server.choose(obs,family_calls))
            png=w0.png_bytes(record)
            (out/f"{mode}-{family}.png").write_bytes(png)
            write_json(out/f"{mode}-{family}.json",record)
            write_json(out/f"{mode}-{family}-choices.json",family_calls)
            calls.extend(family_calls)
            experiments.append({
                "family":family,
                "raw_choices":[c["choice"] for c in family_calls],
                "applied_choices":[e["applied"] for e in record["events"]],
                "decisions":len(family_calls),
                "metrics":record["metrics"],
                "png_sha256":hashlib.sha256(png).hexdigest(),
            })
    lat=[c["latency_seconds"] for c in calls if c.get("status")=="ok"]
    tps=[c["input_tokens_per_second"] for c in calls if c.get("input_tokens_per_second")]
    choices=[c["choice"] for c in calls if c.get("status")=="ok"]
    return {
        "mode":mode,"flow":list(FLOWS[mode]),"executed_layer_steps":len(FLOWS[mode]),
        "unique_source_layers":len(set(FLOWS[mode])),
        "skipped_source_layers":24-len(set(FLOWS[mode])),
        "duplicate_executions":len(FLOWS[mode])-len(set(FLOWS[mode])),
        "model_decisions":len(calls),"valid_decisions":len(lat),
        "mean_seconds":statistics.mean(lat) if lat else None,
        "p50_seconds":statistics.median(lat) if lat else None,
        "p95_seconds":percentile(lat,.95),
        "median_input_tokens_per_second":statistics.median(tps) if tps else None,
        "choice_counts":{a:choices.count(a) for a in ACTION_DESCRIPTIONS},
        "experiments":experiments,"wall_seconds":time.monotonic()-t0,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--mode",choices=tuple(x for x in FLOWS if x not in ("stock","scrambled_eggs")),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source-sha",required=True)
    a=p.parse_args()
    from experiments.driv3r.backend import PINS
    if sha256_file(a.model)!=PINS["model"]["sha256"]: raise ValueError("Q8 donor bytes changed")
    if a.output.exists(): raise RuntimeError("Refuse evidence overwrite")
    a.output.mkdir(parents=True)
    w0=load_w0(a.w0)
    stock=run_arm(a.binary,a.model,w0,"stock",a.output/"stock")
    mutant=run_arm(a.binary,a.model,w0,a.mode,a.output/a.mode)
    report={"schema":"cybersand-d1-frankenlayers-yolo/1","source_sha":a.source_sha,
            "q8_sha256":PINS["model"]["sha256"],"llama_source_commit":"8642541b255bfb5dc39cd351a178be09013a55d6",
            "mode":a.mode,"stock":stock,"mutant":mutant,
            "speedup_p50":stock["p50_seconds"]/mutant["p50_seconds"] if mutant["p50_seconds"] else None,
            "actual_systemone_inference":True,"native_workbench":False,"training":False}
    write_json(a.output/"comparison.json",report)
    print(json.dumps({"mode":a.mode,"speedup_p50":report["speedup_p50"],
                      "stock_p50":stock["p50_seconds"],"mutant_p50":mutant["p50_seconds"],
                      "mutant_choices":mutant["choice_counts"]},sort_keys=True))

if __name__=="__main__":
    main()
