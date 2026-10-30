#!/usr/bin/env python3
"""DETECTIVE-001B4: pinned LFM2.5-1.2B conventional scripting control."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, re, resource, subprocess, sys, time, urllib.request
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.detective_001_contract import ContractError, execute_program, parse_program  # noqa
SPEC=ROOT/"experiments/detective-001/lfm-scripter.json"; CASE=ROOT/"experiments/detective-001/pilot-case.json"
PINS=ROOT/"config/pins.env"

def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def sha_obj(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def write_json(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
def load_env(p):
    out={}
    for line in Path(p).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            k,v=line.split("=",1); out[k]=v
    return out
def pairs(items):
    out={}
    for k,v in items:
        if k in out: raise ContractError("duplicate JSON key")
        out[k]=v
    return out
def finite(x):
    v=float(x)
    if not math.isfinite(v): raise ContractError("nonfinite JSON")
    return v
def reject_const(x): raise ContractError("nonfinite literal "+x)

def load_spec():
    s,c=read_json(SPEC),read_json(CASE); pins=load_env(PINS)
    if (s.get("schema"),s.get("experiment"),s.get("parent_issue"))!=("detective-lfm-scripter-v1","DETECTIVE-001B4",109):
        raise ContractError("identity drift")
    m=s["model"]; r=s["runtime"]
    checks={"LFM_REPO":m["repo"],"LFM_REV":m["revision"],"LFM_FILE":m["file"],"LFM_SHA256":m["sha256"],
            "LLAMA_TAG":r["tag"],"LLAMA_COMMIT":r["commit"],"LLAMA_SHA256":r["archive_sha256"]}
    for k,v in checks.items():
        if pins.get(k)!=v: raise ContractError("pin drift "+k)
    if [x["id"] for x in s["tasks"]] != ["L01","L02","L03","L04","L05"]: raise ContractError("task drift")
    if s["decode"]!={"seed":24301,"max_tokens":96,"temperature":0.0,"top_p":1.0}: raise ContractError("decode drift")
    if s["limits"]!={"tasks":5,"max_generated_tokens":480,"threads":4,"context":2048,"max_job_minutes":15,"training_steps":0,"paid_api_calls":0}: raise ContractError("budget drift")
    if s["comparison"]!={"sprocket_dataset":"datasets/detective-001-sprocket-scripter","wire_matched":False}: raise ContractError("comparison relabeled")
    return s,c,pins

def script_prompt(task):
    return (
      "You control a FICTIONAL detective-game evidence interface. "
      "Return ONLY one JSON object, no markdown or prose. Exact outer schema: "
      '{"schema":"detective-script-v1","steps":[...]}. '
      "Allowed step ops and exact fields: "
      'read_asset(asset_id), correlate_assets(left_id,right_id), '
      'probe_terminal(terminal_id,credential). '
      "Currently accessible evidence IDs are dock-clock-ledger and transfer-card. "
      "Do not invent IDs, paths, shell commands, network actions, or credentials. "
      "If no game action is required, steps must be an empty array. "
      "Player request: "+task["request"]
    )
def prompt(task):
    if task["mode"]=="script": return script_prompt(task)
    return (
      "You are reporting an AUTHORITATIVE result from a FICTIONAL detective-game tool. "
      "Do not invent additional case facts, tools, files, people, or hidden evidence. "
      "Answer in plain text, briefly. "+task["request"]
    )
def normalize_script(raw):
    try:
        obj=json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=reject_const)
    except Exception as e: raise ContractError("strict JSON parse failed") from e
    parse_program(obj); return obj
def same_step(got,want):
    if got==want:return True
    if want and got.get("op")==want.get("op")=="correlate_assets":
        return {got.get("left_id"),got.get("right_id")}=={want["left_id"],want["right_id"]}
    return False
def score(s,case,task,raw):
    base={"task_id":task["id"],"mode":task["mode"],"raw_sha256":hashlib.sha256(raw.encode()).hexdigest()}
    if task["mode"]=="script":
        try:
            program=normalize_script(raw)
            steps=program["steps"]; expected=task["expected_step"]
            correct=(steps==[] if expected is None else len(steps)==1 and same_step(steps[0],expected))
            receipt=execute_program(case,program,["dock-clock-ledger","transfer-card"])
            return {**base,"schema_valid":True,"selection_correct":bool(correct),"program":program,"receipt":receipt}
        except Exception as e:
            return {**base,"schema_valid":False,"selection_correct":False,"error_type":type(e).__name__,"error":str(e)[:180]}
    lo=raw.casefold()
    hits=[x for x in task["expected_markers"] if x.casefold() in lo]
    bad=[x for x in task["forbidden_markers"] if x.casefold() in lo]
    return {**base,"result_use_success":bool(raw.strip()) and len(hits)==len(task["expected_markers"]) and not bad,
            "expected_marker_hits":hits,"forbidden_marker_hits":bad}

def plan():
    s,c,p=load_spec()
    return {"schema":"detective-lfm-control-plan-v1","spec_sha256":sha_file(SPEC),"case_sha256":sha_file(CASE),
            "model":s["model"],"runtime":s["runtime"],"tasks":[{"id":t["id"],"mode":t["mode"],"prompt":prompt(t),"prompt_sha256":hashlib.sha256(prompt(t).encode()).hexdigest()} for t in s["tasks"]],
            "limits":s["limits"],"wire_matched":False}

def save_partial(out,run):
    raw=canon(run).encode()
    if len(raw)>160000: raise ContractError("raw cap")
    write_json(Path(out)/"raw/run.json",run)

def archive(out,run,s):
    d=Path(out)/"dataset"; (d/"raw").mkdir(parents=True,exist_ok=True); (d/"provenance").mkdir(parents=True,exist_ok=True)
    write_json(d/"raw/run.json",run)
    summary={"tasks":5,"schema_valid":sum(x.get("schema_valid") is True for x in run["results"]),
             "selection_correct":sum(x.get("selection_correct") is True for x in run["results"]),
             "result_use_success":sum(x.get("result_use_success") is True for x in run["results"]),
             "generated_tokens":run["total_generated_tokens"],"training_steps":0}
    write_json(d/"manifest.json",{"schema":"detective-lfm-scripter-archive-v1","experiment":"DETECTIVE-001B4","admission":"EXPLORATORY_NOT_ADMITTED",
      "run_sha256":sha_obj(run),"spec_sha256":sha_file(SPEC),"plan_sha256":run["plan_sha256"],"source_head_sha":run["repository"]["head_sha"],"summary":summary})
    write_json(d/"provenance/source-artifacts.json",{"repository":"techrote/neurounner","head_sha":run["repository"]["head_sha"],
      "workflow_run_id":run["repository"]["run_id"],"workflow_run_url":run["repository"]["run_url"],
      "model":s["model"],"runtime":s["runtime"],"model_sha256_verified":run["model_sha256_verified"],
      "llama_version":run["llama_version"],"script_sha256":run["script_sha256"],"case_sha256":run["case_sha256"]})
    (d/"README.md").write_text("# DETECTIVE-001B4 LFM conventional scripter control\n\nPinned LFM2.5-1.2B-Instruct QAD GGUF on five fictional detective semantic task families. This is a task-level, not wire-matched, control against Sprocket. Model output is inert text and only strict detective-script-v1 JSON enters the synthetic authority. No training, paid API, real host scripting, or production admission.\n",encoding="utf-8")
    rows=[]
    for f in sorted(x for x in d.rglob("*") if x.is_file()):
        rel=f.relative_to(d).as_posix()
        if rel!="provenance/hashes.sha256": rows.append(f"{sha_file(f)}  {rel}")
    (d/"provenance/hashes.sha256").write_text("\n".join(rows)+"\n",encoding="utf-8")
    return d

def server_chat(port,prompt_text,s):
    body={"model":"local","messages":[{"role":"user","content":prompt_text}],"max_tokens":s["decode"]["max_tokens"],
          "temperature":s["decode"]["temperature"],"top_p":s["decode"]["top_p"],"seed":s["decode"]["seed"],"stream":False}
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"},method="POST")
    t=time.monotonic()
    with urllib.request.urlopen(req,timeout=120) as resp: payload=json.loads(resp.read())
    elapsed=time.monotonic()-t; msg=payload["choices"][0]["message"]; usage=payload.get("usage") or {}
    return {"request":body,"response":msg.get("content") or "","reasoning_content":msg.get("reasoning_content") or "",
            "completion_tokens":int(usage.get("completion_tokens") or 0),"elapsed_seconds":elapsed,
            "finish_reason":payload["choices"][0].get("finish_reason"),"raw_api":payload}

def run(out):
    s,case,pins=load_spec(); paths=load_env(Path(os.environ.get("NEURO_CACHE",ROOT/".cache/neurounner"))/"paths.env")
    model=Path(paths["LFM_MODEL"]); server=Path(paths["LLAMA_SERVER"])
    if sha_file(model)!=s["model"]["sha256"]: raise ContractError("model SHA mismatch")
    ver=subprocess.run([str(server),"--version"],capture_output=True,text=True,check=False).stdout.strip()
    port=18431; log=Path(out)/"server.log"; log.parent.mkdir(parents=True,exist_ok=True); fh=log.open("w")
    proc=subprocess.Popen([str(server),"-m",str(model),"--host","127.0.0.1","--port",str(port),"-t","4","-c","2048"],
                          stdout=fh,stderr=subprocess.STDOUT,text=True)
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if proc.poll() is not None: raise RuntimeError("llama-server exited")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=1) as rr:
                    if rr.status==200: break
            except Exception: time.sleep(.25)
        else: raise RuntimeError("server startup timeout")
        rid=os.environ.get("GITHUB_RUN_ID"); head=os.environ.get("DETECTIVE_HEAD_SHA") or os.environ.get("GITHUB_SHA") or "local"
        run={"schema":"detective-lfm-scripter-run-v1","experiment":"DETECTIVE-001B4","admission":"EXPLORATORY_NOT_ADMITTED",
             "complete":False,"plan_sha256":sha_obj(plan()),"spec_sha256":sha_file(SPEC),"case_sha256":sha_file(CASE),
             "script_sha256":sha_file(Path(__file__)),"repository":{"head_sha":head,"run_id":rid,"run_url":f"https://github.com/techrote/neurounner/actions/runs/{rid}" if rid else None},
             "model":s["model"],"runtime_pin":s["runtime"],"model_sha256_verified":sha_file(model),"llama_version":ver,
             "host":{"platform":platform.platform(),"cpu_count":os.cpu_count()},"raw":[],"results":[],"total_generated_tokens":0}
        save_partial(out,run)
        for task in s["tasks"]:
            got=server_chat(port,prompt(task),s)
            if run["total_generated_tokens"]+got["completion_tokens"]>s["limits"]["max_generated_tokens"]: raise ContractError("token budget")
            rec={"task_id":task["id"],"mode":task["mode"],"prompt":prompt(task),"prompt_sha256":hashlib.sha256(prompt(task).encode()).hexdigest(),**got}
            run["raw"].append(rec); run["results"].append(score(s,case,task,got["response"])); run["total_generated_tokens"]+=got["completion_tokens"]; save_partial(out,run)
        run["complete"]=True; save_partial(out,run); d=archive(out,run,s)
        return {"dataset":str(d),"tokens":run["total_generated_tokens"],"schema_valid":sum(x.get("schema_valid") is True for x in run["results"]),
                "selection_correct":sum(x.get("selection_correct") is True for x in run["results"]),"result_use_success":sum(x.get("result_use_success") is True for x in run["results"])}
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(5)
            except subprocess.TimeoutExpired: proc.kill()
        fh.close()

def main():
    a=argparse.ArgumentParser(); a.add_argument("--mode",choices=["check","run"],default="check"); a.add_argument("--out",type=Path,default=Path("out/detective-001b4")); x=a.parse_args()
    print(json.dumps(plan() if x.mode=="check" else run(x.out),indent=2,sort_keys=True))
if __name__=="__main__": main()
