#!/usr/bin/env python3
from __future__ import annotations
import json, os, platform, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(os.environ.get("NEURO_CACHE",ROOT/".cache"/"neurounner"))
RESULTS=ROOT/"results"
LOGS=ROOT/"logs"

def env(path):
    d={}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            k,v=line.split("=",1); d[k]=v
    return d

def rss(pid):
    p=Path(f"/proc/{pid}/status")
    if not p.exists(): return None
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("VmRSS:"): return int(line.split()[1])
    return None

def cpu():
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"): return line.split(":",1)[1].strip()
    return "unknown"

def extract_json(text):
    a,b=text.find("{"),text.rfind("}")
    if a<0 or b<=a: return None
    try:
        x=json.loads(text[a:b+1]); return x if isinstance(x,dict) else None
    except json.JSONDecodeError: return None

def main():
    if len(sys.argv)!=3:
        raise SystemExit("usage: panel_judge.py MODEL_ID MODEL_PATH")
    model_id=sys.argv[1]; model=Path(sys.argv[2]).resolve()
    specs=json.loads((ROOT/"config/candidates.json").read_text())
    spec=specs[model_id]
    population=json.loads((ROOT/"experiments/panel-001/population.json").read_text())
    paths=env(CACHE/"paths.env"); server=Path(paths["LLAMA_SERVER"])
    port=18400
    LOGS.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    log=(LOGS/f"panel-{model_id}.log").open("w")
    proc=subprocess.Popen([str(server),"-m",str(model),"--host","127.0.0.1","--port",str(port),"-t","4","-c","8192"],stdout=log,stderr=subprocess.STDOUT,text=True)
    start=time.monotonic()
    try:
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if proc.poll() is not None:
                log.close(); raise RuntimeError(Path(log.name).read_text(errors="replace")[-6000:])
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=1) as r:
                    if r.status==200: break
            except Exception: pass
            time.sleep(.25)
        else: raise RuntimeError("server startup timeout")
        startup=time.monotonic()-start; resident=rss(proc.pid)

        compact=json.dumps(population["candidates"],separators=(",",":"))
        prompt=f"""You are one independent blinded judge in a behaviour-kernel research panel.
Every judge receives exactly the same immutable candidate population. You cannot see other judges.

Evaluate mechanisms, not prose style. Prefer local deterministic behaviours that could produce rich or surprising visible behaviour from little state. Penalize genericness, hidden global knowledge, excessive complexity, incoherence and mechanisms whose claimed failure mode does not follow from the rule.

For EACH candidate report:
- verdict: preserve, repair, or destroy
- preserved: list of concrete mechanisms worth retaining
- destroyed: list of mechanisms/assumptions that should be removed
- repair: a concise repaired mechanism, or empty string if no repair is needed/useful
- best_failure: the most interesting plausible emergent failure
- scores: novelty, mechanical_clarity, emergence, simplicity, each integer 0..5

Then select exactly THREE finalists by immutable candidate ID. Finally create ONE recombination using mechanisms from at least two finalists. Do not invent or alter candidate IDs.

Return JSON only with this shape:
{{"reviews":{{"K01":{{"verdict":"...","preserved":[],"destroyed":[],"repair":"...","best_failure":"...","scores":{{"novelty":0,"mechanical_clarity":0,"emergence":0,"simplicity":0}}}}}},"finalists":["K01","K02","K03"],"recombination":{{"parents":["K01","K02"],"name":"...","mechanism":"...","failure":"..."}}}}

CANDIDATES:
{compact}
"""
        body={"model":"local","messages":[{"role":"user","content":prompt}],"max_tokens":1800,"temperature":0.2,"top_p":0.9,"seed":20261024,"stream":False}
        if model_id in {"qwen35-4b","qwen3-4b","smollm3-3b"}:
            body["chat_template_kwargs"]={"enable_thinking":False}
        req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
        infer=time.monotonic()
        with urllib.request.urlopen(req,timeout=900) as r: payload=json.loads(r.read())
        elapsed=time.monotonic()-infer
        msg=payload["choices"][0]["message"]; content=(msg.get("content") or "").strip(); reasoning=(msg.get("reasoning_content") or "").strip()
        usage=payload.get("usage") or {}; tokens=int(usage.get("completion_tokens") or 0); parsed=extract_json(content)
        record={"experiment":"panel-001","model_id":model_id,"spec":spec,"runner":{"platform":platform.platform(),"cpu_count":os.cpu_count(),"cpu_model":cpu()},"model_bytes":model.stat().st_size,"server_startup_seconds":startup,"server_rss_kib":resident,"elapsed_seconds":elapsed,"completion_tokens":tokens,"tokens_per_second":tokens/elapsed if tokens else None,"content":content,"reasoning_content":reasoning,"parsed":parsed,"usage":usage}
        (RESULTS/f"panel-{model_id}.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
        md=[f"# Panel 001 — {spec['label']}","",f"Startup: {startup:.3f}s; RSS: {(resident or 0)/(1024**2):.3f} GiB; completion: {tokens} tokens / {elapsed:.3f}s.","",f"Parsed JSON: **{parsed is not None}**",""]
        if parsed:
            md += ["## Finalists","",", ".join(parsed.get("finalists",[])),"","## Recombination","","~~~json",json.dumps(parsed.get("recombination"),indent=2,ensure_ascii=False),"~~~",""]
            for cid,review in parsed.get("reviews",{}).items():
                md += [f"## {cid} — {review.get('verdict','?')}","",f"Preserved: {review.get('preserved',[])}",f"Destroyed: {review.get('destroyed',[])}",f"Repair: {review.get('repair','')}",f"Failure: {review.get('best_failure','')}",""]
        else:
            md += ["## Raw output","",content,""]
        (RESULTS/f"panel-{model_id}.md").write_text("\n".join(md)+"\n")
        print("\n".join(md))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(5)
        if not log.closed: log.close()

if __name__=="__main__": main()
