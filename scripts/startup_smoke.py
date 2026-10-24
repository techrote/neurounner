#!/usr/bin/env python3
import json, os, platform, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(os.environ.get("NEURO_CACHE", ROOT/".cache"/"neurounner"))
RESULTS=ROOT/"results"; LOGS=ROOT/"logs"

def load_env(p):
    d={}
    for line in p.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            k,v=line.split("=",1); d[k]=v
    return d

def rss(pid):
    p=Path(f"/proc/{pid}/status")
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("VmRSS:"): return int(line.split()[1])
    return None

def cpu():
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"): return line.split(":",1)[1].strip()
    return "unknown"

def main():
    if len(sys.argv)!=3: raise SystemExit("usage: startup_smoke.py MODEL_ID MODEL_PATH")
    model_id=sys.argv[1]; model=Path(sys.argv[2]).resolve()
    spec=json.loads((ROOT/"config/candidates.json").read_text())[model_id]
    server=Path(load_env(CACHE/"paths.env")["LLAMA_SERVER"])
    RESULTS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    log_path=LOGS/f"startup8-{model_id}.log"
    log=log_path.open("w")
    port=18500
    t0=time.monotonic()
    proc=subprocess.Popen([str(server),"-m",str(model),"--host","127.0.0.1","--port",str(port),"-t","4","-c","2048"],stdout=log,stderr=subprocess.STDOUT,text=True)
    try:
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if proc.poll() is not None:
                log.close(); raise RuntimeError(log_path.read_text(errors="replace")[-6000:])
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=1) as r:
                    if r.status==200: break
            except Exception: pass
            time.sleep(.25)
        else: raise RuntimeError("startup timeout")
        startup=time.monotonic()-t0
        resident=rss(proc.pid)
        body={
          "model":"local",
          "messages":[{"role":"user","content":"In one sentence, invent a tiny deterministic physical NPC reflex using only local sensing and one bounded memory value."}],
          "max_tokens":64,"temperature":0.25,"top_p":0.9,"seed":8080,"stream":False
        }
        if model_id in {"qwen35-4b","qwen3-4b","smollm3-3b"}:
            body["chat_template_kwargs"]={"enable_thinking":False}
        req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
        t1=time.monotonic()
        with urllib.request.urlopen(req,timeout=240) as r: payload=json.loads(r.read())
        elapsed=time.monotonic()-t1
        msg=payload["choices"][0]["message"]; usage=payload.get("usage") or {}
        tokens=int(usage.get("completion_tokens") or 0)
        record={
          "experiment":"startup8","model_id":model_id,"label":spec["label"],
          "runner":{"platform":platform.platform(),"cpu_count":os.cpu_count(),"cpu_model":cpu()},
          "model_bytes":model.stat().st_size,"startup_seconds":startup,"rss_kib":resident,
          "inference_seconds":elapsed,"completion_tokens":tokens,
          "tokens_per_second":tokens/elapsed if tokens else None,
          "content":(msg.get("content") or "").strip(),
          "reasoning_content":(msg.get("reasoning_content") or "").strip(),
          "finish_reason":payload["choices"][0].get("finish_reason")
        }
        (RESULTS/f"startup8-{model_id}.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps(record,indent=2,ensure_ascii=False))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(5)
        if not log.closed: log.close()

if __name__=="__main__": main()
