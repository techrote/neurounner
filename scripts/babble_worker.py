#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, platform, subprocess, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(os.environ.get("NEURO_CACHE",ROOT/".cache"/"neurounner"))
RESULTS=ROOT/"results"/"babble001"

def load_env(p):
    out={}
    for raw in p.read_text().splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            k,v=raw.split("=",1); out[k]=v
    return out

def cpu_model():
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":",1)[1].strip()
    return "unknown"

def request(port, guide, variant, worker):
    system=(guide["system_prompt"]+"\nAttitude: "+guide["attitude_guide"]+
            "\nContent pressure: "+guide["content_guide"]+
            "\nMechanical motif: "+guide["mechanical_motif"]+
            "\nDo not clean yourself up. Produce raw evocative fragments, embodied habits, contradictory urges, local memories, "
            "physical reactions, odd rules, obsessions and failure-prone instincts. No JSON. No explanations about being an AI.")
    user=(f"Babble variant {variant}. Spill 6-12 compact fragments for this temperament. "
          f"Let some ideas be broken or mutually inconsistent if they are interesting. Failure flavor: {guide['interesting_failure_flavor']}")
    body={
      "model":"local",
      "messages":[{"role":"system","content":system},{"role":"user","content":user}],
      "max_tokens":180,"temperature":1.15,"top_p":0.97,
      "seed":20261024 + worker*1000 + int(guide["guide_id"][1:])*10 + variant,
      "stream":False
    }
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
      data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
    t0=time.monotonic()
    with urllib.request.urlopen(req,timeout=240) as r: payload=json.loads(r.read())
    elapsed=time.monotonic()-t0
    msg=payload["choices"][0]["message"]
    usage=payload.get("usage") or {}
    n=int(usage.get("completion_tokens") or 0)
    return {
      "elapsed_seconds":elapsed,"completion_tokens":n,
      "tokens_per_second":n/elapsed if n else None,
      "raw_text":(msg.get("content") or "").strip(),
      "finish_reason":payload["choices"][0].get("finish_reason")
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--worker",type=int,required=True)
    ap.add_argument("--guides",required=True)
    args=ap.parse_args()

    guides=json.load(open(args.guides))["guides"]
    chosen=[g for i,g in enumerate(guides) if i % 4 == args.worker]
    env=load_env(CACHE/"paths.env")
    server=Path(env["LLAMA_SERVER"]); model=Path(env["SPROCKET_MODEL"])
    RESULTS.mkdir(parents=True,exist_ok=True)
    port=18900+args.worker
    log=RESULTS/f"worker-{args.worker}-server.log"; fh=log.open("w")
    proc=subprocess.Popen([str(server),"-m",str(model),"--host","127.0.0.1","--port",str(port),"-t","4","-c","4096"],
      stdout=fh,stderr=subprocess.STDOUT,text=True)
    t0=time.monotonic()
    try:
      deadline=time.monotonic()+120
      while time.monotonic()<deadline:
        if proc.poll() is not None: raise RuntimeError(log.read_text(errors="replace")[-6000:])
        try:
          with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=1) as r:
            if r.status==200: break
        except Exception: pass
        time.sleep(.2)
      startup=time.monotonic()-t0
      records=[]
      for guide in chosen:
        for variant in range(4):
          result=request(port,guide,variant,args.worker)
          result.update({
            "babble_id":f"B{args.worker}-{guide['guide_id']}-{variant}",
            "guide_id":guide["guide_id"],
            "theme":guide["theme"],
            "variant":variant
          })
          records.append(result)
      record={
        "experiment":"BABBLE-001","stage":"babble","worker":args.worker,
        "runner":{"cpu_model":cpu_model(),"cpu_count":os.cpu_count(),"platform":platform.platform()},
        "startup_seconds":startup,
        "babble_count":len(records),
        "babbles":records
      }
      out=RESULTS/f"worker-{args.worker}.json"
      out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
      print(json.dumps(record,indent=2,ensure_ascii=False))
    finally:
      if proc.poll() is None:
        proc.terminate()
        try: proc.wait(10)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(5)
      fh.close()

if __name__=="__main__": main()
