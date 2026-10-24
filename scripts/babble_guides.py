#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, platform, subprocess, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(os.environ.get("NEURO_CACHE",ROOT/".cache"/"neurounner"))
RESULTS=ROOT/"results"/"babble001"

PRESSURES=[
 "paranoid scavenger; cautious, acquisitive, suspicious of repeated local patterns",
 "reckless thrill-seeker; impulsive, novelty-hungry, attracted to motion and near-misses",
 "ritualistic stubborn machine; orderly, persevering, repeats procedures long after they stop helping",
 "timid social mimic; submissive, observant, copies nearby successes but startles easily",
 "territorial miser; possessive, defensive, remembers places and resents intrusion",
 "curious self-saboteur; investigates anomalies even when prior investigation went badly",
 "patient ambusher; low sociability, high persistence, waits too long then commits abruptly",
 "compulsive helper; social, over-attentive, interferes with nearby actors whether wanted or not",
 "resentful memory-keeper; slow to forgive pushes, blocks and failed interactions",
 "chaotic opportunist; flighty, fast-switching, exploits temporary openings without planning",
 "orderly anxious caretaker; cautious, tidies local disorder and panics when routines are broken",
 "dreamy inertia-prone obsessive; apathetic until one stimulus captures attention, then fixates excessively"
]

REQUIRED={"theme","system_prompt","attitude_guide","content_guide","mechanical_motif","interesting_failure_flavor"}

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

def find_guide(x):
    if isinstance(x,dict):
        if REQUIRED.issubset(x):
            return {k:x[k] for k in REQUIRED}
        for v in x.values():
            g=find_guide(v)
            if g: return g
    elif isinstance(x,list):
        for v in x:
            g=find_guide(v)
            if g: return g
    return None

def call(port,index,pressure):
    schema={
      "type":"object",
      "properties":{
        "theme":{"type":"string"},
        "system_prompt":{"type":"string"},
        "attitude_guide":{"type":"string"},
        "content_guide":{"type":"string"},
        "mechanical_motif":{"type":"string"},
        "interesting_failure_flavor":{"type":"string"}
      },
      "required":sorted(REQUIRED),"additionalProperties":False
    }
    prompt=f"""Create ONE prompt-guide packet for a tiny chaotic model called Sprocket.
Guide index {index+1}/12. Temperament pressure: {pressure}.

The packet exists to make Sprocket babble raw personality/temperament material for later training-data compilation.
Do not make a polished assistant persona. Favor embodied habits, local memories, obsessions, irrational biases,
physical reactions, contradictory urges and interesting failure modes.

system_prompt: short role pressure for Sprocket, not JSON instructions.
attitude_guide: emotional/temperamental cadence and biases.
content_guide: kinds of physical habits, local memories, motives and reactions it should babble about.
mechanical_motif: one recurring local-state/physical motif.
interesting_failure_flavor: broken/emergent behavior worth preserving.
Do not imitate named fictional characters or living authors. Keep every field concise."""
    body={
      "model":"local","messages":[{"role":"user","content":prompt}],
      "max_tokens":520,"temperature":0.9,"top_p":0.95,"seed":20261024+index,
      "response_format":{"type":"json_schema","schema":schema},"stream":False,
      "chat_template_kwargs":{"enable_thinking":False}
    }
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
      data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
    t0=time.monotonic()
    with urllib.request.urlopen(req,timeout=420) as r: payload=json.loads(r.read())
    elapsed=time.monotonic()-t0
    text=payload["choices"][0]["message"].get("content") or ""
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS/f"guide-{index+1:02d}-raw.txt").write_text(text+"\n")
    a,b=text.find("{"),text.rfind("}")
    if a<0 or b<=a: raise RuntimeError(f"guide {index+1}: no JSON object")
    try:
        parsed=json.loads(text[a:b+1])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"guide {index+1}: invalid JSON: {e}") from e
    guide=find_guide(parsed)
    if not guide: raise RuntimeError(f"guide {index+1}: complete six-field guide not found")
    return guide,elapsed,payload.get("usage") or {}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("model"); args=ap.parse_args()
    env=load_env(CACHE/"paths.env"); server=Path(env["LLAMA_SERVER"])
    port=18800; RESULTS.mkdir(parents=True,exist_ok=True)
    log=RESULTS/"guide-server.log"; fh=log.open("w")
    proc=subprocess.Popen([str(server),"-m",str(Path(args.model).resolve()),"--host","127.0.0.1","--port",str(port),"-t","4","-c","4096"],
      stdout=fh,stderr=subprocess.STDOUT,text=True)
    t0=time.monotonic()
    try:
      deadline=time.monotonic()+180
      while time.monotonic()<deadline:
        if proc.poll() is not None: raise RuntimeError(log.read_text(errors="replace")[-6000:])
        try:
          with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=1) as r:
            if r.status==200: break
        except Exception: pass
        time.sleep(.25)
      startup=time.monotonic()-t0
      all_guides=[]; calls=[]
      for index,pressure in enumerate(PRESSURES):
        g,elapsed,usage=call(port,index,pressure)
        g["guide_id"]=f"G{index+1:02d}"
        g["orchestrator_pressure"]=pressure
        all_guides.append(g)
        calls.append({"guide_id":g["guide_id"],"elapsed_seconds":elapsed,"usage":usage})
      record={"experiment":"BABBLE-001","stage":"guides",
        "runner":{"cpu_model":cpu_model(),"cpu_count":os.cpu_count(),"platform":platform.platform()},
        "startup_seconds":startup,"calls":calls,"guides":all_guides}
      (RESULTS/"guides.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
      print(json.dumps(record,indent=2,ensure_ascii=False))
    finally:
      if proc.poll() is None:
        proc.terminate()
        try: proc.wait(10)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(5)
      fh.close()

if __name__=="__main__": main()
