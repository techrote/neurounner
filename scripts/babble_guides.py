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

def parse_tagged(text):
    mapping={
      "THEME":"theme",
      "SYSTEM":"system_prompt",
      "ATTITUDE":"attitude_guide",
      "CONTENT":"content_guide",
      "MOTIF":"mechanical_motif",
      "FAILURE":"interesting_failure_flavor"
    }
    out={}
    for raw in text.splitlines():
        line=raw.strip()
        for tag,key in mapping.items():
            prefix=tag+":"
            if line.upper().startswith(prefix):
                out[key]=line[len(prefix):].strip()
                break
    return out if REQUIRED.issubset(out) else None

def call(port,index,pressure):
    prompt=f"""Create ONE prompt-guide packet for a tiny chaotic model called Sprocket.
Guide index {index+1}/12. Temperament pressure: {pressure}.

The packet exists to make Sprocket babble raw personality/temperament material for later training-data compilation.
Favor embodied habits, local memories, obsessions, irrational biases, physical reactions, contradictory urges and interesting failure modes.
Do not make a polished helpful assistant persona.

Return EXACTLY six single-line fields and nothing else:
THEME: short thematic label
SYSTEM: short role pressure for Sprocket
ATTITUDE: emotional/temperamental cadence and biases
CONTENT: physical habits, local memories, motives and reactions to babble about
MOTIF: one recurring local-state/physical motif
FAILURE: broken/emergent behavior worth preserving

Do not imitate named fictional characters or living authors. Keep each line concise."""
    for attempt in range(2):
        body={
          "model":"local","messages":[{"role":"user","content":prompt}],
          "max_tokens":320,"temperature":0.85 if attempt==0 else 0.3,"top_p":0.95,
          "seed":20261024+index+attempt*1000,
          "stream":False,"chat_template_kwargs":{"enable_thinking":False}
        }
        req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
          data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
        t0=time.monotonic()
        with urllib.request.urlopen(req,timeout=420) as r: payload=json.loads(r.read())
        elapsed=time.monotonic()-t0
        text=payload["choices"][0]["message"].get("content") or ""
        RESULTS.mkdir(parents=True,exist_ok=True)
        (RESULTS/f"guide-{index+1:02d}-attempt-{attempt+1}-raw.txt").write_text(text+"\n")
        guide=parse_tagged(text)
        if guide:
            return guide,elapsed,payload.get("usage") or {},attempt+1
        prompt += "\nYour previous response did not contain all six exact tags. Obey the six-line format literally."
    raise RuntimeError(f"guide {index+1}: failed tagged format after two attempts")

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
        g,elapsed,usage,attempts=call(port,index,pressure)
        g["guide_id"]=f"G{index+1:02d}"
        g["orchestrator_pressure"]=pressure
        all_guides.append(g)
        calls.append({"guide_id":g["guide_id"],"elapsed_seconds":elapsed,"usage":usage,"attempts":attempts})
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
