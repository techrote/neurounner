#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time, urllib.error, urllib.request
from pathlib import Path

MODEL="minimaxai/minimax-m3.1-flash"
ENDPOINT="https://integrate.api.nvidia.com/v1/chat/completions"

def extract_json(text):
    a=text.find("{"); b=text.rfind("}")
    if a<0 or b<=a: raise ValueError("no JSON object in compiler response")
    return json.loads(text[a:b+1])

def load_inputs(guides_path:Path,babble_root:Path):
    guides={g["guide_id"]:g for g in json.loads(guides_path.read_text())["guides"]}
    babbles={}
    for p in sorted(babble_root.rglob("worker-*.json")):
        obj=json.loads(p.read_text())
        for b in obj["babbles"]:
            babbles.setdefault(b["guide_id"],[]).append(b)
    missing=[gid for gid in guides if len(babbles.get(gid,[]))<1]
    if missing: raise RuntimeError(f"missing babbles for guides: {missing}")
    return guides,babbles

def call_api(api_key,payload,batch_id,outdir):
    body=json.dumps(payload).encode()
    headers={
      "Authorization":"Bearer "+api_key,
      "Content-Type":"application/json",
      "Accept":"application/json",
      "User-Agent":"neurounner-babble001"
    }
    attempts=[]
    for attempt in range(1,7):
        t0=time.monotonic()
        req=urllib.request.Request(ENDPOINT,data=body,headers=headers,method="POST")
        try:
            with urllib.request.urlopen(req,timeout=900) as r:
                raw=r.read().decode("utf-8")
                elapsed=time.monotonic()-t0
                attempts.append({"attempt":attempt,"status":r.status,"elapsed_seconds":elapsed})
                obj=json.loads(raw)
                (outdir/f"compiler-batch-{batch_id}-raw.json").write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
                return obj,attempts
        except urllib.error.HTTPError as e:
            elapsed=time.monotonic()-t0
            detail=e.read().decode("utf-8",errors="replace")[:4000]
            attempts.append({"attempt":attempt,"status":e.code,"elapsed_seconds":elapsed,"detail":detail})
            if e.code not in (408,429,500,502,503,504) or attempt==6:
                raise
            retry=e.headers.get("Retry-After")
            wait=float(retry) if retry and retry.replace(".","",1).isdigit() else min(60,5*(2**(attempt-1)))
            time.sleep(wait)
        except Exception as e:
            attempts.append({"attempt":attempt,"status":"exception","detail":repr(e)})
            if attempt==6: raise
            time.sleep(min(60,5*(2**(attempt-1))))
    raise RuntimeError("unreachable")

def compiler_prompt(items):
    return """You are compiling raw babble from a tiny chaotic local model into reusable supervised-training personality packets.

Do not turn everything into a generic helpful assistant. Preserve distinctive temperament, obsessions, embodied habits, ugly edges, hesitation, irrationality, and failure-prone instincts when they are useful.
Do not invent unrelated lore to make a packet prettier.
When source fragments contradict, retain the contradiction as alternate tendencies, conditional traits, or interesting_broken rather than silently averaging it away.
Source IDs are immutable: every claim that matters should remain traceable to guide_id and babble_id.

Return exactly one packet per guide_id.

Each packet must contain:
- source_guide_id
- source_babble_ids
- name: short original label
- persona_summary: 1-3 sentences
- temperament_axes: integer 0..100 values for caution, impulsivity, curiosity, sociability, territoriality, persistence, orderliness, dominance
- system_prompt: reusable personality system prompt
- attitude_guide: compact tone/temperament guide
- content_guide: what this persona tends to notice, want, remember and react to
- embodied_habits: 4-8 concrete physical/local-state tendencies
- linguistic_habits: 2-5 stylistic tendencies; these may be empty if the personality is intended to be nonverbal
- motivational_biases: 3-6 local motives or priorities
- mechanical_seeds: 3-6 short ideas that could later be compiled into deterministic actor mechanics
- interesting_broken: 1-5 weird contradictions/failure modes worth retaining
- sft_examples: exactly 4 objects with input and target. They should demonstrate the personality in diverse situations and be useful later for small-model SFT. Targets should embody the temperament without mentioning Sprocket or this compilation process.
- negative_examples: exactly 2 objects with input, bad_target, why_bad. Use these to show generic/sanitized behavior that would erase the persona.

Output JSON only as {"packets":[...]}. Do not include markdown."""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--guides",type=Path,required=True)
    ap.add_argument("--babbles",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    key=os.environ.get("NVIDIA_API_KEY","").strip()
    if not key: raise SystemExit("NVIDIA_API_KEY is not configured")
    args.out.mkdir(parents=True,exist_ok=True)
    rawdir=args.out/"raw"; rawdir.mkdir(exist_ok=True)

    guides,babbles=load_inputs(args.guides,args.babbles)
    gids=sorted(guides)
    all_packets=[]; batches=[]
    for batch_idx,start in enumerate(range(0,len(gids),3)):
        batch_gids=gids[start:start+3]
        source=[]
        for gid in batch_gids:
            source.append({
              "guide":guides[gid],
              "babbles":[{"babble_id":b["babble_id"],"raw_text":b["raw_text"]} for b in sorted(babbles[gid],key=lambda x:x["babble_id"])]
            })
        messages=[
          {"role":"system","content":compiler_prompt(source)},
          {"role":"user","content":"Compile these source groups into packets:\n"+json.dumps(source,ensure_ascii=False)}
        ]
        payload={
          "model":MODEL,
          "messages":messages,
          "max_tokens":6000,
          "temperature":1.0,
          "top_p":0.95,
          "stream":False,
          "reasoning_effort":"low"
        }
        obj,attempts=call_api(key,payload,batch_idx,rawdir)
        msg=obj["choices"][0]["message"]
        parsed=extract_json(msg.get("content") or "")
        packets=parsed.get("packets")
        if not isinstance(packets,list): raise RuntimeError(f"batch {batch_idx}: packets not list")
        by_gid={p.get("source_guide_id"):p for p in packets if isinstance(p,dict)}
        if set(by_gid)!=set(batch_gids):
            raise RuntimeError(f"batch {batch_idx}: expected {batch_gids}, got {sorted(by_gid)}")
        for gid in batch_gids:
            p=by_gid[gid]
            expected_ids=sorted(b["babble_id"] for b in babbles[gid])
            actual=sorted(p.get("source_babble_ids") or [])
            if actual!=expected_ids:
                p["source_babble_ids"]=expected_ids
                p["lineage_repaired_by_orchestrator"]=True
            else:
                p["lineage_repaired_by_orchestrator"]=False
            p["packet_id"]=f"P{len(all_packets)+1:02d}"
            p["compiler_model"]=MODEL
            all_packets.append(p)
        batches.append({
          "batch":batch_idx,"guide_ids":batch_gids,"attempts":attempts,
          "usage":obj.get("usage"),"finish_reason":obj["choices"][0].get("finish_reason"),
          "reasoning_present":bool(msg.get("reasoning_content"))
        })

    with (args.out/"packets.jsonl").open("w") as fh:
        for p in all_packets:
            fh.write(json.dumps(p,ensure_ascii=False)+"\n")
    manifest={
      "experiment":"BABBLE-001","compiler_provider":"nvidia","compiler_model":MODEL,
      "packet_count":len(all_packets),"guide_count":len(guides),
      "babble_count":sum(len(x) for x in babbles.values()),"batches":batches
    }
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(manifest,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
