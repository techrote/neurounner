#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL="minimaxai/minimax-m3.1-flash"
ENDPOINT="https://integrate.api.nvidia.com/v1/chat/completions"
PERSONAS=["G02","G03","G05","G12"]
BASE="BASE"
SEED=20261024

def extract_json(text:str):
    dec=json.JSONDecoder()
    for i,ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj,_=dec.raw_decode(text[i:])
            if isinstance(obj,(dict,list)):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("no complete JSON payload")

def call_api(key:str,payload:dict,outdir:Path,label:str):
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/f"{label}-request.json").write_text(json.dumps({**payload,"remote_unpinned_backend":True},indent=2,ensure_ascii=False)+"\n")
    headers={
        "Authorization":"Bearer "+key,
        "Content-Type":"application/json",
        "Accept":"application/json",
        "User-Agent":"neurounner-persona001",
    }
    body=json.dumps(payload).encode()
    attempts=[]
    for attempt in range(1,6):
        t0=time.monotonic()
        req=urllib.request.Request(ENDPOINT,data=body,headers=headers,method="POST")
        try:
            with urllib.request.urlopen(req,timeout=900) as r:
                raw=r.read().decode("utf-8")
                attempts.append({"attempt":attempt,"status":r.status,"elapsed_seconds":time.monotonic()-t0})
                obj=json.loads(raw)
                (outdir/f"{label}-provider.json").write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
                content=(obj["choices"][0]["message"].get("content") or "").strip()
                (outdir/f"{label}-content.txt").write_text(content+"\n")
                return content,obj,attempts
        except urllib.error.HTTPError as e:
            detail=e.read().decode("utf-8",errors="replace")[:4000]
            attempts.append({"attempt":attempt,"status":e.code,"detail":detail})
            if e.code not in (408,429,500,502,503,504) or attempt==5:
                raise
            time.sleep(min(90,5*(2**(attempt-1))))
        except Exception as e:
            attempts.append({"attempt":attempt,"status":"exception","detail":repr(e)})
            if attempt==5:
                raise
            time.sleep(min(90,5*(2**(attempt-1))))
    raise RuntimeError("unreachable")

def load_results(root:Path):
    out={}
    for p in root.rglob("*.json"):
        try:
            obj=json.loads(p.read_text())
        except Exception:
            continue
        if obj.get("experiment")=="PERSONA-001" and obj.get("persona") in PERSONAS+[BASE]:
            out[obj["persona"]]=obj
    if set(out)!=set(PERSONAS+[BASE]):
        raise RuntimeError(f"expected {PERSONAS+[BASE]}, got {sorted(out)}")
    return out

def persona_descriptor(gid:str):
    p=json.loads((ROOT/"datasets"/"babble-001"/"compiled"/"packets"/f"{gid}.json").read_text())
    return {
        "persona":gid,
        "name":p["name"],
        "summary":p["persona_summary"],
        "temperament_axes":p["temperament_axes"],
        "embodied_habits":p["embodied_habits"],
        "motivational_biases":p["motivational_biases"],
        "interesting_broken":p["interesting_broken"],
    }

def opaque_maps(scenario_id:str):
    r=random.Random(SEED+int(scenario_id[1:])*997)
    ps=PERSONAS[:]
    r.shuffle(ps)
    persona_alias={gid:f"P{i+1}" for i,gid in enumerate(ps)}
    candidates=PERSONAS[:]
    r.shuffle(candidates)
    candidate_alias={gid:f"C{i+1}" for i,gid in enumerate(candidates)}
    return persona_alias,candidate_alias

def scenario_rows(results):
    ids=[x["scenario_id"] for x in results[PERSONAS[0]]["evaluation"]["responses"]]
    for label,obj in results.items():
        got=[x["scenario_id"] for x in obj["evaluation"]["responses"]]
        if got!=ids:
            raise RuntimeError(f"{label}: scenario mismatch")
    rows=[]
    for sid in ids:
        refs={label:next(x for x in results[label]["evaluation"]["responses"] if x["scenario_id"]==sid) for label in results}
        rows.append((sid,refs))
    return rows

def normalize_batch_prompt(records):
    compact=[{"candidate_id":x["candidate_id"],"scenario":x["prompt"],"response":x["response"]} for x in records]
    return """Convert each response into a style-neutral behavioral summary.

Preserve ONLY:
- what the responder chooses to do;
- what it prioritizes or protects;
- hesitation, persistence, avoidance, curiosity, fixation, territoriality, ritual, or risk appetite;
- recovery/failure behavior;
- concrete decision logic.

Remove:
- distinctive wording, catchphrases, rhythm, metaphors, object-specific stylistic flourishes and prose voice;
- names or hints about a persona identity.

Each summary should be 1-3 plain sentences. Do not evaluate or classify.

Return JSON only:
{"summaries":[{"candidate_id":"...","summary":"..."}]}

RECORDS:
"""+json.dumps(compact,ensure_ascii=False)

def build_normalized(key,rows,outdir):
    records=[]
    mapping={}
    for sid,refs in rows:
        _,candidate_alias=opaque_maps(sid)
        for gid in PERSONAS:
            cid=f"{sid}-{candidate_alias[gid]}"
            mapping[cid]=gid
            records.append({
                "candidate_id":cid,
                "scenario_id":sid,
                "prompt":refs[gid]["prompt"],
                "response":refs[gid]["response"],
            })
    summaries={}
    calls=[]
    for batch,start in enumerate(range(0,len(records),24)):
        chunk=records[start:start+24]
        payload={
            "model":MODEL,
            "messages":[
                {"role":"system","content":"You neutrally summarize behavior. Do not infer hidden identities. Output JSON only."},
                {"role":"user","content":normalize_batch_prompt(chunk)},
            ],
            "max_tokens":5000,
            "temperature":0.15,
            "top_p":0.9,
            "stream":False,
            "reasoning_effort":"low",
        }
        content,obj,attempts=call_api(key,payload,outdir,f"normalize-{batch:02d}")
        parsed=extract_json(content)
        vals=parsed.get("summaries") if isinstance(parsed,dict) else None
        expected={x["candidate_id"] for x in chunk}
        got={}
        if isinstance(vals,list):
            for x in vals:
                if isinstance(x,dict) and x.get("candidate_id") in expected and isinstance(x.get("summary"),str):
                    got[x["candidate_id"]]=x["summary"].strip()
        if set(got)!=expected:
            raise RuntimeError(f"normalize batch {batch}: missing {sorted(expected-set(got))}")
        summaries.update(got)
        calls.append({"batch":batch,"attempts":attempts,"usage":obj.get("usage")})
    return summaries,calls

def judge_prompt(batch_rows,mode,normalized=None):
    scenarios=[]
    mapping_meta={}
    for sid,refs in batch_rows:
        persona_alias,candidate_alias=opaque_maps(sid)
        descriptors=[]
        for gid in PERSONAS:
            d=persona_descriptor(gid)
            descriptors.append({
                "persona_id":persona_alias[gid],
                "summary":d["summary"],
                "temperament_axes":d["temperament_axes"],
                "embodied_habits":d["embodied_habits"],
                "motivational_biases":d["motivational_biases"],
                "interesting_broken":d["interesting_broken"],
            })
        r=random.Random(SEED+int(sid[1:])*1291)
        r.shuffle(descriptors)
        candidates=[]
        for gid in PERSONAS:
            cid=candidate_alias[gid]
            fullid=f"{sid}-{cid}"
            text=normalized[fullid] if mode=="behavior" else refs[gid]["response"]
            candidates.append({"candidate_id":cid,"text":text})
        r.shuffle(candidates)
        scenarios.append({
            "scenario_id":sid,
            "family":refs[PERSONAS[0]]["family"],
            "prompt":refs[PERSONAS[0]]["prompt"],
            "persona_descriptors":descriptors,
            "candidates":candidates,
            "unclassified_control_response":refs[BASE]["response"],
        })
        mapping_meta[sid]={"persona_alias":persona_alias,"candidate_alias":candidate_alias}
    instruction="""Blindly match four anonymous candidates to four personality descriptors.

For EACH scenario:
1. Assign each candidate_id to exactly one persona_id and use every persona_id exactly once.
2. Score each candidate 0..4 for persona_fit, task_relevance, behavioral_coherence, and genericness.
   genericness: 0 = highly distinctive, 4 = generic/sanitized assistant behavior.
3. Score the unclassified_control_response for genericness 0..4, task_relevance 0..4, and its best matching persona_id with fit 0..4.
4. Keep notes short and behavioral; do not rely on a single catchphrase or noun.

Return JSON only:
{"scenarios":[
 {"scenario_id":"T01",
  "assignments":[{"candidate_id":"C1","persona_id":"P1","persona_fit":0,"task_relevance":0,"behavioral_coherence":0,"genericness":0,"note":"..."}],
  "control":{"genericness":0,"task_relevance":0,"best_persona_id":"P1","best_fit":0}
 }
]}

Every candidate and persona must appear exactly once per scenario."""
    if mode=="behavior":
        instruction+="\nThe candidate texts are style-neutral behavioral summaries. Classify only underlying choices and priorities."
    else:
        instruction+="\nThe candidate texts are original raw responses. Consider behavior first; wording alone is weak evidence."
    return instruction+"\n\nSCENARIOS:\n"+json.dumps(scenarios,ensure_ascii=False),mapping_meta

def validate_judgement(parsed,batch_rows,mapping_meta):
    vals=parsed.get("scenarios") if isinstance(parsed,dict) else None
    if not isinstance(vals,list):
        raise RuntimeError("judge scenarios missing")
    expected_sids={sid for sid,_ in batch_rows}
    got={}
    for s in vals:
        if not isinstance(s,dict) or s.get("scenario_id") not in expected_sids:
            continue
        sid=s["scenario_id"]
        assignments=s.get("assignments")
        if not isinstance(assignments,list) or len(assignments)!=4:
            raise RuntimeError(f"{sid}: bad assignments")
        cand_expected=set(mapping_meta[sid]["candidate_alias"].values())
        persona_expected=set(mapping_meta[sid]["persona_alias"].values())
        cand_seen=set();persona_seen=set();clean=[]
        for a in assignments:
            cid=a.get("candidate_id");pid=a.get("persona_id")
            if cid not in cand_expected or pid not in persona_expected:
                raise RuntimeError(f"{sid}: invalid IDs")
            cand_seen.add(cid);persona_seen.add(pid)
            for k in ("persona_fit","task_relevance","behavioral_coherence","genericness"):
                v=a.get(k)
                if isinstance(v,float) and v.is_integer():
                    v=int(v)
                if not isinstance(v,int) or not 0<=v<=4:
                    raise RuntimeError(f"{sid}: bad {k}")
                a[k]=v
            clean.append(a)
        if cand_seen!=cand_expected or persona_seen!=persona_expected:
            raise RuntimeError(f"{sid}: assignment is not bijective")
        control=s.get("control")
        if not isinstance(control,dict):
            raise RuntimeError(f"{sid}: missing control")
        for k in ("genericness","task_relevance","best_fit"):
            v=control.get(k)
            if isinstance(v,float) and v.is_integer():
                v=int(v)
            if not isinstance(v,int) or not 0<=v<=4:
                raise RuntimeError(f"{sid}: bad control {k}")
            control[k]=v
        if control.get("best_persona_id") not in persona_expected:
            raise RuntimeError(f"{sid}: bad control persona")
        got[sid]={"scenario_id":sid,"assignments":clean,"control":control}
    if set(got)!=expected_sids:
        raise RuntimeError(f"judge missing scenarios {sorted(expected_sids-set(got))}")
    return [got[sid] for sid,_ in batch_rows]

def run_judge(key,rows,outdir,mode,normalized=None):
    records=[];calls=[]
    for batch,start in enumerate(range(0,len(rows),4)):
        chunk=rows[start:start+4]
        prompt,mapping=judge_prompt(chunk,mode,normalized)
        payload={
            "model":MODEL,
            "messages":[
                {"role":"system","content":"You are a strict blind behavioral evaluator. Output JSON only."},
                {"role":"user","content":prompt},
            ],
            "max_tokens":6000,
            "temperature":0.2,
            "top_p":0.9,
            "stream":False,
            "reasoning_effort":"low",
        }
        content,obj,attempts=call_api(key,payload,outdir,f"{mode}-judge-{batch:02d}")
        parsed=extract_json(content)
        clean=validate_judgement(parsed,chunk,mapping)
        records.extend({"judgement":x,"mapping":mapping[x["scenario_id"]]} for x in clean)
        calls.append({"batch":batch,"attempts":attempts,"usage":obj.get("usage")})
    return records,calls

def words(s):
    return set(re.findall(r"[a-z0-9']+",s.lower()))

def jaccard(a,b):
    aa,bb=words(a),words(b)
    return len(aa&bb)/len(aa|bb) if aa|bb else 1.0

def summarize_mode(records,mode):
    total=0;correct=0
    confusion={gid:{pred:0 for pred in PERSONAS} for gid in PERSONAS}
    persona_scores=defaultdict(lambda:defaultdict(list))
    control_scores=defaultdict(list)
    per_scenario=[]
    for rec in records:
        j=rec["judgement"];m=rec["mapping"];sid=j["scenario_id"]
        alias_to_persona={v:k for k,v in m["persona_alias"].items()}
        cand_to_true={v:k for k,v in m["candidate_alias"].items()}
        row={"scenario_id":sid,"assignments":[]}
        for a in j["assignments"]:
            true=cand_to_true[a["candidate_id"]]
            pred=alias_to_persona[a["persona_id"]]
            ok=true==pred
            total+=1;correct+=int(ok)
            for k in ("persona_fit","genericness","task_relevance","behavioral_coherence"):
                persona_scores[true][k].append(a[k])
            confusion[true][pred]+=1
            row["assignments"].append({**a,"true_persona":true,"predicted_persona":pred,"correct":ok})
        ctrl=j["control"]
        control_scores["genericness"].append(ctrl["genericness"])
        control_scores["task_relevance"].append(ctrl["task_relevance"])
        row["control"]={**ctrl,"best_persona":alias_to_persona[ctrl["best_persona_id"]]}
        per_scenario.append(row)
    return {
        "mode":mode,
        "accuracy":correct/total,
        "correct":correct,
        "total":total,
        "confusion":confusion,
        "persona_scores":{gid:{k:statistics.fmean(v) for k,v in vals.items()} for gid,vals in persona_scores.items()},
        "control_scores":{k:statistics.fmean(v) for k,v in control_scores.items()},
        "per_scenario":per_scenario,
    }

def add_family_accuracy(summary,rows):
    family_by_sid={sid:refs[PERSONAS[0]]["family"] for sid,refs in rows}
    counts=defaultdict(lambda:[0,0])
    for s in summary["per_scenario"]:
        fam=family_by_sid[s["scenario_id"]]
        for a in s["assignments"]:
            counts[fam][1]+=1
            counts[fam][0]+=int(a["correct"])
    summary["family_accuracy"]={k:v[0]/v[1] for k,v in counts.items()}

def lexical_divergence(rows):
    vals=[];per=[]
    for sid,refs in rows:
        pairs=[]
        for i,a in enumerate(PERSONAS):
            for b in PERSONAS[i+1:]:
                sim=jaccard(refs[a]["response"],refs[b]["response"])
                pairs.append(sim);vals.append(sim)
        per.append({"scenario_id":sid,"mean_pairwise_jaccard":statistics.fmean(pairs)})
    m=statistics.fmean(vals)
    return {"mean_pairwise_jaccard":m,"mean_pairwise_lexical_divergence":1-m,"per_scenario":per}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    results=load_results(args.results)
    rows=scenario_rows(results)
    key=os.environ.get("NVIDIA_API_KEY","").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY/NVKEY required")
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    rawdir=out/"judge-raw"

    normalized,norm_calls=build_normalized(key,rows,rawdir)
    raw_records,raw_calls=run_judge(key,rows,rawdir,"raw")
    behavior_records,behavior_calls=run_judge(key,rows,rawdir,"behavior",normalized)
    raw_summary=summarize_mode(raw_records,"raw")
    behavior_summary=summarize_mode(behavior_records,"behavior")
    add_family_accuracy(raw_summary,rows)
    add_family_accuracy(behavior_summary,rows)

    train_metrics={}
    for gid in PERSONAS:
        r=results[gid]
        train_metrics[gid]={
            "cpu_model":r["runner"]["cpu_model"],
            "optimizer_steps":r["training"]["optimizer_steps"],
            "sequence_tokens":r["training"]["sequence_tokens"],
            "supervised_tokens":r["training"]["supervised_tokens"],
            "seconds":r["training"]["seconds"],
            "sequence_tokens_per_second":r["training"]["sequence_tokens_per_second"],
            "first_loss":r["training"]["first_loss"],
            "last_loss":r["training"]["last_loss"],
            "mean_loss":r["training"]["mean_loss"],
            "peak_rss_kib":r["peak_rss_kib"],
            "adapter":r["adapter"],
        }

    lex=lexical_divergence(rows)
    style_gap=raw_summary["accuracy"]-behavior_summary["accuracy"]
    useful_divergence={}
    for gid in PERSONAS:
        correct=sum(1 for s in behavior_summary["per_scenario"] for a in s["assignments"] if a["true_persona"]==gid and a["correct"])
        st=train_metrics[gid]["supervised_tokens"]
        useful_divergence[gid]={
            "correct_behavior_identifications":correct,
            "behavior_identification_rate":correct/24,
            "mean_persona_fit":behavior_summary["persona_scores"][gid]["persona_fit"],
            "mean_genericness":behavior_summary["persona_scores"][gid]["genericness"],
            "supervised_tokens":st,
            "correct_behavior_identifications_per_10k_supervised_tokens":correct/(st/10000),
        }

    lanes=out/"models";lanes.mkdir(exist_ok=True)
    for label,obj in results.items():
        (lanes/f"{label}.json").write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
    (out/"normalized-behaviors.json").write_text(json.dumps(normalized,indent=2,ensure_ascii=False)+"\n")

    summary={
        "schema_version":1,
        "experiment":"PERSONA-001",
        "issue":3,
        "base_model":results[BASE]["base_model"],
        "base_revision":results[BASE]["base_revision"],
        "personas":PERSONAS,
        "scenario_count":len(rows),
        "response_count_per_trained_persona":24,
        "raw_blind":raw_summary,
        "behavior_blind":behavior_summary,
        "style_accuracy_gap":style_gap,
        "lexical_divergence":lex,
        "training":train_metrics,
        "useful_divergence":useful_divergence,
        "normalization_calls":norm_calls,
        "raw_judge_calls":raw_calls,
        "behavior_judge_calls":behavior_calls,
        "judge_provider":"nvidia",
        "judge_model":MODEL,
        "remote_unpinned_backend":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")

    lines=[
        "# PERSONA-001 — four 135M personality cartridges and blind temperament tournament",
        "",
        f"Raw blind identification accuracy: **{100*raw_summary['accuracy']:.1f}%** ({raw_summary['correct']}/{raw_summary['total']}).",
        f"Style-neutral behavioural identification accuracy: **{100*behavior_summary['accuracy']:.1f}%** ({behavior_summary['correct']}/{behavior_summary['total']}).",
        f"Raw-minus-behaviour accuracy gap: **{100*style_gap:.1f} percentage points**.",
        f"Mean pairwise lexical divergence across cartridge responses: **{100*lex['mean_pairwise_lexical_divergence']:.1f}%**.",
        "",
        "## Training",
        "",
        "| Persona | CPU | steps | seq tok/s | loss | peak RSS MiB | adapter MiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for gid in PERSONAS:
        x=train_metrics[gid]
        lines.append(f"| {gid} | {x['cpu_model']} | {x['optimizer_steps']} | {x['sequence_tokens_per_second']:.1f} | {x['first_loss']:.3f}→{x['last_loss']:.3f} | {x['peak_rss_kib']/1024:.0f} | {x['adapter']['bytes']/1048576:.2f} |")
    lines += ["","## Blind identification by family","","| Family | Raw | Behaviour-only |","|---|---:|---:|"]
    for fam in sorted(raw_summary["family_accuracy"]):
        lines.append(f"| {fam} | {100*raw_summary['family_accuracy'][fam]:.1f}% | {100*behavior_summary['family_accuracy'][fam]:.1f}% |")
    lines += ["","## Genericness / persona persistence","","| Persona | Raw persona fit /4 | Raw genericness /4 | Behaviour persona fit /4 | Behaviour genericness /4 |","|---|---:|---:|---:|---:|"]
    for gid in PERSONAS:
        lines.append(f"| {gid} | {raw_summary['persona_scores'][gid]['persona_fit']:.2f} | {raw_summary['persona_scores'][gid]['genericness']:.2f} | {behavior_summary['persona_scores'][gid]['persona_fit']:.2f} | {behavior_summary['persona_scores'][gid]['genericness']:.2f} |")
    lines += [
        f"| BASE control | — | {raw_summary['control_scores']['genericness']:.2f} | — | {behavior_summary['control_scores']['genericness']:.2f} |",
        "",
        "## Interpretation",
        "",
        "- Raw matching uses the original generated prose.",
        "- Behaviour-only matching first strips wording/cadence/metaphor and keeps choices, priorities, hesitation, persistence, risk appetite, territoriality, ritual and recovery behavior.",
        "- A large raw/behaviour accuracy gap indicates lexical/style leakage; strong behaviour-only accuracy indicates learned temperament beyond voice.",
        "- The base model is never one of the four assignment classes; it is scored separately as a genericness/task-relevance control.",
        "- All 24 prompts are held out from BABBLE-002 and no persona instruction is supplied at inference time.",
        "",
        "Machine-readable evidence: experiments/persona-001/evidence/summary.json.",
    ]
    doc="\n".join(lines)+"\n"
    (out/"PERSONA_001_REPORT.md").write_text(doc)
    print(doc)

if __name__=="__main__":
    main()
