#!/usr/bin/env python3
"""GP350-011-R: verify completed synthetic W0 inference in a FAILED CI artifact.

The original inference stage completed. A Python Path/string error in the
subsequent verification step failed the original workflow. This audit reuses
immutable recorded logits, not new full-model forwards; do not count any
of the inherited reference checkpoints as fresh inference.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ACTIONS=("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP")
FAMILIES=("greebles","pipework","architecture")
FRESH=("gp350010-step24","gp350010-step48")
RUN=43867250107
ARTIFACT=14226204589
SOURCE="b99d214ce292b7b3c1d8a18e6030c38316c3b489"
PARENT="ad5781667e2b4fdd29521a21bcd13adda5d5511c"
REVISION="9921cd4d82c0e3986887aa4b376f9446729ef8a3"
CHECKPOINTS={
 "24":"047b2bacb61f6f04fb359e4d9839f6c198a17ba7db6dfb451d88e48846c66430",
 "48":"159892e47cf76cef24274cf8d76881fb56a2b20052ae82eae228b7c036669e7b",
 "72":"a829b47fdf424b66fa8ddedf59323d5e10406e0afd8d881a423194ea790dc82a"}

def check(ok,msg):
    if not ok: raise ValueError(msg)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

def ray2(o):
    dx,dy=((1,0),(0,1),(-1,0),(0,-1))[o["heading"]%4]
    return o["local_5x5"][2+2*dy][2+2*dx]

def verify_episode(root,row):
    name=row["png"]
    png=root/name
    check(png.read_bytes()[:8]==b"\x89PNG\r\n\x1a\n" and sha(png)==row["png_sha256"],"PNG mismatch "+name)
    rec=load(root/(name[:-4]+".json"))
    scores=rec["scores"]
    events=rec["record"]["events"]
    check(len(scores)==len(events)==row["decisions"]==64,"decision count "+name)
    choices=Counter()
    seen=Counter()
    pressure=Counter()
    bursts=[]
    burst=0
    raw_illegal=0
    for i,(x,e) in enumerate(zip(scores,events)):
        o=x["obs"]
        legal=o["legal"]
        values=x["six_logits"]
        check(len(values)==6 and all(math.isfinite(v) for v in values),"bad logits "+name)
        check(bool(legal) and set(legal)<=set(ACTIONS),"bad host legal menu "+name)
        raw=ACTIONS[max(range(6),key=lambda j:values[j])]
        chosen=max(legal,key=lambda a:values[ACTIONS.index(a)])
        check(x["raw"]==raw and x["applied"]==chosen and x["raw_legal"]==(raw in legal),"score/choice mismatch "+name)
        check(e["observation"]==o and e["applied"]==chosen==e["raw_choice"] and e["raw_legal"] is True,"W0 event mismatch "+name)
        check(x["ray2"]==ray2(o),"derived ray mismatch "+name)
        choices[chosen]+=1
        seen[x["ray2"]]+=1
        raw_illegal+=raw not in legal
        if x["ray2"] in ("#","+"):
            pressure["encounters"]+=1
            if chosen in ("LEFT","RIGHT"):pressure["anticipatory_turns"]+=1
            if chosen=="BRANCH":pressure["branch_on_pressure"]+=1
        if chosen=="BRANCH": burst+=1
        elif burst:bursts.append(burst);burst=0
    if burst:bursts.append(burst)
    check(dict(choices)==row["choices"] and dict(seen)==row["ray2_counts"],"action/observation count "+name)
    check(choices["BRANCH"]==row["branches"] and choices["RETRACE"]==row["retraces"],"branch/retrace count "+name)
    check(raw_illegal==row["raw_illegal"] and row["host_illegal"]==0,"host/raw legality "+name)
    return choices,pressure,bursts,raw_illegal

def recover(root):
    fresh=root/"fresh-results"
    prior=root/"original-gp350010"/"original-experiment"
    s=load(fresh/"summary.json")
    o=load(prior/"summary.json")
    check(s["status"]=="complete" and s["source_sha"]==SOURCE,"GP011 exact source/complete")
    check(s["schema"]=="driv3r.lfm350.gp350011-checkpoint-sweep/1","GP011 schema")
    check(s["GP350010_exact_source_sha"]==PARENT and s["GP350010_original_artifact"]==14221088357,"GP010 custody")
    check(s["model"]=="LiquidAI/LFM2.5-350M" and s["revision"]==REVISION and not s["native"],"pinned model")
    check(s["new_optimizer_updates"]==0 and s["fresh_model_decisions"]==1536 and s["fresh_dev_probes"]==144 and s["original_pngs_new"]==24,"new evidence scope")
    check(set(s["fresh_checkpoint_stages"])==set(FRESH),"missing fresh checkpoint")
    check(o["status"]=="complete" and o["source_sha"]==PARENT and o["updates"]==o["backwards"]==72 and o["original_pngs"]==60,"GP010 prior")
    check(o["revision"]==REVISION and o["parent_adapter_sha256"]=="436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c","GP010 pretrained parent")
    for step,want in CHECKPOINTS.items():
        file=prior/("adapter-step-"+step)/"adapter_model.safetensors"
        check(sha(file)==want==o["adapter_sha256"]["adapter-step-"+step+"/adapter_model.safetensors"],"adapter hash "+step)
    result={}
    images=0
    decisions=0
    probes=0
    png_by_stage={}
    for stage in FRESH:
        info=s["fresh_checkpoint_stages"][stage]
        pr=load(fresh/(stage+"-probe.json"))
        check(len(pr["rows"])==pr["n"]==72 and pr["teacher_matches"]==info["teacher_matches"],"probe receipt "+stage)
        matched=0
        raw_bad=0
        for row in pr["rows"]:
            values=row["six_logits"]
            check(len(values)==6 and all(math.isfinite(v) for v in values),"bad probe logits")
            check(row["raw"]==ACTIONS[max(range(6),key=lambda i:values[i])],"probe raw score mismatch")
            check(row["raw_legal"]==(row["raw"]==row["legal"]),"probe raw legality mismatch")
            check(row["match"]==(row["legal"]==row["target"]),"probe target mismatch")
            matched+=row["match"]
            raw_bad+=not row["raw_legal"]
        check(matched==info["teacher_matches"] and raw_bad==pr["raw_illegal"],"probe score tally")
        probes+=72
        check(len(info["episodes"])==12,"missing 12 original episodes "+stage)
        family={}
        for fam in FAMILIES:
            episodes=[x for x in info["episodes"] if x["family"]==fam]
            check(len(episodes)==4,"wrong family count "+fam)
            count=Counter()
            pressure=Counter()
            bursts=[]
            rawbad=0
            unique=0
            for row in episodes:
                check(row["stage"]==stage and row["seed"] in (9503,9521) and row["size"] in (32,48),"seed/size/stage drift")
                choices,p,bad_bursts,illegals=verify_episode(fresh,row)
                count.update(choices)
                pressure.update(p)
                bursts.extend(bad_bursts)
                rawbad+=illegals
                unique+=row["unique_cells"]
                images+=1
                decisions+=64
                png_by_stage[(stage,row["size"],row["seed"],fam)]=row["png_sha256"]
            original=info["family"][fam]
            check(dict(count)==original["actions"] and dict(pressure)==original["pressure"],"family stats drift "+fam)
            check(original["raw_illegal"]==rawbad and original["host_illegal"]==0,"family illegal "+fam)
            check(original["mean_unique_cells"]==unique/4,"family coverage drift")
            check(original["branch_burst_max"]==max(bursts,default=0),"branch burst max")
            check(abs(original["branch_burst_mean"]-(sum(bursts)/len(bursts) if bursts else 0))<1e-9,"branch burst mean")
            family[fam]={"actions":dict(count),"pressure":dict(pressure),"mean_unique_cells":unique/4,"raw_illegal":rawbad,"branch_burst_max":max(bursts,default=0)}
        result[stage]={"heldout_teacher_matches":matched,"probe_raw_illegal":raw_bad,"family":family,"wall_seconds":info["wall_seconds"]}
    for key,rows in (("gp350006",o["baseline"]),("gp350010-step72",o["trained"])):
        check(len(rows)==12 and s["reused_immutable_original_reference"][key]["episodes"]==rows,"original reference evidence drift")
        for row in rows:
            check(sha(prior/row["png"])==row["png_sha256"],"original reference PNG mismatch")
            images+=1
    check(len(o["controls"])==36,"prior controls count")
    for row in o["controls"]:
        check(sha(prior/row["png"])==row["png_sha256"],"prior control PNG mismatch")
        images+=1
    check(images==84 and decisions==1536 and probes==144,"independent recovered evidence totals")
    different=sum(png_by_stage[(FRESH[0],n,seed,fam)]!=png_by_stage[(FRESH[1],n,seed,fam)]
                  for n in (32,48) for seed in (9503,9521) for fam in FAMILIES)
    return {"schema":"driv3r.lfm350.gp350011-recovered/1",
            "original_failed_run":RUN,"original_artifact":ARTIFACT,
            "original_source_sha":SOURCE,"original_parent_source_sha":PARENT,
            "model":"LiquidAI/LFM2.5-350M","revision":REVISION,
            "artifact_integrity":"source pinned; archive digest checked by workflow",
            "fresh_inference_original_scores_verified":decisions,
            "new_observation_probes_verified":probes,
            "original_png_sha256_verified":images,
            "original_adapter_sha256":CHECKPOINTS,
            "matched_step24_step48_pngs_different":different,
            "stage_results":result,"new_optimizer_updates":0,
            "host_illegal_actions":0,"d1_matched":"NOT RUN","native":False,
            "model_execution":"original run completed; no new model inference in recovery",
            "original_ci_failure":"post-inference Path/string concatenation TypeError"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    check(args.source.is_dir(),"downloaded immutable artifact missing")
    check(not args.out.exists(),"never overwrite immutable audit")
    r=recover(args.source)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"GP350011_RECOVERY":"PASS","fresh_inference_records":r["fresh_inference_original_scores_verified"],"PNG_sha256":r["original_png_sha256_verified"],"matched_changed_PNGs":r["matched_step24_step48_pngs_different"],"heldout":{s:x["heldout_teacher_matches"] for s,x in r["stage_results"].items()}}))
if __name__=="__main__":main()
