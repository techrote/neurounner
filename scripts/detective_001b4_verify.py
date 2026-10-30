#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import detective_001b4_lfm as s
class ReadbackError(ValueError): pass
def need(x,m):
    if not x: raise ReadbackError(m)
def verify(d:Path):
    spec,case,_=s.load_spec(); pl=s.plan()
    files={"README.md","manifest.json","raw/run.json","provenance/source-artifacts.json","provenance/hashes.sha256"}
    present={p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}; need(present==files,"file set")
    inv=(d/"provenance/hashes.sha256").read_text().splitlines()
    need(inv==[f"{s.sha_file(d/r)}  {r}" for r in sorted(files-{"provenance/hashes.sha256"})],"hash inventory")
    run=s.read_json(d/"raw/run.json"); man=s.read_json(d/"manifest.json"); prov=s.read_json(d/"provenance/source-artifacts.json")
    need(run["complete"] is True and run["admission"]=="EXPLORATORY_NOT_ADMITTED","run state")
    need(run["plan_sha256"]==s.sha_obj(pl) and run["spec_sha256"]==s.sha_file(s.SPEC) and run["case_sha256"]==s.sha_file(s.CASE) and run["script_sha256"]==s.sha_file(Path(s.__file__)),"source hashes")
    need(run["model"]==spec["model"] and run["runtime_pin"]==spec["runtime"] and run["model_sha256_verified"]==spec["model"]["sha256"],"model pin")
    need(re.fullmatch(r"[0-9a-f]{40}",run["repository"].get("head_sha","")) is not None and str(run["repository"].get("run_id","")).isdigit(),"run identity")
    need(len(run["raw"])==len(run["results"])==5,"task count")
    total=0; replay=[]
    for task,raw,stored in zip(spec["tasks"],run["raw"],run["results"]):
        need(raw["task_id"]==task["id"] and raw["mode"]==task["mode"],"task identity")
        need(raw["prompt"]==s.prompt(task) and raw["prompt_sha256"]==__import__("hashlib").sha256(s.prompt(task).encode()).hexdigest(),"prompt drift")
        need(raw["request"]["messages"]==[{"role":"user","content":s.prompt(task)}],"API prompt drift")
        need(raw["request"]["max_tokens"]==96 and raw["request"]["temperature"]==0.0 and raw["request"]["top_p"]==1.0 and raw["request"]["seed"]==24301,"decode drift")
        need(isinstance(raw["response"],str) and isinstance(raw["completion_tokens"],int) and 0<=raw["completion_tokens"]<=96,"raw output")
        sc=s.score(spec,case,task,raw["response"]); need(sc==stored,"score replay"); replay.append(sc); total+=raw["completion_tokens"]
    need(total==run["total_generated_tokens"] and total<=480,"token budget")
    summary={"tasks":5,"schema_valid":sum(x.get("schema_valid") is True for x in replay),"selection_correct":sum(x.get("selection_correct") is True for x in replay),"result_use_success":sum(x.get("result_use_success") is True for x in replay),"generated_tokens":total,"training_steps":0}
    need(man["summary"]==summary and man["run_sha256"]==s.sha_obj(run) and man["source_head_sha"]==run["repository"]["head_sha"],"manifest")
    need(prov["head_sha"]==run["repository"]["head_sha"] and prov["model_sha256_verified"]==spec["model"]["sha256"] and prov["script_sha256"]==s.sha_file(Path(s.__file__)),"provenance")
    return {"verdict":"SOURCE_BOUND_LFM_CONTROL_READBACK_PASS",**summary,"training_performed":False,"wire_matched_to_sprocket":False}
def main():
    a=argparse.ArgumentParser(); a.add_argument("--dataset",type=Path,required=True); x=a.parse_args(); print(json.dumps(verify(x.dataset),indent=2,sort_keys=True))
if __name__=="__main__": main()
