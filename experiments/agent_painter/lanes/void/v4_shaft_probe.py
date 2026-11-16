#!/usr/bin/env python3
"""VOID A02 single-vertical-shaft source-only pilot. No FINAL, training or native."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compiler import compile_program, parse_program, structural_signature, digest
from v3_research_probe import MANIFEST, TRAIN, DEV, COMPILER, NATIVE, sha, facing_equal_room_near_alias

SEEDS = (104729, 130363)
MODEL = "dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563"
SCHEMA = "void-architecture-shaft-v4/1"
STATUS = "PENDING_GLOBAL_SPATIAL_AND_HELDOUT_REVIEW"

def shaft_witness(p):
    """Structural precheck; *not* proof of independent global composition or playability."""
    if p.get("family") != "architecture": return False
    c = p.get("components", [])
    rooms = [x for x in c if x.get("kind") == "room"]
    decks = [x for x in c if x.get("kind") == "platform"]
    supports = [x for x in c if x.get("kind") == "support"]
    if len(rooms)!=1 or len(decks)!=3 or len(supports)!=2 or len(c)!=6: return False
    room = rooms[0]
    x,y,w,h = room["box"]
    if h < 70 or h <= w+25 or room["thickness"] != 2: return False
    if [o["side"] for o in room["openings"]] != ["top","bottom"]: return False
    if any(o["size"]<7 or o["offset"] != 17 for o in room["openings"]): return False
    if facing_equal_room_near_alias(p): return False
    r=p.get("requirements",{})
    if r.get("clearance")!=5 or r.get("route")!=[[48,17],[48,77]]: return False
    if r.get("region")!=[24,4,52,88]: return False
    if [z["box"][1] for z in decks] != [28,45,62]: return False
    if [z["box"][0] for z in decks] != [31,55,31]: return False
    # The 5-wide clear core spans columns 46..50 at its canonical route x=48.
    if any(z["box"][0] <= 50 and z["box"][0]+z["box"][2] > 46 for z in decks+supports):
        return False
    return True

def verify(corpus: Path, proposals: Path, previous: Path):
    # Never open original FINAL JSONL/images/targets. TRAIN/DEV metadata only.
    manifest = corpus/"admitted-corpus.json"
    if manifest.is_symlink() or sha(manifest)!=MANIFEST:
        raise ValueError("original native corpus manifest is not immutable")
    m=json.loads(manifest.read_text(encoding="utf-8"))
    if m["source_sha"]!=NATIVE or m["state"]!="native-admitted":
        raise ValueError("original native source/state mismatch")
    prior=set()
    splits={}
    for split,expected in (("TRAIN",TRAIN),("DEVELOPMENT",DEV)):
        path=corpus/(split.lower()+".jsonl")
        meta=m["split_files"][split]
        if path.is_symlink() or meta["path"]!=path.name or meta["sha256"]!=expected or meta["count"]!=12 or sha(path)!=expected:
            raise ValueError("frozen public split custody mismatch: "+split)
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows)!=12 or len({row["group_id"] for row in rows})!=6 or any(row["split"]!=split for row in rows):
            raise ValueError("frozen public structural grouping mismatch")
        prior.update(row["structural_signature_sha256"] for row in rows)
        splits[split]={"rows":len(rows),"structural_groups":len({row["group_id"] for row in rows}),"sha256":expected}
    old=json.loads(previous.read_text(encoding="utf-8"))
    oldA=next(z for z in old["candidates"] if z["id"]=="void-v3-A01")
    if oldA["review_status"]!="QUARANTINE_POTENTIAL_NEAR_ALIAS" or oldA["training_admitted"] is not False:
        raise ValueError("A01 quarantine is not preserved")
    oldsign=digest(structural_signature(oldA["program"]))
    doc=json.loads(proposals.read_text(encoding="utf-8"))
    if (doc["schema"]!=SCHEMA or doc["state"]!="unadmitted-research"
        or doc["model"]!=MODEL or doc["original_native_source"]!=NATIVE
        or doc["original_manifest_sha256"]!=MANIFEST or doc["compiler_sha256"]!=COMPILER
        or doc["new_shared_train_groups"]!=0 or doc["model_optimizer_updates"]!=0
        or doc["final_targets_opened"] is not False or doc["development_targets_used"] is not False):
        raise ValueError("source, model, split or no-training identity changed")
    if len(doc["candidates"])!=1: raise ValueError("exactly one novel v4 hypothesis required")
    c=doc["candidates"][0]
    if (c["id"]!="void-v4-A02" or c["family"]!="architecture" or
        c["review_status"]!=STATUS or c["training_admitted"] is not False or
        c["native_qualified"] is not False or c["classification"]!="single-vertical-shaft-new-composition-hypothesis"):
        raise ValueError("v4 candidate/review disposition tampered")
    p=parse_program(json.dumps(c["program"],sort_keys=True,separators=(",",":")))
    if not shaft_witness(p): raise ValueError("shaft geometry/intent guard refused")
    sig=digest(structural_signature(p))
    if sig in prior or sig==oldsign: raise ValueError("public or quarantined A01 structural signature collision")
    views=[]
    for seed in SEEDS:
        compiled=compile_program(p,seed=seed,focus=(0,0,96,96),material=58,
                                 allowed_materials=(58,),max_operations=512,max_cell_work=100000)
        metrics=compiled["structural_preflight"]["metrics"]
        if (compiled["compiler_sha256"]!=COMPILER or
            not compiled["structural_preflight"]["passed"] or
            metrics["room_count"]!=1 or metrics["opening_count"]!=2 or
            not metrics["clearance_route_passed"] or metrics["clearance_cells"]!=5):
            raise ValueError("strict two-seed source route/opening/clearance failed")
        views.append({"seed":seed,"program_sha256":compiled["program_sha256"],
            "operations_sha256":compiled["operations_sha256"],
            "rectangles":compiled["operation_count"],"material_cells":compiled["unique_cells"],
            "clearance_route_passed":True,"native_executed":False})
    return {"status":"SOURCE_ONLY_PASS","source_model":MODEL,
        "original_manifest_sha256":MANIFEST,"original_split_custody":splits,
        "candidate_id":c["id"],"signature_sha256":sig,
        "quarantined_old_architecture":"void-v3-A01",
        "review_status":STATUS,"original_public_exact_signature_collision":False,
        "global_independence_approved":False,"compiler_seed_views":views,
        "seed_preflights":2,"independent_composition_hypotheses":1,
        "native_positive_views":0,"new_train_groups":0,"optimizer_updates":0,
        "model_generated_paintings":0,"final_opened":False}

def main():
    a=argparse.ArgumentParser()
    for name in ("corpus","proposals","previous","output"):
        a.add_argument("--"+name,required=True,type=Path)
    x=a.parse_args()
    x.output.parent.mkdir(parents=True,exist_ok=True)
    try: r=verify(x.corpus,x.proposals,x.previous)
    except Exception as exc:
        r={"status":"FAIL","error":str(exc),"new_train_groups":0,
           "native_positive_views":0,"optimizer_updates":0,"final_opened":False}
    x.output.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(r,sort_keys=True))
    return 0 if r["status"]=="SOURCE_ONLY_PASS" else 1

if __name__=="__main__": raise SystemExit(main())
