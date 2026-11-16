#!/usr/bin/env python3
"""VOID v5: frozen A02 support correction, source-only. Never opens sealed FINAL."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from compiler import compile_program,parse_program
from v4_shaft_probe import verify as verify_original, shaft_witness, SEEDS, MANIFEST, MODEL, COMPILER

ROOT=Path(__file__).resolve().parent
OLD_SHA="d8f8d0ff810e006d6a52e38396790a73f15b9999e73fe0b59ca110c985cc1cd9"
OLD_OPS="e63e1cabe640a3b0b78e876258a780e36e19aa8251f68b8973d7e808034e5cf8"
ANCHORS=[[33,30,2,52],[63,47,2,35]]
def witness(old,new):
    if not shaft_witness(old) or not shaft_witness(new): return False
    expected=copy.deepcopy(old)
    for i,box in enumerate(ANCHORS,start=4): expected["components"][i]["box"]=box
    if new != expected: return False
    floor=old["components"][0]["box"][1]+old["components"][0]["box"][3]-old["components"][0]["thickness"]
    for deck,support in zip(new["components"][1:3],new["components"][4:]):
        dx,dy,dw,dh=deck["box"]
        sx,sy,sw,sh=support["box"]
        if not (sy==dy+dh and sy+sh==floor and dx<=sx and sx+sw<=dx+dw and (sx+sw<=46 or sx>=51)):
            return False
    return True

def verify(corpus,proposals,parent,prior):
    original_receipt=verify_original(corpus,parent,prior)
    if original_receipt["status"]!="SOURCE_ONLY_PASS":
        raise ValueError("original A02 source not qualified")
    old_doc=json.loads(parent.read_text(encoding="utf-8"))
    old=old_doc["candidates"][0]["program"]
    old_comp=compile_program(old,seed=104729,focus=(0,0,96,96),material=58,allowed_materials=(58,))
    if old_comp["program_sha256"]!=OLD_SHA or old_comp["operations_sha256"]!=OLD_OPS:
        raise ValueError("immutable original program/operations changed")
    doc=json.loads(proposals.read_text(encoding="utf-8"))
    p=doc.get("parent",{})
    if (doc.get("schema")!="void-architecture-floor-anchored-v5/1" or
        doc.get("state")!="unadmitted-research" or doc.get("model")!=MODEL or
        doc.get("original_native_source")!=old_doc["original_native_source"] or
        doc.get("original_manifest_sha256")!=MANIFEST or doc.get("compiler_sha256")!=COMPILER or
        doc.get("new_shared_train_groups")!=0 or doc.get("model_optimizer_updates")!=0 or
        doc.get("final_targets_opened") is not False or doc.get("development_targets_used") is not False or
        p.get("id")!="void-v4-A02" or p.get("original_program_sha256")!=OLD_SHA or
        p.get("native_run")!=43592303518 or p.get("native_artifact_id")!=14113396057 or
        p.get("native_producer")!="4c1a70f806c6ba173098cee22fce770b3c4662ae"):
        raise ValueError("v5 source, parent, split or no-training contract changed")
    if len(doc.get("candidates",[]))!=1: raise ValueError("one nonindependent revision required")
    c=doc["candidates"][0]
    if (c.get("id")!="void-v5-A02-floor-anchored" or c.get("parent_id")!="void-v4-A02" or
        c.get("classification")!="NONINDEPENDENT_SUPPORT_REVISION" or
        c.get("review_status")!="PENDING_NEW_NATIVE_AND_GLOBAL_REVIEW" or
        c.get("training_admitted") is not False or c.get("native_qualified") is not False):
        raise ValueError("v5 candidate misclassified")
    new=parse_program(json.dumps(c["program"],separators=(",",":")))
    if not witness(old,new): raise ValueError("support floor anchoring or parent geometry violated")
    views=[]
    for seed in SEEDS:
        result=compile_program(new,seed=seed,focus=(0,0,96,96),material=58,
                        allowed_materials=(58,),max_operations=512,max_cell_work=100000)
        m=result["structural_preflight"]["metrics"]
        if (not result["structural_preflight"]["passed"] or result["compiler_sha256"]!=COMPILER or
            m["room_count"]!=1 or m["opening_count"]!=2 or
            not m["clearance_route_passed"] or m["clearance_cells"]!=5 or
            not 524<result["unique_cells"]<800):
            raise ValueError("failed new source geometry/clearance check")
        views.append({"seed":seed,"program_sha256":result["program_sha256"],
                      "operations_sha256":result["operations_sha256"],
                      "compiler_material_cells":result["unique_cells"],
                      "native_executed":False})
    return {"status":"SOURCE_ONLY_PASS","model":MODEL,"parent_program_sha256":OLD_SHA,
            "original_manifest_sha256":MANIFEST,"original_public_splits":original_receipt["original_split_custody"],
            "original_A01_quarantined":original_receipt["quarantined_old_architecture"],
            "source_views":views,"floor_anchored_posts":2,"central_clearance_cells":5,
            "independent_new_structures":0,"native_positive_views":0,
            "train_admitted_groups":0,"optimizer_steps":0,"model_generated_paintings":0,
            "final_opened":False,"global_review":"HOLD"}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try: r=verify(a.corpus,ROOT/"v5_support_proposals.json",ROOT/"v4_shaft_proposals.json",ROOT/"v3_research_proposals.json")
    except Exception as e: r={"status":"FAIL","error":str(e),"native_positive_views":0,"train_admitted_groups":0,"optimizer_steps":0,"final_opened":False}
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(r,sort_keys=True))
    return 0 if r["status"]=="SOURCE_ONLY_PASS" else 1
if __name__=="__main__": raise SystemExit(main())
