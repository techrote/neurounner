#!/usr/bin/env python3
"""Read-only v5 service-slot check against original source-built Art material bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

HERE=Path(__file__).resolve().parent
PROGRAM_SHA="9df29e3d7f5d97d68ee69708882e3bd6b7b19a5ee3d019ca0a2a86145cc1aeba"
CROP_SHA="14738d7bd6747f2137c2604e1e7afa615622a7d31c2adf785bd64f95e3c67cf7"
PRODUCER="64b5f1ca5b0f65ffbf9a7645a0d0828d35657f08"
CORPUS="2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
MODEL="dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563"
SEEDS=(104729,130363)

def need(value,message):
    if not value: raise ValueError(message)

def sha(data): return hashlib.sha256(data).hexdigest()

def program_witness(d):
    need(d.get("model")==MODEL and d.get("original_manifest_sha256")==CORPUS,"foreign model/corpus")
    need(d.get("new_shared_train_groups")==d.get("model_optimizer_updates")==0,"unapproved shared TRAIN/optimizer claim")
    need(d.get("final_targets_opened") is False,"FINAL access claimed")
    rows=d.get("candidates",[])
    need(len(rows)==1 and rows[0].get("classification")=="NONINDEPENDENT_SUPPORT_REVISION"
         and rows[0].get("training_admitted") is False and rows[0].get("native_qualified") is False,
         "original research classification required")
    p=rows[0]["program"]
    digest=sha(json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
    need(digest==PROGRAM_SHA,"original v5 program changed")
    parts=p["components"]
    need([part["kind"] for part in parts]==["room","platform","platform","platform","support","support"],
         "unexpected architecture layout")
    room=parts[0]
    need(room["box"]==[28,8,42,76] and room["thickness"]==2
         and parts[4]["box"]==[33,30,2,52] and parts[5]["box"]==[63,47,2,35],
         "room/posts changed")
    inside_left=room["box"][0]+room["thickness"]
    inside_right=room["box"][0]+room["box"][2]-room["thickness"]-1
    widths=[parts[4]["box"][0]-inside_left,
            inside_right-(parts[5]["box"][0]+parts[5]["box"][2])+1]
    floor=room["box"][1]+room["box"][3]-room["thickness"]
    need(widths==[3,3] and p["requirements"]["clearance"]==5
         and floor==82 and all(c["box"][1]+c["box"][3]==floor for c in parts[4:]),
         "non-traversable slot width/floor contact contract")
    return {"source_program_sha256":digest,"side_service_slot_widths":widths,
            "central_geometric_clearance":5,"classification":"NON_TRAVERSABLE_SERVICE_SLOTS"}

def material_witness(values):
    need(len(values)==96*96 and set(values)<={0,58},"invalid original crop")
    def get(x,y): return values[y*96+x]
    for y in range(65,82):
        for x in range(30,33): need(get(x,y)==0,"left service slot occupied")
        for x in range(65,68): need(get(x,y)==0,"right service slot occupied")
        for x in (29,33,34,63,64,68): need(get(x,y)==58,"wall/post missing")
    for x in (33,34,63,64):
        need(get(x,81)==get(x,82)==58,"post-floor contact missing")
    for y in range(17,78):
        for x in range(46,51): need(get(x,y)==0,"central route obstructed")
    need(values.count(58)==662,"original native material count changed")
    return {"side_slots":[[30,65,3,17],[65,65,3,17]],
            "traversable_side_rooms":False,"original_hard_wall_cells":662,
            "central_geometric_clearance_cells":5,
            "physics_load_or_jump_reachability_proven":False}

def member(root,name):
    need(isinstance(name,str) and bool(name),"missing member")
    root=root.resolve()
    p=(root/name).resolve()
    need(p.is_relative_to(root) and p.is_file(),"foreign evidence member")
    return p

def native_audit(root):
    matches=list(root.rglob("native-verification.json"))
    need(len(matches)==1,"exactly one original native report required")
    packet=matches[0].parent
    report=json.loads(matches[0].read_text())
    need(report.get("passed") is True and report.get("source_sha")==PRODUCER
         and not report.get("failures"),"original native qualification/source mismatch")
    gates=report.get("global_gates",{})
    need(len(gates)==16 and all(value is True for value in gates.values()),"native safety gates incomplete")
    rows=report.get("cases",[])
    need(len(rows)==2 and {r.get("id") for r in rows}==
         {"void-v5-A02-floor-anchored-s"+str(s) for s in SEEDS},"host-seed cases changed")
    for r in rows:
        need(r.get("success") is True and r.get("filled_cells")==662
             and r.get("affected_cells")==r.get("acknowledged_cells")==662
             and r.get("program_sha256")==PROGRAM_SHA
             and r.get("split")=="UNADMITTED_TRAIN_PROPOSAL","native case/receipt mismatch")
        data=member(packet,r["raw_materials"]).read_bytes()
        need(len(data)==96*96*4 and sha(data)==CROP_SHA==r.get("readback_sha256"),
             "native material bytes differ from immutable crop")
        for name,key in (("png","image_sha256"),("receipt","receipt_sha256")):
            need(sha(member(packet,r[name]).read_bytes())==r.get(key),"original image/receipt hash mismatch")
        material_witness(list(struct.unpack("<"+"i"*(96*96),data)))
    return {"original_run_id":43601866007,"original_artifact_id":14116463694,
            "original_native_producer":PRODUCER,"original_crop_sha256":CROP_SHA,
            "original_zip_sha256_previously_qualified":"d8f1c1dbce65ef14d392a7bd557bd89e8af21563da14d4a5a61b84386285944b",
            "full_zip_rehashed_this_run":False,
            "independently_checked_original_native_views":2,"new_native_art_operations":0}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-artifact",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        p=program_witness(json.loads((HERE/"v5_support_proposals.json").read_text()))
        n=native_audit(a.original_artifact)
        result={"status":"READ_ONLY_SEMANTICS_PASS","global_id":"ARCH-568-R01",
                "program":p,"native":n,"admission":"REVIEW_ONLY",
                "shared_train_groups":0,"optimizer_steps":0,
                "model_paintings":0,"frozen_useful_model":False,"final_opened":False,
                "next_gate":"shepherd global split and positive admission review"}
    except (ValueError,KeyError,TypeError,OSError,struct.error) as e:
        result={"status":"FAIL","error":str(e),"shared_train_groups":0,"optimizer_steps":0,
                "final_opened":False}
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"]=="READ_ONLY_SEMANTICS_PASS" else 1

if __name__=="__main__": raise SystemExit(main())
