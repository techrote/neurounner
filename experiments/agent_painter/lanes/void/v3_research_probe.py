#!/usr/bin/env python3
"""VOID v3 SOURCE-ONLY research screen. No TRAIN admission, native Art or model run."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compiler import parse_program, compile_program, structural_signature, digest

MANIFEST="2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
TRAIN="5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1"
DEV="8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572"
COMPILER="4d8559805df1ff5f3f23277a7bca98e47fa8feb53337aa79e8546d0bb89132f6"
NATIVE="633920373a3d406a2ede61ce4137c324f76be23b"
EXPECTED={"void-v3-G01":"greebles","void-v3-G02":"greebles","void-v3-A01":"architecture","void-v3-P05-fitted":"pipework"}
EXPECTED_REVIEW={"void-v3-G01":"PENDING_GLOBAL_REVIEW",
                 "void-v3-G02":"PENDING_GLOBAL_REVIEW",
                 "void-v3-A01":"QUARANTINE_POTENTIAL_NEAR_ALIAS",
                 "void-v3-P05-fitted":"STYLE_VARIANT_NOT_NEW_GROUP"}
SEEDS=(104729,130363)
# Independent reviewer 2026-11-14: native-positive does not confer split
# independence. The paired-facing-room A01 is quarantined for near-alias risk.
QUARANTINED_NEAR_ALIAS=("void-v3-A01",)

def require_reference_manifest_sha256(screened):
    """Only publish the digest that the frozen-corpus screen actually verified."""
    actual=screened.get("original_manifest_sha256")
    if actual != MANIFEST:
        raise ValueError("missing/foreign verified original native manifest SHA-256")
    return actual


def facing_equal_room_near_alias(program):
    """Witness the risky *motif*, not graph equality with original held-outs.

    Two equal-level/equal-sized rooms with opposed facing side doors and
    horizontal circulation resembles original TRAIN paired_doorway_rooms and
    DEVELOPMENT bridge_between_towers. A different SHA is not split safety.
    """
    if program["family"] != "architecture":
        return False
    rooms=[c for c in program["components"] if c["kind"]=="room"]
    if len(rooms)!=2:
        return False
    left,right=sorted(rooms,key=lambda c:c["box"][0])
    lbox,rbox=left["box"],right["box"]
    doors_l=left.get("openings",[])
    doors_r=right.get("openings",[])
    route=program.get("requirements",{}).get("route",[])
    return bool(
        lbox[1]==rbox[1] and lbox[2:]==rbox[2:]
        and len(doors_l)==len(doors_r)==1
        and doors_l[0]["side"]=="right" and doors_r[0]["side"]=="left"
        and len(route)==2 and route[0][1]==route[1][1]
    )

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(corpus, proposals, old_proposals):
    # TRAIN and DEVELOPMENT only: not even open FINAL JSONL, images or targets.
    mpath=corpus/"admitted-corpus.json"
    if mpath.is_symlink() or sha(mpath)!=MANIFEST: raise ValueError("immutable native manifest mismatch")
    m=json.loads(mpath.read_text())
    if m["source_sha"]!=NATIVE or m["state"]!="native-admitted": raise ValueError("native source mismatch")
    historical=set()
    for split,expected in (("TRAIN",TRAIN),("DEVELOPMENT",DEV)):
        meta=m["split_files"][split]; p=corpus/(split.lower()+".jsonl")
        if p.is_symlink() or meta["path"]!=p.name or meta["count"]!=12 or meta["sha256"]!=expected or sha(p)!=expected: raise ValueError("original split custody mismatch: "+split)
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        if len(rows)!=12 or len({r["group_id"] for r in rows})!=6 or any(r["split"]!=split for r in rows): raise ValueError("original groups differ: "+split)
        historical.update(r["structural_signature_sha256"] for r in rows)
    d=json.loads(proposals.read_text())
    if (d["schema"]!="void-three-family-v3-research-proposals/1" or d["state"]!="unadmitted-research"
            or d["original_native_source"]!=NATIVE or d["original_manifest_sha256"]!=MANIFEST
            or d["compiler_sha256"]!=COMPILER or d["new_shared_train_groups"]!=0
            or d["model_optimizer_updates"]!=0 or d["development_target_usage"] is not False
            or d["final_targets_opened"] is not False):
        raise ValueError("v3 status/source mismatch")
    old=json.loads(old_proposals.read_text())
    oldp=next(x["program"] for x in old["candidates"] if x["id"]=="void-pipe-P05")
    oldsig=digest(structural_signature(oldp))
    if len(d["candidates"])!=4 or {c["id"] for c in d["candidates"]}!=set(EXPECTED): raise ValueError("candidate IDs differ")
    signatures=set(); out=[]
    for c in d["candidates"]:
        name=c["id"]
        if c["training_admitted"] is not False: raise ValueError("false TRAIN admission")
        if c.get("review_status") != EXPECTED_REVIEW[name]: raise ValueError("global review/quarantine disposition changed")
        p=parse_program(json.dumps(c["program"],sort_keys=True,separators=(",",":")))
        if p["family"]!=EXPECTED[name]: raise ValueError("wrong family")
        sig=digest(structural_signature(p))
        if name=="void-v3-A01" and not facing_equal_room_near_alias(p):
            raise ValueError("A01 quarantine witness changed; require new peer split review")
        if name=="void-v3-P05-fitted":
            if (c["classification"]!="style-variant-not-a-new-train-group"
                    or c["same_graph_as"]!="void-pipe-P05" or sig!=oldsig
                    or not p["components"][0]["fittings"] or p["components"][0]["diameter"]!=3):
                raise ValueError("P05 fitted style is not identical structural graph")
        else:
            if c["classification"]!="new-composition-hypothesis" or sig in signatures or sig in historical:
                raise ValueError("original public structural signature collision: "+name)
            signatures.add(sig)
        views=[]
        for seed in SEEDS:
            v=compile_program(p,seed=seed,focus=(0,0,96,96),material=58,allowed_materials=(58,),max_operations=512,max_cell_work=100000)
            s=v["semantic"]; metrics=v["structural_preflight"]["metrics"]
            if v["compiler_sha256"]!=COMPILER or not v["structural_preflight"]["passed"]: raise ValueError("compiler/semantic contract differs")
            if p["family"]=="greebles" and (metrics["panel_count"]!=len(p["components"]) or metrics["detail_kinds"]<4 or metrics["negative_space_percent"]<65):
                raise ValueError("greeble panel/detail/negative-space witness failed")
            if p["family"]=="architecture" and (metrics["room_count"]!=2 or metrics["opening_count"]!=2 or not metrics.get("clearance_route_passed")):
                raise ValueError("architecture native-equivalent route/opening preflight failed")
            if p["family"]=="pipework" and (not s["fittings"] or len(s["endpoints"])!=1 or len(s["junctions"])!=2 or s["expected_cycles"]!=2 or metrics["diameter_probes_passed"]!=len(s["edges"])):
                raise ValueError("pipe fitting/topology/diameter witness failed")
            views.append({"seed":seed,"operations_sha256":v["operations_sha256"],"rectangles":v["operation_count"],"cells":v["unique_cells"],"metrics":metrics,"fitting_count":len(s.get("fittings",[])),"native_executed":False})
        out.append({"id":name,"program_sha256":digest(p),"signature_sha256":sig,"classification":c["classification"],"original_public_signature_collision":sig in historical,"global_peer_review":"pending","near_alias_review":("QUARANTINE_POTENTIAL_NEAR_ALIAS" if name in QUARANTINED_NEAR_ALIAS else "PENDING"),"views":views})
    return {"status":"SOURCE_ONLY_PASS","cases":out,"original_manifest_sha256":MANIFEST,
            "quarantined_candidate_ids":list(QUARANTINED_NEAR_ALIAS),
            "globally_admitted_signatures":0,"putative_new_signatures":len(signatures),
            "styles_not_new_groups":1,"seed_preflights":len(out)*2,
            "original_groups":{"TRAIN":6,"DEVELOPMENT":6},"native_positive":0,
            "new_train_groups":0,"optimizer_updates":0,"model_generated_paintings":0,"final_opened":False}

def main():
    a=argparse.ArgumentParser()
    for name in ("corpus","proposals","old-proposals","output"): a.add_argument("--"+name,required=True,type=Path)
    x=vars(a.parse_args()); output=x.pop("output"); output.parent.mkdir(parents=True,exist_ok=True)
    try: result=verify(x["corpus"],x["proposals"],x["old_proposals"])
    except Exception as error: result={"status":"FAIL","error":str(error),"native_positive":0,"new_train_groups":0,"optimizer_updates":0,"final_opened":False}
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:result.get(k) for k in ("status","error","seed_preflights")}))
    return 0 if result["status"]=="SOURCE_ONLY_PASS" else 1

if __name__=="__main__": raise SystemExit(main())
