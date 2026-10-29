#!/usr/bin/env python3
"""DETECTIVE-001R1 independent relevance review custody/aggregation.

No model inference and no scoring by this script. It verifies frozen opaque
packets against raw evidence, emits identity-free rating templates, and only
reveals source mapping after complete external integer scores are supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
C2=ROOT/"experiments/detective-001/training-c2-evidence"
C3=ROOT/"experiments/detective-001/training-c3-evidence"


class ReviewError(ValueError):
    pass


def canonical(v:Any)->str:
    return json.dumps(v,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False)


def sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()


def read(p:Path)->Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write(p:Path,v:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,sort_keys=True,
                            ensure_ascii=False)+"\n",encoding="utf-8")


def packet_paths(label:str)->tuple[Path,Path]:
    if label=="C2": return C2/"relevance-packet.json",C2/"raw-results.json"
    if label=="C3": return C3/"relevance-packet.json",C3/"raw-results.json"
    raise ReviewError("packet must be C2 or C3")


def reconstruct(label:str)->tuple[dict,dict[str,dict]]:
    packet_path,raw_path=packet_paths(label)
    packet=read(packet_path);raw=read(raw_path)
    rows=[]
    if label=="C2":
        if raw.get("schema")!="detective-c2-evaluation-raw-v1":
            raise ReviewError("unexpected C2 raw schema")
        for lane in raw.get("lanes",[]):
            if lane.get("lane_kind") not in {"parent","trained"}:
                continue
            if lane.get("member_id") not in {"V-G02-A","V-G09-C"}:
                raise ReviewError("unexpected C2 member")
            refs=lane.get("reference_responses")
            if not isinstance(refs,list) or len(refs)!=4:
                raise ReviewError("C2 reference cardinality mismatch")
            for r in refs:
                rows.append({
                    "member_id":lane["member_id"],
                    "lane_kind":lane["lane_kind"],
                    "scenario_id":r["scenario_id"],
                    "prompt":r["prompt"],"response":r["response"]
                })
        rows.sort(key=lambda x:(x["prompt"],x["response"],x["member_id"],
                                x["lane_kind"],x["scenario_id"]))
        expected_schema="detective-c2-post-blind-relevance-v1"
        expected_n=16
    else:
        if raw.get("schema")!="detective-c3-evaluation-raw-v1":
            raise ReviewError("unexpected C3 raw schema")
        for lane in raw.get("lanes",[]):
            if lane.get("member_id")!="V-G02-A" or lane.get("lane_kind") not in {"parent","trained"}:
                continue
            refs=lane.get("reference_responses")
            if not isinstance(refs,list) or len(refs)!=4:
                raise ReviewError("C3 reference cardinality mismatch")
            for r in refs:
                rows.append({
                    "member_id":"V-G02-A",
                    "lane_kind":lane["lane_kind"],
                    "scenario_id":r["scenario_id"],
                    "prompt":r["prompt"],"response":r["response"]
                })
        rows.sort(key=lambda x:(x["prompt"],x["response"],x["lane_kind"],
                                x["scenario_id"]))
        expected_schema="detective-c3-blind-relevance-v1"
        expected_n=8
    if len(rows)!=expected_n:
        raise ReviewError(label+" reconstructed item count mismatch")
    mapping={}
    expected_items=[]
    for i,row in enumerate(rows,1):
        item_id=f"R{i:03d}"
        mapping[item_id]={
            "member_id":row["member_id"],
            "lane_kind":row["lane_kind"],
            "scenario_id":row["scenario_id"]
        }
        expected_items.append({
            "item_id":item_id,"prompt":row["prompt"],
            "response":row["response"],"score_0_to_4":None,
            "reviewer_note":""
        })
    if (packet.get("schema")!=expected_schema
            or packet.get("identity_mapping_committed") is not False
            or packet.get("items")!=expected_items):
        raise ReviewError(label+" opaque packet does not reconstruct from raw evidence")
    rendered=canonical(packet)
    for secret in ("V-G02-A","V-G09-C",'"member_id"','"lane_kind"'):
        if secret in rendered:
            raise ReviewError(label+" opaque packet leaks source identity")
    return packet,mapping


def template(label:str)->dict:
    packet,_=reconstruct(label)
    return {
        "schema":"detective-independent-relevance-ratings-v1",
        "experiment":"DETECTIVE-001R1",
        "issue":121,
        "packet":label,
        "packet_sha256":sha_file(packet_paths(label)[0]),
        "reviewer_label":"",
        "rubric":packet["rubric"],
        "reviewer_instruction":packet["reviewer_instruction"],
        "ratings":[{
            "item_id":x["item_id"],"score_0_to_4":None,"note":""
        } for x in packet["items"]]
    }


def validate_ratings(label:str,path:Path)->tuple[dict,dict[str,dict]]:
    expected=template(label)
    actual=read(path)
    if (actual.get("schema")!=expected["schema"]
            or actual.get("experiment")!="DETECTIVE-001R1"
            or actual.get("issue")!=expected["issue"]
            or actual.get("packet")!=label
            or actual.get("packet_sha256")!=expected["packet_sha256"]):
        raise ReviewError(label+" rating source identity/hash mismatch")
    if not isinstance(actual.get("reviewer_label"),str):
        raise ReviewError("reviewer_label must be text")
    rows=actual.get("ratings")
    if not isinstance(rows,list) or len(rows)!=len(expected["ratings"]):
        raise ReviewError(label+" requires every rating")
    expected_ids=[x["item_id"] for x in expected["ratings"]]
    if [x.get("item_id") for x in rows]!=expected_ids:
        raise ReviewError(label+" rating order/item IDs changed")
    seen=set()
    for r in rows:
        if set(r)!={"item_id","score_0_to_4","note"}:
            raise ReviewError("rating row field drift")
        if r["item_id"] in seen: raise ReviewError("duplicate rating")
        seen.add(r["item_id"])
        score=r["score_0_to_4"]
        if type(score) is not int or not 0<=score<=4:
            raise ReviewError("every score must be integer 0..4")
        if not isinstance(r["note"],str):
            raise ReviewError("rating note must be text")
    _,mapping=reconstruct(label)
    return actual,mapping


def means(label:str,ratings:dict,mapping:dict[str,dict])->dict:
    groups={}
    for row in ratings["ratings"]:
        src=mapping[row["item_id"]]
        key=(src["member_id"],src["lane_kind"])
        groups.setdefault(key,[]).append(row["score_0_to_4"])
    result={}
    expected=(("V-G02-A","parent"),("V-G02-A","trained"))
    if label=="C2":
        expected+=(("V-G09-C","parent"),("V-G09-C","trained"))
    for key in expected:
        values=groups.get(key)
        if values is None or len(values)!=4:
            raise ReviewError(label+" hidden mapping/rating group incomplete "+repr(key))
        result[f"{key[0]}:{key[1]}"]={
            "mean":sum(values)/4,"scores":values,"items":4
        }
    return result


def aggregate(c2_path:Path,c3_path:Path)->dict:
    c2,c2map=validate_ratings("C2",c2_path)
    c3,c3map=validate_ratings("C3",c3_path)
    m2=means("C2",c2,c2map);m3=means("C3",c3,c3map)
    g02_c1=m2["V-G02-A:parent"]["mean"]
    g02_c2=m2["V-G02-A:trained"]["mean"]
    g02_c3_parent=m3["V-G02-A:parent"]["mean"]
    g02_c3=m3["V-G02-A:trained"]["mean"]
    g09_c1=m2["V-G09-C:parent"]["mean"]
    g09_c2=m2["V-G09-C:trained"]["mean"]
    # The C2 G02 model was regenerated in C3; deterministic greedy outputs
    # should normally agree, but relevance custody does not require text/score
    # identity across separate inference runs. Report both means.
    local_g02_ok=g02_c3>=g02_c3_parent-0.5
    cumulative_g02_ok=g02_c3>=g02_c1-0.5
    g09_ok=g09_c2>=g09_c1-0.5
    expansion_relevance_gate=cumulative_g02_ok and g09_ok
    return {
        "schema":"detective-independent-relevance-summary-v1",
        "experiment":"DETECTIVE-001R1","issue":121,
        "packet_sha256":{
            "C2":sha_file(packet_paths("C2")[0]),
            "C3":sha_file(packet_paths("C3")[0])
        },
        "rating_file_sha256":{
            "C2":sha_file(c2_path),"C3":sha_file(c3_path)
        },
        "means":{"C2":m2,"C3":m3},
        "g02":{
            "original_c1_parent_mean":g02_c1,
            "c2_mean":g02_c2,
            "c3_local_parent_mean":g02_c3_parent,
            "c3_candidate_mean":g02_c3,
            "c3_minus_original_c1":g02_c3-g02_c1,
            "c3_minus_local_c2_parent":g02_c3-g02_c3_parent,
            "local_nonregression_pass":local_g02_ok,
            "cumulative_nonregression_pass":cumulative_g02_ok
        },
        "g09":{
            "original_c1_parent_mean":g09_c1,
            "c2_candidate_mean":g09_c2,
            "c2_minus_original_c1":g09_c2-g09_c1,
            "nonregression_pass":g09_ok
        },
        "maximum_allowed_regression":0.5,
        "candidate_set_relevance_gate":expansion_relevance_gate,
        "population_expansion_admitted_by_this_review":False,
        "note":"Relevance is one prerequisite only; proposition/leakage gates remain independently authoritative."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--make-template",choices=["C2","C3"])
    ap.add_argument("--out",type=Path)
    ap.add_argument("--c2-ratings",type=Path)
    ap.add_argument("--c3-ratings",type=Path)
    args=ap.parse_args()
    if args.make_template:
        if args.out is None: raise SystemExit("--out required")
        write(args.out,template(args.make_template))
        print(json.dumps({"packet":args.make_template,
                          "template_sha256":sha_file(args.out),
                          "scores_filled":0},sort_keys=True))
    elif args.c2_ratings and args.c3_ratings:
        result=aggregate(args.c2_ratings,args.c3_ratings)
        if args.out: write(args.out,result)
        print(json.dumps(result,indent=2,sort_keys=True))
    else:
        p2,_=reconstruct("C2");p3,_=reconstruct("C3")
        print(json.dumps({
            "C2_items":len(p2["items"]),"C3_items":len(p3["items"]),
            "C2_sha256":sha_file(packet_paths("C2")[0]),
            "C3_sha256":sha_file(packet_paths("C3")[0]),
            "ratings_present":False,
            "population_expansion_admitted":False
        },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
