#!/usr/bin/env python3
"""R10 source-bound visible-split + original native-byte review. Not TRAIN.

Only corpus/train.jsonl and corpus/development.jsonl are opened. FINAL is sealed.
Two host seeds are two physical views of ONE authored design, not two structures.
"""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import zipfile
from PIL import Image

CORPUS_ZIP="50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39"
RICH_ZIP="f7e673c414487668598eb26dc618daa019b2fc83bc70ed373cd574049f20e53c"
PROGRAM="0f52d4f6b252d17f9f335e8c8a65a44f78626e79cd26298881a78446f3f7b745"
SOURCE="ab101910921a7cf7f30b3dc83d150d214aaa060d"
RICH_BLOB="37d9b7d3f698d5e851e213043d8d43c5f2c9750a"
SEEDS=(104729,130363)

def must(value, reason):
    if not value: raise ValueError(reason)

def digest(data): return sha256(data).hexdigest()

def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))

def topology(program):
    must(program["family"]=="pipework" and len(program["components"])==1,"not one pipe network")
    c=program["components"][0]; n=len(c["nodes"]); e=c["edges"]
    degrees=Counter(i for a,b in e for i in (a,b))
    must(len(degrees)==n, "isolated pipe node")
    return {"nodes":n,"edges":len(e),"cycles":len(e)-n+1,
            "endpoints":sum(v==1 for v in degrees.values()),
            "degree_histogram":dict(sorted(Counter(degrees.values()).items())),
            "diameter":c["diameter"]}

def evaluate(corpus_path, native_path):
    from peer_topology import PEER_GRAPH_EDGES,isomorphic
    from rich_pipework_v2 import program_for
    import hashlib
    src=Path(__file__).with_name("rich_pipework_v2.py").read_bytes()
    must(hashlib.sha1(b"blob "+str(len(src)).encode()+b"\0"+src).hexdigest()==RICH_BLOB,
         "original rich study source Git blob drift")
    must(digest(corpus_path.read_bytes())==CORPUS_ZIP, "original corpus archive hash drift")
    must(digest(native_path.read_bytes())==RICH_ZIP, "original R09/R10 native archive hash drift")
    program=program_for("R10")
    must(digest(canonical(program).encode())==PROGRAM, "original R10 teacher source changed")
    f=topology(program)
    must(f=={"nodes":10,"edges":12,"cycles":3,"endpoints":2,
             "degree_histogram":{1:2,2:2,3:6},"diameter":5}, "R10 topology changed")
    matches=[]
    for name,(n,e) in PEER_GRAPH_EDGES.items():
        if isomorphic(10,program["components"][0]["edges"],n,e): matches.append(name)
    must(not matches,"R10 is isomorphic to pinned VOID peer")
    visible=[]
    with zipfile.ZipFile(corpus_path) as z:
        for split in ("train","development"):
            rows=[json.loads(line) for line in z.read("corpus/"+split+".jsonl").splitlines() if line.strip()]
            must(len(rows)==12 and len({x["structural_group"] for x in rows})==6,
                 "original visible partition group or row count drift")
            seen=set()
            for row in rows:
                key=row["structural_group"]
                if key in seen: continue
                seen.add(key)
                p=row["program"]
                if p["family"]!="pipework": continue
                old=p["components"][0]
                must(not isomorphic(10,program["components"][0]["edges"],
                                    len(old["nodes"]),old["edges"]),
                     "R10 isomorphic to visible original pipe group")
                visible.append({"split":split,"structural_group":key,**topology(p)})
    must(len(visible)==4,"expected two visible pipe structures per partition")
    cases=[]
    with zipfile.ZipFile(native_path) as z:
        must(z.testzip() is None,"original research ZIP CRC failure")
        folder="sprocket-rich-v2-native/"
        summary=json.loads(z.read(folder+"qualification-summary.json"))
        verified=json.loads(z.read(folder+"native-verification.json"))
        produced=json.loads(z.read(folder+"candidates.json"))
        must(summary["source_sha"]==verified["source_sha"]==SOURCE
             and summary["native_verified"] is True and verified["passed"] is True
             and summary["shared_train_groups_admitted"]==0
             and summary["model_training"] is False
             and summary["final_targets_opened"] is False,
             "original research source/status mismatch")
        examples={r["id"]:r for r in produced["examples"]}
        checks={r["id"]:r for r in verified["cases"]}
        for seed in SEEDS:
            name=f"sprocket-R10-s{seed}"
            must(name in examples and name in checks,"original seed case missing")
            a,b=examples[name],checks[name]
            must(a["seed"]==seed and a["split"]=="PROPOSAL_ONLY"
                 and a["program"]==program and b["seed"]==seed
                 and b["program_sha256"]==PROGRAM
                 and b["structure"]["passed"] is True,
                 "original R10 program/native verification mismatch")
            raw=z.read(folder+name+".materials.i32le")
            mat=z.read(folder+name+".materials.png")
            rgb=z.read(folder+name+".png")
            receipt=z.read(folder+name+".receipt.json")
            cells=struct.unpack("<9216i",raw)
            must(set(cells)=={0,58} and cells.count(58)==1724,
                 "original native material domain/count mismatch")
            aimg=Image.open(BytesIO(mat)); vimg=Image.open(BytesIO(rgb))
            must(aimg.mode=="L" and aimg.size==(96,96) and
                 vimg.mode=="RGB" and vimg.size==(96,96)
                 and aimg.tobytes()==bytes(cells),
                 "material PNG pixel disagreement")
            must(all((vimg.getpixel((i%96,i//96))==(232,226,205))==(v==58)
                     for i,v in enumerate(cells)),
                 "render PNG foreground disagrees with stored native cells")
            must(digest(raw)==b["readback_sha256"]
                 and digest(rgb)==b["image_sha256"]
                 and digest(receipt)==b["receipt_sha256"]
                 and b["acknowledged_cells"]==1724,
                 "native receipt/PNG/readback hashes disagree")
            cases.append({"seed":seed,"acknowledged_cells":1724,
                          "readback_sha256":digest(raw),"png_sha256":digest(rgb),
                          "receipt_sha256":digest(receipt)})
    return {"schema":"cybersand.sprocket-r10-visible-native-review/1",
            "status":"REVIEW_ONLY_VISIBLE_SPLIT_AND_NATIVE_BYTES_PASS",
            "original_corpus_zip_sha256":CORPUS_ZIP,
            "original_native_zip_sha256":RICH_ZIP,
            "original_native_producer_source":SOURCE,
            "program_sha256":PROGRAM,
            "topology":f,"visible_original_pipe_groups":visible,
            "pinned_void_isomorphism_matches":matches,
            "native_seed_cases":cases,"independent_structures":1,"seed_views":2,
            "proven_native_research_views":2,"model_outputs":0,"optimizer_steps":0,
            "new_shared_train_groups":0,"final_targets_opened":False,
            "limitation":"Not a full global peer/silhouette verdict, a newly executed Art run, a trained model painting, or shared TRAIN admission",
            "next":"Reviewer decide global whole-structure/split safety; shepherd authorize newly source-built shared TRAIN admission and common 60-request DEVELOPMENT freeze"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus-zip",type=Path,required=True)
    p.add_argument("--rich-zip",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    must(not a.output.exists(),"refuse evidence overwrite")
    result=evaluate(a.corpus_zip,a.rich_zip)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print({k:result[k] for k in ("status","independent_structures","seed_views","new_shared_train_groups")})

if __name__=="__main__":main()
