#!/usr/bin/env python3
"""Exact-blob, source-only SPROCKET G09 vs SMOL G06 geometry witness.

No peer code execution, FINAL read, model operation, native claim or TRAIN admission.
"""
import argparse
import ast
from hashlib import sha1, sha256
import json
from pathlib import Path

G09_BLOB = "b3ac98a8cd06158b39a2f58776fab845046f245f"
G06_BLOB = "0b0adae3c02d1ccc9bda2cdca393f272ca7ecf63"

def blob(data):
    return sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def programs(g09_source, g06_source):
    a=g09_source.read_bytes()
    b=g06_source.read_bytes()
    if blob(a)!=G09_BLOB or blob(b)!=G06_BLOB:
        raise ValueError("pinned peer or lane Git blob differs")
    tree=ast.parse(a.decode())
    studies=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=="STUDIES" for t in n.targets)]
    if len(studies)!=1 or set(studies[0])!={"G08","G09","A07","A08"}:
        raise ValueError("G09 source definition drift")
    peer=json.loads(b)
    g06=[x["program"] for x in peer["proposals"] if x.get("id")=="smol_g06_stepped_absence"]
    if len(g06)!=1: raise ValueError("SMOL G06 source missing/ambiguous")
    return studies[0]["G09"]["program"],g06[0],{"sprocket":sha256(a).hexdigest(),"smol":sha256(b).hexdigest()}

def boxes(program):
    if program.get("version")!=2 or program.get("family")!="greebles" or len(program.get("components",[]))!=3:
        raise ValueError("only three-panel v2 Greebles may be compared")
    result=[]
    for panel in program["components"]:
        b=panel.get("box")
        if panel.get("kind")!="panel" or not isinstance(b,list) or len(b)!=4 or any(type(v)!=int for v in b):
            raise ValueError("invalid panel component/coordinate")
        x,y,w,h=b
        if x<0 or y<0 or w<=0 or h<=0 or x+w>96 or y+h>96:
            raise ValueError("out-of-bounds panel")
        result.append((x,y,w,h))
    return sorted(result,key=lambda b:(b[1],b[0]))

def raster(boxlist):
    return {(x,y) for x0,y0,w,h in boxlist for x in range(x0,x0+w) for y in range(y0,y0+h)}

def compare(g09,smol):
    a,b=boxes(g09),boxes(smol)
    widths=[x[2] for x in a]
    lefts=[x[0] for x in a]
    gaps=[a[i+1][1]-(a[i][1]+a[i][3]) for i in range(2)]
    centers=[2*x+w for x,y,w,h in b]
    if not (all(w>2*h and w>=40 for x,y,w,h in a)
            and widths[0]>widths[1]>widths[2] and lefts[0]<lefts[1]<lefts[2]
            and all(g>=8 for g in gaps)):
        raise ValueError("G09 broad descending three-band structure violated")
    if not (all(w<30 for x,y,w,h in b) and centers[0]>centers[1]>centers[2]
            and centers[0]-centers[2]>70):
        raise ValueError("SMOL G06 compact ascending cassette structure violated")
    am,bm=raster(a),raster(b)
    return {
      "schema":"cybersand.sprocket-g09-smol-g06-source-contrast/1",
      "disposition":"PAIRWISE_SOURCE_DISTINCT_GLOBAL_REVIEW_HOLD",
      "sprocket_boxes":a,"smol_boxes":b,"sprocket_widths":widths,
      "smol_widths":[x[2] for x in b],"sprocket_gaps":gaps,
      "sprocket_box_cells":len(am),"smol_box_cells":len(bm),
      "shared_box_cells":len(am&bm),
      "box_iou":round(len(am&bm)/len(am|bm),8),
      "direction":"G09 top-to-bottom right; G06 top-to-bottom left",
      "scope":"Visible pinned source panel boxes only, NOT native painted pixels",
      "native_research_run_ids":[43587600137,43601281142],
      "original_native_producers":["63ee8b2b561068a72e0652e54b6e4193cb47d5e6","6e412178c9b71cb0196d8e35408470077b50f884"],
      "sprocket_blob_sha1":G09_BLOB,"smol_blob_sha1":G06_BLOB,
      "globally_distinct":False,"new_train_groups":0,"new_native_views":0,
      "new_model_outputs":0,"optimizer_updates":0,"final_opened":False,
      "next_gate":"Reviewer inspect original native PNGs and visible held-outs; shepherd decide global split and shared admission"
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sprocket",type=Path,required=True)
    p.add_argument("--smol",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError("refuse evidence overwrite")
    g09,g06,sha=programs(args.sprocket,args.smol)
    result=compare(g09,g06)
    result["source_sha256"]=sha
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"disposition":result["disposition"],"box_iou":result["box_iou"],"train":0}))

if __name__=="__main__":
    main()
