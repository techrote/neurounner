#!/usr/bin/env python3
"""R10 original native material-mask proximity to visible TRAIN/DEVELOPMENT only.

No FINAL read, training, model inference, new native application or admission.
"""
import argparse
import json
import struct
import zipfile
from pathlib import Path
from io import BytesIO
from PIL import Image
from r10_split_native_review import evaluate, digest, must, SEEDS, CORPUS_ZIP, RICH_ZIP, PROGRAM

GROUPS = {
    "train": {"pipework/two_elbow_route/v1", "pipework/tee_distribution/v1"},
    "development": {"pipework/three_branch_manifold/v1", "pipework/closed_loop_with_tap/v1"},
}
MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
SIDE = 96

def cells(data):
    must(len(data) == SIDE*SIDE*4, "invalid readback length")
    values = struct.unpack("<9216i", data)
    must(set(values) == {0,58}, "native material domain violation")
    return {i for i,v in enumerate(values) if v==58}, values

def vicinity(points, radius=3):
    must(isinstance(radius,int) and 0<=radius<=8, "invalid vicinity radius")
    result=set()
    for p in points:
        x,y=p%SIDE,p//SIDE
        for yy in range(max(0,y-radius),min(SIDE,y+radius+1)):
            for xx in range(max(0,x-radius),min(SIDE,x+radius+1)):
                result.add(yy*SIDE+xx)
    return result

def metrics(a,b,radius=3):
    must(bool(a) and bool(b), "empty native material mask")
    must(all(type(p)==int and 0<=p<SIDE*SIDE for p in a|b), "invalid native pixel")
    i=len(a&b)
    ca=len(a&vicinity(b,radius))
    cb=len(b&vicinity(a,radius))
    return {
        "candidate_cells":len(a),"reference_cells":len(b),
        "intersection_cells":i,"union_cells":len(a|b),
        "iou":round(i/len(a|b),6),
        "candidate_within_reference_radius3_cells":ca,
        "candidate_within_reference_radius3_fraction":round(ca/len(a),6),
        "reference_within_candidate_radius3_cells":cb,
        "reference_within_candidate_radius3_fraction":round(cb/len(b),6)
    }

def audit(corpus,native):
    old=evaluate(corpus,native)
    must(old["status"]=="REVIEW_ONLY_VISIBLE_SPLIT_AND_NATIVE_BYTES_PASS",
         "original R10 custody not qualified")
    rows_out=[]
    with zipfile.ZipFile(corpus) as z, zipfile.ZipFile(native) as n:
        man=z.read("corpus/admitted-corpus.json")
        must(digest(man)==MANIFEST, "original corpus manifest drift")
        m=json.loads(man)
        must(m["source_sha"]==NATIVE_SOURCE, "original corpus source drift")
        mask={}
        for seed in SEEDS:
            mask[seed],_=cells(n.read(f"sprocket-rich-v2-native/sprocket-R10-s{seed}.materials.i32le"))
            must(len(mask[seed])==1724,"original R10 population drift")
        for split,expected in GROUPS.items():
            blob=z.read("corpus/"+split+".jsonl")
            must(digest(blob)==m["split_files"][split.upper()]["sha256"],
                 "visible split source hash mismatch")
            rows=[json.loads(line) for line in blob.splitlines() if line.strip()]
            must(len(rows)==12, "visible split count mismatch")
            chosen=[x for x in rows if x["family"]=="pipework"]
            must(len(chosen)==4 and {x["structural_group"] for x in chosen}==expected,
                 "visible structural groups changed")
            must({(x["structural_group"],x["seed"]) for x in chosen}
                 == {(g,s) for g in expected for s in SEEDS}, "seed coverage mismatch")
            for row in sorted(chosen,key=lambda r:(r["structural_group"],r["seed"])):
                name=row["native_case_id"]
                must(name==row["id"] and name.startswith("pipework-"),"invalid native case")
                admission=row["native_admission"]
                must(row["native_qualified"] is True and admission["native_execution"] is True
                     and admission["verified"] is True and admission["state"]=="PASS"
                     and admission["source_sha"]==NATIVE_SOURCE,
                     "original native admission mismatch")
                raw=z.read(name+".materials.i32le")
                png=z.read(name+".png")
                rec=z.read(name+".receipt.json")
                mat=z.read(name+".materials.png")
                must(digest(raw)==admission["readback_sha256"]
                     and digest(png)==admission["png_sha256"]
                     and digest(rec)==admission["receipt_sha256"],
                     "original native receipt/readback hash mismatch")
                r=json.loads(rec)
                b,values=cells(raw)
                must(r["passed"] is True and r["id"]==name and r["terminal"]["status"]=="completed"
                     and r["crop_i32le_sha256"]==digest(raw) and len(b)==r["filled_cells"],
                     "original native receipt not positive")
                img=Image.open(BytesIO(mat))
                must(img.mode=="L" and img.size==(SIDE,SIDE) and img.tobytes()==bytes(values),
                     "original material PNG/readback mismatch")
                display=Image.open(BytesIO(png))
                must(display.mode=="RGB" and display.size==(SIDE,SIDE)
                     and all((display.getpixel((i%SIDE,i//SIDE))==(232,226,205))==(v==58)
                             for i,v in enumerate(values)),
                     "original display PNG pixels not material readback")
                rows_out.append({"split":split.upper(),"group":row["structural_group"],
                                 "seed":row["seed"],"reference_readback_sha256":digest(raw),
                                 "reference_png_sha256":digest(png),
                                 **metrics(mask[row["seed"]],b)})
    must(len(rows_out)==8,"incorrect comparison denominator")
    return {"schema":"cybersand.sprocket-r10-visible-native-proximity/1",
            "status":"REVIEW_ONLY_ORIGINAL_NATIVE_SPATIAL_PROXIMITY",
            "program_sha256":PROGRAM,"original_corpus_zip_sha256":CORPUS_ZIP,
            "original_native_zip_sha256":RICH_ZIP,
            "original_corpus_source_sha":NATIVE_SOURCE,
            "original_corpus_manifest_sha256":MANIFEST,
            "image_size":[SIDE,SIDE],"proximity":"Chebyshev <=3 pixels, both directions",
            "distinct_reference_groups":4,"reference_native_seed_views":8,
            "candidate_independent_compositions":1,"results":rows_out,
            "new_shared_train_groups":0,"new_native_executions":0,
            "model_outputs":0,"optimizer_updates":0,"final_targets_opened":False,
            "disposition":"REVIEW_ONLY_GLOBAL_SPLIT_DECISION_RESERVED_TO_SHEPHERD_REVIEWER",
            "limitation":"No threshold-based admission; morphology near-alias may exist without graph isomorphism."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus-zip",type=Path,required=True)
    p.add_argument("--rich-zip",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    must(not a.output.exists(),"refuse overwrite")
    report=audit(a.corpus_zip,a.rich_zip)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(report["status"],len(report["results"]),"comparisons")

if __name__=="__main__":
    main()
