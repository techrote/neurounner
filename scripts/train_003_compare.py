#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

METRICS=("json_rate","exact_schema_rate","core_schema_rate","decision_accuracy","mechanical_score","overall_score")
def load(p): return json.loads(Path(p).read_text())
def vals(e): return {k:float(e["evaluation"][k]) for k in METRICS}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--baseline",required=True); ap.add_argument("--candidate",required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    b=load(a.baseline); c=load(a.candidate)
    for name,x in (("baseline",b),("candidate",c)):
        if x.get("status")!="success" or x.get("experiment")!="TRAIN-003": raise SystemExit(f"{name} evidence invalid")
    comparable=(b["base_model"]==c["base_model"] and b["base_revision"]==c["base_revision"] and b["evaluator_version"]==c["evaluator_version"] and b["dataset"]["manifest_sha256"]==c["dataset"]["manifest_sha256"] and b["dataset"]["shard_sha256"]==c["dataset"]["shard_sha256"])
    if not comparable: raise SystemExit("baseline/candidate evaluation contracts differ")
    bm=vals(b); cm=vals(c); delta={k:cm[k]-bm[k] for k in METRICS}
    by={}
    for label in c["evaluation"]["by_decision"]:
        x=b["evaluation"]["by_decision"][label]; y=c["evaluation"]["by_decision"][label]
        by[label]={"baseline":x,"candidate":y,"delta":{k:float(y[k])-float(x[k]) for k in METRICS}}
    out={"schema_version":1,"experiment":"TRAIN-003","status":"success","comparison_contract_identical":True,"base_model":c["base_model"],"base_revision":c["base_revision"],"dataset":c["dataset"],"baseline":{"identity":b["identity"],"source_commit":b.get("source_commit"),"checkpoint_metadata":b.get("checkpoint_metadata"),"metrics":bm},"candidate":{"identity":c["identity"],"source_commit":c.get("source_commit"),"training":c.get("training"),"adapter":c.get("adapter"),"metrics":cm},"delta_candidate_minus_train001":delta,"by_decision":by,"semantic_improvement_observed":delta["mechanical_score"]>0 and delta["overall_score"]>0,"notes":["Mechanical score is independent of the new decision field and exact six-key schema, so TRAIN-001 receives credit for mechanically correct five-field output.","Held-out mechanism families are absent from TRAIN-003 training.","Raw held-out outputs remain in the baseline and candidate evidence files."]}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": main()
