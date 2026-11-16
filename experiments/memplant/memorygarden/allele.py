"""Counterfactual Allele-001: frozen Qwen source-state vector arithmetic.

Learn global heat/vibration *additive* state deltas from the original DG002
training families, apply to different, already-encoded worlds. No new text
prefill, no model gradient, no native KV surgery; readout is a train-only ridge.
The DG002 sealed worlds were previously inspected: results are EXPLORATORY.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.memplant.memorygarden.subjects import normalize
from experiments.memplant.dreamgraft002.source import save_json

LAMBDA=1.0
RNG_SEED=23017


def load_all(directory):
    d=Path(directory)
    manifest=json.loads((d/"manifest.json").read_text())
    payload=(d/"features.npz").read_bytes()
    if hashlib.sha256(payload).hexdigest()!=manifest["feature_container_hash"]:
        raise RuntimeError("original donor feature archive hash mismatch")
    with np.load(d/"features.npz",allow_pickle=False) as f:
        features={k:np.asarray(f[k],dtype=np.float64)
                  for k in ("v","hidden","embedding")}
    for key,x in features.items():
        if x.shape!=(128,2048):
            raise RuntimeError("source feature shape changed")
        if hashlib.sha256(x.astype("float32").tobytes()).hexdigest()!=manifest["channels"][key]["sha256"]:
            raise RuntimeError("source tensor checksum mismatch")
    return manifest,features


def train_readout(features,rows):
    ix=np.array([i for i,c in enumerate(rows) if c["split"]=="train"])
    y=np.array([[1 if c["heat"]=="hardens" else -1,
                 1 if c["vibration"]=="floats" else -1] for c in rows],float)
    x=normalize(features)
    bank=x[ix]
    alpha=np.linalg.solve(bank@bank.T/2048+LAMBDA*np.eye(len(ix)),y[ix])
    def predict(a):
        return (normalize(a)@bank.T/2048)@alpha
    return predict,y


def evaluate(predict,features,rows,split,delta_h,delta_v,rng):
    ids=np.array([i for i,c in enumerate(rows) if c["split"]==split])
    by_id={r["case_id"]:i for i,r in enumerate(rows)}
    original=features[ids];before=predict(original)
    conditions={}
    for axis,delta in (("heat",delta_h),("vibration",delta_v)):
        change_axis=0 if axis=="heat" else 1
        keep_axis=1-change_axis
        source_direction=np.array([
            1 if (rows[i]["heat"]=="hardens" if axis=="heat"
                  else rows[i]["vibration"]=="floats") else -1
            for i in ids])
        targets=np.array([by_id[rows[i]["heat_twin" if axis=="heat" else "vibe_twin"]]
                          for i in ids])
        expected=np.array([[1 if rows[j]["heat"]=="hardens" else -1,
                            1 if rows[j]["vibration"]=="floats" else -1] for j in targets],float)
        original_y=np.array([[1 if rows[i]["heat"]=="hardens" else -1,
                              1 if rows[i]["vibration"]=="floats" else -1] for i in ids],float)
        applied=original+source_direction[:,None]*delta
        changed=predict(applied)
        random_delta=rng.normal(size=delta.shape)
        random_delta*=np.linalg.norm(delta)/(np.linalg.norm(random_delta)+1e-15)
        wrong_delta=delta_v if axis=="heat" else delta_h
        wrong_delta*=np.linalg.norm(delta)/(np.linalg.norm(wrong_delta)+1e-15)
        def metrics(scores):
            signs=np.where(scores>=0,1,-1)
            oldsigns=np.where(before>=0,1,-1)
            return dict(
                n=len(ids),
                changed_target_correct=int((signs[:,change_axis]==expected[:,change_axis]).sum()),
                untouched_correct=int((signs[:,keep_axis]==original_y[:,keep_axis]).sum()),
                both_correct=int(((signs==expected).all(axis=1)).sum()),
                changed_prediction_count=int((signs[:,change_axis]!=oldsigns[:,change_axis]).sum()),
                preserved_other_prediction_count=int((signs[:,keep_axis]==oldsigns[:,keep_axis]).sum()))
        conditions[axis]={
            "delta_l2":float(np.linalg.norm(delta)),
            "additive":metrics(changed),
            "reencoded_target":metrics(predict(features[targets])),
            "random_same_norm":metrics(predict(original+source_direction[:,None]*random_delta)),
            "wrong_axis_same_norm":metrics(predict(original+source_direction[:,None]*wrong_delta)),
            "mean_cosine_to_actual_reencoding":float(
                (normalize(applied)*normalize(features[targets])).sum(axis=1).mean()/2048),
            "source_families":len({rows[i]["family"] for i in ids}),
        }
    return conditions


def run(source,out):
    meta,features=load_all(source)
    rows=meta["cases"]
    byid={r["case_id"]:i for i,r in enumerate(rows)}
    train=[i for i,r in enumerate(rows) if r["split"]=="train"]
    if len(train)!=80:
        raise RuntimeError("not the fixed DG002 training dataset")
    response={
        "schema":"memplant-allele-001-v1",
        "status":"COMPLETED_SOURCE_SIDE_EXPLORATORY",
        "real_donor_sha256":meta["channels"]["v"]["sha256"],
        "source_model_revision":meta["donor"]["revision"],
        "dataset_families":{"train":20,"validation":6,"previously_inspected_test":6},
        "statistical_limit":"several factorial samples per same world; test previously explored",
        "no_receiver_model_executed":True,
        "native_KV_cache_edited":False,
        "readout_supervised_on_training_only":True,
        "methods":{},
    }
    rng=np.random.default_rng(RNG_SEED)
    for method,x in features.items():
        pred,y=train_readout(x,rows)
        heat=[];vibration=[]
        for i in train:
            item=rows[i]
            if item["heat"]=="hardens":
                heat.append(x[byid[item["heat_twin"]]]-x[i])
            if item["vibration"]=="floats":
                vibration.append(x[byid[item["vibe_twin"]]]-x[i])
        if len(heat)!=40 or len(vibration)!=40:
            raise RuntimeError("counterfactual training pairs incomplete")
        dh=np.mean(heat,axis=0)
        dv=np.mean(vibration,axis=0)
        response["methods"][method]={
            "delta_heat_sha256":hashlib.sha256(dh.astype("float32").tobytes()).hexdigest(),
            "delta_vibration_sha256":hashlib.sha256(dv.astype("float32").tobytes()).hexdigest(),
            "validation":evaluate(pred,x,rows,"validation",dh,dv,rng),
            "previously_inspected_test":evaluate(pred,x,rows,"sealed",dh,dv,rng),
        }
    save_json(out,response)
    for name,p in response["methods"].items():
        c=p["previously_inspected_test"]
        print(name, "heat:",c["heat"]["additive"], "vibration:",c["vibration"]["additive"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="runtime/memplant/garden/source")
    ap.add_argument("--out",default="runtime/memplant/garden/allele.json")
    a=ap.parse_args()
    run(a.source,a.out)
