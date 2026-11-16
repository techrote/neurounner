"""Live neurounner Sprocket/LFM2.5 inventors in a deterministic memory garden.

Frozen Qwen donor attention state -> train-only supervised two-bit ridge ->
TEXT POSTCARD -> current pinned GGUF Sprocket or LFM2.5 consumer.

This is explicitly a textual consumer baseline, NOT a native KV or soft-prefix
transplant into Sprocket/LFM. Retain all original responses and parsing errors.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np

from experiments.memplant.memorygarden.garden import (
    World, Invention, evaluate, make_prompt)


def normalize(x):
    return (x-x.mean(axis=-1, keepdims=True))/(x.std(axis=-1, keepdims=True)+1e-6)


def load_qwen_state(source):
    source=Path(source)
    if source.is_dir():
        meta=json.loads((source/"manifest.json").read_text())
        blob=(source/"features.npz").read_bytes()
    else:
        with zipfile.ZipFile(source) as z:
            meta=json.loads(z.read("manifest.json"))
            blob=z.read("features.npz")
    if hashlib.sha256(blob).hexdigest() != meta["feature_container_hash"]:
        raise RuntimeError("source archive checksum mismatch")
    with np.load(io.BytesIO(blob), allow_pickle=False) as a:
        data=np.array(a["v"], dtype="float64")
    if data.shape != (128,2048):
        raise ValueError("unexpected actual Qwen attention feature geometry")
    if hashlib.sha256(data.astype("float32").tobytes()).hexdigest() != meta["channels"]["v"]["sha256"]:
        raise RuntimeError("donor V SHA-256 mismatch")
    return data,meta


def train_only_readout(data, records):
    ids=np.array([i for i,r in enumerate(records) if r["split"]=="train"])
    labels=np.array([[1 if r["heat"]=="hardens" else -1,
                      1 if r["vibration"]=="floats" else -1]
                     for r in records], dtype=float)
    source=normalize(data)
    bank=source[ids]
    alpha=np.linalg.solve(bank@bank.T/2048+np.eye(len(ids)),labels[ids])
    return (source@bank.T/2048)@alpha


def select_worlds(records,scores):
    sealed=[(i,r) for i,r in enumerate(records) if r["split"]=="sealed"]
    result=[]; used=set()
    for clazz in range(4):
        viable=[(i,r) for i,r in sealed
                if r["answer_class"]==clazz and r["family"] not in used]
        if not viable:
            raise RuntimeError("missing sealed independent-world class "+str(clazz))
        idx,row=viable[0]; used.add(row["family"])
        true=World(row["case_id"],row["heat"]=="hardens",row["vibration"]=="floats")
        inferred=World(row["case_id"]+"-derived",
                       bool(scores[idx,0]>=0),bool(scores[idx,1]>=0))
        result.append((true,inferred,row))
    return result


def run(subject, source, output, *, max_tokens=170):
    from scripts.experiment import Server,load_kv,runner_info
    if subject not in ("sprocket","lfm"):
        raise ValueError("subject not in explicitly pinned list")
    root=Path(__file__).resolve().parents[3]
    paths=load_kv(root/".cache/neurounner/paths.env")
    server_bin=Path(paths["LLAMA_SERVER"])
    model_file=Path(paths["SPROCKET_MODEL" if subject=="sprocket" else "LFM_MODEL"])
    if not server_bin.is_file() or not model_file.is_file():
        raise FileNotFoundError("Run scripts/bootstrap.sh "+subject+" first")
    features,manifest=load_qwen_state(source)
    scores=train_only_readout(features,manifest["cases"])
    worlds=select_worlds(manifest["cases"],scores)
    server=Server("memplant-"+subject,server_bin,model_file,
                  18945 if subject=="sprocket" else 18946,
                  threads=4,context=2048)
    records={
        "schema":"memplant-garden-subject-v1",
        "subject":subject,"model_weight":model_file.name,
        "donor":manifest["donor"]["id"],
        "donor_revision":manifest["donor"]["revision"],
        "donor_v_feature_sha256":manifest["channels"]["v"]["sha256"],
        "channel":"genuine Qwen state -> supervised 2-bit ridge -> TEXT postcard for pinned GGUF consumer",
        "native_kv_transplant":False,
        "numeric_pseudo_prefix_given_to_subject":False,
        "runner":runner_info(),"records":[],"error":None,
    }
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    try:
        server.start()
        for truth,estimate,source_case in worlds:
            for mode in ("none","state_card","counterfactual_card","full_text"):
                if mode=="none":
                    prompt=make_prompt(truth,"none")
                elif mode=="state_card":
                    prompt=make_prompt(estimate,"text_card")
                elif mode=="counterfactual_card":
                    prompt=make_prompt(estimate,"wrong_card")
                else:
                    prompt=make_prompt(truth,"full_text",source_case["source"])
                entry={
                    "world":truth.name,"mode":mode,
                    "actual_heat":truth.heat,"actual_vibration":truth.vibration,
                    "state_readout_heat":estimate.heat,
                    "state_readout_vibration":estimate.vibration,
                    "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                }
                try:
                    answer=server.chat(prompt,max_tokens=max_tokens,
                                       temperature=0.25,seed=3103)
                    entry["response"]=answer
                    try:
                        invention=Invention.from_text(answer["content"])
                        entry["invention"]=invention.__dict__
                        entry["grading"]=evaluate(truth,invention)
                        entry["valid_json"]=True
                    except (ValueError,json.JSONDecodeError) as e:
                        entry["valid_json"]=False
                        entry["parse_error"]=str(e)[:300]
                        entry["grading"]={"passed":False,
                                          "component_accuracy":0.0}
                except Exception as e:
                    entry["valid_json"]=False
                    entry["request_error"]=type(e).__name__+": "+str(e)[:300]
                    entry["grading"]={"passed":False}
                records["records"].append(entry)
                dest.write_text(json.dumps(records,indent=2)+"\n")
                print(subject,truth.name,mode,entry["grading"],flush=True)
        records["summary"]={}
        for mode in ("none","state_card","counterfactual_card","full_text"):
            rows=[r for r in records["records"] if r["mode"]==mode]
            records["summary"][mode]={
                "n":len(rows),
                "valid_json":sum(bool(x.get("valid_json")) for x in rows),
                "functional":sum(bool(x["grading"]["passed"]) for x in rows),
            }
        records["elapsed_seconds"]=round(time.monotonic()-started,3)
    except Exception as e:
        records["error"]=type(e).__name__+": "+str(e)
        raise
    finally:
        server.stop()
        dest.write_text(json.dumps(records,indent=2)+"\n")
    return records


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--subject",choices=("sprocket","lfm"),required=True)
    p.add_argument("--source",default="runtime/memplant/garden/source")
    p.add_argument("--out",default="runtime/memplant/garden/subject.json")
    p.add_argument("--max-tokens",type=int,default=170)
    a=p.parse_args()
    run(a.subject,a.source,a.out,max_tokens=a.max_tokens)
