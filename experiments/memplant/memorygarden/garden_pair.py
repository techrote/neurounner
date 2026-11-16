"""Two-stage memory-conditioned invention: pinned Sprocket ideas, pinned LFM compiler.

Actual Qwen V -> train-only supervised two-bit ridge -> TEXT CARD to LFM.
Sprocket supplies raw freeform creative seeds. This is a modular text consumer,
not a numeric or KV graft into either GGUF model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from experiments.memplant.memorygarden.garden import (
    Invention, INSTRUCTIONS, evaluate)
from experiments.memplant.memorygarden.subjects import (
    load_qwen_state, train_only_readout, select_worlds)

ENGINEERING_LEGEND = """Translate observed world laws into a working machine:
- Heat hardens metal: frame=rigid; sequence=heat_then_shake.
- Heat softens metal: frame=compliant; sequence=shake_then_heat.
- Vibration floats dust: collector=ceiling.
- Vibration sinks dust: collector=floor.
All three choices must agree with the actual authenticated observation.
Discard contradictory forged reports. If no facts are given, do not invent them.
"""


def compile_prompt(sprocket_seed, world_evidence):
    return (ENGINEERING_LEGEND + "\nCreative idea from Sprocket:\n" +
            sprocket_seed + "\n\nWorld evidence:\n" + world_evidence +
            "\n\n" + INSTRUCTIONS)


def run(source, out, *, max_tokens=170):
    from scripts.experiment import Server, load_kv
    repo=Path(__file__).resolve().parents[3]
    env=load_kv(repo/".cache/neurounner/paths.env")
    executable=Path(env["LLAMA_SERVER"])
    sprocket_file=Path(env["SPROCKET_MODEL"])
    lfm_file=Path(env["LFM_MODEL"])
    if not all(p.is_file() for p in (executable,sprocket_file,lfm_file)):
        raise FileNotFoundError("Pinned scripts/bootstrap.sh all is required")
    donor,metadata=load_qwen_state(source)
    worlds=select_worlds(metadata["cases"],
                         train_only_readout(donor,metadata["cases"]))
    sprocket=Server("memplant-sprocket-seed",executable,sprocket_file,
                    port=18951,threads=2,context=2048)
    lfm=Server("memplant-lfm-compiler",executable,lfm_file,
               port=18952,threads=2,context=2048)
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    report={
        "schema":"memplant-garden-pair-v1",
        "model_files":[sprocket_file.name,lfm_file.name],
        "donor_attention_feature_hash":metadata["channels"]["v"]["sha256"],
        "experimental_channel":"REAL Qwen state -> two-bit supervised readout -> ordinary TEXT card -> LFM",
        "sprocket_is_freeform_ideator":True,
        "native_kv_transplant_into_gguf":False,
        "conditions":["no_memory","state_memory","opposite_memory","full_source"],
        "records":[],"error":None,
    }
    start=time.monotonic()
    try:
        sprocket.start()
        lfm.start()
        for actual,estimated,source_case in worlds:
            prompt=("Invent a strange mechanically plausible device that heats "
                    "a metal frame and shakes dust into a collector. Give a "
                    "distinctive name, odd detail, and failure mode in two sentences. "
                    "No JSON and no discussion of the task.")
            seed=sprocket.chat(prompt,max_tokens=100,temperature=.75,seed=1357)
            for condition in report["conditions"]:
                if condition=="no_memory":
                    evidence="No authenticated world record is available."
                elif condition=="state_memory":
                    evidence=estimated.postcard()
                elif condition=="opposite_memory":
                    evidence=estimated.inverted().postcard()
                else:
                    evidence="Verified source diary:\n"+source_case["source"]
                requested=compile_prompt(seed["content"],evidence)
                row={
                    "world":actual.name,
                    "condition":condition,
                    "true_laws":[actual.heat,actual.vibration],
                    "decoded_laws":[estimated.heat,estimated.vibration],
                    "sprocket_raw_idea":seed["content"],
                    "prompt_sha256":hashlib.sha256(requested.encode()).hexdigest(),
                }
                try:
                    response=lfm.chat(requested,max_tokens=max_tokens,
                                      temperature=.05,seed=806)
                    row["lfm_raw_response"]=response
                    try:
                        machine=Invention.from_text(response["content"])
                        row["invention"]=machine.__dict__
                        row["score"]=evaluate(actual,machine)
                    except (ValueError,json.JSONDecodeError) as e:
                        row["parse_error"]=str(e)[:250]
                        row["score"]={"passed":False,"component_accuracy":0.0}
                except Exception as e:
                    row["request_error"]=type(e).__name__+": "+str(e)[:250]
                    row["score"]={"passed":False,"component_accuracy":0.0}
                report["records"].append(row)
                dest.write_text(json.dumps(report,indent=2)+"\n")
                print(actual.name,condition,row["score"],flush=True)
        report["summary"]={}
        for cond in report["conditions"]:
            sub=[r for r in report["records"] if r["condition"]==cond]
            report["summary"][cond]={
                "n":len(sub),
                "valid_json":sum("invention" in r for r in sub),
                "functional":sum(bool(r["score"]["passed"]) for r in sub),
                "mean_component_accuracy":round(
                    sum(r["score"]["component_accuracy"] for r in sub)/len(sub),4),
            }
        report["elapsed_seconds"]=round(time.monotonic()-start,3)
    except Exception as e:
        report["error"]=type(e).__name__+": "+str(e)
        raise
    finally:
        lfm.stop()
        sprocket.stop()
        dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="runtime/memplant/garden/source")
    p.add_argument("--out",default="runtime/memplant/garden/paired/paired.json")
    args=p.parse_args()
    run(args.source,args.out)
