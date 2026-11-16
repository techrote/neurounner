#!/usr/bin/env python3
"""Wave-16 real-model Q4 residual training and separate held-out W0 evaluation.

Train frozen-base external 55-parameter adapter on 2718/architecture with real
System One probabilities. Evaluate distinct seed 3891/pipework, shifted local
obstacles, and matched Q8/Q4/raw/adapted controls on the same hosted CPU.
This is NOT LoRA nor d1 base-weight fine-tuning nor native CyberSand integration.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from experiments.driv3r.backend import PINS,sha256_file
from experiments.driv3r.frankenlayers import infer as paired
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
from experiments.driv3r.frankenlayers_lab import counterfactual as cf
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS,validate_flows
from experiments.driv3r.frankenlayers_lab.infer_q4_causal import pareto_axis
from experiments.driv3r.frankenlayers_lab.infer_quant import validate_quant_receipt
from experiments.driv3r.frankenlayers_lab.repair_adapter import train,predict,digest

MODE="aero4_no2"
TRAIN_SEED=2718
TRAIN_FAMILY="architecture"
EVAL_SEED=3891
EVAL_FAMILY="pipework"


def evaluation_obstacles(variant,size=cf.SIZE):
    if variant not in cf.ALL_VARIANTS or size!=cf.SIZE:raise ValueError("Invalid heldout geometry")
    c=size//2
    return {
      "east_open":set(),"east_blocked":{(10,c)},
      "north_blocked":{(8,c-2),(9,c-2)},
      "south_blocked":{(8,c+2),(9,c+2)},
      "far_open":set(),"far_blocked":{(16,c)}
    }[variant]


@contextmanager
def heldout():
    old=cf.SEED,cf.FAMILY,cf.variant_obstacles
    try:
        cf.SEED,cf.FAMILY,cf.variant_obstacles=EVAL_SEED,EVAL_FAMILY,evaluation_obstacles
        yield
    finally:
        cf.SEED,cf.FAMILY,cf.variant_obstacles=old


@contextmanager
def adapt_server(checkpoint):
    original=paired.Server
    class Adapted(original):
        def choose(self,obs,records):
            raw=super().choose(obs,records)
            rec=records[-1]
            choice,explanation=predict(obs,raw,rec["parsed"]["probabilities"],checkpoint)
            rec["base_model_choice"]=raw
            rec["adapter"]=explanation
            rec["choice"]=choice
            return choice
    paired.Server=Adapted
    try:yield
    finally:paired.Server=original


def source_records(root):
    result=[]
    for variant in cf.ALL_VARIANTS:
        path=root/variant/(MODE+"-"+TRAIN_FAMILY+"-choices.json")
        for call in json.loads(path.read_text()):
            if call.get("status")!="ok" or call["choice"] not in call["observation"]["legal"]:
                raise ValueError("Illegal or failed source model decision")
            result.append({"observation":call["observation"],"choice":call["choice"],
                           "probabilities":call["parsed"]["probabilities"]})
    if len(result)<48:raise ValueError("Insufficient actual System One samples")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--model-q8",type=Path,required=True)
    p.add_argument("--model-q4",type=Path,required=True)
    p.add_argument("--quant-receipt",type=Path,required=True)
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--mode",choices=(MODE,),required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--source-sha",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.seed!=EVAL_SEED or a.output.exists():
        raise ValueError("Held-out seed changed or evidence overwrite")
    if sha256_file(a.model_q8)!=PINS["model"]["sha256"]:
        raise ValueError("Q8 donor hash mismatch")
    receipt=json.loads(a.quant_receipt.read_text())
    validate_quant_receipt(receipt,a.model_q8,a.model_q4)
    validate_flows()
    if any(name in FLOWS for name in LAB_FLOWS):
        raise ValueError("Lab flow collides with parent mode")
    FLOWS.update(LAB_FLOWS)
    w0=paired.load_w0(a.w0)
    a.output.mkdir(parents=True)

    # Model-origin training fold, disjoint from all held-out evaluation observations.
    train_path=a.output/"train-fold"
    cf.run_all(a.binary,a.model_q4,w0,MODE,train_path,TRAIN_SEED)
    source=source_records(train_path)
    checkpoint=train(source)
    paired.write_json(a.output/"adapter-checkpoint.json",checkpoint)
    paired.write_json(a.output/"train-samples.json",source)

    with heldout():
        q8=cf.run_all(a.binary,a.model_q8,w0,"stock",a.output/"eval-q8-stock",EVAL_SEED)
        q4=cf.run_all(a.binary,a.model_q4,w0,"stock",a.output/"eval-q4-stock",EVAL_SEED)
        raw=cf.run_all(a.binary,a.model_q4,w0,MODE,a.output/"eval-q4-mutant-raw",EVAL_SEED)
        with adapt_server(checkpoint):
            repaired=cf.run_all(a.binary,a.model_q4,w0,MODE,a.output/"eval-q4-mutant-adapted",EVAL_SEED)

    flags=[]
    for v in cf.ALL_VARIANTS:
        f=a.output/"eval-q4-mutant-adapted"/v/(MODE+"-"+EVAL_FAMILY+"-choices.json")
        for call in json.loads(f.read_text()):
            if call["choice"] not in call["observation"]["legal"]:
                raise AssertionError("Adapter illegally bypassed action contract")
            flags.append(bool(call["adapter"]["overrode"]))

    arms={"q8_stock":q8,"q4_stock":q4,"q4_mutant_raw":raw,"q4_mutant_repaired":repaired}
    axes={name:pareto_axis(arm) for name,arm in arms.items()}
    result={
      "schema":"frankenlayers-wave16-frozen-base-repair/1",
      "source_sha":a.source_sha,"mode":MODE,"flow":list(LAB_FLOWS[MODE]),
      "immutable_q8_sha256":PINS["model"]["sha256"],"quant_receipt":receipt,
      "immutable_w0_sha256":sha256_file(a.w0),
      "train_seed":TRAIN_SEED,"train_family":TRAIN_FAMILY,
      "heldout_seed":EVAL_SEED,"heldout_family":EVAL_FAMILY,
      "heldout_geometry":{v:sorted(list(evaluation_obstacles(v))) for v in cf.ALL_VARIANTS},
      "train_records":len(source),"train_data_sha256":digest(source),
      "adapter_weights_sha256":checkpoint["weights_sha256"],
      "adapter_train_accuracy":checkpoint["train_accuracy"],
      "adapter_overrides":sum(flags),"adapter_test_decisions":len(flags),
      "arms":arms,"pareto_axes":axes,
      "real_systemone_inference":True,"real_adapter_training":True,
      "base_2b_weights_modified":False,"real_lora_training":False,
      "native_game":False,"game_code_modified":False,
      "limitations":"Synthetic W0 only: supervised local exploration/recovery target, no global goal competence. "
        "An external learned residual scorer can override System One outputs; it is not d1 weight repair. "
        "Per-decision latency speedups do not measure completed episodes."
    }
    paired.write_json(a.output/"comparison.json",result)
    print(json.dumps({"train_records":len(source),"train_accuracy":checkpoint["train_accuracy"],
      "overrides":sum(flags),"eval_decisions":len(flags),"axes":axes},sort_keys=True))


if __name__=="__main__":
    main()
