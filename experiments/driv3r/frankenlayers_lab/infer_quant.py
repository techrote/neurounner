#!/usr/bin/env python3
"""Three-way actual System One Q8-stock / Q4-stock / Q4-mutant comparison.

No model training here, no game edits. A Q4 model is ephemeral; the source
donor hash, transformation receipt and actual Q4 digest are retained.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from experiments.driv3r.backend import PINS, sha256_file
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
from experiments.driv3r.frankenlayers.infer import load_w0, run_arm, write_json
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows

# Qualified Q8 survivors; all Q4 attempts require original same-runner Q8/Q4 controls.
QUANT_MODES = ("aero6", "skele6", "aero5_no23", "tailflip4", "echo4", "aero4_no10")

def validate_quant_receipt(receipt: dict, q8: Path, q4: Path) -> None:
    if receipt.get("schema") != "cybersand-franken-q4-requant/1":
        raise ValueError("Invalid quantisation receipt schema")
    if receipt.get("quant") != "Q4_K_M" or receipt.get("allow_requantize") is not True:
        raise ValueError("Q4 receipt is not a supported reproducible conversion")
    if receipt.get("q8_sha256") != PINS["model"]["sha256"]:
        raise ValueError("Original donor SHA does not match immutable pin")
    if sha256_file(q8) != PINS["model"]["sha256"]:
        raise ValueError("Q8 donor bytes altered")
    if not q4.is_file() or q4.stat().st_size < 1000000:
        raise ValueError("Q4 quantised model missing or implausibly small")
    if receipt.get("q4_sha256") != sha256_file(q4):
        raise ValueError("Q4 actual bytes do not match conversion receipt")

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--model-q8",type=Path,required=True)
    p.add_argument("--model-q4",type=Path,required=True)
    p.add_argument("--quant-receipt",type=Path,required=True)
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--mode",choices=QUANT_MODES,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=570)
    p.add_argument("--source-sha",required=True)
    a=p.parse_args()
    validate_flows()
    if set(FLOWS).intersection(LAB_FLOWS):
        raise RuntimeError("Phase2 registry collision")
    FLOWS.update(LAB_FLOWS)
    receipt=json.loads(a.quant_receipt.read_text())
    validate_quant_receipt(receipt,a.model_q8,a.model_q4)
    if a.output.exists():
        raise RuntimeError("Refusing evidence overwrite")
    a.output.mkdir(parents=True)
    w0=load_w0(a.w0)
    # All arms same source-built runtime, CPU runner, seed, legal-action host
    # and tokenised observations. Absolute speed is still CPU-load sensitive.
    q8_stock=run_arm(a.binary,a.model_q8,w0,"stock",a.output/"q8-stock",a.seed)
    q4_stock=run_arm(a.binary,a.model_q4,w0,"stock",a.output/"q4-stock",a.seed)
    q4_mutant=run_arm(a.binary,a.model_q4,w0,a.mode,a.output/"q4-mutant",a.seed)
    def ratio(left,right):
        return left/right if left is not None and right else None
    report={
        "schema":"cybersand-frankenlayers-q4-three-way/1",
        "source_sha":a.source_sha,
        "llama_source_commit":"8642541b255bfb5dc39cd351a178be09013a55d6",
        "seed":a.seed,"mode":a.mode,"quant_receipt":receipt,
        "q8_stock":q8_stock,"q4_stock":q4_stock,"q4_mutant":q4_mutant,
        "speedup_q4_mutant_vs_q8_stock_p50":ratio(q8_stock["p50_seconds"],q4_mutant["p50_seconds"]),
        "speedup_q4_mutant_vs_q4_stock_p50":ratio(q4_stock["p50_seconds"],q4_mutant["p50_seconds"]),
        "speedup_q4_stock_vs_q8_stock_p50":ratio(q8_stock["p50_seconds"],q4_stock["p50_seconds"]),
        "actual_systemone_inference":True,"native_workbench":False,
        "training":False,"game_code_modified":False,
        "quantization_note":"Q4_K_M requantised from Q8 donor; distinct from F16-source Q4, potential additional degradation",
    }
    write_json(a.output/"comparison.json",report)
    print(json.dumps({
        "mode":a.mode,"q8_stock_p50":q8_stock["p50_seconds"],
        "q4_stock_p50":q4_stock["p50_seconds"],
        "q4_mutant_p50":q4_mutant["p50_seconds"],
        "speedup_vs_q8_stock":report["speedup_q4_mutant_vs_q8_stock_p50"],
        "speedup_vs_q4_stock":report["speedup_q4_mutant_vs_q4_stock_p50"],
        "mutant_choices":q4_mutant["choice_counts"],
    },sort_keys=True))

if __name__=="__main__":
    main()
