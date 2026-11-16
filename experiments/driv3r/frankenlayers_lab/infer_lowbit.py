#!/usr/bin/env python3
"""Model-only IQ3_XS W0 screen with same-runner Q8 and IQ3 stock controls.

This is deliberately a 3x16-decision screen, not a claim of trained
competence, native gameplay, or held-out recovery. Requantizing Q8 rather
than original FP16 may damage the model; log failures as evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.driv3r.backend import PINS, sha256_file
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
from experiments.driv3r.frankenlayers.infer import load_w0, run_arm, write_json
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows

MODES = ("aero4_no10", "aero4_no2")


def validate_iq3_receipt(receipt: dict, q8: Path, iq3: Path, imatrix: Path, calibration: Path) -> None:
    if receipt.get("schema") != "frankenlayers-iq3-requant/2":
        raise ValueError("Unrecognised low-bit conversion receipt")
    if receipt.get("quant") != "IQ3_XS" or receipt.get("allow_requantize") is not True:
        raise ValueError("Only source-pinned Q8 -> IQ3_XS requantisation authorised")
    if receipt.get("q8_sha256") != PINS["model"]["sha256"]:
        raise ValueError("IQ3 recipe points to the wrong Q8 donor")
    if sha256_file(q8) != PINS["model"]["sha256"]:
        raise ValueError("Q8 donor content mismatch")
    if not iq3.is_file() or iq3.stat().st_size < 1_000_000:
        raise ValueError("IQ3 file missing or implausibly small")
    if receipt.get("iq3_sha256") != sha256_file(iq3):
        raise ValueError("Actual IQ3 file differs from conversion receipt")
    if receipt.get("imatrix_source") != "Q8 donor, train-only synthetic game prompts":
        raise ValueError("IQ3 must use the declared source-pinned calibration")
    if receipt.get("calibration_chunks") != 8:
        raise ValueError("Unexpected calibration chunk budget")
    if not imatrix.is_file() or imatrix.stat().st_size < 1000:
        raise ValueError("Missing calibration importance matrix")
    if receipt.get("imatrix_sha256") != sha256_file(imatrix):
        raise ValueError("Importance matrix differs from receipt")
    if not calibration.is_file() or calibration.stat().st_size < 15000:
        raise ValueError("Missing game calibration text")
    if receipt.get("calibration_sha256") != sha256_file(calibration):
        raise ValueError("Calibration text differs from receipt")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model-q8", type=Path, required=True)
    p.add_argument("--model-iq3", type=Path, required=True)
    p.add_argument("--quant-receipt", type=Path, required=True)
    p.add_argument("--imatrix", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--w0", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, choices=(570, 571), required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Refusing to overwrite any experiment evidence")
    validate_flows()
    if any(name in FLOWS for name in LAB_FLOWS):
        raise ValueError("Inherited and lab model registries collide")
    receipt = json.loads(a.quant_receipt.read_text())
    validate_iq3_receipt(receipt, a.model_q8, a.model_iq3, a.imatrix, a.calibration)
    FLOWS.update(LAB_FLOWS)
    a.output.mkdir(parents=True)
    w0 = load_w0(a.w0)

    # All three arms share the same pinned source-built llama runtime, model
    # family, W0 game host, runner, seed and experimental output directory.
    # The IQ3 stock arm isolates quantisation from layer surgery.
    stock_q8 = run_arm(a.binary, a.model_q8, w0, "stock", a.output / "q8-stock", a.seed)
    stock_iq3 = run_arm(a.binary, a.model_iq3, w0, "stock", a.output / "iq3-stock", a.seed)
    mutant_iq3 = run_arm(a.binary, a.model_iq3, w0, a.mode, a.output / "iq3-mutant", a.seed)

    def ratio(baseline, candidate):
        return baseline / candidate if baseline is not None and candidate else None

    report = {
        "schema": "frankenlayers-iq3-w0-same-runner/1",
        "source_sha": a.source_sha,
        "seed": a.seed,
        "mode": a.mode,
        "flow": list(LAB_FLOWS[a.mode]),
        "quant_receipt": receipt,
        "immutable_w0_sha256": sha256_file(a.w0),
        "immutable_q8_sha256": PINS["model"]["sha256"],
        "q8_stock": stock_q8,
        "iq3_stock": stock_iq3,
        "iq3_mutant": mutant_iq3,
        "speedup_vs_q8_stock_p50": ratio(stock_q8["p50_seconds"], mutant_iq3["p50_seconds"]),
        "speedup_vs_iq3_stock_p50": ratio(stock_iq3["p50_seconds"], mutant_iq3["p50_seconds"]),
        "real_inference": True,
        "real_training": False,
        "native_workbench": False,
        "game_code_modified": False,
        "limitations": "W0 3x16 only. IQ3_XS is requantized from Q8 with a bounded synthetic game-prompt imatrix. "
                       "Legal actions, nonterminal survival and latency do not establish useful game strategy.",
    }
    write_json(a.output / "comparison.json", report)
    print(json.dumps({
        "mode": a.mode, "seed": a.seed,
        "iq3_mutant_choices": mutant_iq3["choice_counts"],
        "iq3_mutant_valid_decisions": mutant_iq3["valid_decisions"],
        "ratio_to_q8_stock": report["speedup_vs_q8_stock_p50"],
        "ratio_to_iq3_stock": report["speedup_vs_iq3_stock_p50"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
