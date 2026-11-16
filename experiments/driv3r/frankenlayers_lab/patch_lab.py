#!/usr/bin/env python3
"""Extend the pinned #626 *decision-only* runtime patch with 6/4-layer flows.

No changes to GGUF bytes or the game's native, physics, Godot or Workbench code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from experiments.driv3r.frankenlayers.patch_runtime import (
    PINNED_LLAMA_COMMIT, TARGET, patch as patch_phase2,
)
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows

UNKNOWN = '''        } else {
            throw std::runtime_error("Unknown CYBERSAND_D1_FRANKEN_FLOW: " + mode);
'''
LOOP = '''    for (size_t d1_step = 0; d1_step < d1_flow.size(); ++d1_step) {
'''
# The original graph constructs both full and sliding attention input tensors;
# a sparse flow must actually consume each, otherwise ggml may abort at startup.
ATTN_GUARD = '''    if constexpr (decision) {
        bool d1_has_full = false;
        bool d1_has_sliding = false;
        for (const int source_il : d1_flow) {
            if (source_il < 0 || source_il >= n_layer) {
                throw std::runtime_error("Experimental flow layer out of range");
            }
            if (!hparams.is_recr(source_il)) {
                if (hparams.is_swa(source_il)) {
                    d1_has_sliding = true;
                } else {
                    d1_has_full = true;
                }
            }
        }
        if (!d1_has_full || !d1_has_sliding) {
            throw std::runtime_error("Experimental flow lacks full/sliding attention ballast");
        }
    }

'''

def cpp_list(indices: tuple[int, ...]) -> str:
    return "{" + ",".join(str(i) for i in indices) + "}"

def patch(root: Path) -> dict:
    validate_flows()
    base_receipt = patch_phase2(root)
    path = root / TARGET
    original_patched = path.read_text()
    if original_patched.count(UNKNOWN) != 1 or original_patched.count(LOOP) != 1:
        raise RuntimeError("Expected exact phase-2 switch/loop markers; fail closed")
    extras = "".join(
        f'        }} else if (mode == "{name}") {{\n'
        f'            d1_flow = {cpp_list(layers)};\n'
        for name, layers in LAB_FLOWS.items()
    )
    patched = original_patched.replace(UNKNOWN, extras + UNKNOWN)
    patched = patched.replace(LOOP, ATTN_GUARD + LOOP)
    path.write_text(patched)
    return {
        "schema": "cybersand-frankenlayers-lab-patch/1",
        "base_phase2": base_receipt,
        "llama_source_commit": PINNED_LLAMA_COMMIT,
        "target": str(TARGET),
        "phase2_patched_sha256": hashlib.sha256(original_patched.encode()).hexdigest(),
        "lab_patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "lab_flows": {name: list(layers) for name, layers in LAB_FLOWS.items()},
        "attention_guard": "full-and-sliding-required-by-model-metadata",
        "decision_only": True,
        "weights_modified": False,
        "game_code_modified": False,
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llama-root", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    args = p.parse_args()
    receipt = patch(args.llama_root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))

if __name__ == "__main__":
    main()
