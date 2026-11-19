#!/usr/bin/env python3
"""Rebuild exact H52 209-parameter exposed-only output-head candidate from authentic H45.

Requires original H45 trained checkpoint from preserved H50/H51 evidence;
does not perform retraining, change hidden layers or create new model families.
"""
import hashlib, struct, sys
from pathlib import Path
import torch

SOURCE_SHA = "7eaae96ba2b8e73bbfdb144f1d86eb78d101d887463b5002e6a2291d819458a6"
CANDIDATE_SHA = "b1662812437ae4a83e16f0f345e2c8291acf48e8748776340d98f4be9a9fc9fc"
HEAD_FLOAT32_LE_HEX = "2d0ab840 43b43540 68703e41 b5b534c1 491014c0 6a343fc1 58565ec1 016a9241"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rebuild(src, dst):
    assert sha(src) == SOURCE_SHA, "Original H45 checkpoint SHA-256 mismatch"
    state = torch.load(src, weights_only=True, map_location="cpu")
    assert sum(x.numel() for x in state.values()) == 209
    raw = bytes.fromhex(HEAD_FLOAT32_LE_HEX)
    state["tile.4.weight"] = torch.tensor(
        struct.unpack("<8f", raw), dtype=torch.float32
    ).reshape(1, 8)
    torch.save(state, dst)
    assert sha(dst) == CANDIDATE_SHA, (
        "Reconstructed checkpoint mismatch; inspect PyTorch serialization/runtime"
    )
    print("H52 reconstruction SHA-256 verified:", CANDIDATE_SHA)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python h52_rebuild_head_patch.py ORIGINAL_H45.pt "
            "H52-exposed-head-LP-candidate-26011640.pt"
        )
    rebuild(Path(sys.argv[1]), Path(sys.argv[2]))
