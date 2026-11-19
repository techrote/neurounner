#!/usr/bin/env python3
"""Restore exact H36 trained 209-parameter receiver numerical tensor states.

Only final eight weights differ from Git-backed H33b numerical custody.
Original serialized H36 .pt bytes remain in the H36 external evidence capsule.
No oracle, family, law/source labels or training data enter reconstruction.
"""
import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

def base_state(seed, source):
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    assert data["format"] == "dg020-ieee754-f32-le-v1"
    record = data["models"][f"h33b-{seed}"]
    raw = zlib.decompress(base64.b64decode(record["zlib_b64"], validate=True))
    assert hashlib.sha256(raw).hexdigest() == record["raw_sha256"]
    floats = np.frombuffer(raw, dtype="<f4")
    assert floats.size == 209
    state = {}
    index = 0
    for key, shape in data["keys"]:
        count = int(np.prod(shape))
        state[key] = torch.from_numpy(floats[index:index+count].copy().reshape(shape))
        index += count
    assert index == 209
    return state

def restore(seed, base, heads):
    data = json.loads(Path(heads).read_text(encoding="utf-8"))
    assert data["format"] == "dg020-h36-8-output-head-f32-le-v1"
    record = data["seeds"][str(seed)]
    state = base_state(seed, base)
    weight = np.frombuffer(bytes.fromhex(record["head_8_f32_le_hex"]), dtype="<f4").copy().reshape((1, 8))
    state["tile.4.weight"] = torch.from_numpy(weight)
    raw = b"".join(t.detach().numpy().astype("<f4").tobytes() for t in state.values())
    assert len(raw) == 209 * 4
    assert hashlib.sha256(raw).hexdigest() == record["all_tensors_ordered_sha256"]
    return state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=HERE / "H33_FROZEN_RECEIVER_TENSOR_CUSTODY.json")
    parser.add_argument("--heads", type=Path, default=HERE / "H36_FROZEN_RECEIVER_HEAD_PATCHES.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in (26011639, 26011640, 26011641):
        state = restore(seed, args.base, args.heads)
        dest = args.out / f"h36-numeric-restored-{seed}.pt"
        torch.save(state, dest)
        print(seed, "209 bit-exact numerical tensor parameters restored:", dest)

if __name__ == "__main__":
    main()
