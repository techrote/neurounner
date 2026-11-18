#!/usr/bin/env python3
"""Restore H33/H33b bit-exact float32 receiver state tensors from Git custody.

This restores the *numerical checkpoint state*; PyTorch .pt container-byte
hashes may differ from the original serialized files. No law IDs, oracle data,
training samples or costs are used in reconstruction. Replayer must separately
provide genuine frozen donors and original exposed H27 training observations.
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
ARCHIVE = HERE / "H33_FROZEN_RECEIVER_TENSOR_CUSTODY.json"


def load_state(name, source=ARCHIVE):
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    assert data["format"] == "dg020-ieee754-f32-le-v1"
    model = data["models"][name]
    raw = zlib.decompress(base64.b64decode(model["zlib_b64"], validate=True))
    assert hashlib.sha256(raw).hexdigest() == model["raw_sha256"]
    assert len(raw) == 4 * sum(int(np.prod(shape)) for _, shape in data["keys"])
    floats = np.frombuffer(raw, dtype="<f4")
    state = {}
    offset = 0
    for key, shape in data["keys"]:
        count = int(np.prod(shape))
        state[key] = torch.from_numpy(floats[offset:offset + count].copy().reshape(shape))
        offset += count
    assert offset == 209
    assert sum(v.numel() for v in state.values()) == 209
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for variant in ("h33", "h33b"):
        for seed in (26011639, 26011640, 26011641):
            name = f"{variant}-{seed}"
            state = load_state(name)
            target = args.out / f"receiver-{name}-restored.pt"
            torch.save(state, target)
            print(name, target, "209 bit-exact numerical weights restored")


if __name__ == "__main__":
    main()
