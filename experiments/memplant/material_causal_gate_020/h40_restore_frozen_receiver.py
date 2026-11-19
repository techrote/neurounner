#!/usr/bin/env python3
"""DG020 H39/H40 exact numerical receiver restoration from Git text custody.

No source donor/checkpoint binaries, family labels, or oracle answers are needed.
Uses original H33b 209-float32 tensor custody, H36 eight-coefficient frozen head
patches and H39 three-float32-element patches. PyTorch .pt container serialization
may differ from the historical binary; numerical tensor bit patterns MUST match.
H39 is a fitted exposed-training-only correction, not by itself a prospective gate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from h33_restore_receiver_weights import load_state

HERE = Path(__file__).resolve().parent
H36 = HERE / "H36_FROZEN_RECEIVER_HEAD_PATCHES.json"
H39 = HERE / "H39_FROZEN_EXPOSED_RECEIVER_TENSOR_CUSTODY.json"


def tensor_sha(state):
    raw = b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in state.values())
    return hashlib.sha256(raw).hexdigest()


def restore(seed, h36=H36, h39=H39):
    assert seed in (26011639, 26011640, 26011641)
    state = load_state(f"h33b-{seed}")
    assert sum(v.numel() for v in state.values()) == 209
    doc36 = json.loads(Path(h36).read_text(encoding="utf-8"))["seeds"][str(seed)]
    bits = bytes.fromhex(doc36["head_8_f32_le_hex"])
    head = np.frombuffer(bits, dtype="<f4").copy()
    assert len(head) == 8 and tuple(state["tile.4.weight"].shape) == (1, 8)
    state["tile.4.weight"] = torch.from_numpy(head.reshape(1, 8))
    assert tensor_sha(state) == doc36["all_tensors_ordered_sha256"], "H36 reconstruction failed"
    if seed == 26011641:
        patch = json.loads(Path(h39).read_text(encoding="utf-8"))
        assert patch["original_h36_serialized_checkpoint_sha256"] == doc36["original_serialized_pt_sha256"]
        assert patch["changed_tensors"] == ["tile.0.weight", "tile.0.bias"]
        for key, info in patch["replacement_elements_uint32_le"].items():
            arr = state[key].detach().cpu().numpy().copy()
            assert list(arr.shape) == info["shape"]
            u = arr.reshape(-1).view("<u4")
            ix = info["changed_flat_indices"]
            assert u[ix].tolist() == info["original_base_elements_uint32_le"], "wrong H36 baseline element"
            u[ix] = np.asarray(info["replacement_elements_uint32_le"], dtype="<u4")
            assert hashlib.sha256(arr.tobytes()).hexdigest() == info["full_restored_tensor_float32_le_sha256"]
            state[key] = torch.from_numpy(arr)
        assert tensor_sha(state) == patch["concatenated_complete_state_float32_le_sha256"], "H39 reconstruction failed"
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in (26011639, 26011640, 26011641):
        state = restore(seed)
        name = "h39" if seed == 26011641 else "h36"
        dest = args.out / f"receiver-{name}-reconstructed-{seed}.pt"
        torch.save(state, dest)
        print(seed, "209 restored float32 tensors verified", tensor_sha(state), dest)


if __name__ == "__main__":
    main()
