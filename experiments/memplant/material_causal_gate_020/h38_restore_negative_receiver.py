#!/usr/bin/env python3
"""Restore the H38 NEGATIVE diagnostic from its verified H36 neural checkpoint.
This is NOT a new scientific validation; never promote H38 as a receiver PASS.
Requires CPU PyTorch, NumPy, and original H36 seed 26011641 .pt from H36 archive.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--h36-original",type=Path,required=True)
    p.add_argument("--tensor-custody",type=Path,default=Path(__file__).with_name("H38_FROZEN_NEGATIVE_RECEIVER_TENSOR_CUSTODY.json"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    spec=json.loads(args.tensor_custody.read_text())
    if sha(args.h36_original)!=spec["baseline_h36_checkpoint_sha256"]:
        raise ValueError("H36 exact original checkpoint bytes do not match")
    base=torch.load(args.h36_original,map_location="cpu",weights_only=True)
    original={name:value.clone() for name,value in base.items()}
    if sum(v.numel() for v in base.values())!=209: raise ValueError("unexpected receiver parameter count")
    allowed={"tile.0.bias","tile.2.bias","tile.4.weight"}
    if set(spec["replacement_tensors_uint32_le"])!=allowed: raise ValueError("tensor patch scope mismatch")
    for name,bits in spec["replacement_tensors_uint32_le"].items():
        vals=np.array(bits,dtype=np.uint32).view(np.float32)
        if vals.size!=base[name].numel(): raise ValueError(f"shape mismatch {name}")
        base[name]=torch.tensor(vals.reshape(base[name].shape).copy(),dtype=torch.float32)
        if not np.array_equal(base[name].numpy().view(np.uint32).ravel(),np.array(bits,dtype=np.uint32)):
            raise AssertionError(f"float32 payload mismatch {name}")
    for name in original:
        if name not in allowed and not torch.equal(base[name],original[name]):
            raise AssertionError(f"unauthorized alteration {name}")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(base,args.output)
    print(json.dumps({"classification":"H38 NEGATIVE TRAINING-ONLY; NOT SCIENTIFIC ACCEPTANCE",
      "tensors_restored":sorted(allowed),"params":209,
      "float32_bit_exact":True,"original_sha256":sha(args.h36_original),
      "serialized_reconstruction_sha256":sha(args.output),
      "reference_serialized_sha256":spec["h38_negative_checkpoint_sha256"]},indent=2))

if __name__=="__main__":main()
