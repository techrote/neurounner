#!/usr/bin/env python3
"""Rusty-spoon M0: anatomically selected, in-place Q8_0/F16/F32 GGUF weight carving.

Copy -> mutate -> verify: never alter the immutable d1 donor. This deliberately
does NOT claim that changing quantized values changes the number of matmuls.
The Q8_0 34-byte block contains FP16 scale followed by 32 signed 8-bit values.
We edit ONLY valid quantized integer codes; original scales and metadata survive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import shutil
import struct

SPECS = {
    "mlp_scoop": {
        "layers": tuple(range(8, 16)), "suffixes": ("ffn_gate", "ffn_up", "ffn_down"),
        "op": "attenuate", "fraction": .25, "gain": .25,
        "intent": "Scoop first quarter of each middle-band FFN projection to one quarter strength",
    },
    "conv_scrape": {
        "layers": (1, 5, 9, 13, 17, 21),
        "suffixes": ("shortconv.in_proj", "shortconv.out_proj", "shortconv.conv", "ssm_in", "ssm_out", "ssm_conv1d"),
        "op": "zero", "fraction": .50,
        "intent": "Erase half of selected every-fourth early/middle/late convolution-family tensors",
    },
    "conv_scrape_10": {
        "layers": (1, 5, 9, 13, 17, 21),
        "suffixes": ("shortconv.in_proj", "shortconv.out_proj", "shortconv.conv"),
        "op": "zero", "fraction": .10,
        "intent": "Gentle 10% first-channel convolution scoop, same anatomy as the original 50% destructive specimen",
    },
    "conv_scrape_25": {
        "layers": (1, 5, 9, 13, 17, 21),
        "suffixes": ("shortconv.in_proj", "shortconv.out_proj", "shortconv.conv"),
        "op": "zero", "fraction": .25,
        "intent": "Moderate 25% first-channel convolution scoop, same anatomy as the original 50% destructive specimen",
    },
    "attn_gouge": {
        "layers": (2, 6, 10, 14, 18, 22),
        "suffixes": ("attn_q", "attn_k", "attn_output", "attn_out", "attn_o"),
        "op": "invert", "fraction": .25,
        "intent": "Reverse the signed contribution of first quarter of selected full-attention projections",
    },
}
GGUF_TYPE_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
TENSOR_TYPE_NAMES = {0:"F32", 1:"F16", 8:"Q8_0"}
PATTERN = re.compile(r"^blk\.(\d+)\.(.+)$")


class GGUFError(ValueError):
    pass


def inspect(path: Path):
    """Bounded metadata scanner: no model library and no full-weight import."""
    with path.open("rb") as f:
        def read(n):
            if not 0 <= n <= 1_000_000_000:
                raise GGUFError("Invalid GGUF length")
            b = f.read(n)
            if len(b) != n:
                raise GGUFError("Truncated GGUF file")
            return b
        def u32(): return struct.unpack("<I", read(4))[0]
        def u64(): return struct.unpack("<Q", read(8))[0]
        def string():
            n = u64()
            if n > 100_000_000:
                raise GGUFError("Oversized GGUF string")
            return read(n).decode("utf-8")
        def value(typ, depth=0):
            if depth > 2:
                raise GGUFError("Nested GGUF metadata beyond expected depth")
            if typ in GGUF_TYPE_SIZES:
                raw = read(GGUF_TYPE_SIZES[typ])
                return int.from_bytes(raw, "little") if typ in (0, 2, 4, 7, 10) else None
            if typ == 8:
                return string()
            if typ == 9:
                item_type, count = u32(), u64()
                if count > 10_000_000:
                    raise GGUFError("Oversized GGUF metadata array")
                if item_type in GGUF_TYPE_SIZES:
                    read(count * GGUF_TYPE_SIZES[item_type])
                    return None
                for _ in range(count):
                    value(item_type, depth+1)
                return None
            raise GGUFError(f"Unknown GGUF metadata type {typ}")
        if read(4) != b"GGUF":
            raise GGUFError("Missing GGUF magic")
        version = u32()
        if version not in (2,3):
            raise GGUFError(f"Unsupported GGUF version {version}")
        n_tensors, n_meta = u64(), u64()
        if not 100 <= n_tensors <= 10000 or n_meta > 10000:
            raise GGUFError("Unreasonable d1 GGUF tensor/metadata count")
        alignment = 32
        for _ in range(n_meta):
            key, typ = string(), u32()
            v = value(typ)
            if key == "general.alignment" and type(v) is int:
                alignment = v
        if alignment not in (16,32,64,128,256):
            raise GGUFError("Unsupported GGUF alignment")
        tensors = []
        for _ in range(n_tensors):
            name = string()
            dims_count = u32()
            if not 1 <= dims_count <= 4:
                raise GGUFError("Malformed tensor dimensions")
            shape = tuple(u64() for _ in range(dims_count))
            typ, rel = u32(), u64()
            if any(v == 0 for v in shape):
                raise GGUFError("Zero-dimensional tensor")
            n = math.prod(shape)
            if typ == 8:
                if n % 32:
                    raise GGUFError("Q8_0 tensor does not contain full 32-element blocks")
                length = n // 32 * 34
            elif typ in (0,1):
                length = n * (4 if typ == 0 else 2)
            else:
                length = None  # not a corruption target; don't infer unknown packed size
            tensors.append({"name":name, "shape":shape, "type":typ,
                            "type_name":TENSOR_TYPE_NAMES.get(typ, f"other-{typ}"),
                            "relative_offset":rel, "n_elements":n, "n_bytes":length})
        head = f.tell()
        base = ((head + alignment - 1) // alignment) * alignment
    size = path.stat().st_size
    for t in tensors:
        t["offset"] = base + t["relative_offset"]
        if t["n_bytes"] is not None and t["offset"] + t["n_bytes"] > size:
            raise GGUFError("Tensor extends beyond original GGUF")
    return {"version":version, "n_tensors":n_tensors, "alignment":alignment,
            "data_offset":base, "bytes":size, "tensors":tensors}


def selected(tensor, spec):
    m = PATTERN.match(tensor["name"])
    if not m or int(m.group(1)) not in spec["layers"]:
        return False
    suffix = m.group(2)
    return (tensor["type"] in (0,1,8) and
            any(suffix == name + ".weight" or suffix == name
                for name in spec["suffixes"]))


def mutate_values(view, op, gain=None):
    import numpy as np
    if op == "zero":
        view[:] = 0
    elif op == "attenuate":
        view[:] = np.rint(view.astype(np.float32) * gain).astype(view.dtype)
    elif op == "invert":
        if np.issubdtype(view.dtype, np.integer):
            view[:] = np.clip(-view.astype(np.int16), -127, 127).astype(view.dtype)
        else:
            view[:] = -view
    else:
        raise GGUFError("Unknown mutation operation")


def carve(donor: Path, output: Path, mode: str, proof: Path):
    import numpy as np
    if mode not in SPECS or donor.resolve() == output.resolve():
        raise GGUFError("Invalid mutation mode/donor")
    if output.exists():
        raise GGUFError("Refuse overwriting existing model")
    meta = inspect(donor)
    spec = SPECS[mode]
    matches = [t for t in meta["tensors"] if selected(t, spec)]
    inventory = [{"name":t["name"],"type":t["type_name"],"shape":t["shape"],
                  "n_elements":t["n_elements"]} for t in matches]
    proof.parent.mkdir(parents=True,exist_ok=True)
    receipt = {"schema":"driv3r-d1-mutagen-m0/1","mode":mode,"spec":spec,
               "donor_bytes":meta["bytes"],"gguf_version":meta["version"],
               "tensor_count":meta["n_tensors"],"eligible_tensors":inventory,
               "non_native":True,"not_training":True,"changes_compute_graph":False,
               "quantization_mutation":"Q8_0 signed qcodes, FP16 scales unchanged; FP16/F32 numerical edits",
               "status":"preflight"}
    if not matches:
        proof.write_text(json.dumps(receipt,indent=2,default=list)+"\n")
        raise GGUFError(f"No eligible actual tensor for {mode}; inspect GGUF inventory first")
    with donor.open("rb") as original_stream:
        donor_hash = hashlib.file_digest(original_stream,"sha256").hexdigest()
    receipt["donor_sha256"]=donor_hash
    shutil.copyfile(donor,output)
    changes=[]
    with output.open("r+b") as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_WRITE) as mapped:
            for t in matches:
                count=max(1,round(t["n_elements"]*spec["fraction"]))
                if t["type"] == 8:
                    # Whole Q8 blocks: no touching the FP16 scale or GGUF framing.
                    blocks=max(1,round((count // 32)))
                    count=blocks*32
                    for start in range(0,blocks,65536):
                        sub=min(65536,blocks-start)
                        offset=t["offset"]+start*34
                        block=np.ndarray((sub,34),dtype=np.uint8,buffer=mapped,offset=offset)
                        q=block[:,2:].view(np.int8)
                        mutate_values(q,spec["op"],spec.get("gain"))
                        del q, block
                elif t["type"] in (0,1):
                    dtype="<f4" if t["type"] == 0 else "<f2"
                    for start in range(0,count,65536):
                        sub=min(65536,count-start)
                        vals=np.ndarray((sub,),dtype=dtype,buffer=mapped,
                                        offset=t["offset"]+start*(4 if t["type"]==0 else 2))
                        mutate_values(vals,spec["op"],spec.get("gain"))
                        del vals
                changes.append({"name":t["name"],"elements_targeted":count,
                                "type":t["type_name"],"total_elements":t["n_elements"]})
            mapped.flush()
    check=inspect(output)
    if check["bytes"] != meta["bytes"] or check["n_tensors"] != meta["n_tensors"] or check["data_offset"] != meta["data_offset"]:
        raise GGUFError("Mutated file's structure drifted")
    with output.open("rb") as mutant_stream:
        mutated_hash=hashlib.file_digest(mutant_stream,"sha256").hexdigest()
    if mutated_hash==donor_hash:
        raise GGUFError("Mutagen produced no actual file changes")
    receipt.update({"status":"MUTATED_UNQUALIFIED_MODEL_INFERENCE_PENDING",
                    "candidate_sha256":mutated_hash,
                    "candidate_bytes":check["bytes"],
                    "modified_tensors":changes,
                    "modified_tensor_count":len(changes),
                    "total_targeted_scalars":sum(x["elements_targeted"] for x in changes),
                    "original_checkpoint_unchanged":True})
    proof.write_text(json.dumps(receipt,sort_keys=True,indent=2,default=list)+"\n")
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--donor",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=tuple(SPECS),required=True)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--inventory",type=Path,required=True)
    args=p.parse_args()
    inv=inspect(args.donor)
    args.inventory.parent.mkdir(parents=True,exist_ok=True)
    args.inventory.write_text(json.dumps({
        "schema":"driv3r-d1-gguf-inventory/1","model_bytes":inv["bytes"],
        "tensors": [{k:t[k] for k in ("name","shape","type_name","n_elements","n_bytes")}
                    for t in inv["tensors"]]
    },indent=2)+"\n")
    info=carve(args.donor,args.output,args.mode,args.receipt)
    print(json.dumps({"status":info["status"],"mode":args.mode,
                      "tensors":info["modified_tensor_count"],"targeted_scalars":info["total_targeted_scalars"],
                      "mutant_sha256":info["candidate_sha256"]}))


if __name__=="__main__":
    main()
