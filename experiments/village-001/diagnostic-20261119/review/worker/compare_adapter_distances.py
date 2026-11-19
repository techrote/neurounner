#!/usr/bin/env python3
"""Describe numerical F adapter distances from accepted TAR bytes; never load a model.

No exact reproduction flag or frozen decision threshold is changed. The left
adapter is the reference for relative L2. F32 parameters are converted exactly
to float64 before subtraction; squared norms are accumulated per tensor with
NumPy and combined across tensors with math.fsum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import struct
import tarfile

import numpy as np

PANEL = ["V-G01-A", "V-G03-B", "V-G05-C", "V-G07-A", "V-G09-B", "V-G11-C"]
REL = Path("experiments/village-001/diagnostic-20261119")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def unique(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate safetensors header key")
        result[key] = value
    return result


def load_arrays(archive, expected_asset, accepted):
    """Read one bounded regular TAR member without extracting any path."""
    require(archive.is_file() and not archive.is_symlink(), "Missing/nonregular adapter archive")
    require(archive.stat().st_size == expected_asset["bytes"] and sha(archive) == expected_asset["sha256"],
            "Accepted archive size/hash mismatch: " + archive.name)
    with tarfile.open(archive, "r:gz") as tar:
        members, seen, weights = tar.getmembers(), set(), []
        require(0 < len(members) <= 1000 and sum(x.size for x in members) <= 256 * 1024 * 1024,
                "Adapter TAR exceeds bound")
        for member in members:
            name = member.name.rstrip("/")
            path = PurePosixPath(name)
            require(name and not path.is_absolute() and all(p not in ("", ".", "..") for p in name.split("/"))
                    and "\\" not in name and ":" not in name and "\x00" not in name, "Unsafe TAR path")
            require(name not in seen and (member.isdir() or member.isfile()), "Duplicate/link/special TAR entry")
            require(0 <= member.size <= 128 * 1024 * 1024, "Invalid TAR member size")
            seen.add(name)
            if path.name == "adapter_model.safetensors":
                require(member.isfile(), "Safetensors member is not a regular file")
                weights.append(member)
        require(len(weights) == 1, "Adapter must contain exactly one safetensors payload")
        stream = tar.extractfile(weights[0])
        require(stream is not None, "Missing safetensors stream")
        with stream:
            data = stream.read()
        require(len(data) == weights[0].size, "Truncated TAR payload")
    require(sha_bytes(data) == accepted["safetensors_file_sha256"], "Safetensors differs from acceptance")
    require(len(data) >= 8, "Missing safetensors length")
    header_length = struct.unpack("<Q", data[:8])[0]
    require(0 < header_length <= 16 * 1024 * 1024 and 8 + header_length < len(data), "Invalid header length")
    header = json.loads(data[8:8 + header_length], object_pairs_hook=unique)
    require(isinstance(header, dict), "Invalid safetensors header")
    if "__metadata__" in header:
        require(isinstance(header["__metadata__"], dict), "Invalid safetensors metadata")
    payload = memoryview(data)[8 + header_length:]
    arrays, fingerprints, intervals = {}, {}, []
    aggregate = hashlib.sha256()
    for name, row in sorted(header.items()):
        if name == "__metadata__":
            continue
        require(isinstance(name, str) and name and isinstance(row, dict)
                and set(row) == {"dtype", "shape", "data_offsets"}, "Invalid tensor descriptor")
        require(row["dtype"] == "F32", "This descriptive comparison is restricted to the accepted F32 adapters")
        shape, offsets = row["shape"], row["data_offsets"]
        require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), "Invalid shape")
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(n) is int for n in offsets), "Invalid offsets")
        start, end = offsets
        require(0 <= start <= end <= len(payload) and end - start == math.prod(shape) * 4, "Invalid tensor range")
        intervals.append((start, end))
        raw = payload[start:end]
        fingerprint = {"dtype": "torch.float32", "shape": shape, "sha256": sha_bytes(raw)}
        fingerprints[name] = fingerprint
        aggregate.update(canonical({"name": name, **fingerprint}))
        array = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        require(np.isfinite(array).all(), "Nonfinite parameter value")
        arrays[name] = array
    cursor = 0
    for start, end in sorted(intervals):
        require(start == cursor, "Overlapping/gapped tensor payload")
        cursor = end
    require(cursor == len(payload) and len(arrays) == accepted["tensor_count"] == 420,
            "Incomplete accepted tensor inventory")
    require(aggregate.hexdigest() == accepted["canonical_tensor_sha256"], "Canonical tensor fingerprint differs from acceptance")
    receipt = {"archive": archive.name, "archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive),
        "safetensors_tar_member": weights[0].name, "safetensors_bytes": len(data),
        "safetensors_file_sha256": sha_bytes(data), "canonical_tensor_sha256": aggregate.hexdigest(),
        "adapter_directory_sha256": accepted["directory_sha256"], "tensor_count": len(arrays),
        "elements": sum(x.size for x in arrays.values()), "dtype": "F32"}
    return arrays, fingerprints, receipt


def compare(left, right, left_fingerprints, right_fingerprints):
    require(left.keys() == right.keys(), "Compared tensor names differ")
    rows, squares, left_squares, right_squares = [], [], [], []
    for name in sorted(left):
        a, b = left[name], right[name]
        left_fp, right_fp = left_fingerprints[name], right_fingerprints[name]
        require(left_fp["shape"] == right_fp["shape"] and a.shape == b.shape, "Compared tensor shapes differ")
        delta = b - a
        squared = float(np.sum(delta * delta, dtype=np.float64))
        norm_a = float(np.sum(a * a, dtype=np.float64))
        norm_b = float(np.sum(b * b, dtype=np.float64))
        distance, baseline = math.sqrt(squared), math.sqrt(norm_a)
        squares.append(squared)
        left_squares.append(norm_a)
        right_squares.append(norm_b)
        rows.append({"name": name, "dtype": "F32", "shape": left_fp["shape"], "elements": int(a.size),
            "left_sha256": left_fp["sha256"], "right_sha256": right_fp["sha256"],
            "hash_equal": left_fp["sha256"] == right_fp["sha256"], "numeric_changed_elements": int(np.count_nonzero(delta)),
            "max_absolute_difference": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "l2_difference": distance, "left_l2_norm": baseline, "right_l2_norm": math.sqrt(norm_b),
            "relative_l2_difference": distance / baseline if baseline else None})
    distance, norm_a, norm_b = (math.sqrt(math.fsum(v)) for v in (squares, left_squares, right_squares))
    return {"tensor_count": len(rows), "elements": sum(r["elements"] for r in rows),
        "hash_differing_tensors": sum(not r["hash_equal"] for r in rows),
        "numeric_changed_elements": sum(r["numeric_changed_elements"] for r in rows),
        "max_absolute_difference": max(r["max_absolute_difference"] for r in rows),
        "l2_difference": distance, "left_l2_norm": norm_a, "right_l2_norm": norm_b,
        "relative_l2_difference": distance / norm_a if norm_a else None, "per_tensor": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--input-assets", type=Path, required=True)
    parser.add_argument("--reference-assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists() and args.repo.resolve() not in args.out.resolve().parents, "Use a new scratch output path")
    accepted = read(args.acceptance)
    require(accepted["input_acceptance"] == "pass" and accepted["adapter_bytes"]["pass"]
            and accepted["input_release"]["pass"], "Independent input/adapter acceptance is required first")
    plan_path = args.repo / REL / "worker-plan.json"
    plan = read(plan_path)
    require(sha(plan_path) == accepted["source"]["worker_plan_sha256"], "Worker plan differs from acceptance")
    accepted_adapters = {(r["member_id"], r["stage"]): r for r in accepted["adapter_bytes"]["adapters"]}
    originals = {r["member_id"]: r for r in plan["adapter_downloads"]["D_ORIGINAL"]["assets"]}
    findings, asset_receipts = [], []
    for member in PANEL:
        loaded = {}
        for stage in ("D_ORIGINAL", "D_REPRODUCED", "D_CORRECTED"):
            row = accepted_adapters[(member, stage)]
            require(row["all_tensor_bytes_match_native_receipts"], "Adapter did not pass byte-level acceptance")
            asset = originals[member] if stage == "D_ORIGINAL" else accepted["input_release"]["verified_assets"][row["asset"]]
            require(asset["file"] == row["asset"], "Accepted archive name mismatch")
            directory = args.reference_assets if stage == "D_ORIGINAL" else args.input_assets
            arrays, fingerprints, receipt = load_arrays(directory / asset["file"], asset, row)
            loaded[stage] = (arrays, fingerprints)
            asset_receipts.append({"member_id": member, "stage": stage, **receipt,
                "release_tag": plan["adapter_downloads"]["D_ORIGINAL"]["release_tag"] if stage == "D_ORIGINAL" else accepted["input_release"]["tag"]})
        comparisons = {}
        for label, left, right in (("original_D_vs_reproduced_D", "D_ORIGINAL", "D_REPRODUCED"),
                                  ("reproduced_D_vs_corrected_D", "D_REPRODUCED", "D_CORRECTED")):
            comparisons[label] = {"left_reference": left, "right": right,
                **compare(loaded[left][0], loaded[right][0], loaded[left][1], loaded[right][1])}
        findings.append({"member_id": member, **comparisons})
    report = {"schema": "village-001f-descriptive-adapter-distances-v1",
        "input_publication_commit": accepted["audit_head"], "input_release_tag": accepted["input_release"]["tag"],
        "scientific_source_commit": accepted["source_freeze_commit"], "acceptance_report_sha256": sha(args.acceptance),
        "worker_plan_sha256": sha(plan_path), "analysis_script_sha256": sha(Path(__file__)),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "method": "Read verified regular TAR safetensors bytes directly; verify F32 canonical fingerprints; convert F32 exactly to float64 before right-minus-left subtraction; NumPy float64 sum within tensors and math.fsum across tensors; L2=sqrt(sum squared differences), relative L2=L2/left-reference L2 norm. A zero reference norm yields null relative L2.",
        "assets": asset_receipts, "per_member": findings,
        "interpretation_limits": ["Descriptive numerical distances have no prespecified acceptance threshold; exact byte/tensor reproduction flags remain unchanged.",
            "Adapter-parameter distance is not a task-quality metric or a distance between full merged model functions. LoRA factor parameterization affects its scale.",
            "Original-D and reproduced-D execution environments differ; these distances do not identify the source of that difference. The paired mask arms use the actual matched host/runtime.",
            "No held-out quality conclusion follows until the frozen blinded judging is complete."],
        "raw_evidence_modified": False, "model_loaded": False, "new_model_tokenizer_or_provider_calls": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out), "sha256": sha(args.out), "members": [{"member_id": row["member_id"],
        **{key: {k: v for k, v in value.items() if k != "per_tensor"} for key, value in row.items() if isinstance(value, dict)}}
        for row in findings]}, indent=2))


if __name__ == "__main__":
    main()
