"""MZ-016: deterministic, reversible-from-source low-bit weight audit.

Research-only compression screening of the original MZ-015 closed-loop policies.
No training or Workbench integration; weights reconstructed from an owner-supplied
immutable checkpoint. This does not use compressed CPU kernels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import statistics
import time
from pathlib import Path

from experiments.driv3r.model_zoo.spatial_paint_mz013.run import models
from experiments.driv3r.model_zoo.turning_maps_mz015.run import rollout, D

SCHEMA = "MZ016-LOWBIT-FROZEN-v1"
SOURCE_SCHEMA = "MZ015-TURNING-MAPS-v1"
SOURCE_SHA = "807ac25373c31649dd5a92a9aaa503098a86183b"
EXPECTED_SOURCE_SHA256 = "0c3d60f6259e9b9504843883c1bc06459720b0789748d22dcdf1ac4bb3071c5f"


def quantize(weight, *, bits, threshold=0.5):
    """Output-channel-wise quantization, returned as dequantized float32 tensor.

    Ternary: mask |w| >= threshold * per-row mean |w|; retained values
    become +/- mean retained |w|. 4/8-bit: symmetric max-abs signed grid.
    Zero rows remain zero. No mutation of the original tensor.
    """
    if weight.ndim != 2 or bits not in (2, 4, 8):
        raise ValueError("only 2-D Linear/GRU matrices and 2/4/8 bits")
    w = weight.detach().float()
    if bits == 2:
        keep = w.abs() >= threshold * w.abs().mean(dim=1, keepdim=True)
        count = keep.sum(dim=1, keepdim=True).clamp(min=1)
        scale = (w.abs() * keep).sum(dim=1, keepdim=True) / count
        out = w.sign() * keep * scale
    else:
        levels = 7 if bits == 4 else 127
        scale = w.abs().amax(dim=1, keepdim=True) / levels
        scale = scale.clamp(min=1e-12)
        out = (w / scale).round().clamp(-levels, levels) * scale
    return out.to(weight.dtype)


def group_for(name, model):
    if model == "diagonal":
        if name.startswith("cells.0."):
            return "recurrent_0"
        if name.startswith("cells.1."):
            return "recurrent_1"
        if name.startswith("head."):
            return "head"
    elif model == "gru":
        if name.startswith("core.weight_ih_"):
            return "input_recurrence"
        if name.startswith("core.weight_hh_"):
            return "state_recurrence"
        if name.startswith("head."):
            return "head"
    return None


def convert(net, model, *, bits, threshold, selection):
    """Return an independent dequantized evaluation copy and storage estimate."""
    import torch
    clone = copy.deepcopy(net)
    total_params = sum(p.numel() for p in clone.parameters())
    q_count = zeros = rows = l1_error = 0
    names = []
    with torch.no_grad():
        for name, parameter in clone.named_parameters():
            group = group_for(name, model) if ("weight" in name) else None
            if parameter.ndim != 2 or group is None:
                continue
            if selection not in ("all", group):
                continue
            quantized = quantize(parameter, bits=bits, threshold=threshold)
            q_count += parameter.numel()
            zeros += int((quantized == 0).sum())
            rows += parameter.shape[0]
            l1_error += float((quantized - parameter).abs().sum())
            names.append(name)
            parameter.copy_(quantized)
    if not names:
        raise ValueError(f"No matrices selected: {model}/{selection}")
    original_bytes = total_params * 4
    packed_lower_bytes = (total_params - q_count) * 4 + q_count * bits / 8 + rows * 4
    return clone, {
        "matrix_names": names,
        "quantized_values": q_count,
        "original_parameters": total_params,
        "zero_fraction_in_quantized": round(zeros / q_count, 6),
        "mean_abs_quant_error": round(l1_error / q_count, 8),
        "nominal_fp32_bytes": original_bytes,
        "estimated_packed_bytes": round(packed_lower_bytes),
        "estimated_packed_ratio": round(packed_lower_bytes / original_bytes, 5),
        "physical_torch_storage": "float32 dequantized; no packed kernel or runtime-memory saving",
    }


def timing(torch, net, decisions=40, repetitions=13):
    net.eval()
    x = torch.zeros(1, D)
    vals = []
    with torch.inference_mode():
        for rep in range(repetitions):
            state = None
            t0 = time.perf_counter_ns()
            for _ in range(decisions):
                _, state = net.step(x, state)
            if rep >= 3:
                vals.append((time.perf_counter_ns()-t0) / 1e6 / decisions)
    return {"p50_ms": round(statistics.median(vals), 5),
            "p95_ms": round(sorted(vals)[-1], 5),
            "measurement": "float32/dequantized batch1 cached single CPU thread"}


def main(args):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(16)
    path = Path(args.checkpoint)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256 and not args.allow_unverified:
        raise ValueError("Input checkpoint SHA256 does not match MZ-015 immutable evidence")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SOURCE_SCHEMA or payload.get("source_sha") != SOURCE_SHA:
        raise ValueError("Wrong checkpoint schema or training source commit")
    seeds = list(range(9000, 9008)) + list(range(9100, 9108))
    results = {}
    for model in ("diagonal", "gru"):
        baseline = models(torch)[model]
        baseline.load_state_dict(payload["models"][model], strict=True)
        baseline.eval()
        ref = rollout(torch, baseline, seeds, ood=True)
        if ref["clean"] != len(seeds) * 4:
            raise AssertionError(f"MZ-015 baseline unexpectedly failed: {model} {ref['clean']}")
        cases = {"float32": {"scores": {k:ref[k] for k in (
            "episodes", "success", "clean", "collisions", "paints", "raw_illegal")},
             "cpu": timing(torch, baseline)}}
        selections = (["all", "recurrent_0", "recurrent_1", "head"] if model == "diagonal"
                      else ["all", "input_recurrence", "state_recurrence", "head"])
        conditions = [(2, 0.5, sel) for sel in selections]
        conditions += [(2, 1.0, "all"), (4, 0.0, "all"), (8, 0.0, "all")]
        for bits, threshold, selection in conditions:
            specimen, size = convert(baseline, model, bits=bits, threshold=threshold,
                                     selection=selection)
            specimen.eval()
            trial = rollout(torch, specimen, seeds, ood=True)
            key = f"{bits}bit-t{threshold:g}-{selection}"
            cases[key] = {"scores": {k: trial[k] for k in (
                "episodes", "success", "clean", "collisions", "paints", "raw_illegal")},
                "quantization": size, "cpu": timing(torch, specimen)}
        results[model] = cases
    report = {
        "schema": SCHEMA,
        "script_source_sha": os.environ.get("GITHUB_SHA", "LOCAL-UNVERIFIED"),
        "source_training_sha": SOURCE_SHA,
        "source_checkpoint_sha256": source_sha256,
        "input_verified": source_sha256 == EXPECTED_SOURCE_SHA256,
        "evaluation_seeds": seeds,
        "episode_definition": "16 OOD map seeds x 4 hidden cue/inversion settings, 64 rollouts per case",
        "parameter_updates": 0,
        "research_question": "Does a source-pinned MZ-015 recurrent game policy tolerate rowwise ternary/int4/int8 Linear/GRU weight substitution?",
        "models": results,
        "peak_process_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
        "limitations": [
            "No training or quantization-aware finetuning; this is a frozen model surgery screen",
            "Same MZ-015 turning-map grammar; these seeds are unseen, not independent topology families",
            "Quantized tensors are immediately dequantized into float32 for PyTorch inference",
            "Packed byte estimates are theoretical lower bounds, not actual memory or speed gains",
            "No special low-bit CPU kernels, physical hardware acceptance or CyberSand integration",
            "Teacher data and prior model weights originated in a synthetic host, not the game",
        ],
    }
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# MZ-016 deterministic frozen-policy low-bit audit", "", f"MZ-015 checkpoint SHA-256: `{source_sha256}`", "",
             "| Architecture | Treatment | Clean/64 | Collisions | Paints | Raw illegal |",
             "|---|---|---:|---:|---:|---:|"]
    for model, treatments in results.items():
        for treatment, result in treatments.items():
            score = result["scores"]
            lines.append(f"| {model} | {treatment} | {score['clean']}/64 | {score['collisions']} | {score['paints']} | {score['raw_illegal']} |")
    lines += ["", "No quantized checkpoint is generated: source weights plus this deterministic transform reproduce each specimen.",
              "No runtime speed/footprint gain claimed. Research only; Draft/HOLD."]
    (target / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Original MZ-015 models.pt from its immutable Actions artifact")
    parser.add_argument("--out", default="mz016-evidence")
    parser.add_argument("--allow-unverified", action="store_true", help="Explicit audit-only escape hatch; marks results unverified")
    main(parser.parse_args())
