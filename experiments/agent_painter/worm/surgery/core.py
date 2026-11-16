"""Deterministic LoRA-only mutation toolkit for WORM-001 SPROCKET neural surgery.

All mutations operate on private in-memory copies of the W2 LoRA adapter, never on
frozen base parameters, original release files or the training checkpoint. This
is an intentionally *destructive* research apparatus, not production code.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class Arm:
    name: str
    method: str
    fraction: float
    seed: int
    secondary_fraction: float = 0.0


TARGETED = (
    Arm("control", "identity", 0, 612),
    Arm("attributed-zero10", "targeted-zero", 0.10, 612),
    Arm("attributed-zero10-plus-crowbar50", "targeted-hybrid", 0.10, 612, 0.50),
    Arm("attributed-extreme50", "targeted-extreme", 0.50, 612),
    Arm("attributed-mid25", "targeted-mid", 0.25, 612),
)
CHAOS = (
    Arm("control", "identity", 0, 612),
    Arm("global-adapter-extreme50", "random-extreme", 0.50, 613),
    Arm("global-adapter-mid25", "random-mid", 0.25, 617),
    Arm("global-adapter-random25", "random-resample", 0.25, 619),
    Arm("global-adapter-zero50", "random-zero", 0.50, 621),
)


def lora_parameters(model):
    import torch
    parameters = [(name, param) for name, param in model.named_parameters()
                  if (".lora_A." in name or ".lora_B." in name) and name.endswith(".weight")]
    if not parameters or any(param.dtype != torch.float32 or param.device.type != "cpu" for _, param in parameters):
        raise ValueError("Expected exclusively finite CPU FP32 LoRA weights")
    if any(not torch.isfinite(p).all().item() for _, p in parameters):
        raise ValueError("Original LoRA adapter already nonfinite")
    return sorted(parameters, key=lambda pair: pair[0])


def flatten(parameters):
    import torch
    return torch.cat([param.detach().reshape(-1).clone().to(device="cpu") for _, param in parameters])


def apply_flat(parameters, values):
    import torch
    if values.dtype != torch.float32 or not values.is_contiguous():
        raise ValueError("Unexpected surgery weight format")
    expected = sum(param.numel() for _, param in parameters)
    if values.numel() != expected:
        raise ValueError("Surgery cannot change tensor inventory")
    with torch.no_grad():
        offset = 0
        for _, param in parameters:
            n = param.numel()
            param.copy_(values[offset:offset+n].reshape(param.shape))
            offset += n


def finite_tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def selected_indices(scores, fraction):
    """Right-promoting-only top fraction, deterministic ties on CPU."""
    import torch
    if not 0 <= fraction <= 1:
        raise ValueError("Invalid selection fraction")
    if scores is None or scores.ndim != 1:
        raise ValueError("Missing one-dimensional RIGHT attribution")
    finite = torch.isfinite(scores) & (scores > 0)
    count = int(finite.sum().item())
    if count == 0 and fraction:
        raise ValueError("No positive RIGHT-promoting LoRA attributions")
    keep = min(count, max(1, round(count * fraction))) if fraction else 0
    if keep == 0:
        return torch.empty(0, dtype=torch.long)
    ranked = torch.argsort(scores, descending=True, stable=True)
    return ranked[:keep]


def random_indices(n, fraction, seed):
    import torch
    if not 0 <= fraction <= 1:
        raise ValueError("Invalid corruption fraction")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(n, generator=generator)[:round(n*fraction)]


def tensor_midpoints(parameters, original):
    """Per-tensor midpoint; avoids overflow in min+max when range is wide."""
    import torch
    values = torch.empty_like(original)
    offset = 0
    for _, param in parameters:
        n = param.numel()
        view = original[offset:offset+n]
        midpoint = float((view.min() / 2 + view.max() / 2).item())
        values[offset:offset+n] = midpoint
        offset += n
    return values


def empirical_random_values(parameters, original, seed):
    """Random values from tensor-local existing min/max, not fake precision or global noise."""
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x54A4)
    values = torch.empty_like(original)
    offset = 0
    for _, param in parameters:
        n = param.numel()
        reference = original[offset:offset+n]
        lo, hi = reference.min(), reference.max()
        values[offset:offset+n] = lo + (hi-lo) * torch.rand(n, generator=generator)
        offset += n
    return values


def mutate(original, parameters, arm, scores=None):
    """Return mutated weights and exact mutation receipt; does not edit arguments."""
    import torch
    if original.ndim != 1 or original.numel() != sum(p.numel() for _, p in parameters):
        raise ValueError("Cannot operate on an incomplete adapter")
    if not torch.isfinite(original).all().item():
        raise ValueError("Original weight inventory nonfinite")
    out = original.clone()
    finfo_min = torch.finfo(torch.float32).min  # deliberately absurd but finite
    if arm.method.startswith("targeted"):
        ids = selected_indices(scores, arm.fraction)
    elif arm.method.startswith("random"):
        ids = random_indices(original.numel(), arm.fraction, arm.seed)
    else:
        ids = torch.empty(0, dtype=torch.long)
    extreme_ids = torch.empty(0, dtype=torch.long)
    if arm.method in ("targeted-zero", "random-zero"):
        out[ids] = 0
    elif arm.method in ("targeted-extreme", "random-extreme"):
        out[ids] = finfo_min
        extreme_ids = ids
    elif arm.method in ("targeted-mid", "random-mid"):
        mid = tensor_midpoints(parameters, original)
        out[ids] = mid[ids]
    elif arm.method == "random-resample":
        replacement = empirical_random_values(parameters, original, arm.seed)
        out[ids] = replacement[ids]
    elif arm.method == "targeted-hybrid":
        # Two disjoint damages: top 10% RIGHT-attributed zeros,
        # then the next 50% RIGHT-attributed weights receive fp32 minimum.
        ordered = selected_indices(scores, 1.0)
        n = len(ordered)
        upto = min(n, len(ids) + round(n * arm.secondary_fraction))
        extreme_ids = ordered[len(ids):upto]
        out[ids] = 0
        out[extreme_ids] = finfo_min
    elif arm.method != "identity":
        raise ValueError("Unknown, unreviewed destructive intervention")
    if not torch.isfinite(out).all().item():
        raise ValueError("Mutation left nonfinite scalar weights; abort")
    changed = int((out != original).sum().item())
    return out, {
        "name": arm.name, "method": arm.method, "seed": arm.seed,
        "selected_count": int(len(ids)), "extreme_count": int(len(extreme_ids)),
        "changed_count": changed, "total_lora_weights": original.numel(),
        "selected_fraction_of_all_lora": len(ids) / original.numel(),
        "modified_fraction_of_all_lora": changed / original.numel(),
        "minimum_finite_fp32": finfo_min if len(extreme_ids) else None,
        "original_sha256": finite_tensor_sha256(original),
        "mutated_sha256": finite_tensor_sha256(out),
        "frozen_base_mutated": False, "mutation_device": "CPU",
    }
