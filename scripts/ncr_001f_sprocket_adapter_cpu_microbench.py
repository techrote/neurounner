#!/usr/bin/env python3
"""NCR-001F: measured CPU adapter-only forward microbench, NEVER model-host parity.

The frozen LM is not loaded. Pinned genuine NCA and gated MLP PyTorch modules
process deterministic synthetic tap activations, not any held-out fixture.
The result is not a #72 conformance or #73 matched-host compute budget PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time
import traceback

import torch

import ncr_001f_sprocket_mac_proxy as pins
from ncr_001_core import PersistentDepthNCA
from ncr_001f_controls import GatedMLPControl

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261029
STATUS = "ADAPTER_ONLY_CPU_FORWARD_MICROBENCH__FROZEN_HOST_NOT_EXECUTED__NO_TRAIN"
FAIL_STATUS = "FAILED_ADAPTER_ONLY_CPU_FORWARD_MICROBENCH__NO_TRAIN"
LENGTHS = (1, 16, 64)
SOURCE_PATHS = (
    "scripts/ncr_001f_sprocket_adapter_cpu_microbench.py",
    "scripts/ncr_001f_sprocket_mac_proxy.py",
    "scripts/ncr_001_core.py",
    "scripts/ncr_001f_controls.py",
    "experiments/ncr-001/protocol.json",
    "experiments/ncr-001/source-pins.json",
)


def refuse(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("NCR001F_CPU_MICROBENCH_REFUSED: " + message)


def validate_budget(lengths: tuple[int, ...], warmups: int, repeats: int) -> None:
    refuse(type(lengths) is tuple and len(lengths) > 0
           and all(type(n) is int and n in LENGTHS for n in lengths)
           and len(set(lengths)) == len(lengths),
           "only declared synthetic token lengths 1/16/64 are permitted")
    refuse(type(warmups) is int and 1 <= warmups <= 5
           and type(repeats) is int and 2 <= repeats <= 10,
           "bounded warmup/sample budget invalid")


def source_sha256s() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


def make_modules() -> tuple[PersistentDepthNCA, GatedMLPControl]:
    torch.manual_seed(SEED)
    nca = PersistentDepthNCA(1280, grid_size=8, channels=4,
                             tap_count=3, steps_per_tap=3,
                             observation_gain=0.2, update_gain=0.25)
    mlp = GatedMLPControl(1280, bottleneck=256, tap_count=3)
    proxy = pins.mac_proxy(1280, taps=3, steps=3)
    refuse(nca.trainable_parameters() == proxy["nca_trainable_parameter_formula"]
           and mlp.trainable_parameters() == proxy["mlp_trainable_parameter_formula"],
           "actual module trainable parameter counts contradict pinned geometry")
    refuse(nca.config()["steps_per_tap"] == 3
           and mlp.config()["bottleneck"] == 256
           and nca.config()["tap_count"] == mlp.config()["tap_count"] == 3,
           "modules no longer implement the preregistered three-tap geometry")
    nca.eval()
    mlp.eval()
    return nca, mlp


def synthetic_hidden(length: int) -> tuple[torch.Tensor, ...]:
    refuse(type(length) is int and length in LENGTHS,
           "unregistered synthetic sequence length")
    generator = torch.Generator(device="cpu").manual_seed(SEED + length)
    return tuple(torch.randn(1, length, 1280, generator=generator)
                 for _ in range(3))


def three_taps(kind: str, module, depths: tuple[torch.Tensor, ...], *,
               validate: bool = True) -> torch.Tensor:
    refuse(kind in ("nca", "mlp") and len(depths) == 3,
           "invalid adapter identity or missing depth activation")
    state = None
    with torch.inference_mode():
        for tap, hidden in enumerate(depths):
            if kind == "nca":
                output, state = module.step(hidden, state, tap_index=tap)
            else:
                output = module.step(hidden, tap_index=tap)
            # Both modules have zero-initialized readouts. Any nonidentity
            # indicates corrupted initialization or a changed implementation.
            if validate:
                refuse(torch.equal(output, hidden),
                       "zero-readout no-op parity failed at depth tap")
        if kind == "nca" and validate:
            refuse(state is not None and state.next_tap == 3
                   and bool(torch.isfinite(state.field).all()),
                   "cell field nonfinite or depth state did not reach tap 3")
    return output


def measured_case(nca, mlp, length: int, warmups: int, repeats: int) -> dict:
    depths = synthetic_hidden(length)
    modules = {"nca": nca, "mlp": mlp}
    # Alternating order is essential to avoid assigning all first-run noise
    # to one adapter. Never interpret these kernel times as model throughput.
    for _ in range(warmups):
        for kind in ("nca", "mlp"):
            three_taps(kind, modules[kind], depths)
    times = {"nca": [], "mlp": []}
    for i in range(repeats):
        order = ("nca", "mlp") if i % 2 == 0 else ("mlp", "nca")
        for kind in order:
            start = time.perf_counter_ns()
            # Timed work is module forwards only, not torch.equal/isfinite assertions.
            three_taps(kind, modules[kind], depths, validate=False)
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            refuse(elapsed_ms > 0, "nonpositive monotonic CPU measurement")
            times[kind].append(round(elapsed_ms, 6))
    # Fail closed after timing, too. Checks remain mandatory but outside samples.
    for kind in ("nca", "mlp"):
        three_taps(kind, modules[kind], depths)
    med = {k: statistics.median(t) for k, t in times.items()}
    return {
        "synthetic_shape": [1, length, 1280],
        "tap_count": 3, "nca_steps_per_tap": 3,
        "warmup_rounds_per_adapter": warmups,
        "measured_rounds_per_adapter": repeats,
        "raw_forward_three_tap_ms": times,
        "median_forward_three_tap_ms": med,
        "median_nca_over_mlp_wall_ratio": round(med["nca"] / med["mlp"], 6),
        "includes_frozen_transformer": False,
        "includes_backward_or_optimizer": False,
        "timed_interval_includes_noop_or_finite_checks": False,
        "warmup_and_postflight_noop_state_checks": True,
    }


def base_receipt() -> dict:
    return {
        "experiment": "NCR-001F", "issue": 73,
        "status": "STARTED_ADAPTER_ONLY_CPU_MICROBENCH",
        "source_variants_verified_not_executed": {},
        "frozen_model_weights_downloaded": False,
        "frozen_model_inference_executed": False,
        "tokenizers_executed": False,
        "heldouts_opened": False,
        "optimizer_steps": 0, "training_performed": False,
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "actual_matched_frozen_host_budget_pass": False,
        "adapter_only_proxy_not_model_latency": True,
        "seed": SEED,
        "runner": {
            "github_sha": os.environ.get("GITHUB_SHA", "local"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
            "github_job": os.environ.get("GITHUB_JOB", "local"),
            "runner_os": os.environ.get("RUNNER_OS", platform.system()),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "cpu_model": next(
                (s.split(":", 1)[1].strip()
                 for s in Path("/proc/cpuinfo").read_text().splitlines()
                 if s.startswith("model name")), "unknown"
            ) if Path("/proc/cpuinfo").exists() else "unknown",
        },
    }


def run(lengths: tuple[int, ...] = LENGTHS,
        warmups: int = 2, repeats: int = 5) -> dict:
    validate_budget(lengths, warmups, repeats)
    checked = pins.audit()  # Verifies source Git blobs, 4-lane protocol and weights.
    refuse(checked["train_authorized"] is False
           and checked["full_72_conformance_pass"] is False
           and set(checked["lanes"]) ==
               {"sprocket_chat", "sprocket_instruct"},
           "source or four-lane scientific admission contract changed")
    receipt = base_receipt()
    receipt["source_git_blobs"] = checked["source_git_blobs"]
    receipt["source_sha256"] = source_sha256s()
    receipt["source_variants_verified_not_executed"] = {
        lane: {
            "source_revision": info["source_revision"],
            "weight_path": info["weight_path"],
            "weight_sha256": info["weight_sha256"],
        } for lane, info in checked["lanes"].items()
    }
    nca, mlp = make_modules()
    receipt["model_free_shared_geometry"] = pins.mac_proxy(1280)
    receipt["synthetic_cases"] = [
        measured_case(nca, mlp, length, warmups, repeats)
        for length in lengths
    ]
    receipt["peak_process_rss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss
    receipt["status"] = STATUS
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = base_receipt()
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    try:
        torch.set_num_threads(4)
        report = run()
    except BaseException as error:
        report["status"] = FAIL_STATUS
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()[-6000:]
        raise
    finally:
        report["peak_process_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
        args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
