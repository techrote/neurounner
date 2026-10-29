#!/usr/bin/env python3
"""NCR-001B CPU synthetic-only core smoke. Does not load or train LMs."""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch

from ncr_001_core import PersistentDepthNCA

ROOT = Path(__file__).resolve().parents[1]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runtime/ncr-001/core-smoke.json")
    parser.add_argument("--sequence-length", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.sequence_length <= 128:
        parser.error("Only 1..128 synthetic positions are allowed")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.manual_seed(20261026)
    config = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    assert config["train_authorized"] is False
    widths = {name: data["expected_config"]["hidden_size"]
              for name, data in config["models"].items()}
    out = {
        "experiment": "NCR-001B",
        "classification": "synthetic_cpu_pretraining_probe_not_lm_conformance",
        "training_performed": False,
        "model_weights_loaded": False,
        "protocol_training_authorized": False,
        "torch_version": torch.__version__,
        "runner": {"cpu_model": cpu_model(),
                   "threads": torch.get_num_threads(),
                   "platform": platform.platform()},
        "field": {"height": 8, "width": 8, "channels": 4,
                  "bytes_per_token_fp32": 8 * 8 * 4 * 4},
        "lanes": {},
    }
    for name, width in widths.items():
        core = PersistentDepthNCA(width).eval()
        x = torch.randn(1, args.sequence_length, width)
        with torch.no_grad():
            s = None
            # Warm up the kernels and verify zero-initialized identity.
            for tap in range(3):
                y, s = core.step(x, s, tap_index=tap)
            if not torch.equal(x, y):
                raise AssertionError(f"source no-op failed: {name}")
            durations = []
            for _ in range(3):
                state = None
                start = time.monotonic()
                for tap in range(3):
                    y, state = core.step(x, state, tap_index=tap)
                durations.append(time.monotonic() - start)
                assert torch.equal(x, y)
                assert state is not None and state.next_tap == 3
                assert bool(torch.isfinite(state.field).all())
        out["lanes"][name] = {
            "hidden_size": width,
            "trainable_parameters": core.trainable_parameters(),
            "sequence_positions": args.sequence_length,
            "field_bytes_per_batch": args.sequence_length * 1024,
            "three_taps_three_steps_wall_seconds": durations,
            "no_op_exact_identity": True,
            "state_shape": list(state.field.shape),
            "note": "Wall seconds represent only synthetic NCA kernels (no LLM).",
        }
        print(f"{name}: {width=} trainable={core.trainable_parameters()} "
              f"synthetic_three_taps_seconds={[round(v, 4) for v in durations]}",
              flush=True)
    out["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("NCR-001B synthetic-only probe PASS; no model weights or training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
