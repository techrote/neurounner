#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

LANE_ORDER = [
    "ubuntu-24.04-x64",
    "windows-2025-x64",
    "macos-15-arm64",
]
EXPECTED_LANES = set(LANE_ORDER)
EXPECTED_REPLICATES = {1, 2, 3}
METRICS = [
    "generation_tok_s",
    "lora_tok_s",
    "torch_gflops",
    "json_mib_s",
    "sha_mib_s",
    "zlib_mib_s",
    "model_load_s",
    "wall_s",
]


def get(d: dict[str, Any], path: str):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def ratio(value, baseline):
    if value is None or baseline in (None, 0):
        return None
    return value / baseline


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def metric_stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"median": None, "min": None, "max": None}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def fmt_range(stats):
    if stats["median"] is None:
        return "n/a"
    return f"{stats['median']:.2f} [{stats['min']:.2f}–{stats['max']:.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--md-out", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.input_dir.rglob("runner001-*.json"))
    rows = []
    canonical_hashes = set()

    for path in files:
        if path.name == "runner001-summary.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lane = data["machine"]["lane"]
        replicate = int(data["machine"]["replicate"])
        canonical_hashes.add(data["source"]["dataset_canonical_sha256"])
        rows.append({
            "lane": lane,
            "replicate": replicate,
            "runner_os": data["machine"].get("runner_os"),
            "runner_arch": data["machine"].get("runner_arch"),
            "cpu": data["machine"].get("cpu_model"),
            "logical_cpus": data["machine"].get("logical_cpus"),
            "memory_gib": data["machine"].get("memory_bytes", 0) / 1024**3,
            "python": data["machine"].get("python"),
            "torch": get(data, "benchmarks.local_model.versions.torch"),
            "dataset_raw_sha256": data["source"].get("dataset_raw_sha256"),
            "dataset_canonical_sha256": data["source"].get("dataset_canonical_sha256"),
            "json_mib_s": get(data, "benchmarks.json_roundtrip.mib_per_second"),
            "sha_mib_s": get(data, "benchmarks.sha256.mib_per_second"),
            "zlib_mib_s": get(data, "benchmarks.zlib.mib_per_second"),
            "torch_gflops": get(data, "benchmarks.torch_matmul.estimated_gflops"),
            "model_load_s": get(data, "benchmarks.local_model.load.seconds"),
            "generation_tok_s": get(data, "benchmarks.local_model.generation.median_tokens_per_second"),
            "lora_tok_s": get(data, "benchmarks.local_model.lora_training.sequence_tokens_per_second"),
            "wall_s": data.get("wall_seconds"),
            "source_file": str(path),
        })

    if not rows:
        raise SystemExit("no RUNNER-001 result JSON files found")

    actual_lanes = {r["lane"] for r in rows}
    if actual_lanes != EXPECTED_LANES:
        raise SystemExit(
            f"RUNNER-001 expected lanes {sorted(EXPECTED_LANES)}, got {sorted(actual_lanes)}"
        )
    if len(canonical_hashes) != 1:
        raise SystemExit(
            f"RUNNER-001 corpus semantic identity mismatch: {sorted(canonical_hashes)}"
        )

    aggregates = []
    for lane in LANE_ORDER:
        lane_rows = [r for r in rows if r["lane"] == lane]
        reps = {r["replicate"] for r in lane_rows}
        if reps != EXPECTED_REPLICATES or len(lane_rows) != len(EXPECTED_REPLICATES):
            raise SystemExit(
                f"RUNNER-001 {lane} expected replicas {sorted(EXPECTED_REPLICATES)}, "
                f"got {sorted(reps)} across {len(lane_rows)} rows"
            )

        aggregate = {
            "lane": lane,
            "replicates": len(lane_rows),
            "runner_arches": sorted({str(r["runner_arch"]) for r in lane_rows}),
            "logical_cpus": sorted({int(r["logical_cpus"]) for r in lane_rows}),
            "cpu_models": sorted({str(r["cpu"]) for r in lane_rows}),
            "memory_gib": metric_stats([r["memory_gib"] for r in lane_rows]),
            "metrics": {
                metric: metric_stats([r[metric] for r in lane_rows])
                for metric in METRICS
            },
        }
        aggregates.append(aggregate)

    baseline = next(a for a in aggregates if a["lane"] == "ubuntu-24.04-x64")
    for aggregate in aggregates:
        aggregate["relative_to_ubuntu"] = {
            "generation": ratio(
                aggregate["metrics"]["generation_tok_s"]["median"],
                baseline["metrics"]["generation_tok_s"]["median"],
            ),
            "lora": ratio(
                aggregate["metrics"]["lora_tok_s"]["median"],
                baseline["metrics"]["lora_tok_s"]["median"],
            ),
            "torch": ratio(
                aggregate["metrics"]["torch_gflops"]["median"],
                baseline["metrics"]["torch_gflops"]["median"],
            ),
        }

    rows.sort(key=lambda r: (LANE_ORDER.index(r["lane"]), r["replicate"]))
    out = {
        "schema": "neurounner.runner001.summary.v3",
        "baseline_lane": baseline["lane"],
        "dataset_canonical_sha256": next(iter(canonical_hashes)),
        "replicates_per_lane": len(EXPECTED_REPLICATES),
        "aggregates": aggregates,
        "rows": rows,
        "github_models": {
            "status": "retired",
            "retired_date": "2026-07-30",
            "docs": "https://docs.github.com/en/github-models",
        },
        "interpretation": [
            "Three fresh hosted VMs are measured per runner family; medians are primary and min/max show host-allocation variance.",
            "Within each VM, generation uses a full-shape warmup and the median of five deterministic 48-token generations.",
            "Generation and LoRA measurements are local CPU work and are the primary Neurounner platform comparisons.",
            "Model-load and total-wall measurements include network/package/cache effects and should not be treated as pure CPU metrics.",
            "GitHub Models cannot supply a current remote-control benchmark because GitHub fully retired the service on 2026-07-30.",
            "macos-15 is Apple Silicon arm64 with three CPUs versus four x64 CPUs on Linux and Windows, so this measures the real runner-family choice rather than an OS-only causal effect.",
        ],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RUNNER-001 cross-platform results",
        "",
        f"Three fresh VMs per platform. Ubuntu is the ratio baseline. Canonical corpus SHA-256: `{next(iter(canonical_hashes))}`.",
        "",
        "| Lane | Gen tok/s median [range] | vs Ubuntu | LoRA seq tok/s median [range] | vs Ubuntu | Torch GFLOPS median [range] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in aggregates:
        rel = a["relative_to_ubuntu"]
        lines.append(
            f"| {a['lane']} | {fmt_range(a['metrics']['generation_tok_s'])} | "
            f"{fmt(rel['generation'])}× | {fmt_range(a['metrics']['lora_tok_s'])} | "
            f"{fmt(rel['lora'])}× | {fmt_range(a['metrics']['torch_gflops'])} |"
        )

    lines += [
        "",
        "## Supporting medians",
        "",
        "| Lane | JSON MiB/s | SHA-256 MiB/s | zlib MiB/s | Model load s | Benchmark wall s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in aggregates:
        m = a["metrics"]
        lines.append(
            f"| {a['lane']} | {fmt(m['json_mib_s']['median'])} | "
            f"{fmt(m['sha_mib_s']['median'])} | {fmt(m['zlib_mib_s']['median'])} | "
            f"{fmt(m['model_load_s']['median'])} | {fmt(m['wall_s']['median'])} |"
        )

    lines += [
        "",
        "## Fresh-VM evidence",
        "",
        "| Lane | Rep | CPU | Gen tok/s | LoRA tok/s | Torch GFLOPS | Load s | Wall s |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['lane']} | {r['replicate']} | {r['cpu']} | "
            f"{fmt(r['generation_tok_s'])} | {fmt(r['lora_tok_s'])} | "
            f"{fmt(r['torch_gflops'])} | {fmt(r['model_load_s'])} | {fmt(r['wall_s'])} |"
        )

    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Primary conclusions use three fresh-VM medians, not a single lucky or unlucky host allocation.",
        "- Within-VM generation uses a same-shape warmup and five timed deterministic generations; the per-VM median is aggregated again across VMs.",
        "- Local generation and LoRA timings are the main Neurounner metrics; generic JSON/hash/compression/matmul numbers are diagnostic.",
        "- GitHub Models is not benchmarked: GitHub fully retired the playground, catalog and inference API on 2026-07-30.",
        "- `macos-15` uses Apple Silicon and fewer CPUs than the x64 Linux/Windows lanes, so this is a practical runner-family comparison rather than a pure OS causal experiment.",
        "- The Intel macOS pilot exposed a separate compatibility cost: the same current PyTorch stack was unavailable there, so it is not mixed into the performance table.",
        "",
    ]
    args.md_out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
