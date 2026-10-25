#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_LANES = {
    "ubuntu-24.04-x64",
    "windows-2025-x64",
    "macos-15-arm64",
}


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
        canonical_hashes.add(data["source"]["dataset_canonical_sha256"])
        rows.append({
            "lane": lane,
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
            "generation_tok_s": get(data, "benchmarks.local_model.generation.tokens_per_second"),
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

    baseline = next(r for r in rows if r["lane"] == "ubuntu-24.04-x64")
    for row in rows:
        row["relative_to_ubuntu"] = {
            "generation": ratio(row["generation_tok_s"], baseline["generation_tok_s"]),
            "lora": ratio(row["lora_tok_s"], baseline["lora_tok_s"]),
            "torch": ratio(row["torch_gflops"], baseline["torch_gflops"]),
            "json": ratio(row["json_mib_s"], baseline["json_mib_s"]),
            "sha": ratio(row["sha_mib_s"], baseline["sha_mib_s"]),
            "zlib": ratio(row["zlib_mib_s"], baseline["zlib_mib_s"]),
        }

    rows.sort(key=lambda r: ["ubuntu-24.04-x64", "windows-2025-x64", "macos-15-arm64"].index(r["lane"]))
    out = {
        "schema": "neurounner.runner001.summary.v2",
        "baseline_lane": baseline["lane"],
        "dataset_canonical_sha256": next(iter(canonical_hashes)),
        "rows": rows,
        "github_models": {
            "status": "retired",
            "retired_date": "2026-07-30",
            "docs": "https://docs.github.com/en/github-models",
        },
        "interpretation": [
            "Generation and LoRA measurements are local CPU work and are the primary Neurounner platform comparisons.",
            "GitHub Models cannot supply a current remote-control benchmark because GitHub fully retired the service on 2026-07-30.",
            "macos-15 is Apple Silicon arm64 with three CPUs versus four x64 CPUs on the Linux and Windows standard public-repository runners, so this is not an OS-only causal comparison.",
            "A macos-15-intel pilot was excluded from the comparable matrix because the image did not provide the same modern PyTorch wheel as the other lanes.",
        ],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RUNNER-001 cross-platform results",
        "",
        f"Ubuntu baseline: `{baseline['lane']}`. Canonical corpus SHA-256: `{next(iter(canonical_hashes))}`.",
        "",
        "| Lane | Arch | CPUs | Local gen tok/s | vs Ubuntu | LoRA seq tok/s | vs Ubuntu | Torch GFLOPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        rel = r["relative_to_ubuntu"]
        lines.append(
            f"| {r['lane']} | {r['runner_arch']} | {r['logical_cpus']} | "
            f"{fmt(r['generation_tok_s'])} | {fmt(rel['generation'])}× | "
            f"{fmt(r['lora_tok_s'])} | {fmt(rel['lora'])}× | "
            f"{fmt(r['torch_gflops'])} |"
        )
    lines += [
        "",
        "## CPU / runner identity",
        "",
        "| Lane | CPU | RAM GiB | JSON MiB/s | SHA-256 MiB/s | zlib MiB/s | Model load s | Benchmark wall s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['lane']} | {r['cpu']} | {fmt(r['memory_gib'])} | "
            f"{fmt(r['json_mib_s'])} | {fmt(r['sha_mib_s'])} | {fmt(r['zlib_mib_s'])} | "
            f"{fmt(r['model_load_s'])} | {fmt(r['wall_s'])} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Local generation and LoRA timings are the primary platform-sensitive Neurounner measurements.",
        "- GitHub Models is not benchmarked: GitHub fully retired the playground, catalog and inference API on 2026-07-30.",
        "- `macos-15` uses Apple Silicon and fewer CPUs than the x64 Linux/Windows lanes, so the result measures the real hosted-runner choice rather than a pure OS causal effect.",
        "- The Intel macOS pilot exposed a separate compatibility cost: a source-identical modern PyTorch stack was unavailable there, so it is not mixed into the performance table.",
        "",
    ]
    args.md_out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
