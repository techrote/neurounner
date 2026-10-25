#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    for path in files:
        if path.name == "runner001-summary.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lane = data["machine"]["lane"]
        rows.append({
            "lane": lane,
            "runner_os": data["machine"].get("runner_os"),
            "runner_arch": data["machine"].get("runner_arch"),
            "cpu": data["machine"].get("cpu_model"),
            "logical_cpus": data["machine"].get("logical_cpus"),
            "memory_gib": data["machine"].get("memory_bytes", 0) / 1024**3,
            "json_mib_s": get(data, "benchmarks.json_roundtrip.mib_per_second"),
            "sha_mib_s": get(data, "benchmarks.sha256.mib_per_second"),
            "zlib_mib_s": get(data, "benchmarks.zlib.mib_per_second"),
            "torch_gflops": get(data, "benchmarks.torch_matmul.estimated_gflops"),
            "model_load_s": get(data, "benchmarks.local_model.load.seconds"),
            "generation_tok_s": get(data, "benchmarks.local_model.generation.tokens_per_second"),
            "lora_tok_s": get(data, "benchmarks.local_model.lora_training.sequence_tokens_per_second"),
            "github_models_status": get(data, "benchmarks.github_models.status"),
            "github_models_median_s": get(data, "benchmarks.github_models.median_seconds"),
            "wall_s": data.get("wall_seconds"),
            "source_file": str(path),
        })

    if not rows:
        raise SystemExit("no RUNNER-001 result JSON files found")

    baseline = next((r for r in rows if r["lane"] == "ubuntu-24.04-x64"), rows[0])
    for row in rows:
        row["relative_to_ubuntu"] = {
            "generation": ratio(row["generation_tok_s"], baseline["generation_tok_s"]),
            "lora": ratio(row["lora_tok_s"], baseline["lora_tok_s"]),
            "torch": ratio(row["torch_gflops"], baseline["torch_gflops"]),
            "json": ratio(row["json_mib_s"], baseline["json_mib_s"]),
            "sha": ratio(row["sha_mib_s"], baseline["sha_mib_s"]),
            "zlib": ratio(row["zlib_mib_s"], baseline["zlib_mib_s"]),
        }

    out = {
        "schema": "neurounner.runner001.summary.v1",
        "baseline_lane": baseline["lane"],
        "rows": rows,
        "interpretation": [
            "Generation and LoRA measurements are local CPU work and are the primary Neurounner platform comparisons.",
            "GitHub Models inference is remote; its latency is a runner/network/client-path measurement, not local model compute throughput.",
            "macos-15 is Apple Silicon arm64 with a different standard-runner CPU count from the x64 Linux/Windows lanes, so it is not an OS-only causal comparison.",
            "macos-15-intel is included as a partial architecture control.",
        ],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RUNNER-001 cross-platform results",
        "",
        f"Ubuntu baseline: `{baseline['lane']}`.",
        "",
        "| Lane | Arch | CPUs | Local gen tok/s | vs Ubuntu | LoRA seq tok/s | vs Ubuntu | Torch GFLOPS | GitHub Models median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        rel = r["relative_to_ubuntu"]
        lines.append(
            f"| {r['lane']} | {r['runner_arch']} | {r['logical_cpus']} | "
            f"{fmt(r['generation_tok_s'])} | {fmt(rel['generation'])}× | "
            f"{fmt(r['lora_tok_s'])} | {fmt(rel['lora'])}× | "
            f"{fmt(r['torch_gflops'])} | "
            f"{fmt(r['github_models_median_s'], 3) if r['github_models_status'] == 'success' else r['github_models_status']} |"
        )
    lines += [
        "",
        "## CPU / runner identity",
        "",
        "| Lane | CPU | RAM GiB | JSON MiB/s | SHA-256 MiB/s | zlib MiB/s | Model load s | Total wall s |",
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
        "- GitHub Models is remote inference. Runner-to-runner differences there mostly reflect request setup/network/service variance.",
        "- `macos-15` is an Apple-Silicon runner, so comparing it directly with x64 Linux/Windows confounds OS, ISA, CPU implementation, and CPU count.",
        "- `macos-15-intel` is retained as a partial x64 control rather than pretending the primary three lanes are hardware-identical.",
        "",
    ]
    args.md_out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
