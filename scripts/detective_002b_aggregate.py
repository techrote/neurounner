#!/usr/bin/env python3
"""Aggregate DETECTIVE-002B seven-body runtime smoke into durable evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-002/runtime-smoke.json"
SOURCE = ROOT / "experiments/detective-002/source-evidence/source-evidence.json"


class AggregateError(ValueError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_results(results_dir: Path) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    cfg = read_json(CONFIG)
    source = read_json(SOURCE)
    authority = {x["candidate_id"]: x for x in source["candidates"]}
    found: dict[str, dict] = {}
    paths = list(results_dir.rglob("result.json"))
    if len(paths) != len(cfg["cohort"]):
        raise AggregateError(
            f"expected {len(cfg['cohort'])} result files, found {len(paths)}"
        )
    for path in paths:
        row = read_json(path)
        cid = row.get("candidate_id")
        if cid in found:
            raise AggregateError("duplicate body result " + str(cid))
        if cid not in cfg["cohort"]:
            raise AggregateError("unexpected body result " + str(cid))
        found[cid] = row
    if set(found) != set(cfg["cohort"]):
        raise AggregateError("body result set differs from preregistered cohort")

    for cid in cfg["cohort"]:
        row = found[cid]
        src = authority[cid]
        if (
            row.get("schema") != "detective-002b-body-runtime-v1"
            or row.get("experiment") != "DETECTIVE-002B"
            or row.get("issue") != 130
            or row.get("source", {}).get("repo") != src["repo"]
            or row.get("source", {}).get("revision") != src["resolved_revision"]
            or row.get("source", {}).get("variant") != src["variant"]
            or row.get("source", {}).get("weight") != src["weight"]
            or row.get("optimizer_steps") != 0
            or row.get("paid_api_calls") != 0
            or row.get("relevance_mapping_accessed") is not False
            or row.get("population_expansion_admitted") is not False
        ):
            raise AggregateError(cid + ": body/source/custody record mismatch")
        if row.get("status") not in {
            "runtime_pass", "runtime_fail", "protocol_specialization_required"
        }:
            raise AggregateError(cid + ": invalid runtime classification")
        if row["status"] == "runtime_pass":
            tasks = row.get("tasks", [])
            if len(tasks) != 4:
                raise AggregateError(cid + ": runtime_pass missing four tasks")
            if row.get("total_generated_tokens", 999) > 256:
                raise AggregateError(cid + ": generation budget exceeded")
            env = row.get("environment") or {}
            if (
                env.get("torch_threads") != 4
                or env.get("cuda_available") is not False
            ):
                raise AggregateError(cid + ": four-CPU/CPU-only contract mismatch")
            for task in tasks:
                if (
                    task.get("generated_tokens", 999) > 64
                    or task.get("prompt_token_count", 999) > 512
                    or len(task.get("generated_token_ids", []))
                       != task.get("generated_tokens")
                ):
                    raise AggregateError(cid + ": task token custody mismatch")

    chat = found["sprocket-500m-chat"]["source"]["weight"]
    instruct = found["sprocket-500m-instruct"]["source"]["weight"]
    if (
        chat["path"] == instruct["path"]
        or chat["lfs_sha256"] == instruct["lfs_sha256"]
    ):
        raise AggregateError("Sprocket variants collapsed in runtime evidence")
    return cfg, authority, found


def body_summary(cid: str, row: dict, source: dict) -> dict:
    tasks = row.get("tasks", [])
    prefill = [
        float(x["prompt_tokens_per_second"])
        for x in tasks if x.get("prompt_tokens_per_second") is not None
    ]
    completion = [
        float(x["completion_tokens_per_second"])
        for x in tasks if x.get("completion_tokens_per_second") is not None
    ]
    by_id = {x["task_id"]: x for x in tasks}
    b02 = by_id.get("B02", {}).get("observations", {})
    b03 = by_id.get("B03", {}).get("observations", {})
    b04 = by_id.get("B04", {}).get("observations", {})
    return {
        "candidate_id": cid,
        "status": row["status"],
        "family": source["family"],
        "variant": source["variant"],
        "parameters": source["safetensors_parameters"],
        "license": source["license"],
        "production_license_gate": source["production_license_gate"],
        "weight_sha256": source["weight"]["lfs_sha256"],
        "weight_bytes": source["weight"]["size"],
        "model_load_seconds": row.get("environment", {}).get("model_load_seconds"),
        "rss_after_load_kib": row.get("environment", {}).get("rss_after_load_kib"),
        "peak_rss_kib": row.get("peak_rss_kib"),
        "mean_prompt_tokens_per_second": _mean(prefill),
        "mean_completion_tokens_per_second": _mean(completion),
        "total_generated_tokens": sum(x.get("generated_tokens", 0) for x in tasks),
        "truncated_tasks": sum(bool(x.get("hit_max_new_tokens")) for x in tasks),
        "repeated_sentences": sum(
            x.get("observations", {}).get("repeated_sentence_count", 0)
            for x in tasks
        ),
        "memory_time_marker": bool(b02.get("required_hits")),
        "memory_uncertainty_qualifier": bool(b02.get("qualifier_hits")),
        "physical_record_corrected_marker": bool(b03.get("required_hits")),
        "physical_record_stale_marker_also_present": bool(b03.get("stale_hits")),
        "rumor_qualifier": bool(b04.get("qualifier_hits")),
        "runtime_failure": row.get("runtime_failure"),
        "training_selection_allowed": False,
        "population_expansion_admitted": False,
    }


def markdown_report(manifest: dict) -> str:
    lines = [
        "# DETECTIVE-002B — seven-body no-training runtime/dialogue smoke",
        "",
        "This report is a runtime/behavior characterization, not an NPC ranking "
        "or training admission. DETECTIVE-001R1 #121 remains independently "
        "reviewer-blocked.",
        "",
        "| Body | Runtime | Params | Load s | RSS GiB | Peak GiB | Mean completion tok/s | Trunc. | Repeats | Memory uncertainty | Corrected record | Rumor qualifier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in manifest["bodies"]:
        def num(v, digits=2):
            return "—" if v is None else f"{v:.{digits}f}"
        rss = (
            None if row["rss_after_load_kib"] is None
            else row["rss_after_load_kib"] / 1024 / 1024
        )
        peak = (
            None if row["peak_rss_kib"] is None
            else row["peak_rss_kib"] / 1024 / 1024
        )
        lines.append(
            f"| {row['candidate_id']} | {row['status']} | "
            f"{row['parameters'] or '—'} | {num(row['model_load_seconds'])} | "
            f"{num(rss)} | {num(peak)} | "
            f"{num(row['mean_completion_tokens_per_second'])} | "
            f"{row['truncated_tasks']} | {row['repeated_sentences']} | "
            f"{'yes' if row['memory_uncertainty_qualifier'] else 'no'} | "
            f"{'yes' if row['physical_record_corrected_marker'] else 'no'} | "
            f"{'yes' if row['rumor_qualifier'] else 'no'} |"
        )
    lines += [
        "",
        "Mechanical markers above are not human quality judgments. A response "
        "can mention the corrected timestamp while still being incoherent; raw "
        "outputs are the authority.",
        "",
        "## Boundaries",
        "",
        "- optimizer steps: **0**;",
        "- paid API calls: **0**;",
        "- relevance mapping accessed: **false**;",
        "- population expansion admitted: **false**;",
        "- body training selection made: **false**.",
        "",
        "LFM2.5-350M remains research-runtime eligible but production/game "
        "distribution is still license-gated under lfm1.0. Qwen3-0.6B "
        "remains an explicit near-band exception at 751,632,384 parameters. "
        "Sprocket Chat and Instruct remain distinct checkpoint hashes.",
        "",
        "Any later body-selection decision must separately reconcile runtime "
        "evidence with DETECTIVE-001R1 relevance and C1/C2/C3 policy evidence.",
    ]
    return "\n".join(lines) + "\n"


def assemble(results_dir: Path, out: Path, source_head: str, run_id: str,
             run_url: str) -> dict:
    cfg, authority, found = load_results(results_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "raw").mkdir(parents=True)
    (out / "provenance").mkdir(parents=True)

    bodies = []
    for cid in cfg["cohort"]:
        row = found[cid]
        write_json(out / "raw" / f"{cid}.json", row)
        bodies.append(body_summary(cid, row, authority[cid]))

    status_counts: dict[str, int] = {}
    for body in bodies:
        status_counts[body["status"]] = status_counts.get(body["status"], 0) + 1

    manifest = {
        "schema": "detective-002b-runtime-archive-v1",
        "experiment": "DETECTIVE-002B",
        "issue": 130,
        "source_head": source_head,
        "workflow_run_id": str(run_id),
        "workflow_run_url": run_url,
        "config_sha256": file_sha(CONFIG),
        "source_evidence_sha256": file_sha(SOURCE),
        "cohort": cfg["cohort"],
        "status_counts": status_counts,
        "bodies": bodies,
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "relevance_mapping_accessed": False,
        "population_expansion_admitted": False,
        "body_training_selection_made": False,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "provenance" / "source-artifacts.json", {
        "source_head": source_head,
        "workflow_run_id": str(run_id),
        "workflow_run_url": run_url,
        "config_path": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": file_sha(CONFIG),
        "source_evidence_path": SOURCE.relative_to(ROOT).as_posix(),
        "source_evidence_sha256": file_sha(SOURCE),
        "stage_a_merge_commit": cfg["source_authority"]["stage_a_merge_commit"],
        "candidate_weight_sha256": {
            cid: authority[cid]["weight"]["lfs_sha256"]
            for cid in cfg["cohort"]
        },
        "external_provider_calls": 0,
        "training_steps": 0,
        "relevance_mapping_accessed": False,
    })
    (out / "REPORT.md").write_text(markdown_report(manifest), encoding="utf-8")
    inventory = []
    for path in sorted(x for x in out.rglob("*") if x.is_file()):
        rel = path.relative_to(out).as_posix()
        if rel != "provenance/hashes.sha256":
            inventory.append(f"{file_sha(path)}  {rel}")
    (out / "provenance" / "hashes.sha256").write_text(
        "\n".join(inventory) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-head", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-url", required=True)
    args = ap.parse_args()
    m = assemble(
        args.results_dir, args.out, args.source_head, args.run_id, args.run_url
    )
    print(json.dumps({
        "bodies": len(m["bodies"]),
        "status_counts": m["status_counts"],
        "optimizer_steps": 0,
        "population_expansion_admitted": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
