#!/usr/bin/env python3
"""Independent standard-library readback for DETECTIVE-002B runtime evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-002/runtime-smoke.json"
SOURCE = ROOT / "experiments/detective-002/source-evidence/source-evidence.json"


class VerifyError(ValueError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise VerifyError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repeated_sentence_count(text: str) -> int:
    parts = [
        " ".join(x.casefold().split())
        for x in re.split(r"(?:[.!?]+|\n+)", text)
        if x.strip()
    ]
    seen: set[str] = set()
    repeats = 0
    for part in parts:
        if len(part) < 4:
            continue
        if part in seen:
            repeats += 1
        else:
            seen.add(part)
    return repeats


def hits(task_cfg: dict, response: str, key: str) -> list[str]:
    lower = response.casefold()
    return [
        marker for marker in task_cfg["observations"].get(key, [])
        if marker.casefold() in lower
    ]


def recompute_observations(task_cfg: dict, response: str) -> dict:
    return {
        "required_hits": hits(task_cfg, response, "required_markers"),
        "alternative_hits": hits(task_cfg, response, "alternative_markers"),
        "qualifier_hits": hits(task_cfg, response, "qualifier_markers"),
        "stale_hits": hits(task_cfg, response, "stale_markers"),
        "forbidden_hits": hits(task_cfg, response, "forbidden_markers"),
        "starts_json_object": response.lstrip().startswith("{"),
        "control_token_literals": sorted(
            set(re.findall(r"<\|[^|>]+\|>", response))
        ),
        "repeated_sentence_count": repeated_sentence_count(response),
        "characters": len(response),
        "whitespace_words": len(response.split()),
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def recompute_body(cid: str, raw: dict, source: dict, cfg: dict) -> dict:
    tasks = raw.get("tasks", [])
    if raw["status"] == "runtime_pass":
        need(len(tasks) == 4, cid + ": runtime_pass task count")
    by_id = {x["task_id"]: x for x in tasks}
    for task in tasks:
        task_cfg = next(
            (x for x in cfg["battery"] if x["id"] == task["task_id"]), None
        )
        need(task_cfg is not None, cid + ": unknown task ID")
        need(
            task["observations"]
            == recompute_observations(task_cfg, task["response"]),
            cid + ": stored observations do not replay",
        )
        need(
            task["generated_tokens"] == len(task["generated_token_ids"]),
            cid + ": generated token count mismatch",
        )
        need(task["generated_tokens"] <= 64, cid + ": per-task decode budget")
        need(task["prompt_token_count"] <= 512, cid + ": prompt budget")
        need(
            bool(task["hit_max_new_tokens"])
            == (task["generated_tokens"] == 64),
            cid + ": truncation marker mismatch",
        )
        need(
            re.fullmatch(r"[0-9a-f]{64}", task["prompt_token_sha256"]) is not None,
            cid + ": prompt token digest missing",
        )
    prefill = [
        float(x["prompt_tokens_per_second"])
        for x in tasks if x.get("prompt_tokens_per_second") is not None
    ]
    completion = [
        float(x["completion_tokens_per_second"])
        for x in tasks if x.get("completion_tokens_per_second") is not None
    ]
    b02 = by_id.get("B02", {}).get("observations", {})
    b03 = by_id.get("B03", {}).get("observations", {})
    b04 = by_id.get("B04", {}).get("observations", {})
    return {
        "candidate_id": cid,
        "status": raw["status"],
        "family": source["family"],
        "variant": source["variant"],
        "parameters": source["safetensors_parameters"],
        "license": source["license"],
        "production_license_gate": source["production_license_gate"],
        "weight_sha256": source["weight"]["lfs_sha256"],
        "weight_bytes": source["weight"]["size"],
        "model_load_seconds": raw.get("environment", {}).get("model_load_seconds"),
        "rss_after_load_kib": raw.get("environment", {}).get("rss_after_load_kib"),
        "peak_rss_kib": raw.get("peak_rss_kib"),
        "mean_prompt_tokens_per_second": mean(prefill),
        "mean_completion_tokens_per_second": mean(completion),
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
        "runtime_failure": raw.get("runtime_failure"),
        "training_selection_allowed": False,
        "population_expansion_admitted": False,
    }


def verify(root: Path) -> dict:
    cfg = read_json(CONFIG)
    source_doc = read_json(SOURCE)
    authority = {x["candidate_id"]: x for x in source_doc["candidates"]}
    expected_files = {
        "manifest.json",
        "REPORT.md",
        "provenance/source-artifacts.json",
        "provenance/hashes.sha256",
        *{f"raw/{cid}.json" for cid in cfg["cohort"]},
    }
    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*") if p.is_file()
    }
    need(present == expected_files, "DETECTIVE-002B archive file set mismatch")
    expected_inventory = [
        f"{file_sha(root / rel)}  {rel}"
        for rel in sorted(expected_files - {"provenance/hashes.sha256"})
    ]
    recorded = (root / "provenance/hashes.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    need(recorded == expected_inventory, "archive hash inventory mismatch")

    manifest = read_json(root / "manifest.json")
    provenance = read_json(root / "provenance/source-artifacts.json")
    need(
        manifest.get("schema") == "detective-002b-runtime-archive-v1"
        and manifest.get("experiment") == "DETECTIVE-002B"
        and manifest.get("issue") == 130,
        "manifest identity mismatch",
    )
    need(
        manifest.get("config_sha256") == file_sha(CONFIG)
        and manifest.get("source_evidence_sha256") == file_sha(SOURCE),
        "manifest source/config hash mismatch",
    )
    need(
        manifest.get("cohort") == cfg["cohort"],
        "manifest cohort/order mismatch",
    )
    need(
        manifest.get("optimizer_steps") == 0
        and manifest.get("paid_api_calls") == 0
        and manifest.get("relevance_mapping_accessed") is False
        and manifest.get("population_expansion_admitted") is False
        and manifest.get("body_training_selection_made") is False,
        "manifest widened experiment authority",
    )
    need(
        re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_head", "")) is not None
        and str(manifest.get("workflow_run_id", "")).isdigit(),
        "manifest source/run identity missing",
    )

    bodies = []
    status_counts: dict[str, int] = {}
    for cid in cfg["cohort"]:
        raw = read_json(root / "raw" / f"{cid}.json")
        src = authority[cid]
        need(
            raw.get("schema") == "detective-002b-body-runtime-v1"
            and raw.get("candidate_id") == cid
            and raw.get("source", {}).get("repo") == src["repo"]
            and raw.get("source", {}).get("revision") == src["resolved_revision"]
            and raw.get("source", {}).get("weight") == src["weight"],
            cid + ": raw source identity mismatch",
        )
        need(
            raw.get("source_evidence_sha256") == file_sha(SOURCE)
            and raw.get("config_sha256") == file_sha(CONFIG),
            cid + ": raw source/config digest mismatch",
        )
        need(
            raw.get("optimizer_steps") == 0
            and raw.get("paid_api_calls") == 0
            and raw.get("relevance_mapping_accessed") is False
            and raw.get("population_expansion_admitted") is False,
            cid + ": raw authority widened",
        )
        need(
            raw["status"] in {
                "runtime_pass", "runtime_fail", "protocol_specialization_required"
            },
            cid + ": invalid status",
        )
        if raw["status"] == "runtime_pass":
            env = raw["environment"]
            need(
                env["torch_threads"] == 4 and env["cuda_available"] is False,
                cid + ": CPU/thread contract mismatch",
            )
            need(raw["total_generated_tokens"] <= 256, cid + ": total decode budget")
        body = recompute_body(cid, raw, src, cfg)
        bodies.append(body)
        status_counts[body["status"]] = status_counts.get(body["status"], 0) + 1

    need(manifest["bodies"] == bodies, "manifest body table does not replay")
    need(
        manifest["status_counts"] == status_counts,
        "manifest status counts do not replay",
    )

    chat = next(x for x in bodies if x["candidate_id"] == "sprocket-500m-chat")
    instruct = next(
        x for x in bodies if x["candidate_id"] == "sprocket-500m-instruct"
    )
    need(
        chat["weight_sha256"] != instruct["weight_sha256"],
        "Sprocket runtime variants collapsed",
    )
    lfm = next(x for x in bodies if x["candidate_id"] == "lfm25-350m")
    need(lfm["production_license_gate"] is True, "LFM license gate lost")
    qwen3 = next(x for x in bodies if x["candidate_id"] == "qwen3-0.6b")
    need(qwen3["parameters"] == 751_632_384, "Qwen3 parameter identity drift")

    need(
        provenance.get("source_head") == manifest["source_head"]
        and provenance.get("workflow_run_id") == manifest["workflow_run_id"]
        and provenance.get("config_sha256") == file_sha(CONFIG)
        and provenance.get("source_evidence_sha256") == file_sha(SOURCE)
        and provenance.get("stage_a_merge_commit")
            == cfg["source_authority"]["stage_a_merge_commit"]
        and provenance.get("external_provider_calls") == 0
        and provenance.get("training_steps") == 0
        and provenance.get("relevance_mapping_accessed") is False,
        "provenance mismatch",
    )
    expected_weights = {
        cid: authority[cid]["weight"]["lfs_sha256"] for cid in cfg["cohort"]
    }
    need(
        provenance.get("candidate_weight_sha256") == expected_weights,
        "provenance weight map mismatch",
    )
    return {
        "verdict": "DETECTIVE_002B_RUNTIME_READBACK_PASS",
        "bodies": len(bodies),
        "status_counts": status_counts,
        "runtime_pass": status_counts.get("runtime_pass", 0),
        "optimizer_steps": 0,
        "relevance_mapping_accessed": False,
        "population_expansion_admitted": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(verify(args.evidence), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
