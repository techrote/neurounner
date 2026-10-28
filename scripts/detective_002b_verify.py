#!/usr/bin/env python3
"""Independent DETECTIVE-002B runtime-evidence assembly/readback."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def repeated_sentences(text: str) -> int:
    parts = [
        " ".join(x.casefold().split())
        for x in re.split(r"[.!?]+", text)
        if x.strip()
    ]
    return len(parts) - len(set(parts))


def recompute_observation(task: dict, response: str,
                          generated_tokens: int) -> dict:
    obs = task["observation"]
    lower = response.casefold()
    out = {
        "kind": obs["kind"] if "kind" in obs else "markers",
        "response_words": len(response.split()),
        "repeated_sentence_count": repeated_sentences(response),
        "truncated": generated_tokens >= 64,
    }
    if task["id"] == "B02":
        out.update({
            "remembered_marker_hit":
                obs["required_marker"].casefold() in lower,
            "qualifier_hits": [
                x for x in obs["qualifier_any"]
                if x.casefold() in lower
            ],
        })
    elif task["id"] == "B03":
        out.update({
            "corrected_marker_hit":
                obs["corrected_marker"].casefold() in lower,
            "stale_marker_hit":
                obs["stale_marker"].casefold() in lower,
        })
    elif task["id"] == "B04":
        out.update({
            "unsupported_marker_hit":
                obs["unsupported_marker"].casefold() in lower,
            "qualifier_hits": [
                x for x in obs["qualifier_any"]
                if x.casefold() in lower
            ],
        })
    return out


def load_authority() -> tuple[dict, dict[str, dict]]:
    cfg = read_json(CONFIG)
    source = read_json(SOURCE)
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-npc-runtime-smoke-v1", "DETECTIVE-002B", 130
    ):
        raise VerifyError("wrong DETECTIVE-002B identity")
    if sha_file(SOURCE) != cfg["source_authority"]["sha256"]:
        raise VerifyError("source evidence hash drift")
    rows = {x["candidate_id"]: x for x in source["candidates"]}
    if set(cfg["candidates"]) != {
        cid for cid, x in rows.items()
        if x["runtime_eligibility"] == "stage_b_candidate"
    }:
        raise VerifyError("runtime candidate set no longer matches Stage A")
    return cfg, rows


def validate_result(path: Path, cfg: dict,
                    source_rows: dict[str, dict]) -> dict:
    doc = read_json(path)
    cid = doc.get("candidate_id")
    if cid not in cfg["candidates"]:
        raise VerifyError("unexpected runtime candidate " + str(cid))
    source = source_rows[cid]
    if (
        doc.get("schema") != "detective-npc-runtime-result-v1"
        or doc.get("experiment") != "DETECTIVE-002B"
        or doc.get("issue") != 130
        or doc.get("config_sha256") != sha_file(CONFIG)
        or doc.get("source_evidence_sha256") != sha_file(SOURCE)
        or doc.get("source") != source
        or doc.get("runtime_budget") != cfg["runtime"]
        or doc.get("training_steps") != 0
        or doc.get("paid_api_calls") != 0
        or doc.get("gpu_used") is not False
        or doc.get("relevance_mapping_accessed") is not False
        or doc.get("population_expansion_admitted") is not False
    ):
        raise VerifyError("runtime result identity/gate mismatch " + cid)
    selected = doc.get("selected_weight_verified")
    if selected != {
        "path": source["weight"]["path"],
        "sha256": source["weight"]["lfs_sha256"],
        "bytes": source["weight"]["size"],
    }:
        raise VerifyError("selected weight verification mismatch " + cid)
    runner = doc.get("runner") or {}
    if (
        runner.get("torch_threads") != 4
        or runner.get("torch_interop_threads") != 1
    ):
        raise VerifyError("runtime thread budget mismatch " + cid)
    if doc.get("runtime_status") not in {
        "runtime_pass", "runtime_fail",
        "protocol_specialization_required",
    }:
        raise VerifyError("invalid runtime status " + cid)

    tasks = doc.get("tasks")
    if not isinstance(tasks, list):
        raise VerifyError("tasks must be list " + cid)
    expected_tasks = cfg["tasks"]
    if len(tasks) > len(expected_tasks):
        raise VerifyError("too many runtime tasks " + cid)
    for index, row in enumerate(tasks):
        task = expected_tasks[index]
        if (
            row.get("task_id") != task["id"]
            or row.get("family") != task["family"]
            or row.get("prompt") != task["prompt"]
        ):
            raise VerifyError("runtime task identity drift " + cid)
        count = row.get("generated_tokens")
        ids = row.get("generated_token_ids")
        if type(count) is not int or not 0 <= count <= 64:
            raise VerifyError("generated token count invalid " + cid)
        if not isinstance(ids, list) or len(ids) != count:
            raise VerifyError("generated token IDs mismatch " + cid)
        response = row.get("response")
        if not isinstance(response, str):
            raise VerifyError("response missing " + cid)
        expected_obs = recompute_observation(task, response, count)
        if row.get("observation") != expected_obs:
            raise VerifyError("mechanical observation does not replay " + cid)
        if not isinstance(row.get("elapsed_seconds"), (int, float)):
            raise VerifyError("task elapsed time missing " + cid)

    total = sum(x["generated_tokens"] for x in tasks)
    if doc.get("total_generated_tokens") != total or total > 256:
        raise VerifyError("aggregate token budget mismatch " + cid)

    if doc["runtime_status"] == "runtime_pass":
        if len(tasks) != 4:
            raise VerifyError("runtime_pass requires complete battery " + cid)
        for field in (
            "model_load_seconds", "rss_after_load_kib",
            "peak_rss_kib", "loaded_parameter_count",
        ):
            if not isinstance(doc.get(field), (int, float)):
                raise VerifyError("runtime_pass metric missing "
                                  + cid + ": " + field)
        if doc.get("protocol_specialization_required") is not False:
            raise VerifyError("runtime_pass mislabeled protocol " + cid)
    else:
        if not isinstance(doc.get("error"), str):
            raise VerifyError("runtime failure lacks retained error " + cid)

    if cid == "qwen3-0.6b":
        need(source["safetensors_parameters"] == 751632384,
             "Qwen3 near-band source count drift")
    if cid == "lfm25-350m":
        need(source["production_license_gate"] is True,
             "LFM production license gate disappeared")
    return doc


def result_paths(root: Path, cfg: dict) -> dict[str, Path]:
    files = list(root.rglob("*.json"))
    by = {}
    for path in files:
        try:
            doc = read_json(path)
        except Exception:
            continue
        cid = doc.get("candidate_id")
        if cid in cfg["candidates"]:
            if cid in by:
                raise VerifyError("duplicate result for " + cid)
            by[cid] = path
    if set(by) != set(cfg["candidates"]):
        missing = sorted(set(cfg["candidates"]) - set(by))
        raise VerifyError("runtime result set incomplete: " + ",".join(missing))
    return by


def candidate_summary(doc: dict) -> dict:
    tasks = doc["tasks"]
    elapsed = sum(float(x["elapsed_seconds"]) for x in tasks)
    tokens = sum(int(x["generated_tokens"]) for x in tasks)
    by_id = {x["task_id"]: x for x in tasks}
    summary = {
        "runtime_status": doc["runtime_status"],
        "model_load_seconds": doc.get("model_load_seconds"),
        "rss_after_load_kib": doc.get("rss_after_load_kib"),
        "peak_rss_kib": doc.get("peak_rss_kib"),
        "loaded_parameter_count": doc.get("loaded_parameter_count"),
        "generated_tokens": tokens,
        "generation_seconds": elapsed,
        "aggregate_tokens_per_second":
            tokens / elapsed if elapsed else None,
        "truncated_tasks": sum(
            bool(x["observation"]["truncated"]) for x in tasks),
        "repeated_sentences": sum(
            int(x["observation"]["repeated_sentence_count"])
            for x in tasks),
        "memory_marker_hit": None,
        "memory_qualifier_hit": None,
        "corrected_marker_hit": None,
        "stale_marker_hit": None,
        "rumor_qualifier_hit": None,
        "unsupported_marker_hit": None,
    }
    if "B02" in by_id:
        o = by_id["B02"]["observation"]
        summary["memory_marker_hit"] = o["remembered_marker_hit"]
        summary["memory_qualifier_hit"] = bool(o["qualifier_hits"])
    if "B03" in by_id:
        o = by_id["B03"]["observation"]
        summary["corrected_marker_hit"] = o["corrected_marker_hit"]
        summary["stale_marker_hit"] = o["stale_marker_hit"]
    if "B04" in by_id:
        o = by_id["B04"]["observation"]
        summary["rumor_qualifier_hit"] = bool(o["qualifier_hits"])
        summary["unsupported_marker_hit"] = o["unsupported_marker_hit"]
    return summary


def markdown_report(summary: dict) -> str:
    lines = [
        "# DETECTIVE-002B — source-frozen four-CPU runtime smoke",
        "",
        "This is a no-training runtime/behavior smoke, not a model ranking or "
        "detective-NPC admission decision.",
        "",
        "| Candidate | Status | Load s | RSS MiB | Peak MiB | Decode tok/s | "
        "Memory+qualifier | Correction | Rumor qualifier | Truncated |",
        "|---|---|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for cid in summary["candidate_order"]:
        row = summary["candidates"][cid]
        def num(value, div=1.0):
            return "n/a" if value is None else f"{float(value)/div:.2f}"
        mq = (
            "n/a" if row["memory_marker_hit"] is None
            else f"{int(bool(row['memory_marker_hit']))}/"
                 f"{int(bool(row['memory_qualifier_hit']))}"
        )
        correction = (
            "n/a" if row["corrected_marker_hit"] is None
            else f"{int(bool(row['corrected_marker_hit']))}/"
                 f"stale:{int(bool(row['stale_marker_hit']))}"
        )
        rumor = (
            "n/a" if row["rumor_qualifier_hit"] is None
            else str(int(bool(row["rumor_qualifier_hit"])))
        )
        lines.append(
            f"| {cid} | {row['runtime_status']} | "
            f"{num(row['model_load_seconds'])} | "
            f"{num(row['rss_after_load_kib'], 1024)} | "
            f"{num(row['peak_rss_kib'], 1024)} | "
            f"{num(row['aggregate_tokens_per_second'])} | "
            f"{mq} | {correction} | {rumor} | "
            f"{row['truncated_tasks']} |"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        f"- runtime pass: **{summary['runtime_passes']}/{len(summary['candidate_order'])}**;",
        f"- protocol specialization required: **{summary['protocol_specialization_required']}**;",
        f"- runtime failures: **{summary['runtime_failures']}**;",
        "- optimizer steps: **0**;",
        "- paid API calls: **0**;",
        "- hidden DETECTIVE-001R1 relevance mapping accessed: **no**;",
        "- population expansion admitted: **no**.",
        "",
        "Marker columns are mechanical observations only. They are not human "
        "quality scores and do not select a body for future training.",
    ]
    return "\n".join(lines) + "\n"


def hash_inventory(root: Path) -> list[str]:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel != "hashes.sha256":
            rows.append(f"{sha_file(path)}  {rel}")
    return rows


def assemble(results_root: Path, out: Path, source_head: str,
             run_id: str, run_url: str) -> dict:
    cfg, source_rows = load_authority()
    paths = result_paths(results_root, cfg)
    if out.exists():
        shutil.rmtree(out)
    (out / "raw").mkdir(parents=True)
    candidates = {}
    result_sha = {}
    for cid in cfg["candidates"]:
        doc = validate_result(paths[cid], cfg, source_rows)
        destination = out / "raw" / f"{cid}.json"
        shutil.copy2(paths[cid], destination)
        candidates[cid] = candidate_summary(doc)
        result_sha[cid] = sha_file(destination)
    summary = {
        "schema": "detective-npc-runtime-summary-v1",
        "experiment": "DETECTIVE-002B",
        "issue": 130,
        "config_sha256": sha_file(CONFIG),
        "source_evidence_sha256": sha_file(SOURCE),
        "candidate_order": cfg["candidates"],
        "candidates": candidates,
        "runtime_passes": sum(
            x["runtime_status"] == "runtime_pass"
            for x in candidates.values()),
        "runtime_failures": sum(
            x["runtime_status"] == "runtime_fail"
            for x in candidates.values()),
        "protocol_specialization_required": sum(
            x["runtime_status"] == "protocol_specialization_required"
            for x in candidates.values()),
        "training_steps": 0,
        "paid_api_calls": 0,
        "relevance_mapping_accessed": False,
        "population_expansion_admitted": False,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "provenance.json", {
        "source_head": source_head,
        "workflow_run_id": run_id,
        "workflow_run_url": run_url,
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha_file(CONFIG),
        "source_evidence":
            SOURCE.relative_to(ROOT).as_posix(),
        "source_evidence_sha256": sha_file(SOURCE),
        "raw_result_sha256": result_sha,
        "runtime_candidates": cfg["candidates"],
        "excluded": cfg["excluded"],
        "training_steps": 0,
        "paid_api_calls": 0,
        "relevance_mapping_accessed": False,
        "population_expansion_admitted": False,
    })
    (out / "REPORT.md").write_text(
        markdown_report(summary), encoding="utf-8")
    (out / "hashes.sha256").write_text(
        "\n".join(hash_inventory(out)) + "\n", encoding="utf-8")
    return summary


def verify_evidence(root: Path) -> dict:
    cfg, source_rows = load_authority()
    expected = {
        "REPORT.md", "hashes.sha256", "provenance.json", "summary.json",
        *{f"raw/{cid}.json" for cid in cfg["candidates"]},
    }
    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*") if p.is_file()
    }
    if present != expected:
        raise VerifyError("runtime evidence file set mismatch")
    recorded = (root / "hashes.sha256").read_text(
        encoding="utf-8").splitlines()
    if recorded != hash_inventory(root):
        raise VerifyError("runtime evidence hash inventory mismatch")
    raw_docs = {}
    for cid in cfg["candidates"]:
        path = root / "raw" / f"{cid}.json"
        raw_docs[cid] = validate_result(path, cfg, source_rows)
    expected_candidates = {
        cid: candidate_summary(raw_docs[cid])
        for cid in cfg["candidates"]
    }
    summary = read_json(root / "summary.json")
    if (
        summary.get("schema") != "detective-npc-runtime-summary-v1"
        or summary.get("experiment") != "DETECTIVE-002B"
        or summary.get("issue") != 130
        or summary.get("config_sha256") != sha_file(CONFIG)
        or summary.get("source_evidence_sha256") != sha_file(SOURCE)
        or summary.get("candidate_order") != cfg["candidates"]
        or summary.get("candidates") != expected_candidates
        or summary.get("runtime_passes") != sum(
            x["runtime_status"] == "runtime_pass"
            for x in expected_candidates.values())
        or summary.get("runtime_failures") != sum(
            x["runtime_status"] == "runtime_fail"
            for x in expected_candidates.values())
        or summary.get("protocol_specialization_required") != sum(
            x["runtime_status"] == "protocol_specialization_required"
            for x in expected_candidates.values())
        or summary.get("training_steps") != 0
        or summary.get("paid_api_calls") != 0
        or summary.get("relevance_mapping_accessed") is not False
        or summary.get("population_expansion_admitted") is not False
    ):
        raise VerifyError("runtime aggregate summary does not replay")
    if (root / "REPORT.md").read_text(
            encoding="utf-8") != markdown_report(summary):
        raise VerifyError("runtime report does not replay")
    provenance = read_json(root / "provenance.json")
    if (
        provenance.get("config_sha256") != sha_file(CONFIG)
        or provenance.get("source_evidence_sha256") != sha_file(SOURCE)
        or provenance.get("runtime_candidates") != cfg["candidates"]
        or provenance.get("excluded") != cfg["excluded"]
        or provenance.get("training_steps") != 0
        or provenance.get("paid_api_calls") != 0
        or provenance.get("relevance_mapping_accessed") is not False
        or provenance.get("population_expansion_admitted") is not False
        or re.fullmatch(r"[0-9a-f]{40}",
                        str(provenance.get("source_head", ""))) is None
        or not str(provenance.get("workflow_run_id", "")).isdigit()
    ):
        raise VerifyError("runtime provenance mismatch")
    for cid, digest in provenance["raw_result_sha256"].items():
        if cid not in cfg["candidates"]:
            raise VerifyError("unknown provenance candidate " + cid)
        if sha_file(root / "raw" / f"{cid}.json") != digest:
            raise VerifyError("raw result provenance hash mismatch " + cid)
    return {
        "verdict": "DETECTIVE_002B_RUNTIME_READBACK_PASS",
        "candidates": len(cfg["candidates"]),
        "runtime_passes": summary["runtime_passes"],
        "runtime_failures": summary["runtime_failures"],
        "protocol_specialization_required":
            summary["protocol_specialization_required"],
        "training_steps": 0,
        "paid_api_calls": 0,
        "population_expansion_admitted": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--source-head")
    ap.add_argument("--run-id")
    ap.add_argument("--run-url")
    ap.add_argument("--verify", type=Path)
    args = ap.parse_args()
    if args.verify:
        print(json.dumps(
            verify_evidence(args.verify), indent=2, sort_keys=True))
    elif args.results_root and args.out:
        if not args.source_head or not args.run_id or not args.run_url:
            raise SystemExit(
                "--source-head/--run-id/--run-url required for assembly")
        summary = assemble(
            args.results_root, args.out,
            args.source_head, args.run_id, args.run_url)
        print(json.dumps({
            "runtime_passes": summary["runtime_passes"],
            "runtime_failures": summary["runtime_failures"],
            "protocol_specialization_required":
                summary["protocol_specialization_required"],
            "training_steps": 0,
            "population_expansion_admitted": False,
        }, indent=2, sort_keys=True))
    else:
        cfg, _ = load_authority()
        print(json.dumps({
            "contract": "PASS",
            "candidates": len(cfg["candidates"]),
            "training_steps": 0,
            "population_expansion_admitted": False,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
