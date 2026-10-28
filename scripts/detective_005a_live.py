#!/usr/bin/env python3
"""DETECTIVE-005A live no-training lineage runner.

Runs exactly one VILLAGE lineage (A/B/C siblings) against the frozen four-prompt
fun screen. All model imports happen only in --run mode. Raw text, special-token
text and output token IDs are preserved. No optimizer, model judge or game-state
mutation exists in this runner.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import resource
import tarfile
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_005a_fun_screen as screen

LINEAGE_RX = re.compile(r"^G(0[1-9]|1[0-2])$")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(file_sha(item)))
    return h.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(x.stat().st_size for x in path.rglob("*") if x.is_file())


def cpu_id() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def peak_rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def safe_unpack(archive: Path, destination: Path, source: dict) -> None:
    screen.require(archive.is_file(), "adapter archive missing")
    screen.require(archive.stat().st_size == source["asset_bytes"] and
                   file_sha(archive) == source["asset_sha256"],
                   "adapter archive byte/hash mismatch")
    screen.require(not destination.exists(), "adapter destination already exists")
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        names = set()
        screen.require(2 <= len(members) <= 12,
                       "unexpected adapter tar member count")
        for item in members:
            parts = Path(item.name).parts
            screen.require(
                item.isfile() and len(parts) == 1 and parts[0] not in {".", ".."} and
                item.name not in names and item.size <= 30_000_000 and
                re.fullmatch(r"[A-Za-z0-9_.-]+", item.name) is not None,
                "unsafe adapter archive entry")
            names.add(item.name)
        screen.require(
            {"adapter_model.safetensors", "adapter_config.json"} <= names,
            "adapter archive missing required files")
        for item in members:
            source_fh = tf.extractfile(item)
            screen.require(source_fh is not None, "adapter member unreadable")
            with source_fh, (destination / item.name).open("wb") as target:
                while True:
                    block = source_fh.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
    screen.require(
        directory_hash(destination) == source["adapter_directory_sha256"] and
        directory_bytes(destination) == source["adapter_directory_bytes"],
        "unpacked adapter directory identity mismatch")


def append_completed_raw(path: Path, row: dict) -> None:
    """Fsync a validated response before the next inference begins.

    The complete 12-row results.jsonl is still the only assemblable lineage
    evidence; partial records remain recovery evidence, never admission.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(screen.canonical(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_interrupted_receipt(lineage: str, out: Path, error: Exception) -> dict:
    """Make a failed model/load run visible in the always-uploaded artifact."""
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "results.partial.jsonl"
    preserved = screen.read_results(partial) if partial.is_file() else []
    receipt = {
        "schema": "detective-005a-interrupted-lineage-v1",
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "lineage": lineage,
        "repository_head_sha": (
            os.environ.get("DETECTIVE_HEAD_SHA")
            or os.environ.get("GITHUB_SHA")
            or "local"),
        "actions_run_id": os.environ.get("GITHUB_RUN_ID"),
        "completed_raw_results": len(preserved),
        "partial_results_sha256": (
            file_sha(partial) if partial.is_file() else None),
        "error_type": type(error).__name__,
        "error_detail": str(error)[:300],
        "optimizer_steps": 0,
        "model_judge_calls": 0,
        "production_admission": False,
        "population_expansion_admitted": False,
        "durably_promoted": False,
        "status": "incomplete_artifact_only",
    }
    write_json(out / "interrupted.json", receipt)
    return receipt


def lineage_member_ids(cfg: dict, lineage: str) -> list[str]:
    screen.require(LINEAGE_RX.fullmatch(lineage) is not None,
                   "lineage must be G01..G12")
    ids = [m["member_id"] for m in cfg["source_population"]["members"]
           if m["member_id"].split("-")[1] == lineage]
    screen.require(ids == [f"V-{lineage}-{s}" for s in "ABC"],
                   "lineage member cohort drift")
    return ids


def source_plan(lineage: str) -> dict:
    cfg = screen.load_contract()
    ids = lineage_member_ids(cfg, lineage)
    return {
        "schema": "detective-005a-live-lineage-plan-v1",
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "lineage": lineage,
        "member_ids": ids,
        "source_config_sha256": screen.sha256_file(screen.CONFIG),
        "release_manifest_git_blob_sha1":
            cfg["source_population"]["release_manifest"]["git_blob_sha1"],
        "member_contract_sha256":
            cfg["source_population"]["release_manifest"]["member_contract_sha256"],
        "base_weights_sha256":
            cfg["source_population"]["base_model"]["weights_sha256"],
        "prompts": [p["id"] for p in cfg["prompts"]],
        "expected_results": 12,
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "subjective_fun_is_ranking_signal": True,
        "objective_relevance_threshold": None,
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n",
                    encoding="utf-8")


def run_lineage(lineage: str, out: Path, download_dir: Path) -> dict:
    cfg = screen.load_contract()
    plan = source_plan(lineage)
    member_ids = plan["member_ids"]
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "results.partial.jsonl"
    screen.require(
        not partial_path.exists() and
        not (out / "results.jsonl").exists() and
        not (out / "manifest.json").exists(),
        "refuse to overwrite pre-existing lineage evidence")

    from huggingface_hub import snapshot_download
    from peft import PeftModel
    import peft
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(min(cfg["limits"]["max_threads"],
                              os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg["decode"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    base = cfg["source_population"]["base_model"]
    snapshot = Path(snapshot_download(
        repo_id=base["repo"],
        revision=base["revision"],
        allow_patterns=[
            base["weights_file"], "config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "generation_config.json",
        ],
    ))
    weights = snapshot / base["weights_file"]
    screen.require(weights.is_file() and
                   file_sha(weights) == base["weights_sha256"],
                   "base model weight SHA mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    head_sha = (os.environ.get("DETECTIVE_HEAD_SHA") or
                os.environ.get("GITHUB_SHA") or "local")
    run_id = os.environ.get("GITHUB_RUN_ID")
    common_runner = {
        "repository_head_sha": head_sha,
        "actions_run_id": run_id,
        "actions_run_url":
            (f"https://github.com/techrote/neurounner/actions/runs/{run_id}"
             if run_id else None),
        "platform": platform.platform(),
        "cpu_model": cpu_id(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "threads": torch.get_num_threads(),
        "base_weights_sha256_verified": file_sha(weights),
    }

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    member_summaries = []
    total_generated = 0
    for member_id in member_ids:
        source = screen.expected_source(cfg, member_id)
        archive = download_dir / source["asset_name"]
        adapter_dir = out / "work" / member_id
        safe_unpack(archive, adapter_dir, source)

        model_base = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(
            model_base, adapter_dir, is_trainable=False)
        model.config.use_cache = True
        model.eval()

        member_rows = []
        for prompt in cfg["prompts"]:
            messages = screen.prompt_messages(cfg, prompt["id"])
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt")
            screen.require(
                int(ids.shape[-1]) <= cfg["decode"]["max_prompt_tokens"],
                "prompt exceeds frozen token budget")
            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    ids,
                    do_sample=False,
                    max_new_tokens=cfg["decode"]["max_new_tokens"],
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.monotonic() - started
            tail = generated[0, ids.shape[-1]:].tolist()
            total_generated += len(tail)
            raw = tokenizer.decode(tail, skip_special_tokens=False)
            clean = tokenizer.decode(tail, skip_special_tokens=True)
            if tail and tail[-1] == tokenizer.eos_token_id:
                finish = "eos"
            elif len(tail) >= cfg["decode"]["max_new_tokens"]:
                finish = "length"
            else:
                finish = "stopped"
            row = {
                "schema": "detective-005a-raw-response-v1",
                "experiment": "DETECTIVE-005A",
                "member_id": member_id,
                "prompt_id": prompt["id"],
                "source": source,
                "request": screen.expected_request(cfg, prompt["id"]),
                "response": {
                    "text": clean,
                    "raw_with_special_tokens": raw,
                    "token_ids": tail,
                    "finish_reason": finish,
                    "prompt_tokens": int(ids.shape[-1]),
                    "completion_tokens": len(tail),
                    "elapsed_seconds": elapsed,
                },
                "runner": {
                    **common_runner,
                    "prompt_token_ids": ids[0].tolist(),
                    "prompt_ids_sha256": hashlib.sha256(
                        ids.numpy().tobytes()).hexdigest(),
                    "adapter_archive_sha256_verified": file_sha(archive),
                    "adapter_directory_sha256_verified":
                        directory_hash(adapter_dir),
                },
            }
            observation = screen.validate_result(cfg, row)
            append_completed_raw(partial_path, row)
            rows.append(row)
            member_rows.append(observation)

        member_summaries.append({
            "member_id": member_id,
            "hard_floor_pass":
                all(x["hard_floor_pass"] for x in member_rows),
            "hard_floor_failures":
                [x["prompt_id"] for x in member_rows
                 if not x["hard_floor_pass"]],
            "generated_tokens":
                sum(next(r for r in rows
                         if r["member_id"] == member_id and
                         r["prompt_id"] == x["prompt_id"])
                    ["response"]["completion_tokens"]
                    for x in member_rows),
        })
        del model
        del model_base
        gc.collect()

    screen.require(len(rows) == 12 and
                   len({(r["member_id"], r["prompt_id"]) for r in rows}) == 12,
                   "lineage result cardinality mismatch")
    results_path = out / "results.jsonl"
    results_path.write_text(
        "".join(screen.canonical(row) + "\n" for row in rows),
        encoding="utf-8")
    screen.require(results_path.read_bytes() == partial_path.read_bytes(),
                   "incremental raw archive disagrees with final results")
    manifest = {
        "schema": "detective-005a-live-lineage-manifest-v1",
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "lineage": lineage,
        "source_plan": plan,
        "runner_script_sha256": screen.sha256_file(Path(__file__)),
        "results_jsonl_sha256": screen.sha256_file(results_path),
        "results": len(rows),
        "members": member_summaries,
        "total_generated_tokens": total_generated,
        "peak_rss_kib": peak_rss_kib(),
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "model_judge_calls": 0,
        "production_admission": False,
        "population_expansion_admitted": False,
    }
    write_json(out / "manifest.json", manifest)
    return {
        "lineage": lineage,
        "members": len(member_ids),
        "results": len(rows),
        "generated_tokens": total_generated,
        "hard_floor_clean_members":
            sum(x["hard_floor_pass"] for x in member_summaries),
        "results_jsonl_sha256": manifest["results_jsonl_sha256"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out/detective-005a"))
    ap.add_argument("--download-dir", type=Path,
                    default=Path("downloads/detective-005a"))
    args = ap.parse_args()
    if args.run:
        try:
            result = run_lineage(args.lineage, args.out, args.download_dir)
        except Exception as exc:
            write_interrupted_receipt(args.lineage, args.out, exc)
            raise
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(source_plan(args.lineage),
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
