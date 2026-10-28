#!/usr/bin/env python3
"""DETECTIVE-002C exact LFM2.5-350M Transformers-v5 runtime requalification.

Stage A: small frozen tokenizer/config files only.
Stage B: exact frozen weight is downloaded only after tokenizer/rendering PASS.
No training, no relevance mapping, no production-license clearance.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.detective_002b_runtime import (  # noqa: E402
    cpu_model,
    current_rss_kib,
    marker_observations,
)

CONFIG = ROOT / "experiments/detective-002/lfm-v5-runtime.json"
SOURCE = ROOT / "experiments/detective-002/source-evidence/source-evidence.json"
BATTERY = ROOT / "experiments/detective-002/runtime-smoke.json"
HISTORICAL = ROOT / "datasets/detective-002b/raw/lfm25-350m.json"


class SourceIntegrityError(ValueError):
    """Hard source/custody failure; never classify as model incompatibility."""


class ContractError(ValueError):
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


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"))
    h.update(data)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    )


def load_contract() -> tuple[dict, dict, dict]:
    cfg = read_json(CONFIG)
    source = read_json(SOURCE)
    battery = read_json(BATTERY)
    historical = read_json(HISTORICAL)

    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-002c-lfm-runtime-v1", "DETECTIVE-002C", 133
    ):
        raise ContractError("DETECTIVE-002C identity drift")

    authority = cfg["source_authority"]
    expected_blobs = {
        SOURCE: authority["stage_a_git_blob_sha1"],
        BATTERY: authority["battery_git_blob_sha1"],
        HISTORICAL: authority["historical_result_git_blob_sha1"],
    }
    for path, expected in expected_blobs.items():
        if git_blob_sha1(path) != expected:
            raise SourceIntegrityError(
                f"frozen source blob changed: {path.relative_to(ROOT)}"
            )

    if authority["stage_a_merge_commit"] != (
        "0993f0dfd859f67e88353bde0eeb2f6f377ccfac"
    ):
        raise ContractError("Stage-A merge authority drift")

    candidate = next(
        (x for x in source.get("candidates", [])
         if x.get("candidate_id") == "lfm25-350m"),
        None,
    )
    if candidate is None:
        raise SourceIntegrityError("Stage-A LFM2.5-350M source record missing")

    model = cfg["model"]
    expected_model = {
        "candidate_id": "lfm25-350m",
        "repo": "LiquidAI/LFM2.5-350M",
        "revision": "9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
        "architecture": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "safetensors_parameters": 354483968,
        "weight_path": "model.safetensors",
        "weight_sha256":
            "1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e",
        "weight_bytes": 708984464,
        "tokenizer_config_sha256":
            "3701f370c70034d06e28947ff51c1063983393319f429e3ce58f06a45fae67cc",
        "tokenizer_config_bytes": 595,
        "chat_template_sha256":
            "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6",
        "chat_template_bytes": 5487,
        "config_sha256":
            "720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8",
        "config_bytes": 1281,
        "license": "lfm1.0",
        "production_license_gate": True,
    }
    if model != expected_model:
        raise ContractError("frozen LFM model contract drift")

    if (
        candidate.get("repo") != model["repo"]
        or candidate.get("resolved_revision") != model["revision"]
        or candidate.get("architectures") != model["architecture"]
        or candidate.get("model_type") != model["model_type"]
        or candidate.get("safetensors_parameters")
            != model["safetensors_parameters"]
        or candidate.get("license") != model["license"]
        or candidate.get("production_license_gate") is not True
        or candidate.get("weight", {}).get("path") != model["weight_path"]
        or candidate.get("weight", {}).get("lfs_sha256")
            != model["weight_sha256"]
        or candidate.get("weight", {}).get("size") != model["weight_bytes"]
        or candidate.get("tokenizer_config", {}).get("sha256")
            != model["tokenizer_config_sha256"]
        or candidate.get("tokenizer_config", {}).get("bytes")
            != model["tokenizer_config_bytes"]
        or candidate.get("chat_template_file", {}).get("sha256")
            != model["chat_template_sha256"]
        or candidate.get("chat_template_file", {}).get("bytes")
            != model["chat_template_bytes"]
        or candidate.get("config", {}).get("sha256")
            != model["config_sha256"]
        or candidate.get("config", {}).get("bytes")
            != model["config_bytes"]
        or candidate.get("runtime_eligibility") != "stage_b_candidate"
        or candidate.get("source_gate_pass") is not True
    ):
        raise SourceIntegrityError("Stage-A LFM source record differs from 002C pin")

    historical_cfg = cfg["historical_control"]
    failure = historical.get("runtime_failure") or {}
    if (
        historical.get("schema") != "detective-002b-body-runtime-v1"
        or historical.get("experiment") != historical_cfg["experiment"]
        or historical.get("candidate_id") != "lfm25-350m"
        or historical.get("status") != historical_cfg["status"]
        or historical.get("runtime_contract", {}).get("transformers")
            != historical_cfg["transformers"]
        or failure.get("type") != historical_cfg["error_type"]
        or historical_cfg["error_contains"] not in failure.get("message", "")
        or historical.get("optimizer_steps") != 0
        or historical.get("relevance_mapping_accessed") is not False
    ):
        raise SourceIntegrityError("historical 002B LFM failure identity drift")

    # Same semantic prompts/system; runtime package change is the experimental variable.
    if (
        battery.get("schema") != "detective-002b-runtime-v1"
        or battery.get("experiment") != "DETECTIVE-002B"
        or len(battery.get("battery", [])) != 4
        or [x["id"] for x in battery["battery"]]
            != ["B01", "B02", "B03", "B04"]
    ):
        raise SourceIntegrityError("merged 002B battery identity drift")

    runtime = cfg["runtime"]
    if runtime != {
        "os": "ubuntu-24.04",
        "python": "3.12",
        "torch_cpu": "2.14.1+cpu",
        "transformers": "5.20.1",
        "transformers_wheel_sha256":
            "5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19",
        "threads": 4,
        "seed": 260213,
        "do_sample": False,
        "max_prompt_tokens": 512,
        "max_new_tokens_per_prompt": 64,
        "max_generated_tokens_total": 256,
        "max_minutes": 25,
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "gpu_required": False,
        "dtype": "float32",
    }:
        raise ContractError("002C runtime envelope drift")

    if cfg["stages"] != {
        "tokenizer_only_first": True,
        "download_weight_only_after_tokenizer_pass": True,
        "require_tokenizer_backend": "tokenizers",
        "expected_tokenizer_class": "TokenizersBackend",
        "same_battery_as_002b": True,
    }:
        raise ContractError("002C stage contract drift")

    if cfg["decisions"] != {
        "allowed": [
            "runtime_compatibility_pass",
            "tokenizer_incompatible",
            "model_runtime_incompatible",
        ],
        "source_integrity_fail_is_hard_error": True,
        "population_expansion_admitted": False,
        "training_selection_allowed": False,
        "production_license_cleared": False,
        "relevance_mapping_accessed": False,
    }:
        raise ContractError("002C decision boundary drift")
    return cfg, candidate, battery


def verify_small_file(path: Path, expected_sha: str, expected_bytes: int,
                      label: str) -> None:
    if (
        not path.is_file()
        or file_sha(path) != expected_sha
        or path.stat().st_size != expected_bytes
    ):
        raise SourceIntegrityError(f"{label} bytes differ from frozen Stage-A source")


def small_snapshot(candidate: dict, stage: Path) -> dict:
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(
        repo_id=candidate["repo"],
        revision=candidate["resolved_revision"],
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ],
    ))
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for src in snap.iterdir():
        if src.is_file():
            shutil.copy2(src, stage / src.name)

    verify_small_file(
        stage / "config.json",
        candidate["config"]["sha256"],
        candidate["config"]["bytes"],
        "config.json",
    )
    verify_small_file(
        stage / "tokenizer_config.json",
        candidate["tokenizer_config"]["sha256"],
        candidate["tokenizer_config"]["bytes"],
        "tokenizer_config.json",
    )
    verify_small_file(
        stage / "chat_template.jinja",
        candidate["chat_template_file"]["sha256"],
        candidate["chat_template_file"]["bytes"],
        "chat_template.jinja",
    )
    return {
        "source_snapshot": str(snap),
        "config_sha256": file_sha(stage / "config.json"),
        "tokenizer_config_sha256": file_sha(stage / "tokenizer_config.json"),
        "chat_template_sha256": file_sha(stage / "chat_template.jinja"),
        "weight_downloaded": False,
    }


def render_prompts(tok, cfg: dict, battery: dict) -> list[dict]:
    rendered = []
    for task in battery["battery"]:
        messages = [
            {"role": "system", "content": battery["system"]},
            {"role": "user", "content": task["user"]},
        ]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        count = int(ids.shape[-1])
        if count > cfg["runtime"]["max_prompt_tokens"]:
            raise ContractError("rendered prompt exceeds 002B/002C context ceiling")
        rendered.append({
            "task_id": task["id"],
            "family": task["family"],
            "messages": messages,
            "input_ids": ids,
            "prompt_tokens": count,
            "prompt_token_sha256": hashlib.sha256(
                ids.numpy().tobytes()
            ).hexdigest(),
        })
    return rendered


def tokenizer_stage(cfg: dict, candidate: dict, battery: dict, stage: Path) -> dict:
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != cfg["runtime"]["transformers"]:
        raise SourceIntegrityError(
            f"unexpected Transformers runtime {transformers.__version__}"
        )

    source = small_snapshot(candidate, stage)
    started = time.monotonic()
    try:
        tok = AutoTokenizer.from_pretrained(
            stage, local_files_only=True, trust_remote_code=False
        )
        seconds = time.monotonic() - started
    except Exception as exc:
        return {
            "status": "tokenizer_incompatible",
            "source": source,
            "transformers": transformers.__version__,
            "tokenizer_error": {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            },
            "tokenizer_load_seconds": time.monotonic() - started,
            "rendered_prompts": [],
        }

    class_name = type(tok).__name__
    backend = getattr(tok, "backend", None)
    # Some implementations expose the backend enum/object rather than a plain string.
    backend_text = (
        backend if isinstance(backend, str)
        else getattr(backend, "value", None) or type(backend).__name__
        if backend is not None else None
    )
    if class_name != cfg["stages"]["expected_tokenizer_class"]:
        return {
            "status": "tokenizer_incompatible",
            "source": source,
            "transformers": transformers.__version__,
            "tokenizer_class": class_name,
            "tokenizer_backend": backend_text,
            "tokenizer_error": {
                "type": "TokenizerClassMismatch",
                "message": (
                    f"expected {cfg['stages']['expected_tokenizer_class']}, "
                    f"got {class_name}"
                ),
            },
            "tokenizer_load_seconds": seconds,
            "rendered_prompts": [],
        }

    try:
        rendered = render_prompts(tok, cfg, battery)
    except Exception as exc:
        return {
            "status": "tokenizer_incompatible",
            "source": source,
            "transformers": transformers.__version__,
            "tokenizer_class": class_name,
            "tokenizer_backend": backend_text,
            "tokenizer_error": {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            },
            "tokenizer_load_seconds": seconds,
            "rendered_prompts": [],
        }

    return {
        "status": "tokenizer_pass",
        "source": source,
        "transformers": transformers.__version__,
        "tokenizer_class": class_name,
        "tokenizer_backend": backend_text,
        "tokenizer_load_seconds": seconds,
        "rendered_prompts": [
            {
                k: v for k, v in row.items()
                if k != "input_ids"
            } for row in rendered
        ],
        "_tokenizer": tok,
        "_rendered": rendered,
    }


def download_weight(candidate: dict, stage: Path) -> dict:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=candidate["repo"],
        filename=candidate["weight"]["path"],
        revision=candidate["resolved_revision"],
    ))
    if (
        file_sha(path) != candidate["weight"]["lfs_sha256"]
        or path.stat().st_size != candidate["weight"]["size"]
    ):
        raise SourceIntegrityError("LFM weight bytes differ from frozen Stage-A identity")
    shutil.copy2(path, stage / "model.safetensors")
    if file_sha(stage / "model.safetensors") != candidate["weight"]["lfs_sha256"]:
        raise SourceIntegrityError("staged LFM weight hash mismatch")
    return {
        "selected_weight_path": candidate["weight"]["path"],
        "selected_weight_sha256": file_sha(path),
        "selected_weight_bytes": path.stat().st_size,
        "staged_weight_sha256": file_sha(stage / "model.safetensors"),
    }


def run(out: Path) -> dict:
    cfg, candidate, battery = load_contract()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if transformers.__version__ != cfg["runtime"]["transformers"]:
        raise SourceIntegrityError("Transformers version does not match 002C pin")
    if torch.cuda.is_available():
        raise ContractError("DETECTIVE-002C is CPU-only")
    torch.set_num_threads(cfg["runtime"]["threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg["runtime"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    result = {
        "schema": "detective-002c-lfm-runtime-result-v1",
        "experiment": "DETECTIVE-002C",
        "issue": 133,
        "status": "RUNNING",
        "config_sha256": file_sha(CONFIG),
        "source_evidence_sha256": file_sha(SOURCE),
        "battery_sha256": file_sha(BATTERY),
        "historical_result_sha256": file_sha(HISTORICAL),
        "model": cfg["model"],
        "runtime_contract": cfg["runtime"],
        "historical_control": cfg["historical_control"],
        "relevance_mapping_accessed": False,
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "production_license_cleared": False,
        "training_selection_allowed": False,
        "population_expansion_admitted": False,
        "tokenizer_stage": None,
        "model_stage": None,
    }
    write_json(out, result)

    started_wall = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="detective002c-lfm-") as td:
        stage = Path(td) / "model"
        token = tokenizer_stage(cfg, candidate, battery, stage)
        serial_token = {
            k: v for k, v in token.items()
            if not k.startswith("_")
        }
        result["tokenizer_stage"] = serial_token
        write_json(out, result)

        if token["status"] != "tokenizer_pass":
            result["status"] = "tokenizer_incompatible"
            result["wall_seconds"] = time.monotonic() - started_wall
            result["peak_rss_kib"] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            write_json(out, result)
            return result

        # Weight download is intentionally after tokenizer PASS.
        weight = download_weight(candidate, stage)
        result["tokenizer_stage"]["source"]["weight_downloaded"] = True
        result["weight_stage"] = weight
        write_json(out, result)

        tok = token["_tokenizer"]
        rendered = token["_rendered"]
        try:
            load_started = time.monotonic()
            model = AutoModelForCausalLM.from_pretrained(
                stage,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.float32,
            )
            model.config.use_cache = True
            model.eval()
            load_seconds = time.monotonic() - load_started
            env = {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu": cpu_model(),
                "cpu_count": os.cpu_count(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "cuda_available": torch.cuda.is_available(),
                "model_load_seconds": load_seconds,
                "rss_after_load_kib": current_rss_kib(),
                "peak_rss_after_load_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            }
            result["model_stage"] = {
                "status": "RUNNING",
                "environment": env,
                "tasks": [],
            }
            write_json(out, result)

            total = 0
            for task_cfg, rendered_row in zip(battery["battery"], rendered):
                ids = rendered_row["input_ids"]
                prefill_started = time.monotonic()
                with torch.inference_mode():
                    prefill = model(input_ids=ids, use_cache=False)
                prefill_seconds = time.monotonic() - prefill_started
                del prefill

                generation_started = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(
                        ids,
                        max_new_tokens=cfg["runtime"]["max_new_tokens_per_prompt"],
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=tok.eos_token_id,
                        return_dict_in_generate=True,
                    )
                elapsed = time.monotonic() - generation_started
                tail = generated.sequences[0, ids.shape[-1]:].tolist()
                total += len(tail)
                if total > cfg["runtime"]["max_generated_tokens_total"]:
                    raise ContractError("002C generated token budget exceeded")
                response = tok.decode(tail, skip_special_tokens=True)
                raw_special = tok.decode(tail, skip_special_tokens=False)
                row = {
                    "task_id": task_cfg["id"],
                    "family": task_cfg["family"],
                    "messages": rendered_row["messages"],
                    "prompt_token_count": rendered_row["prompt_tokens"],
                    "prompt_token_sha256":
                        rendered_row["prompt_token_sha256"],
                    "prefill_seconds": prefill_seconds,
                    "prompt_tokens_per_second": (
                        rendered_row["prompt_tokens"] / prefill_seconds
                        if prefill_seconds else None
                    ),
                    "generation_seconds": elapsed,
                    "generated_token_ids": tail,
                    "generated_tokens": len(tail),
                    "completion_tokens_per_second": (
                        len(tail) / elapsed if elapsed else None
                    ),
                    "hit_max_new_tokens":
                        len(tail) == cfg["runtime"]["max_new_tokens_per_prompt"],
                    "response": response,
                    "response_with_special_tokens": raw_special,
                    "observations": marker_observations(task_cfg, response),
                }
                result["model_stage"]["tasks"].append(row)
                write_json(out, result)

            if len(result["model_stage"]["tasks"]) != 4:
                raise ContractError("002C incomplete dialogue battery")
            result["model_stage"]["status"] = "runtime_compatibility_pass"
            result["model_stage"]["total_generated_tokens"] = total
            result["status"] = "runtime_compatibility_pass"
            result["wall_seconds"] = time.monotonic() - started_wall
            result["peak_rss_kib"] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            write_json(out, result)
            del model
            gc.collect()
            return result
        except (SourceIntegrityError, ContractError):
            raise
        except Exception as exc:
            result["model_stage"] = result["model_stage"] or {"tasks": []}
            result["model_stage"]["status"] = "model_runtime_incompatible"
            result["model_stage"]["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            }
            result["status"] = "model_runtime_incompatible"
            result["wall_seconds"] = time.monotonic() - started_wall
            result["peak_rss_kib"] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            write_json(out, result)
            return result


def archive(result_path: Path, out: Path, source_head: str,
            run_id: str, run_url: str) -> dict:
    cfg, _, _ = load_contract()
    result = read_json(result_path)
    if result.get("schema") != "detective-002c-lfm-runtime-result-v1":
        raise ContractError("002C result schema mismatch")
    if result.get("status") not in cfg["decisions"]["allowed"]:
        raise ContractError("002C result classification not admitted")
    if out.exists():
        shutil.rmtree(out)
    (out / "provenance").mkdir(parents=True)
    shutil.copy2(result_path, out / "result.json")
    manifest = {
        "schema": "detective-002c-runtime-archive-v1",
        "experiment": "DETECTIVE-002C",
        "issue": 133,
        "source_head": source_head,
        "workflow_run_id": str(run_id),
        "workflow_run_url": run_url,
        "status": result["status"],
        "config_sha256": file_sha(CONFIG),
        "source_evidence_sha256": file_sha(SOURCE),
        "battery_sha256": file_sha(BATTERY),
        "historical_result_sha256": file_sha(HISTORICAL),
        "result_sha256": file_sha(out / "result.json"),
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "relevance_mapping_accessed": False,
        "production_license_cleared": False,
        "training_selection_allowed": False,
        "population_expansion_admitted": False,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "provenance" / "source-artifacts.json", {
        "source_head": source_head,
        "workflow_run_id": str(run_id),
        "workflow_run_url": run_url,
        "stage_a_git_blob_sha1":
            cfg["source_authority"]["stage_a_git_blob_sha1"],
        "battery_git_blob_sha1":
            cfg["source_authority"]["battery_git_blob_sha1"],
        "historical_result_git_blob_sha1":
            cfg["source_authority"]["historical_result_git_blob_sha1"],
        "transformers": cfg["runtime"]["transformers"],
        "transformers_wheel_sha256":
            cfg["runtime"]["transformers_wheel_sha256"],
        "model_weight_sha256": cfg["model"]["weight_sha256"],
        "training_steps": 0,
        "external_provider_calls": 0,
        "relevance_mapping_accessed": False,
    })
    (out / "README.md").write_text(
        "# DETECTIVE-002C LFM2.5-350M runtime compatibility\n\n"
        "Focused source-bound requalification of the exact DETECTIVE-002A "
        "LFM2.5-350M checkpoint after DETECTIVE-002B recorded a Transformers "
        "4.57.6 TokenizersBackend construction failure. DETECTIVE-002C changes "
        "only the frozen Transformers runtime to 5.20.1 and verifies that wheel "
        "before installation. Tokenizer/rendering must pass before model weight "
        "download. No training, relevance mapping, body selection, license "
        "clearance or population admission occurs here.\n",
        encoding="utf-8",
    )
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
    ap.add_argument("--out", type=Path)
    ap.add_argument("--archive-from", type=Path)
    ap.add_argument("--archive-out", type=Path)
    ap.add_argument("--source-head")
    ap.add_argument("--run-id")
    ap.add_argument("--run-url")
    args = ap.parse_args()
    if args.archive_from:
        if not all((args.archive_out, args.source_head, args.run_id, args.run_url)):
            raise SystemExit("archive mode requires archive-out/source-head/run-id/run-url")
        m = archive(
            args.archive_from, args.archive_out, args.source_head,
            args.run_id, args.run_url,
        )
        print(json.dumps(m, indent=2, sort_keys=True))
    else:
        if args.out is None:
            raise SystemExit("--out required")
        result = run(args.out)
        print(json.dumps({
            "status": result["status"],
            "tokenizer_status": result["tokenizer_stage"]["status"],
            "model_status": (
                result.get("model_stage", {}) or {}
            ).get("status"),
            "optimizer_steps": 0,
            "production_license_cleared": False,
            "population_expansion_admitted": False,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
