#!/usr/bin/env python3
"""DETECTIVE-002B source-frozen no-training compact-body runtime smoke."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-002/runtime-smoke.json"
SOURCE = ROOT / "experiments/detective-002/source-evidence/source-evidence.json"


class RuntimeSmokeError(ValueError):
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


def current_rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def load_contract() -> tuple[dict, dict[str, dict]]:
    cfg = read_json(CONFIG)
    source = read_json(SOURCE)
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-002b-runtime-v1", "DETECTIVE-002B", 130
    ):
        raise RuntimeSmokeError("runtime-smoke identity drift")
    authority = cfg["source_authority"]
    if (
        authority["stage_a_issue"] != 127
        or authority["stage_a_pr"] != 129
        or authority["stage_a_merge_commit"]
        != "0993f0dfd859f67e88353bde0eeb2f6f377ccfac"
        or authority["path"] != SOURCE.relative_to(ROOT).as_posix()
        or authority["git_blob_sha1"] != "a09003d37dd3ef4f2bcd0f6a8aafd1de6457c01c"
        or git_blob_sha1(SOURCE) != authority["git_blob_sha1"]
        or source.get("schema") != authority["expected_schema"]
        or source.get("experiment") != authority["expected_experiment"]
        or source.get("issue") != 127
    ):
        raise RuntimeSmokeError("Stage-A source authority mismatch")
    if source.get("weight_bytes_downloaded") != 0:
        raise RuntimeSmokeError("Stage-A source archive unexpectedly downloaded weights")
    if source.get("training_steps") != 0:
        raise RuntimeSmokeError("Stage-A source archive contains training")
    if source.get("relevance_mapping_accessed") is not False:
        raise RuntimeSmokeError("Stage-A relevance mapping boundary violated")
    runtime = cfg["runtime"]
    if runtime != {
        "os": "ubuntu-24.04",
        "torch_cpu": "2.14.1+cpu",
        "transformers": "4.57.6",
        "huggingface_hub": "0.36.2",
        "safetensors": "0.8.0",
        "threads": 4,
        "seed": 260213,
        "do_sample": False,
        "max_new_tokens_per_prompt": 64,
        "prompts": 4,
        "max_generated_tokens_total": 256,
        "max_prompt_tokens": 512,
        "max_minutes_per_body": 25,
        "paid_api_calls": 0,
        "optimizer_steps": 0,
        "gpu_required": False,
        "torch_dtype": "float32",
    }:
        raise RuntimeSmokeError("runtime envelope drift")
    if len(cfg["battery"]) != 4 or [x["id"] for x in cfg["battery"]] != [
        "B01", "B02", "B03", "B04"
    ]:
        raise RuntimeSmokeError("battery drift")
    candidates = {x["candidate_id"]: x for x in source.get("candidates", [])}
    if set(cfg["cohort"]) != {
        "smollm2-135m-instruct",
        "smollm2-360m-instruct",
        "sprocket-500m-chat",
        "sprocket-500m-instruct",
        "qwen25-0.5b-instruct",
        "qwen3-0.6b",
        "lfm25-350m",
    }:
        raise RuntimeSmokeError("cohort drift")
    if not set(cfg["cohort"]) <= set(candidates):
        raise RuntimeSmokeError("Stage-A source archive missing cohort member")
    for cid in cfg["cohort"]:
        row = candidates[cid]
        if row.get("source_gate_pass") is not True:
            raise RuntimeSmokeError(cid + ": Stage-A source gate not passed")
        if row.get("runtime_eligibility") != "stage_b_candidate":
            raise RuntimeSmokeError(cid + ": not Stage-B candidate")
        if row.get("weight_bytes_downloaded") != 0 or row.get("training_steps") != 0:
            raise RuntimeSmokeError(cid + ": Stage-A execution boundary drift")
        weight = row.get("weight") or {}
        if (
            not re.fullmatch(r"[0-9a-f]{40}", row.get("resolved_revision", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", weight.get("lfs_sha256", ""))
            or not isinstance(weight.get("size"), int)
            or weight["size"] <= 0
        ):
            raise RuntimeSmokeError(cid + ": incomplete frozen source identity")
    chat = candidates["sprocket-500m-chat"]["weight"]
    instruct = candidates["sprocket-500m-instruct"]["weight"]
    if (
        chat["path"] == instruct["path"]
        or chat["lfs_sha256"] == instruct["lfs_sha256"]
        or chat["path"] != cfg["sprocket"]["chat_weight_path"]
        or instruct["path"] != cfg["sprocket"]["instruct_weight_path"]
    ):
        raise RuntimeSmokeError("Sprocket Chat/Instruct checkpoint conflation")
    qwen3 = candidates["qwen3-0.6b"]
    if qwen3.get("safetensors_parameters") != 751_632_384:
        raise RuntimeSmokeError("Qwen3 near-band parameter count drift")
    if cfg["qwen3"]["enable_thinking"] is not False:
        raise RuntimeSmokeError("Qwen3 matched battery must disable thinking")
    if candidates["lfm25-350m"].get("production_license_gate") is not True:
        raise RuntimeSmokeError("LFM production license gate unexpectedly cleared")
    return cfg, candidates


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


def marker_observations(task: dict, response: str) -> dict:
    lower = response.casefold()

    def hits(key: str) -> list[str]:
        return [
            marker for marker in task["observations"].get(key, [])
            if marker.casefold() in lower
        ]

    return {
        "required_hits": hits("required_markers"),
        "alternative_hits": hits("alternative_markers"),
        "qualifier_hits": hits("qualifier_markers"),
        "stale_hits": hits("stale_markers"),
        "forbidden_hits": hits("forbidden_markers"),
        "starts_json_object": response.lstrip().startswith("{"),
        "control_token_literals": sorted(
            set(re.findall(r"<\|[^|>]+\|>", response))
        ),
        "repeated_sentence_count": repeated_sentence_count(response),
        "characters": len(response),
        "whitespace_words": len(response.split()),
    }


def snapshot_and_stage(candidate: dict, stage: Path) -> dict:
    from huggingface_hub import snapshot_download

    weight = candidate["weight"]
    patterns = [
        weight["path"],
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
    ]
    snap = Path(snapshot_download(
        repo_id=candidate["repo"],
        revision=candidate["resolved_revision"],
        allow_patterns=patterns,
    ))
    original_weight = snap / weight["path"]
    if not original_weight.is_file():
        raise RuntimeSmokeError("selected frozen weight was not downloaded")
    actual_sha = file_sha(original_weight)
    if (
        actual_sha != weight["lfs_sha256"]
        or original_weight.stat().st_size != weight["size"]
    ):
        raise RuntimeSmokeError(
            f"{candidate['candidate_id']}: selected weight bytes differ from Stage-A LFS identity"
        )
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for src in snap.iterdir():
        if src.is_file() and src.name != "model.safetensors":
            shutil.copy2(src, stage / src.name)
    # Sprocket Chat uses chat/model.safetensors. All other selected weights are
    # staged under the standard Transformers filename.
    shutil.copy2(original_weight, stage / "model.safetensors")
    if file_sha(stage / "model.safetensors") != weight["lfs_sha256"]:
        raise RuntimeSmokeError("staged model weight hash mismatch")

    for key, filename in (
        ("config", "config.json"),
        ("tokenizer_config", "tokenizer_config.json"),
        ("chat_template_file", "chat_template.jinja"),
    ):
        meta = candidate.get(key) or {}
        if meta.get("status") == "present":
            target = stage / filename
            if not target.is_file():
                raise RuntimeSmokeError(
                    f"{candidate['candidate_id']}: frozen {filename} missing at runtime"
                )
            if file_sha(target) != meta["sha256"] or target.stat().st_size != meta["bytes"]:
                raise RuntimeSmokeError(
                    f"{candidate['candidate_id']}: frozen {filename} identity mismatch"
                )
    return {
        "source_snapshot": str(snap),
        "selected_weight_path": weight["path"],
        "selected_weight_sha256": actual_sha,
        "selected_weight_bytes": original_weight.stat().st_size,
        "staged_weight_sha256": file_sha(stage / "model.safetensors"),
    }


def render_prompt(tok, cfg: dict, candidate_id: str, task: dict):
    messages = [
        {"role": "system", "content": cfg["system"]},
        {"role": "user", "content": task["user"]},
    ]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if candidate_id == "qwen3-0.6b":
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(messages, **kwargs), messages


def save_partial(out: Path, record: dict) -> None:
    write_json(out, record)


def run(candidate_id: str, out: Path) -> dict:
    cfg, candidates = load_contract()
    if candidate_id not in cfg["cohort"]:
        raise RuntimeSmokeError("candidate not in frozen Stage-B cohort")
    candidate = candidates[candidate_id]

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        raise RuntimeSmokeError("DETECTIVE-002B is CPU-only")
    torch.set_num_threads(cfg["runtime"]["threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg["runtime"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    record = {
        "schema": "detective-002b-body-runtime-v1",
        "experiment": "DETECTIVE-002B",
        "issue": 130,
        "candidate_id": candidate_id,
        "status": "RUNNING",
        "source_evidence_sha256": file_sha(SOURCE),
        "config_sha256": file_sha(CONFIG),
        "source": {
            "repo": candidate["repo"],
            "revision": candidate["resolved_revision"],
            "variant": candidate["variant"],
            "architecture": candidate["architectures"],
            "model_type": candidate["model_type"],
            "safetensors_parameters": candidate["safetensors_parameters"],
            "license": candidate["license"],
            "production_license_gate": candidate["production_license_gate"],
            "weight": candidate["weight"],
        },
        "runtime_contract": cfg["runtime"],
        "relevance_mapping_accessed": False,
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "population_expansion_admitted": False,
        "tasks": [],
    }
    save_partial(out, record)

    start_wall = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"detective002b-{candidate_id}-") as td:
        stage = Path(td) / "model"
        try:
            source_runtime = snapshot_and_stage(candidate, stage)
            record["source_runtime"] = source_runtime
            save_partial(out, record)

            t0 = time.monotonic()
            tok = AutoTokenizer.from_pretrained(
                stage, local_files_only=True, trust_remote_code=False
            )
            tokenizer_load = time.monotonic() - t0
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token

            # Template conformance is checked before the expensive model load.
            rendered = []
            for task in cfg["battery"]:
                ids, messages = render_prompt(tok, cfg, candidate_id, task)
                if ids.shape[-1] > cfg["runtime"]["max_prompt_tokens"]:
                    raise RuntimeSmokeError(
                        f"{candidate_id}: rendered prompt exceeds fixed context budget"
                    )
                rendered.append((task, ids, messages))

            t0 = time.monotonic()
            model = AutoModelForCausalLM.from_pretrained(
                stage,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
            )
            model.config.use_cache = True
            model.eval()
            model_load = time.monotonic() - t0
            rss_after_load = current_rss_kib()
            peak_after_load = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            record["environment"] = {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu": cpu_model(),
                "cpu_count": os.cpu_count(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "cuda_available": torch.cuda.is_available(),
                "tokenizer_load_seconds": tokenizer_load,
                "model_load_seconds": model_load,
                "rss_after_load_kib": rss_after_load,
                "peak_rss_after_load_kib": peak_after_load,
            }
            save_partial(out, record)

            total_generated = 0
            for task, ids, messages in rendered:
                prefill_started = time.monotonic()
                with torch.inference_mode():
                    prefill = model(input_ids=ids, use_cache=False)
                prefill_seconds = time.monotonic() - prefill_started
                del prefill

                started = time.monotonic()
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
                elapsed = time.monotonic() - started
                tail = generated.sequences[0, ids.shape[-1]:].tolist()
                response = tok.decode(tail, skip_special_tokens=True)
                raw_special = tok.decode(tail, skip_special_tokens=False)
                total_generated += len(tail)
                if total_generated > cfg["runtime"]["max_generated_tokens_total"]:
                    raise RuntimeSmokeError("generated-token budget exceeded")
                task_record = {
                    "task_id": task["id"],
                    "family": task["family"],
                    "messages": messages,
                    "prompt_token_count": int(ids.shape[-1]),
                    "prompt_token_sha256": hashlib.sha256(
                        ids.numpy().tobytes()
                    ).hexdigest(),
                    "prefill_seconds": prefill_seconds,
                    "prompt_tokens_per_second": (
                        int(ids.shape[-1]) / prefill_seconds
                        if prefill_seconds else None
                    ),
                    "generation_seconds": elapsed,
                    "generated_token_ids": tail,
                    "generated_tokens": len(tail),
                    "completion_tokens_per_second": (
                        len(tail) / elapsed if elapsed else None
                    ),
                    "hit_max_new_tokens": (
                        len(tail) == cfg["runtime"]["max_new_tokens_per_prompt"]
                    ),
                    "response": response,
                    "response_with_special_tokens": raw_special,
                    "observations": marker_observations(task, response),
                }
                record["tasks"].append(task_record)
                save_partial(out, record)

            if len(record["tasks"]) != 4:
                raise RuntimeSmokeError("incomplete dialogue battery")
            record["status"] = "runtime_pass"
            record["runtime_failure"] = None
            record["total_generated_tokens"] = total_generated
            record["wall_seconds"] = time.monotonic() - start_wall
            record["peak_rss_kib"] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            record["decision"] = {
                "classification": "runtime_pass",
                "training_selection_allowed": False,
                "population_expansion_admitted": False,
            }
            save_partial(out, record)
            del model
            gc.collect()
            return record
        except RuntimeSmokeError:
            raise
        except Exception as exc:
            # Preserve runtime/protocol incompatibilities as experimental
            # outcomes rather than dropping the body from the matrix.
            record["status"] = (
                "protocol_specialization_required"
                if "chat template" in str(exc).casefold()
                or "apply_chat_template" in str(exc).casefold()
                else "runtime_fail"
            )
            record["runtime_failure"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            record["wall_seconds"] = time.monotonic() - start_wall
            record["peak_rss_kib"] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            record["decision"] = {
                "classification": record["status"],
                "training_selection_allowed": False,
                "population_expansion_admitted": False,
            }
            save_partial(out, record)
            return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = run(args.candidate, args.out)
    print(json.dumps({
        "candidate_id": result["candidate_id"],
        "status": result["status"],
        "tasks": len(result.get("tasks", [])),
        "wall_seconds": result.get("wall_seconds"),
        "peak_rss_kib": result.get("peak_rss_kib"),
        "optimizer_steps": result["optimizer_steps"],
        "population_expansion_admitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
