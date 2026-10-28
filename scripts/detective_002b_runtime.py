#!/usr/bin/env python3
"""DETECTIVE-002B source-frozen no-training compact-body runtime smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-002/runtime-smoke.json"
SOURCE = ROOT / "experiments/detective-002/source-evidence/source-evidence.json"


class RuntimeContractError(ValueError):
    pass


class ProtocolSpecialization(RuntimeError):
    pass


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


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(
                errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def current_rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(
                errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def repeated_sentences(text: str) -> int:
    parts = [
        " ".join(x.casefold().split())
        for x in re.split(r"[.!?]+", text)
        if x.strip()
    ]
    return len(parts) - len(set(parts))


def load_authority(candidate_id: str) -> tuple[dict, dict]:
    cfg = read_json(CONFIG)
    source = read_json(SOURCE)
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-npc-runtime-smoke-v1", "DETECTIVE-002B", 130
    ):
        raise RuntimeContractError("wrong DETECTIVE-002B identity")
    if sha_file(SOURCE) != cfg["source_authority"]["sha256"]:
        raise RuntimeContractError("DETECTIVE-002A source evidence hash drift")
    if cfg["runtime"] != {
        "cpu_threads": 4,
        "interop_threads": 1,
        "max_job_minutes": 25,
        "max_new_tokens_per_task": 64,
        "max_generated_tokens_total": 256,
        "seed": 24402,
        "do_sample": False,
        "torch_dtype": "float32",
        "trust_remote_code": False,
        "paid_api_calls": 0,
        "training_steps": 0,
        "gpu_required": False,
    }:
        raise RuntimeContractError("runtime budget drift")
    if [x["id"] for x in cfg["tasks"]] != ["B01", "B02", "B03", "B04"]:
        raise RuntimeContractError("runtime battery drift")
    if candidate_id not in cfg["candidates"]:
        raise RuntimeContractError("candidate not admitted to Stage B matrix")
    rows = {
        x["candidate_id"]: x for x in source.get("candidates", [])
    }
    if set(cfg["candidates"]) - set(rows):
        raise RuntimeContractError("source evidence missing Stage B candidate")
    row = rows[candidate_id]
    if (
        row.get("source_gate_pass") is not True
        or row.get("runtime_eligibility") != "stage_b_candidate"
        or row.get("weight_bytes_downloaded") != 0
        or row.get("training_steps") != 0
    ):
        raise RuntimeContractError("candidate lacks Stage A runtime authority")
    if row["candidate_id"] == "qwen3-0.6b":
        if (
            row.get("admission_class") != "near_band_exception"
            or row.get("safetensors_parameters") != 751632384
        ):
            raise RuntimeContractError("Qwen3 near-band identity drift")
    if row["candidate_id"] == "lfm25-350m":
        if (
            row.get("license") != "lfm1.0"
            or row.get("production_license_gate") is not True
        ):
            raise RuntimeContractError("LFM production license gate drift")
    if row["candidate_id"].startswith("sprocket-500m-"):
        other_id = (
            "sprocket-500m-instruct"
            if row["candidate_id"] == "sprocket-500m-chat"
            else "sprocket-500m-chat"
        )
        other = rows[other_id]
        if (
            row["resolved_revision"] != other["resolved_revision"]
            or row["weight"]["path"] == other["weight"]["path"]
            or row["weight"]["lfs_sha256"]
                == other["weight"]["lfs_sha256"]
        ):
            raise RuntimeContractError("Sprocket Chat/Instruct collapsed")
    return cfg, row


def verify_small_file(path: Path, expected: dict, label: str) -> None:
    if expected["status"] != "present":
        return
    if sha_file(path) != expected["sha256"]:
        raise RuntimeContractError(label + " SHA mismatch")
    if path.stat().st_size != expected["bytes"]:
        raise RuntimeContractError(label + " byte count mismatch")


def observe(task: dict, response: str, generated_tokens: int,
            max_new_tokens: int) -> dict:
    lower = response.casefold()
    obs = task["observation"]
    out = {
        "kind": obs["kind"] if "kind" in obs else "markers",
        "response_words": len(response.split()),
        "repeated_sentence_count": repeated_sentences(response),
        "truncated": generated_tokens >= max_new_tokens,
    }
    if task["id"] == "B02":
        marker = obs["required_marker"]
        hits = [x for x in obs["qualifier_any"] if x.casefold() in lower]
        out.update({
            "remembered_marker_hit": marker.casefold() in lower,
            "qualifier_hits": hits,
        })
    elif task["id"] == "B03":
        out.update({
            "corrected_marker_hit":
                obs["corrected_marker"].casefold() in lower,
            "stale_marker_hit": obs["stale_marker"].casefold() in lower,
        })
    elif task["id"] == "B04":
        hits = [x for x in obs["qualifier_any"] if x.casefold() in lower]
        out.update({
            "unsupported_marker_hit":
                obs["unsupported_marker"].casefold() in lower,
            "qualifier_hits": hits,
        })
    return out


def render_prompt(tok, cfg: dict, candidate_id: str, task: dict):
    messages = [
        {"role": "system", "content": cfg["system"]},
        {"role": "user", "content": task["prompt"]},
    ]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if candidate_id == "qwen3-0.6b":
        kwargs["enable_thinking"] = False
    try:
        ids = tok.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        raise ProtocolSpecialization(
            "chat-template rendering failed: " + str(exc)[:300]) from exc
    if ids is None or getattr(ids, "numel", lambda: 0)() == 0:
        raise ProtocolSpecialization("chat template produced no prompt tokens")
    return ids, messages


def run(candidate_id: str, out: Path) -> dict:
    cfg, source = load_authority(candidate_id)

    # Heavy imports occur only after the complete source/budget contract passes.
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from safetensors import __version__ as safetensors_version
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import huggingface_hub

    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    try:
        torch.set_num_interop_threads(cfg["runtime"]["interop_threads"])
    except RuntimeError:
        pass
    torch.manual_seed(cfg["runtime"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repo = source["repo"]
    revision = source["resolved_revision"]
    selected = source["weight"]
    if not isinstance(selected, dict):
        raise RuntimeContractError("Stage B candidate lacks selected weight")

    # Exact selected weight is downloaded first and independently checked.
    weight_path = Path(hf_hub_download(
        repo, filename=selected["path"], revision=revision))
    if sha_file(weight_path) != selected["lfs_sha256"]:
        raise RuntimeContractError("selected weight SHA mismatch")
    if weight_path.stat().st_size != selected["size"]:
        raise RuntimeContractError("selected weight byte count mismatch")

    # Verify the small metadata files that Stage A hashed.
    config_path = Path(hf_hub_download(
        repo, filename="config.json", revision=revision))
    verify_small_file(config_path, source["config"], "config.json")
    tokenizer_config_path = Path(hf_hub_download(
        repo, filename="tokenizer_config.json", revision=revision))
    verify_small_file(
        tokenizer_config_path, source["tokenizer_config"],
        "tokenizer_config.json")

    chat_template_path = None
    if source.get("chat_template_file", {}).get("status") == "present":
        chat_template_path = Path(hf_hub_download(
            repo, filename="chat_template.jinja", revision=revision))
        verify_small_file(
            chat_template_path, source["chat_template_file"],
            "chat_template.jinja")

    base_record = {
        "schema": "detective-npc-runtime-result-v1",
        "experiment": "DETECTIVE-002B",
        "issue": 130,
        "candidate_id": candidate_id,
        "config_sha256": sha_file(CONFIG),
        "source_evidence_sha256": sha_file(SOURCE),
        "source": source,
        "selected_weight_verified": {
            "path": selected["path"],
            "sha256": sha_file(weight_path),
            "bytes": weight_path.stat().st_size,
        },
        "runtime_budget": cfg["runtime"],
        "runner": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "packages": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "safetensors": safetensors_version,
        },
        "training_steps": 0,
        "paid_api_calls": 0,
        "gpu_used": False,
        "relevance_mapping_accessed": False,
        "population_expansion_admitted": False,
        "tasks": [],
        "runtime_status": "incomplete",
    }
    write_json(out, base_record)

    try:
        started = time.monotonic()
        tok = AutoTokenizer.from_pretrained(
            repo, revision=revision, trust_remote_code=False)
        if (
            not getattr(tok, "chat_template", None)
            and chat_template_path is not None
        ):
            tok.chat_template = chat_template_path.read_text(
                encoding="utf-8")
        if not getattr(tok, "chat_template", None):
            raise ProtocolSpecialization(
                "frozen source has no usable chat template")

        auto_config = AutoConfig.from_pretrained(
            repo, revision=revision, trust_remote_code=False)
        weight_rel = Path(selected["path"])
        if weight_rel.parent.as_posix() == ".":
            model = AutoModelForCausalLM.from_pretrained(
                repo, revision=revision, trust_remote_code=False,
                torch_dtype=torch.float32)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                repo, revision=revision,
                subfolder=weight_rel.parent.as_posix(),
                config=auto_config,
                trust_remote_code=False,
                torch_dtype=torch.float32)
        model.config.use_cache = True
        model.eval()
        load_seconds = time.monotonic() - started
        rss_after_load = current_rss_kib()
        parameter_count = sum(p.numel() for p in model.parameters())

        total_generated = 0
        rows = []
        for task in cfg["tasks"]:
            ids, messages = render_prompt(
                tok, cfg, candidate_id, task)
            prompt_count = int(ids.shape[-1])
            task_started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    ids,
                    max_new_tokens=cfg["runtime"]["max_new_tokens_per_task"],
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=(
                        tok.pad_token_id
                        if tok.pad_token_id is not None
                        else tok.eos_token_id
                    ),
                    eos_token_id=tok.eos_token_id,
                )
            elapsed = time.monotonic() - task_started
            tail = generated[0, ids.shape[-1]:]
            generated_tokens = int(tail.numel())
            total_generated += generated_tokens
            if total_generated > cfg["runtime"]["max_generated_tokens_total"]:
                raise RuntimeContractError(
                    "aggregate generated-token budget exceeded")
            response = tok.decode(
                tail, skip_special_tokens=True).strip()
            raw_response = tok.decode(
                tail, skip_special_tokens=False)
            rows.append({
                "task_id": task["id"],
                "family": task["family"],
                "prompt": task["prompt"],
                "messages": messages,
                "prompt_tokens": prompt_count,
                "prompt_token_ids_sha256": hashlib.sha256(
                    ids.to(dtype=torch.int64).cpu().numpy().tobytes()
                ).hexdigest(),
                "generated_token_ids": tail.tolist(),
                "generated_tokens": generated_tokens,
                "elapsed_seconds": elapsed,
                "tokens_per_second": (
                    generated_tokens / elapsed if elapsed else None),
                "response": response,
                "response_with_special_tokens": raw_response,
                "observation": observe(
                    task, response, generated_tokens,
                    cfg["runtime"]["max_new_tokens_per_task"]),
            })
            base_record["tasks"] = rows
            write_json(out, base_record)

        if len(rows) != 4:
            raise RuntimeContractError("incomplete runtime battery")
        base_record.update({
            "runtime_status": "runtime_pass",
            "model_load_seconds": load_seconds,
            "rss_after_load_kib": rss_after_load,
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "loaded_parameter_count": parameter_count,
            "total_generated_tokens": total_generated,
            "protocol_specialization_required": False,
        })
    except ProtocolSpecialization as exc:
        base_record.update({
            "runtime_status": "protocol_specialization_required",
            "protocol_specialization_required": True,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "total_generated_tokens": sum(
                x.get("generated_tokens", 0)
                for x in base_record["tasks"]),
        })
    except Exception as exc:
        base_record.update({
            "runtime_status": "runtime_fail",
            "protocol_specialization_required": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:800],
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "total_generated_tokens": sum(
                x.get("generated_tokens", 0)
                for x in base_record["tasks"]),
        })

    write_json(out, base_record)
    return base_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        cfg, source = load_authority(args.candidate)
        print(json.dumps({
            "candidate": args.candidate,
            "source_revision": source["resolved_revision"],
            "weight_sha256": source["weight"]["lfs_sha256"],
            "tasks": len(cfg["tasks"]),
            "training_steps": 0,
            "population_expansion_admitted": False,
        }, indent=2, sort_keys=True))
    else:
        result = run(args.candidate, args.out)
        print(json.dumps({
            "candidate": args.candidate,
            "runtime_status": result["runtime_status"],
            "tasks": len(result["tasks"]),
            "total_generated_tokens":
                result.get("total_generated_tokens", 0),
            "training_steps": 0,
            "population_expansion_admitted": False,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
