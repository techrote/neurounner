#!/usr/bin/env python3
"""DETECTIVE-003D: source-native Sprocket root-Instruct live scripter bridge.

One publisher-native generation is preserved exactly. Strict native parsing and
one-call normalization are recorded separately. Only a valid normalized
detective-script-v1 programme may enter DETECTIVE-003C.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import detective_001b3_sprocket as b3  # noqa: E402
from scripts import detective_003a_session as session_api  # noqa: E402
from scripts import detective_003c_scripter_bridge as bridge  # noqa: E402
from scripts.detective_001_contract import ContractError  # noqa: E402

CONFIG = ROOT / "experiments/detective-003/sprocket-live-scripter.json"
CASE = ROOT / "experiments/detective-001/pilot-case.json"
B3_SPEC = ROOT / "experiments/detective-001/sprocket-scripter.json"
B3_RUNNER = ROOT / "scripts/detective_001b3_sprocket.py"
BRIDGE_CONFIG = ROOT / "experiments/detective-003/scripter-bridge.json"
BRIDGE_MODULE = ROOT / "scripts/detective_003c_scripter_bridge.py"
SESSION_MODULE = ROOT / "scripts/detective_003a_session.py"


class LiveScripterError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


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
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"))
    h.update(data)
    return h.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def current_rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def load_contract() -> tuple[dict, dict, dict]:
    cfg = read_json(CONFIG)
    case = read_json(CASE)
    b3_spec, b3_case = b3.load_spec()
    bridge_cfg, bridge_case = bridge.load_contract()

    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-003d-sprocket-live-bridge-v1",
        "DETECTIVE-003D",
        146,
    ):
        raise LiveScripterError("003D config identity drift")
    if canonical(case) != canonical(b3_case) or canonical(case) != canonical(bridge_case):
        raise LiveScripterError("case authority mismatch across source layers")

    pins = cfg["authority"]
    checks = (
        (B3_SPEC, pins["b3_spec"]["path"], pins["b3_spec"]["git_blob_sha1"]),
        (B3_RUNNER, pins["b3_native_wire"]["path"],
         pins["b3_native_wire"]["git_blob_sha1"]),
        (BRIDGE_CONFIG, pins["bridge_config"]["path"],
         pins["bridge_config"]["git_blob_sha1"]),
        (BRIDGE_MODULE, pins["bridge_module"]["path"],
         pins["bridge_module"]["git_blob_sha1"]),
        (SESSION_MODULE, pins["session_module"]["path"],
         pins["session_module"]["git_blob_sha1"]),
        (CASE, pins["case"]["path"], pins["case"]["git_blob_sha1"]),
    )
    for path, declared_path, expected_blob in checks:
        if declared_path != path.relative_to(ROOT).as_posix():
            raise LiveScripterError("authority path drift")
        if git_blob_sha1(path) != expected_blob:
            raise LiveScripterError("authority Git blob drift: " + declared_path)
    if case["case_id"] != pins["case"]["case_id"]:
        raise LiveScripterError("case ID drift")

    if cfg["model"] != b3_spec["model"]:
        raise LiveScripterError("003D model pins differ from frozen B3 authority")
    if cfg["publisher"] != {
        "repo": b3_spec["publisher"]["repo"],
        "commit": b3_spec["publisher"]["commit"],
        "sft_path": b3_spec["publisher"]["sft_path"],
        "sft_git_blob": b3_spec["publisher"]["sft_git_blob"],
        "harvest_path": b3_spec["publisher"]["harvest_path"],
        "harvest_git_blob": b3_spec["publisher"]["harvest_git_blob"],
    }:
        raise LiveScripterError("publisher source pins differ from B3 authority")
    if cfg["special_token_ids"] != b3_spec["special_token_ids"]:
        raise LiveScripterError("publisher special-token IDs drift")
    if cfg["model"]["weight_file"] != "model.safetensors":
        raise LiveScripterError("003D must use root Instruct checkpoint")
    if cfg["model"]["chat_checkpoint_forbidden"] != "chat/model.safetensors":
        raise LiveScripterError("Chat checkpoint exclusion drift")
    if cfg["normalization"] != {
        "one_native_call_only": True,
        "admitted_tools": ["read_asset", "correlate_assets", "probe_terminal"],
        "textual_repair": False,
        "fuzzy_mapping": False,
        "argument_repair": False,
    }:
        raise LiveScripterError("normalization authority widened")
    if cfg["acceptance"] != {
        "correct_tool_selection_required": False,
        "access_grant_required": False,
        "evidence_application_required": False,
        "one_generation_required": True,
        "exact_raw_output_required": True,
        "deterministic_replay_required": True,
        "host_execution": False,
        "training_admitted": False,
        "body_selection_made": False,
        "relevance_mapping_accessed": False,
        "learner_final_accessed": False,
        "population_expansion_admitted": False,
    }:
        raise LiveScripterError("003D acceptance authority widened")
    return cfg, case, b3_spec


def build_request(cfg: dict, case: dict) -> tuple[dict, dict]:
    session = session_api.start_session(
        case,
        cfg["session"]["participants"],
        cfg["session"]["session_key"],
    )
    req = cfg["game_request"]
    request = bridge.build_scripter_request(
        case,
        session,
        req["player_request"],
        req["visible_asset_ids"],
        req["disclosed_terminal_ids"],
        req["player_known_credentials"],
    )
    return session, request


def render_native_turns(cfg: dict, request: dict) -> tuple[list[dict], str]:
    user_content = (
        cfg["prompt"]["instruction"]
        + "\n\nPlayer request:\n"
        + request["player_request"]
        + "\n\nCurrent fictional game view:\n"
        + canonical(request["view"])
    )
    turns = [
        {"role": "system", "content": b3.manifest()},
        {"role": "user", "content": user_content},
    ]
    prompt_text = b3.render_text(turns)
    return turns, prompt_text


def normalize_native_output(raw_output: str, b3_spec: dict) -> dict:
    """Strictly parse one publisher-native call and normalize without repair."""
    try:
        call = b3.parse_native_tool_call(raw_output)
    except ContractError as exc:
        return {
            "native_parse_status": "rejected",
            "normalization_status": "not_attempted",
            "native_call": None,
            "program": None,
            "program_text": None,
            "error_type": "ContractError",
            "error": str(exc)[:256],
        }
    try:
        program = b3.tool_call_to_program(call, b3_spec)
    except ContractError as exc:
        return {
            "native_parse_status": "parsed",
            "normalization_status": "rejected",
            "native_call": call,
            "program": None,
            "program_text": None,
            "error_type": "ContractError",
            "error": str(exc)[:256],
        }
    return {
        "native_parse_status": "parsed",
        "normalization_status": "normalized",
        "native_call": call,
        "program": program,
        "program_text": canonical(program),
        "error_type": None,
        "error": None,
    }


def exact_probe_selected(cfg: dict, normalization: dict) -> bool:
    if normalization["normalization_status"] != "normalized":
        return False
    call = normalization["native_call"]
    expected = {
        "name": "probe_terminal",
        "arguments": {
            "terminal_id": "camera-console",
            "credential": "OXIDATION",
        },
    }
    return call == expected


def verify_publisher_checkout(cfg: dict, publisher_dir: Path) -> dict:
    def git(*args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", "-C", str(publisher_dir), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise LiveScripterError("publisher source Git verification failed") from exc
        return proc.stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != cfg["publisher"]["commit"]:
        raise LiveScripterError("publisher checkout commit mismatch")
    blobs = {}
    for path_key, blob_key in (
        ("sft_path", "sft_git_blob"),
        ("harvest_path", "harvest_git_blob"),
    ):
        path = cfg["publisher"][path_key]
        actual = git("hash-object", path)
        expected = cfg["publisher"][blob_key]
        if actual != expected:
            raise LiveScripterError("publisher source blob mismatch: " + path)
        blobs[path] = actual
    return {"commit": head, "git_blobs": blobs}


def _hash_inventory(root: Path) -> str:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel != "provenance/hashes.sha256":
            rows.append(f"{file_sha(path)}  {rel}")
    return "\n".join(rows) + "\n"


def archive_dataset(out_root: Path, result: dict, cfg: dict) -> Path:
    dataset = out_root / "dataset"
    if dataset.exists():
        import shutil
        shutil.rmtree(dataset)
    (dataset / "raw").mkdir(parents=True)
    (dataset / "provenance").mkdir(parents=True)
    write_json(dataset / "raw" / "run.json", result)
    manifest = {
        "schema": "detective-003d-sprocket-live-archive-v1",
        "experiment": "DETECTIVE-003D",
        "issue": 146,
        "status": result["status"],
        "outcome": result["outcome"],
        "source_head": result["repository"]["head_sha"],
        "workflow_run_id": result["repository"]["run_id"],
        "workflow_run_url": result["repository"]["run_url"],
        "config_sha256": file_sha(CONFIG),
        "model": cfg["model"],
        "raw_generated_tokens": result["generation"]["generated_tokens"],
        "native_parse_status": result["normalization"]["native_parse_status"],
        "normalization_status": result["normalization"]["normalization_status"],
        "exact_probe_selected": result["exact_probe_selected"],
        "transaction_status": (
            result["transaction"]["status"]
            if result["transaction"] is not None else None
        ),
        "camera_cache_visible_after_transaction": (
            result["camera_cache_visible_after_transaction"]
        ),
        "evidence_applied": result["evidence_application"]["applied"],
        "final_session_sha256": result["final_session"]["sha256"],
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "host_execution": False,
        "training_admitted": False,
        "body_selection_made": False,
        "relevance_mapping_accessed": False,
        "learner_final_accessed": False,
        "population_expansion_admitted": False,
    }
    write_json(dataset / "manifest.json", manifest)
    write_json(dataset / "provenance" / "source-artifacts.json", {
        "repository": "techrote/neurounner",
        "head_sha": result["repository"]["head_sha"],
        "workflow_run_id": result["repository"]["run_id"],
        "workflow_run_url": result["repository"]["run_url"],
        "config_sha256": file_sha(CONFIG),
        "authority_git_blobs": {
            key: value["git_blob_sha1"]
            for key, value in cfg["authority"].items()
        },
        "model_files": result["source"]["model_files"],
        "publisher_checkout": result["source"]["publisher_checkout"],
        "external_provider_calls": 0,
        "training_steps": 0,
        "host_execution": False,
        "relevance_mapping_accessed": False,
        "learner_final_accessed": False,
    })
    (dataset / "README.md").write_text(
        "# DETECTIVE-003D source-native Sprocket live scripter bridge\n\n"
        "One exact root Sprocket-500M Instruct generation on its publisher-native "
        "tool-call wire, followed by strict native parsing and one-call "
        "normalization into DETECTIVE-003C. Correct detective tool selection is "
        "not required for integration PASS. Raw control-token output and token "
        "IDs are preserved. Any valid normalized programme executes only through "
        "the fictional DETECTIVE authority; evidence is presented to a witness "
        "only if the deterministic receipt actually grants it and the game "
        "explicitly applies it. No model training, host execution, hidden "
        "relevance/learner-final access or population admission occurs.\n",
        encoding="utf-8",
    )
    (dataset / "provenance" / "hashes.sha256").write_text(
        _hash_inventory(dataset), encoding="utf-8"
    )
    return dataset


def live_run(publisher_dir: Path, out_root: Path) -> dict:
    cfg, case, b3_spec = load_contract()
    session, request = build_request(cfg, case)
    session_before = copy.deepcopy(session)
    turns, prompt_text = render_native_turns(cfg, request)

    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        raise LiveScripterError("003D is CPU-only")
    torch.set_num_threads(cfg["runtime"]["threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg["prompt"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    source = {
        "publisher_checkout": verify_publisher_checkout(cfg, publisher_dir),
        "model_files": {},
    }
    for name, expected_sha in (
        ("config.json", None),
        ("tokenizer.json", cfg["model"]["tokenizer_sha256"]),
        ("tokenizer_config.json", cfg["model"]["tokenizer_config_sha256"]),
        ("special_tokens_map.json", cfg["model"]["special_tokens_map_sha256"]),
        (cfg["model"]["weight_file"], cfg["model"]["weight_sha256"]),
    ):
        path = Path(hf_hub_download(
            cfg["model"]["repo"],
            filename=name,
            revision=cfg["model"]["revision"],
        ))
        digest = file_sha(path)
        if expected_sha is not None and digest != expected_sha:
            raise LiveScripterError("model source SHA mismatch: " + name)
        if (
            name == cfg["model"]["weight_file"]
            and path.stat().st_size != cfg["model"]["weight_bytes"]
        ):
            raise LiveScripterError("root Instruct weight byte count mismatch")
        source["model_files"][name] = {
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    if cfg["model"]["chat_checkpoint_forbidden"] in source["model_files"]:
        raise LiveScripterError("Chat checkpoint entered root-Instruct run")

    tok = AutoTokenizer.from_pretrained(
        cfg["model"]["repo"],
        revision=cfg["model"]["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    b3.verify_tokenizer(tok, b3_spec)
    ids_list = b3.prompt_ids(tok, b3_spec, turns)
    if len(ids_list) > cfg["prompt"]["max_prompt_tokens"]:
        raise LiveScripterError("publisher prompt exceeds 003D budget")
    ids = torch.tensor([ids_list], dtype=torch.long)

    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["repo"],
        revision=cfg["model"]["revision"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.config.use_cache = True
    model.eval()
    load_seconds = time.monotonic() - load_started

    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            ids,
            max_new_tokens=cfg["prompt"]["max_new_tokens"],
            do_sample=False,
            use_cache=True,
            eos_token_id=cfg["prompt"]["eos_token_id"],
            pad_token_id=cfg["special_token_ids"]["<|pad|>"],
        )
    generation_seconds = time.monotonic() - started
    tail = output[0, ids.shape[-1]:].tolist()
    raw_output = tok.decode(tail, skip_special_tokens=False)

    normalization = normalize_native_output(raw_output, b3_spec)
    transaction = None
    session_after = session
    witness_prompt = None
    camera_visible = False
    if normalization["normalization_status"] == "normalized":
        transaction = bridge.execute_scripter_transaction(
            case,
            session,
            request,
            normalization["program_text"],
        )
        if session != session_before:
            raise LiveScripterError("transaction execution mutated session")
        camera_visible = (
            cfg["application"]["private_asset_id"]
            in transaction["visible_asset_ids_after"]
        )
        if camera_visible and cfg["application"]["apply_if_visible"]:
            session_after = bridge.apply_transaction_evidence(
                case,
                session,
                request,
                transaction,
                cfg["application"]["witness_id"],
                cfg["application"]["private_asset_id"],
            )
            witness_prompt = session_api.build_prompt(
                case,
                session_after,
                cfg["application"]["witness_id"],
                "What does the newly presented archive excerpt show?",
            )

    if normalization["normalization_status"] != "normalized":
        outcome = "native_or_normalization_rejected"
    elif camera_visible and witness_prompt is not None:
        outcome = "normalized_executed_grant_applied"
    elif transaction is not None:
        outcome = "normalized_executed_no_grant"
    else:
        raise LiveScripterError("invalid 003D outcome state")

    head_sha = (
        os.environ.get("DETECTIVE_HEAD_SHA")
        or os.environ.get("GITHUB_SHA")
        or "local"
    )
    run_id = os.environ.get("GITHUB_RUN_ID")
    result = {
        "schema": "detective-003d-sprocket-live-run-v1",
        "experiment": "DETECTIVE-003D",
        "issue": 146,
        "status": "integration_pass",
        "outcome": outcome,
        "repository": {
            "head_sha": head_sha,
            "run_id": run_id,
            "run_url": (
                f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{run_id}"
                if run_id and os.environ.get("GITHUB_REPOSITORY") else None
            ),
        },
        "source": source,
        "request_receipt": request,
        "publisher_turns": turns,
        "publisher_prompt_text": prompt_text,
        "prompt_token_count": len(ids_list),
        "prompt_token_sha256": hashlib.sha256(
            torch.tensor(ids_list, dtype=torch.int64).numpy().tobytes()
        ).hexdigest(),
        "generation": {
            "generated_token_ids": tail,
            "generated_tokens": len(tail),
            "raw_output_with_special_tokens": raw_output,
            "generation_seconds": generation_seconds,
            "completion_tokens_per_second": (
                len(tail) / generation_seconds if generation_seconds else None
            ),
            "hit_max_new_tokens": len(tail) == cfg["prompt"]["max_new_tokens"],
        },
        "normalization": normalization,
        "exact_probe_selected": exact_probe_selected(cfg, normalization),
        "transaction": transaction,
        "camera_cache_visible_after_transaction": camera_visible,
        "evidence_application": {
            "applied": witness_prompt is not None,
            "selected_asset_id": (
                cfg["application"]["private_asset_id"]
                if witness_prompt is not None else None
            ),
            "witness_id": (
                cfg["application"]["witness_id"]
                if witness_prompt is not None else None
            ),
        },
        "final_session": session_after,
        "witness_prompt_after_application": witness_prompt,
        "case_sha256": hashlib.sha256(canonical(case).encode("utf-8")).hexdigest(),
        "environment": {
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
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "decision": {
            "correct_tool_selection_required": False,
            "exact_probe_selected": exact_probe_selected(cfg, normalization),
            "access_granted": camera_visible,
            "evidence_applied": witness_prompt is not None,
            "host_execution": False,
            "training_admitted": False,
            "body_selection_made": False,
            "relevance_mapping_accessed": False,
            "learner_final_accessed": False,
            "population_expansion_admitted": False,
        },
    }
    dataset = archive_dataset(out_root, result, cfg)

    del model
    gc.collect()
    return {
        "dataset": str(dataset),
        "status": result["status"],
        "outcome": outcome,
        "generated_tokens": len(tail),
        "native_parse_status": normalization["native_parse_status"],
        "normalization_status": normalization["normalization_status"],
        "exact_probe_selected": result["exact_probe_selected"],
        "camera_cache_visible": camera_visible,
        "evidence_applied": witness_prompt is not None,
        "optimizer_steps": 0,
        "population_expansion_admitted": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publisher-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(
        live_run(args.publisher_dir, args.out),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
