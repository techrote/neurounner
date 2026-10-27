#!/usr/bin/env python3
"""DETECTIVE-001B3: publisher-wire Sprocket root Instruct fictional scripter smoke.

No training. No model-authored host process/file/network execution. The model may
only propose a publisher-native tool call, which is parsed into DETECTIVE-001A's
typed fictional IR and evaluated by the deterministic case authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.detective_001_contract import ContractError, execute_program, parse_program  # noqa: E402

SPEC = ROOT / "experiments/detective-001/sprocket-scripter.json"
CASE = ROOT / "experiments/detective-001/pilot-case.json"
DATASET_NAME = "detective-001-sprocket-scripter"

ROLE = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "memory": "<|memory_read|>",
    "tool": "<|tool_result|>",
}
CONTROL = (
    "<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>",
    "<|end|>", "<|tool_call|>", "<|tool_result|>", "<think>", "</think>",
    "<|memory_read|>", "<|memory_write|>",
)
ACTION_MARKERS = ("<|tool_call|>", "<|memory_read|>", "<|memory_write|>")
CONTROL_RX = re.compile("|".join(re.escape(x) for x in CONTROL))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ContractError("duplicate JSON key")
        out[key] = value
    return out


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ContractError("non-finite JSON number")
    return number


def _constant(value):
    raise ContractError("non-finite JSON literal: " + value)


def load_spec() -> tuple[dict, dict]:
    spec, case = read_json(SPEC), read_json(CASE)
    if (spec.get("schema"), spec.get("experiment"), spec.get("parent_issue")) != (
        "detective-sprocket-scripter-v1", "DETECTIVE-001B3", 106
    ):
        raise ContractError("wrong experiment identity")
    model = spec["model"]
    expected_model = {
        "repo": "kandivault/sprocket-500m",
        "revision": "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
        "variant": "root-instruct",
        "weight_file": "model.safetensors",
        "weight_bytes": 1002207984,
        "weight_sha256": "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
        "tokenizer_sha256": "178a63d94a7de4aa273a6a9c218180c3b7d1912f5a18a91be34bab252f7277eb",
        "tokenizer_config_sha256": "1d1c0733848e0c78a2ec48a7f20cd55ec5ee0b4cbcda9785249af9e489823408",
        "special_tokens_map_sha256": "c575235775a1998faf1256ea0b6f791116ea9d89cc34238ea3bd532b8190773b",
        "context": 2048,
        "chat_checkpoint_forbidden": "chat/model.safetensors",
    }
    if model != expected_model:
        raise ContractError("Sprocket root Instruct pin drift")
    publisher = spec["publisher"]
    if (publisher["repo"] != "kandivault-source/Sprocket-500M"
        or publisher["commit"] != "a003767e70d766bc14b40c89792543ee8641c8bc"
        or publisher["sft_git_blob"] != "90bddf8976b9d6410fe7c5c1f084708e1cebab73"
        or publisher["harvest_git_blob"] != "97811aa4245ad3f05f66ce0c2d8aa8c353619267"
        or publisher["source_is_training_build_attestation"] is not False):
        raise ContractError("publisher source contract drift")
    if spec["special_token_ids"] != {
        "<|endoftext|>": 0, "<|pad|>": 1, "<|system|>": 2, "<|user|>": 3,
        "<|assistant|>": 4, "<|end|>": 5, "<|tool_call|>": 6,
        "<|tool_result|>": 7, "<think>": 8, "</think>": 9,
        "<|memory_read|>": 10, "<|memory_write|>": 11,
    }:
        raise ContractError("special token identity drift")
    if spec["case_path"] != CASE.relative_to(ROOT).as_posix():
        raise ContractError("case path changed")
    if spec["authority"] != {
        "initially_authorized_assets": ["dock-clock-ledger", "transfer-card"],
        "allowed_tool_names": ["read_asset", "correlate_assets", "probe_terminal"],
        "host_executes_only_typed_detective_ir": True,
        "arbitrary_host_execution": False,
    }:
        raise ContractError("authority widened")
    if [t["id"] for t in spec["tasks"]] != ["S01", "S02", "S03", "S04", "S05"]:
        raise ContractError("task suite changed")
    if spec["decode"] != {
        "seed": 24301, "do_sample": False, "max_new_tokens": 96,
        "max_prompt_tokens": 768, "eos_token_id": 5
    }:
        raise ContractError("decode budget changed")
    if spec["limits"] != {
        "tasks": 5, "max_generated_tokens": 480, "max_threads": 4,
        "max_job_minutes": 20, "paid_api_calls": 0, "training_steps": 0,
        "output_bytes": 160000,
    }:
        raise ContractError("resource budget changed")
    hc = spec["historical_control"]
    if (hc["source_dataset"] != "datasets/detective-001-smoke"
        or hc["source_run_id"] != 40579442013
        or hc["strict_script_attempts"] != 2
        or hc["strict_script_valid"] != 0
        or hc["comparison_is_contextual_not_matched"] is not True):
        raise ContractError("historical control relabeled")
    if spec["acceptance"]["no_training"] is not True:
        raise ContractError("training admission not allowed")
    return spec, case


def manifest() -> str:
    tools = [
        {
            "name": "read_asset",
            "description": "Read a fictional detective-case evidence record that the game currently permits.",
            "parameters": {
                "type": "object",
                "properties": {"asset_id": {"type": "string"}},
                "required": ["asset_id"],
            },
        },
        {
            "name": "correlate_assets",
            "description": "Compare two currently permitted fictional evidence records for shared canonical fact IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "left_id": {"type": "string"},
                    "right_id": {"type": "string"},
                },
                "required": ["left_id", "right_id"],
            },
        },
        {
            "name": "probe_terminal",
            "description": "Attempt a fictional terminal access using a credential supplied by the player. Never invent a real credential.",
            "parameters": {
                "type": "object",
                "properties": {
                    "terminal_id": {"type": "string"},
                    "credential": {"type": "string"},
                },
                "required": ["terminal_id", "credential"],
            },
        },
    ]
    return "You have access to these tools:\n" + "\n".join(
        json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in tools
    )


def _validate_host_content(text: str) -> None:
    if (not isinstance(text, str) or not text.strip() or "\x00" in text or "\r" in text):
        raise ContractError("invalid Sprocket turn content")
    if CONTROL_RX.search(text):
        raise ContractError("host content attempts raw control-token injection")


def validate_turns(turns: list[dict]) -> None:
    if not isinstance(turns, list) or not 1 <= len(turns) <= 12:
        raise ContractError("bounded Sprocket turns required")
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
            raise ContractError("turn field mismatch")
        role, text = turn["role"], turn["content"]
        if role not in ROLE:
            raise ContractError("unknown Sprocket role")
        if role == "assistant":
            if not isinstance(text, str) or not text.strip():
                raise ContractError("empty assistant source turn")
            matches = [x.group() for x in CONTROL_RX.finditer(text)]
            actions = [x for x in matches if x in ACTION_MARKERS]
            if len(actions) != 1 or len(matches) != 1 or not text.startswith(actions[0]):
                raise ContractError("synthetic assistant action has invalid control placement")
            if actions[0] != "<|tool_call|>":
                raise ContractError("this battery injects tool calls only")
            parse_native_tool_call(text)
        else:
            _validate_host_content(text)
    if turns[-1]["role"] == "assistant":
        raise ContractError("prompt must end on a host turn before assistant cue")


def render_text(turns: list[dict]) -> str:
    validate_turns(turns)
    return "".join(ROLE[t["role"]] + t["content"] + "<|end|>" for t in turns) + "<|assistant|>"


def verify_tokenizer(tok, spec: dict) -> None:
    for marker, expected in spec["special_token_ids"].items():
        try:
            converted = tok.convert_tokens_to_ids(marker)
            encoded = tok.encode(marker, add_special_tokens=False)
            decoded = tok.convert_ids_to_tokens(expected)
        except Exception as exc:
            raise ContractError("special token lookup failed: " + marker) from exc
        if converted != expected or encoded != [expected] or decoded != marker:
            raise ContractError("special token identity mismatch: " + marker)


def prompt_ids(tok, spec: dict, turns: list[dict]) -> list[int]:
    text = render_text(turns)
    ids = tok.encode(text, add_special_tokens=False)
    if not ids or ids[-1] != spec["special_token_ids"]["<|assistant|>"]:
        raise ContractError("publisher prompt lacks dangling assistant cue")
    if len(ids) > spec["decode"]["max_prompt_tokens"]:
        raise ContractError("publisher prompt exceeds preregistered token budget")
    return ids


def parse_native_tool_call(text: str) -> dict:
    if not isinstance(text, str):
        raise ContractError("tool output must be text")
    value = text.strip()
    if value.endswith("<|end|>"):
        value = value[:-len("<|end|>")].rstrip()
    if "<|end|>" in value:
        raise ContractError("tool output contains premature end token")
    if not value.startswith("<|tool_call|>"):
        raise ContractError("assistant did not begin with native tool marker")
    if any(marker in value[len("<|tool_call|>"):]
           for marker in CONTROL if marker != "<|tool_call|>"):
        raise ContractError("unexpected control token after tool marker")
    payload = value[len("<|tool_call|>"):].strip()
    try:
        obj = json.loads(payload, object_pairs_hook=_pairs, parse_float=_finite,
                         parse_constant=_constant)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ContractError("native tool call JSON invalid") from exc
    if (not isinstance(obj, dict) or set(obj) != {"name", "arguments"}
        or not isinstance(obj["name"], str) or not obj["name"].strip()
        or not isinstance(obj["arguments"], dict)):
        raise ContractError("native tool call requires name and arguments only")
    return obj


def tool_call_text(call: dict) -> str:
    return "<|tool_call|>" + json.dumps(
        call, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def tool_call_to_program(call: dict, spec: dict) -> dict:
    name, args = call["name"], call["arguments"]
    if name not in spec["authority"]["allowed_tool_names"]:
        raise ContractError("tool not admitted by game")
    fields = {
        "read_asset": {"asset_id"},
        "correlate_assets": {"left_id", "right_id"},
        "probe_terminal": {"terminal_id", "credential"},
    }
    if set(args) != fields[name] or any(not isinstance(v, str) or not v for v in args.values()):
        raise ContractError("native tool arguments violate typed game IR")
    program = {"schema": "detective-script-v1", "steps": [{"op": name, **args}]}
    parse_program(program)
    return program


def execute_call(case: dict, spec: dict, call: dict) -> dict:
    program = tool_call_to_program(call, spec)
    return execute_program(
        case, program, spec["authority"]["initially_authorized_assets"]
    )


def result_payload(receipt: dict) -> str:
    events = receipt.get("events")
    if not isinstance(events, list) or len(events) != 1:
        raise ContractError("tool receipt must contain one event")
    return canonical(events[0]["result"])


def task_turns(spec: dict, case: dict, task: dict) -> list[dict]:
    turns = [{"role": "system", "content": manifest()},
             {"role": "user", "content": task["request"]}]
    if task["mode"] == "host_result":
        call = task["synthetic_prior_call"]
        receipt = execute_call(case, spec, call)
        turns.extend([
            {"role": "assistant", "content": tool_call_text(call)},
            {"role": "tool", "content": result_payload(receipt)},
        ])
    validate_turns(turns)
    return turns


def _matches_expected_args(got: dict, expected: dict) -> bool:
    if got == expected:
        return True
    # Correlation is symmetric in the current fictional interpreter.
    if set(got) == {"left_id", "right_id"} and set(expected) == set(got):
        return {got["left_id"], got["right_id"]} == {
            expected["left_id"], expected["right_id"]
        }
    return False


def score_task(spec: dict, case: dict, task: dict, raw: str) -> dict:
    base = {"task_id": task["id"], "family": task["family"],
            "mode": task["mode"], "raw_sha256": hashlib.sha256(
                raw.encode("utf-8")).hexdigest()}
    if task["mode"] == "model_call":
        try:
            call = parse_native_tool_call(raw)
            program = tool_call_to_program(call, spec)
            receipt = execute_call(case, spec, call)
            expected = (call["name"] == task["expected_tool"]
                        and _matches_expected_args(
                            call["arguments"], task["expected_arguments"]))
            status = receipt["events"][0]["result"]["status"]
            return {**base, "native_call_valid": True,
                    "tool_name": call["name"], "arguments": call["arguments"],
                    "task_selection_correct": expected,
                    "authority_status": status,
                    "typed_program": program, "receipt": receipt}
        except (ContractError, ValueError, TypeError) as exc:
            return {**base, "native_call_valid": False,
                    "task_selection_correct": False,
                    "error_type": type(exc).__name__, "error": str(exc)[:180]}
    if task["mode"] == "no_tool":
        stripped = raw.strip().removesuffix("<|end|>").rstrip()
        control = [m.group() for m in CONTROL_RX.finditer(stripped)]
        return {**base, "native_call_valid": None,
                "tool_restraint": bool(stripped) and "<|tool_call|>" not in stripped
                                  and not any(x in ACTION_MARKERS for x in control),
                "control_markers": control}
    stripped = raw.strip().removesuffix("<|end|>").rstrip()
    markers = [x.casefold() for x in task["expected_text_markers"]]
    forbidden = [x.casefold() for x in task["forbidden_text_markers"]]
    lower = stripped.casefold()
    return {**base, "native_call_valid": None,
            "result_use_success": bool(stripped)
                and "<|tool_call|>" not in stripped
                and all(x in lower for x in markers)
                and not any(x in lower for x in forbidden),
            "expected_marker_hits": [x for x in task["expected_text_markers"]
                                     if x.casefold() in lower],
            "forbidden_marker_hits": [x for x in task["forbidden_text_markers"]
                                      if x.casefold() in lower]}


def plan() -> dict:
    spec, case = load_spec()
    tasks = []
    for task in spec["tasks"]:
        turns = task_turns(spec, case, task)
        tasks.append({
            "id": task["id"], "family": task["family"], "mode": task["mode"],
            "turns": turns, "rendered_text_sha256": hashlib.sha256(
                render_text(turns).encode("utf-8")).hexdigest(),
        })
    return {
        "schema": "detective-sprocket-scripter-plan-v1",
        "spec_sha256": file_sha(SPEC), "case_sha256": file_sha(CASE),
        "model": spec["model"], "publisher": spec["publisher"],
        "tasks": tasks, "limits": spec["limits"],
        "historical_control": spec["historical_control"],
    }


def save_partial(out: Path, data: dict, cap: int) -> None:
    raw = canonical(data).encode("utf-8")
    if len(raw) > cap:
        raise ContractError("raw evidence byte cap exceeded")
    write_json(out / "raw" / "run.json", data)


def archive(out: Path, run: dict, spec: dict) -> Path:
    dataset = out / "dataset"
    (dataset / "raw").mkdir(parents=True, exist_ok=True)
    (dataset / "provenance").mkdir(parents=True, exist_ok=True)
    write_json(dataset / "raw/run.json", run)
    scores = run["results"]
    summary = {
        "tasks": len(scores),
        "native_tool_calls_valid": sum(x.get("native_call_valid") is True for x in scores),
        "tool_selection_correct": sum(x.get("task_selection_correct") is True for x in scores),
        "tool_restraint_success": sum(x.get("tool_restraint") is True for x in scores),
        "tool_result_or_denial_use_success": sum(x.get("result_use_success") is True for x in scores),
        "generated_tokens": run["total_generated_tokens"],
        "training_steps": 0,
    }
    write_json(dataset / "manifest.json", {
        "schema": "detective-sprocket-scripter-archive-v1",
        "experiment": "DETECTIVE-001B3",
        "admission": "EXPLORATORY_NOT_ADMITTED",
        "run_sha256": sha_obj(run), "spec_sha256": file_sha(SPEC),
        "plan_sha256": run["plan_sha256"],
        "source_head_sha": run["repository"]["head_sha"],
        "summary": summary,
    })
    write_json(dataset / "provenance/source-artifacts.json", {
        "repository": "techrote/neurounner",
        "head_sha": run["repository"]["head_sha"],
        "workflow_run_id": run["repository"]["run_id"],
        "workflow_run_url": run["repository"]["run_url"],
        "model": spec["model"], "publisher": spec["publisher"],
        "model_files": run["model_files"],
        "publisher_checkout": run["publisher_checkout"],
        "script_sha256": run["script_sha256"],
        "case_sha256": run["case_sha256"],
    })
    (dataset / "README.md").write_text(
        "# DETECTIVE-001B3 Sprocket Instruct scripter smoke\n\n"
        "Raw, source-bound root Sprocket-500M Instruct output against five "
        "fictional detective evidence-tool situations. Publisher-native "
        "<|tool_call|>/host <|tool_result|> wire is preserved. Calls are parsed "
        "into inert typed DETECTIVE-001A IR and executed only against the synthetic "
        "case authority; there is no model-authored shell, filesystem, network or "
        "real credential action. This is a development smoke, not final scripting "
        "or NPC admission. The public publisher source link is not treated as a "
        "cryptographic training-build attestation. Negative outputs remain raw.\n",
        encoding="utf-8",
    )
    inventory = []
    for item in sorted(x for x in dataset.rglob("*") if x.is_file()):
        rel = item.relative_to(dataset).as_posix()
        if rel != "provenance/hashes.sha256":
            inventory.append(f"{file_sha(item)}  {rel}")
    (dataset / "provenance/hashes.sha256").write_text(
        "\n".join(inventory) + "\n", encoding="utf-8"
    )
    return dataset


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def live_run(out: Path, publisher_dir: Path) -> dict:
    spec, case = load_spec()
    p = plan()
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import transformers

    torch.set_num_threads(min(spec["limits"]["max_threads"], os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(spec["decode"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Download exact root Instruct source files. Chat subfolder is never requested.
    files = {}
    for name, expected_hash in (
        ("config.json", None),
        ("tokenizer.json", spec["model"]["tokenizer_sha256"]),
        ("tokenizer_config.json", spec["model"]["tokenizer_config_sha256"]),
        ("special_tokens_map.json", spec["model"]["special_tokens_map_sha256"]),
        (spec["model"]["weight_file"], spec["model"]["weight_sha256"]),
    ):
        path = Path(hf_hub_download(spec["model"]["repo"], filename=name,
                                    revision=spec["model"]["revision"]))
        digest = file_sha(path)
        if expected_hash and digest != expected_hash:
            raise ContractError(f"source file SHA mismatch: {name}")
        if name == spec["model"]["weight_file"] and path.stat().st_size != spec["model"]["weight_bytes"]:
            raise ContractError("root Instruct checkpoint byte count mismatch")
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}

    pub_head = os.popen(f"git -C {publisher_dir} rev-parse HEAD").read().strip()
    if pub_head != spec["publisher"]["commit"]:
        raise ContractError("publisher checkout is not the pinned source")
    pub_blobs = {}
    for path, expected in (
        (spec["publisher"]["sft_path"], spec["publisher"]["sft_git_blob"]),
        (spec["publisher"]["harvest_path"], spec["publisher"]["harvest_git_blob"]),
    ):
        blob = os.popen(f"git -C {publisher_dir} hash-object {path}").read().strip()
        if blob != expected:
            raise ContractError("publisher source blob drift: " + path)
        pub_blobs[path] = blob

    tok = AutoTokenizer.from_pretrained(
        spec["model"]["repo"], revision=spec["model"]["revision"],
        local_files_only=True, trust_remote_code=False
    )
    verify_tokenizer(tok, spec)
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"]["repo"], revision=spec["model"]["revision"],
        local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32
    )
    model.config.use_cache = True
    model.eval()

    source_sha = os.environ.get("DETECTIVE_HEAD_SHA") or os.environ.get("GITHUB_SHA") or "local"
    run_id = os.environ.get("GITHUB_RUN_ID")
    run = {
        "schema": "detective-sprocket-scripter-run-v1",
        "experiment": spec["experiment"], "admission": "EXPLORATORY_NOT_ADMITTED",
        "complete": False, "plan_sha256": sha_obj(p), "spec_sha256": file_sha(SPEC),
        "case_sha256": file_sha(CASE), "script_sha256": file_sha(Path(__file__)),
        "repository": {
            "head_sha": source_sha, "run_id": run_id,
            "run_url": f"https://github.com/techrote/neurounner/actions/runs/{run_id}" if run_id else None,
        },
        "model": spec["model"], "model_files": files,
        "publisher": spec["publisher"],
        "publisher_checkout": {"commit": pub_head, "git_blobs": pub_blobs},
        "runtime": {
            "platform": platform.platform(), "cpu": cpu_model(),
            "threads": torch.get_num_threads(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
        },
        "raw_generations": [], "results": [], "total_generated_tokens": 0,
    }
    save_partial(out, run, spec["limits"]["output_bytes"])

    for task in spec["tasks"]:
        turns = task_turns(spec, case, task)
        ids_list = prompt_ids(tok, spec, turns)
        ids = torch.tensor([ids_list], dtype=torch.long)
        if run["total_generated_tokens"] + spec["decode"]["max_new_tokens"] > spec["limits"]["max_generated_tokens"]:
            raise ContractError("decode budget would be exceeded")
        started = time.monotonic()
        with torch.inference_mode():
            generated_all = model.generate(
                ids, max_new_tokens=spec["decode"]["max_new_tokens"],
                do_sample=False, use_cache=True,
                eos_token_id=spec["decode"]["eos_token_id"],
                pad_token_id=spec["special_token_ids"]["<|pad|>"],
            )
        elapsed = time.monotonic() - started
        tail = generated_all[0, ids.shape[-1]:].tolist()
        decoded = tok.decode(tail, skip_special_tokens=False)
        score = score_task(spec, case, task, decoded)
        raw = {
            "task_id": task["id"], "family": task["family"], "mode": task["mode"],
            "turns": turns, "rendered_prompt_sha256": hashlib.sha256(
                render_text(turns).encode("utf-8")).hexdigest(),
            "prompt_token_ids_sha256": hashlib.sha256(
                torch.tensor(ids_list, dtype=torch.int64).numpy().tobytes()
            ).hexdigest(),
            "prompt_tokens": len(ids_list),
            "generated_token_ids": tail, "generated_tokens": len(tail),
            "decoded_with_special_tokens": decoded,
            "elapsed_seconds": elapsed,
        }
        run["raw_generations"].append(raw)
        run["results"].append(score)
        run["total_generated_tokens"] += len(tail)
        save_partial(out, run, spec["limits"]["output_bytes"])

    if len(run["results"]) != 5 or len(run["raw_generations"]) != 5:
        raise ContractError("incomplete scripted battery")
    run["runtime"]["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    run["complete"] = True
    save_partial(out, run, spec["limits"]["output_bytes"])
    dataset = archive(out, run, spec)
    return {
        "dataset": str(dataset), "tasks": 5,
        "generated_tokens": run["total_generated_tokens"],
        "native_calls_valid": sum(x.get("native_call_valid") is True for x in run["results"]),
        "tool_selections_correct": sum(x.get("task_selection_correct") is True for x in run["results"]),
        "restraint_success": sum(x.get("tool_restraint") is True for x in run["results"]),
        "result_use_success": sum(x.get("result_use_success") is True for x in run["results"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "run"], default="check")
    parser.add_argument("--out", type=Path, default=Path("out/detective-001b3"))
    parser.add_argument("--publisher-dir", type=Path, default=Path("publisher"))
    args = parser.parse_args()
    payload = plan() if args.mode == "check" else live_run(args.out, args.publisher_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
