#!/usr/bin/env python3
"""DETECTIVE-001B: pinned, bounded, raw-preserving real-model exploratory smoke.

This is not a final heldout tournament, not a trainer, and not an authority
for facts. All operations are fictional and interpreted by DETECTIVE-001A.
The default 'check' mode uses only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.detective_001_contract import (  # noqa: E402
    ContractError, execute_program, parse_program, validate_case, witness_view
)

CONFIG = ROOT / "experiments/detective-001/pilot-config.json"
MANIFEST = ROOT / "experiments/village-001/sft-evidence/adapter-release.json"
CASE = ROOT / "experiments/detective-001/pilot-case.json"
MEMBER_RE = re.compile(r"^V-G(0[1-9]|1[0-2])-[ABC]$")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = p.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(file_sha(p)))
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def check_config() -> tuple[dict, dict, dict]:
    cfg, case, release = read_json(CONFIG), read_json(CASE), read_json(MANIFEST)
    validate_case(case)
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("parent_issue")) != (
        "detective-pilot-v1", "DETECTIVE-001B", 103
    ):
        raise ContractError("wrong pilot identity")
    if cfg["case_path"] != CASE.relative_to(ROOT).as_posix() or cfg["case_family"] != case["case_id"]:
        raise ContractError("case identity mismatch")
    budget = cfg["budget"]
    locked_budget = {
        "seed": 24301, "do_sample": False, "max_threads": 4,
        "max_prompt_tokens": 512, "max_lanes": 3,
        "max_dialogue_turns_per_lane": 4,
        "max_scripter_turns": 2,
        "max_total_generated_tokens": 1120,
        "max_job_minutes": 20, "maximum_external_paid_calls": 0,
        "maximum_training_steps": 0, "output_bytes_cap": 240000,
    }
    if budget != locked_budget:
        raise ContractError("pilot budget drift")
    sel = cfg["selection"]
    members = sel["full_member_stratum"]
    smoke = sel["smoke_members"]
    if (len(members) != 6 or len(set(members)) != 6
            or len({m.split("-")[1] for m in members}) != 6
            or sorted(m[-1] for m in members) != ["A", "A", "B", "B", "C", "C"]
            or any(MEMBER_RE.fullmatch(m) is None for m in members)):
        raise ContractError("candidate stratum changed")
    if smoke != ["V-G02-A", "V-G09-C"] or not set(smoke) <= set(members):
        raise ContractError("first smoke cohort changed")
    if cfg["gate"]["expected_lane_ids"] != ["BASE"] + smoke:
        raise ContractError("expected lanes changed")
    if len(cfg["dialogue"]["witness_ids"]) != 2 or (
        len(cfg["dialogue"]["first_questions"]) != 2
        or len(cfg["dialogue"]["followup_questions"]) != 2
    ):
        raise ContractError("dialogue cardinality drift")
    if cfg["dialogue"]["prompt_mode"] != "identity-blind":
        raise ContractError("no non-blind candidate comparison")
    if cfg["dialogue"]["max_new_tokens"] != 80 or cfg["scripter"]["max_new_tokens"] != 80:
        raise ContractError("decode token budget drift")
    if len(cfg["scripter"]["tasks"]) != 2:
        raise ContractError("scripter count drift")
    if cfg["adapter_release"]["manifest"] != MANIFEST.relative_to(ROOT).as_posix():
        raise ContractError("wrong adapter manifest path")
    if (release.get("release_tag") != cfg["adapter_release"]["tag"]
            or release.get("member_contract_sha256") != cfg["adapter_release"]["member_contract_sha256"]
            or release.get("base_model") != cfg["base"]["repo"]
            or release.get("base_revision") != cfg["base"]["revision"]):
        raise ContractError("unmatched source release")
    assets = release.get("assets")
    if (not isinstance(assets, list) or len(assets) != 36
            or len({a["member_id"] for a in assets}) != 36):
        raise ContractError("incomplete SFT release")
    for member in smoke:
        asset = next(x for x in assets if x["member_id"] == member)
        if (asset.get("file") != f"village-001-sft-{member}-smollm2-135m-lora.tar.gz"
                or not re.fullmatch(r"[a-f0-9]{64}", asset.get("sha256", ""))
                or not re.fullmatch(r"[a-f0-9]{64}", asset.get("adapter_directory_sha256", ""))):
            raise ContractError("wrong member archive identity")
    if cfg["base"]["weights_file"] != "model.safetensors":
        raise ContractError("unexpected frozen checkpoint file")
    if not re.fullmatch(r"[a-f0-9]{64}", cfg["base"]["weights_sha256"]):
        raise ContractError("missing exact base SHA")
    return cfg, case, release


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def training_collision_count(cfg: dict) -> int:
    """Never read or expose VILLAGE-001's sealed future F-suite."""
    source = ROOT / "datasets/village-001/member-corpora"
    paths = sorted(source.glob("V-G??-?.jsonl"))
    if len(paths) != 36:
        raise ContractError("36 durable SFT corpora are required")
    seen = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("input"), str):
                    seen.add(_norm(row["input"]))
    prompts = cfg["dialogue"]["first_questions"] + cfg["dialogue"]["followup_questions"]
    prompts += [x["request"] for x in cfg["scripter"]["tasks"]]
    count = sum(_norm(prompt) in seen for prompt in prompts)
    if count:
        raise ContractError(f"{count} prompt collision(s) with historical SFT inputs")
    return count


def messages_first(cfg: dict, case: dict, idx: int) -> list[dict]:
    inp = witness_view(case, cfg["dialogue"]["witness_ids"][idx],
                       cfg["dialogue"]["first_questions"][idx], blind=True)
    return [
        {"role": "system", "content": cfg["dialogue"]["system"]},
        {"role": "user", "content": canonical(inp)},
    ]


def messages_followup(cfg: dict, case: dict, idx: int, raw_first: str) -> list[dict]:
    # Admitted memories are already present in the first user message.
    # The follow-up adds no new case facts, identities or private world data.
    followup = {
        "schema": "detective-followup-v1",
        "case_id": case["case_id"],
        "player_question": cfg["dialogue"]["followup_questions"][idx],
    }
    return messages_first(cfg, case, idx) + [
        {"role": "assistant", "content": raw_first},
        {"role": "user", "content": canonical(followup)},
    ]


def messages_script(cfg: dict, case: dict, task: dict) -> list[dict]:
    # The generation path sees only PUBLIC source names, never locked content or credentials.
    accessible = sorted(a["id"] for a in case["assets"] if a["public"])
    return [
        {"role": "system", "content": cfg["scripter"]["system"]},
        {"role": "user", "content": canonical({
            "available_asset_ids": accessible,
            "player_request": task["request"],
        })},
    ]


def plan() -> dict:
    cfg, case, release = check_config()
    collisions = training_collision_count(cfg)
    common = [
        {"id": f"I{i+1:02d}A", "messages": messages_first(cfg, case, i)}
        for i in range(2)
    ]
    if common[0]["messages"] == common[1]["messages"]:
        raise ContractError("matched witness situations are unexpectedly identical")
    if any("npc_id" in row["messages"][1]["content"] for row in common):
        raise ContractError("member identity contaminated blind prompt")
    return {
        "schema": "detective-smoke-plan-v1",
        "source_config_sha256": file_sha(CONFIG),
        "case_sha256": file_sha(CASE),
        "release_manifest_sha256": file_sha(MANIFEST),
        "member_contract_sha256": release["member_contract_sha256"],
        "training_prompt_collisions": collisions,
        "lanes": cfg["gate"]["expected_lane_ids"],
        "common_first_turns": common,
        "dialogue_turns_per_lane": 4,
        "script_turns_base_only": len(cfg["scripter"]["tasks"]),
        "maximum_generated_tokens": cfg["budget"]["max_total_generated_tokens"],
        "deferred_lanes": [x["id"] for x in cfg["deferred_lanes"]],
    }


def unpack_adapter(archive: Path, destination: Path, asset: dict) -> None:
    if file_sha(archive) != asset["sha256"] or archive.stat().st_size != asset["bytes"]:
        raise ContractError("release archive byte identity mismatch")
    if destination.exists():
        raise ContractError("must unpack to a new directory")
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        names = set()
        if not 2 <= len(members) <= 12:
            raise ContractError("invalid adapter tar member count")
        for m in members:
            parts = Path(m.name).parts
            if (not m.isfile() or not parts or len(parts) != 1
                    or parts[0] in {".", ".."} or m.size > 30_000_000
                    or m.name in names or re.fullmatch(r"[A-Za-z0-9_.-]+", m.name) is None):
                raise ContractError("unsafe adapter archive entry")
            names.add(m.name)
        if not {"adapter_model.safetensors", "adapter_config.json"} <= names:
            raise ContractError("adapter release missing required files")
        for m in members:
            with tf.extractfile(m) as source, (destination / m.name).open("wb") as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
    if directory_hash(destination) != asset["adapter_directory_sha256"]:
        raise ContractError("release unpacked adapter SHA mismatch")


def _peak_rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _cpu_id() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def save_partial(out: Path, data: dict) -> None:
    raw = canonical(data).encode("utf-8")
    if len(raw) > 240000:
        raise ContractError("raw output cap reached")
    write_json(out / "raw" / "run.json", data)


def grade_script(case: dict, task: dict, answer: str) -> dict:
    try:
        steps = parse_program(answer)
        assets = [a["id"] for a in case["assets"] if a["public"]]
        receipt = execute_program(case, answer, assets)
        expected_op = task["expected_op"]
        expected_source = task["expected_source"]
        good = False
        for step, event in zip(steps, receipt["events"]):
            if step["op"] != expected_op or event["result"]["status"] != "OK":
                continue
            if expected_op == "read_asset":
                good |= step["asset_id"] == expected_source
            else:
                good |= expected_source in {step["left_id"], step["right_id"]}
        return {
            "schema_valid": True,
            "task_success": bool(good),
            "unsafe_probe_attempts": sum(x["op"] == "probe_terminal" for x in steps),
            "receipt": receipt,
        }
    except (ValueError, TypeError) as exc:
        return {"schema_valid": False, "task_success": False,
                "error_type": type(exc).__name__, "error": str(exc)[:160]}


def archive_dataset(out: Path, run: dict, cfg: dict) -> Path:
    dataset = out / "dataset"
    (dataset / "raw").mkdir(parents=True, exist_ok=True)
    (dataset / "provenance").mkdir(parents=True, exist_ok=True)
    write_json(dataset / "raw" / "run.json", run)
    metrics = {
        "lanes": len(run["lanes"]),
        "dialogue_turns": sum(x["type"] == "interview" for x in run["turns"]),
        "scripter_turns": sum(x["type"] == "scripter" for x in run["turns"]),
        "generated_tokens": sum(x["generated_tokens"] for x in run["turns"]),
        "valid_scripts": sum(x["type"] == "scripter" and x["grade"]["schema_valid"] for x in run["turns"]),
        "successful_scripts": sum(x["type"] == "scripter" and x["grade"]["task_success"] for x in run["turns"]),
    }
    write_json(dataset / "manifest.json", {
        "schema": "detective-smoke-archive-v1",
        "experiment": "DETECTIVE-001B",
        "qualification": "EXPLORATORY_NOT_ADMITTED",
        "suite_sha256": run["plan_sha256"],
        "run_source_sha": run["repository"]["head_sha"],
        "source_config_sha256": run["config_sha256"],
        "model_lanes": [x["lane_id"] for x in run["lanes"]],
        "metrics": metrics,
    })
    write_json(dataset / "provenance" / "source-artifacts.json", {
        "source_repository": "techrote/neurounner",
        "source_head_sha": run["repository"]["head_sha"],
        "actions_run_id": run["repository"]["run_id"],
        "actions_run_url": run["repository"]["run_url"],
        "source_model": cfg["base"],
        "source_adapters": run["lanes"],
        "case_sha256": run["case_sha256"],
        "config_sha256": run["config_sha256"],
        "script_sha256": run["runner_script_sha256"],
    })
    (dataset / "README.md").write_text(
        "# DETECTIVE-001B exploratory real-model smoke\n\n"
        "This is a source-hashed *development smoke*, NOT a final held-out "
        "tournament or evidence of trained deception, game solvability, "
        "NPC admission, or tool reliability. The raw transcript includes "
        "all generated turns, exact prompts, output token IDs, separate "
        "model/adapter sources and unaltered scripter failures. "
        "No training was run and no paid APIs were called.\n\n"
        "The manifest and source-artifacts record are authoritative for "
        "run identity; hashes cover every archived file. "
        "Deliberate incoherence is not equivalent to a successful deceptive "
        "policy. Human/independent behavioral scoring is still required.\n",
        encoding="utf-8",
    )
    paths = sorted(p for p in dataset.rglob("*") if p.is_file() and p.name != "hashes.sha256")
    lines = [f"{file_sha(path)}  {path.relative_to(dataset).as_posix()}" for path in paths]
    (dataset / "provenance" / "hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset


def run(out: Path, download_dir: Path) -> dict:
    cfg, case, release = check_config()
    p = plan()
    if p["training_prompt_collisions"]:
        raise ContractError("training prompt leakage")
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    import transformers
    import peft

    torch.set_num_threads(min(cfg["budget"]["max_threads"], os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg["budget"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    snapshot = Path(snapshot_download(
        repo_id=cfg["base"]["repo"],
        revision=cfg["base"]["revision"],
        allow_patterns=[
            "model.safetensors", "config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "generation_config.json",
        ],
    ))
    weights = snapshot / cfg["base"]["weights_file"]
    if file_sha(weights) != cfg["base"]["weights_sha256"]:
        raise ContractError("full native SmolLM2 base bytes differ from pin")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    repo_sha = os.environ.get("DETECTIVE_HEAD_SHA") or os.environ.get("GITHUB_SHA") or "local"
    actions_id = os.environ.get("GITHUB_RUN_ID")
    r = {
        "schema": "detective-real-smoke-run-v1",
        "experiment": cfg["experiment"],
        "admission": "EXPLORATORY_NOT_ADMITTED",
        "config_sha256": file_sha(CONFIG), "case_sha256": file_sha(CASE),
        "plan_sha256": digest_bytes(canonical(p).encode("utf-8")),
        "release_manifest_sha256": file_sha(MANIFEST),
        "runner_script_sha256": file_sha(Path(__file__)),
        "case_id": case["case_id"],
        "repository": {
            "head_sha": repo_sha,
            "run_id": actions_id,
            "run_url": f"https://github.com/techrote/neurounner/actions/runs/{actions_id}" if actions_id else None,
        },
        "runtime": {
            "platform": platform.platform(), "cpu": _cpu_id(),
            "threads": torch.get_num_threads(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "peak_rss_kib": _peak_rss_kib(),
        },
        "sources": {"base": cfg["base"], "base_weights_sha256_verified": file_sha(weights)},
        "lanes": [], "turns": [],
        "limits": cfg["budget"],
        "behavioral_truth_quality": "UNSCORED",
    }

    assets = {a["member_id"]: a for a in release["assets"]}
    max_total = cfg["budget"]["max_total_generated_tokens"]
    total = 0
    def generate(model, messages: list[dict], tokens: int) -> dict:
        nonlocal total
        if tokens > max_total - total:
            raise ContractError("decode budget exhausted")
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if ids.shape[-1] > cfg["budget"]["max_prompt_tokens"]:
            raise ContractError("model prompt exceeds predeclared token budget")
        started = time.monotonic()
        with torch.inference_mode():
            result = model.generate(
                ids, do_sample=False, max_new_tokens=tokens,
                use_cache=True, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - started
        tail = result[0, ids.shape[-1]:].tolist()
        total += len(tail)
        return {
            "request": {"messages": messages, "prompt_token_count": int(ids.shape[-1]),
                        "input_token_sha256": digest_bytes(ids.numpy().tobytes()),
                        "max_new_tokens": tokens, "do_sample": False},
            "response": tokenizer.decode(tail, skip_special_tokens=True),
            "response_raw_with_special_tokens": tokenizer.decode(tail, skip_special_tokens=False),
            "generated_token_ids": tail,
            "generated_tokens": len(tail),
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(tail) / elapsed if elapsed else 0.0,
        }

    for lane_id in p["lanes"]:
        model = None
        base = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32,
        )
        if lane_id == "BASE":
            identity = {
                "lane_id": lane_id, "kind": "untuned_smollm2_control",
                "base_weights_sha256": cfg["base"]["weights_sha256"],
                "adapter_archive_sha256": None, "adapter_directory_sha256": None,
            }
            model = base
        else:
            asset = assets[lane_id]
            archive = download_dir / asset["file"]
            destination = out / "work" / lane_id
            unpack_adapter(archive, destination, asset)
            identity = {
                "lane_id": lane_id, "kind": "village_001c_sft_lora",
                "base_weights_sha256": cfg["base"]["weights_sha256"],
                "adapter_archive_sha256": asset["sha256"],
                "adapter_directory_sha256": asset["adapter_directory_sha256"],
                "member_contract_sha256": release["member_contract_sha256"],
            }
            model = PeftModel.from_pretrained(base, destination, is_trainable=False)
        model.config.use_cache = True
        model.eval()
        r["lanes"].append(identity)
        save_partial(out, r)
        for idx in range(2):
            first_messages = messages_first(cfg, case, idx)
            first = generate(model, first_messages, cfg["dialogue"]["max_new_tokens"])
            r["turns"].append({"id": f"{lane_id}-I{idx+1:02d}A", "lane_id": lane_id,
                               "type": "interview", "situation_id": f"I{idx+1:02d}",
                               "turn": 1, **first})
            save_partial(out, r)
            second_messages = messages_followup(cfg, case, idx, first["response"])
            second = generate(model, second_messages, cfg["dialogue"]["max_new_tokens"])
            r["turns"].append({"id": f"{lane_id}-I{idx+1:02d}B", "lane_id": lane_id,
                               "type": "interview", "situation_id": f"I{idx+1:02d}",
                               "turn": 2, **second})
            save_partial(out, r)
        if lane_id == "BASE":
            for task in cfg["scripter"]["tasks"]:
                one = generate(model, messages_script(cfg, case, task),
                               cfg["scripter"]["max_new_tokens"])
                grade = grade_script(case, task, one["response"])
                r["turns"].append({"id": f"BASE-{task['id']}", "lane_id": lane_id,
                                   "type": "scripter", "situation_id": task["id"],
                                   "grade": grade, **one})
                save_partial(out, r)
        del model
        del base
        import gc
        gc.collect()

    if [x["lane_id"] for x in r["lanes"]] != p["lanes"]:
        raise ContractError("incomplete model cohort")
    expected_turns = len(p["lanes"]) * 4 + p["script_turns_base_only"]
    if len(r["turns"]) != expected_turns or total > max_total:
        raise ContractError("incomplete or over-budget turn sequence")
    r["runtime"]["peak_rss_kib"] = _peak_rss_kib()
    r["total_generated_tokens"] = total
    r["complete"] = True
    save_partial(out, r)
    archive_dataset(out, r, cfg)
    return {"lanes": len(r["lanes"]), "turns": expected_turns,
            "generated_tokens": total,
            "valid_scripts": sum(x["type"] == "scripter" and x["grade"]["schema_valid"] for x in r["turns"]),
            "archive_path": str(out / "dataset")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "smoke"], default="check")
    ap.add_argument("--out", type=Path, default=Path("out/detective-001b"))
    ap.add_argument("--download-dir", type=Path, default=Path("downloads/detective-001b"))
    args = ap.parse_args()
    if args.mode == "check":
        print(json.dumps(plan(), indent=2, sort_keys=True))
    else:
        print(json.dumps(run(args.out, args.download_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
