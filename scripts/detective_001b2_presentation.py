#!/usr/bin/env python3
"""DETECTIVE-001B2: fixed JSON vs natural witness-projection comparison.

Offline check by default. Live run is explicitly bounded; not a training job,
not a hidden heldout, not an operational oracle for case truth.
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
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import detective_001b_pilot as parent  # noqa: E402
from scripts.detective_001_contract import ContractError, witness_view  # noqa: E402

SPEC = ROOT / "experiments/detective-001/presentation-ab.json"
DATASET = "detective-001-presentation-ab"

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def sha_file(path: Path) -> str:
    return parent.file_sha(path)

def canonical(obj: Any) -> str:
    return parent.canonical(obj)

def sha_obj(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()

def write_json(path: Path, obj: Any) -> None:
    parent.write_json(path, obj)

def load_spec() -> tuple[dict, dict, dict]:
    spec = read_json(SPEC)
    old, case, release = parent.check_config()
    if (spec.get("schema"), spec.get("experiment"), spec.get("parent_issue")) != (
        "detective-presentation-ab-v1", "DETECTIVE-001B2", 106
    ):
        raise ContractError("wrong experiment/version")
    if (spec["reference"]["original_run_id"] != 40579442013
        or spec["reference"]["original_source_sha"] != "5fe18428beba882696de5f36a8bce781d3688597"
        or spec["reference"]["original_dataset"] != "datasets/detective-001-smoke"
        or spec["reference"]["original_json_like_interviews"] != 8
        or spec["reference"]["original_interviews"] != 12):
        raise ContractError("original evidence reference drift")
    source = spec["source"]
    if (source["base_repo"] != old["base"]["repo"]
        or source["base_revision"] != old["base"]["revision"]
        or source["base_weight_file"] != old["base"]["weights_file"]
        or source["base_weight_sha256"] != old["base"]["weights_sha256"]
        or source["release_tag"] != release["release_tag"]
        or source["release_manifest"] != parent.MANIFEST.relative_to(ROOT).as_posix()
        or source["member_contract_sha256"] != release["member_contract_sha256"]):
        raise ContractError("source identity mismatch")
    if spec["case_path"] != parent.CASE.relative_to(ROOT).as_posix():
        raise ContractError("case changed")
    if spec["lanes"] != ["BASE", "V-G02-A", "V-G09-C"]:
        raise ContractError("cohort changed")
    expected_situations = [
        {"id": f"I{i+1:02d}", "witness_id": old["dialogue"]["witness_ids"][i],
         "question": old["dialogue"]["first_questions"][i]}
        for i in range(2)
    ]
    if spec["situations"] != expected_situations or spec["conditions"] != ["json", "natural"]:
        raise ContractError("situations/conditions drift")
    if spec["system"] != old["dialogue"]["system"]:
        raise ContractError("system semantics drift")
    if spec["decoding"] != {
        "do_sample": False, "seed": 24301, "max_new_tokens": 80, "max_prompt_tokens": 512
    }:
        raise ContractError("decoding contract drift")
    if spec["limits"] != {
        "max_lanes": 3, "max_situations": 2, "max_conditions": 2, "expected_turns": 12,
        "max_total_generated_tokens": 960, "max_threads": 4, "max_job_minutes": 20,
        "paid_api_calls": 0, "training_steps": 0, "raw_bytes": 120000
    }:
        raise ContractError("budget drift")
    if spec["measurements"]["primary"] != "output_prefix_structured_object":
        raise ContractError("primary metric drift")
    if spec["go_no_go"] != {
        "archive_only_if_complete_and_source_verified": True,
        "report_all_failures": True, "no_model_training_or_persona_selection": True,
        "nonparametric_sample_small": True, "sealed_final_holdout_not_included": True
    }:
        raise ContractError("qualification limit drift")
    return spec, case, release

def _safe_line(text: str) -> str:
    if (not isinstance(text, str) or not text or len(text) > 1024 or
        any(ord(x) < 32 or ord(x) == 127 for x in text)):
        raise ContractError("non-literal or multiline witness data in natural renderer")
    return text

def render_natural(view: dict) -> str:
    """Same admitted facts/question, with no private attributes or newly invented clues."""
    if set(view) != {"schema", "case_id", "admitted_memories", "player_question"}:
        raise ContractError("renderer received non-blind view")
    if view["schema"] != "detective-interview-v1":
        raise ContractError("unexpected interview schema")
    memories = view["admitted_memories"]
    if not isinstance(memories, list) or len(memories) > 8:
        raise ContractError("unexpected memory list")
    rows = [
        f"Case: {_safe_line(view['case_id'])}",
        "Record format: detective-interview-v1",
        "Admitted witness memories:",
    ]
    for item in memories:
        if not isinstance(item, dict) or set(item) != {"id", "text"}:
            raise ContractError("invalid admitted memory")
        rows.append(f"- {_safe_line(item['id'])}: {_safe_line(item['text'])}")
    rows.append(f"Player question: {_safe_line(view['player_question'])}")
    return "\n".join(rows)

def messages(spec: dict, case: dict, scenario_index: int, condition: str) -> list[dict]:
    item = spec["situations"][scenario_index]
    blind = witness_view(case, item["witness_id"], item["question"], blind=True)
    if condition == "json":
        user = canonical(blind)
    elif condition == "natural":
        user = render_natural(blind)
    else:
        raise ContractError("unknown rendering condition")
    return [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": user},
    ]

def plan() -> dict:
    spec, case, release = load_spec()
    condition_rows = []
    for i, item in enumerate(spec["situations"]):
        for cond in spec["conditions"]:
            msgs = messages(spec, case, i, cond)
            condition_rows.append({
                "situation_id": item["id"], "condition": cond, "messages": msgs,
                "messages_sha256": sha_obj(msgs),
            })
        canonical_blind = witness_view(case, item["witness_id"], item["question"], blind=True)
        natural = render_natural(canonical_blind)
        if (not all(natural.count(x["text"]) == 1 for x in canonical_blind["admitted_memories"])
            or natural.count(item["question"]) != 1):
            raise ContractError("natural language conversion loses/repeats facts")
        other = {x["id"] for x in case["facts"]} - {
            m["id"] for m in canonical_blind["admitted_memories"]
        }
        if any(s in natural for s in other):
            raise ContractError("unadmitted fact ID leaked into natural renderer")
        if any(other_npc["id"] in natural for other_npc in case["npcs"]):
            raise ContractError("actor identity leaked")
        if any(t["credential"] in natural for t in case["terminals"]):
            raise ContractError("game credentials leaked")
    if len(condition_rows) != 4:
        raise ContractError("wrong pair count")
    return {
        "schema": "detective-presentation-plan-v1",
        "config_sha256": sha_file(SPEC),
        "old_config_sha256": sha_file(parent.CONFIG),
        "case_sha256": sha_file(parent.CASE),
        "release_sha256": sha_file(parent.MANIFEST),
        "member_contract_sha256": release["member_contract_sha256"],
        "lanes": spec["lanes"],
        "paired_messages": condition_rows,
        "expected_turns": spec["limits"]["expected_turns"],
        "max_total_generated_tokens": spec["limits"]["max_total_generated_tokens"],
    }

def echo_signature(output: str) -> dict:
    if not isinstance(output, str):
        raise ContractError("output must be raw decoded text")
    return {
        "starts_object": output.lstrip().startswith("{"),
        "echoed_schema_keys": sorted([
            k for k in ("admitted_memories", "case_id", "player_question", "schema")
            if re.search(r'(?<![a-zA-Z])' + re.escape(k) + r'(?![a-zA-Z])', output)
        ]),
        "characters": len(output),
        "whitespace_words": len(output.split()),
    }

def persist_raw(out: Path, run_data: dict, cap: int) -> None:
    data = canonical(run_data).encode("utf-8")
    if len(data) > cap:
        raise ContractError("raw evidence byte cap exceeded")
    write_json(out / "raw" / "run.json", run_data)

def promote_local_archive(out: Path, data: dict, spec: dict) -> Path:
    dataset = out / "dataset"
    (dataset / "provenance").mkdir(parents=True, exist_ok=True)
    (dataset / "raw").mkdir(parents=True, exist_ok=True)
    write_json(dataset / "raw" / "run.json", data)
    report = {
        "experiment": "DETECTIVE-001B2",
        "scope": "presentation-only-development-smoke",
        "admission": "NOT_EVALUATED",
        "lanes": spec["lanes"],
        "turns": len(data["turns"]),
        "total_generated_tokens": sum(t["generated_tokens"] for t in data["turns"]),
        "echo_prefix_by_condition": {
            cond: {
                "starts_object": sum(
                    t["echo_signature"]["starts_object"]
                    for t in data["turns"] if t["condition"] == cond
                ),
                "total": sum(t["condition"] == cond for t in data["turns"]),
            } for cond in spec["conditions"]
        },
        "case_family_count": 1,
        "training_steps": 0,
    }
    write_json(dataset / "manifest.json", {
        "schema": "detective-presentation-archive-v1", "run_sha256": sha_obj(data),
        "config_sha256": sha_file(SPEC),
        "plan_sha256": data["plan_sha256"],
        "source_head_sha": data["repository"]["head_sha"],
        "summary": report,
    })
    write_json(dataset / "provenance/source-artifacts.json", {
        "repo": "techrote/neurounner", "head_sha": data["repository"]["head_sha"],
        "workflow_run_id": data["repository"]["run_id"], "workflow_run_url": data["repository"]["run_url"],
        "base_model": spec["source"], "model_lanes": data["lanes"],
        "script_sha256": data["script_sha256"],
        "source_config_sha256": data["config_sha256"],
        "source_case_sha256": data["case_sha256"],
    })
    (dataset / "README.md").write_text(
        "# DETECTIVE-001B2 raw presentation A/B\n\n"
        "Original model output from pinned SmolLM2 base and two independently trained "
        "Village SFT cartridges on the same two fictional witness situations "
        "in JSON vs natural-language renderings. This is an exploratory **one-case-family** "
        "presentation experiment. Model falsehood, personal intent, truth, plot solvability "
        "and persona quality are deliberately NOT evaluated. All original prompt/response "
        "text, output tokens, checkpoint/adapter source identities, and measured failures "
        "are retained. Original DETECTIVE-001B archive is unmodified.\n",
        encoding="utf-8",
    )
    inventory = []
    for entry in sorted(x for x in dataset.rglob("*") if x.is_file()):
        relative = entry.relative_to(dataset).as_posix()
        if relative != "provenance/hashes.sha256":
            inventory.append(f"{sha_file(entry)}  {relative}")
    (dataset / "provenance/hashes.sha256").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    return dataset

def run(out: Path, download_dir: Path) -> dict:
    spec, case, release = load_spec()
    source_plan = plan()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    import transformers
    import peft

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(spec["decoding"]["seed"])

    native = Path(snapshot_download(
        repo_id=spec["source"]["base_repo"], revision=spec["source"]["base_revision"],
        allow_patterns=["model.safetensors", "config.json", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json",
                        "generation_config.json"],
    ))
    if sha_file(native / "model.safetensors") != spec["source"]["base_weight_sha256"]:
        raise ContractError("native backbone weights SHA changed")
    tok = AutoTokenizer.from_pretrained(native, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    source_sha = os.environ.get("DETECTIVE_HEAD_SHA") or os.environ.get("GITHUB_SHA") or "local"
    run_id = os.environ.get("GITHUB_RUN_ID")
    result = {
        "schema": "detective-presentation-run-v1", "experiment": spec["experiment"],
        "admission": "NOT_EVALUATED", "complete": False,
        "plan_sha256": sha_obj(source_plan), "config_sha256": sha_file(SPEC),
        "case_sha256": sha_file(parent.CASE),
        "release_sha256": sha_file(parent.MANIFEST),
        "script_sha256": sha_file(Path(__file__)),
        "reference": spec["reference"], "sources": spec["source"],
        "repository": {"head_sha": source_sha, "run_id": run_id,
                       "run_url": f"https://github.com/techrote/neurounner/actions/runs/{run_id}" if run_id else None},
        "runtime": {
            "platform": platform.platform(), "cpu": parent._cpu_id(),
            "threads": torch.get_num_threads(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "lanes": [], "turns": [],
    }

    assets = {a["member_id"]: a for a in release["assets"]}
    total_tokens = 0
    for lid in spec["lanes"]:
        base = AutoModelForCausalLM.from_pretrained(
            native, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32
        )
        if lid == "BASE":
            model = base
            identity = {"lane_id": lid, "kind": "untuned",
                        "adapter_tar_sha256": None, "adapter_dir_sha256": None}
        else:
            asset = assets[lid]
            path = download_dir / asset["file"]
            location = out / "work" / lid
            parent.unpack_adapter(path, location, asset)
            model = PeftModel.from_pretrained(base, location, is_trainable=False)
            identity = {"lane_id": lid, "kind": "VILLAGE-001C-SFT",
                        "adapter_tar_sha256": asset["sha256"],
                        "adapter_dir_sha256": asset["adapter_directory_sha256"]}
        model.config.use_cache = True
        model.eval()
        result["lanes"].append(identity)
        persist_raw(out, result, spec["limits"]["raw_bytes"])

        for i, situation in enumerate(spec["situations"]):
            for condition in spec["conditions"]:
                msg = messages(spec, case, i, condition)
                ids = tok.apply_chat_template(
                    msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                if ids.shape[-1] > spec["decoding"]["max_prompt_tokens"]:
                    raise ContractError("prompt exceeds token ceiling")
                if total_tokens + spec["decoding"]["max_new_tokens"] > spec["limits"]["max_total_generated_tokens"]:
                    raise ContractError("decode would exceed total token ceiling")
                tic = time.monotonic()
                with torch.inference_mode():
                    generation = model.generate(
                        ids, max_new_tokens=spec["decoding"]["max_new_tokens"],
                        do_sample=False, use_cache=True,
                        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                    )
                elapsed = time.monotonic() - tic
                raw_tokens = generation[0, ids.shape[-1]:].tolist()
                total_tokens += len(raw_tokens)
                answer = tok.decode(raw_tokens, skip_special_tokens=True)
                record = {
                    "id": f"{lid}-{situation['id']}-{condition}",
                    "lane_id": lid, "situation_id": situation["id"],
                    "condition": condition,
                    "request": {
                        "messages": msg, "input_token_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                        "input_tokens": int(ids.shape[-1]),
                        "max_new_tokens": spec["decoding"]["max_new_tokens"],
                        "do_sample": False,
                    },
                    "response": answer,
                    "response_raw_with_special_tokens": tok.decode(raw_tokens, skip_special_tokens=False),
                    "generated_token_ids": raw_tokens, "generated_tokens": len(raw_tokens),
                    "elapsed_seconds": elapsed,
                    "echo_signature": echo_signature(answer),
                }
                result["turns"].append(record)
                persist_raw(out, result, spec["limits"]["raw_bytes"])
        del model
        del base
        import gc
        gc.collect()
    if (len(result["lanes"]) != 3 or len(result["turns"]) != 12 or
        total_tokens > spec["limits"]["max_total_generated_tokens"]):
        raise ContractError("incomplete or oversized native pilot")
    result["runtime"]["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["total_generated_tokens"] = total_tokens
    result["complete"] = True
    persist_raw(out, result, spec["limits"]["raw_bytes"])
    dataset = promote_local_archive(out, result, spec)
    return {"turns": len(result["turns"]), "tokens": total_tokens,
            "dataset": str(dataset), "manifest": str(dataset / "manifest.json")}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check","run"], default="check")
    parser.add_argument("--out", type=Path, default=Path("out/detective-001b2"))
    parser.add_argument("--download-dir", type=Path, default=Path("downloads/detective-001b2"))
    a = parser.parse_args()
    print(json.dumps(plan() if a.mode == "check" else run(a.out,a.download_dir),
                     indent=2,sort_keys=True))

if __name__ == "__main__":
    main()
