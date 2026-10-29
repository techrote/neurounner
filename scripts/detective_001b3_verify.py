#!/usr/bin/env python3
"""Independent standard-library readback for DETECTIVE-001B3 Sprocket scripter evidence."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import detective_001b3_sprocket as s  # noqa: E402


class ReadbackError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ReadbackError(message)


def verify(dataset: Path) -> dict:
    spec, case = s.load_spec()
    plan = s.plan()
    expected_files = {
        "README.md", "manifest.json", "raw/run.json",
        "provenance/source-artifacts.json", "provenance/hashes.sha256",
    }
    present = {p.relative_to(dataset).as_posix()
               for p in dataset.rglob("*") if p.is_file()}
    require(present == expected_files, "archive file set mismatch")

    inventory = (dataset / "provenance/hashes.sha256").read_text(
        encoding="utf-8").splitlines()
    expected_inventory = [
        f"{s.file_sha(dataset / rel)}  {rel}"
        for rel in sorted(expected_files - {"provenance/hashes.sha256"})
    ]
    require(inventory == expected_inventory, "archive hash inventory mismatch")

    run = s.read_json(dataset / "raw/run.json")
    manifest = s.read_json(dataset / "manifest.json")
    provenance = s.read_json(dataset / "provenance/source-artifacts.json")

    require(run.get("schema") == "detective-sprocket-scripter-run-v1"
            and run.get("experiment") == "DETECTIVE-001B3"
            and run.get("admission") == "EXPLORATORY_NOT_ADMITTED"
            and run.get("complete") is True,
            "run completeness or admission label invalid")
    require(run["plan_sha256"] == s.sha_obj(plan)
            and run["spec_sha256"] == s.file_sha(s.SPEC)
            and run["case_sha256"] == s.file_sha(s.CASE)
            and run["script_sha256"] == s.file_sha(Path(s.__file__)),
            "source plan/config/case/script drift")
    require(run["model"] == spec["model"]
            and run["publisher"] == spec["publisher"],
            "model/publisher source identity drift")

    repository = run["repository"]
    require(re.fullmatch(r"[0-9a-f]{40}", repository.get("head_sha", "")) is not None,
            "missing source head")
    require(isinstance(repository.get("run_id"), str)
            and repository["run_id"].isdecimal(),
            "missing Actions run identity")

    model_files = run["model_files"]
    for filename, expected in (
        ("tokenizer.json", spec["model"]["tokenizer_sha256"]),
        ("tokenizer_config.json", spec["model"]["tokenizer_config_sha256"]),
        ("special_tokens_map.json", spec["model"]["special_tokens_map_sha256"]),
        ("model.safetensors", spec["model"]["weight_sha256"]),
    ):
        require(filename in model_files
                and model_files[filename]["sha256"] == expected,
                "model file hash mismatch: " + filename)
    require(model_files["model.safetensors"]["bytes"] == spec["model"]["weight_bytes"],
            "root Instruct checkpoint size mismatch")
    require(spec["model"]["chat_checkpoint_forbidden"] not in model_files,
            "Chat checkpoint leaked into root Instruct run")

    publisher = run["publisher_checkout"]
    require(publisher["commit"] == spec["publisher"]["commit"]
            and publisher["git_blobs"] == {
                spec["publisher"]["sft_path"]: spec["publisher"]["sft_git_blob"],
                spec["publisher"]["harvest_path"]: spec["publisher"]["harvest_git_blob"],
            }, "publisher checkout/blob mismatch")

    tasks = spec["tasks"]
    raw = run["raw_generations"]
    results = run["results"]
    require(len(raw) == len(results) == len(tasks) == 5, "task count mismatch")
    total = 0
    reproduced = []
    for task, attempt, stored_score in zip(tasks, raw, results):
        require(attempt["task_id"] == task["id"]
                and attempt["family"] == task["family"]
                and attempt["mode"] == task["mode"],
                "raw task identity mismatch")
        turns = s.task_turns(spec, case, task)
        require(attempt["turns"] == turns, "model saw non-preregistered turns")
        require(attempt["rendered_prompt_sha256"] ==
                __import__("hashlib").sha256(
                    s.render_text(turns).encode("utf-8")).hexdigest(),
                "rendered prompt digest mismatch")
        require(0 < attempt["prompt_tokens"] <= spec["decode"]["max_prompt_tokens"],
                "prompt token budget invalid")
        require(re.fullmatch(r"[0-9a-f]{64}",
                             attempt["prompt_token_ids_sha256"]) is not None,
                "prompt token digest absent")
        token_ids = attempt["generated_token_ids"]
        require(isinstance(token_ids, list)
                and len(token_ids) == attempt["generated_tokens"]
                and len(token_ids) <= spec["decode"]["max_new_tokens"]
                and all(type(x) is int and 0 <= x < 32000 for x in token_ids),
                "generated token record invalid")
        require(isinstance(attempt["decoded_with_special_tokens"], str)
                and isinstance(attempt["elapsed_seconds"], (int, float))
                and attempt["elapsed_seconds"] >= 0,
                "raw decoded output/timing missing")
        score = s.score_task(
            spec, case, task, attempt["decoded_with_special_tokens"])
        require(stored_score == score, "stored deterministic score does not replay")
        reproduced.append(score)
        total += len(token_ids)

    require(run["total_generated_tokens"] == total
            and total <= spec["limits"]["max_generated_tokens"],
            "total decode budget mismatch")

    summary = {
        "tasks": 5,
        "native_tool_calls_valid": sum(
            x.get("native_call_valid") is True for x in reproduced),
        "tool_selection_correct": sum(
            x.get("task_selection_correct") is True for x in reproduced),
        "tool_restraint_success": sum(
            x.get("tool_restraint") is True for x in reproduced),
        "tool_result_or_denial_use_success": sum(
            x.get("result_use_success") is True for x in reproduced),
        "generated_tokens": total,
        "training_steps": 0,
    }
    require(manifest["schema"] == "detective-sprocket-scripter-archive-v1"
            and manifest["experiment"] == "DETECTIVE-001B3"
            and manifest["admission"] == "EXPLORATORY_NOT_ADMITTED"
            and manifest["run_sha256"] == s.sha_obj(run)
            and manifest["spec_sha256"] == s.file_sha(s.SPEC)
            and manifest["plan_sha256"] == run["plan_sha256"]
            and manifest["source_head_sha"] == repository["head_sha"]
            and manifest["summary"] == summary,
            "manifest does not reproduce from raw evidence")

    require(provenance["repository"] == "techrote/neurounner"
            and provenance["head_sha"] == repository["head_sha"]
            and provenance["workflow_run_id"] == repository["run_id"]
            and provenance["model"] == spec["model"]
            and provenance["publisher"] == spec["publisher"]
            and provenance["model_files"] == run["model_files"]
            and provenance["publisher_checkout"] == run["publisher_checkout"]
            and provenance["script_sha256"] == s.file_sha(Path(s.__file__))
            and provenance["case_sha256"] == s.file_sha(s.CASE),
            "provenance does not match raw source")

    # This run must never rewrite the contextual historical SmolLM control.
    hc = spec["historical_control"]
    require(hc["strict_script_attempts"] == 2
            and hc["strict_script_valid"] == 0
            and hc["comparison_is_contextual_not_matched"] is True,
            "historical control semantics changed")

    return {
        "verdict": "SOURCE_BOUND_SPROCKET_SCRIPT_READBACK_PASS",
        "tasks": 5, "generated_tokens": total,
        **summary,
        "training_performed": False,
        "host_execution_performed": False,
        "behavior_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
