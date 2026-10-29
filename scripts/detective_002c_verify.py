#!/usr/bin/env python3
"""Independent standard-library readback for DETECTIVE-002C LFM v5 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.detective_002b_runtime import marker_observations  # noqa: E402

CONFIG = ROOT / "experiments/detective-002/lfm-v5-runtime.json"
SOURCE = ROOT / "experiments/detective-002/source-evidence/source-evidence.json"
BATTERY = ROOT / "experiments/detective-002/runtime-smoke.json"
HISTORICAL = ROOT / "datasets/detective-002b/raw/lfm25-350m.json"


class VerifyError(ValueError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise VerifyError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def verify_archive(root: Path) -> dict:
    cfg = read_json(CONFIG)
    source = read_json(SOURCE)
    battery = read_json(BATTERY)
    historical = read_json(HISTORICAL)

    expected_files = {
        "README.md",
        "manifest.json",
        "result.json",
        "provenance/source-artifacts.json",
        "provenance/hashes.sha256",
    }
    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*") if p.is_file()
    }
    need(present == expected_files, "002C archive file set mismatch")
    expected_inventory = [
        f"{file_sha(root / rel)}  {rel}"
        for rel in sorted(expected_files - {"provenance/hashes.sha256"})
    ]
    need(
        (root / "provenance/hashes.sha256").read_text(
            encoding="utf-8"
        ).splitlines() == expected_inventory,
        "002C archive hash inventory mismatch",
    )

    authority = cfg["source_authority"]
    need(
        git_blob_sha1(SOURCE) == authority["stage_a_git_blob_sha1"]
        and git_blob_sha1(BATTERY) == authority["battery_git_blob_sha1"]
        and git_blob_sha1(HISTORICAL)
            == authority["historical_result_git_blob_sha1"],
        "002C frozen source blob drift",
    )

    candidate = next(
        x for x in source["candidates"]
        if x["candidate_id"] == "lfm25-350m"
    )
    model = cfg["model"]
    need(
        candidate["repo"] == model["repo"]
        and candidate["resolved_revision"] == model["revision"]
        and candidate["weight"]["path"] == model["weight_path"]
        and candidate["weight"]["lfs_sha256"] == model["weight_sha256"]
        and candidate["weight"]["size"] == model["weight_bytes"]
        and candidate["tokenizer_config"]["sha256"]
            == model["tokenizer_config_sha256"]
        and candidate["chat_template_file"]["sha256"]
            == model["chat_template_sha256"]
        and candidate["config"]["sha256"] == model["config_sha256"]
        and candidate["production_license_gate"] is True,
        "002C Stage-A model authority mismatch",
    )

    failure = historical["runtime_failure"]
    need(
        historical["experiment"] == "DETECTIVE-002B"
        and historical["status"] == "runtime_fail"
        and historical["runtime_contract"]["transformers"] == "4.57.6"
        and failure["type"] == "ValueError"
        and "Tokenizer class TokenizersBackend" in failure["message"],
        "002C historical control drift",
    )

    result = read_json(root / "result.json")
    manifest = read_json(root / "manifest.json")
    provenance = read_json(root / "provenance/source-artifacts.json")

    need(
        result.get("schema") == "detective-002c-lfm-runtime-result-v1"
        and result.get("experiment") == "DETECTIVE-002C"
        and result.get("issue") == 133,
        "002C result identity mismatch",
    )
    need(
        result.get("status") in cfg["decisions"]["allowed"],
        "002C unapproved outcome classification",
    )
    need(
        result.get("config_sha256") == file_sha(CONFIG)
        and result.get("source_evidence_sha256") == file_sha(SOURCE)
        and result.get("battery_sha256") == file_sha(BATTERY)
        and result.get("historical_result_sha256") == file_sha(HISTORICAL),
        "002C result source hash mismatch",
    )
    need(
        result.get("model") == model
        and result.get("runtime_contract") == cfg["runtime"]
        and result.get("historical_control") == cfg["historical_control"],
        "002C model/runtime/historical contract mismatch",
    )
    need(
        result.get("optimizer_steps") == 0
        and result.get("paid_api_calls") == 0
        and result.get("relevance_mapping_accessed") is False
        and result.get("production_license_cleared") is False
        and result.get("training_selection_allowed") is False
        and result.get("population_expansion_admitted") is False,
        "002C experiment authority widened",
    )
    packages = result.get("resolved_packages") or {}
    need(
        packages.get("transformers") == "5.20.1"
        and packages.get("torch") == "2.14.1+cpu"
        and packages.get("python", "").startswith("3.12"),
        "002C resolved core package identity mismatch",
    )

    token = result.get("tokenizer_stage") or {}
    need(
        token.get("status") in {"tokenizer_pass", "tokenizer_incompatible"},
        "002C tokenizer stage status invalid",
    )
    need(
        token.get("transformers") == "5.20.1",
        "002C tokenizer stage did not run on v5.20.1",
    )
    source_stage = token.get("source") or {}
    need(
        source_stage.get("config_sha256") == model["config_sha256"]
        and source_stage.get("tokenizer_config_sha256")
            == model["tokenizer_config_sha256"]
        and source_stage.get("chat_template_sha256")
            == model["chat_template_sha256"],
        "002C tokenizer small-source identity mismatch",
    )

    if result["status"] == "tokenizer_incompatible":
        need(
            token["status"] == "tokenizer_incompatible",
            "002C top-level/tokenizer outcome mismatch",
        )
        need(
            source_stage.get("weight_downloaded") is False,
            "002C tokenizer failure downloaded model weight",
        )
        need(
            "weight_stage" not in result
            and result.get("model_stage") is None,
            "002C tokenizer failure improperly entered model stage",
        )
        need(
            isinstance(token.get("tokenizer_error"), dict),
            "002C tokenizer failure missing raw error",
        )
    else:
        need(
            token["status"] == "tokenizer_pass",
            "002C model-stage result without tokenizer PASS",
        )
        need(
            token.get("tokenizer_class")
                == cfg["stages"]["expected_tokenizer_class"]
            and token.get("tokenizer_backend")
                == cfg["stages"]["require_tokenizer_backend"],
            "002C tokenizer class/backend mismatch",
        )
        rendered = token.get("rendered_prompts")
        need(
            isinstance(rendered, list)
            and len(rendered) == 4
            and [x["task_id"] for x in rendered]
                == ["B01", "B02", "B03", "B04"],
            "002C tokenizer render packet incomplete",
        )
        for row in rendered:
            need(
                0 < row["prompt_tokens"] <= 512
                and re.fullmatch(
                    r"[0-9a-f]{64}", row["prompt_token_sha256"]
                ) is not None,
                "002C rendered prompt token custody invalid",
            )

        weight = result.get("weight_stage") or {}
        need(
            source_stage.get("weight_downloaded") is True
            and weight.get("selected_weight_path") == model["weight_path"]
            and weight.get("selected_weight_sha256") == model["weight_sha256"]
            and weight.get("selected_weight_bytes") == model["weight_bytes"]
            and weight.get("staged_weight_sha256") == model["weight_sha256"],
            "002C model stage lacks exact frozen weight proof",
        )
        model_stage = result.get("model_stage") or {}
        if result["status"] == "model_runtime_incompatible":
            need(
                model_stage.get("status") == "model_runtime_incompatible"
                and isinstance(model_stage.get("error"), dict),
                "002C model incompatibility missing raw error",
            )
        else:
            need(
                result["status"] == "runtime_compatibility_pass"
                and model_stage.get("status") == "runtime_compatibility_pass",
                "002C PASS classification mismatch",
            )
            env = model_stage.get("environment") or {}
            need(
                env.get("transformers") == "5.20.1"
                and env.get("torch_threads") == 4
                and env.get("cuda_available") is False,
                "002C PASS environment mismatch",
            )
            tasks = model_stage.get("tasks")
            need(
                isinstance(tasks, list)
                and len(tasks) == 4
                and [x["task_id"] for x in tasks]
                    == ["B01", "B02", "B03", "B04"],
                "002C PASS task set incomplete",
            )
            total = 0
            for task_cfg, rendered_row, task in zip(
                battery["battery"], rendered, tasks
            ):
                need(
                    task["prompt_token_count"]
                        == rendered_row["prompt_tokens"]
                    and task["prompt_token_sha256"]
                        == rendered_row["prompt_token_sha256"],
                    "002C model prompt differs from tokenizer-only gate",
                )
                need(
                    task["messages"] == rendered_row["messages"],
                    "002C model messages differ from tokenizer-only gate",
                )
                need(
                    task["generated_tokens"]
                        == len(task["generated_token_ids"])
                    and 0 <= task["generated_tokens"] <= 64,
                    "002C generated token record invalid",
                )
                need(
                    bool(task["hit_max_new_tokens"])
                    == (task["generated_tokens"] == 64),
                    "002C truncation marker invalid",
                )
                need(
                    task["observations"]
                    == marker_observations(task_cfg, task["response"]),
                    "002C mechanical observations do not replay",
                )
                total += task["generated_tokens"]
            need(
                total == model_stage.get("total_generated_tokens")
                and total <= 256,
                "002C total decode budget mismatch",
            )

    need(
        manifest.get("schema") == "detective-002c-runtime-archive-v1"
        and manifest.get("experiment") == "DETECTIVE-002C"
        and manifest.get("issue") == 133
        and manifest.get("status") == result["status"]
        and manifest.get("result_sha256") == file_sha(root / "result.json")
        and manifest.get("config_sha256") == file_sha(CONFIG)
        and manifest.get("source_evidence_sha256") == file_sha(SOURCE)
        and manifest.get("battery_sha256") == file_sha(BATTERY)
        and manifest.get("historical_result_sha256")
            == file_sha(HISTORICAL),
        "002C manifest/result/source mismatch",
    )
    need(
        manifest.get("optimizer_steps") == 0
        and manifest.get("paid_api_calls") == 0
        and manifest.get("relevance_mapping_accessed") is False
        and manifest.get("production_license_cleared") is False
        and manifest.get("training_selection_allowed") is False
        and manifest.get("population_expansion_admitted") is False,
        "002C manifest widened authority",
    )
    need(
        re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_head", ""))
        is not None
        and str(manifest.get("workflow_run_id", "")).isdigit(),
        "002C archive source/run identity missing",
    )

    need(
        provenance.get("source_head") == manifest["source_head"]
        and provenance.get("workflow_run_id")
            == manifest["workflow_run_id"]
        and provenance.get("stage_a_git_blob_sha1")
            == cfg["source_authority"]["stage_a_git_blob_sha1"]
        and provenance.get("battery_git_blob_sha1")
            == cfg["source_authority"]["battery_git_blob_sha1"]
        and provenance.get("historical_result_git_blob_sha1")
            == cfg["source_authority"]["historical_result_git_blob_sha1"]
        and provenance.get("transformers") == "5.20.1"
        and provenance.get("transformers_wheel_sha256")
            == cfg["runtime"]["transformers_wheel_sha256"]
        and provenance.get("model_weight_sha256")
            == model["weight_sha256"]
        and provenance.get("training_steps") == 0
        and provenance.get("external_provider_calls") == 0
        and provenance.get("relevance_mapping_accessed") is False,
        "002C provenance mismatch",
    )

    return {
        "verdict": "DETECTIVE_002C_SOURCE_READBACK_PASS",
        "status": result["status"],
        "historical_status": "runtime_fail",
        "historical_transformers": "4.57.6",
        "requalification_transformers": "5.20.1",
        "tokenizer_status": token["status"],
        "weight_downloaded": source_stage.get("weight_downloaded"),
        "optimizer_steps": 0,
        "production_license_cleared": False,
        "training_selection_allowed": False,
        "population_expansion_admitted": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(
        verify_archive(args.evidence), indent=2, sort_keys=True
    ))


if __name__ == "__main__":
    main()
