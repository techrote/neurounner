#!/usr/bin/env python3
"""Independent standard-library replay for DETECTIVE-003D live Sprocket evidence."""
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

from scripts import detective_003a_session as session_api  # noqa: E402
from scripts import detective_003c_scripter_bridge as bridge  # noqa: E402
from scripts import detective_003d_sprocket_live as live  # noqa: E402


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
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root: Path) -> list[str]:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel != "provenance/hashes.sha256":
            rows.append(f"{file_sha(path)}  {rel}")
    return rows


def verify(evidence: Path) -> dict:
    cfg, case, b3_spec = live.load_contract()
    expected_files = {
        "README.md",
        "manifest.json",
        "raw/run.json",
        "provenance/source-artifacts.json",
        "provenance/hashes.sha256",
    }
    present = {
        p.relative_to(evidence).as_posix()
        for p in evidence.rglob("*") if p.is_file()
    }
    need(present == expected_files, "003D archive file set mismatch")
    need(
        (evidence / "provenance/hashes.sha256").read_text(
            encoding="utf-8"
        ).splitlines() == inventory(evidence),
        "003D archive hash inventory mismatch",
    )

    manifest = read_json(evidence / "manifest.json")
    raw = read_json(evidence / "raw/run.json")
    provenance = read_json(evidence / "provenance/source-artifacts.json")

    need(
        manifest.get("schema") == "detective-003d-sprocket-live-archive-v1"
        and manifest.get("experiment") == "DETECTIVE-003D"
        and manifest.get("issue") == 146
        and manifest.get("status") == "integration_pass",
        "003D manifest identity/status mismatch",
    )
    need(
        raw.get("schema") == "detective-003d-sprocket-live-run-v1"
        and raw.get("experiment") == "DETECTIVE-003D"
        and raw.get("issue") == 146
        and raw.get("status") == "integration_pass",
        "003D raw identity/status mismatch",
    )
    need(
        manifest["config_sha256"] == file_sha(live.CONFIG),
        "003D config digest mismatch",
    )
    need(
        manifest["model"] == cfg["model"],
        "003D manifest model identity mismatch",
    )
    need(
        re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_head", ""))
        is not None
        and str(manifest.get("workflow_run_id", "")).isdigit(),
        "003D source/run identity missing",
    )
    need(
        raw["repository"]["head_sha"] == manifest["source_head"]
        and str(raw["repository"]["run_id"])
            == str(manifest["workflow_run_id"])
        and raw["repository"]["run_url"] == manifest["workflow_run_url"],
        "003D raw/manifest source identity mismatch",
    )

    for field in ("optimizer_steps", "paid_api_calls"):
        need(manifest[field] == 0, "003D nonzero " + field)
    for field in (
        "host_execution",
        "training_admitted",
        "body_selection_made",
        "relevance_mapping_accessed",
        "learner_final_accessed",
        "population_expansion_admitted",
    ):
        need(manifest[field] is False, "003D authority widened: " + field)

    source = raw["source"]
    need(
        source["publisher_checkout"]["commit"] == cfg["publisher"]["commit"],
        "003D publisher checkout mismatch",
    )
    need(
        source["publisher_checkout"]["git_blobs"] == {
            cfg["publisher"]["sft_path"]: cfg["publisher"]["sft_git_blob"],
            cfg["publisher"]["harvest_path"]:
                cfg["publisher"]["harvest_git_blob"],
        },
        "003D publisher blob map mismatch",
    )
    model_files = source["model_files"]
    for filename, expected in (
        ("tokenizer.json", cfg["model"]["tokenizer_sha256"]),
        ("tokenizer_config.json", cfg["model"]["tokenizer_config_sha256"]),
        ("special_tokens_map.json", cfg["model"]["special_tokens_map_sha256"]),
        ("model.safetensors", cfg["model"]["weight_sha256"]),
    ):
        need(
            model_files[filename]["sha256"] == expected,
            "003D model source hash mismatch: " + filename,
        )
    need(
        model_files["model.safetensors"]["bytes"]
        == cfg["model"]["weight_bytes"],
        "003D root checkpoint byte mismatch",
    )
    need(
        cfg["model"]["chat_checkpoint_forbidden"] not in model_files,
        "003D Chat checkpoint contamination",
    )

    session, request = live.build_request(cfg, case)
    need(
        raw["request_receipt"] == request,
        "003D 003C request receipt does not reproduce",
    )
    turns, prompt_text = live.render_native_turns(cfg, request)
    need(raw["publisher_turns"] == turns, "003D publisher turns differ")
    need(
        raw["publisher_prompt_text"] == prompt_text,
        "003D publisher prompt text does not reproduce",
    )
    need(
        0 < raw["prompt_token_count"] <= cfg["prompt"]["max_prompt_tokens"]
        and re.fullmatch(r"[0-9a-f]{64}", raw["prompt_token_sha256"])
        is not None,
        "003D prompt token record invalid",
    )

    gen = raw["generation"]
    need(
        gen["generated_tokens"] == len(gen["generated_token_ids"])
        <= cfg["prompt"]["max_new_tokens"],
        "003D generation token record invalid",
    )
    need(
        gen["hit_max_new_tokens"]
        == (gen["generated_tokens"] == cfg["prompt"]["max_new_tokens"]),
        "003D truncation marker mismatch",
    )
    need(
        isinstance(gen["raw_output_with_special_tokens"], str),
        "003D exact raw output missing",
    )

    normalization = live.normalize_native_output(
        gen["raw_output_with_special_tokens"],
        b3_spec,
    )
    need(
        normalization == raw["normalization"],
        "003D native parse/normalization does not replay",
    )
    exact_probe = live.exact_probe_selected(cfg, normalization)
    need(
        exact_probe == raw["exact_probe_selected"]
        == manifest["exact_probe_selected"],
        "003D exact probe observation mismatch",
    )

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
        need(
            transaction == raw["transaction"],
            "003D 003C transaction does not replay",
        )
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
    else:
        need(raw["transaction"] is None, "003D rejected output has transaction")

    need(
        raw["camera_cache_visible_after_transaction"] == camera_visible,
        "003D camera visibility observation mismatch",
    )
    need(
        raw["evidence_application"] == {
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
        "003D evidence-application metadata mismatch",
    )
    need(
        raw["final_session"] == session_after,
        "003D final session does not replay",
    )
    need(
        raw["witness_prompt_after_application"] == witness_prompt,
        "003D witness prompt does not replay",
    )

    if witness_prompt is not None:
        blob = json.dumps(witness_prompt["prompt"], sort_keys=True)
        need("scanned twice" in blob, "003D granted prompt lacks archive excerpt")
        for forbidden in (
            "camera-cache",
            "F-CAMERA",
            "camera-console",
            "OXIDATION",
        ):
            need(
                forbidden not in blob,
                "003D witness prompt leaked authority token: " + forbidden,
            )

    if normalization["normalization_status"] != "normalized":
        expected_outcome = "native_or_normalization_rejected"
    elif camera_visible and witness_prompt is not None:
        expected_outcome = "normalized_executed_grant_applied"
    elif transaction is not None:
        expected_outcome = "normalized_executed_no_grant"
    else:
        raise VerifyError("invalid reconstructed 003D outcome")
    need(
        raw["outcome"] == manifest["outcome"] == expected_outcome,
        "003D outcome classification mismatch",
    )
    need(
        manifest["raw_generated_tokens"] == gen["generated_tokens"]
        and manifest["native_parse_status"]
            == normalization["native_parse_status"]
        and manifest["normalization_status"]
            == normalization["normalization_status"]
        and manifest["transaction_status"]
            == (transaction["status"] if transaction is not None else None)
        and manifest["camera_cache_visible_after_transaction"] == camera_visible
        and manifest["evidence_applied"] == (witness_prompt is not None)
        and manifest["final_session_sha256"] == session_after["sha256"],
        "003D manifest derived fields mismatch",
    )

    decision = raw["decision"]
    need(
        decision == {
            "correct_tool_selection_required": False,
            "exact_probe_selected": exact_probe,
            "access_granted": camera_visible,
            "evidence_applied": witness_prompt is not None,
            "host_execution": False,
            "training_admitted": False,
            "body_selection_made": False,
            "relevance_mapping_accessed": False,
            "learner_final_accessed": False,
            "population_expansion_admitted": False,
        },
        "003D decision record mismatch",
    )

    expected_case_sha = hashlib.sha256(
        live.canonical(case).encode("utf-8")
    ).hexdigest()
    need(
        raw["case_sha256"] == expected_case_sha,
        "003D case digest mismatch",
    )
    need(
        raw["environment"]["torch_threads"] == 4
        and raw["environment"]["cuda_available"] is False,
        "003D CPU/thread contract mismatch",
    )

    expected_blobs = {
        key: value["git_blob_sha1"]
        for key, value in cfg["authority"].items()
    }
    need(
        provenance["repository"] == "techrote/neurounner"
        and provenance["head_sha"] == manifest["source_head"]
        and str(provenance["workflow_run_id"])
            == str(manifest["workflow_run_id"])
        and provenance["config_sha256"] == file_sha(live.CONFIG)
        and provenance["authority_git_blobs"] == expected_blobs
        and provenance["model_files"] == source["model_files"]
        and provenance["publisher_checkout"] == source["publisher_checkout"]
        and provenance["external_provider_calls"] == 0
        and provenance["training_steps"] == 0
        and provenance["host_execution"] is False
        and provenance["relevance_mapping_accessed"] is False
        and provenance["learner_final_accessed"] is False,
        "003D provenance mismatch",
    )

    return {
        "verdict": "DETECTIVE_003D_SPROCKET_LIVE_READBACK_PASS",
        "outcome": expected_outcome,
        "generated_tokens": gen["generated_tokens"],
        "native_parse_status": normalization["native_parse_status"],
        "normalization_status": normalization["normalization_status"],
        "exact_probe_selected": exact_probe,
        "access_granted": camera_visible,
        "evidence_applied": witness_prompt is not None,
        "final_session_sha256": session_after["sha256"],
        "optimizer_steps": 0,
        "population_expansion_admitted": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(verify(args.evidence), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
