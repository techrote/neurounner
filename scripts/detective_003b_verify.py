#!/usr/bin/env python3
"""Independent standard-library replay for DETECTIVE-003B live-session evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from scripts import detective_003a_session as session_api
from scripts import detective_003b_live_session as live


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
    cfg, case, _, _ = live.load_contract()
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
    need(present == expected_files, "003B archive file set mismatch")
    need(
        (evidence / "provenance/hashes.sha256").read_text(
            encoding="utf-8"
        ).splitlines() == inventory(evidence),
        "003B archive hash inventory mismatch",
    )

    manifest = read_json(evidence / "manifest.json")
    raw = read_json(evidence / "raw/run.json")
    provenance = read_json(evidence / "provenance/source-artifacts.json")

    need(
        manifest.get("schema") == "detective-003b-live-session-archive-v1"
        and manifest.get("experiment") == "DETECTIVE-003B"
        and manifest.get("issue") == 142
        and manifest.get("status") == "integration_pass",
        "003B manifest identity/status mismatch",
    )
    need(
        raw.get("schema") == "detective-003b-live-session-run-v1"
        and raw.get("experiment") == "DETECTIVE-003B"
        and raw.get("issue") == 142
        and raw.get("status") == "integration_pass",
        "003B raw identity/status mismatch",
    )
    need(
        manifest["config_sha256"] == file_sha(live.CONFIG)
        == raw["config_sha256"],
        "003B config hash mismatch",
    )
    need(
        manifest["case_sha256"] == file_sha(live.CASE)
        == raw["case_file_sha256"],
        "003B case-file hash mismatch",
    )
    need(
        manifest["session_module_git_blob_sha1"]
        == cfg["authority"]["session_module"]["git_blob_sha1"],
        "003B session authority mismatch",
    )
    need(
        re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_head", ""))
        is not None
        and str(manifest.get("workflow_run_id", "")).isdigit(),
        "003B source/run identity missing",
    )
    need(
        raw["repository"]["head_sha"] == manifest["source_head"]
        and str(raw["repository"]["run_id"]) == str(manifest["workflow_run_id"])
        and raw["repository"]["run_url"] == manifest["workflow_run_url"],
        "003B raw/manifest source identity mismatch",
    )

    for field in (
        "optimizer_steps",
        "paid_api_calls",
    ):
        need(manifest[field] == 0 and raw[field] == 0, field + " must be zero")
    for field in (
        "relevance_mapping_accessed",
        "learner_final_accessed",
        "body_selection_made",
        "population_expansion_admitted",
    ):
        need(
            manifest[field] is False and raw[field] is False,
            "003B authority widened: " + field,
        )

    need(
        manifest["base_model"] == cfg["base_model"],
        "003B base model manifest drift",
    )
    for npc_id, meta in cfg["witnesses"].items():
        stored = manifest["witnesses"][npc_id]
        need(
            stored == {
                "adapter_name": meta["adapter_name"],
                "release_tag": meta["release_tag"],
                "archive_sha256": meta["archive_sha256"],
                "directory_sha256": meta["directory_sha256"],
            },
            npc_id + ": manifest adapter identity mismatch",
        )
        source_key = "g02_adapter" if npc_id == "V-G02-A" else "g09_adapter"
        need(
            raw["source"][source_key]["archive_sha256"]
            == meta["archive_sha256"]
            and raw["source"][source_key]["archive_bytes"]
            == meta["archive_bytes"]
            and raw["source"][source_key]["directory_sha256"]
            == meta["directory_sha256"],
            npc_id + ": raw adapter source mismatch",
        )
    need(
        raw["source"]["base_weight"]["sha256"]
        == cfg["base_model"]["weight_sha256"],
        "003B base weight source mismatch",
    )

    need(
        raw["environment"]["torch_threads"] == 4
        and raw["environment"]["cuda_available"] is False,
        "003B CPU/thread contract mismatch",
    )
    need(
        len(raw["model_turns"]) == cfg["runtime"]["model_turns"] == 3,
        "003B model turn count mismatch",
    )
    need(
        raw["total_generated_tokens"]
        == sum(x["generated_tokens"] for x in raw["model_turns"])
        <= cfg["runtime"]["max_generated_tokens_total"],
        "003B total token budget mismatch",
    )

    # Reconstruct the entire session from immutable stored raw responses and
    # the fixed sequence. No model/tokenizer is needed.
    session = session_api.start_session(
        case,
        cfg["session"]["participants"],
        cfg["session"]["session_key"],
    )
    turn_i = 0
    evidence_i = 0
    for step in cfg["session"]["sequence"]:
        if step["action"] == "evidence":
            need(
                evidence_i < len(raw["evidence_actions"]),
                "003B missing evidence action",
            )
            stored = raw["evidence_actions"][evidence_i]
            evidence_i += 1
            challenged = live.challenge_ids_for_g02(session, case, cfg)
            need(
                stored == {
                    "step_id": step["id"],
                    "npc_id": step["npc_id"],
                    "asset_id": step["asset_id"],
                    "challenge_policy": step["challenge_policy"],
                    "challenged_claim_ids": challenged,
                    "session_sha256_after": stored["session_sha256_after"],
                },
                "003B evidence action metadata mismatch",
            )
            session = session_api.present_evidence(
                case,
                session,
                step["npc_id"],
                step["asset_id"],
                challenged_claim_ids=challenged,
            )
            need(
                session["sha256"] == stored["session_sha256_after"],
                "003B evidence session hash mismatch",
            )
            continue

        need(turn_i < len(raw["model_turns"]), "003B missing model turn")
        stored = raw["model_turns"][turn_i]
        turn_i += 1
        witness = cfg["witnesses"][step["npc_id"]]
        need(
            stored["step_id"] == step["id"]
            and stored["npc_id"] == step["npc_id"]
            and stored["adapter_name"] == witness["adapter_name"]
            and stored["adapter_directory_sha256"]
                == witness["directory_sha256"],
            "003B turn/witness/adapter mapping mismatch",
        )
        need(
            stored["session_sha256_before"] == session["sha256"],
            "003B stored pre-turn session hash mismatch",
        )
        prompt_receipt = session_api.build_prompt(
            case,
            session,
            step["npc_id"],
            step["question"],
            include_history=step["include_history"],
        )
        need(
            stored["prompt_receipt"] == prompt_receipt,
            "003B prompt receipt does not reproduce",
        )
        rendered = live.render_session_prompt(prompt_receipt["prompt"], cfg)
        need(
            stored["rendered_user_prompt"] == rendered,
            "003B natural renderer does not reproduce",
        )
        need(
            stored["messages"] == [
                {
                    "role": "system",
                    "content": cfg["prompt_renderer"]["system"],
                },
                {"role": "user", "content": rendered},
            ],
            "003B model messages differ from fixed projection",
        )
        for forbidden in (
            "V-G02-A",
            "V-G09-C",
            "F-CLOCK",
            "F-SWAP",
            "F-CAMERA",
            "F-PRIVATE",
            "transfer-card",
            "camera-cache",
            "camera-console",
            "OXIDATION",
            "claim_id",
            "contradiction_id",
            "epistemic_class",
        ):
            need(
                forbidden.casefold()
                not in stored["rendered_user_prompt"].casefold(),
                "003B rendered prompt leaked private metadata: " + forbidden,
            )

        need(
            isinstance(stored["prompt_token_count"], int)
            and 0 < stored["prompt_token_count"]
            <= cfg["runtime"]["max_prompt_tokens"]
            and re.fullmatch(
                r"[0-9a-f]{64}", stored["prompt_token_sha256"]
            ) is not None,
            "003B prompt-token record invalid",
        )
        need(
            stored["generated_tokens"]
            == len(stored["generated_token_ids"])
            <= cfg["runtime"]["max_new_tokens_per_turn"],
            "003B generated-token record invalid",
        )
        need(
            stored["hit_max_new_tokens"]
            == (
                stored["generated_tokens"]
                == cfg["runtime"]["max_new_tokens_per_turn"]
            ),
            "003B truncation marker mismatch",
        )
        need(
            isinstance(stored["response"], str),
            "003B raw response missing",
        )
        extraction = live.extract_time_annotation(stored["response"], cfg)
        need(
            stored["annotation"] == extraction,
            "003B literal-time annotation does not replay",
        )
        session = session_api.record_turn(
            case,
            session,
            prompt_receipt,
            stored["response"],
            extraction["annotations"],
        )
        need(
            session["sha256"] == stored["session_sha256_after"],
            "003B post-turn session hash mismatch",
        )

    need(turn_i == 3, "003B reconstructed model turn count mismatch")
    need(
        evidence_i == len(raw["evidence_actions"]) == 1,
        "003B reconstructed evidence-action count mismatch",
    )
    derived = session_api.replay_session(case, session)
    need(
        session == raw["final_session"],
        "003B final session event log does not reconstruct",
    )
    need(
        derived == raw["final_derived"],
        "003B final derived replay does not reconstruct",
    )
    need(
        manifest["final_session_sha256"] == session["sha256"]
        and manifest["last_event_sha256"] == derived["last_event_sha256"]
        and manifest["claim_count"] == len(derived["claims"])
        and manifest["contradiction_count"] == len(derived["contradictions"]),
        "003B final manifest counts/hashes mismatch",
    )

    expected_case_sha = live.case_sha(case)
    need(
        raw["case_before_sha256"] == expected_case_sha
        == raw["case_after_sha256"],
        "003B canonical case hash changed",
    )
    need(
        raw["mandatory_public_routes_intact"] is True
        and manifest["case_unchanged"] is True
        and manifest["mandatory_public_routes_intact"] is True
        and live.mandatory_public_routes_intact(case),
        "003B mandatory public route/case integrity failed",
    )

    decision = raw["decision"]
    need(
        decision["integration_pass"] is True
        and decision["claim_count"] == len(derived["claims"])
        and decision["contradiction_count"] == len(derived["contradictions"])
        and decision["minimum_claim_count_required"] == 0
        and decision["minimum_contradiction_count_required"] == 0
        and decision["training_admitted"] is False
        and decision["body_selection_made"] is False
        and decision["population_expansion_admitted"] is False,
        "003B decision record widened authority or drifted",
    )

    need(
        provenance["repository"] == "techrote/neurounner"
        and provenance["head_sha"] == manifest["source_head"]
        and str(provenance["workflow_run_id"])
            == str(manifest["workflow_run_id"])
        and provenance["config_sha256"] == file_sha(live.CONFIG)
        and provenance["case_git_blob_sha1"]
            == cfg["authority"]["case"]["git_blob_sha1"]
        and provenance["session_module_git_blob_sha1"]
            == cfg["authority"]["session_module"]["git_blob_sha1"]
        and provenance["c2_release_git_blob_sha1"]
            == cfg["authority"]["c2_release"]["git_blob_sha1"]
        and provenance["c3_release_git_blob_sha1"]
            == cfg["authority"]["c3_release"]["git_blob_sha1"]
        and provenance["base_weight_sha256"]
            == cfg["base_model"]["weight_sha256"]
        and provenance["external_provider_calls"] == 0
        and provenance["training_steps"] == 0
        and provenance["relevance_mapping_accessed"] is False
        and provenance["learner_final_accessed"] is False,
        "003B provenance mismatch",
    )

    return {
        "verdict": "DETECTIVE_003B_LIVE_SESSION_READBACK_PASS",
        "model_turns": len(raw["model_turns"]),
        "generated_tokens": raw["total_generated_tokens"],
        "claim_count": len(derived["claims"]),
        "contradiction_count": len(derived["contradictions"]),
        "final_session_sha256": session["sha256"],
        "case_unchanged": True,
        "mandatory_public_routes_intact": True,
        "optimizer_steps": 0,
        "relevance_mapping_accessed": False,
        "population_expansion_admitted": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(verify(args.evidence), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
