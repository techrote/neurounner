#!/usr/bin/env python3
"""Read-only NCR-001F synthetic cross-wrapper CI log custody auditor.

Original CI source head is pinned independently of this later archival commit.
This reads Git blobs/log text only: no torch/model import, optimizer or holdouts.
Git object IDs cover complete, unnormalized file bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/ncr-001/controls-evidence/shared-backbone-ownership"
PROVENANCE = ARCHIVE / "provenance.json"
EXPECTED_SOURCE = "fd4b0c3731af029b68589205a79a7488bdca74a0"
EXPECTED_RUN = 40652259144
EXPECTED_JOB = 122694854315
EXPECTED_LOG_GIT_SHA1 = "0e5ebeaf933a8ff54d5988de898769112e8dd2ed"
CASES = (
    "test_second_wrapper_cannot_install_overlapping_hooks",
    "test_disabled_reference_paths_also_refuse_foreign_active_hooks",
    "test_distinct_frozen_backbones_remain_independently_usable",
    "test_exception_always_releases_shared_owner_and_detaches_hooks",
    "test_same_wrapper_lock_behavior_preserved",
)


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError("NCR-001F original synthetic CI archive REFUSED: " + why)


def git_blob_sha1(path: Path) -> str:
    require(path.is_file() and path.resolve().is_relative_to(ROOT.resolve()),
            "missing or out-of-repository source blob")
    return subprocess.check_output(
        ["git", "hash-object", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()


def audit(*, record: dict | None = None, raw_log: bytes | None = None) -> dict:
    p = json.loads(PROVENANCE.read_text(encoding="utf-8")) if record is None else record
    log_data = (ARCHIVE / "ci-40652259144-job-122694854315.log").read_bytes() if raw_log is None else raw_log
    rec = p["recorded_log_fetch"]
    require(p["schema_version"] == 1 and p["experiment"] == "NCR-001F"
            and p["issue"] == 73 and p["review_candidate_pr"] == 151
            and p["original_source_pr"] == 96
            and p["original_source_head"] ==
            "f4f4b5becaebf17d89f11dc8db0952c31cbe451f",
            "record source/PR identity changed")
    require(p["exact_tested_head"] == EXPECTED_SOURCE
            and rec["source_sha_in_log"] == EXPECTED_SOURCE
            and rec["workflow_run_id"] == EXPECTED_RUN
            and rec["workflow_job_id"] == EXPECTED_JOB
            and rec["job_status"] == "completed"
            and rec["job_conclusion"] == "success"
            and rec["git_blob_sha1"] == EXPECTED_LOG_GIT_SHA1
            and rec["path"] ==
            "experiments/ncr-001/controls-evidence/shared-backbone-ownership/"
            "ci-40652259144-job-122694854315.log",
            "original exact-head Actions job provenance changed")
    require(git_blob_sha1(ARCHIVE / "ci-40652259144-job-122694854315.log")
            == EXPECTED_LOG_GIT_SHA1,
            "committed original decoded log bytes no longer match their Git blob")
    # When passed an altered in-memory fixture, do not substitute the on-disk
    # blob for the fixture: compare exact complete SHA-1 bytes through git below.
    require(raw_log is None or raw_log == (
            ARCHIVE / "ci-40652259144-job-122694854315.log").read_bytes(),
            "supplied CI log bytes differ from accepted original")
    log = log_data.decode("utf-8-sig")
    require("NCR_EXPECTED_HEAD: " + EXPECTED_SOURCE in log
            and "ref: " + EXPECTED_SOURCE in log,
            "CI did not check out original source commit exactly")
    require(log.count("Ran 7 tests") == 1 and log.count("Ran 5 tests") == 1,
            "expected historical 7+5 source-only tests were not recorded")
    require(log.count("\n2026-10-28T07:47:05.0204452Z OK") == 1
            and log.count("\n2026-10-28T07:47:06.0331808Z OK") == 1,
            "original two unittest success terminators missing")
    for name in CASES:
        require(name in log and (name + " (") in log,
                "original negative/refusal case missing: " + name)

    # This archive witnesses a historical execution. Later source fixes must
    # never be compared against the old tested source; reconstruct the original
    # immutable Git tree and fail if its ancestry is missing (CI fetch-depth: 0).
    original_workflow = ".github/workflows/ncr-001f-shared-backbone-ownership.yml"
    require(set(p["source_git_blobs_sha1"]) == {
        "scripts/ncr_001f_frozen_hooks.py",
        "tests/test_ncr_001f_frozen_hooks.py",
        "tests/test_ncr_001f_shared_backbone_ownership.py",
        original_workflow,
        "experiments/ncr-001/protocol.json",
        "experiments/ncr-001/source-pins.json",
    }, "original source inventory is incomplete or has extra files")
    for rel, expected in p["source_git_blobs_sha1"].items():
        try:
            original_blob = subprocess.check_output(
                ["git", "rev-parse", "--verify", f"{EXPECTED_SOURCE}:{rel}"],
                cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "NCR-001F original synthetic CI archive REFUSED: "
                "original tested Git source unavailable: " + rel
            ) from exc
        require(original_blob == expected,
                "historical tested source Git blob differs: " + rel)
        if rel == original_workflow:
            require(git_blob_sha1(ARCHIVE / "original-tested-workflow.yml") == expected,
                    "preserved original tested workflow source differs")
    require(p["source_git_blobs_sha1"]["experiments/ncr-001/protocol.json"]
            == "14ce5fbaa38f8806c8c8738f19c01b91385eebaf",
            "frozen protocol blob substitution")
    require(p["source_git_blobs_sha1"]["experiments/ncr-001/source-pins.json"]
            == "59fff1aed872b5563aa6898171983973e0cdb02c",
            "frozen source pins blob substitution")
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    require(protocol["train_authorized"] is False
            and set(protocol["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "training permission/four frozen-model separation changed")
    result = p["observations"]
    require(result["old_synthetic_tests_passed"] == 7
            and result["new_shared_backbone_synthetic_tests_passed"] == 5
            and result["actual_frozen_model_loaded"] is False
            and result["optimizer_steps"] == 0
            and result["heldout_fixtures_accessed"] is False
            and result["independent_actual_host_conformance"] is False
            and result["source_only_test_result_is_not_independent_final_review"] is True,
            "source-only test outcomes falsely upgraded")
    return {
        "status": "PASS_ORIGINAL_EXACT_HEAD_SYNTHETIC_GUARD_CI_READBACK_ONLY",
        "run_id": EXPECTED_RUN, "job_id": EXPECTED_JOB,
        "tested_source_commit": EXPECTED_SOURCE,
        "source_only_tests": 12,
        "model_execution": False,
        "train_authorized": False,
        "final_72_or_73_accepted": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
