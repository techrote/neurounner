#!/usr/bin/env python3
"""NCR-001F event-aware, exact-head Actions evidence audit. No model or training.

Audit the complete historical head-SHA run inventory instead of treating a green
unrelated PR check as domain qualification or losing source-specific push jobs.
Live GitHub metadata is read-only; all scientific admission remains blocked.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO = "techrote/neurounner"
TARGETS = {
    99: {
        "head": "463b55410a1eb9f962960c605e7fa52844ffc2f5",
        "branch": "review/ncr-001f-void-byte-mlp-real-20261027",
        "domain_run": 40818768640,
        "domain_path": ".github/workflows/ncr-001f-void-archive.yml",
        "domain_job": 123266025303,
        "domain_job_name": "Independent original-run JSON/SHA/causality and false-success sentinels",
        "unrelated_run": 40818773779,
        "unrelated_job": 123266042116,
    },
    198: {
        "head": "5ea19b3f9b04fa8542ff3d4b9e70882a811522a9",
        "branch": "review/ncr001f-corrected-cpu-custody-20261029",
        "domain_run": 40875571415,
        "domain_path": ".github/workflows/ncr001f-corrected-cpu-archival-bootstrap.yml",
        "domain_job": 123453388884,
        "domain_job_name": "original-custody-readonly",
        "unrelated_run": 40875664960,
        "unrelated_job": 123453711985,
    },
}
UNRELATED_PATH = ".github/workflows/village-001-preflight.yml"


class EvidenceError(RuntimeError):
    """Provenance or source-scope mismatch; no scientific qualification allowed."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def validate(pr_number: int, pull: dict, inventory: dict, jobs_by_run: dict) -> dict:
    """Require every expected run/job to be exact-head, successful and correctly scoped."""
    need(pr_number in TARGETS, "unknown target PR")
    t = TARGETS[pr_number]
    need(type(pull) is dict and type(pull.get("number")) is int
         and pull["number"] == pr_number, "PR number mismatch")
    need(pull.get("state") == "open" and pull.get("draft") is True,
         "PR is not the frozen open draft")
    need(isinstance(pull.get("head"), dict) and
         pull["head"].get("sha") == t["head"] and
         pull["head"].get("ref") == t["branch"], "PR head or branch changed")
    need(type(inventory) is dict and type(inventory.get("total_count")) is int,
         "missing full head-run inventory count")
    runs = inventory.get("workflow_runs")
    need(type(runs) is list and len(runs) == inventory["total_count"],
         "incomplete/paginated head-run inventory")
    expected = {t["domain_run"], t["unrelated_run"]}
    ids = [r.get("id") for r in runs if type(r) is dict]
    need(len(runs) == 2 and len(ids) == 2 and
         all(type(i) is int for i in ids) and set(ids) == expected,
         "unexpected, missing or duplicate head run")
    entries = {r["id"]: r for r in runs}
    for key, event, path in (
        ("domain_run", "push", t["domain_path"]),
        ("unrelated_run", "pull_request", UNRELATED_PATH),
    ):
        run = entries[t[key]]
        need(run.get("head_sha") == t["head"] and
             run.get("head_branch") == t["branch"], "cross-head or cross-branch run")
        source_repo = run.get("head_repository")
        need(type(source_repo) is dict and source_repo.get("full_name") == REPO,
             "foreign repository run")
        need(run.get("event") == event and run.get("path") == path,
             "event/workflow path misattribution")
        need(run.get("status") == "completed" and run.get("conclusion") == "success" and
             run.get("run_attempt") == 1, "non-success or rerun not pinned")
    need(type(jobs_by_run) is dict and set(jobs_by_run) == expected,
         "run-job evidence incomplete")
    for key, job_id, job_name in (
        ("domain_run", t["domain_job"], t["domain_job_name"]),
        ("unrelated_run", t["unrelated_job"], "population-contract"),
    ):
        job_collection = jobs_by_run[t[key]]
        need(type(job_collection) is dict and
             type(job_collection.get("total_count")) is int and
             job_collection["total_count"] == 1, "job inventory incomplete")
        jobs = job_collection.get("jobs")
        need(type(jobs) is list and len(jobs) == 1, "missing or duplicate job")
        job = jobs[0]
        need(type(job) is dict, "invalid job object")
        need(type(job.get("id")) is int and
             job["id"] == job_id and job.get("name") == job_name,
             "job identity mismatch")
        need(job.get("head_sha") == t["head"] and
             job.get("status") == "completed" and job.get("conclusion") == "success",
             "wrong-head/skipped/failed job")
    return {
        "pr": pr_number, "exact_original_head": t["head"],
        "domain_push_run": t["domain_run"], "domain_push_job": t["domain_job"],
        "domain_workflow": t["domain_path"],
        "unrelated_pr_run": t["unrelated_run"],
        "unrelated_pr_workflow": UNRELATED_PATH,
        "head_run_inventory_complete": True,
        "domain_push_source_audit_verified": True,
        "domain_pull_request_check_present": False,
        "original_physical_model_reexecuted_in_this_audit": False,
        "classification": "EXACT_HEAD_PUSH_CUSTODY_PASS__NO_PR_DOMAIN_CHECK__NO_ADMISSION",
    }


def github_json(path: str, token: str) -> dict:
    url = "https://api.github.com/repos/" + REPO + path
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "neurounner-ncr001f-event-auditor"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urlopen(Request(url, headers=headers), timeout=25) as response:
            return json.load(response)
    except (HTTPError, URLError, OSError, ValueError) as exc:
        raise EvidenceError("GitHub read failed (no admission): " + str(exc)) from exc


def collect(pr_number: int, token: str) -> dict:
    t = TARGETS[pr_number]
    pull = github_json("/pulls/" + str(pr_number), token)
    query = urlencode({"head_sha": t["head"], "per_page": 100})
    inventory = github_json("/actions/runs?" + query, token)
    # Exactly two known historical runs at the frozen head. Refuse future drift,
    # even if a new green badge appears; do not make a partial snapshot pass.
    jobs = {run_id: github_json(f"/actions/runs/{run_id}/jobs?per_page=100", token)
            for run_id in (t["domain_run"], t["unrelated_run"])}
    return validate(pr_number, pull, inventory, jobs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "scope": "NCR-001F #73 SOURCE ONLY; historical exact-head Action-event readback",
        "repo": REPO,
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "actual_compute_budget_qualified": False,
        "model_executed": False,
        "optimizer_steps": 0,
        "heldouts_opened": False,
        "scientific_admission": False,
        "results": [],
    }
    try:
        for number in sorted(TARGETS):
            report["results"].append(collect(number, os.getenv("GH_TOKEN", "")))
        report["status"] = "PASS_EXACT_HEAD_ACTION_EVENT_ATTRIBUTION__TRAINING_BLOCKED"
        code = 0
    except EvidenceError as exc:
        report["status"] = "FAIL_CLOSED_EXACT_HEAD_ACTION_EVENT_ATTRIBUTION"
        report["error"] = str(exc)
        code = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
