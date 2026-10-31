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
        "original_tree": "9edbb5a090e6f6cf3d2d267a3db2d806e1421859",
        "original_parent": "325365d10b68392f37e40ea3cf0e9970b8eb3fb6",
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
        "original_tree": "0a37ddeb1566140750d21cbf6f9e784aa3d1005a",
        "original_parent": "c59add922f46c43c9cf4e6bc3342b334c0a3926a",
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


def validate(pr_number: int, pull: dict, inventory: dict,
             jobs_by_run: dict, source_commit: dict) -> dict:
    """Verify immutable ORIGINAL Actions evidence; classify mutable PR state.

    A later merge/recomposition or unrelated workflow cannot erase the two
    pinned historical runs. It ALSO cannot turn a new check into accepted NCR
    qualification. The complete current inventory is still required to make
    unreviewed additional runs visible rather than silently dropping them.
    """
    need(pr_number in TARGETS, "unknown target PR")
    t = TARGETS[pr_number]
    need(type(pull) is dict and type(pull.get("number")) is int
         and pull["number"] == pr_number, "PR number mismatch")
    head = pull.get("head")
    need(type(head) is dict and head.get("ref") == t["branch"],
         "PR head or branch changed")
    current_sha = head.get("sha")
    need(type(current_sha) is str and len(current_sha) == 40
         and all(c in "0123456789abcdef" for c in current_sha),
         "malformed current PR head")
    state = pull.get("state")
    merged = pull.get("merged")
    need(state in ("open", "closed") and type(merged) is bool
         and not (state == "open" and merged),
         "invalid current PR lifecycle state")
    head_repo = head.get("repo")
    # GitHub can null the current head repo when a merged PR branch is deleted.
    # Only the verified original Git commit/run/job establishes historic source;
    # do not infer that the inaccessible current branch has any qualified CI.
    if head_repo is None:
        need(state == "closed" and merged is True,
             "missing current PR repository without verified merge state")
    else:
        need(type(head_repo) is dict
             and head_repo.get("full_name") == REPO,
             "foreign PR source repository")
    need(type(pull.get("draft")) is bool, "invalid current PR draft status")

    # This ORIGINAL Git commit remains immutable even after GitHub mutates PR
    # readiness, base, branch head, merge state or ref-based workflow inventory.
    # Pin both original tree and first parent to distinguish a fabricated
    # unrelated SHA/source from the historic run-under-review.
    need(type(source_commit) is dict
         and source_commit.get("sha") == t["head"], "original source commit SHA")
    tree = source_commit.get("tree")
    parents = source_commit.get("parents")
    need(type(tree) is dict and tree.get("sha") == t["original_tree"]
         and type(parents) is list and len(parents) >= 1
         and type(parents[0]) is dict
         and parents[0].get("sha") == t["original_parent"],
         "original source tree or ancestry drift")

    need(type(inventory) is dict and type(inventory.get("total_count")) is int
         and 0 <= inventory["total_count"] <= 100,
         "missing or paginated all-event inventory")
    runs = inventory.get("workflow_runs")
    need(type(runs) is list and len(runs) == inventory["total_count"],
         "incomplete/paginated head-run inventory")
    ids = [r.get("id") for r in runs if type(r) is dict]
    need(len(ids) == len(runs)
         and all(type(i) is int for i in ids)
         and len(set(ids)) == len(ids),
         "unexpected, missing or duplicate head run")
    expected = {t["domain_run"], t["unrelated_run"]}
    need(expected.issubset(ids), "unexpected or missing pinned original head run")
    entries = {r["id"]: r for r in runs}
    # Even unreviewed new runs must be on the same original SHA, branch and
    # repository, but their success must NEVER be accepted as domain evidence.
    for run in runs:
        need(run.get("head_sha") == t["head"]
             and run.get("head_branch") == t["branch"],
             "cross-head or cross-branch run")
        source_repo = run.get("head_repository")
        need(type(source_repo) is dict
             and source_repo.get("full_name") == REPO,
             "foreign repository run")
    for key, event, path in (
        ("domain_run", "push", t["domain_path"]),
        ("unrelated_run", "pull_request", UNRELATED_PATH),
    ):
        run = entries[t[key]]
        need(run.get("event") == event and run.get("path") == path,
             "event/workflow path misattribution")
        need(run.get("status") == "completed"
             and run.get("conclusion") == "success"
             and type(run.get("run_attempt")) is int
             and run["run_attempt"] == 1,
             "non-success or rerun not pinned")

    need(type(jobs_by_run) is dict and set(jobs_by_run) == expected,
         "run-job evidence incomplete")
    for key, job_id, job_name in (
        ("domain_run", t["domain_job"], t["domain_job_name"]),
        ("unrelated_run", t["unrelated_job"], "population-contract"),
    ):
        job_collection = jobs_by_run[t[key]]
        need(type(job_collection) is dict
             and type(job_collection.get("total_count")) is int
             and job_collection["total_count"] == 1, "job inventory incomplete")
        jobs = job_collection.get("jobs")
        need(type(jobs) is list and len(jobs) == 1, "missing or duplicate job")
        job = jobs[0]
        need(type(job) is dict, "invalid job object")
        need(type(job.get("id")) is int
             and job["id"] == job_id and job.get("name") == job_name,
             "job identity mismatch")
        need(job.get("head_sha") == t["head"]
             and job.get("status") == "completed"
             and job.get("conclusion") == "success",
             "wrong-head/skipped/failed job")

    unreviewed = sorted(set(ids) - expected)
    original_is_current = current_sha == t["head"]
    return {
        "pr": pr_number, "exact_original_head": t["head"],
        "original_source_commit_verified": True,
        "original_source_tree": t["original_tree"],
        "current_pr_head": current_sha,
        "original_head_still_current_pr_head": original_is_current,
        "current_pr_state": state, "current_pr_draft": pull["draft"],
        "current_pr_merged": merged,
        "domain_push_run": t["domain_run"], "domain_push_job": t["domain_job"],
        "domain_workflow": t["domain_path"],
        "unrelated_pr_run": t["unrelated_run"],
        "unrelated_pr_workflow": UNRELATED_PATH,
        "head_run_inventory_complete": True,
        "observed_head_run_count": len(runs),
        "additional_runs_unreviewed": unreviewed,
        "domain_push_source_audit_verified": True,
        # If new runs exist, it is UNKNOWN whether one is a domain PR check.
        # Either way none qualifies this frozen original-domain proof.
        "domain_pull_request_check_present": (False if not unreviewed else None),
        "domain_pull_request_check_verified": False,
        "original_physical_model_reexecuted_in_this_audit": False,
        "classification": "EXACT_ORIGINAL_SHA_PUSH_CUSTODY_PASS__"
                          "PR_LIFECYCLE_INDEPENDENT__EXTRA_RUNS_UNQUALIFIED__NO_ADMISSION",
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
    original_commit = github_json("/git/commits/" + t["head"], token)
    query = urlencode({"head_sha": t["head"], "per_page": 100})
    inventory = github_json("/actions/runs?" + query, token)
    # Only the two PINNED historical runs qualify. A complete all-event query
    # also exposes any later runs, but they are marked unreviewed and excluded.
    jobs = {run_id: github_json(f"/actions/runs/{run_id}/jobs?per_page=100", token)
            for run_id in (t["domain_run"], t["unrelated_run"])}
    return validate(pr_number, pull, inventory, jobs, original_commit)


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
