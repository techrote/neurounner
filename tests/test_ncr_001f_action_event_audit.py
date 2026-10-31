"""Offline adversarial tests for event/branch/head/job attribution (NO MODEL)."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ncr_001f_action_event_audit import (TARGETS, REPO, UNRELATED_PATH,
                                          EvidenceError, validate)


def fixture(number: int = 99):
    t = TARGETS[number]
    def run(id_, event, path):
        return {"id": id_, "event": event, "path": path, "head_sha": t["head"],
                "head_branch": t["branch"], "head_repository": {"full_name": REPO},
                "status": "completed", "conclusion": "success", "run_attempt": 1}
    def job(id_, name):
        return {"total_count": 1, "jobs": [{"id": id_, "name": name,
                "head_sha": t["head"], "status": "completed", "conclusion": "success"}]}
    pull = {"number": number, "state": "open", "draft": True,
            "head": {"sha": t["head"], "ref": t["branch"]}}
    runs = {"total_count": 2, "workflow_runs": [
        run(t["domain_run"], "push", t["domain_path"]),
        run(t["unrelated_run"], "pull_request", UNRELATED_PATH),
    ]}
    jobs = {
        t["domain_run"]: job(t["domain_job"], t["domain_job_name"]),
        t["unrelated_run"]: job(t["unrelated_job"], "population-contract"),
    }
    return pull, runs, jobs


class EventEvidenceTests(unittest.TestCase):
    def test_both_pinned_original_heads_pass_with_conservative_scope(self):
        for number in TARGETS:
            with self.subTest(pr=number):
                result = validate(number, *fixture(number))
                self.assertTrue(result["domain_push_source_audit_verified"])
                self.assertFalse(result["domain_pull_request_check_present"])
                self.assertFalse(result["original_physical_model_reexecuted_in_this_audit"])
                self.assertIn("NO_ADMISSION", result["classification"])

    def refuses(self, edit, contains, number=99):
        data = copy.deepcopy(fixture(number))
        edit(*data)
        with self.assertRaisesRegex(EvidenceError, contains):
            validate(number, *data)

    def test_wrong_pr_head(self):
        self.refuses(lambda p, r, j: p["head"].update(sha="f" * 40), "head or branch")
    def test_wrong_pr_branch(self):
        self.refuses(lambda p, r, j: p["head"].update(ref="main"), "head or branch")
    def test_promoted_or_closed_draft(self):
        self.refuses(lambda p, r, j: p.update(draft=False), "open draft")
    def test_incomplete_pagination(self):
        self.refuses(lambda p, r, j: r.update(total_count=3), "incomplete")
    def test_duplicate_run_id(self):
        self.refuses(lambda p, r, j: r["workflow_runs"].__setitem__(
            1, copy.deepcopy(r["workflow_runs"][0])), "unexpected")
    def test_extra_or_unknown_run(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0].update(id=123), "unexpected")
    def test_wrong_head_on_domain_push(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0].update(head_sha="0" * 40),
                     "cross-head")
    def test_foreign_repository(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0]["head_repository"].update(
            full_name="another/repo"), "foreign repository")
    def test_treating_village_pr_as_domain_check(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][1].update(
            path=TARGETS[99]["domain_path"]), "event/workflow")
    def test_wrong_event_on_source_domain(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0].update(event="pull_request"),
                     "event/workflow")
    def test_failed_domain_run(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0].update(conclusion="failure"),
                     "non-success")
    def test_modified_run_attempt(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0].update(run_attempt=2),
                     "non-success")
    def test_mismatched_job_identity(self):
        def edit(p, r, j):
            j[TARGETS[99]["domain_run"]]["jobs"][0]["id"] = 9
        self.refuses(edit, "job identity")
    def test_skipped_custody_job(self):
        def edit(p, r, j):
            j[TARGETS[99]["domain_run"]]["jobs"][0]["conclusion"] = "skipped"
        self.refuses(edit, "skipped")
    def test_partial_job_inventory(self):
        def edit(p, r, j):
            j[TARGETS[99]["domain_run"]]["total_count"] = 2
        self.refuses(edit, "job inventory")
    def test_null_repository_metadata_fails_closed(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0].update(
            head_repository=None), "foreign repository")
    def test_null_job_data_fails_closed(self):
        def edit(p, r, j):
            j[TARGETS[99]["domain_run"]]["jobs"][0] = None
        self.refuses(edit, "invalid job object")
    def test_tampered_198_workflow(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0].update(
            path=".github/workflows/village-001-preflight.yml"),
            "event/workflow", number=198)
    def test_boolean_id_rejected(self):
        self.refuses(lambda p, r, j: r["workflow_runs"][0].update(id=True),
                     "unexpected")
    def test_wrong_198_job_head(self):
        def edit(p, r, j):
            j[TARGETS[198]["domain_run"]]["jobs"][0]["head_sha"] = "1" * 40
        self.refuses(edit, "wrong-head", number=198)


if __name__ == "__main__":
    unittest.main()
