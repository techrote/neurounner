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
    pull = {"number": number, "state": "open", "merged": False, "draft": True,
            "head": {"sha": t["head"], "ref": t["branch"], "repo": {"full_name": REPO}}}
    runs = {"total_count": 2, "workflow_runs": [
        run(t["domain_run"], "push", t["domain_path"]),
        run(t["unrelated_run"], "pull_request", UNRELATED_PATH),
    ]}
    jobs = {
        t["domain_run"]: job(t["domain_job"], t["domain_job_name"]),
        t["unrelated_run"]: job(t["unrelated_job"], "population-contract"),
    }
    original = {"sha": t["head"], "tree": {"sha": t["original_tree"]},
                "parents": [{"sha": t["original_parent"]}]}
    return pull, runs, jobs, original


class EventEvidenceTests(unittest.TestCase):
    def test_both_pinned_original_heads_pass_with_conservative_scope(self):
        for number in TARGETS:
            with self.subTest(pr=number):
                result = validate(number, *fixture(number))
                self.assertTrue(result["domain_push_source_audit_verified"])
                self.assertIs(result["domain_pull_request_check_present"], False)
                self.assertTrue(result["original_source_commit_verified"])
                self.assertFalse(result["domain_pull_request_check_verified"])
                self.assertEqual(result["additional_runs_unreviewed"], [])
                self.assertFalse(result["original_physical_model_reexecuted_in_this_audit"])
                self.assertIn("NO_ADMISSION", result["classification"])

    def refuses(self, edit, contains, number=99):
        data = copy.deepcopy(fixture(number))
        edit(*data[:3])
        with self.assertRaisesRegex(EvidenceError, contains):
            validate(number, *data)

    def test_original_source_remains_valid_after_pr_head_recomposition(self):
        p, r, j, c = fixture()
        p["head"]["sha"] = "f" * 40
        p["state"], p["draft"], p["merged"] = "closed", False, True
        report = validate(99, p, r, j, c)
        self.assertFalse(report["original_head_still_current_pr_head"])
        self.assertTrue(report["current_pr_merged"])
        self.assertFalse(report["domain_pull_request_check_verified"])

    def test_malformed_current_pr_head_is_refused(self):
        self.refuses(lambda p, r, j: p["head"].update(sha="f" * 39),
                     "malformed current PR head")
    def test_wrong_pr_branch(self):
        self.refuses(lambda p, r, j: p["head"].update(ref="main"), "head or branch")
    def test_ready_or_closed_pr_does_not_erase_original_evidence(self):
        for state, draft, merged in (("open", False, False),
                                     ("closed", False, True),
                                     ("closed", True, False)):
            with self.subTest(state=state, draft=draft, merged=merged):
                p, r, j, c = fixture()
                p.update(state=state, draft=draft, merged=merged)
                report = validate(99, p, r, j, c)
                self.assertEqual(report["current_pr_state"], state)
                self.assertFalse(report["domain_pull_request_check_verified"])

    def test_merged_pr_deleted_head_repository_retains_original_source_proof(self):
        p, r, j, c = fixture()
        p.update(state="closed", merged=True, draft=False)
        p["head"]["repo"] = None
        report = validate(99, p, r, j, c)
        self.assertTrue(report["original_source_commit_verified"])
        self.assertTrue(report["current_pr_merged"])
        self.assertFalse(report["domain_pull_request_check_verified"])

    def test_unmerged_pr_with_missing_source_repo_refused(self):
        self.refuses(lambda p, r, j: p["head"].update(repo=None),
                     "missing current PR repository")

    def test_inconsistent_merged_open_pr_is_refused(self):
        self.refuses(lambda p, r, j: p.update(merged=True),
                     "lifecycle state")
    def test_incomplete_pagination(self):
        self.refuses(lambda p, r, j: r.update(total_count=3), "incomplete")

    def test_new_unreviewed_pr_domain_run_does_not_replace_original_push(self):
        p, r, j, c = fixture()
        new = copy.deepcopy(r["workflow_runs"][1])
        new.update(id=987654321, path=TARGETS[99]["domain_path"])
        r["workflow_runs"].append(new)
        r["total_count"] = 3
        report = validate(99, p, r, j, c)
        self.assertEqual(report["additional_runs_unreviewed"], [987654321])
        self.assertIsNone(report["domain_pull_request_check_present"])
        self.assertFalse(report["domain_pull_request_check_verified"])
        self.assertEqual(report["observed_head_run_count"], 3)

    def test_new_failed_unreviewed_run_does_not_rewrite_original_result(self):
        p, r, j, c = fixture(198)
        new = copy.deepcopy(r["workflow_runs"][1])
        new.update(id=987654322, status="completed", conclusion="failure")
        r["workflow_runs"].append(new)
        r["total_count"] = 3
        report = validate(198, p, r, j, c)
        self.assertEqual(report["additional_runs_unreviewed"], [987654322])
        self.assertFalse(report["domain_pull_request_check_verified"])

    def test_forged_original_source_tree_and_parent_are_refused(self):
        for field in ("tree", "parents"):
            with self.subTest(field=field):
                p, r, j, c = fixture()
                if field == "tree":
                    c["tree"]["sha"] = "f" * 40
                else:
                    c["parents"][0]["sha"] = "f" * 40
                with self.assertRaisesRegex(EvidenceError, "tree or ancestry"):
                    validate(99, p, r, j, c)

    def test_forged_original_commit_sha_is_refused(self):
        p, r, j, c = fixture()
        c["sha"] = "f" * 40
        with self.assertRaisesRegex(EvidenceError, "commit SHA"):
            validate(99, p, r, j, c)
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
    def test_foreign_current_pr_repository_is_refused(self):
        self.refuses(lambda p, r, j: p["head"]["repo"].update(full_name="other/repo"),
                     "foreign PR source repository")

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
