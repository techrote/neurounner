import importlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
op = importlib.import_module("village_001e_recovery_operator")


def run(ident, number, **changes):
    value = {"id": ident, "run_number": number, "head_sha": str(number) * 40,
             "path": op.WORKFLOW, "head_branch": op.BRANCH, "status": "completed",
             "run_attempt": 1, "repository": {"id": op.REPO_ID, "full_name": op.REPO},
             "head_repository": {"id": op.REPO_ID, "full_name": op.REPO}}
    return {**value, **changes}


class RecoveryHistoryTests(unittest.TestCase):
    def setUp(self):
        self.current = run(102, 2, status="in_progress")
        self.previous = run(101, 1)
        self.req = {"manifest_sha256": "a" * 64, "original_run_id": op.ORIGINAL_RUN,
                    "execution_revision": 2, "resume_recovery_run_id": 101}
        self.old_req = {**self.req, "execution_revision": 1, "resume_recovery_run_id": None}
        self.environment = patch.dict(os.environ, {"GITHUB_RUN_ID": "102", "GITHUB_SHA": "2" * 40})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()

    def exercise(self, *, history=None, jobs=None, request=None, old_request=None):
        history = history if history is not None else [self.previous, self.current]
        jobs = jobs if jobs is not None else [{"id": 9, "name": "recover", "conclusion": "failure",
                                              "steps": [{"name": op.PROVIDER_STEP, "conclusion": "failure"}]}]
        def pages(resource, key):
            return history if key == "workflow_runs" else jobs
        with patch.object(op, "paged", side_effect=pages), patch.object(op.subprocess, "run"), \
                patch.object(op.subprocess, "check_output", return_value=json.dumps(old_request or self.old_req)):
            op.verify_run_history(request or self.req)

    def test_first_run_requires_no_prior_and_revision_one(self):
        self.exercise(history=[self.current], request={**self.req, "execution_revision": 1, "resume_recovery_run_id": None})

    def test_fresh_second_run_cannot_reset_prior_spending(self):
        with self.assertRaisesRegex(RuntimeError, "latest earlier provider execution"):
            self.exercise(request={**self.req, "resume_recovery_run_id": None})

    def test_latest_spending_source_is_accepted(self):
        self.exercise()

    def test_failed_offline_run_does_not_require_nonexistent_model_artifact(self):
        self.exercise(jobs=[{"id": 8, "name": "offline", "conclusion": "failure", "steps": []}], request={**self.req, "resume_recovery_run_id": None})

    def test_skipped_provider_step_does_not_spend(self):
        self.exercise(jobs=[{"id": 9, "name": "recover", "conclusion": "failure", "steps": [{"name": op.PROVIDER_STEP, "conclusion": "skipped"}]}], request={**self.req, "resume_recovery_run_id": None})

    def test_ambiguous_prior_step_cannot_be_called_zero_spending(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be established"):
            self.exercise(jobs=[{"id": 9, "name": "recover", "conclusion": "cancelled", "steps": []}])

    def test_prior_rerun_cannot_hide_first_attempt_spending(self):
        with self.assertRaisesRegex(RuntimeError, "one unambiguous attempt"):
            self.exercise(history=[{**self.previous, "run_attempt": 2}, self.current])

    def test_another_active_run_prevents_overlap(self):
        with self.assertRaisesRegex(RuntimeError, "pending or newer"):
            self.exercise(history=[{**self.previous, "status": "in_progress"}, self.current])

    def test_replacing_manifest_cannot_reset_budget(self):
        with self.assertRaisesRegex(RuntimeError, "another budget manifest"):
            self.exercise(old_request={**self.old_req, "manifest_sha256": "b" * 64})

    def test_revision_must_follow_latest_request(self):
        with self.assertRaisesRegex(RuntimeError, "advance exactly once"):
            self.exercise(request={**self.req, "execution_revision": 4})

    def test_foreign_repository_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "repository mismatch"):
            op.check_run(run(101, 1, repository={"id": 7, "full_name": "someone/else"}), run_id=101, workflow=op.WORKFLOW)


class RecoveryForestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.custody = self.root / "custody"
        self.custody.mkdir()
        self.rows = [{"artifact_name": "village001e-judge-tournament-F0" + str(i), "lane": "F0" + str(i), "status": "pending"} for i in range(1, 4)]
        self.manifest = {"lanes": self.rows}
        self.req = {"resume_recovery_run_id": None}
        self.original = self.root / "original"
        for row in self.rows:
            p = self.original / row["artifact_name"]
            p.mkdir(parents=True)
            (p / "original.txt").write_text(row["lane"])
        self.patches = [patch.object(op, "CUSTODY", self.custody), patch.object(op, "request", side_effect=lambda: (self.req, self.manifest, self.root / "config.json")), patch.object(op.recovery, "preserve_original")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_declared_prior_cannot_be_omitted_before_any_provider_call(self):
        self.req["resume_recovery_run_id"] = 101
        with patch.object(op.recovery, "run_lane") as lane:
            with self.assertRaisesRegex(RuntimeError, "declared prior recovery"):
                op.run_all(self.original, self.root / "out")
            lane.assert_not_called()

    def test_unsolicited_prior_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unsolicited"):
            op.run_all(self.original, self.root / "out", self.root / "prior")

    def test_prior_receipt_is_required(self):
        self.req["resume_recovery_run_id"] = 101
        prior = self.root / "prior"
        prior.mkdir()
        with self.assertRaisesRegex(RuntimeError, "custody receipt missing"):
            op.run_all(self.original, self.root / "out", prior)

    def test_entire_prior_forest_survives_early_interruption(self):
        self.req["resume_recovery_run_id"] = 101
        prior = self.root / "prior"
        shutil.copytree(self.original, prior / "inherited")
        # A lane completed late in an older invocation must survive even if
        # the new invocation is interrupted at its first lane.
        late = self.rows[-1]["artifact_name"]
        shutil.copytree(prior / "inherited" / late, prior / "lanes" / late)
        (prior / "lanes" / late / "older-provider-response.json").write_text('{"first_valid":true}\n')
        files = [{"path": name, **row} for name, row in op.recovery.tree(prior).items()]
        archive = self.root / "prior-recovery.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for p in prior.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(prior).as_posix())
        receipt = {"run": run(101, 1), "artifact": {"workflow_run": {"id": 101}, "size_in_bytes": archive.stat().st_size, "digest": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()}, "files": files}
        (self.custody / "resume-recovery-101.json").write_text(json.dumps(receipt))
        out = self.root / "out"
        def interrupt(*args, **kwargs):
            self.assertEqual(len(list((out / "inherited").iterdir())), 3)
            self.assertTrue((out / "inherited" / late / "older-provider-response.json").is_file())
            raise KeyboardInterrupt()
        with patch.object(op.recovery, "run_lane", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                op.run_all(self.original, out, prior)
        self.assertTrue((out / "inherited" / late / "older-provider-response.json").is_file())
        self.assertEqual((out / "custody/recovery-run-101.zip").read_bytes(), archive.read_bytes())
        self.assertTrue((out / "custody/resume-recovery-101.json").is_file())

    def test_partial_copy_cannot_discard_older_supplemental_history(self):
        self.req["resume_recovery_run_id"] = 101
        prior = self.root / "prior"
        shutil.copytree(self.original, prior / "inherited")
        name = self.rows[0]["artifact_name"]
        shutil.copytree(prior / "inherited" / name, prior / "lanes" / name)
        (prior / "inherited" / name / "older-supplemental-request.json").write_text('{}\n')
        files = [{"path": name, **row} for name, row in op.recovery.tree(prior).items()]
        receipt = {"run": run(101, 1), "artifact": {"workflow_run": {"id": 101}}, "files": files}
        (self.custody / "resume-recovery-101.json").write_text(json.dumps(receipt))
        with patch.object(op.recovery, "run_lane") as lane:
            with self.assertRaisesRegex(RuntimeError, "lost immutable inherited history"):
                op.run_all(self.original, self.root / "out", prior)
            lane.assert_not_called()

    def test_failed_started_tournament_stops_untouched_spending(self):
        for row in self.rows:
            row["kind"] = "tournament"
        self.rows[-1]["status"] = "unstarted"
        with patch.object(op.recovery, "run_lane", side_effect=RuntimeError("format remains incomplete")) as lane:
            with self.assertRaisesRegex(RuntimeError, "incomplete lanes"):
                op.run_all(self.original, self.root / "out")
            self.assertEqual(lane.call_count, 1)
        self.assertTrue((self.root / "out/recovery-failures.json").is_file())

    def test_prior_custody_byte_drift_stops_before_calls(self):
        self.req["resume_recovery_run_id"] = 101
        prior = self.root / "prior"
        shutil.copytree(self.original, prior / "inherited")
        receipt = {"run": run(101, 1), "artifact": {"workflow_run": {"id": 101}}, "files": []}
        (self.custody / "resume-recovery-101.json").write_text(json.dumps(receipt))
        with patch.object(op.recovery, "run_lane") as lane:
            with self.assertRaisesRegex(RuntimeError, "verified ZIP receipt"):
                op.run_all(self.original, self.root / "out", prior)
            lane.assert_not_called()


if __name__ == "__main__":
    unittest.main()
