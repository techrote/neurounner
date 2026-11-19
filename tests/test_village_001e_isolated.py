import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
operator = importlib.import_module("village_001e_isolated")


class IsolatedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        source = Path(__file__).resolve().parents[1]
        for name in ("experiments/village-001/preference-evidence/summary.json", "experiments/village-001/preference-evidence/adapter-release.json", "experiments/village-001/tournament-config.json", *(str(operator.REL / n) for n in ("request.json", "config.json", "final-eval.json"))):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        self.campaign = self.root / operator.REL
        req = json.loads((self.campaign / "request.json").read_text())
        req.update(phase="generate", resume_run_id=None)
        (self.campaign / "request.json").write_text(json.dumps(req))
        self.patches = [patch.object(operator, "ROOT", self.root), patch.object(operator, "CAMPAIGN", self.campaign), patch.dict(os.environ, {"GITHUB_REPOSITORY": operator.REPO, "GITHUB_REF_NAME": operator.BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_SHA": "a" * 40, "GITHUB_OUTPUT": str(self.root / "output"), "RUNNER_TEMP": str(self.root)})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_valid_plan_has_one_serial_matrix_and_all_generations(self):
        with patch.object(operator.subprocess, "check_output", side_effect=["a" * 40 + " refs/heads/x", ""]), patch.object(operator.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr="gh: Not Found (HTTP 404)")):
            operator.validate_live_request()
        outputs = dict(line.split("=", 1) for line in (self.root / "output").read_text().splitlines())
        self.assertEqual(len(json.loads(outputs["members"])), 37)
        self.assertEqual(len(json.loads(outputs["ecology"])), 24)
        self.assertEqual(len(json.loads(outputs["providers"])["include"]), 48)

    def test_cli_subcommands_parse_without_argument_conflicts(self):
        script = Path(__file__).resolve().parents[1] / "scripts/village_001e_isolated.py"
        for command in ("recover-lane", "attach-custody", "asset-list"):
            result = subprocess.run([sys.executable, str(script), command, "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_repeated_custody_attachments_preserve_prior_receipts(self):
        out = self.root / "lane"
        out.mkdir()
        receipt = self.root / "custody.json"
        receipt.write_text('{"run_id":1}\n')
        operator.attach_custody(receipt, out)
        first = list(out.iterdir())[0]
        before = first.read_bytes()
        receipt.write_text('{"run_id":2}\n')
        operator.attach_custody(receipt, out)
        operator.attach_custody(receipt, out)
        self.assertEqual(len(list(out.iterdir())), 2)
        self.assertEqual(first.read_bytes(), before)

    def test_parent_byte_drift_rejected(self):
        with (self.root / "experiments/village-001/preference-evidence/summary.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "D summary changed"):
            operator.request()

    def test_frozen_gate_change_rejected(self):
        path = self.campaign / "config.json"
        cfg = json.loads(path.read_text())
        cfg["tournament"]["individual_quality_required"] = 1
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "gate drift"):
            operator.request()

    def test_moved_branch_stops_before_publication_lookup(self):
        with patch.object(operator.subprocess, "check_output", return_value="b" * 40 + " refs/heads/x"), patch.object(operator.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "branch moved"):
                operator.validate_live_request()
            run.assert_not_called()

    def test_existing_tag_stops_before_execution(self):
        with patch.object(operator.subprocess, "check_output", side_effect=["a" * 40 + " refs/heads/x", "b" * 40 + " refs/tags/x"]):
            with self.assertRaisesRegex(ValueError, "tag already exists"):
                operator.validate_live_request()

    def test_ambiguous_release_error_is_not_absence(self):
        with patch.object(operator.subprocess, "check_output", side_effect=["a" * 40 + " refs/heads/x", ""]), patch.object(operator.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr="gh: gateway error (HTTP 502)")):
            with self.assertRaisesRegex(ValueError, "absence cannot"):
                operator.validate_live_request()

    def test_completed_generation_cannot_be_silently_regenerated(self):
        (self.campaign / "generation").mkdir()
        with patch.object(operator.subprocess, "check_output", side_effect=["a" * 40 + " refs/heads/x", ""]), patch.object(operator.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr="gh: Not Found (HTTP 404)")):
            with self.assertRaisesRegex(ValueError, "already frozen"):
                operator.validate_live_request()

    def test_foreign_resume_run_rejected_before_artifact_download(self):
        path = self.campaign / "request.json"
        req = json.loads(path.read_text())
        req["resume_run_id"] = 123
        path.write_text(json.dumps(req))
        run = {"path": operator.WORKFLOW, "head_branch": operator.BRANCH, "status": "completed", "repository": {"full_name": "foreign/repo", "id": 1}}
        with patch.object(operator, "get_bytes", return_value=json.dumps(run).encode()) as get:
            with self.assertRaisesRegex(ValueError, "repository mismatch"):
                operator.recover_lane("responses", "BASE", self.root / "resume")
            self.assertEqual(get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
