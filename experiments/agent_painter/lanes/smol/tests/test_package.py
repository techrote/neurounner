"""Publication corruption checks only; these fixtures are never training proof."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

LANE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LANE))
SPEC = importlib.util.spec_from_file_location("smol_package", LANE / "package.py")
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class PublicationCustody(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.tree, self.run, self.attempt = "a" * 40, "b" * 40, 12345, 2
        self.request = {"purpose": "unit-test protocol metadata only"}
        selected = package.common.profile("smol", require_supported=True)
        for arm in package.ARMS:
            folder = self.root / arm / "environment"
            folder.mkdir(parents=True)
            receipt = {
                "schema_version": 1, "lane": "smol", "arm": arm,
                "repository": package.REPOSITORY, "source_sha": self.source, "source_tree": self.tree,
                "run_id": self.run, "run_attempt": self.attempt, "job_id": "models",
                "runner_name": "test-runner-" + arm, "hostname": "test-vm-" + arm,
                "runner_environment": "github-hosted", "os": "Linux", "arch": "X64",
                "corpus": {"source_sha": package.contracts.FOUNDATION_SOURCE,
                           "run_id": package.CORPUS_RUN, "artifact": package.CORPUS_ARTIFACT},
                "model": {"model_id": package.contracts.MODEL_ID, "revision": package.contracts.MODEL_REVISION},
                "request": self.request,
            }
            self.write(folder / "run-receipt.json", receipt)
            self.write(folder / "model-download.json", {
                "model_id": selected["model_id"], "revision": selected["revision"], "trust_remote_code": False,
                "files": {name: {"sha256": digest, "bytes": 1} for name, digest in selected["required_hashes"].items()},
            })
            for name in ("cpu.txt", "pip-freeze.txt"):
                (folder / name).write_text("unit-test metadata; no model execution\n")
            (folder / "requirements.sha256").write_text(package.sha(package.PAINTER / "training/requirements.lock") + "  requirements.lock\n")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def verify(self):
        return package.verify_environments(self.root, self.source, self.tree, self.run, self.attempt, self.request)

    def mutate_receipt(self, arm, key, value):
        path = self.root / arm / "environment/run-receipt.json"
        receipt = json.loads(path.read_text())
        receipt[key] = value
        self.write(path, receipt)

    def test_matching_custody_accepts_distinct_vm_metadata(self):
        receipts, _ = self.verify()
        self.assertNotEqual(receipts["baseline"]["hostname"], receipts["candidate"]["hostname"])

    def job_record(self):
        jobs = []
        for index, arm in enumerate((*package.ARMS, "native")):
            jobs.append({
                "id": 100 + index, "run_id": self.run, "run_attempt": self.attempt, "head_sha": self.source,
                "name": arm + " on independent CPU VM" if arm in package.ARMS else package.NATIVE_JOB,
                "runner_id": 1000 + index, "runner_name": "test-runner-" + arm,
                "runner_group_name": "GitHub Actions", "labels": ["ubuntu-24.04"],
                "status": "completed", "conclusion": "success" if arm in package.ARMS else "failure",
                "steps": [{"name": package.NATIVE_STEP, "status": "completed", "conclusion": "success"},
                          {"name": package.PACKAGING_STEP, "status": "completed", "conclusion": "failure"}] if arm == "native" else [],
            })
        return {"schema_version": 1, "repository": package.REPOSITORY, "source_sha": self.source,
                "run_id": self.run, "run_attempt": self.attempt,
                "api_path": f"/repos/{package.REPOSITORY}/actions/runs/{self.run}/jobs",
                "response": {"total_count": 3, "jobs": jobs}}

    def test_repeated_guest_hostname_with_distinct_api_registrations_accepts(self):
        self.mutate_receipt("candidate", "hostname", "test-vm-baseline")
        receipts, _ = self.verify()
        assignments = package.verify_job_assignments(self.job_record(), receipts, self.source, self.run, self.attempt, recovery=True)
        self.assertNotEqual(assignments["baseline"]["runner_id"], assignments["candidate"]["runner_id"])

    def test_same_runner_registration_refused(self):
        self.mutate_receipt("candidate", "runner_name", "test-runner-baseline")
        with self.assertRaisesRegex(ValueError, "separate runner registrations"):
            self.verify()

    def test_same_numeric_runner_even_with_different_names_refused(self):
        receipts, _ = self.verify()
        jobs = self.job_record()
        jobs["response"]["jobs"][1]["runner_id"] = jobs["response"]["jobs"][0]["runner_id"]
        with self.assertRaisesRegex(ValueError, "share a job or runner"):
            package.verify_job_assignments(jobs, receipts, self.source, self.run, self.attempt, recovery=True)

    def test_authoritative_job_source_run_attempt_mismatch_refused(self):
        receipts, _ = self.verify()
        for key, value in (("head_sha", "d" * 40), ("run_id", 9999), ("run_attempt", 1)):
            jobs = self.job_record()
            jobs["response"]["jobs"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "wrong producer job"):
                package.verify_job_assignments(jobs, receipts, self.source, self.run, self.attempt, recovery=True)

    def test_native_failure_cannot_be_hidden_as_packaging_repair(self):
        receipts, _ = self.verify()
        jobs = self.job_record()
        jobs["response"]["jobs"][2]["steps"][0]["conclusion"] = "failure"
        with self.assertRaisesRegex(ValueError, "native execution/build step failed"):
            package.verify_job_assignments(jobs, receipts, self.source, self.run, self.attempt, recovery=True)

    def test_wrong_source_or_run_attempt_refused(self):
        for key, value in (("source_sha", "c" * 40), ("run_attempt", 1), ("run_id", 999)):
            with self.subTest(key=key):
                path = self.root / "candidate/environment/run-receipt.json"
                original = json.loads(path.read_text())
                self.mutate_receipt("candidate", key, value)
                with self.assertRaisesRegex(ValueError, "custody mismatch"):
                    self.verify()
                self.write(path, original)

    def test_changed_base_model_bytes_refused(self):
        path = self.root / "candidate/environment/model-download.json"
        download = json.loads(path.read_text())
        name = next(iter(download["files"]))
        download["files"][name]["sha256"] = "0" * 64
        self.write(path, download)
        with self.assertRaisesRegex(ValueError, "reviewed pin"):
            self.verify()


class ArchiveCustody(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_original_bytes_survive_zip_readback_and_output_is_immutable(self):
        original = b"original binary evidence\x00\xff\n"
        (self.evidence / "readback.bin").write_bytes(original)
        output = self.root / "evidence.zip"
        result = package.archive(output, package.inventory_files(self.evidence))
        self.assertEqual(package.sha(output), result["sha256"])
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read("readback.bin"), original)
            self.assertIsNone(archive.testzip())
        with self.assertRaisesRegex(ValueError, "overwrite"):
            package.archive(output, package.inventory_files(self.evidence))

    def test_final_target_refused(self):
        (self.evidence / "final.jsonl").write_text("unit test sentinel\n")
        with self.assertRaisesRegex(ValueError, "FINAL targets"):
            package.inventory_files(self.evidence)

    def test_symlink_refused(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside fixture")
        (self.evidence / "link.txt").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            package.inventory_files(self.evidence)

    def test_unsafe_zip_member_refused(self):
        path = self.evidence / "receipt.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "unsafe archive member"):
            package.archive(self.root / "invalid.zip", [("../escaped.json", path)])

    def test_checkpoint_inventory_tampering_refused_before_tensor_load(self):
        checkpoint = self.evidence / "checkpoint-000001"
        (checkpoint / "adapter").mkdir(parents=True)
        names = ("adapter/adapter_model.safetensors", "adapter/adapter_config.json", "training_state.pt", "receipt.json")
        for name in names:
            (checkpoint / name).write_bytes(b"unit-test inventory fixture only")
        inventory = {name: {"sha256": package.sha(checkpoint / name), "bytes": (checkpoint / name).stat().st_size}
                     for name in names}
        (checkpoint / "inventory.json").write_text(json.dumps(inventory))
        (checkpoint / "training_state.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "custody hash mismatch"):
            package.common.verify_checkpoint(checkpoint)

    def artifact_record(self):
        source, run_id, attempt = "a" * 40, 12345, 2
        downloads = self.evidence / "producer-downloads"
        downloads.mkdir()
        entries = []
        for index, component in enumerate(package.ARMS):
            folder = self.evidence / component
            folder.mkdir()
            original = folder / "receipt.json"
            original.write_text('{"fixture":"unit-test only"}')
            archive = downloads / (component + ".zip")
            with zipfile.ZipFile(archive, "w") as stream:
                stream.write(original, "receipt.json")
            digest, size = package.sha(archive), archive.stat().st_size
            entries.append({"component": component, "archive_path": "producer-downloads/" + archive.name,
                            "archive_sha256": digest, "archive_bytes": size, "member_prefix": "",
                            "artifact": {"id": 2000 + index,
                                         "name": f"smol-{component}-{source}-{run_id}-{attempt}",
                                         "size_in_bytes": size, "digest": "sha256:" + digest,
                                         "expired": False, "workflow_run": {"id": run_id, "head_sha": source}}})
        return {"schema_version": 1, "repository": package.REPOSITORY, "source_sha": source,
                "run_id": run_id, "run_attempt": attempt, "artifacts": entries}

    def test_producer_artifacts_bind_original_zip_and_extracted_bytes(self):
        record = self.artifact_record()
        result = package.verify_artifact_custody(self.evidence, record, "a" * 40, 12345, 2, recovery=False)
        self.assertEqual(len(result), 2)
        (self.evidence / "candidate/receipt.json").write_text("changed after download")
        with self.assertRaisesRegex(ValueError, "extracted producer bytes differ"):
            package.verify_artifact_custody(self.evidence, record, "a" * 40, 12345, 2, recovery=False)

    def test_recovery_requires_original_native_and_build_artifacts(self):
        record = self.artifact_record()
        with self.assertRaisesRegex(ValueError, "component inventory incomplete"):
            package.verify_artifact_custody(self.evidence, record, "a" * 40, 12345, 2, recovery=True)

    def test_artifact_from_different_producer_cannot_be_relabelled(self):
        record = self.artifact_record()
        record["artifacts"][0]["artifact"]["workflow_run"]["head_sha"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "artifact API identity mismatch"):
            package.verify_artifact_custody(self.evidence, record, "a" * 40, 12345, 2, recovery=False)

    def test_missing_or_failed_retained_regression_refused(self):
        labels = ("art_batch", "stored_material", "deferred_manual", "undo_1", "undo_4",
                  "combined_author", "combined_reopen")
        results = [{"label": label, "status": "passed", "exit_code": 0,
                    "completion": {"checks": 1, "failures": 0}} for label in labels]
        for label in labels:
            (self.evidence / (label + ".log")).write_text("unit-test protocol metadata\n")
        path = self.evidence / "results.json"
        path.write_text(json.dumps(results))
        self.assertEqual(package.verify_regressions(self.evidence)["checks"], 7)
        path.write_text(json.dumps(results[:-1]))
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            package.verify_regressions(self.evidence)
        results[0]["completion"]["failures"] = 1
        path.write_text(json.dumps(results))
        with self.assertRaisesRegex(ValueError, "regression failed"):
            package.verify_regressions(self.evidence)


class ProducerPublisherSeparation(unittest.TestCase):
    def test_publication_fix_can_change_without_relabelling_execution_code(self):
        old = {"experiments/agent_painter/lanes/smol/package.py": "old",
               "experiments/agent_painter/lanes/smol/evaluate.py": "same"}
        new = {**old, "experiments/agent_painter/lanes/smol/package.py": "repair",
               "experiments/agent_painter/lanes/smol/publication.py": "new"}
        self.assertEqual(len(package.validate_publication_delta(old, new)), 2)
        new["experiments/agent_painter/lanes/smol/evaluate.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "non-publication execution source"):
            package.validate_publication_delta(old, new)

    def test_workflow_repair_is_limited_to_packaging_step(self):
        name = ".github/workflows/agent-painter-smol-native.yml"
        before = "header\n      - name: " + package.NATIVE_STEP + "\n        run: native\n      - name: " + package.PACKAGING_STEP + "\n        run: old-package\n      - name: Publish\n        run: release\n"
        after = before.replace("run: old-package", "run: corrected-package")
        self.assertEqual(package.validate_publication_delta({name: "old"}, {name: "new"}, before, after), [name])
        after = after.replace("run: native", "run: changed-native")
        with self.assertRaisesRegex(ValueError, "outside the packaging step"):
            package.validate_publication_delta({name: "old"}, {name: "new"}, before, after)


if __name__ == "__main__":
    unittest.main()
