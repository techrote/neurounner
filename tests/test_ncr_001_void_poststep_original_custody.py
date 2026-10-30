"""No-model adversarial tests for immutable original #210 void.1 poststep evidence.

Only original committed ZIP/log bytes and source metadata are read.
No HF, weights, optimizer, tokenizer or heldout fixture is imported.
"""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_poststep_original_custody as custody  # noqa: E402


class OriginalPoststepCustodyTests(unittest.TestCase):
    @contextmanager
    def copied_originals(self):
        with tempfile.TemporaryDirectory(prefix="ncr001-original-") as temporary:
            evidence = Path(temporary) / "evidence"
            shutil.copytree(custody.EVIDENCE, evidence)
            yield evidence

    def test_original_permanent_bytes_and_report_boundaries(self):
        report, files = custody.read_and_check(verify_git=False)
        self.assertEqual(set(files), set(custody.FILES))
        self.assertEqual(report["original_run"], 41058648266)
        self.assertEqual(report["original_head"], custody.EXPECTED_HEAD)
        self.assertEqual(report["source_and_host_jobs"], [124072474714, 124072552187])
        self.assertEqual(report["original_zips_and_logs"], 4)
        self.assertEqual(report["original_zip_members"], 8)
        self.assertEqual(report["synthetic_nca_steps"], 1)
        self.assertEqual(report["cache_checked_tail_tokens"], 4)
        self.assertEqual(report["context_units"], 31)
        self.assertLessEqual(report["poststep_cache_max_abs"], 5e-4)
        self.assertTrue(report["separate_process_cold_adapter"])
        for key in ("train_authorized", "full_72_conformance_pass",
                    "per_model_gate", "actual_compute_budget_qualified",
                    "original_training_impact_measured",
                    "scientific_review_complete", "fresh_physical_execution"):
            self.assertIs(report[key], False, key)

    def test_literal_committed_git_blobs_not_only_manifest_hashes(self):
        report = custody.audit(live=False)
        self.assertFalse(report["original_ledger_freshly_rechecked"])
        self.assertEqual(report["status"],
                         "PASS_ORIGINAL_POSTSTEP_COLD_BYTES__SCIENTIFIC_72_BLOCKED")

    def test_all_four_original_zip_and_decoded_log_mutations_refused(self):
        for name in custody.FILES:
            with self.subTest(name=name), self.copied_originals() as evidence:
                path = evidence / name
                path.write_bytes(path.read_bytes() + b"UNAUTHORIZED")
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(evidence)

    def test_missing_original_streams_fail_closed(self):
        for name in custody.FILES:
            with self.subTest(name=name), self.copied_originals() as evidence:
                (evidence / name).unlink()
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(evidence)

    def test_symlink_substitutions_refused_even_when_original_bytes_match(self):
        for name in ("host-original.zip", "host-original-job.log"):
            with self.subTest(name=name), self.copied_originals() as evidence:
                original = evidence / name
                moved = evidence / (name + ".renamed")
                original.rename(moved)
                original.symlink_to(moved.name)
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(evidence)

    def test_original_manifest_forgery_refused_even_if_valid_json(self):
        changes = (
            ("source_head", "f" * 40),
            ("run", 41058648267),
            ("physical_host_run_job_success", False),
            ("retention_of_original_zips_and_logs", "ephemeral_only"),
        )
        for key, value in changes:
            with self.subTest(key=key), self.copied_originals() as evidence:
                path = evidence / "manifest.json"
                data = json.loads(path.read_bytes())
                data[key] = value
                path.write_text(json.dumps(data, sort_keys=True) + "\n")
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(evidence)

    def test_false_training_or_scientific_admission_in_manifest_refused(self):
        for key in custody.BOUNDARY:
            with self.subTest(key=key), self.copied_originals() as evidence:
                path = evidence / "manifest.json"
                data = json.loads(path.read_bytes())
                data["scientific_boundary"][key] = True
                path.write_text(json.dumps(data, sort_keys=True) + "\n")
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(evidence)

    def test_rehashed_but_replaced_original_log_is_refused(self):
        with self.copied_originals() as evidence:
            name = "host-original-job.log"
            file = evidence / name
            fake = file.read_bytes().replace(custody.EXPECTED_HEAD.encode(),
                                             b"f" * 40)
            self.assertNotEqual(fake, file.read_bytes())
            file.write_bytes(fake)
            manifest_path = evidence / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["files"][name] = {
                "size": len(fake), "sha256": custody.sha(fake),
                "git_blob": custody.blob(fake),
            }
            manifest["original"]["host"]["decoded_log_sha256"] = custody.sha(fake)
            manifest["original"]["host"]["decoded_log_bytes"] = len(fake)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                custody.read_and_check(evidence)

    def test_exact_zip_inventory_no_persisted_adapter(self):
        _, files = custody.read_and_check(verify_git=False)
        with zipfile.ZipFile(io.BytesIO(files["host-original.zip"])) as archive:
            members = set(archive.namelist())
            self.assertEqual(len(members), 7)
            self.assertIn("ncr001-void-poststep-cold/producer.json", members)
            self.assertIn("ncr001-void-poststep-cold/consumer.json", members)
            self.assertFalse(any(n.endswith((".pt", ".safetensors", "metadata.json"))
                                 for n in members))
            producer = json.loads(archive.read(
                "ncr001-void-poststep-cold/producer.json"))
            consumer = json.loads(archive.read(
                "ncr001-void-poststep-cold/consumer.json"))
            self.assertEqual(producer["diagnostic_optimizer_steps"], 1)
            self.assertEqual(consumer["diagnostic_optimizer_steps"], 0)
            self.assertEqual(producer["sample"]["real_eos_id"], 258)
            self.assertEqual(producer["sample"]["supervised_after_shift"], 21)
            self.assertNotEqual(producer["producer_pid"], consumer["cold_process_pid"])


if __name__ == "__main__":
    unittest.main()
