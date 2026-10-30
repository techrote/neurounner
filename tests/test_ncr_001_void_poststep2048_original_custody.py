"""No-model adversarial tests for permanent original poststep2048 custody."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_poststep2048_original_custody as custody
import ncr_001_void_poststep2048_bootstrap_custody as acquisition


class Poststep2048CustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.originals = custody.read_and_check(verify_git=True)
        with zipfile.ZipFile(io.BytesIO(cls.originals["host-original.zip"])) as z:
            cls.producer = json.loads(z.read(
                "ncr001-void-poststep2048/producer.json"))
            cls.consumer = json.loads(z.read(
                "ncr001-void-poststep2048/consumer.json"))

    @contextmanager
    def copied_evidence(self):
        with tempfile.TemporaryDirectory(prefix="ncr-poststep2048-") as td:
            evidence = Path(td) / "evidence"
            shutil.copytree(custody.EVIDENCE, evidence)
            with mock.patch.object(custody, "EVIDENCE", evidence), \
                 mock.patch.object(custody, "MANIFEST", evidence / "manifest.json"):
                yield evidence

    def test_original_four_streams_and_physical_result_are_exact(self):
        r = self.report
        self.assertEqual(
            r["status"],
            "PASS_ORIGINAL_POSTSTEP2048_COMPLETE_BYTES__SCIENTIFIC_72_UNADMITTED")
        self.assertEqual(r["original_run"], 41133450850)
        self.assertEqual(r["original_head"],
                         "520c4ae5b3152aff8d94760dab2e3103746f7334")
        self.assertEqual(r["original_zips_and_logs"], 4)
        self.assertEqual(r["original_zip_members"], 9)
        self.assertTrue(r["permanent_git_custody"])
        p = r["physical_result"]
        self.assertEqual(p["nca_only_optimizer_steps"], 1)
        self.assertEqual(p["public_training_units"], 31)
        self.assertEqual(p["supervised_after_shift"], 21)
        self.assertEqual(p["true_eos_id"], 258)
        self.assertEqual(p["poststep_2048_incremental_positions"], 2048)
        self.assertAlmostEqual(p["poststep_2048_cache_max_abs"],
                               7.522106170654297e-05)
        self.assertEqual(p["masked_batch"]["left"],
                         {"max_batch_vs_single_abs": 0.0,
                          "max_row_swap_abs": 0.0})
        self.assertEqual(p["masked_batch"]["right"],
                         {"max_batch_vs_single_abs": 0.0,
                          "max_row_swap_abs": 0.0})
        self.assertTrue(p["future_suffix_earlier_exact"])
        self.assertEqual(p["future_suffix_units"], 64)
        self.assertEqual(p["generate_prefix_units"], 2044)
        self.assertEqual(p["generate_new_units"], 4)
        self.assertEqual(p["generated_ids"], [105,110,115,112])
        self.assertFalse(p["generation_actual_eos_id_emitted"])
        self.assertAlmostEqual(p["generation_cache_score_max_abs"],
                               5.7220458984375e-06)

    def test_all_original_files_are_literal_committed_regular_blobs(self):
        custody.git_entry(custody.MANIFEST, custody.MANIFEST_BLOB)
        for name, (_, _, blob) in custody.FILES.items():
            custody.git_entry(custody.EVIDENCE / name, blob)

    def test_all_four_whole_original_stream_mutations_refused(self):
        for name in custody.FILES:
            with self.subTest(name=name), self.copied_evidence() as evidence:
                path = evidence / name
                path.write_bytes(path.read_bytes() + b"UNAUTHORIZED")
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(verify_git=False)

    def test_missing_original_streams_refused(self):
        for name in custody.FILES:
            with self.subTest(name=name), self.copied_evidence() as evidence:
                (evidence / name).unlink()
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(verify_git=False)

    def test_symlink_substitutions_refused(self):
        for name in ("host-original.zip", "host-original-job.log"):
            with self.subTest(name=name), self.copied_evidence() as evidence:
                path = evidence / name
                saved = evidence / (name + ".saved")
                path.rename(saved)
                path.symlink_to(saved.name)
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(verify_git=False)

    def test_manifest_forgery_refused_before_semantic_promotion(self):
        for field, value in (
            ("run", 1),
            ("source_head", "f"*40),
            ("retention_of_original_zips_and_logs", "ephemeral"),
            ("physical_host_job_success", False),
        ):
            with self.subTest(field=field), self.copied_evidence() as evidence:
                path = evidence / "manifest.json"
                data = json.loads(path.read_text())
                data[field] = value
                path.write_text(json.dumps(data, sort_keys=True) + "\n")
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(verify_git=False)

    def test_false_science_training_compute_or_eos_promotion_refused(self):
        for key in custody.BOUNDARY:
            with self.subTest(key=key), self.copied_evidence() as evidence:
                path = evidence / "manifest.json"
                data = json.loads(path.read_text())
                data["scientific_boundary"][key] = True
                path.write_text(json.dumps(data, sort_keys=True) + "\n")
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    custody.read_and_check(verify_git=False)

    def test_physical_semantics_reject_short_or_partial_cache_relabel(self):
        for key, value in (
            ("context_units", 31),
            ("incremental_positions", 4),
            ("incremental_positions", True),
            ("max_abs", 0.1),
            ("all_positions_compared", False),
        ):
            p, c = copy.deepcopy(self.producer), copy.deepcopy(self.consumer)
            c["poststep_2048_all_position_cache"][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    RuntimeError, "REFUSED"):
                acquisition.require_semantics(p, c)

    def test_physical_semantics_reject_mask_suffix_generation_forgery(self):
        mutations = (
            lambda c: c["poststep_masked_batch"].pop("left"),
            lambda c: c["poststep_masked_batch"]["right"].__setitem__(
                "batch_size", 1),
            lambda c: c["poststep_masked_batch"]["left"].__setitem__(
                "explicit_attention_mask", False),
            lambda c: c["poststep_masked_batch"]["right"].__setitem__(
                "max_row_swap_abs", 0.1),
            lambda c: c["poststep_2048_suffix"].__setitem__(
                "earlier_logits_exact", False),
            lambda c: c["poststep_2048_suffix"].__setitem__(
                "mutated_suffix_units", 1),
            lambda c: c["poststep_native_generation"].__setitem__(
                "prefix_units", 512),
            lambda c: c["poststep_native_generation"].__setitem__(
                "cache_on_off_token_ids_equal", False),
            lambda c: c["poststep_native_generation"].__setitem__(
                "max_score_abs", 0.1),
        )
        for mutation in mutations:
            p, c = copy.deepcopy(self.producer), copy.deepcopy(self.consumer)
            mutation(c)
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                acquisition.require_semantics(p, c)

    def test_physical_semantics_reject_false_optimizer_or_variant(self):
        mutations = (
            lambda p,c: p.__setitem__("diagnostic_optimizer_steps", True),
            lambda p,c: p.__setitem__("diagnostic_optimizer_steps", 2),
            lambda p,c: p.__setitem__("model_repo", "kandivault/sprocket-500m"),
            lambda p,c: p["public_learning_sample"].__setitem__("real_eos_id", 62),
            lambda p,c: c.__setitem__("frozen_backbone_sha256", "f"*64),
            lambda p,c: c.__setitem__("poststep_adapter_reload_exact", False),
            lambda p,c: c.__setitem__("consumer_pid", p["producer_pid"]),
        )
        for mutation in mutations:
            p, c = copy.deepcopy(self.producer), copy.deepcopy(self.consumer)
            mutation(p, c)
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                acquisition.require_semantics(p, c)

    def test_original_host_zip_contains_no_temporary_adapter_checkpoint(self):
        with zipfile.ZipFile(io.BytesIO(
                self.originals["host-original.zip"])) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), 8)
            self.assertFalse(any(name.endswith(
                (".pt", ".safetensors", "metadata.json")) for name in names))
            self.assertIn("ncr001-void-poststep2048/producer.json", names)
            self.assertIn("ncr001-void-poststep2048/consumer.json", names)

    def test_audit_is_source_only_and_does_not_promote_science(self):
        r = custody.audit(live=False)
        self.assertFalse(r["original_ledger_freshly_rechecked"])
        for key in (
            "fresh_physical_execution", "train_authorized", "per_model_gate",
            "full_72_conformance_pass", "independent_scientific_review",
            "comparative_training", "heldouts_opened",
            "actual_compute_budget_qualified", "generated_eos_skill_qualified",
            "historical_train002b_eos_effect_measured",
        ):
            self.assertIs(r[key], False, key)


if __name__ == "__main__":
    unittest.main()
