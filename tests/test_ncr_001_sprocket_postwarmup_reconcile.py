"""Adversarial offline checks for original Chat/Instruct diagnostic reconciliation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_sprocket_postwarmup_reconcile as recon
import ncr_001_sprocket_learning_artifact_readback as original
import ncr_001_sprocket_learning_git_custody as custody


class SprocketPostWarmupReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, members = custody.load_archive()
        cls.original_receipts = {
            lane: json.loads(members[lane]["receipt.json"])
            for lane in ("chat", "instruct")
        }

    def test_original_source_bound_two_lane_diagnostic_only(self):
        result = recon.reconcile()
        self.assertEqual(result["status"], recon.FINAL)
        self.assertEqual(result["physical_source_run"], 40710171622)
        self.assertEqual(set(result["lanes"]), {
            "sprocket_chat", "sprocket_instruct"})
        self.assertNotEqual(
            result["lanes"]["sprocket_chat"]["original_checkpoint_sha256"],
            result["lanes"]["sprocket_instruct"]["original_checkpoint_sha256"])
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["full_72_conformance_pass"])
        self.assertFalse(result["comparative_nca_mlp_training"])
        self.assertFalse(result["new_physical_model_execution"])
        self.assertEqual(result["new_optimizer_steps"], 0)

    def test_narrow_protocol_labels_never_promote_lane(self):
        for lane in recon.reconcile()["lanes"].values():
            self.assertEqual(lane["base_weight_unchanged_after_step"],
                             recon.DIAGNOSTIC_OBSERVED)
            self.assertEqual(lane["nonzero_adapter_gradient_after_warmup"],
                             recon.DIAGNOSTIC_OBSERVED)
            self.assertEqual(lane["finite_state_and_loss"], recon.PARTIAL_LOSS)
            self.assertFalse(lane["full_72_per_model_gate"])
            self.assertFalse(lane["tool_memory_semantics_qualified"])

    def test_original_historical_adjudication_cohorts_not_rewritten(self):
        report = recon.reconcile()
        self.assertEqual(report["historical_formal_pretraining_adjudications"]
                         ["sprocket_chat"]["head"],
                         "b2461a66241bc2fd0fbc389d7b3b2d10f729d2ab")
        self.assertEqual(report["historical_formal_pretraining_adjudications"]
                         ["sprocket_instruct"]["head"],
                         "90f3006dd59b2a731db19064472d9f2215ac62d9")

    def mutate_receipt(self, lane, change):
        receipt = copy.deepcopy(self.original_receipts[lane])
        change(receipt)
        with self.assertRaises((AssertionError, KeyError, ValueError)):
            original.receipt_gate(receipt, lane, source_tree=False)

    def test_cross_variant_weight_substitution_refused(self):
        self.mutate_receipt("chat", lambda r: r["source_identity"].update(
            weight_path="model.safetensors"))
        self.mutate_receipt("instruct", lambda r: r["source_identity"].update(
            weight_path="chat/model.safetensors"))

    def test_missing_postwarmup_upstream_gradient_refused(self):
        self.mutate_receipt("chat", lambda r: r["post_warmup"]
                            ["gradient_abs_sums"].update({"observe.weight": 0}))
        self.mutate_receipt("instruct", lambda r: r["post_warmup"]
                            ["gradient_abs_sums"].update({"rule.weight": 0}))

    def test_forged_step_or_frozen_backbone_refused(self):
        self.mutate_receipt("chat", lambda r: r.update(
            diagnostic_optimizer_steps=2))
        self.mutate_receipt("instruct", lambda r: r.update(
            backbone_after_sha256="0"*64))

    def test_false_scientific_or_train_permission_refused(self):
        self.mutate_receipt("chat", lambda r: r.update(
            full_72_conformance_pass=True))
        self.mutate_receipt("instruct", lambda r: r.update(
            train_authorized=True))

    def test_future_lane_eos_and_heldout_substitution_refused(self):
        self.mutate_receipt("instruct", lambda r: r["synthetic_sample"].update(
            assistant_eos_id=258))
        self.mutate_receipt("chat", lambda r: r["synthetic_sample"].update(
            heldout=True))

    def test_raw_committed_manifest_and_archive_whitespace_refused(self):
        self._mutate_file("manifest.json", lambda b: b + b"\n")
        self._mutate_file("original-receipt-bundle.part001.b64",
                          lambda b: b + b"\n")

    def test_raw_original_positive_and_failed_job_log_mutations_refused(self):
        self._mutate_file(
            "actions-decoded/chat-original-job-122893504080.log",
            lambda b: b + b"\nFORGED PASS")
        self._mutate_file(
            "actions-decoded/initial-readback-formatter-hash-failure-job-122910071921.log",
            lambda b: b + b"\nFORGED NEGATIVE")


    def test_archived_job_ids_are_bound_to_original_run_log_contents(self):
        manifest = json.loads((recon.EVIDENCE / "manifest.json").read_text())
        self.assertEqual(recon.check_archived_job_bindings(manifest), {
            "source": 122892989282,
            "chat": 122893504080,
            "instruct": 122893504180,
        })
        report = recon.reconcile()
        self.assertEqual(report["lanes"]["sprocket_chat"]["physical_job_id"],
                         122893504080)
        self.assertEqual(report["lanes"]["sprocket_instruct"]["physical_job_id"],
                         122893504180)
        for lane in report["lanes"].values():
            self.assertEqual(lane["job_id_provenance"],
                             "PINNED_GITHUB_ACTIONS_RUN_LEDGER_PLUS_ORIGINAL_DECODED_LOG")

    def test_swapped_or_forged_original_job_id_ledger_refused(self):
        manifest = json.loads((recon.EVIDENCE / "manifest.json").read_text())
        changed = copy.deepcopy(manifest)
        changed["original_decoded_actions_jobs"]["chat"] = 122893504180
        with self.assertRaisesRegex(ValueError, "job-ID ledger changed"):
            recon.check_archived_job_bindings(changed)

    def test_archived_log_variant_or_job_source_substitution_refused(self):
        manifest = json.loads((recon.EVIDENCE / "manifest.json").read_text())
        logs = {}
        for variant, job_id in recon.ORIGINAL_JOBS.items():
            filename = "actions-decoded/" + (
                "source-original-job-" if variant == "source"
                else variant + "-original-job-"
            ) + str(job_id) + ".log"
            logs[variant] = (recon.EVIDENCE / filename).read_text(
                encoding="utf-8-sig")
        swapped = dict(logs)
        swapped["chat"], swapped["instruct"] = logs["instruct"], logs["chat"]
        with self.assertRaisesRegex(ValueError, "job log/source/variant"):
            recon.check_archived_job_bindings(manifest, log_texts=swapped)
        missing_source = dict(logs)
        missing_source["chat"] = missing_source["chat"].replace(
            recon.SOURCE_HEAD, "0" * 40)
        with self.assertRaisesRegex(ValueError, "job log/source/variant"):
            recon.check_archived_job_bindings(manifest, log_texts=missing_source)

    def test_original_job_uploaded_zip_digest_ledger_forgery_refused(self):
        manifest = json.loads((recon.EVIDENCE / "manifest.json").read_text())
        modified = copy.deepcopy(manifest)
        modified["original_actions_artifacts"]["instruct"]["zip_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "artifact digest ledger changed"):
            recon.check_archived_job_bindings(modified)

    def _mutate_file(self, name, change):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            for source in recon.ORIGINAL_BLOBS:
                target = dest / source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(recon.EVIDENCE / source, target)
            target = dest / name
            target.write_bytes(change(target.read_bytes()))
            with self.assertRaisesRegex(ValueError,
                                        "changed original committed Git blob"):
                recon.check_committed_bytes(dest)


if __name__ == "__main__":
    unittest.main()
