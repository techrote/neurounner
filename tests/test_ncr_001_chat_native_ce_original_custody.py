"""Fail-closed original Sprocket CHAT physical CE evidence custody tests.

All tests run offline on the exact original Actions ZIP and decoded job log.
No original frozen model, tokenizer, optimizer or held-out data is executed.
"""
from __future__ import annotations

import copy
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
import ncr_001_chat_native_ce_original_custody as audit


class OriginalChatCEArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = audit.base64.b64decode(
            (audit.EVIDENCE / audit.BUNDLE).read_bytes().strip(), validate=True)
        with zipfile.ZipFile(io.BytesIO(cls.original)) as z:
            cls.receipt = json.loads(z.read("receipt.json"))

    def tamper_receipt(self, mutation):
        value = copy.deepcopy(self.receipt)
        mutation(value)
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            audit.validate_receipt(value)

    def mutate_dir(self, mutation):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "source"
            shutil.copytree(audit.EVIDENCE, dest)
            mutation(dest)
            with self.assertRaises((ValueError, FileNotFoundError)):
                audit.audit(dest)

    def test_original_six_members_and_entire_log_are_source_bound(self):
        out = audit.audit()
        self.assertEqual(out["original"]["original_raw_members"], 6)
        self.assertEqual(out["original"]["assistant_supervised_targets"], 16)
        self.assertEqual(out["original"]["original_zip_sha256"], audit.ORIGINAL_ZIP_SHA256)
        self.assertFalse(out["train_authorized"])
        self.assertFalse(out["full_72_conformance_pass"])

    def test_wrong_chat_vs_root_instruct_checkpoint_refused(self):
        self.tamper_receipt(lambda d: d["sources"].__setitem__(
            "chat_checkpoint_sha256", audit.INSTRUCT_SHA))
        self.tamper_receipt(lambda d: d.__setitem__("lane", "sprocket_instruct"))

    def test_true_original_train_gate_cannot_open(self):
        self.tamper_receipt(lambda d: d.__setitem__("train_authorized", True))
        self.tamper_receipt(lambda d: d.__setitem__("optimizer_steps", 1))
        self.tamper_receipt(lambda d: d.__setitem__("full_72_conformance_pass", True))

    def test_false_scientific_or_tool_competence_refused(self):
        self.tamper_receipt(lambda d: d.__setitem__("independent_scientific_review", True))
        self.tamper_receipt(lambda d: d.__setitem__("tool_memory_behaviour_qualified", True))
        self.tamper_receipt(lambda d: d.__setitem__("comparative_nca_mlp_training", True))

    def test_corrupted_native_or_manual_ce_refused(self):
        self.tamper_receipt(lambda d: d["ce"].__setitem__("native", 0.0))
        self.tamper_receipt(lambda d: d["ce"].__setitem__("abs_difference", 0.1))
        self.tamper_receipt(lambda d: d["ce"].__setitem__("targets", 17))

    def test_original_attention_or_shifted_supervision_changed_refused(self):
        self.tamper_receipt(lambda d: d["synthetic_transport"].__setitem__(
            "attention_mask_sha256", "0"*64))
        self.tamper_receipt(lambda d: d["synthetic_transport"].__setitem__(
            "supervised_targets", 0))
        self.tamper_receipt(lambda d: d["synthetic_transport"].__setitem__(
            "attended_true_end_id5", False))

    def test_frozen_backbone_or_gated_mlp_parity_corruption_refused(self):
        self.tamper_receipt(lambda d: d.__setitem__("backbone_after_sha256", "0"*64))
        self.tamper_receipt(lambda d: d["controls"]["mlp"].__setitem__(
            "disabled_logits_exact", False))

    def test_original_runner_or_source_head_mismatch_refused(self):
        self.tamper_receipt(lambda d: d["runner"].__setitem__("run", "40798938606"))
        self.tamper_receipt(lambda d: d.__setitem__("source_head", "0"*40))

    def test_original_complete_decoded_job_log_tamper_refused(self):
        def mutation(evidence):
            p = evidence / audit.JOB_LOG
            p.write_bytes(p.read_bytes() + b"\nFORGED")
        self.mutate_dir(mutation)

    def test_original_manifest_altered_even_with_rehash_refused(self):
        def mutation(evidence):
            p = evidence / audit.MANIFEST
            d = json.loads(p.read_text())
            d["full_72_conformance_pass"] = True
            p.write_text(json.dumps(d))
        self.mutate_dir(mutation)

    def test_original_base64_archive_byte_flip_refused(self):
        def mutation(evidence):
            p = evidence / audit.BUNDLE
            raw = bytearray(p.read_bytes())
            raw[13] = ord("B") if raw[13] != ord("B") else ord("C")
            p.write_bytes(raw)
        self.mutate_dir(mutation)

    def test_original_actions_member_omission_refused(self):
        with zipfile.ZipFile(io.BytesIO(self.original)) as z:
            originals = {n: z.read(n) for n in z.namelist()}
        with io.BytesIO() as dest:
            with zipfile.ZipFile(dest, "w") as out:
                for name, contents in originals.items():
                    if name != "physical-host.log":
                        out.writestr(name, contents)
            with self.assertRaisesRegex(ValueError, "REFUSED"):
                audit.validate_zip(dest.getvalue())

    def test_missing_original_zip_is_never_a_pass(self):
        self.mutate_dir(lambda evidence: (evidence / audit.BUNDLE).unlink())

    def test_original_source_receipt_has_separate_chat_and_root_hashes(self):
        valid = audit.validate_receipt(self.receipt)
        self.assertEqual(valid["native_ce"], 4.101235866546631)
        self.assertNotEqual(audit.CHAT_SHA, audit.INSTRUCT_SHA)
        self.assertFalse(valid["train_authorized"])


if __name__ == "__main__":
    unittest.main()
