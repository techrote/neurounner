"""Offline original-host evidence retention/adversarial checks, no model imports."""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_nca_original_custody as custody  # noqa: E402


class OriginalVoidDiagnosticCustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = custody.audit()
        cls.manifest = json.loads(custody.MANIFEST.read_text())

    def test_original_archive_identity(self):
        self.assertEqual(self.manifest["original_run"], 40765025332)
        self.assertEqual(
            custody.sha(custody.original_zip("source")), custody.SHAS["source"])
        self.assertEqual(
            custody.sha(custody.original_zip("host")), custody.SHAS["host"])

    def test_four_lane_policy_and_non_admission(self):
        p = json.loads(custody.PROTOCOL.read_text())
        self.assertFalse(p["train_authorized"])
        self.assertEqual(set(p["models"]),
                         {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        result = self.original
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["full_72_conformance_pass"])
        self.assertFalse(result["comparative_training_performed"])
        self.assertFalse(result["independent_scientific_review"])
        self.assertTrue(result["model_training_experiment"] is False)
        self.assertEqual(result["diagnostic_optimizer_steps"], 1)

    def test_exact_source_and_actual_eos(self):
        result = self.original
        self.assertEqual(result["actual_source_files"]["model.safetensors"]["sha256"],
                         custody.WEIGHT)
        self.assertEqual(result["actual_source_files"]["tokenization_byte.py"]["sha256"],
                         custody.TOKENIZER)
        self.assertEqual(result["synthetic_sample"]["real_eos_id"], 258)
        self.assertEqual(result["synthetic_sample"]["supervised_after_shift"], 21)

    def test_post_warmup_signal_but_no_quality_result(self):
        result = self.original
        self.assertEqual(result["backbone_before_sha256"],
                         result["backbone_after_sha256"])
        self.assertGreater(
            result["post_warmup"]["gradient_l1"]["observe.weight"], 0)
        self.assertGreater(
            result["post_warmup"]["gradient_l1"]["rule.weight"], 0)
        self.assertEqual(len(result["post_warmup"]["ordered_field_trace"]), 3)

    def test_original_semantic_falsifications_rejected(self):
        result = custody.adversarial()
        self.assertEqual(result["negative_mutations_refused"], 14)
        self.assertFalse(result["full_72_conformance_pass"])

    def test_archive_zip_byte_corruption_rejected(self):
        archive = custody.original_zip("host")
        changed = archive[:-1] + bytes([archive[-1] ^ 1])
        self.assertNotEqual(custody.sha(changed), custody.SHAS["host"])

    def test_original_receipt_falsified_eos_rejected(self):
        changed = json.loads(json.dumps(self.original))
        changed["synthetic_sample"]["real_eos_id"] = 62
        with self.assertRaisesRegex(ValueError, "synthetic byte/EOS"):
            custody.inspect_receipt(changed)


if __name__ == "__main__":
    unittest.main()
