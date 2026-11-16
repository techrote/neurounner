"""Strict positive/adversarial original-native A07/A08 visible-split audit."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from a07_a08_original_native_visible import audit, require_exact_cases, original_pixels, SEEDS

class ArchitectureOriginalVisibleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Path(os.environ["SPROCKET_A_CORPUS_ZIP"])
        cls.native = Path(os.environ["SPROCKET_A_NATIVE_ZIP"])

    def test_exact_two_designs_four_candidate_views_eight_reference_views(self):
        x = audit(self.corpus, self.native)
        self.assertEqual((x["candidate_structures"], x["candidate_seed_views"],
                          x["reference_structures"], x["reference_seed_views"],
                          x["pairwise_comparisons"]), (2, 4, 4, 8, 16))
        self.assertEqual({r["candidate"] for r in x["results"]}, {"A07","A08"})
        self.assertEqual({r["reference_seed"] for r in x["results"]}, set(SEEDS))
        self.assertEqual({r["acknowledged_cells"] for r in x["original_native_cases"]},
                         {781,924})
        self.assertEqual((x["new_native_executions"], x["new_model_outputs"],
                          x["optimizer_updates"], x["new_shared_train_admissions"]), (0,0,0,0))
        self.assertFalse(x["final_targets_opened"])

    def test_final_member_not_read_or_decompressed(self):
        read = zipfile.ZipFile.read
        observed = []
        def guard(zipobj, name, *args, **kwargs):
            observed.append(str(name))
            self.assertNotIn("final.jsonl", str(name).lower())
            return read(zipobj, name, *args, **kwargs)
        with patch.object(zipfile.ZipFile, "read", guard):
            audit(self.corpus, self.native)
        self.assertTrue(any("development.jsonl" in n for n in observed))

    def test_missing_second_seed_is_rejected(self):
        group = "architecture/a/v1"
        rows = [{"structural_group":group,"seed":SEEDS[0]}]
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            require_exact_cases(rows, {group})

    def test_duplicate_reference_seed_is_rejected(self):
        group = "architecture/a/v1"
        rows = [{"structural_group":group,"seed":SEEDS[0]} for _ in range(2)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            require_exact_cases(rows, {group})

    def test_tampered_original_corpus_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"tampered.zip"
            path.write_bytes(self.corpus.read_bytes() + b"extra")
            with self.assertRaisesRegex(ValueError, "corpus ZIP hash drift"):
                audit(path, self.native)

    def test_tampered_original_native_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"tampered.zip"
            path.write_bytes(self.native.read_bytes() + b"extra")
            with self.assertRaisesRegex(ValueError, "native ZIP hash drift"):
                audit(self.corpus, path)

if __name__ == "__main__":
    unittest.main()
