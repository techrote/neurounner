"""R10 native original-byte spatial proximity positive/adversarial cases."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from r10_native_mask_proximity import audit, metrics, vicinity

class R10SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus=Path(os.environ["SPROCKET_R10_CORPUS_ZIP"])
        cls.native=Path(os.environ["SPROCKET_R10_NATIVE_ZIP"])

    def test_original_8_native_pairs_one_design(self):
        report=audit(self.corpus,self.native)
        self.assertEqual((report["distinct_reference_groups"],
                          report["reference_native_seed_views"],
                          report["candidate_independent_compositions"]),(4,8,1))
        self.assertFalse(report["final_targets_opened"])
        self.assertEqual(report["new_shared_train_groups"],0)
        self.assertTrue(all(r["candidate_cells"]==1724 for r in report["results"]))

    def test_sealed_final_not_read(self):
        read=zipfile.ZipFile.read
        paths=[]
        def check(obj,name,*args,**kwargs):
            paths.append(str(name))
            self.assertNotIn("final.jsonl",str(name).lower())
            return read(obj,name,*args,**kwargs)
        with patch.object(zipfile.ZipFile,"read",check):
            audit(self.corpus,self.native)
        self.assertTrue(any("development.jsonl" in p for p in paths))

    def test_directionality_and_pixel_boundaries(self):
        a,b={0},{1,2}
        self.assertEqual(metrics(a,b,0)["iou"],0)
        self.assertEqual(metrics(a,b,1)["candidate_within_reference_radius3_cells"],1)
        self.assertEqual(metrics(a,b,1)["reference_within_candidate_radius3_cells"],1)
        self.assertNotIn(96,vicinity({95},1))
        self.assertIn(191,vicinity({95},1))
        self.assertEqual(metrics({0},{0},0)["iou"],1)

    def test_bad_masks_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"empty"):
            metrics(set(),{0})
        with self.assertRaisesRegex(ValueError,"pixel"):
            metrics({9216},{0})
        with self.assertRaisesRegex(ValueError,"radius"):
            metrics({0},{1},-1)

    def test_tampered_corpus_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/"bad.zip"
            bad.write_bytes(self.corpus.read_bytes()+b"x")
            with self.assertRaisesRegex(ValueError,"corpus archive hash drift"):
                audit(bad,self.native)

    def test_tampered_research_native_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/"bad.zip"
            bad.write_bytes(self.native.read_bytes()+b"x")
            with self.assertRaisesRegex(ValueError,"native archive hash drift"):
                audit(self.corpus,bad)
