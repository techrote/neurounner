"""Adversarial archival split/native R10 review tests; no FINAL access or training."""
import os
from pathlib import Path
import tempfile
import unittest
from r10_split_native_review import evaluate, topology

class R10VisibleNativeTests(unittest.TestCase):
    def setUp(self):
        self.corpus=Path(os.environ["SPROCKET_R10_CORPUS_ZIP"])
        self.native=Path(os.environ["SPROCKET_R10_NATIVE_ZIP"])

    def test_real_original_archives_are_one_native_structure(self):
        report=evaluate(self.corpus,self.native)
        self.assertEqual(report["independent_structures"],1)
        self.assertEqual(report["seed_views"],2)
        self.assertEqual(report["proven_native_research_views"],2)
        self.assertEqual(report["new_shared_train_groups"],0)
        self.assertFalse(report["final_targets_opened"])
        self.assertEqual(len(report["visible_original_pipe_groups"]),4)
        self.assertEqual(report["pinned_void_isomorphism_matches"],[])
        self.assertTrue(all(r["acknowledged_cells"]==1724 for r in report["native_seed_cases"]))

    def test_tampered_corpus_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/"corrupt-corpus.zip"
            bad.write_bytes(self.corpus.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError,"corpus archive hash drift"):
                evaluate(bad,self.native)

    def test_tampered_native_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/"corrupt-native.zip"
            bad.write_bytes(self.native.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError,"native archive hash drift"):
                evaluate(self.corpus,bad)

    def test_topology_rejects_isolated_node(self):
        from rich_pipework_v2 import program_for
        p=program_for("R10")
        p["components"][0]["edges"]=[e for e in p["components"][0]["edges"] if 9 not in e]
        with self.assertRaisesRegex(ValueError,"isolated"):
            topology(p)

if __name__=="__main__":unittest.main()
