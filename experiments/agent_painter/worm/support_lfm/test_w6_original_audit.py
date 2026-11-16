#!/usr/bin/env python3
"""Source-only adversarial geometry and fail-closed evidence tests."""
import tempfile
import unittest
from pathlib import Path
from w6_original_audit import spatial, gallery_digest, training_evidence

class W6SourceTests(unittest.TestCase):
    def test_bounded_24_prefix_and_48_decision_reach(self):
        trajectory=[[7,48]] + [[8+i,48] for i in range(48)]
        e=spatial(trajectory)
        self.assertEqual(e["full"]["bbox_width"],49)
        self.assertEqual(e["full"]["bbox_height"],1)
        self.assertEqual(e["prefix24"]["executed_steps"],24)
        self.assertEqual(e["prefix24"]["max_manhattan_from_start"],24)

    def test_short_trace_does_not_claim_full_budget(self):
        e=spatial([[7,48],[8,48],[7,48]])
        self.assertEqual(e["prefix24"]["executed_steps"],2)
        self.assertEqual(e["prefix24"]["unique_positions"],2)

    def test_invalid_positions(self):
        for bad in ([],[[7,48],[96,1]],[(7,48)],[[7,True]],[[7,48]]*514):
            with self.subTest(bad=str(bad)[:20]), self.assertRaises(ValueError):
                spatial(bad)
        with self.assertRaises(ValueError):spatial([[7,48]], prefix=True)

    def test_missing_symlink_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):gallery_digest(root)
            (root/"a").write_text("a")
            (root/"b").symlink_to("a")
            with self.assertRaises(ValueError):gallery_digest(root)
            with self.assertRaises(ValueError):training_evidence(root)

if __name__=="__main__":unittest.main()
