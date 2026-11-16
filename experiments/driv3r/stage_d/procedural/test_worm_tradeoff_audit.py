"""Source-replayed audit regression tests, with optional external evidence archive."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from worm_sketch import canonical, digest
from worm_horizon_study import corpus
from worm_tradeoff_audit import paired_report


class WormTradeoffAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        corpus(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exactly_sixty_source_verified_cases(self):
        result = paired_report(self.root)
        self.assertEqual(len(result['cases']), 12)
        self.assertEqual([len(list(self.root.glob('*/manifest.json')))], [60])
        self.assertEqual(result['qualification'], 'POSTHOC_DEVELOPMENT_SYNTHETIC_ONLY')
        self.assertTrue(all(row['sham_identical'] for row in result['cases']))
        self.assertEqual(result['totals']['delta_painted'], 136)
        self.assertEqual(result['totals']['delta_repeats'], -171)

    def test_repeated_report_byte_identical(self):
        self.assertEqual(canonical(paired_report(self.root)), canonical(paired_report(self.root)))

    def test_ink_algebra_and_divergence(self):
        report = paired_report(self.root)
        for row in report['cases']:
            self.assertEqual(row['delta_painted'], row['new_ink_not_in_baseline'] - row['baseline_ink_missing_in_new'])
            self.assertLessEqual(row['ink_jaccard_numerator'], row['ink_jaccard_denominator'])
            if row['scene'] == 'baffled' and row['seed'] == 79:
                self.assertIsNone(row['first_cursor_divergence_step'])
            else:
                self.assertIsInstance(row['first_cursor_divergence_step'], int)

    def test_drop_case_fails_closed(self):
        original = self.root / 'open-seed23-strong_x16'
        moved = self.root / '__moved'
        original.rename(moved)
        try:
            with self.assertRaises(ValueError):
                paired_report(self.root)
        finally:
            moved.rename(original)

    def test_rehashed_artifact_tamper_fails_closed(self):
        folder = self.root / 'open-seed23-strong_x16'
        path = folder / 'trajectory.json'
        old = path.read_bytes()
        mpath = folder / 'manifest.json'
        old_manifest = mpath.read_bytes()
        try:
            path.write_bytes(old + b' ')
            manifest = json.loads(old_manifest)
            manifest['artifacts']['trajectory.json'] = digest(path.read_bytes())
            mpath.write_bytes(canonical(manifest))
            with self.assertRaises(ValueError):
                paired_report(self.root)
        finally:
            path.write_bytes(old)
            mpath.write_bytes(old_manifest)

    def test_no_native_or_model_evidence_promotion(self):
        report = paired_report(self.root)
        self.assertIsNone(report['native'])
        self.assertIsNone(report['model'])
        self.assertIsNone(report['human_art_judgment'])
        self.assertIn('ink_overlap', report['not_preregistered'])


if __name__ == '__main__':
    unittest.main()
