"""Fail-closed evaluation checks on real P0 source features; no native or inference execution."""
import unittest
from pathlib import Path
import diagnose_proxies

HERE = Path(__file__).parent


class ProxyFalsificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = diagnose_proxies.diagnose(HERE / 'source_features.csv')

    def test_exact_source_and_truth_labels(self):
        r = self.report
        self.assertEqual(r['dataset_sha256'], diagnose_proxies.rank.DATA_SHA)
        self.assertEqual(r['claims']['source_worlds'], 8)
        self.assertEqual(r['claims']['learned_model_images'], 8)
        self.assertEqual(r['claims']['model_free_control_images'], 24)
        self.assertEqual(r['human_owner_ratings'], 0)
        self.assertFalse(r['trajectory_recovery_or_stop_measured'])
        self.assertEqual(r['native_godot_receipts'], 0)

    def test_repeated_wall_morphology_blocks_naive_generalisation_claim(self):
        c = self.report['claims']
        self.assertEqual(c['distinct_wall_control_feature_vectors'], 2)
        self.assertEqual(c['largest_repeated_wall_feature_vector'], 7)
        self.assertEqual(len(c['wall_feature_aliases'][0]), 7)

    def test_learned_extent_and_weave_control_ablation(self):
        extent=self.report['axes']['spatial_extent']['matched_learned_vs_control']
        weave=self.report['axes']['local_weave']['matched_learned_vs_control']
        for control in ('random','fidget','wall'):
            self.assertEqual(extent[control]['non_tied_pairs'],8)
            self.assertEqual(extent[control]['learned_proxy_wins'],0)
        self.assertEqual(sum(weave[c]['learned_proxy_wins'] for c in ('random','fidget')),12)
        self.assertEqual(weave['wall']['learned_proxy_wins'],0)

    def test_leave_family_out_is_genuinely_isolated(self):
        for axis in diagnose_proxies.AXES:
            a=self.report['axes'][axis]
            self.assertEqual(a['same_world_pairs'],48)
            for family,result in a['leave_entire_model_family_out'].items():
                self.assertNotEqual(result['training_family'], family)
                self.assertEqual(result['pairs'],24)
                self.assertEqual(result['train_pairs'],24)


if __name__ == '__main__':
    unittest.main()
