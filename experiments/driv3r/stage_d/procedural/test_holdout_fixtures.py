"""Input-freeze tests (deliberately no policy import)."""
import unittest
from holdout_fixtures import SPEC, shape, preregister, canonical, digest_points, digest_guide

class IndependentFixtureTests(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        a = preregister()
        b = preregister()
        self.assertEqual(canonical(a), canonical(b))
        self.assertEqual(len(a['cases']), 12)
        self.assertEqual(len({c['initial_sha256'] for c in a['cases']}), 12)
        self.assertEqual(len({c['seed'] for c in a['cases']}), 12)
        self.assertEqual(len({c['family'] for c in a['cases']}), 7)

    def test_positive_u16_hard_permissions_and_custody(self):
        for registered in preregister()['cases']:
            initial, allowed, guide = shape(registered['family'], registered['seed'])
            self.assertTrue(initial <= allowed)
            self.assertEqual(registered['initial_sha256'], digest_points(initial))
            self.assertEqual(registered['permission_sha256'], digest_points(allowed))
            self.assertEqual(registered['guidance_u16le_sha256'], digest_guide(guide))
            self.assertTrue(all(0 < x <= 65535 for x in guide))
            self.assertGreater(len(initial), 0)

    def test_early_stop_negative_control(self):
        a, permitted, _ = shape('disjoint_blocks', 21012)
        self.assertEqual(a, permitted)

    def test_unknown_shape_refused(self):
        with self.assertRaises(ValueError):
            shape('fictional', 6)

if __name__ == '__main__': unittest.main()
