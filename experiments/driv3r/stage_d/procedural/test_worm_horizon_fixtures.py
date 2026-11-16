"""Deterministic input-only test; independent of later policy outcomes."""
import unittest
from worm_horizon_fixtures import SCENES, SEEDS, SIZE, fixture, preregistration
from worm_sketch import DIRS, digest, canonical, environment


class FixtureTests(unittest.TestCase):
    def test_registered_and_reproducible(self):
        a = canonical(preregistration())
        self.assertEqual(a, canonical(preregistration()))
        self.assertEqual(len(preregistration()['cases']), 12)
        for scene in SCENES:
            for seed in SEEDS:
                spec, legal, guide = fixture(scene, seed)
                self.assertIn(tuple(spec['origin']), legal)
                self.assertTrue(all(0 <= v <= 65535 for v in guide.values()))
                self.assertEqual(len(guide), SIZE[0] * SIZE[1])
                self.assertEqual(spec['permitted_sha256'], digest(canonical(sorted(map(list, legal)))))

    def test_original_three_masks_are_unchanged(self):
        for scene in ('open', 'slats', 'ring'):
            self.assertEqual(fixture(scene, 23)[1], environment(80, 80, scene))

    def test_novel_scenes_are_distinct_and_connected_at_origin(self):
        masks = [fixture(scene, 23)[1] for scene in SCENES]
        self.assertEqual(len(set(map(frozenset, masks))), len(SCENES))
        for scene in SCENES:
            spec, legal, _ = fixture(scene, 23)
            origin = tuple(spec['origin'])
            frontier = [origin]
            seen = {origin}
            for point in frontier:
                for dx, dy in DIRS:
                    q = point[0] + dx, point[1] + dy
                    if q in legal and q not in seen:
                        seen.add(q)
                        frontier.append(q)
            self.assertGreater(len(seen), 100)

    def test_unknown_inputs_rejected(self):
        with self.assertRaises(ValueError):
            fixture('unknown', 23)
        with self.assertRaises(ValueError):
            fixture('open', 0)
        with self.assertRaises(ValueError):
            fixture('open', True)
