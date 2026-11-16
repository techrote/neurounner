"""Fail-closed split, geometry, legality and reproducibility checks for v5."""
import tempfile
import unittest
from pathlib import Path
from transfer_orientation_v5 import (make_vertical_comb_world, validate,
    oracle_goal_rollout, run)
from domain_recovery_v3 import make_slalom_world,world_digest
from recovery_curriculum import distance_field

class OrientationTests(unittest.TestCase):
    def test_generator_deterministic_and_distinct(self):
        for seed in (8500,8501,8502):
            a=make_vertical_comb_world(seed);b=make_vertical_comb_world(seed)
            self.assertEqual(world_digest(a),world_digest(b))
            self.assertNotEqual(world_digest(a),world_digest(make_slalom_world(seed)))
            self.assertFalse(a['obstacles'] & a['protected'])
            self.assertNotIn(a['start'],a['obstacles']|a['protected'])
            self.assertNotIn(a['goal'],a['obstacles']|a['protected'])
            self.assertIsNotNone(distance_field(a).get((*a['start'],0)))
    def test_seed_overlap_fails_before_training(self):
        with self.assertRaisesRegex(ValueError,'seed overlap'):
            validate((1000,),(2500,),(7500,),(8500,),(1000,),(),(),192)
    def test_arms_same_world_count(self):
        with self.assertRaisesRegex(ValueError,'size-matched'):
            validate((1000,),(2500,2501),(7500,),(8500,),(),(),(),192)
    def test_budget_guard_fails_before_training(self):
        with self.assertRaisesRegex(ValueError,'budget must'):
            validate((1000,),(2500,),(7500,),(8500,),(),(),(9500,),135)
    def test_privileged_oracle_is_goal_achieving(self):
        for seed in (9500,9501):
            w=make_slalom_world(seed)
            self.assertEqual(oracle_goal_rollout(w,192),distance_field(w).get((*w['start'],0)))
    def test_immutable_output_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                run(Path(temp))

if __name__=='__main__':unittest.main()
