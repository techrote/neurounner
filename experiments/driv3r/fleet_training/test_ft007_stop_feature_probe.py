"""Adversarial tests for frozen-checkpoint time-feature diagnostic custody."""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

import numpy as np
import ft007_stop_feature_probe as f


class StopFeatureProbeTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((Path(__file__).parent / "request_ft007.json").read_text())

    def test_request_rejects_any_source_budget_or_optimization_mutation(self):
        self.assertEqual(f.check_request(self.request), self.request)
        for key, value in (
            ("ft003_artifact", 0), ("ft003_archive_sha256", "a"*64),
            ("source_blob", "0"*40), ("optimizer_steps", 1),
            ("native_material_edits", 1), ("max_runner_jobs", 4),
            ("max_seconds", 3600), ("probe_worlds_per_scenario", 999),
            ("early_grid", [0, 1]), ("frozen_doses", [48,48]),
        ):
            changed = copy.deepcopy(self.request)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                f.check_request(changed)

    def test_probe_state_cannot_masquerade_as_autonomous_or_training(self):
        def stable(*parts):
            return int.from_bytes(hashlib.sha256(
                "|".join(map(str, parts)).encode()).digest()[:8], "big") % 2**32
        pop = types.SimpleNamespace(stable_seed=stable)
        self.assertNotEqual(
            f.world_seed(pop,"PROBE",9401,"navigator","long60",0),
            f.world_seed(pop,"AUTONOMOUS",9401,"navigator","long60",0))
        for args in [
            ("TRAIN",9401,"navigator","long60",0),
            ("PROBE",9999,"navigator","long60",0),
            ("PROBE",9401,"navigator","long60",999),
            ("PROBE",9401,"navigator","nonsense",0),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                f.world_seed(pop,*args)
        self.assertEqual(len(f.custody_digest(pop)), 64)

    def test_host_only_legal_teacher_fallbacks(self):
        pop = types.SimpleNamespace(
            DIRECTIONS={"N":(0,-1), "E":(1,0), "S":(0,1), "W":(-1,0)},
            stable_seed=lambda *parts: 0)
        for turn, menu, expected in (
            (1, ["E","STOP"], "E"), (8,["RETRACE","STOP"],"RETRACE"),
            (0,["STOP"],"STOP"), (8,["N","RETRACE","STOP"],"RETRACE")):
            w = types.SimpleNamespace(steps=turn,legal=lambda m=menu:list(m))
            self.assertEqual(f.teacher_action(pop,w,42),expected)

    def test_counterfactual_rank_flip_is_not_an_actual_world_step(self):
        class ScoringPolicy:
            def logits(self,X):
                # Fixed nonSTOP score 0; STOP crosses it when early exceeds .5.
                return X[:,3] * (X[:,12] - .5)
        X = np.zeros((2,20))
        X[1,3] = 1
        X[:,12] = .2
        old = X.copy()
        panel = f.score_state(ScoringPolicy(),X,["E","STOP"],5,"digest",77,12)
        self.assertTrue(panel["stop_rank_flip"])
        self.assertTrue(panel["any_action_flip"])
        self.assertEqual(panel["original_choice"],"E")
        np.testing.assert_array_equal(X,old)
        self.assertEqual(len(panel["grid"]),5)
        self.assertIsNone(f.score_state(ScoringPolicy(),X[:1],["STOP"],5,"d",77,12))

    def test_common_time_bias_does_not_change_action_ranking(self):
        class SharedBias:
            def logits(self,X):
                return 3 * X[:,12] + X[:,1]
        X = np.zeros((2,20))
        X[0,1] = 1
        p = f.score_state(SharedBias(),X,["E","STOP"],0,"digest",1,12)
        self.assertFalse(p["any_action_flip"])
        self.assertFalse(p["stop_rank_flip"])
        self.assertAlmostEqual(p["stop_margin_span"],0)


if __name__ == "__main__":
    unittest.main()
