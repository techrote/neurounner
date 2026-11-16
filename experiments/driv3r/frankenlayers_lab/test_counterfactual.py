"""Cheap source-only contract tests for paired-geometry game-policy probes."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.driv3r.frankenlayers import infer as paired
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS
from experiments.driv3r.frankenlayers_lab.test_holdout import MiniW0
from experiments.driv3r.frankenlayers_lab.counterfactual import (
    ALL_VARIANTS, PAIRS, FAMILY, SEED, SIZE, STEPS, CausalHost,
    intervention_world, variant_obstacles, first_difference, pair_diagnostic,
)


class CausalTests(unittest.TestCase):
    def test_interventions_are_paired_and_bounded(self):
        self.assertEqual(len(PAIRS), 3)
        self.assertEqual(len(ALL_VARIANTS), 6)
        base = {(7, SIZE // 2), (9, SIZE // 2), (42, 42)}
        for variant in ALL_VARIANTS:
            blocks = intervention_world(base, variant)
            self.assertNotIn((7, SIZE // 2), blocks)
            self.assertIn((42, 42), blocks)
            self.assertTrue(all(0 <= x < SIZE and 0 <= y < SIZE for x, y in blocks))
        self.assertNotEqual(variant_obstacles("east_open"),
                            variant_obstacles("east_blocked"))
        self.assertNotEqual(variant_obstacles("north_blocked"),
                            variant_obstacles("south_blocked"))
        self.assertNotEqual(variant_obstacles("far_open"),
                            variant_obstacles("far_blocked"))
        with self.assertRaises(ValueError):
            variant_obstacles("imposter")

    def test_fixed_family_seed_and_restoration(self):
        w0 = MiniW0()
        orig = w0.initial_world.__func__
        for variant in ALL_VARIANTS:
            host = CausalHost(w0, variant)
            record = host.run(SEED, mode="callback", family=FAMILY, steps=STEPS,
                              selector=lambda obs: "STRAIGHT")
            self.assertEqual(record["research_intervention"]["fixed_family"], FAMILY)
            self.assertFalse(record["research_intervention"]["native_game"])
            self.assertEqual(w0.initial_world.__func__, orig)
            self.assertEqual(host.png_bytes(record), b"png")
        with self.assertRaises(ValueError):
            CausalHost(w0, "east_blocked").run(
                570, mode="callback", family=FAMILY, steps=STEPS,
                selector=lambda obs: "STRAIGHT")

    def test_diagnostic_does_not_confuse_nondeterminism_with_geometry(self):
        first = lambda observation, actions: {
            "calls": [{"observation": o} for o in observation],
            "arm": {"experiments": [{
                "raw_choices": actions,
                "metrics": {"new_steps": 1, "terminated_by_stop": False}}]}}
        a = first([{"local_5x5": "A"}, {"local_5x5": "B"}], ["BRANCH", "RIGHT"])
        b = first([{"local_5x5": "C"}, {"local_5x5": "B"}], ["BRANCH", "LEFT"])
        d = pair_diagnostic(a, b)
        self.assertEqual(d["first_observation_difference"], 0)
        self.assertEqual(d["first_action_difference"], 1)
        self.assertTrue(d["action_divergence_after_observation"])
        self.assertFalse(d["unexplained_early_action_divergence"])
        e = first([{"local_5x5": "A"}], ["RIGHT"])
        f = first([{"local_5x5": "A"}], ["LEFT"])
        self.assertTrue(pair_diagnostic(e, f)["unexplained_early_action_divergence"])
        self.assertIsNone(first_difference([1, 2], [1, 2]))

    def test_actual_inherited_runner_positional_seed_contract(self):
        class FakeServer:
            def __init__(self, binary, model, mode, root, seed):
                self.root = root
            def __enter__(self):
                self.root.mkdir(parents=True, exist_ok=True)
                return self
            def choose(self, obs, records):
                records.append({"choice": "STRAIGHT", "status": "ok",
                                "latency_seconds": .01,
                                "input_tokens_per_second": 100.0})
                return "STRAIGHT"
            def __exit__(self, *args):
                return False

        w0 = MiniW0()
        old_steps, old_families = paired.STEPS, paired.FAMILIES
        with TemporaryDirectory() as tmp, patch.object(paired, "Server", FakeServer), \
                patch.dict(paired.FLOWS, LAB_FLOWS):
            try:
                paired.STEPS, paired.FAMILIES = STEPS, (FAMILY,)
                arm = paired.run_arm(Path("binary"), Path("model"),
                    CausalHost(w0, "east_blocked"), "aero4_no10",
                    Path(tmp) / "trial", SEED)
            finally:
                paired.STEPS, paired.FAMILIES = old_steps, old_families
            self.assertEqual(arm["valid_decisions"], 1)
            self.assertEqual(arm["experiments"][0]["family"], FAMILY)
            self.assertTrue((Path(tmp) / "trial" /
                "aero4_no10-architecture-choices.json").is_file())


if __name__ == "__main__":
    unittest.main()
