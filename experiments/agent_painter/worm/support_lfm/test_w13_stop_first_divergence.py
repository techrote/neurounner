#!/usr/bin/env python3
"""Adversarial STOP-restoration tests; no trained model or optimizer dispatch."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path

import audit_w13_original as original
import w13_stop_first_divergence as probe


def fixture(u: float, count=128):
    legal = ["STRAIGHT", "LEFT", "STOP"]
    sampled = ["STRAIGHT", "LEFT"]
    logits = {a: 0.0 for a in original.ACTIONS}
    masked = original.mixture(logits, sampled)[1]
    action = original.sample(masked, sampled, u)
    decisions, events = [], []
    for tick in range(count):
        decisions.append({
            "tick": tick, "legal": legal,
            "raw_choice_selected_by_host_sampler": action,
            "model_six_code_scores": {"candidate_logits": logits},
            "sampling_receipt": {
                "legal": legal, "sampled_legal": sampled, "choice": action,
                "sampler": "learned", "STOP_deferred_unless_forced": True,
                "temperature": 2.5, "model_fraction": 0.7, "host_u": u,
            },
        })
        events.append({"observation": {"tick": tick, "legal": legal},
                       "raw_choice": action, "applied": action})
    return decisions, events


class StopFirstDivergence(unittest.TestCase):
    def test_first_restored_STOP_is_causally_at_tick_zero(self):
        d, e = fixture(.90)
        row = probe.choices_until_first_divergence(d, e)
        self.assertEqual(row["first_divergence"]["tick"], 0)
        self.assertEqual(row["first_divergence"]["original_action"], "LEFT")
        self.assertEqual(row["first_divergence"]["restored_action"], "STOP")
        self.assertEqual(row["same_state_STOP_selections_NOT_rollouts"], 128)

    def test_nonstop_reweighting_may_diverge_first(self):
        d, e = fixture(.45)
        row = probe.choices_until_first_divergence(d, e)
        self.assertEqual(row["first_divergence"]["tick"], 0)
        self.assertEqual(row["first_divergence"]["original_action"], "STRAIGHT")
        self.assertEqual(row["first_divergence"]["restored_action"], "LEFT")
        self.assertEqual(row["same_state_STOP_selections_NOT_rollouts"], 0)

    def test_no_divergence_must_not_be_fabricated(self):
        d, e = fixture(.1)
        with self.assertRaisesRegex(ValueError, "no STOP-unmasking causal divergence"):
            probe.choices_until_first_divergence(d, e)

    def test_mask_mutation_refused(self):
        d, e = fixture(.9)
        d[0]["sampling_receipt"]["sampled_legal"].append("STOP")
        with self.assertRaises(ValueError):
            probe.choices_until_first_divergence(d, e)

    def test_bad_recorded_original_choice_refused(self):
        d, e = fixture(.9)
        d[5]["raw_choice_selected_by_host_sampler"] = "STRAIGHT"
        with self.assertRaises(ValueError):
            probe.choices_until_first_divergence(d, e)

    def test_wrong_128_turn_budget_refused(self):
        d, e = fixture(.9, 127)
        with self.assertRaises(ValueError):
            probe.choices_until_first_divergence(d, e)

    def test_true_w0_one_step_prefix_replay(self):
        if len(sys.argv) < 2 or not Path(sys.argv[1]).is_dir():
            self.skipTest("W0 source directory not supplied")
        sys.path.insert(0, sys.argv[1])
        import worm
        source = worm.run(seed=6111, family="greebles", mode="wall",
                          steps=128, size=96)
        actions = [e["applied"] for e in source["events"]]
        self.assertEqual(len(actions), 128)
        tick = 2
        self.assertIn("STOP", source["events"][tick]["observation"]["legal"])
        decision = {"first_divergence": {
            "tick": tick, "original_action": actions[tick],
            "restored_action": "STOP",
            "original_observation": source["events"][tick]["observation"],
        }}
        base, alt = probe.paired_prefix(worm, 6111, "greebles", actions, decision)
        self.assertEqual(len(base["events"]), tick+1)
        self.assertEqual(len(alt["events"]), tick+1)
        self.assertTrue(alt["metrics"]["terminated_by_stop"])
        self.assertFalse(base["metrics"]["terminated_by_stop"])
        self.assertNotEqual(worm.png_bytes(base), worm.png_bytes(alt))
        altered = dict(decision)
        altered["first_divergence"] = dict(decision["first_divergence"],
                                            original_observation={"invalid": True})
        with self.assertRaises(ValueError):
            probe.paired_prefix(worm, 6111, "greebles", actions, altered)


if __name__ == "__main__":
    w0_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.argv = [sys.argv[0], w0_dir]
    unittest.main(argv=[sys.argv[0]], verbosity=2)
