"""WORM-001 LFM support: executable source-only behaviour and split checks."""
from __future__ import annotations
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from curriculum import (TRAIN_SEEDS, DEV_SEEDS, canonical, checked_splits,
                        evaluate_policy, fresh_directions, oracle, prompt, adverse_rows,
                        rollout, rows, save_gallery, statistics, worm, split_distinct_actions)


class LfmWormSupportTests(unittest.TestCase):
    def test_schema_and_reproducible_disjoint_seed_groups(self):
        train, dev, _audit = split_distinct_actions()
        checked_splits(train, dev)
        self.assertFalse(set(TRAIN_SEEDS) & set(DEV_SEEDS))
        self.assertEqual(len({r["id"] for r in train+dev}), len(train+dev))
        self.assertEqual(canonical(train), canonical(split_distinct_actions()[0]))
        self.assertTrue(all(r["target_action"] in r["observation"]["legal"] for r in train+dev))
        self.assertTrue(all("teacher" not in r["prompt"].lower() for r in train+dev))

    def test_vocabulary_and_prompt_are_host_local(self):
        for family in worm.FAMILIES:
            rec = worm.run(1102, "wall", family, 20)
            for event in rec["events"]:
                o = event["observation"]
                self.assertIn(oracle(o), o["legal"])
                self.assertIn(oracle(o), worm.ACTIONS)
                self.assertIn("Action:", prompt(o))
                self.assertNotIn("target_action", prompt(o))

    def test_online_rollout_observation_is_authentic_replay(self):
        record = rollout(2101, "pipework", 30, oracle)
        actions = [e["raw_choice"] for e in record["events"]]
        replay = worm.run(2101, "replay", "pipework", 30, raw_actions=actions)
        self.assertEqual({k:v for k,v in record.items() if k != "mode"},
                         {k:v for k,v in replay.items() if k != "mode"})
        self.assertEqual(record["mode"], "callback")
        self.assertTrue(all(e["raw_choice"] == oracle(e["observation"]) for e in record["events"]))

    def test_raw_invalid_never_converted_to_legal_model_claim(self):
        record = rollout(2101, "greebles", 16, lambda _: "TELEPORT")
        self.assertEqual(record["events"][0]["raw_choice"], "TELEPORT")
        self.assertFalse(record["events"][0]["raw_legal"])
        self.assertEqual(record["metrics"]["invalid_raw_choices"], 1)
        self.assertIn(record["events"][0]["applied"], ("STOP", "RETRACE"))
        self.assertEqual(statistics(record)["invalid_raw_count"], 1)

    def test_no_cross_split_policy_evaluation(self):
        with self.assertRaises(ValueError):
            evaluate_policy("lfm", oracle, seeds=(TRAIN_SEEDS[0],))
        with self.assertRaises(ValueError):
            rows("final")
        with self.assertRaises(ValueError):
            rollout(2101, "pipework", 513, oracle)
        with self.assertRaises(TypeError):
            rollout(2101, "pipework", 3, lambda _: None)

    def test_real_pngs_equal_obstacle_and_decision_budgets(self):
        with TemporaryDirectory() as tmp:
            summary = save_gallery(Path(tmp), steps=20)
            self.assertEqual(sum(f["file"].endswith(".png") for f in summary["files"]), 24)
            self.assertEqual(len(summary["reports"]), 24)
            self.assertEqual(len({(r["family"],r["seed"],r["policy"])
                                 for r in summary["reports"]}), 24)
            for f in summary["files"]:
                data = (Path(tmp)/f["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), f["sha256"])
                if f["file"].endswith(".png"):
                    self.assertEqual(data[:8], bytes((137,80,78,71,13,10,26,10)))
            for r in summary["reports"]:
                self.assertLessEqual(r["metrics"]["decision_count"], 20)
                self.assertGreaterEqual(r["metrics"]["coverage_free_fraction"], 0)
                self.assertLessEqual(r["metrics"]["invalid_raw_rate"], 1)

    def test_exact_prompt_disjointness_v2_includes_initial_world_aliases(self):
        # W0 identical local observations can occur under *different* world seeds.
        raw_train, raw_dev = rows("train", steps=12), rows("development", steps=12)
        self.assertTrue({x["prompt"] for x in raw_train} &
                        {x["prompt"] for x in raw_dev})
        train, dev, audit = split_distinct_actions()
        self.assertEqual(audit["literal_cross_split_prompt_overlap"], 0)
        self.assertGreater(audit["distinct_overlapping_prompt_count"], 0)
        self.assertGreater(audit["removed_train_for_exact_development_prompt"], 0)
        self.assertEqual(len(train), len({x["prompt"] for x in train}))
        self.assertEqual(len(dev), len({x["prompt"] for x in dev}))
        self.assertFalse({x["prompt"] for x in train} & {x["prompt"] for x in dev})
        self.assertEqual(len(audit["retained_train_worlds"]), 12)
        checked_splits(train + adverse_rows("train"),
                       dev + adverse_rows("development"))

    def test_raw_literal_prompt_leakage_is_fail_closed(self):
        train, dev = rows("train", steps=4), rows("development", steps=4)
        self.assertTrue({r["prompt"] for r in train} & {r["prompt"] for r in dev})
        with self.assertRaisesRegex(ValueError, "literal prompt"):
            checked_splits(train, dev)

    def test_raw_model_collapse_not_hidden_by_host_fallback(self):
        # An endlessly repeated raw STRAIGHT eventually hits a boundary; host
        # RETRACE is not evidence of learned action diversity or recovery.
        rec = rollout(2101, "architecture", 120, lambda _: "STRAIGHT")
        scores = statistics(rec)
        self.assertEqual(scores["raw_action_counts"], {"STRAIGHT": 120})
        self.assertEqual(scores["max_same_raw_streak"], 120)
        self.assertEqual(scores["raw_action_entropy_bits"], 0)
        self.assertEqual(scores["raw_straight_fraction"], 1)
        self.assertGreater(scores["host_fallback_count"], 0)
        self.assertGreater(scores["action_entropy_bits"], 0)
        self.assertEqual(scores["invalid_raw_count"], scores["host_fallback_count"])

    def test_heldout_train_leakage_is_fail_closed(self):
        a = rows("train", 3)
        b = rows("development", 3)
        b[0]["world_group"] = a[0]["world_group"]
        with self.assertRaises(ValueError):
            checked_splits(a, b)

    def test_separate_adverse_local_motifs_and_honest_provenance(self):
        train, dev = adverse_rows("train"), adverse_rows("development")
        self.assertEqual(len(train), 72)
        self.assertEqual(len(dev), 48)
        self.assertFalse({x["world_group"] for x in train} &
                         {x["world_group"] for x in dev})
        self.assertIn("STOP", {x["target_action"] for x in train})
        self.assertIn("RETRACE", {x["target_action"] for x in train})
        self.assertTrue(all(x["observation_source"].startswith("counterfactual")
                            for x in train + dev))
        self.assertTrue(all(x["target_action"] in x["host_legal"] for x in train + dev))
        replay_train, replay_dev, _audit = split_distinct_actions()
        checked_splits(replay_train + train, replay_dev + dev)

    def test_adverse_obstacle_and_recovery_action(self):
        o = {"heading":0, "family":"architecture", "tick":9,
             "local_5x5":[".....","..#..",".#@#.","..#..","....."],
             "legal":["RETRACE","STOP"], "last_actions":[]}
        self.assertEqual(fresh_directions(o), [])
        self.assertEqual(oracle(o), "RETRACE")
        o["legal"]=["STOP"]
        self.assertEqual(oracle(o), "STOP")


if __name__ == "__main__":
    unittest.main()
