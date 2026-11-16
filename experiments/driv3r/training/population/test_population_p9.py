"""Guardrails for P9 adaptive environment curricula, state splits and real updates."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from population_lab import World, NICHES, Policy, stable_seed
from population_p9 import (ARMS,CARDS,scenario_world,play,train_episode,split_seeds,
                           offered_cards,choose_card,run_p9)


class P9Tests(unittest.TestCase):
    def test_synthetic_generator_preserves_protected_and_start(self):
        for seed in (1,21,847):
            for card in range(len(CARDS)):
                w=scenario_world(seed,card,32)
                self.assertFalse(w.painted & (w.obstacles|w.protected))
                self.assertIn((3,11),w.painted)
                self.assertIn("STOP",w.legal())
                self.assertTrue(w.goal not in w.protected and w.goal not in w.obstacles)
                self.assertTrue(w.second_goal not in w.protected and w.second_goal not in w.obstacles)
        with self.assertRaises(ValueError):
            scenario_world(1,999)

    def test_legal_action_mask_during_training_and_play(self):
        for niche in NICHES:
            for card in range(len(CARDS)):
                p=Policy(stable_seed("p9",niche,card))
                t=train_episode(p,niche,138,card,32)
                self.assertGreater(t["steps"],0)
                e=play(p,niche,139,card,32)
                self.assertFalse(e["world"].painted & (e["world"].protected|e["world"].obstacles))
                self.assertTrue(all(h["action"] in ("N","E","S","W","RETRACE","STOP")
                                    for h in e["world"].history))

    def test_seed_disjoint_and_card_offer_arm_independent(self):
        s=split_seeds(640,2,6,3,4)
        flat=sum(s["train"],[])+s["selection"]+s["holdout"]
        self.assertEqual(len(flat),len(set(flat)))
        a=offered_cards(640,"navigator",0,0,1)
        self.assertEqual(a,offered_cards(640,"navigator",0,0,1))
        self.assertEqual(len(set(a)),3)
        self.assertTrue(all(0<=v<6 for v in a))

    def test_adaptive_selection_recovers_scored_best(self):
        p=Policy(8)
        frozen=Policy(9)
        offered=[0,1,2]
        cache={}
        for arm in ARMS:
            c,record=choose_card(arm,p,frozen,"navigator",123,offered,0,32,cache)
            self.assertIn(c,offered)
            self.assertEqual(len(record["live"]),len(offered))
            if arm=="adaptive_hardness":
                self.assertEqual(c,offered[int(np.argmax(record["live"]))])
            if arm=="frozen_hardness":
                self.assertEqual(c,offered[int(np.argmax(record["frozen"]))])
        self.assertEqual(len(cache),3)

    def test_small_tournament_real_training_frozen_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p0=root/"p0"
            (p0/"models").mkdir(parents=True)
            for niche in NICHES:
                (p0/"models"/(niche+".json")).write_text(json.dumps(
                    Policy(stable_seed("p9",niche,"parent")).serialized()))
            (p0/"results.json").write_text(json.dumps(dict(
                format="population-research-v1",language_model=False,seed=619,
                train_seed_count=4,selection_seed_count=3,heldout_seed_count=3)))
            r=run_p9(p0,root/"p9",seed=647,generations=2,offspring=2,
                     train_episodes=4,selection_episodes=3,holdout_episodes=4,
                     horizon=32,portfolio=2,proposals=2)
            self.assertEqual(r["candidates"],6*3*2*2)
            self.assertGreater(r["adam_updates"],0)
            self.assertFalse(r["native"])
            self.assertFalse(r["language_model"])
            self.assertFalse(r["actual_self_play"])
            self.assertFalse(r["game_code_modified"])
            self.assertTrue(all(rec["initial_sha256"]!=rec["final_sha256"] for rec in r["lineage"]))
            self.assertEqual(len(list((root/"p9"/"models").glob("*.json"))),r["candidates"])
            self.assertEqual(len(list((root/"p9"/"gallery").glob("*-trace.json"))),6*3*2)
            self.assertTrue(all(set(r["comparison"][n])==set(ARMS) for n in NICHES))
            self.assertTrue(all(set(r["comparison"][n][arm]) >= {"familiar","unseen_geometry"}
                                for n in NICHES for arm in ARMS))
            self.assertEqual(len(set(x["training"]["probe_policy_rollouts"] for x in r["lineage"])),1)

    def test_source_boundary_bounded_campaign(self):
        with self.assertRaises(ValueError):
            split_seeds(1,0,0,0,0) if False else scenario_world(1,0,500)

if __name__=="__main__":
    unittest.main()
