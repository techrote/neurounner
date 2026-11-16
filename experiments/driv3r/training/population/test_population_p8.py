"""P8 temporal encoder and evolutionary control regression/guardrail tests."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from population_lab import Policy, World, NICHES, stable_seed
from population_p8 import (ARMS, sequence, disorder, windows, fit_temporal, temporal_cell,
                           encode, parent_for, select_ids, run_p8)
from population_p6 import split_seeds


class P8Tests(unittest.TestCase):
    def test_true_order_vs_permuted_same_actions(self):
        w=World.generate(29)
        for _ in range(8):
            options=[a for a in w.legal() if a != "STOP"]
            if not options:
                break
            w.step(options[0])
        tokens,n=sequence(w)
        self.assertGreater(n,1)
        r=np.random.default_rng(7)
        for name in ("reverse","shuffle"):
            changed=disorder(tokens,n,name,r)
            np.testing.assert_array_equal(changed.sum(axis=0),tokens.sum(axis=0))
            np.testing.assert_array_equal(changed[n:],tokens[n:])
        self.assertEqual(windows(tokens[None]).shape,(1,30,30))
        with self.assertRaises(ValueError):
            windows(tokens[:5][None])

    def test_encoder_train_only_determinism_and_permutation_sensitivity(self):
        parents={n:Policy(stable_seed("p8",n,"test")) for n in NICHES}
        a=fit_temporal(parents,640,probe_count=4,probe_worlds=3,steps=40,batch_size=18)
        b=fit_temporal(parents,640,probe_count=4,probe_worlds=3,steps=40,batch_size=18)
        self.assertEqual(a["sha256"],b["sha256"])
        self.assertGreater(a["optimizer_steps"],0)
        self.assertTrue(np.isfinite(a["positive_variances"]).all())
        self.assertEqual(len(a["cuts"]),(3))
        self.assertGreater(a["encoder_validation_accuracy"],0)
        transcript=type("Trace",(),{"history":[{"action":k} for k in
             ("N","N","E","S","E","RETRACE","W","STOP")]})()
        token,n=sequence(transcript)
        z=encode(token,n,a)
        inverted=encode(disorder(token,n,"reverse",np.random.default_rng(1)),n,a)
        self.assertGreater(float(np.linalg.norm(z-inverted)),1e-6)
        cell,_=temporal_cell([(token,n)],a)
        self.assertEqual(len(cell),3)
        self.assertTrue(all(0<=v<=3 for v in cell))

    def test_legal_world_and_sequence_input_rejection(self):
        w=World.generate(12)
        self.assertFalse(w.painted & w.protected)
        w.history.append(dict(action="INJECT_ILLEGAL"))
        with self.assertRaises(ValueError):
            sequence(w)
        with self.assertRaises(ValueError):
            fit_temporal({},640,steps=1)

    def test_parent_and_frozen_portfolio_controls(self):
        rec=lambda name,score:dict(id=name,selection={"mean_reward":score})
        r=[rec("a",1),rec("b",3),rec("c",2)]
        arc={(0,):r[0],(1,):r[1]}
        self.assertIsNone(parent_for("temporal_map",0,0,640,"navigator",arc,r[1],r))
        self.assertEqual(parent_for("reward_pbt",1,0,640,"navigator",arc,r[1],r),r[1])
        self.assertIn(parent_for("temporal_map",1,0,640,"navigator",arc,r[1],r),r)
        for arm in ARMS:
            ids=select_ids(r,arc,arm,2,7)
            self.assertEqual(len(ids),len(set(ids)))

    def test_unseen_worlds_disjoint_from_encoder_and_p0(self):
        s=split_seeds(638,4,3,3,2)
        train=[stable_seed("p8",638,"encoder_train","world",i) for i in range(3)]
        valid=[stable_seed("p8",638,"encoder_validation","world",i) for i in range(3)]
        pca=[stable_seed("p7",638,"encoder_train",i) for i in range(3)]
        flat=sum(s["train"],[])+s["selection"]+s["holdout"]+train+valid+pca
        self.assertEqual(len(flat),len(set(flat)))

    def test_mini_training_evidence_all_niches_and_no_game_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            p0=root/"p0"
            (p0/"models").mkdir(parents=True)
            for niche in NICHES:
                (p0/"models"/(niche+".json")).write_text(json.dumps(
                    Policy(stable_seed(niche,619)).serialized()))
            (p0/"results.json").write_text(json.dumps(dict(
                format="population-research-v1",language_model=False,seed=619,
                train_seed_count=4,selection_seed_count=3,heldout_seed_count=3)))
            r=run_p8(p0,root/"p8",seed=649,generations=2,offspring=3,
                     train_episodes=4,selection_episodes=3,holdout_episodes=3,
                     horizon=31,portfolio=2,encoder_steps=40,probe_count=4,probe_worlds=3)
            self.assertEqual(r["candidate_count"],6*5*2*3)
            self.assertEqual(r["adam_encoder_updates"],40)
            self.assertGreater(r["adam_policy_updates"],0)
            self.assertFalse(r["game_code_modified"])
            self.assertFalse(r["language_model"])
            self.assertFalse(r["self_play"])
            self.assertTrue(all(x["initial_sha256"] != x["final_sha256"] for x in r["records"]))
            self.assertEqual(set(r["arms"]),set(ARMS))
            traces=list((root/"p8"/"gallery").glob("*-trace.json"))
            self.assertEqual(len(traces),6*5*2)
            for path in traces:
                t=json.loads(path.read_text())
                self.assertFalse(set(map(tuple,t["painted"])) & set(map(tuple,t["obstacles"])))
                self.assertFalse(set(map(tuple,t["painted"])) & set(map(tuple,t["protected"])))
                self.assertTrue(all(h["action"] in ("N","E","S","W","RETRACE","STOP")
                                    for h in t["trajectory"]))
                self.assertFalse(t["native"])

if __name__=="__main__":
    unittest.main()
