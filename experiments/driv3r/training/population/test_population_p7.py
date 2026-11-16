"""P7 tests: no holdout contamination, fitted encoder, ancestry, trained weights."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from population_lab import NICHES, World, Policy, stable_seed
from population_p7 import (ARMS, archive_add, choose_parent, choose_portfolio,
                           fingerprint, fit_encoder, learned_cell, run_p7)
from population_p6 import split_seeds


class P7Tests(unittest.TestCase):
    def test_fingerprint_normalised_and_spatial(self):
        w = World.generate(33)
        a = fingerprint(w)
        self.assertEqual(a.shape, (31,))
        self.assertAlmostEqual(a[:25].sum(), 1.)
        self.assertEqual(a[25:].sum(), 0.)
        act = next(x for x in w.legal() if x != "STOP")
        w.step(act)
        b = fingerprint(w)
        self.assertAlmostEqual(b[25:].sum(), 1.)
        self.assertNotEqual(a.tolist(), b.tolist())

    def test_train_only_embedding_determinism_and_bins(self):
        parents = {n: Policy(stable_seed(n, 2)) for n in NICHES}
        a = fit_encoder(parents, 638, probe_count=4, probe_worlds=3)
        b = fit_encoder(parents, 638, probe_count=4, probe_worlds=3)
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["samples"], 30)
        self.assertTrue(0 < a["top3_fraction_variance"] <= 1.)
        c, z = learned_cell(np.zeros(31), a)
        self.assertEqual(len(c), 3)
        self.assertTrue(all(0 <= x <= 3 for x in c))
        self.assertTrue(np.isfinite(z).all())
        with self.assertRaises(ValueError):
            learned_cell(np.zeros(14), a)

    def test_archive_and_parent_logic(self):
        def rec(ident, reward, raw, learned):
            return dict(id=ident, selection={"mean_reward":reward}, raw_cell=raw,
                        learned_cell=learned)
        r = [rec("a", 1., [0], [0]), rec("b", 2., [0], [1]), rec("c", 3., [1], [1])]
        ar = archive_add({}, r, "learned_cell")
        self.assertEqual(ar[(1,)]["id"], "c")
        self.assertIsNone(choose_parent("learned_map",0,0,636,"navigator",{},None,[]))
        self.assertEqual(choose_parent("reward_pbt",1,0,636,"navigator",ar,r[1],r)["id"],"b")
        self.assertIn(choose_parent("learned_map",1,0,636,"navigator",ar,r[1],r),r)
        self.assertIn(choose_parent("random_parents",1,0,636,"navigator",ar,r[1],r),r)
        p = choose_portfolio(r, ar, "learned_map",2,99)
        self.assertEqual(len(p), len(set(p)))

    def test_train_selection_encoder_holdout_separation(self):
        s = split_seeds(636, 4, 3, 3, 2)
        p = [stable_seed("p7",636,"encoder_train",i) for i in range(3)]
        flat = sum(s["train"],[]) + s["selection"] + s["holdout"] + p
        self.assertEqual(len(flat),len(set(flat)))

    def test_input_bounds(self):
        with self.assertRaises(ValueError):
            run_p7("invalid", "unused", generations=1)
        with self.assertRaises(ValueError):
            run_p7("invalid", "unused", portfolio=20)

    def test_mini_campaign_provenance_and_legal_actions(self):
        # Fixture sourced from P0-style serialization; no game dependency.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p0=root/"p0"
            (p0/"models").mkdir(parents=True)
            for n in NICHES:
                (p0/"models"/(n+".json")).write_text(json.dumps(
                    Policy(stable_seed(n,619)).serialized()))
            (p0/"results.json").write_text(json.dumps(dict(
                format="population-research-v1",language_model=False,seed=619,
                train_seed_count=4,selection_seed_count=3,heldout_seed_count=3)))
            r = run_p7(p0,root/"p7", seed=649, generations=2, offspring=3,
                       train_episodes=4, selection_episodes=3, holdout_episodes=3,
                       horizon=31, portfolio=2, probe_count=4, probe_worlds=3)
            self.assertEqual(r["candidate_count"],6*4*2*3)
            self.assertGreater(r["adam_updates"],0)
            self.assertEqual(set(r["arms"]),set(ARMS))
            self.assertFalse(r["game_code_modified"])
            self.assertFalse(r["language_model"])
            self.assertEqual(len(r["records"]),144)
            self.assertTrue(all(x["initial_sha256"] != x["final_sha256"] for x in r["records"]))
            self.assertTrue(all(len(x["learned_cell"])==3 for x in r["records"]))
            traces=list((root/"p7"/"gallery").glob("*-trace.json"))
            self.assertEqual(len(traces),6*4*2)
            for p in traces:
                data=json.loads(p.read_text())
                painted={tuple(x) for x in data["painted"]}
                self.assertFalse(painted & {tuple(x) for x in data["protected"]})
                self.assertFalse(painted & {tuple(x) for x in data["obstacles"]})
                self.assertTrue(all(x["action"] in ("N","E","S","W","RETRACE","STOP")
                                    for x in data["trajectory"]))


if __name__ == "__main__":
    unittest.main()