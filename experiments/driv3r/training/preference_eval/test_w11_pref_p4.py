"""P4 adversarial source and real CPU learning qualification; no game/native writes."""
import os
import tempfile
import unittest
from pathlib import Path
import w11_pref_p4 as p

class W11P4Tests(unittest.TestCase):
    def test_source_custody_and_unique_models(self):
        rows=p.records()
        self.assertEqual(len(rows),45)
        self.assertEqual(len({(r["family"],r["seed"],r["mode"]) for r in rows}),45)
        self.assertEqual(set(r["seed"] for r in rows),set(p.SEEDS))
        self.assertEqual(len(p.DATA.encode()),1844)
    def test_tampered_frozen_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"byte identity"):
            p.records(p.DATA.replace("A 1901 g 32", "A 1901 g 31"))
        with self.assertRaisesRegex(ValueError,"byte identity"):
            p.records(p.DATA.replace("P 1931 9", "G 1931 9"))
    def test_future_horizon_pairing_and_abstention(self):
        rows=p.records()
        for r in rows:
            self.assertLessEqual(r["future_new"],r["turns"]-32)
            self.assertLessEqual(r["future_branch"]+r["future_retrace"],r["turns"]-32)
        pairs_b,ties_b=p.pairs(rows,"future_branch")
        pairs_n,ties_n=p.pairs(rows,"future_new")
        self.assertEqual((len(pairs_b),ties_b),(79,11))
        self.assertEqual((len(pairs_n),ties_n),(85,5))
        self.assertTrue(all(a["family"]==b["family"] and a["seed"]==b["seed"] for _,a,b,_ in pairs_n))
    def test_real_adam_pairs_and_seed_holdouts(self):
        pairs,_=p.pairs(p.records(),"future_branch")
        a=p.fit(pairs,steps=40);b=p.fit(pairs,steps=40)
        self.assertEqual(a,b)
        self.assertEqual(a["Adam_updates"],40)
        self.assertEqual(a["balanced_training_pairs"],2*len(pairs))
        self.assertTrue(any(abs(w)>1e-4 for w in a["weights"]))
        self.assertLess(a["loss"][-1][1],a["loss"][0][1])
        for seed in p.SEEDS:
            tr=[z for z in pairs if z[0][1]!=seed]
            te=[z for z in pairs if z[0][1]==seed]
            self.assertTrue({z[0] for z in tr}.isdisjoint({z[0] for z in te}))
    def test_immutable_artifact_report_and_negative_controls(self):
        root=os.environ.get("RUNNER_TEMP")
        if root:
            dest=Path(root)/"pref-p1"/"p4"
            report=p.run(dest)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                report=p.run(tmp)
        self.assertEqual(report["new_model_forwards"],0)
        self.assertEqual(report["native_Godot_receipts"],0)
        self.assertEqual(report["human_preference_ratings"],0)
        self.assertEqual(report["original_W11_episode_count_including_controls"],72)
        self.assertEqual(report["matched_greedy_vs_sampled"]["T07_more_future_branches"],9)
        self.assertEqual(report["matched_greedy_vs_sampled"]["T07_more_future_new"],4)
        self.assertEqual(report["matched_greedy_vs_sampled"]["T07_less_future_new"],5)
        for axis in p.AXES:
            self.assertEqual(len(report["axes"][axis]["heldout_seed_folds"]),3)
            self.assertEqual(report["axes"][axis]["actual_cpu_Adam_updates_including_three_folds"],960)

if __name__=="__main__": unittest.main()
