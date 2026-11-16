#!/usr/bin/env python3
"""Host-selection unit tests; no model download or native engine required."""
import unittest
import worm_train_w7_v1 as w7
import worm_w8_scored_crn_v1 as crn

class TestScoredCRN(unittest.TestCase):
    def setUp(self):
        self.scores={"candidate_logits":{a:float(i) for i,a in enumerate(w7.ACTIONS)}}
    def test_repeat_and_family_independent_random(self):
        self.assertEqual(crn.common_u(3101,2),crn.common_u(3101,2))
        self.assertNotEqual(crn.common_u(3101,2),crn.common_u(3101,3))
    def test_flat_equal_and_exactly_one_choice(self):
        legal=["STRAIGHT","LEFT","RIGHT","BRANCH","STOP"]
        chosen,d=crn.scored_policy(None,legal,"flat",3101,0)
        self.assertIn(chosen,legal)
        self.assertNotIn("STOP",d["sampled_legal"])
        self.assertAlmostEqual(sum(d["sample_probs"].values()),1.0,places=12)
        self.assertEqual(set(d["sample_probs"]),set(legal)-{"STOP"})
        self.assertEqual(set(d["sample_probs"].values()),{0.25})
    def test_learned_legal_and_mixture(self):
        legal=["RIGHT","LEFT","RETRACE","STOP"]
        chosen,d=crn.scored_policy(self.scores,legal,"learned",3101,1)
        self.assertIn(chosen,legal)
        self.assertIn(chosen,d["sampled_legal"])
        self.assertTrue(all(v>0 for v in d["sample_probs"].values()))
        self.assertAlmostEqual(sum(d["sample_probs"].values()),1.0,places=12)
        self.assertGreater(d["sample_probs"]["RETRACE"],d["sample_probs"]["LEFT"])
    def test_forced_stop(self):
        for mode in ("flat","learned"):
            selected,d=crn.scored_policy(self.scores,["STOP"],mode,3101,1)
            self.assertEqual(selected,"STOP")
            self.assertEqual(d["sample_probs"],{"STOP":1.0})
    def test_empty_refuses(self):
        for mode in ("flat","learned"):
            with self.assertRaises(ValueError):
                crn.scored_policy(self.scores,[] ,mode,3101,0)
    def test_does_not_modify_logits(self):
        import copy
        before=copy.deepcopy(self.scores)
        crn.scored_policy(self.scores,["LEFT","RIGHT","BRANCH"],"learned",3109,12)
        self.assertEqual(self.scores,before)
    def test_action_distribution_not_host_random_confounded(self):
        self.assertEqual(crn.common_u(3101,0),crn.common_u(3101,0))
        self.assertNotEqual(crn.common_u(3101,0),crn.common_u(3109,0))

if __name__=="__main__":
    unittest.main()
