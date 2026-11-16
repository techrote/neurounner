"""FT-003 custody, legal budgets and matched-world negative controls."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import training_dose as d


class TestTrainingDose(unittest.TestCase):
    def setUp(self):
        self.req = json.loads((Path(__file__).parent / "request_dose.json").read_text())

    def test_exact_source_and_four_distinct_job_leases(self):
        d.validated_request(self.req)
        leases = [f"{d.REQUEST_ID}/{dose}/{seed}"
                  for dose in self.req["train_episodes"] for seed in self.req["seeds"]]
        self.assertEqual(len(leases), 4)
        self.assertEqual(len(set(leases)), 4)

    def test_no_extra_optimizers_or_horizon_expansion(self):
        for field, value in (("seeds", [9401, 9401]), ("train_episodes", [12, 1000]),
                             ("max_parallel", 4), ("upstream_commit", "0"*40)):
            r = copy.deepcopy(self.req)
            r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.validated_request(r)

    def test_provenance_nonspoofed_split_hash(self):
        x = {"seed": 9401, "train_seed_count": 12, "selection_seed_count": 4,
             "heldout_seed_count": 12,
             "seed_hashes": {k:d.expected_seed_hash(9401,k,n)
                            for k,n in (("train",12),("selection",4),("heldout",12))}}
        self.assertIs(d.validate_split(x,9401,12,self.req),x)
        x["seed_hashes"]["heldout"] = "0"*64
        with self.assertRaises(AssertionError):
            d.validate_split(x,9401,12,self.req)

    def test_upstream_heldout_uses_test_seed_namespace(self):
        self.assertEqual(d.expected_seed_hash(9401,"heldout",12),
                         "72f4d195538cf58ebfbd8c865b1d35340efd4413fb0c71e006c7ef1d140a0dec")

    def test_matched_holdout_controls_must_agree(self):
        def result(dose):
            scores = {k: {"mean_reward": 1.0, "mean_painted": 3,
                          "mean_retrace": 1, "early_stop_rate": 0.0} for k in d.NICHES}
            return {"seed":9401,"train_seed_count":dose,
                    "selection_seed_count":4,"heldout_seed_count":12,
                    "seed_hashes":{"selection":"sel","heldout":"test"},
                    "heldout_champions":scores,
                    "heldout_controls":{k:{"nonstop_random":{"mean_reward":0.5}} for k in d.NICHES}}
        short,long = result(12),result(48)
        self.assertEqual(d.compare_pair(9401,short,long)["ties"],6)
        long["heldout_controls"]["navigator"]["nonstop_random"]["mean_reward"] = -1
        with self.assertRaises(AssertionError):
            d.compare_pair(9401,short,long)

    def test_real_regression_is_not_rejected(self):
        def result(reward):
            return {"seed":9403,"selection_seed_count":4,"heldout_seed_count":12,
                    "seed_hashes":{"selection":"sel","heldout":"test"},
                    "heldout_champions":{n:{"mean_reward":reward,"mean_painted":2,
                                           "mean_retrace":0,"early_stop_rate":0.25}
                                         for n in d.NICHES},
                    "heldout_controls":{n:{"nonstop_random":{"mean_reward":0.0}}
                                         for n in d.NICHES}}
        rows = d.compare_pair(9403,result(2.0),result(-3.0))
        self.assertEqual(rows["regressed"],6)
        self.assertEqual(rows["improved"],0)
        self.assertTrue(all(x["reward_delta"] == -5 for x in rows["niches"]))


if __name__ == "__main__":
    unittest.main()
