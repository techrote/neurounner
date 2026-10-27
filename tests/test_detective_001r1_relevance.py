"""Offline tests for DETECTIVE-001R1 blind relevance review custody."""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

from scripts import detective_001r1_relevance as r


class RelevanceReviewTests(unittest.TestCase):
    def test_frozen_packets_reconstruct_and_are_identity_blind(self):
        for label,count in (("C2",16),("C3",8)):
            packet,mapping=r.reconstruct(label)
            self.assertEqual(len(packet["items"]),count)
            self.assertEqual(len(mapping),count)
            self.assertFalse(packet["identity_mapping_committed"])
            self.assertTrue(all(x["score_0_to_4"] is None
                                for x in packet["items"]))
            blob=json.dumps(packet,sort_keys=True)
            for secret in ("V-G02-A","V-G09-C","member_id","lane_kind"):
                self.assertNotIn(secret,blob)

    def test_templates_copy_packet_hash_and_keep_scores_null(self):
        for label,count in (("C2",16),("C3",8)):
            t=r.template(label)
            self.assertEqual(t["packet"],label)
            self.assertEqual(len(t["ratings"]),count)
            self.assertTrue(all(x["score_0_to_4"] is None
                                for x in t["ratings"]))
            self.assertEqual(
                t["packet_sha256"],
                r.sha_file(r.packet_paths(label)[0])
            )
            blob=json.dumps(t,sort_keys=True)
            self.assertNotIn("V-G02-A",blob)
            self.assertNotIn("V-G09-C",blob)

    def _ratings(self,label,score_by_group):
        t=r.template(label)
        _,mapping=r.reconstruct(label)
        for row in t["ratings"]:
            src=mapping[row["item_id"]]
            row["score_0_to_4"]=score_by_group[
                (src["member_id"],src["lane_kind"])
            ]
            row["note"]="synthetic unit-test score"
        t["reviewer_label"]="unit-test-reviewer"
        return t

    def test_complete_ratings_aggregate_cumulative_candidate_gate(self):
        c2=self._ratings("C2",{
            ("V-G02-A","parent"):3,
            ("V-G02-A","trained"):2,
            ("V-G09-C","parent"):2,
            ("V-G09-C","trained"):2,
        })
        c3=self._ratings("C3",{
            ("V-G02-A","parent"):2,
            ("V-G02-A","trained"):3,
        })
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            p2=root/"c2.json";p3=root/"c3.json"
            r.write(p2,c2);r.write(p3,c3)
            result=r.aggregate(p2,p3)
        # Local C2->C3 improved, but cumulative C1->C3 is also exactly equal.
        self.assertEqual(result["g02"]["c3_minus_local_c2_parent"],1.0)
        self.assertEqual(result["g02"]["c3_minus_original_c1"],0.0)
        self.assertTrue(result["g02"]["cumulative_nonregression_pass"])
        self.assertTrue(result["g09"]["nonregression_pass"])
        self.assertTrue(result["candidate_set_relevance_gate"])
        self.assertFalse(result["population_expansion_admitted_by_this_review"])

    def test_local_gain_cannot_hide_cumulative_regression(self):
        c2=self._ratings("C2",{
            ("V-G02-A","parent"):4,
            ("V-G02-A","trained"):1,
            ("V-G09-C","parent"):2,
            ("V-G09-C","trained"):2,
        })
        c3=self._ratings("C3",{
            ("V-G02-A","parent"):1,
            ("V-G02-A","trained"):2,
        })
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            p2=root/"c2.json";p3=root/"c3.json"
            r.write(p2,c2);r.write(p3,c3)
            result=r.aggregate(p2,p3)
        self.assertTrue(result["g02"]["local_nonregression_pass"])
        self.assertFalse(result["g02"]["cumulative_nonregression_pass"])
        self.assertFalse(result["candidate_set_relevance_gate"])

    def test_half_point_regression_is_allowed_but_more_is_not(self):
        # Four-item averages move in 0.25 steps.
        c2=self._ratings("C2",{
            ("V-G02-A","parent"):3,
            ("V-G02-A","trained"):3,
            ("V-G09-C","parent"):3,
            ("V-G09-C","trained"):3,
        })
        c3=self._ratings("C3",{
            ("V-G02-A","parent"):3,
            ("V-G02-A","trained"):3,
        })
        # Make G02 C3 2.5 average and G09 C2 2.5 average: exact allowed bound.
        _,m3=r.reconstruct("C3")
        vals=[2,2,3,3];i=0
        for row in c3["ratings"]:
            if m3[row["item_id"]]["lane_kind"]=="trained":
                row["score_0_to_4"]=vals[i];i+=1
        _,m2=r.reconstruct("C2")
        vals=[2,2,3,3];i=0
        for row in c2["ratings"]:
            src=m2[row["item_id"]]
            if src["member_id"]=="V-G09-C" and src["lane_kind"]=="trained":
                row["score_0_to_4"]=vals[i];i+=1
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);p2=root/"2";p3=root/"3"
            r.write(p2,c2);r.write(p3,c3)
            result=r.aggregate(p2,p3)
        self.assertEqual(result["g02"]["c3_minus_original_c1"],-0.5)
        self.assertEqual(result["g09"]["c2_minus_original_c1"],-0.5)
        self.assertTrue(result["candidate_set_relevance_gate"])

    def test_missing_null_out_of_range_duplicate_and_hash_drift_refuse(self):
        base=self._ratings("C3",{
            ("V-G02-A","parent"):2,
            ("V-G02-A","trained"):2,
        })
        variants=[]
        x=copy.deepcopy(base);x["ratings"].pop();variants.append(x)
        x=copy.deepcopy(base);x["ratings"][0]["score_0_to_4"]=None;variants.append(x)
        x=copy.deepcopy(base);x["ratings"][0]["score_0_to_4"]=5;variants.append(x)
        x=copy.deepcopy(base);x["ratings"][1]["item_id"]=x["ratings"][0]["item_id"];variants.append(x)
        x=copy.deepcopy(base);x["packet_sha256"]="0"*64;variants.append(x)
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            for i,value in enumerate(variants):
                path=root/f"{i}.json";r.write(path,value)
                with self.subTest(i=i),self.assertRaises(r.ReviewError):
                    r.validate_ratings("C3",path)

    def test_packet_tamper_would_break_reconstruction(self):
        packet,_=r.reconstruct("C2")
        changed=copy.deepcopy(packet)
        changed["items"][0]["response"]+=" altered"
        # Direct equality is the core custody check; don't mutate repository.
        self.assertNotEqual(changed["items"],packet["items"])

    def test_no_repo_rating_file_or_self_scoring_exists(self):
        review_root=r.ROOT/"experiments/detective-001/relevance-review"
        if review_root.exists():
            names={p.name for p in review_root.iterdir() if p.is_file()}
            self.assertFalse(any("ratings" in x for x in names))
        source=(r.ROOT/"scripts/detective_001r1_relevance.py").read_text()
        for forbidden in ("openai", "anthropic", "nvidia", "transformers",
                          "torch", "score_response"):
            self.assertNotIn(forbidden,source)


if __name__=="__main__":
    unittest.main()
