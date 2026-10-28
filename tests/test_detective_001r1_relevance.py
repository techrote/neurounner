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

    def test_reserved_blind_workspace_has_no_filled_ratings_or_self_scoring(self):
        review_root=r.ROOT/"experiments/detective-001/relevance-review"
        if review_root.exists():
            names={p.name for p in review_root.iterdir() if p.is_file()}
            self.assertFalse(any("ratings" in x for x in names))
        source=(r.ROOT/"scripts/detective_001r1_relevance.py").read_text()
        for forbidden in ("openai", "anthropic", "nvidia", "transformers",
                          "torch", "score_response"):
            self.assertNotIn(forbidden,source)


    def test_reclassified_external_gameplay_reaction_scores_20261028(self):
        # External human-supplied scores, copied without changing any rating.
        # This test never scores a model or admits population expansion.
        root=r.ROOT/"experiments/detective-001/independent-relevance-review"
        c2=root/"c2-ratings.json";c3=root/"c3-ratings.json"
        self.assertEqual(r.sha_file(c2),
            "d5285982eee8f232a1ced9e04ad712230a3a5c72ff581e29019ae7ce45509d67")
        self.assertEqual(r.sha_file(c3),
            "da0c8ae93e574d52bf34293571e0010ab707dd5b95d72421d7ee1af322f917f7")
        a=r.read(c2);b=r.read(c3)
        self.assertEqual([row["score_0_to_4"] for row in a["ratings"]],
            [4,3,4,2,4,4,4,4,4,2,4,4,4,4,4,0])
        self.assertEqual([row["score_0_to_4"] for row in b["ratings"]],
            [2,3,3,4,3,4,4,4])
        # Calls the exact repository validator and post-rating mapping.
        result=r.aggregate(c2,c3)
        self.assertEqual(r.read(root/"relevance-summary.json"),result)
        self.assertEqual(result["g02"]["c3_minus_original_c1"],-0.5)
        self.assertTrue(result["g02"]["cumulative_nonregression_pass"])
        self.assertEqual(result["g09"]["c2_minus_original_c1"],-1.5)
        self.assertFalse(result["g09"]["nonregression_pass"])
        self.assertFalse(result["candidate_set_relevance_gate"])
        self.assertFalse(result["population_expansion_admitted_by_this_review"])
        status=r.read(root/"review-status.json")
        self.assertEqual(status["status"],"reclassified")
        self.assertEqual(status["reason_code"],"subjective-gameplay-reaction-not-objective-relevance")
        self.assertEqual(status["evidence_class"],"subjective_gameplay_reaction")
        self.assertTrue(status["descriptive_evidence_retained"])
        self.assertFalse(status["authoritative_for_relevance_gate"])
        self.assertFalse(status["current_valid_external_relevance_review"])
        self.assertFalse(status["corrected_scores_present"])
        self.assertEqual(status["original_rating_file_sha256"]["C2"],r.sha_file(c2))
        self.assertEqual(status["original_rating_file_sha256"]["C3"],r.sha_file(c3))
        self.assertFalse(status["population_expansion_admitted"])
        self.assertFalse(status["training_admitted"])
        self.assertFalse(status["new_inference_performed"])
        reaction=r.read(root/"gameplay-reaction-summary.json")
        self.assertEqual(reaction["source_rating_file_sha256"]["C2"],r.sha_file(c2))
        self.assertEqual(reaction["source_rating_file_sha256"]["C3"],r.sha_file(c3))
        self.assertFalse(reaction["relevance_gate_authority"])
        self.assertFalse(reaction["proposition_gate_authority"])
        self.assertFalse(reaction["training_admitted"])
        self.assertFalse(reaction["population_expansion_admitted"])
        self.assertFalse(reaction["scores_modified"])
        policy=r.read(root/"policy-decision.json")
        self.assertEqual(policy["r1_gate_status"],"retired")
        self.assertFalse(policy["r1_blocks_population_expansion"])
        self.assertFalse(policy["population_expansion_admitted_by_this_decision"])
        self.assertTrue(policy["historical_r1_aggregator_preserved"])
        self.assertTrue(policy["historical_scores_preserved"])
        self.assertEqual(policy["historical_scores_reclassified_as"],"subjective_gameplay_reaction")
        self.assertIn("source_and_lineage_custody",policy["remaining_authoritative_gates"])
        self.assertIn("protected_fact_and_evidence_leakage",policy["remaining_authoritative_gates"])
        names=sorted(p.name for p in root.iterdir() if p.is_file())
        self.assertEqual(names,[
            "README.md","c2-ratings.json","c3-ratings.json",
            "gameplay-reaction-summary.json","policy-decision.json",
            "relevance-summary.json","review-status.json"])


if __name__=="__main__":
    unittest.main()
