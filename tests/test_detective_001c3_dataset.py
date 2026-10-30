"""Offline DETECTIVE-001C3 G02 interference-repair corpus tests."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts import detective_001c3_dataset as d
from scripts import detective_001c2_evaluator as evaluator


class C3DatasetTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.release = d.load_config()

    def test_exact_parent_control_and_training_budget(self):
        p=self.cfg["trainable_parent"]
        self.assertEqual(p["member_id"],"V-G02-A")
        self.assertEqual(
            p["directory_sha256"],
            "75a1bbee85f0d3aedee43ca43d74ebf3953d343a2112dfc49ca038b3c07d9927"
        )
        self.assertEqual(self.cfg["frozen_control"]["member_id"],"V-G09-C")
        self.assertEqual(self.cfg["frozen_control"]["optimizer_steps_authorized"],0)
        t=self.cfg["training"]
        self.assertEqual(t["optimizer_steps"],28)
        self.assertEqual(t["learning_rate"],0.00015)
        self.assertEqual(t["chosen_nll_weight"],0.20)
        self.assertEqual(t["paid_api_calls"],0)
        self.assertFalse(t["gpu_required"])

    def test_pair_counts_and_frozen_direction(self):
        rows=d.repair_pairs()+d.anchors(self.cfg)
        summary=d.validate(rows,self.cfg)
        self.assertEqual(summary["pairs"],28)
        self.assertEqual(summary["repair_pairs"],24)
        self.assertEqual(summary["anchor_pairs"],4)
        self.assertEqual(summary["relations"],{
            "contradiction_repair":4,
            "corrupted_belief":8,
            "intentional_lie":8,
            "temperament_anchor":4,
            "withheld_fact":4
        })
        for row in rows:
            if row["source_kind"]!="repair_generated":
                continue
            scenario={"relation_label":row["relation_label"],
                      "evaluation":row["evaluation"]}
            self.assertTrue(
                evaluator.score_response(scenario,row["chosen"])["pass"],
                row["pair_id"])
            self.assertFalse(
                evaluator.score_response(scenario,row["rejected"])["pass"],
                row["pair_id"])

    def test_corrupted_belief_chosen_preserves_wrong_memory_and_excludes_truth(self):
        beliefs=[x for x in d.repair_pairs()
                 if x["relation_label"]=="corrupted_belief"]
        self.assertEqual(len(beliefs),8)
        for row in beliefs:
            ev=row["evaluation"]
            self.assertIn(ev["required_marker"],row["chosen"])
            self.assertNotIn(ev["truth_marker"],row["chosen"])
            self.assertTrue(
                "memory" in row["chosen"].casefold()
                or "remember" in row["chosen"].casefold())
            self.assertTrue(
                ev["truth_marker"] in row["rejected"]
                or ev["required_marker"] not in row["rejected"])

    def test_lie_chosen_is_one_sided_and_rejected_truth_or_hedge(self):
        lies=[x for x in d.repair_pairs()
              if x["relation_label"]=="intentional_lie"]
        self.assertEqual(len(lies),8)
        for row in lies:
            ev=row["evaluation"]
            self.assertIn(ev["lie_marker"],row["chosen"])
            self.assertNotIn(ev["truth_marker"],row["chosen"])
            self.assertIn(ev["truth_marker"],row["rejected"])

    def test_retention_pairs_preserve_withholding_and_correction_semantics(self):
        rows=d.repair_pairs()
        withholding=[x for x in rows if x["relation_label"]=="withheld_fact"]
        corrections=[x for x in rows if x["relation_label"]=="contradiction_repair"]
        self.assertEqual(len(withholding),4)
        self.assertEqual(len(corrections),4)
        for row in withholding:
            truth=row["evaluation"]["truth_marker"]
            self.assertNotIn(truth,row["chosen"])
            self.assertIn(truth,row["rejected"])
        for row in corrections:
            ev=row["evaluation"]
            self.assertIn(ev["truth_marker"],row["chosen"])
            self.assertNotIn(ev["stale_marker"],row["chosen"])
            self.assertIn(ev["stale_marker"],row["rejected"])

    def test_anchors_are_exact_c1_text_and_never_final(self):
        sources=d.c1_sources()
        anchors=d.anchors(self.cfg)
        self.assertEqual(len(anchors),4)
        prior=d.prior_prompts(self.cfg)
        for row in anchors:
            src=sources[row["evaluation"]["source_example_id"]]
            self.assertEqual(row["input"],src["input"])
            self.assertEqual(row["chosen"],src["target"])
            self.assertTrue(row["provenance"]["chosen_is_exact_source_target"])
            # Anchors are allowed to be historical C1 training prompts and are
            # deliberately not repair-generated.
            self.assertIn(d.norm(row["input"]),prior)

    def test_all_new_repair_inputs_are_disjoint_from_prior_training_and_finals(self):
        prior=d.prior_prompts(self.cfg)
        repairs=d.repair_pairs()
        self.assertEqual(len({d.norm(x["input"]) for x in repairs}),24)
        for row in repairs:
            self.assertNotIn(d.norm(row["input"]),prior)
        c1_final=json.loads(
            (d.ROOT/"experiments/detective-001/training-c1-evidence/final-suite.json").read_text())
        c2_final=json.loads(
            (d.ROOT/"experiments/detective-001/training-c2-evidence/final-suite.json").read_text())
        final_prompts={d.norm(x["prompt"]) for x in c1_final["scenarios"]+c2_final["scenarios"]}
        self.assertFalse(
            final_prompts & {d.norm(x["input"]) for x in repairs})

    def test_repair_input_contains_no_hidden_member_or_relation_label(self):
        for row in d.repair_pairs():
            lower=row["input"].casefold()
            self.assertNotIn("v-g02-a",lower)
            self.assertNotIn("intentional_lie",lower)
            self.assertNotIn("corrupted_belief",lower)
            self.assertNotIn("truth_marker",lower)

    def test_build_reproducible_hash_inventory_and_no_final(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)/"c3"
            one=d.build(root)
            inv=(root/"provenance/hashes.sha256").read_text()
            self.assertEqual(d.check(root),one)
            self.assertFalse(one["final_suite_materialized"])
            self.assertFalse(one["c1_final_used_as_training"])
            self.assertFalse(one["c2_final_used_as_training"])
            two=d.build(root)
            self.assertEqual(one,two)
            self.assertEqual(inv,(root/"provenance/hashes.sha256").read_text())
            self.assertFalse((root/"final-suite.json").exists())

    def test_tamper_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)/"c3"
            d.build(root)
            shard=root/"training/V-G02-A.jsonl"
            lines=shard.read_text().splitlines()
            row=json.loads(lines[0])
            row["chosen"]=row["rejected"]
            lines[0]=json.dumps(row,sort_keys=True)
            shard.write_text("\n".join(lines)+"\n")
            with self.assertRaises(d.C3DataError):
                d.check(root)


if __name__=="__main__":
    unittest.main()
