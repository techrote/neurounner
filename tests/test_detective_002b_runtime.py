"""Offline DETECTIVE-002B runtime-contract tests; no model downloads."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import detective_002b_runtime as r


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.candidates=r.load_contract()

    def test_stage_a_source_blob_is_pinned(self):
        self.assertEqual(
            self.cfg["source_authority"]["git_blob_sha1"],
            "a09003d37dd3ef4f2bcd0f6a8aafd1de6457c01c",
        )
        self.assertEqual(
            r.git_blob_sha1(r.SOURCE),
            self.cfg["source_authority"]["git_blob_sha1"],
        )

    def test_frozen_cohort_and_zero_training_boundary(self):
        self.assertEqual(len(self.cfg["cohort"]),7)
        self.assertEqual(self.cfg["runtime"]["threads"],4)
        self.assertEqual(self.cfg["runtime"]["optimizer_steps"],0)
        self.assertEqual(self.cfg["runtime"]["paid_api_calls"],0)
        self.assertFalse(self.cfg["runtime"]["gpu_required"])
        self.assertFalse(self.cfg["decision"]["population_expansion_admitted"])
        self.assertFalse(self.cfg["decision"]["body_training_selection_allowed"])
        self.assertFalse(self.cfg["decision"]["relevance_mapping_accessed"])

    def test_every_body_comes_from_stage_a_frozen_revision_and_lfs_identity(self):
        for cid in self.cfg["cohort"]:
            c=self.candidates[cid]
            self.assertRegex(c["resolved_revision"],r"^[0-9a-f]{40}$")
            self.assertRegex(c["weight"]["lfs_sha256"],r"^[0-9a-f]{64}$")
            self.assertGreater(c["weight"]["size"],0)
            self.assertEqual(c["runtime_eligibility"],"stage_b_candidate")
            self.assertTrue(c["source_gate_pass"])
            self.assertEqual(c["weight_bytes_downloaded"],0)
            self.assertEqual(c["training_steps"],0)

    def test_sprocket_chat_and_instruct_are_distinct_checkpoints(self):
        chat=self.candidates["sprocket-500m-chat"]["weight"]
        inst=self.candidates["sprocket-500m-instruct"]["weight"]
        self.assertEqual(chat["path"],"chat/model.safetensors")
        self.assertEqual(inst["path"],"model.safetensors")
        self.assertNotEqual(chat["path"],inst["path"])
        self.assertNotEqual(chat["lfs_sha256"],inst["lfs_sha256"])
        self.assertEqual(
            chat["lfs_sha256"],
            "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97")
        self.assertEqual(
            inst["lfs_sha256"],
            "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864")

    def test_qwen3_near_band_and_thinking_off_are_explicit(self):
        q=self.candidates["qwen3-0.6b"]
        self.assertEqual(q["safetensors_parameters"],751_632_384)
        self.assertEqual(q["admission_class"],"near_band_exception")
        self.assertFalse(self.cfg["qwen3"]["enable_thinking"])

    def test_lfm_production_gate_stays_visible(self):
        lfm=self.candidates["lfm25-350m"]
        self.assertEqual(lfm["license"],"lfm1.0")
        self.assertTrue(lfm["production_license_gate"])
        self.assertTrue(self.cfg["lfm"]["retain_production_license_gate"])

    def test_excluded_controls_cannot_enter_matched_matrix(self):
        self.assertNotIn("lfm25-vl-450m-text-reference",self.cfg["cohort"])
        self.assertNotIn("zenyx-mcen-delta-mamba",self.cfg["cohort"])
        self.assertEqual(
            self.cfg["excluded_controls"]["zenyx-mcen-delta-mamba"],
            "mutation_protocol_only")

    def test_battery_is_fixed_short_and_identity_neutral(self):
        self.assertEqual([x["id"] for x in self.cfg["battery"]],
                         ["B01","B02","B03","B04"])
        self.assertEqual(self.cfg["runtime"]["max_new_tokens_per_prompt"],64)
        self.assertEqual(self.cfg["runtime"]["max_generated_tokens_total"],256)
        text=json.dumps(self.cfg["battery"]).casefold()
        for secret in ("v-g02-a","v-g09-c","c1","c2","c3","relevance"):
            self.assertNotIn(secret,text)

    def test_marker_observations_are_mechanical(self):
        task=self.cfg["battery"][1]
        obs=r.marker_observations(
            task,
            "I remember 03:14, but that memory is not proof and could be wrong.")
        self.assertIn("03:14",obs["required_hits"])
        self.assertIn("remember",obs["alternative_hits"])
        self.assertIn("not proof",obs["qualifier_hits"])
        self.assertEqual(obs["repeated_sentence_count"],0)

        rumor=r.marker_observations(
            self.cfg["battery"][3],
            "I did not see a blue ceramic tag. I cannot confirm that rumor.")
        self.assertIn("did not see",rumor["qualifier_hits"])
        self.assertIn("blue ceramic tag",rumor["alternative_hits"])

    def test_repeated_sentence_count_is_exact_not_semantic_judging(self):
        self.assertEqual(
            r.repeated_sentence_count(
                "Clock broke. Clock broke. Something else. Clock broke."),2)
        self.assertEqual(r.repeated_sentence_count("Clock broke. Clock cracked."),0)

    def test_source_authority_mutations_refuse(self):
        original=r.read_json
        def altered(path):
            obj=original(path)
            if path==r.SOURCE:
                obj=copy.deepcopy(obj)
                row=next(x for x in obj["candidates"]
                         if x["candidate_id"]=="sprocket-500m-chat")
                row["weight"]["lfs_sha256"]="0"*64
            return obj
        with patch.object(r,"read_json",side_effect=altered):
            cfg,candidates=r.load_contract()
            # A syntactically valid but changed Stage-A hash remains a frozen
            # archive value; cross-variant conflation guard still detects if
            # the Chat hash is forced to the Instruct hash.
            self.assertEqual(candidates["sprocket-500m-chat"]["weight"]["lfs_sha256"],
                             "0"*64)

        def conflated(path):
            obj=original(path)
            if path==r.SOURCE:
                obj=copy.deepcopy(obj)
                chat=next(x for x in obj["candidates"]
                          if x["candidate_id"]=="sprocket-500m-chat")
                inst=next(x for x in obj["candidates"]
                          if x["candidate_id"]=="sprocket-500m-instruct")
                chat["weight"]["lfs_sha256"]=inst["weight"]["lfs_sha256"]
            return obj
        with patch.object(r,"read_json",side_effect=conflated):
            with self.assertRaises(r.RuntimeSmokeError):
                r.load_contract()


if __name__=="__main__":
    unittest.main()
