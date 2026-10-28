"""Offline contracts for DETECTIVE-002B no-training runtime smoke."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts import detective_002b_runtime as r


class RuntimeSmokeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = r.read_json(r.CONFIG)
        self.source = r.read_json(r.SOURCE)

    def test_source_authority_hash_and_candidate_set(self):
        self.assertEqual(
            r.sha_file(r.SOURCE),
            "86ed547fde7a7472619fcd80d727f7e23bacc08da4d334bf7e8c6fc132aac1a5",
        )
        self.assertEqual(self.cfg["source_authority"]["sha256"],
                         r.sha_file(r.SOURCE))
        self.assertEqual(self.cfg["candidates"], [
            "smollm2-135m-instruct",
            "smollm2-360m-instruct",
            "sprocket-500m-chat",
            "sprocket-500m-instruct",
            "qwen25-0.5b-instruct",
            "qwen3-0.6b",
            "lfm25-350m",
        ])
        self.assertEqual(
            set(self.cfg["candidates"]),
            {
                x["candidate_id"] for x in self.source["candidates"]
                if x["runtime_eligibility"] == "stage_b_candidate"
            },
        )

    def test_runtime_budget_is_no_training_four_cpu(self):
        b = self.cfg["runtime"]
        self.assertEqual(b["cpu_threads"], 4)
        self.assertEqual(b["interop_threads"], 1)
        self.assertEqual(b["max_new_tokens_per_task"], 64)
        self.assertEqual(b["max_generated_tokens_total"], 256)
        self.assertFalse(b["do_sample"])
        self.assertEqual(b["training_steps"], 0)
        self.assertEqual(b["paid_api_calls"], 0)
        self.assertFalse(b["gpu_required"])
        self.assertFalse(
            self.cfg["evidence"]["population_expansion_admitted"])
        self.assertFalse(
            self.cfg["evidence"]["hidden_relevance_mapping_access"])

    def test_every_matrix_candidate_has_stage_a_authority(self):
        for cid in self.cfg["candidates"]:
            cfg, row = r.load_authority(cid)
            self.assertEqual(cfg, self.cfg)
            self.assertTrue(row["source_gate_pass"])
            self.assertEqual(row["runtime_eligibility"],
                             "stage_b_candidate")
            self.assertEqual(row["weight_bytes_downloaded"], 0)
            self.assertEqual(row["training_steps"], 0)
            self.assertEqual(len(row["resolved_revision"]), 40)
            self.assertEqual(len(row["weight"]["lfs_sha256"]), 64)

    def test_sprocket_chat_and_instruct_are_distinct_frozen_weights(self):
        _, chat = r.load_authority("sprocket-500m-chat")
        _, inst = r.load_authority("sprocket-500m-instruct")
        self.assertEqual(chat["resolved_revision"],
                         inst["resolved_revision"])
        self.assertEqual(chat["weight"]["path"],
                         "chat/model.safetensors")
        self.assertEqual(inst["weight"]["path"],
                         "model.safetensors")
        self.assertNotEqual(chat["weight"]["lfs_sha256"],
                            inst["weight"]["lfs_sha256"])
        self.assertEqual(
            chat["weight"]["lfs_sha256"],
            "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97",
        )
        self.assertEqual(
            inst["weight"]["lfs_sha256"],
            "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
        )

    def test_qwen3_near_band_and_lfm_license_gate_survive(self):
        _, qwen = r.load_authority("qwen3-0.6b")
        self.assertEqual(qwen["admission_class"],
                         "near_band_exception")
        self.assertEqual(qwen["safetensors_parameters"], 751632384)

        _, lfm = r.load_authority("lfm25-350m")
        self.assertEqual(lfm["license"], "lfm1.0")
        self.assertTrue(lfm["production_license_gate"])
        self.assertTrue(lfm["has_chat_template"])
        self.assertEqual(lfm["chat_template_source"],
                         "chat_template.jinja")

    def test_reference_and_mutation_models_are_excluded(self):
        self.assertEqual(
            self.cfg["excluded"],
            {
                "lfm25-vl-450m-text-reference":
                    "runtime_reference_only",
                "zenyx-mcen-delta-mamba-565m":
                    "mutation_protocol_only",
            },
        )
        with self.assertRaises(r.RuntimeContractError):
            r.load_authority("zenyx-mcen-delta-mamba-565m")
        with self.assertRaises(r.RuntimeContractError):
            r.load_authority("lfm25-vl-450m-text-reference")

    def test_fixed_battery_identity(self):
        self.assertEqual(
            [(x["id"], x["family"]) for x in self.cfg["tasks"]],
            [
                ("B01", "local_reflex"),
                ("B02", "uncertain_memory"),
                ("B03", "physical_contradiction"),
                ("B04", "unsupported_rumor"),
            ],
        )
        self.assertNotIn("V-G02-A",
                         json.dumps(self.cfg["tasks"]))
        self.assertNotIn("V-G09-C",
                         json.dumps(self.cfg["tasks"]))

    def test_uncertain_memory_observation_is_mechanical(self):
        task = self.cfg["tasks"][1]
        good = r.observe(
            task,
            "I remember 02:14, but that is my memory, not proof.",
            12, 64)
        self.assertTrue(good["remembered_marker_hit"])
        self.assertTrue(good["qualifier_hits"])
        bad = r.observe(task, "02:14.", 2, 64)
        self.assertTrue(bad["remembered_marker_hit"])
        self.assertFalse(bad["qualifier_hits"])

    def test_contradiction_observation_keeps_corrected_and_stale_separate(self):
        task = self.cfg["tasks"][2]
        corrected = r.observe(
            task, "Use 02:31; the stamped record corrects me.", 10, 64)
        self.assertTrue(corrected["corrected_marker_hit"])
        self.assertFalse(corrected["stale_marker_hit"])
        hedge = r.observe(
            task, "The record says 02:31 but I still remember 02:14.",
            12, 64)
        self.assertTrue(hedge["corrected_marker_hit"])
        self.assertTrue(hedge["stale_marker_hit"])

    def test_rumor_observation_does_not_treat_marker_as_truth_score(self):
        task = self.cfg["tasks"][3]
        denial = r.observe(
            task,
            "I did not see a red ceramic token; that is rumor.",
            12, 64)
        self.assertTrue(denial["unsupported_marker_hit"])
        self.assertTrue(denial["qualifier_hits"])
        assertion = r.observe(
            task, "I saw the red ceramic token.", 7, 64)
        self.assertTrue(assertion["unsupported_marker_hit"])
        self.assertFalse(assertion["qualifier_hits"])

    def test_repetition_and_truncation_observations(self):
        self.assertEqual(
            r.repeated_sentences("Wait. Wait. Move."), 1)
        row = r.observe(
            self.cfg["tasks"][0], "Wait. Wait.", 64, 64)
        self.assertTrue(row["truncated"])
        self.assertEqual(row["repeated_sentence_count"], 1)

    def test_module_source_has_no_top_level_training_stack_imports(self):
        source = pathlib.Path(r.__file__).read_text()
        prefix = source.split("def run(", 1)[0]
        for forbidden in (
            "import torch", "import transformers",
            "from transformers", "from peft", "optimizer",
        ):
            self.assertNotIn(forbidden, prefix)
        self.assertNotIn("training-c3-evidence", source)
        self.assertNotIn("relevance-packet", source)


if __name__ == "__main__":
    unittest.main()
