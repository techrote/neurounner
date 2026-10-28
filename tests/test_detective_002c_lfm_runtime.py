"""Offline DETECTIVE-002C contract/custody tests; no model/runtime imports."""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from scripts import detective_002c_lfm_runtime as r


class LfmV5ContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.candidate,self.battery=r.load_contract()

    def test_exact_source_battery_and_historical_blobs_are_pinned(self):
        auth=self.cfg["source_authority"]
        self.assertEqual(
            r.git_blob_sha1(r.SOURCE),auth["stage_a_git_blob_sha1"])
        self.assertEqual(
            r.git_blob_sha1(r.BATTERY),auth["battery_git_blob_sha1"])
        self.assertEqual(
            r.git_blob_sha1(r.HISTORICAL),
            auth["historical_result_git_blob_sha1"])
        self.assertEqual(
            auth["stage_a_merge_commit"],
            "0993f0dfd859f67e88353bde0eeb2f6f377ccfac")

    def test_exact_lfm_source_identity_and_license_gate(self):
        m=self.cfg["model"]
        self.assertEqual(m["repo"],"LiquidAI/LFM2.5-350M")
        self.assertEqual(
            m["revision"],"9e6c6ccf47cd318696e137d381a7ded8fe4df09f")
        self.assertEqual(
            m["weight_sha256"],
            "1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e")
        self.assertEqual(m["weight_bytes"],708_984_464)
        self.assertEqual(m["safetensors_parameters"],354_483_968)
        self.assertEqual(m["license"],"lfm1.0")
        self.assertTrue(m["production_license_gate"])
        self.assertEqual(self.candidate["runtime_eligibility"],"stage_b_candidate")

    def test_historical_failure_is_exact_002b_v4576_record(self):
        h=json.loads(r.HISTORICAL.read_text())
        self.assertEqual(h["experiment"],"DETECTIVE-002B")
        self.assertEqual(h["status"],"runtime_fail")
        self.assertEqual(h["runtime_contract"]["transformers"],"4.57.6")
        self.assertEqual(h["runtime_failure"]["type"],"ValueError")
        self.assertIn("Tokenizer class TokenizersBackend",
                      h["runtime_failure"]["message"])

    def test_transformers_v5201_and_wheel_hash_are_frozen(self):
        rt=self.cfg["runtime"]
        self.assertEqual(rt["transformers"],"5.20.1")
        self.assertEqual(
            rt["transformers_wheel_sha256"],
            "5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19")
        self.assertEqual(rt["threads"],4)
        self.assertEqual(rt["optimizer_steps"],0)
        self.assertEqual(rt["paid_api_calls"],0)
        self.assertFalse(rt["gpu_required"])

    def test_tokenizer_stage_precedes_weight_download(self):
        stages=self.cfg["stages"]
        self.assertTrue(stages["tokenizer_only_first"])
        self.assertTrue(stages["download_weight_only_after_tokenizer_pass"])
        self.assertEqual(stages["require_tokenizer_backend"],"tokenizers")
        self.assertEqual(stages["expected_tokenizer_class"],"TokenizersBackend")

    def test_battery_is_exact_002b_semantic_fixture(self):
        b=json.loads(r.BATTERY.read_text())
        self.assertEqual(self.battery,b)
        self.assertEqual([x["id"] for x in b["battery"]],
                         ["B01","B02","B03","B04"])
        self.assertEqual(b["runtime"]["seed"],260213)
        self.assertEqual(self.cfg["runtime"]["seed"],260213)
        self.assertEqual(self.cfg["runtime"]["max_new_tokens_per_prompt"],64)
        self.assertEqual(self.cfg["runtime"]["max_generated_tokens_total"],256)

    def test_decision_boundary_cannot_select_train_clear_license_or_expand(self):
        d=self.cfg["decisions"]
        self.assertEqual(set(d["allowed"]),{
            "runtime_compatibility_pass",
            "tokenizer_incompatible",
            "model_runtime_incompatible",
        })
        self.assertTrue(d["source_integrity_fail_is_hard_error"])
        self.assertFalse(d["population_expansion_admitted"])
        self.assertFalse(d["training_selection_allowed"])
        self.assertFalse(d["production_license_cleared"])
        self.assertFalse(d["relevance_mapping_accessed"])

    def test_source_blob_tamper_is_hard_error(self):
        original=r.read_json
        def changed(path):
            obj=original(path)
            if path==r.SOURCE:
                obj=copy.deepcopy(obj)
                row=next(x for x in obj["candidates"]
                         if x["candidate_id"]=="lfm25-350m")
                row["weight"]["lfs_sha256"]="0"*64
            return obj
        # read_json mocking alone changes parsed content but the frozen blob
        # guard also reads real bytes; it should still reject source-record drift.
        with patch.object(r,"read_json",side_effect=changed):
            with self.assertRaises(r.SourceIntegrityError):
                r.load_contract()

    def test_archive_accepts_only_predeclared_non_admission_outcomes(self):
        base={
            "schema":"detective-002c-lfm-runtime-result-v1",
            "experiment":"DETECTIVE-002C","issue":133,
            "config_sha256":r.file_sha(r.CONFIG),
            "source_evidence_sha256":r.file_sha(r.SOURCE),
            "battery_sha256":r.file_sha(r.BATTERY),
            "historical_result_sha256":r.file_sha(r.HISTORICAL),
            "model":self.cfg["model"],"runtime_contract":self.cfg["runtime"],
            "historical_control":self.cfg["historical_control"],
            "relevance_mapping_accessed":False,"optimizer_steps":0,
            "paid_api_calls":0,"production_license_cleared":False,
            "training_selection_allowed":False,
            "population_expansion_admitted":False,
            "tokenizer_stage":{"status":"tokenizer_incompatible"},
            "model_stage":None,
            "status":"tokenizer_incompatible",
        }
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            result=root/"result.json";r.write_json(result,base)
            archive=root/"archive"
            m=r.archive(result,archive,"a"*40,"123",
                        "https://github.com/techrote/neurounner/actions/runs/123")
            self.assertEqual(m["status"],"tokenizer_incompatible")
            self.assertEqual(m["optimizer_steps"],0)
            self.assertFalse(m["production_license_cleared"])
            self.assertFalse(m["population_expansion_admitted"])

            bad=copy.deepcopy(base);bad["status"]="training_selected"
            r.write_json(result,bad)
            with self.assertRaises(r.ContractError):
                r.archive(result,root/"bad","a"*40,"123","x")


if __name__=="__main__":
    unittest.main()
