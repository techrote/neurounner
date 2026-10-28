"""Offline DETECTIVE-002A compact NPC source-qualification tests."""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from scripts import detective_002a_sources as s


class SourceQualificationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = s.load_config()

    def fake_payload(self, meta, *, sha=None, parameters=None,
                     weight_sha=None, license_id=None):
        resolved = sha or (
            meta["source_ref"] if meta["source_ref"] != "main"
            else hashlib.sha1(meta["repo"].encode()).hexdigest()
        )
        if parameters is None:
            if meta["parameter_count_policy"] == "expected_exact":
                parameters = meta["expected_api_parameters"]
            elif meta["parameter_count_policy"] == "api_required":
                parameters = meta["nominal_parameters"]
        siblings = [
            {"rfilename": "config.json", "blobId": "a" * 40, "size": 321},
            {"rfilename": "tokenizer_config.json",
             "blobId": "b" * 40, "size": 512},
        ]
        if meta.get("weight_path"):
            sha256 = (
                weight_sha or meta.get("expected_weight_sha256")
                or hashlib.sha256(meta["weight_path"].encode()).hexdigest()
            )
            siblings.append({
                "rfilename": meta["weight_path"],
                "blobId": "c" * 40,
                "size": 123456,
                "lfs": {
                    "sha256": sha256,
                    "size": 123456,
                    "pointerSize": 130,
                },
            })
        payload = {
            "id": meta["repo"],
            "sha": resolved,
            "private": False,
            "gated": False,
            "disabled": False,
            "lastModified": "2026-10-27T00:00:00.000Z",
            "library_name": "transformers",
            "pipeline_tag": "text-generation",
            "tags": [],
            "cardData": {
                "license": (
                    license_id if license_id is not None
                    else meta.get("expected_license")
                )
            },
            "config": {
                "architectures": ["FakeForCausalLM"],
                "model_type": "fake",
                "tokenizer_config": {
                    "chat_template": "user {{ content }} tool_call"
                },
            },
            "siblings": siblings,
        }
        if parameters is not None:
            payload["safetensors"] = {
                "parameters": {"BF16": parameters},
                "total": parameters,
            }
        return payload

    def fake_evidence(self):
        rows = []
        for cid, meta in self.cfg["candidates"].items():
            revision = (
                meta["source_ref"] if meta["source_ref"] != "main"
                else hashlib.sha1(cid.encode()).hexdigest()
            )
            if meta["parameter_count_policy"] == "expected_exact":
                params = meta["expected_api_parameters"]
            elif meta["parameter_count_policy"] == "api_required":
                params = meta["nominal_parameters"]
            else:
                params = None
            if meta.get("weight_path"):
                weight_sha = (
                    meta.get("expected_weight_sha256")
                    or hashlib.sha256(cid.encode()).hexdigest()
                )
                weight = {
                    "path": meta["weight_path"],
                    "size": 1000,
                    "blob_id": "d" * 40,
                    "lfs_sha256": weight_sha,
                    "lfs_size": 1000,
                    "pointer_size": 130,
                }
            else:
                weight = None
            if meta["admission_class"] in {
                "direct_candidate", "near_band_exception",
                "existing_control",
            } and meta.get("direct_dialogue_candidate"):
                runtime = "stage_b_candidate"
            elif meta["admission_class"] == "runtime_reference":
                runtime = "stage_b_reference"
            elif meta["admission_class"] == "mutation_only":
                runtime = "mutation_protocol_only"
            else:
                runtime = "blocked"
            rows.append({
                "candidate_id": cid,
                "admission_class": meta["admission_class"],
                "family": meta["family"],
                "repo": meta["repo"],
                "variant": meta["variant"],
                "requested_ref": meta["source_ref"],
                "resolved_revision": revision,
                "last_modified": "2026-10-27T00:00:00.000Z",
                "private": False,
                "gated": False,
                "disabled": False,
                "license": meta.get("expected_license"),
                "production_license_gate":
                    meta.get("production_license_gate", False),
                "library_name": "transformers",
                "pipeline_tag": "text-generation",
                "model_type": "fake",
                "architectures": ["FakeForCausalLM"],
                "safetensors_parameters": params,
                "nominal_parameters": meta["nominal_parameters"],
                "parameter_count_policy": meta["parameter_count_policy"],
                "parameter_gate_pass": True,
                "parameter_gate_reason": "test",
                "weight": weight,
                "expected_weight_sha256":
                    meta.get("expected_weight_sha256"),
                "weight_gate_pass": True,
                "weight_gate_reason": "test",
                "config": {
                    "status": "present", "sha256": "e" * 64,
                    "bytes": 100,
                },
                "tokenizer_config": {
                    "status": "present", "sha256": "f" * 64,
                    "bytes": 100,
                },
                "has_chat_template":
                    meta["admission_class"] != "mutation_only",
                "template_mentions_tools": cid == "qwen3-0.6b",
                "checkpoint_metadata": [],
                "license_gate_pass": True,
                "license_gate_reason": "test",
                "source_gate_pass": True,
                "runtime_eligibility": runtime,
                "requires_custom_runtime":
                    meta.get("requires_custom_runtime", False),
                "instruction_model": meta.get("instruction_model", True),
                "weight_bytes_downloaded": 0,
                "training_steps": 0,
                "paid_api_calls": 0,
            })
        return {
            "schema": "detective-npc-body-source-evidence-v1",
            "experiment": "DETECTIVE-002A",
            "issue": 127,
            "resolver_version": s.RESOLVER_VERSION,
            "config_sha256": s.sha_file(s.CONFIG),
            "weight_bytes_downloaded": 0,
            "training_steps": 0,
            "paid_api_calls": 0,
            "relevance_mapping_accessed": False,
            "population_expansion_admitted": False,
            "candidates": rows,
        }

    def test_cohort_has_explicit_classes_and_zero_training(self):
        self.assertEqual(len(self.cfg["candidates"]), 9)
        self.assertEqual(self.cfg["gates"]["training_steps"], 0)
        self.assertEqual(self.cfg["gates"]["weight_downloads"], 0)
        self.assertTrue(self.cfg["gates"]["no_relevance_mapping_access"])
        self.assertFalse(
            self.cfg["gates"]["population_expansion_admitted"])
        classes = {
            x["admission_class"]
            for x in self.cfg["candidates"].values()
        }
        self.assertEqual(classes, {
            "existing_control", "runtime_reference", "direct_candidate",
            "near_band_exception", "mutation_only",
        })

    def test_sprocket_chat_and_instruct_are_same_revision_distinct_weights(self):
        chat = self.cfg["candidates"]["sprocket-500m-chat"]
        inst = self.cfg["candidates"]["sprocket-500m-instruct"]
        self.assertEqual(chat["repo"], inst["repo"])
        self.assertEqual(chat["source_ref"], inst["source_ref"])
        self.assertNotEqual(chat["weight_path"], inst["weight_path"])
        self.assertEqual(chat["weight_path"], "chat/model.safetensors")
        self.assertEqual(inst["weight_path"], "model.safetensors")

    def test_qwen3_near_band_is_not_hidden_by_model_name(self):
        meta = self.cfg["candidates"]["qwen3-0.6b"]
        self.assertEqual(meta["admission_class"], "near_band_exception")
        self.assertEqual(meta["expected_api_parameters"], 751632384)
        ok, reason = s.parameter_gate(
            meta, 751632384, self.cfg["parameter_band"])
        self.assertTrue(ok)
        self.assertIn("near-band", reason)

        changed = copy.deepcopy(meta)
        changed["admission_class"] = "direct_candidate"
        ok, _ = s.parameter_gate(
            changed, 751632384, self.cfg["parameter_band"])
        self.assertFalse(ok)

    def test_direct_candidate_outside_band_fails_parameter_gate(self):
        meta = self.cfg["candidates"]["smollm2-360m-instruct"]
        ok, _ = s.parameter_gate(
            meta, 900_000_000, self.cfg["parameter_band"])
        self.assertFalse(ok)

    def test_resolve_candidate_records_metadata_without_weight_download(self):
        meta = self.cfg["candidates"]["qwen3-0.6b"]
        payload = self.fake_payload(meta)
        config_bytes = json.dumps({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
        }).encode()
        tokenizer_bytes = json.dumps({
            "chat_template": "{{ tools }} <tool_call>"
        }).encode()
        with patch.object(s, "http_json", return_value=payload), patch.object(
            s, "small_file",
            side_effect=[
                (config_bytes, "present"),
                (tokenizer_bytes, "present"),
            ],
        ):
            row = s.resolve_candidate("qwen3-0.6b", meta, self.cfg)
        self.assertTrue(row["source_gate_pass"])
        self.assertEqual(row["safetensors_parameters"], 751632384)
        self.assertTrue(row["has_chat_template"])
        self.assertTrue(row["template_mentions_tools"])
        self.assertEqual(row["weight_bytes_downloaded"], 0)
        self.assertEqual(row["training_steps"], 0)

    def test_standalone_chat_template_file_is_detected(self):
        meta = self.cfg["candidates"]["lfm25-350m"]
        payload = self.fake_payload(meta)
        payload["config"]["tokenizer_config"].pop("chat_template", None)
        payload["siblings"].append({
            "rfilename": "chat_template.jinja",
            "blobId": "9" * 40,
            "size": 64,
        })
        config_bytes = b'{"architectures":["LfmForCausalLM"],"model_type":"lfm2"}'
        tok_bytes = b'{"eos_token":"<|endoftext|>"}'
        jinja_bytes = b"{{ messages }} {% if tools %}<tool_call>{% endif %}"
        with patch.object(s, "http_json", return_value=payload), patch.object(
            s, "small_file",
            side_effect=[
                (config_bytes, "present"),
                (tok_bytes, "present"),
                (jinja_bytes, "present"),
            ],
        ):
            row = s.resolve_candidate("lfm25-350m", meta, self.cfg)
        self.assertTrue(row["has_chat_template"])
        self.assertEqual(row["chat_template_source"], "chat_template.jinja")
        self.assertEqual(
            row["chat_template_file"]["sha256"],
            s.sha_bytes(jinja_bytes))
        self.assertTrue(row["template_mentions_tools"])

    def test_known_weight_hash_mismatch_is_retained_as_source_failure(self):
        meta = self.cfg["candidates"]["smollm2-135m-instruct"]
        payload = self.fake_payload(meta, weight_sha="0" * 64)
        with patch.object(s, "http_json", return_value=payload), patch.object(
            s, "small_file",
            side_effect=[
                (b'{"model_type":"llama"}', "present"),
                (b'{"chat_template":"x"}', "present"),
            ],
        ):
            row = s.resolve_candidate(
                "smollm2-135m-instruct", meta, self.cfg)
        self.assertFalse(row["weight_gate_pass"])
        self.assertFalse(row["source_gate_pass"])
        self.assertIn("mismatch", row["weight_gate_reason"])

    def test_named_custom_license_normalizes_other_plus_license_name(self):
        payload = {
            "cardData": {
                "license": "other",
                "license_name": "LFM1.0",
            },
            "tags": [],
        }
        self.assertEqual(s.license_from(payload), "lfm1.0")

    def test_lfm_license_gate_and_zenyx_mutation_only_cannot_disappear(self):
        lfm = self.cfg["candidates"]["lfm25-350m"]
        self.assertEqual(lfm["expected_license"], "lfm1.0")
        self.assertTrue(lfm["production_license_gate"])

        zenyx = self.cfg["candidates"]["zenyx-mcen-delta-mamba-565m"]
        self.assertEqual(zenyx["admission_class"], "mutation_only")
        self.assertFalse(zenyx["direct_dialogue_candidate"])
        self.assertTrue(zenyx["requires_custom_runtime"])
        self.assertFalse(zenyx["instruction_model"])

    def test_evidence_readback_preserves_near_band_mutation_and_zero_work(self):
        evidence = self.fake_evidence()
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "evidence.json"
            s.write_json(path, evidence)
            result = s.verify(path)
        self.assertEqual(result["candidates"], 9)
        self.assertEqual(result["training_steps"], 0)
        self.assertEqual(result["weight_bytes_downloaded"], 0)
        self.assertFalse(result["population_expansion_admitted"])
        self.assertNotIn("sprocket-500m-chat",
                         result["unresolved_licenses"])
        self.assertNotIn("sprocket-500m-instruct",
                         result["unresolved_licenses"])

    def test_known_weight_tamper_refuses_readback(self):
        evidence = self.fake_evidence()
        row = next(
            x for x in evidence["candidates"]
            if x["candidate_id"] == "sprocket-500m-instruct"
        )
        row["weight"]["lfs_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "evidence.json"
            s.write_json(path, evidence)
            with self.assertRaises(s.SourceError):
                s.verify(path)

    def test_qwen3_parameter_tamper_refuses_readback(self):
        evidence = self.fake_evidence()
        row = next(
            x for x in evidence["candidates"]
            if x["candidate_id"] == "qwen3-0.6b"
        )
        row["safetensors_parameters"] = 600_000_000
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "evidence.json"
            s.write_json(path, evidence)
            with self.assertRaises(s.SourceError):
                s.verify(path)

    def test_zenyx_direct_promotion_refuses_readback(self):
        evidence = self.fake_evidence()
        row = next(
            x for x in evidence["candidates"]
            if x["candidate_id"] == "zenyx-mcen-delta-mamba-565m"
        )
        row["runtime_eligibility"] = "stage_b_candidate"
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "evidence.json"
            s.write_json(path, evidence)
            with self.assertRaises(s.SourceError):
                s.verify(path)

    def test_archive_hash_and_report_replay(self):
        evidence = self.fake_evidence()
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "archive"
            root.mkdir()
            s.write_json(root / "source-evidence.json", evidence)
            (root / "REPORT.md").write_text(
                s.markdown_report(evidence), encoding="utf-8")
            (root / "hashes.sha256").write_text(
                "\n".join(s.hash_inventory(root)) + "\n",
                encoding="utf-8")
            result = s.verify_archive(root)
            self.assertEqual(
                result["verdict"],
                "DETECTIVE_002A_SOURCE_READBACK_PASS")
            (root / "REPORT.md").write_text("tampered\n")
            with self.assertRaises(s.SourceError):
                s.verify_archive(root)

    def test_api_urls_freeze_exact_refs_and_resolve_main_once(self):
        exact = self.cfg["candidates"]["qwen3-0.6b"]
        url = s.api_url(exact["repo"], exact["source_ref"])
        self.assertIn("/revision/", url)
        self.assertIn(exact["source_ref"], url)

        floating = self.cfg["candidates"]["lfm25-350m"]
        url = s.api_url(floating["repo"], floating["source_ref"])
        self.assertNotIn("/revision/", url)
        self.assertTrue(url.endswith("?blobs=true"))

    def test_small_metadata_helper_refuses_large_payload(self):
        with patch.object(s, "http_bytes", return_value=b"x" * 2_000_001):
            with self.assertRaises(s.SourceError):
                s.small_file("owner/repo", "a" * 40, "config.json")


if __name__ == "__main__":
    unittest.main()
