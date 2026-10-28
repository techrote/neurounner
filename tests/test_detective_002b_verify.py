"""Offline evidence-replay tests for DETECTIVE-002B runtime smoke."""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

from scripts import detective_002b_verify as v


class RuntimeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.sources = v.load_authority()

    def _task_response(self, task_id: str) -> str:
        return {
            "B01": (
                "If the wall sensor trips, I store one direction bit and "
                "step away once."
            ),
            "B02": (
                "I remember 02:14, but that is my memory, not proof."
            ),
            "B03": (
                "Use 02:31; the stamped record corrects my old memory."
            ),
            "B04": (
                "I did not see a red ceramic token; that is rumor."
            ),
        }[task_id]

    def _result(self, cid: str, *, status: str = "runtime_pass") -> dict:
        source = self.sources[cid]
        tasks = []
        if status == "runtime_pass":
            for i, task in enumerate(self.cfg["tasks"], 1):
                response = self._task_response(task["id"])
                count = 8 + i
                tasks.append({
                    "task_id": task["id"],
                    "family": task["family"],
                    "prompt": task["prompt"],
                    "messages": [
                        {"role": "system", "content": self.cfg["system"]},
                        {"role": "user", "content": task["prompt"]},
                    ],
                    "prompt_tokens": 30 + i,
                    "prompt_token_ids_sha256": f"{i:064x}",
                    "generated_token_ids": list(range(count)),
                    "generated_tokens": count,
                    "elapsed_seconds": 1.0 + i / 10,
                    "tokens_per_second": count / (1.0 + i / 10),
                    "response": response,
                    "response_with_special_tokens": response,
                    "observation": v.recompute_observation(
                        task, response, count),
                })
        doc = {
            "schema": "detective-npc-runtime-result-v1",
            "experiment": "DETECTIVE-002B",
            "issue": 130,
            "candidate_id": cid,
            "config_sha256": v.sha_file(v.CONFIG),
            "source_evidence_sha256": v.sha_file(v.SOURCE),
            "source": source,
            "selected_weight_verified": {
                "path": source["weight"]["path"],
                "sha256": source["weight"]["lfs_sha256"],
                "bytes": source["weight"]["size"],
            },
            "runtime_budget": self.cfg["runtime"],
            "runner": {
                "platform": "Linux-test",
                "cpu_count": 4,
                "cpu_model": "synthetic-cpu",
                "torch_threads": 4,
                "torch_interop_threads": 1,
            },
            "packages": {
                "python": "3.12.0",
                "torch": "2.14.1+cpu",
                "transformers": "4.57.6",
                "huggingface_hub": "0.36.2",
                "safetensors": "0.8.0",
            },
            "training_steps": 0,
            "paid_api_calls": 0,
            "gpu_used": False,
            "relevance_mapping_accessed": False,
            "population_expansion_admitted": False,
            "tasks": tasks,
            "runtime_status": status,
            "total_generated_tokens": sum(
                x["generated_tokens"] for x in tasks),
            "peak_rss_kib": 900000,
        }
        if status == "runtime_pass":
            doc.update({
                "model_load_seconds": 1.25,
                "rss_after_load_kib": 800000,
                "loaded_parameter_count": (
                    source["safetensors_parameters"]
                    if source["safetensors_parameters"] is not None
                    else source["nominal_parameters"]
                ),
                "protocol_specialization_required": False,
            })
        elif status == "protocol_specialization_required":
            doc.update({
                "protocol_specialization_required": True,
                "error_type": "ProtocolSpecialization",
                "error": "chat template requires a specialized wire",
            })
        else:
            doc.update({
                "protocol_specialization_required": False,
                "error_type": "RuntimeError",
                "error": "synthetic runtime failure",
            })
        return doc

    def _write_results(self, root: pathlib.Path,
                       overrides: dict[str, str] | None = None) -> None:
        overrides = overrides or {}
        for cid in self.cfg["candidates"]:
            v.write_json(
                root / f"{cid}.json",
                self._result(cid, status=overrides.get(
                    cid, "runtime_pass")),
            )

    def test_synthetic_complete_matrix_assembles_and_replays(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            results = root / "results"
            results.mkdir()
            self._write_results(results)
            evidence = root / "evidence"
            summary = v.assemble(
                results, evidence, "a" * 40, "123456",
                "https://github.com/techrote/neurounner/actions/runs/123456",
            )
            self.assertEqual(summary["runtime_passes"], 7)
            self.assertEqual(summary["runtime_failures"], 0)
            self.assertEqual(
                summary["protocol_specialization_required"], 0)
            replay = v.verify_evidence(evidence)
            self.assertEqual(
                replay["verdict"],
                "DETECTIVE_002B_RUNTIME_READBACK_PASS")
            self.assertEqual(replay["runtime_passes"], 7)
            self.assertEqual(replay["training_steps"], 0)
            self.assertFalse(
                replay["population_expansion_admitted"])

    def test_runtime_failure_is_durable_evidence_not_assembly_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            results = root / "results"
            results.mkdir()
            self._write_results(
                results,
                {"lfm25-350m": "runtime_fail"})
            evidence = root / "evidence"
            summary = v.assemble(
                results, evidence, "b" * 40, "654321",
                "https://github.com/techrote/neurounner/actions/runs/654321",
            )
            self.assertEqual(summary["runtime_passes"], 6)
            self.assertEqual(summary["runtime_failures"], 1)
            replay = v.verify_evidence(evidence)
            self.assertEqual(replay["runtime_failures"], 1)
            raw = v.read_json(
                evidence / "raw/lfm25-350m.json")
            self.assertEqual(raw["runtime_status"], "runtime_fail")
            self.assertIn("synthetic runtime failure", raw["error"])

    def test_protocol_specialization_is_separate_from_runtime_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            results = root / "results"
            results.mkdir()
            self._write_results(
                results,
                {"sprocket-500m-chat":
                    "protocol_specialization_required"})
            evidence = root / "evidence"
            summary = v.assemble(
                results, evidence, "c" * 40, "123123",
                "https://github.com/techrote/neurounner/actions/runs/123123",
            )
            self.assertEqual(summary["runtime_passes"], 6)
            self.assertEqual(summary["runtime_failures"], 0)
            self.assertEqual(
                summary["protocol_specialization_required"], 1)
            v.verify_evidence(evidence)

    def test_mechanical_observation_tamper_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            path = root / "row.json"
            doc = self._result("smollm2-135m-instruct")
            doc["tasks"][1]["observation"][
                "remembered_marker_hit"] = False
            v.write_json(path, doc)
            with self.assertRaises(v.VerifyError):
                v.validate_result(
                    path, self.cfg, self.sources)

    def test_thread_budget_tamper_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            path = root / "row.json"
            doc = self._result("qwen25-0.5b-instruct")
            doc["runner"]["torch_threads"] = 8
            v.write_json(path, doc)
            with self.assertRaises(v.VerifyError):
                v.validate_result(
                    path, self.cfg, self.sources)

    def test_selected_weight_tamper_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            path = root / "row.json"
            doc = self._result("sprocket-500m-chat")
            doc["selected_weight_verified"]["sha256"] = "0" * 64
            v.write_json(path, doc)
            with self.assertRaises(v.VerifyError):
                v.validate_result(
                    path, self.cfg, self.sources)

    def test_summary_tamper_refuses_after_valid_assembly(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            results = root / "results"
            results.mkdir()
            self._write_results(results)
            evidence = root / "evidence"
            v.assemble(
                results, evidence, "d" * 40, "987654",
                "https://github.com/techrote/neurounner/actions/runs/987654",
            )
            summary_path = evidence / "summary.json"
            summary = v.read_json(summary_path)
            summary["runtime_passes"] = 99
            v.write_json(summary_path, summary)
            # Rehashing the inventory cannot rescue a false summary because
            # independent replay recomputes it from raw results.
            (evidence / "hashes.sha256").write_text(
                "\n".join(v.hash_inventory(evidence)) + "\n",
                encoding="utf-8")
            with self.assertRaises(v.VerifyError):
                v.verify_evidence(evidence)

    def test_report_contains_no_ranking_or_population_admission(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            results = root / "results"
            results.mkdir()
            self._write_results(results)
            evidence = root / "evidence"
            summary = v.assemble(
                results, evidence, "e" * 40, "111222",
                "https://github.com/techrote/neurounner/actions/runs/111222",
            )
            report = (evidence / "REPORT.md").read_text()
            self.assertIn("not a model ranking", report)
            self.assertIn(
                "population expansion admitted: **no**", report)
            self.assertFalse(
                summary["population_expansion_admitted"])

    def test_candidate_summary_replays_marker_fields(self):
        doc = self._result("smollm2-360m-instruct")
        row = v.candidate_summary(doc)
        self.assertTrue(row["memory_marker_hit"])
        self.assertTrue(row["memory_qualifier_hit"])
        self.assertTrue(row["corrected_marker_hit"])
        self.assertFalse(row["stale_marker_hit"])
        self.assertTrue(row["rumor_qualifier_hit"])
        self.assertTrue(row["unsupported_marker_hit"])
        self.assertEqual(row["truncated_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
