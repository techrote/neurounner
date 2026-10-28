"""Offline custody qualification for DETECTIVE-005A failed-run recovery."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import detective_005a_assemble as assemble
from scripts import detective_005a_fun_screen as screen
from scripts import detective_005a_live as live
from scripts import detective_005a_recover as recover


HEAD = "1" * 40
RUN = "123456"


class Detective005ARecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = screen.load_contract()

    def row(self, member_id="V-G01-A", prompt_id="F01"):
        src = screen.expected_source(self.cfg, member_id)
        text = "My wrench and the clock both insist the relay is haunted."
        return {
            "schema": "detective-005a-raw-response-v1",
            "experiment": "DETECTIVE-005A",
            "member_id": member_id,
            "prompt_id": prompt_id,
            "source": src,
            "request": screen.expected_request(self.cfg, prompt_id),
            "response": {
                "text": text,
                "raw_with_special_tokens": text,
                "token_ids": [101],
                "finish_reason": "length",
                "prompt_tokens": 32,
                "completion_tokens": 1,
                "elapsed_seconds": 0.001,
            },
            "runner": {
                "repository_head_sha": HEAD,
                "actions_run_id": RUN,
                "base_weights_sha256_verified": src["base_weights_sha256"],
                "adapter_archive_sha256_verified": src["asset_sha256"],
                "adapter_directory_sha256_verified": src[
                    "adapter_directory_sha256"],
                "synthetic_test_only": True,
            },
        }

    def test_interrupted_torn_journal_is_archived_byte_identically(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "artifacts" / "detective005a-G01"
            src.mkdir(parents=True)
            partial = src / "results.partial.jsonl"
            live.append_completed_raw(partial, self.row())
            with partial.open("ab") as fh:
                fh.write(b'{"torn":')
            original = partial.read_bytes()
            with mock.patch.dict(os.environ, {
                    "DETECTIVE_HEAD_SHA": HEAD, "GITHUB_RUN_ID": RUN}):
                live.write_interrupted_receipt(
                    "G01", src, RuntimeError("synthetic failure"))
            out = root / "archive"
            manifest = recover.recover(root / "artifacts", out, HEAD, RUN)
            self.assertEqual(manifest["validated_raw_prefix_rows"], 1)
            self.assertEqual(manifest["complete_lineage_artifacts"], 0)
            self.assertFalse(manifest["review_packet_generated"])
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(),
                original)
            report = screen.read_json(
                out / "provenance/source-artifacts.json")["lineages"][0]
            self.assertEqual(report["journal_tail_problem"],
                             "unterminated_raw_row")
            self.assertEqual(report["custody_warnings"], [])
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 1)
            with partial.open("ab") as fh:
                fh.write(b"more bad bytes")
            # Changing the transient source after archival cannot rewrite it.
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(), original)
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 1)

    def test_unauthorized_source_run_row_is_quarantined_not_discarded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "artifacts" / "detective005a-G01"
            src.mkdir(parents=True)
            bad = self.row()
            bad["runner"]["repository_head_sha"] = "2" * 40
            partial = src / "results.partial.jsonl"
            partial.write_text(screen.canonical(bad) + "\n", encoding="utf-8")
            original = partial.read_bytes()
            out = root / "archive"
            manifest = recover.recover(root / "artifacts", out, HEAD, RUN)
            self.assertEqual(manifest["validated_raw_prefix_rows"], 0)
            report = screen.read_json(
                out / "provenance/source-artifacts.json")["lineages"][0]
            self.assertIn("invalid_response_1:ScreenError",
                          report["custody_warnings"])
            self.assertIn("missing_interruption_receipt",
                          report["custody_warnings"])
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(),
                original)
            recover.verify(out)

    def test_full_lineage_from_failed_assemble_is_only_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "artifacts" / "detective005a-G01"
            src.mkdir(parents=True)
            rows = [self.row(f"V-G01-{s}", p["id"])
                    for s in "ABC" for p in self.cfg["prompts"]]
            result_bytes = "".join(screen.canonical(r) + "\n"
                                   for r in rows).encode()
            (src / "results.jsonl").write_bytes(result_bytes)
            (src / "results.partial.jsonl").write_bytes(result_bytes)
            assemble.write_json(src / "manifest.json", {
                "schema": "detective-005a-live-lineage-manifest-v1",
                "experiment": "DETECTIVE-005A",
                "issue": 176,
                "lineage": "G01",
                "source_plan": live.source_plan("G01"),
                "runner_script_sha256":
                    screen.sha256_file(Path(live.__file__)),
                "results_jsonl_sha256": screen.sha256_bytes(result_bytes),
                "results": 12,
                "members": [],
                "total_generated_tokens": 12,
                "peak_rss_kib": 1,
                "optimizer_steps": 0,
                "paid_api_calls": 0,
                "model_judge_calls": 0,
                "production_admission": False,
                "population_expansion_admitted": False,
            })
            out = root / "archive"
            manifest = recover.recover(root / "artifacts", out, HEAD, RUN)
            self.assertEqual(manifest["complete_lineage_artifacts"], 1)
            self.assertEqual(manifest["validated_raw_prefix_rows"], 12)
            self.assertFalse(manifest["review_packet_generated"])
            self.assertEqual(
                screen.read_json(out / "provenance/source-artifacts.json")
                ["lineages"][0]["status"], "complete_lineage_recovered")
            self.assertFalse((out / "review/fun-packet.json").exists())
            self.assertFalse(recover.verify(out)["review_packet_generated"])

    def test_artifact_api_metadata_and_missing_download_are_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "artifacts" / "detective005a-G01").mkdir(parents=True)
            (root / "artifacts" / "detective005a-G01" /
             "results.partial.jsonl").write_text(
                screen.canonical(self.row()) + "\n", encoding="utf-8")
            metadata = root / "artifact-list.json"
            metadata.write_text(json.dumps({"artifacts": [
                {"name": "detective005a-G02", "id": 711, "digest":
                 "sha256:" + "2" * 64, "size_in_bytes": 10,
                 "expired": False},
            ]}), encoding="utf-8")
            out = root / "archive"
            recover.recover(root / "artifacts", out, HEAD, RUN, metadata)
            provenance = screen.read_json(
                out / "provenance/source-artifacts.json")
            self.assertIn("downloaded_artifact_not_listed",
                          provenance["lineages"][0]["custody_warnings"])
            self.assertIn("listed_artifact_not_downloaded",
                          provenance["lineages"][1]["custody_warnings"])
            self.assertEqual(provenance["lineages"][1]["artifact_api"]["id"],
                             711)
            self.assertEqual((out /
                "provenance/actions-artifacts-api.json").read_bytes(),
                metadata.read_bytes())
            recover.verify(out)

    def test_no_artifacts_never_claims_work_or_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "artifacts").mkdir()
            out = root / "archive"
            manifest = recover.recover(root / "artifacts", out, HEAD, RUN)
            self.assertEqual(manifest["validated_raw_prefix_rows"], 0)
            self.assertEqual(manifest["lineages_with_artifacts"], 0)
            self.assertEqual(manifest["complete_lineage_artifacts"], 0)
            self.assertFalse(manifest["production_admission"])
            recover.verify(out)

    def test_tampered_archive_bytes_fail_readback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / "artifacts" / "detective005a-G01"
            folder.mkdir(parents=True)
            (folder / "results.partial.jsonl").write_text(
                screen.canonical(self.row()) + "\n", encoding="utf-8")
            out = root / "archive"
            recover.recover(root / "artifacts", out, HEAD, RUN)
            raw = out / "raw/G01/results.partial.jsonl"
            raw.write_bytes(raw.read_bytes() + b"x")
            with self.assertRaisesRegex(screen.ScreenError,
                                        "recovered original artifact bytes"):
                recover.verify(out)

    def test_refuse_unpacked_weights_and_duplicate_raw_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            folder = root / "detective005a-G01"
            (folder / "work").mkdir(parents=True)
            (folder / "work" / "adapter_model.safetensors").write_bytes(b"bad")
            with self.assertRaisesRegex(screen.ScreenError, "work folder"):
                recover.artifact_files(root, "G01")
            (folder / "work" / "adapter_model.safetensors").unlink()
            (folder / "work").rmdir()
            (folder / "results.partial.jsonl").write_text("{}\n")
            nested = folder / "out" / "G01"
            nested.mkdir(parents=True)
            (nested / "results.partial.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(screen.ScreenError, "duplicate"):
                recover.artifact_files(root, "G01")

    def test_live_recovery_workflow_guardrails(self):
        workflow = (
            screen.ROOT / ".github/workflows/detective-005a-live.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("recover-failed-evidence:", workflow)
        self.assertIn("always()", workflow)
        self.assertIn("needs.assemble.result != 'success'", workflow)
        self.assertIn("evidence/detective-005a-partial-", workflow)
        self.assertIn("scripts/detective_005a_recover.py", workflow)
        self.assertIn("refuse overwrite", workflow.lower())
        self.assertIn("RUN-144-NO-TRAINING", workflow)
        self.assertIn("refs/heads/main", workflow)


if __name__ == "__main__":
    unittest.main()
