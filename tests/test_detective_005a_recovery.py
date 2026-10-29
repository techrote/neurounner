"""Offline custody qualification for DETECTIVE-005A failed-run recovery."""
from __future__ import annotations

import copy
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
            self.assertEqual(manifest["complete_lineage_artifacts"], 0)
            self.assertEqual(manifest["validated_raw_prefix_rows"], 12)
            self.assertFalse(manifest["review_packet_generated"])
            unbound = screen.read_json(
                out / "provenance/source-artifacts.json")["lineages"][0]
            self.assertEqual(unbound["status"], "recovery_only")
            self.assertIn("artifact_api_metadata_unavailable",
                          unbound["custody_warnings"])
            self.assertFalse(recover.verify(out)["artifact_transport_complete"])
            self.assertFalse((out / "review/fun-packet.json").exists())
            self.assertFalse(recover.verify(out)["review_packet_generated"])

            # A source-complete final JSONL without the mandatory fsynced
            # journal is still preserved, but must not gain complete custody.
            (src / "results.partial.jsonl").unlink()
            degraded = root / "archive-missing-journal"
            degraded_manifest = recover.recover(
                root / "artifacts", degraded, HEAD, RUN)
            self.assertEqual(degraded_manifest["validated_raw_prefix_rows"], 12)
            self.assertEqual(degraded_manifest["complete_lineage_artifacts"], 0)
            saved = screen.read_json(
                degraded / "provenance/source-artifacts.json")["lineages"][0]
            self.assertEqual(saved["status"], "recovery_only")
            self.assertIn("complete_lineage_missing_incremental_journal",
                          saved["custody_warnings"])
            self.assertEqual(
                (degraded / "raw/G01/results.jsonl").read_bytes(),
                result_bytes)
            self.assertFalse(recover.verify(degraded)["review_packet_generated"])

            # A matching GitHub API response can qualify complete custody,
            # but a foreign Actions-run identity must demote, not discard it.
            (src / "results.partial.jsonl").write_bytes(result_bytes)
            metadata = root / "source-api.json"
            original_api = {"artifacts": [{
                "name": "detective005a-G01", "id": 712,
                "digest": "sha256:" + "f" * 64,
                "size_in_bytes": 1200,
                "expired": False,
                "workflow_run": {
                    "id": int(RUN), "head_sha": HEAD,
                    "repository_id": 1432134981,
                },
            }]}
            metadata.write_text(json.dumps(original_api), encoding="utf-8")
            valid_archive = root / "archive-api"
            qualified = recover.recover(root / "artifacts", valid_archive,
                                        HEAD, RUN, metadata, "success", "success")
            self.assertEqual(qualified["complete_lineage_artifacts"], 1)
            self.assertTrue(qualified["artifact_transport_complete"])
            self.assertTrue(recover.require_complete_transport(
                valid_archive)["artifact_transport_complete"])
            foreign_api = copy.deepcopy(original_api)
            foreign_api["artifacts"][0]["workflow_run"]["head_sha"] = "2" * 40
            metadata.write_text(json.dumps(foreign_api), encoding="utf-8")
            foreign_archive = root / "archive-foreign"
            quarantined = recover.recover(root / "artifacts", foreign_archive,
                                           HEAD, RUN, metadata)
            self.assertEqual(quarantined["validated_raw_prefix_rows"], 12)
            self.assertEqual(quarantined["complete_lineage_artifacts"], 0)
            record = screen.read_json(
                foreign_archive / "provenance/source-artifacts.json"
            )["lineages"][0]
            self.assertEqual(record["status"], "recovery_only")
            self.assertIn("artifact_api_source_run_mismatch",
                          record["custody_warnings"])
            self.assertEqual(
                (foreign_archive / "raw/G01/results.jsonl").read_bytes(),
                result_bytes)
            recover.verify(foreign_archive)

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
                 "expired": False, "workflow_run": {
                    "id": int(RUN), "head_sha": HEAD,
                    "repository_id": 1432134981}},
            ]}), encoding="utf-8")
            out = root / "archive"
            archive = recover.recover(root / "artifacts", out, HEAD, RUN,
                                      metadata, "success", "failure")
            self.assertFalse(archive["artifact_transport_complete"])
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
            self.assertFalse(recover.verify(out)["artifact_transport_complete"])
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(out)

            # Even if the download action reports success, a same-run API
            # entry that survived but was not downloaded blocks green custody.
            alternate = root / "archive-download-outcome-success"
            manifest = recover.recover(root / "artifacts", alternate,
                                        HEAD, RUN, metadata,
                                        "success", "success")
            self.assertFalse(manifest["artifact_transport_complete"])
            self.assertFalse(recover.verify(alternate)["artifact_transport_complete"])
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(alternate)

    def test_fatal_download_outcome_with_a_valid_api_row_stays_red(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "artifacts" / "detective005a-G01"
            source.mkdir(parents=True)
            original = screen.canonical(self.row()).encode() + b"\n"
            (source / "results.partial.jsonl").write_bytes(original)
            metadata = root / "api.json"
            metadata.write_text(json.dumps({"artifacts": [{
                "name": "detective005a-G01", "id": 1001,
                "digest": "sha256:" + "f" * 64, "size_in_bytes": 100,
                "expired": False, "workflow_run": {
                    "id": int(RUN), "head_sha": HEAD,
                    "repository_id": 1432134981},
            }]}), encoding="utf-8")
            # All visible source bytes survived, but the failed downloader
            # might have dropped another file inside the artifact. Never green.
            out = root / "archive"
            manifest = recover.recover(root / "artifacts", out, HEAD, RUN,
                                        metadata, "success", "failure")
            self.assertEqual(manifest["validated_raw_prefix_rows"], 1)
            self.assertFalse(manifest["artifact_transport_complete"])
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(), original)
            self.assertFalse(recover.verify(out)["artifact_transport_complete"])
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(out)

            # API lookup failure also blocks green even with clean download.
            missing_api = root / "archive-no-api"
            missing = recover.recover(root / "artifacts", missing_api,
                                      HEAD, RUN, None, "failure", "success")
            self.assertFalse(missing["artifact_transport_complete"])
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(missing_api)

    def test_malformed_artifact_api_keeps_original_partial_and_api_bytes(self):
        """API truncation/schema failure cannot erase surviving model output."""
        cases = (
            ("truncated", b'{"artifacts": ['),
            ("wrong_shape", b'{"artifacts": {"wrong": "type"}}'),
            ("null_entry", b'{"artifacts": [null]}'),
            ("duplicate", json.dumps({"artifacts": [
                {"name": "detective005a-G01"},
                {"name": "detective005a-G01"}]}).encode()),
        )
        for label, original_api in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                source = root / "artifacts" / "detective005a-G01"
                source.mkdir(parents=True)
                original_raw = (screen.canonical(self.row()).encode("utf-8")
                                + b"\n" + b'{"torn":')
                (source / "results.partial.jsonl").write_bytes(original_raw)
                with mock.patch.dict(os.environ, {
                        "DETECTIVE_HEAD_SHA": HEAD, "GITHUB_RUN_ID": RUN}):
                    live.write_interrupted_receipt(
                        "G01", source, RuntimeError("synthetic failure"))
                metadata = root / "original-api.json"
                metadata.write_bytes(original_api)
                out = root / "recovery"
                manifest = recover.recover(
                    root / "artifacts", out, HEAD, RUN, metadata,
                    "success", "success")
                self.assertEqual(manifest["validated_raw_prefix_rows"], 1)
                self.assertTrue(manifest["artifact_api_metadata_invalid"])
                self.assertFalse(manifest["artifact_transport_complete"])
                self.assertEqual(
                    (out / "raw/G01/results.partial.jsonl").read_bytes(),
                    original_raw)
                archived_api = out / "provenance/actions-artifacts-api.json"
                self.assertEqual(archived_api.read_bytes(), original_api)
                provenance = screen.read_json(
                    out / "provenance/source-artifacts.json")
                self.assertTrue(provenance["artifact_api_metadata_invalid"])
                record = provenance["lineages"][0]
                self.assertEqual(record["status"], "recovery_only")
                self.assertIn("artifact_api_metadata_invalid",
                              record["custody_warnings"])
                self.assertNotIn("downloaded_artifact_not_listed",
                                 record["custody_warnings"])
                self.assertEqual(record["validated_raw_prefix_rows"], 1)
                self.assertFalse(recover.verify(out)["artifact_transport_complete"])
                with self.assertRaisesRegex(
                        screen.ScreenError, "incomplete original lineage"):
                    recover.require_complete_transport(out)

                # Original malformed API bytes are included in SHA inventory.
                archived_api.write_bytes(original_api + b"tampered")
                with self.assertRaisesRegex(
                        screen.ScreenError, "hash inventory"):
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

    @unittest.skipUnless(os.name == "posix", "symlink fixtures require POSIX")
    def test_symlinked_recovery_is_not_self_contained_even_with_matching_hash(self):
        """A Git link is not an archived copy, even when its target hashes match."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "artifacts" / "detective005a-G01"
            source.mkdir(parents=True)
            (source / "results.partial.jsonl").write_text(
                screen.canonical(self.row()) + "\n", encoding="utf-8")
            out = root / "archive"
            recover.recover(root / "artifacts", out, HEAD, RUN)
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 1)

            original = out / "raw/G01/results.partial.jsonl"
            external = root / "non-archived-original.jsonl"
            external.write_bytes(original.read_bytes())
            original.unlink()
            original.symlink_to(external)
            self.assertEqual(original.read_bytes(), external.read_bytes())
            with self.assertRaisesRegex(screen.ScreenError, "symlink"):
                recover.verify(out)

            # An entire lineage directory linked to external storage is
            # equally non-durable, even though every source byte still exists.
            original.unlink()
            original.write_bytes(external.read_bytes())
            lineage_dir = out / "raw/G01"
            external_dir = root / "non-archived-lineage"
            lineage_dir.rename(external_dir)
            lineage_dir.symlink_to(external_dir, target_is_directory=True)
            with self.assertRaisesRegex(screen.ScreenError, "symlink"):
                recover.verify(out)

            # The root must also be a real directory, not a linked alias.
            linked_archive = root / "archive-alias"
            linked_archive.symlink_to(out, target_is_directory=True)
            with self.assertRaisesRegex(screen.ScreenError, "physical directory"):
                recover.verify(linked_archive)

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
        self.assertIn("id: artifact-api", workflow)
        self.assertIn("id: download-lineages", workflow)
        self.assertIn("steps.download-lineages.outcome", workflow)
        self.assertIn("steps.artifact-api.outcome", workflow)
        self.assertIn("--require-complete-transport", workflow)
        self.assertIn("refuse overwrite", workflow.lower())
        self.assertIn("RUN-144-NO-TRAINING", workflow)
        self.assertIn("refs/heads/main", workflow)

    def test_push_failure_keeps_a_transient_recovery_fallback(self):
        """Source guard: retained stage precedes git push and final red gate."""
        workflow = (screen.ROOT / ".github/workflows/detective-005a-live.yml").read_text(
            encoding="utf-8")
        recovery = workflow.split("  recover-failed-evidence:\n", 1)[1]
        fallback_name = "      - name: Upload verified recovery staging as transient push-failure fallback"
        push_name = "      - name: Promote failure evidence to fresh run-scoped branch only"
        final_name = "      - name: Fail recovery gate when original artifact custody is incomplete"
        self.assertEqual(recovery.count(fallback_name), 1)
        self.assertLess(recovery.index("      - name: Build and read back quarantine-only durable dataset"),
                        recovery.index(fallback_name))
        self.assertLess(recovery.index(fallback_name), recovery.index(push_name))
        self.assertLess(recovery.index(push_name), recovery.index(final_name))
        fallback = recovery.split(fallback_name, 1)[1].split(push_name, 1)[0]
        self.assertIn("continue-on-error: true", fallback)
        self.assertIn("actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9", fallback)
        self.assertIn("name: detective005a-recovery-fallback-${{ github.run_id }}", fallback)
        self.assertIn("path: staged/detective-005a-recovery", fallback)
        self.assertIn("if-no-files-found: error", fallback)
        self.assertIn("retention-days: 30", fallback)
        self.assertIn("--require-complete-transport", recovery.split(final_name, 1)[1])



if __name__ == "__main__":
    unittest.main()
