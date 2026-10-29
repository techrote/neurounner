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
                "members": assemble.lineage_member_receipts(
                    self.cfg, "G01", rows)[0],
                "total_generated_tokens": assemble.lineage_member_receipts(
                    self.cfg, "G01", rows)[1],
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
            original_api = {"total_count": 1, "artifacts": [{
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
            metadata.write_text(json.dumps({"total_count": 1, "artifacts": [
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

    def test_distinct_lineage_names_cannot_share_one_artifact_id(self):
        """Name matching alone must not certify duplicate GitHub object IDs."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            originals = {}
            for lineage in ("G01", "G02"):
                folder = root / "artifacts" / f"detective005a-{lineage}"
                folder.mkdir(parents=True)
                raw = (
                    screen.canonical(self.row(member_id=f"V-{lineage}-A")) + "\n"
                ).encode("utf-8")
                (folder / "results.partial.jsonl").write_bytes(raw)
                originals[lineage] = raw

            api = {"total_count": 2, "artifacts": [
                {"name": f"detective005a-{lineage}", "id": 5001,
                 "digest": "sha256:" + character * 64,
                 "size_in_bytes": 200, "expired": False,
                 "workflow_run": {
                     "id": int(RUN), "head_sha": HEAD,
                     "repository_id": 1432134981}}
                for lineage, character in (("G01", "a"), ("G02", "b"))
            ]}
            metadata = root / "api.json"
            metadata.write_text(json.dumps(api), encoding="utf-8")
            out = root / "archive"
            report = recover.recover(
                root / "artifacts", out, HEAD, RUN, metadata,
                "success", "success")
            self.assertTrue(report["artifact_api_metadata_invalid"])
            self.assertFalse(report["artifact_transport_complete"])
            self.assertEqual(report["validated_raw_prefix_rows"], 2)
            for lineage, original in originals.items():
                self.assertEqual(
                    (out / "raw" / lineage / "results.partial.jsonl").read_bytes(),
                    original)
            self.assertEqual(
                (out / "provenance/actions-artifacts-api.json").read_bytes(),
                metadata.read_bytes())
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 2)
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(out)

            # Replacing the repeated ID with a second distinct immutable
            # artifact should restore the *transport* gate, not certify
            # either one-row lineage as a completed 144-response screen.
            api["artifacts"][1]["id"] = 5002
            metadata.write_text(json.dumps(api), encoding="utf-8")
            good = recover.recover(
                root / "artifacts", root / "archive-unique", HEAD, RUN,
                metadata, "success", "success")
            self.assertTrue(good["artifact_transport_complete"])
            self.assertEqual(good["complete_lineage_artifacts"], 0)
            recover.verify(root / "archive-unique")

    def test_successful_api_first_page_cannot_hide_later_artifacts(self):
        """HTTP/API success is insufficient if total_count exceeds page rows."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "artifacts/detective005a-G01"
            source.mkdir(parents=True)
            original = screen.canonical(self.row()).encode("utf-8") + b"\n"
            (source / "results.partial.jsonl").write_bytes(original)
            metadata = root / "api-first-page.json"
            # A ?per_page=100 response can be truncated while HTTP 200
            # and the download action both report success.
            response = {"total_count": 101, "artifacts": [{
                "name": "detective005a-G01", "id": 1001,
                "digest": "sha256:" + "f" * 64,
                "size_in_bytes": 100, "expired": False,
                "workflow_run": {
                    "id": int(RUN), "head_sha": HEAD,
                    "repository_id": 1432134981},
            }]}
            metadata.write_text(json.dumps(response), encoding="utf-8")
            out = root / "recovery"
            report = recover.recover(root / "artifacts", out, HEAD, RUN,
                                     metadata, "success", "success")
            self.assertEqual(report["validated_raw_prefix_rows"], 1)
            self.assertTrue(report["artifact_api_metadata_invalid"])
            self.assertFalse(report["artifact_transport_complete"])
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(), original)
            self.assertEqual(
                (out / "provenance/actions-artifacts-api.json").read_bytes(),
                metadata.read_bytes())
            provenance = screen.read_json(
                out / "provenance/source-artifacts.json")
            self.assertIn("artifact_api_metadata_invalid",
                          provenance["lineages"][0]["custody_warnings"])
            self.assertFalse(recover.verify(out)["artifact_transport_complete"])
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(out)

    def test_fatal_download_outcome_with_a_valid_api_row_stays_red(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "artifacts" / "detective005a-G01"
            source.mkdir(parents=True)
            original = screen.canonical(self.row()).encode() + b"\n"
            (source / "results.partial.jsonl").write_bytes(original)
            metadata = root / "api.json"
            metadata.write_text(json.dumps({"total_count": 1, "artifacts": [{
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
            ("missing_count", b'{"artifacts": [{"name": "detective005a-G01"}]}'),
            ("boolean_count", b'{"total_count": true, "artifacts": []}'),
            ("duplicate", json.dumps({"total_count": 2, "artifacts": [
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

    def test_nested_raw_impostor_and_extra_archived_files_fail_custody(self):
        """Neither a nested input nor a rehashed foreign output is admissible."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "artifacts/detective005a-G01"
            source.mkdir(parents=True)
            original = screen.canonical(self.row()).encode() + bytes([10])
            (source / "results.partial.jsonl").write_bytes(original)
            nested = source / "nested"
            nested.mkdir()
            (nested / "results.jsonl").write_bytes(
                b"synthetic-foreign-raw-impostor")
            metadata = root / "api.json"
            metadata.write_text(json.dumps({"total_count": 1, "artifacts": [{
                "name": "detective005a-G01", "id": 1001,
                "digest": "sha256:" + "a" * 64,
                "size_in_bytes": 100, "expired": False,
                "workflow_run": {"id": int(RUN), "head_sha": HEAD,
                                 "repository_id": 1432134981},
            }]}), encoding="utf-8")
            out = root / "recovery"
            result = recover.recover(root / "artifacts", out, HEAD, RUN,
                                     metadata, "success", "success")
            self.assertEqual(result["validated_raw_prefix_rows"], 1)
            self.assertFalse(result["artifact_transport_complete"])
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(), original)
            self.assertFalse((out / "raw/G01/results.jsonl").exists())
            self.assertFalse((out / "raw/G01/nested").exists())
            provenance = screen.read_json(
                out / "provenance/source-artifacts.json")
            self.assertEqual(
                provenance["lineages"][0]["source_structure_warnings"],
                [recover.STRUCTURE_WARNING])
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 1)
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(out)

            # A new file plus a recalculated SHA inventory still must fail:
            # it was never part of the original artifact/provenance contract.
            foreign = out / "raw/G13/foreign.jsonl"
            foreign.parent.mkdir(parents=True)
            foreign.write_bytes(b"unregistered-synthetic-data")
            (out / "provenance/hashes.sha256").write_text(
                assemble.hash_inventory(out), encoding="utf-8")
            with self.assertRaisesRegex(screen.ScreenError, "unregistered"):
                recover.verify(out)

    def test_nested_untrusted_json_cannot_strand_independent_raw_bytes(self):
        """Recursive JSON in raw, receipt, manifest and API stays quarantined."""
        nested = b"[" * 2500 + b"0" + b"]" * 2500 + b"\n"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            g01 = root / "artifacts/detective005a-G01"
            g02 = root / "artifacts/detective005a-G02"
            g01.mkdir(parents=True)
            g02.mkdir(parents=True)
            for filename in ("results.partial.jsonl", "results.jsonl",
                             "interrupted.json", "manifest.json"):
                (g01 / filename).write_bytes(nested)
            sibling = (screen.canonical(self.row(
                member_id="V-G02-A")).encode("utf-8") + b"\n")
            (g02 / "results.partial.jsonl").write_bytes(sibling)
            api = root / "api.json"
            api.write_bytes(nested)
            out = root / "archive"
            manifest = recover.recover(
                root / "artifacts", out, HEAD, RUN, api, "success", "success")
            self.assertEqual(manifest["validated_raw_prefix_rows"], 1)
            self.assertTrue(manifest["artifact_api_metadata_invalid"])
            self.assertFalse(manifest["artifact_transport_complete"])
            for filename in ("results.partial.jsonl", "results.jsonl",
                             "interrupted.json", "manifest.json"):
                self.assertEqual((out / "raw/G01" / filename).read_bytes(),
                                 nested)
            self.assertEqual((out / "raw/G02/results.partial.jsonl").read_bytes(),
                             sibling)
            self.assertEqual(
                (out / "provenance/actions-artifacts-api.json").read_bytes(),
                nested)
            records = screen.read_json(
                out / "provenance/source-artifacts.json")["lineages"]
            self.assertEqual(records[0]["validated_raw_prefix_rows"], 0)
            self.assertIn(records[0]["journal_tail_problem"],
                          {"non_object_raw_row", "unparseable_raw_row"})
            self.assertIn("invalid_interruption_receipt",
                          records[0]["custody_warnings"])
            self.assertIn("invalid_complete_lineage_manifest",
                          records[0]["custody_warnings"])
            self.assertIn("artifact_api_metadata_invalid",
                          records[0]["custody_warnings"])
            self.assertEqual(records[1]["validated_raw_prefix_rows"], 1)
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 1)
            with self.assertRaisesRegex(screen.ScreenError, "incomplete original"):
                recover.require_complete_transport(out)

    def test_oversized_json_integer_does_not_abort_raw_preservation(self):
        """Python integer conversion limits may raise ValueError, not JSONDecodeError."""
        original = b'{"n":' + b"9" * 12000 + b"}\n"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            g01 = root / "artifacts/detective005a-G01"
            g01.mkdir(parents=True)
            (g01 / "results.partial.jsonl").write_bytes(original)
            out = root / "archive"
            manifest = recover.recover(root / "artifacts", out, HEAD, RUN)
            self.assertEqual(manifest["validated_raw_prefix_rows"], 0)
            self.assertFalse(manifest["artifact_transport_complete"])
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(),
                original)
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 0)

    def test_parser_limit_exception_paths_fail_closed(self):
        """Exercise exceptions independently of Python JSON parser depth limits."""
        for error_type in (RecursionError, ValueError, OverflowError):
            with self.subTest(error_type=error_type.__name__):
                with mock.patch.object(
                        live, "intact_partial_raw_rows",
                        side_effect=error_type("synthetic parser limit")):
                    self.assertEqual(
                        recover.intact_prefix_or_quarantine(Path("unused.jsonl")),
                        (0, "unparseable_raw_row"))
        for error_type in (RecursionError, ValueError, OverflowError):
            with self.subTest(api_error=error_type.__name__):
                with mock.patch.object(
                        screen, "read_json",
                        side_effect=error_type("synthetic artifact API limit")):
                    self.assertEqual(
                        recover.load_api_for_custody(Path("original-api.json")),
                        ({}, True))
                    self.assertEqual(
                        recover.unexpected_artifact_sources(
                            None, Path("original-api.json")), [])

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

    def test_foreign_api_and_download_names_block_green_without_copying_foreign(self):
        """A foreign G13 is an unresolved custody gap, not an empty success."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            expected = artifacts / "detective005a-G01"
            expected.mkdir(parents=True)
            raw = screen.canonical(self.row()).encode() + b"\n"
            (expected / "results.partial.jsonl").write_bytes(raw)
            foreign = artifacts / "detective005a-G13"
            foreign.mkdir()
            (foreign / "unknown.bin").write_bytes(b"synthetic-untrusted-bytes")
            metadata = root / "api.json"
            metadata.write_text(json.dumps({"total_count": 2, "artifacts": [
                {"name": "detective005a-G01", "id": 712,
                 "digest": "sha256:" + "a" * 64, "size_in_bytes": 100,
                 "expired": False, "workflow_run": {
                     "id": int(RUN), "head_sha": HEAD,
                     "repository_id": 1432134981}},
                {"name": "detective005a-G13", "id": 713,
                 "digest": "sha256:" + "b" * 64, "size_in_bytes": 100,
                 "expired": False, "workflow_run": {
                     "id": int(RUN), "head_sha": HEAD,
                     "repository_id": 1432134981}},
            ]}), encoding="utf-8")
            out = root / "archive"
            manifest = recover.recover(
                artifacts, out, HEAD, RUN, metadata, "success", "success")
            self.assertEqual(manifest["validated_raw_prefix_rows"], 1)
            self.assertEqual(manifest["unexpected_artifact_sources"], [
                "api:detective005a-G13", "downloaded:detective005a-G13"])
            self.assertFalse(manifest["artifact_transport_complete"])
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(), raw)
            self.assertFalse((out / "raw/G13").exists())
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 1)
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(out)

            # API-only foreign evidence must be caught even when the
            # downloader did not return the unexpected artifact.
            foreign.rename(root / "not-downloaded")
            out_api_only = root / "archive-api-only"
            second = recover.recover(artifacts, out_api_only, HEAD, RUN,
                                     metadata, "success", "success")
            self.assertEqual(second["unexpected_artifact_sources"],
                             ["api:detective005a-G13"])
            self.assertFalse(second["artifact_transport_complete"])
            recover.verify(out_api_only)

    def test_api_repository_mismatch_blocks_source_bound_custody(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "artifacts/detective005a-G01"
            source.mkdir(parents=True)
            (source / "results.partial.jsonl").write_text(
                screen.canonical(self.row()) + "\n", encoding="utf-8")
            metadata = root / "api.json"
            metadata.write_text(json.dumps({"total_count": 1, "artifacts": [{
                "name": "detective005a-G01", "id": 1001,
                "digest": "sha256:" + "f" * 64,
                "size_in_bytes": 120, "expired": False,
                "workflow_run": {"id": int(RUN), "head_sha": HEAD,
                                 "repository_id": 999},
            }]}), encoding="utf-8")
            out = root / "archive"
            manifest = recover.recover(
                root / "artifacts", out, HEAD, RUN, metadata,
                "success", "success")
            self.assertFalse(manifest["artifact_transport_complete"])
            record = screen.read_json(
                out / "provenance/source-artifacts.json")["lineages"][0]
            self.assertIn("artifact_api_repository_mismatch",
                          record["custody_warnings"])
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 1)

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

    def test_unsafe_files_do_not_strand_other_raw_siblings(self):
        """A foreign weight must never erase intact raw outputs from the run."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            originals = {}
            entries = []
            for lineage, member, artifact_id in (
                    ("G01", "V-G01-A", 1001), ("G02", "V-G02-A", 1002)):
                folder = root / "artifacts" / f"detective005a-{lineage}"
                folder.mkdir(parents=True)
                original = (
                    screen.canonical(self.row(member_id=member)) + "\n"
                ).encode("utf-8")
                (folder / "results.partial.jsonl").write_bytes(original)
                originals[lineage] = original
                entries.append({
                    "name": f"detective005a-{lineage}", "id": artifact_id,
                    "digest": "sha256:" + "f" * 64, "size_in_bytes": 100,
                    "expired": False, "workflow_run": {
                        "id": int(RUN), "head_sha": HEAD,
                        "repository_id": 1432134981},
                })
            foreign = root / "artifacts" / "detective005a-G01"
            (foreign / "adapter_model.safetensors").write_bytes(
                b"synthetic-do-not-publish-weights")
            (foreign / "work").mkdir()
            (foreign / "work" / "adapter.safetensors").write_bytes(
                b"synthetic-unpacked-weight")
            (foreign / "unresolved-link").symlink_to(root / "missing")
            metadata = root / "api.json"
            metadata.write_text(json.dumps({"total_count": len(entries), "artifacts": entries}),
                                encoding="utf-8")
            out = root / "recovery"
            report = recover.recover(root / "artifacts", out, HEAD, RUN,
                                     metadata, "success", "success")
            self.assertEqual(report["validated_raw_prefix_rows"], 2)
            self.assertFalse(report["artifact_transport_complete"])
            for lineage, original in originals.items():
                self.assertEqual(
                    (out / "raw" / lineage / "results.partial.jsonl").read_bytes(),
                    original)
            self.assertFalse((out / "raw/G01/adapter_model.safetensors").exists())
            self.assertFalse((out / "raw/G01/work").exists())
            self.assertFalse((out / "raw/G01/unresolved-link").is_symlink())
            provenance = screen.read_json(
                out / "provenance/source-artifacts.json")
            self.assertEqual(
                provenance["lineages"][0]["source_structure_warnings"],
                [recover.STRUCTURE_WARNING])
            self.assertEqual(
                provenance["lineages"][1]["source_structure_warnings"], [])
            self.assertIn(recover.STRUCTURE_WARNING,
                          provenance["lineages"][0]["custody_warnings"])
            self.assertFalse(recover.verify(out)["artifact_transport_complete"])
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(out)

    def test_dangling_lineage_link_cannot_block_other_recovery(self):
        """Even a broken symlink is not a missing or archived lineage."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "detective005a-G01").symlink_to(root / "missing")
            survivor = artifacts / "detective005a-G02"
            survivor.mkdir()
            original = (
                screen.canonical(self.row(member_id="V-G02-A")) + "\n"
            ).encode("utf-8")
            (survivor / "results.partial.jsonl").write_bytes(original)
            out = root / "recovery"
            report = recover.recover(artifacts, out, HEAD, RUN)
            self.assertEqual(report["validated_raw_prefix_rows"], 1)
            self.assertEqual(
                (out / "raw/G02/results.partial.jsonl").read_bytes(), original)
            provenance = screen.read_json(
                out / "provenance/source-artifacts.json")
            self.assertEqual(
                provenance["lineages"][0]["source_structure_warnings"],
                [recover.STRUCTURE_WARNING])
            self.assertFalse(recover.verify(out)["artifact_transport_complete"])
            with self.assertRaisesRegex(screen.ScreenError,
                                        "incomplete original lineage"):
                recover.require_complete_transport(out)

    def test_rehashed_foreign_file_in_expected_lineage_is_not_authorized(self):
        """A self-edited SHA inventory/provenance cannot legalize extra bytes.

        Previously a forged raw/G01/foreign.bin could be added to source
        provenance and the SHA inventory; readback then recomputed the same
        untrusted file list and still reported transport complete.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "artifacts/detective005a-G01"
            source.mkdir(parents=True)
            original = (screen.canonical(self.row()) + "\n").encode("utf-8")
            (source / "results.partial.jsonl").write_bytes(original)
            api = root / "api.json"
            api.write_text(json.dumps({
                "total_count": 1,
                "artifacts": [{
                    "name": "detective005a-G01",
                    "id": 7001,
                    "size_in_bytes": 100,
                    "digest": "sha256:" + "a" * 64,
                    "expired": False,
                    "workflow_run": {
                        "id": int(RUN), "head_sha": HEAD,
                        "repository_id": 1432134981,
                    },
                }],
            }), encoding="utf-8")
            out = root / "archive"
            report = recover.recover(
                root / "artifacts", out, HEAD, RUN,
                api, "success", "success")
            self.assertTrue(report["artifact_transport_complete"])
            self.assertEqual(recover.verify(out)["validated_raw_prefix_rows"], 1)

            # Simulate a post-promotion archive rewrite that updates BOTH
            # the self-reported file inventory and its checksums.
            foreign = out / "raw/G01/foreign.bin"
            foreign.write_bytes(b"synthetic-foreign-payload-not-original-response")
            source_inventory = out / "provenance/source-artifacts.json"
            provenance = screen.read_json(source_inventory)
            provenance["lineages"][0]["files"].append({
                "name": foreign.name,
                "bytes": foreign.stat().st_size,
                "sha256": screen.sha256_file(foreign),
            })
            provenance["lineages"][0]["files"].sort(key=lambda x: x["name"])
            source_inventory.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            (out / "provenance/hashes.sha256").write_text(
                assemble.hash_inventory(out), encoding="utf-8")
            self.assertEqual(
                (out / "raw/G01/results.partial.jsonl").read_bytes(),
                original)
            with self.assertRaisesRegex(
                    screen.ScreenError, "unapproved recovery file provenance"):
                recover.verify(out)
            with self.assertRaisesRegex(
                    screen.ScreenError, "unapproved recovery file provenance"):
                recover.require_complete_transport(out)

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



    def test_remote_branch_readback_and_main_only_synthetic_push_contract(self):
        """Static guard only; a PASS is not a real GitHub branch-push receipt."""
        root = screen.ROOT / ".github/workflows"
        live_workflow = (root / "detective-005a-live.yml").read_text(
            encoding="utf-8")
        complete = live_workflow.split("  assemble:\n", 1)[1].split(
            "  recover-failed-evidence:\n", 1)[0]
        recovery = live_workflow.split("  recover-failed-evidence:\n", 1)[1]
        for section, program in (
            (complete, "detective_005a_assemble.py"),
            (recovery, "detective_005a_recover.py"),
        ):
            self.assertIn('git push --force-with-lease="refs/heads/$TARGET_BRANCH:" origin', section)
            self.assertIn('git fetch --no-tags origin "refs/heads/$TARGET_BRANCH"', section)
            self.assertIn('git archive FETCH_HEAD', section)
            self.assertIn('"$(git rev-parse FETCH_HEAD)" = "$(git rev-parse HEAD)"',
                          section)
            self.assertIn('remote-readback', section)
            self.assertIn('--verify', section.split('git archive FETCH_HEAD', 1)[1])
            self.assertIn(program, section.split('git archive FETCH_HEAD', 1)[1])
            self.assertLess(section.index('git push --force-with-lease='),
                            section.index('git fetch --no-tags origin'))

        protocol = (root / "detective-005a-protocol.yml").read_text(
            encoding="utf-8")
        self.assertIn("  synthetic-durable-branch-probe:\n", protocol)
        probe = protocol.split("  synthetic-durable-branch-probe:\n", 1)[1]
        self.assertIn("github.event_name == 'workflow_dispatch'", probe)
        self.assertIn("github.ref == 'refs/heads/main'", probe)
        self.assertIn("needs: synthetic-artifact-readback", probe)
        self.assertIn("contents: write", probe)
        self.assertIn("evidence/detective-005a-synthetic-probe-", probe)
        self.assertIn("datasets/detective-005a-synthetic-probe-", probe)
        self.assertIn('git push --force-with-lease="refs/heads/$TARGET_BRANCH:" origin "HEAD:refs/heads/$TARGET_BRANCH"', probe)
        self.assertIn('git archive FETCH_HEAD "$TARGET_DATASET"', probe)
        self.assertIn('--require-complete-transport', probe)
        self.assertIn('"complete_lineage_artifacts"] == 0', probe)
        self.assertIn('"optimizer_steps"] == 0', probe)
        self.assertNotIn("--run --out", probe)
        self.assertNotIn("pip install", probe)
        self.assertNotIn("hf_hub_download", probe)



    @unittest.skipUnless(
        __import__("shutil").which("git") and __import__("shutil").which("tar"),
        "Local Git and tar are required for the no-network transport test",
    )
    def test_local_git_push_fetch_archive_uses_persisted_bytes(self):
        """Actual local bare-Git transfer, not a hosted GitHub persistence claim."""
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = root / "origin.git"
            working = root / "working"
            artifact = root / "artifacts" / "detective005a-G01"
            artifact.mkdir(parents=True)
            journal = artifact / "results.partial.jsonl"
            live.append_completed_raw(journal, self.row())
            # Deliberately preserve a torn suffix, as in an interrupted run.
            with journal.open("ab") as fp:
                fp.write(b'{"torn_tail":')
            original = journal.read_bytes()
            dataset_rel = Path("datasets/detective-005a-local-git-probe")
            dataset = working / dataset_rel
            dataset.parent.mkdir(parents=True)
            recover.recover(root / "artifacts", dataset, HEAD, RUN)
            recover.verify(dataset)

            def git(*args, cwd=working):
                return subprocess.run(
                    ["git", *args], cwd=cwd, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ).stdout.strip()

            git("init", "--bare", str(origin), cwd=root)
            working.mkdir(exist_ok=True)
            git("init", "-b", "main")
            git("config", "user.name", "DETECTIVE no-model fixture")
            git("config", "user.email", "fixture@example.invalid")
            git("add", str(dataset_rel))
            git("commit", "-m", "synthetic interrupted recovery, not model data")
            local_head = git("rev-parse", "HEAD")
            git("remote", "add", "origin", str(origin))
            branch = "refs/heads/evidence/detective-005a-local-git-probe"
            git("push", "--force-with-lease=" + branch + ":", "origin", "HEAD:" + branch)
            git("fetch", "--no-tags", "origin", branch)
            self.assertEqual(local_head, git("rev-parse", "FETCH_HEAD"))

            # Mutate the local working file after the commit; the fetched
            # remote Git object must still contain the exact original bytes.
            (dataset / "raw/G01/results.partial.jsonl").write_bytes(b"changed locally")
            archive = subprocess.run(
                ["git", "archive", "FETCH_HEAD", str(dataset_rel)],
                cwd=working, check=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            extracted = root / "readback"
            extracted.mkdir()
            subprocess.run(
                ["tar", "-xf", "-", "-C", str(extracted)], input=archive,
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            fetched = extracted / dataset_rel
            self.assertEqual(
                (fetched / "raw/G01/results.partial.jsonl").read_bytes(), original)
            self.assertEqual(recover.verify(fetched)["validated_raw_prefix_rows"], 1)

            # A corrupted extracted copy must be refused rather than laundered
            # through the otherwise-valid inventory or remote commit identity.
            fetched_raw = fetched / "raw/G01/results.partial.jsonl"
            fetched_raw.write_bytes(original + b"unrecorded change")
            with self.assertRaisesRegex(screen.ScreenError, "bytes drift"):
                recover.verify(fetched)


    @unittest.skipUnless(__import__("shutil").which("git"),
                         "Git is required for the local create-only lease test")
    def test_evidence_branch_create_only_lease_rejects_raced_fast_forward(self):
        """Reject an existing branch even when a plain push would fast-forward.

        The preliminary ls-remote branch absence check is TOCTOU-prone.
        This tests the Git server's absent-ref lease locally, not on GitHub.
        """
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = root / "origin.git"
            work = root / "work"
            work.mkdir()

            def git(*args, cwd=work, check=True):
                return subprocess.run(
                    ["git", *args], cwd=cwd, check=check,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )

            git("init", "--bare", str(origin), cwd=root)
            git("init", "-b", "main")
            git("config", "user.name", "DETECTIVE create-only fixture")
            git("config", "user.email", "fixture@example.invalid")
            git("remote", "add", "origin", str(origin))
            (work / "fixture").write_text("original\n", encoding="utf-8")
            git("add", "fixture")
            git("commit", "-m", "first")
            first = git("rev-parse", "HEAD").stdout.strip()
            branch = "refs/heads/evidence/detective-005a-create-only-race"
            lease = "--force-with-lease=" + branch + ":"
            git("push", lease, "origin", "HEAD:" + branch)
            self.assertEqual(git("--git-dir=" + str(origin), "rev-parse",
                                 branch, cwd=root).stdout.strip(), first)

            # A competing writer has already created the ref. The next
            # attempted push is a valid fast-forward, but must be refused.
            (work / "fixture").write_text("successor\n", encoding="utf-8")
            git("commit", "-am", "second")
            self.assertEqual(git("merge-base", "--is-ancestor",
                                 first.decode("ascii"), "HEAD").returncode, 0)
            rejected = git("push", lease, "origin", "HEAD:" + branch, check=False)
            self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
            self.assertEqual(git("--git-dir=" + str(origin), "rev-parse",
                                 branch, cwd=root).stdout.strip(), first)



    @unittest.skipUnless(__import__("shutil").which("git"),
                         "Git is required for offline rejected-push test")
    def test_local_push_rejection_and_post_push_ref_drift_fail_closed(self):
        """Simulate receive-pack rejection and a changed ref before readback.

        A local bare Git server is not proof of GitHub Actions permissions,
        retention or cancellation behavior. Both negative outcomes must remain
        distinguishable from a remotely read-back custody success.
        """
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            origin = root / "origin.git"
            work = root / "working"
            artifact = root / "artifacts" / "detective005a-G01"
            artifact.mkdir(parents=True)
            journal = artifact / "results.partial.jsonl"
            live.append_completed_raw(journal, self.row())
            with journal.open("ab") as fp:
                fp.write(b'{"torn_tail":')
            original_raw = journal.read_bytes()
            dataset_rel = Path("datasets/detective-005a-rejected-push-probe")
            dataset = work / dataset_rel
            dataset.parent.mkdir(parents=True)
            recover.recover(root / "artifacts", dataset, HEAD, RUN)
            self.assertEqual(recover.verify(dataset)["validated_raw_prefix_rows"], 1)

            def git(*args, cwd=work, check=True):
                return subprocess.run(
                    ["git", *args], cwd=cwd, check=check,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )

            git("init", "--bare", str(origin), cwd=root)
            work.mkdir(exist_ok=True)
            git("init", "-b", "main")
            git("config", "user.name", "DETECTIVE refused-push fixture")
            git("config", "user.email", "fixture@example.invalid")
            git("remote", "add", "origin", str(origin))
            git("add", str(dataset_rel))
            git("commit", "-m", "synthetic original raw custody")
            expected_head = git("rev-parse", "HEAD").stdout.strip()
            branch = "refs/heads/evidence/detective-005a-push-failure"
            lease = "--force-with-lease=" + branch + ":"

            # Remote rejects creation. Local raw recovery must remain intact;
            # a failed push is NEVER a durable evidence-branch receipt.
            hook = origin / "hooks" / "pre-receive"
            hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            hook.chmod(0o755)
            denied = git("push", lease, "origin", "HEAD:" + branch, check=False)
            self.assertNotEqual(denied.returncode, 0)
            self.assertNotEqual(
                git("--git-dir=" + str(origin), "show-ref", "--verify",
                    "--quiet", branch, cwd=root, check=False).returncode, 0)
            self.assertEqual((dataset / "raw/G01/results.partial.jsonl").read_bytes(),
                             original_raw)
            recover.verify(dataset)

            # Retry succeeds once the denial hook is removed, but a competing
            # writer fast-forwards the ref before our fetch: exact-head
            # readback must reject it even though our push succeeded.
            hook.unlink()
            git("push", lease, "origin", "HEAD:" + branch)
            self.assertEqual(
                git("--git-dir=" + str(origin), "rev-parse", branch,
                    cwd=root).stdout.strip(), expected_head)
            (work / "competing-writer-marker").write_text(
                "another writer updated the branch\n", encoding="utf-8")
            git("add", "competing-writer-marker")
            git("commit", "-m", "competing writer advances ref")
            git("push", "origin", "HEAD:" + branch)
            git("fetch", "--no-tags", "origin", branch)
            fetched_head = git("rev-parse", "FETCH_HEAD").stdout.strip()
            self.assertNotEqual(fetched_head, expected_head)
            self.assertEqual(
                git("--git-dir=" + str(origin), "rev-parse", branch,
                    cwd=root).stdout.strip(), fetched_head)
            self.assertEqual((dataset / "raw/G01/results.partial.jsonl").read_bytes(),
                             original_raw)
            recover.verify(dataset)



if __name__ == "__main__":
    unittest.main()
