"""Offline custody regressions for recovering the original trained Village."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import village_001d_assemble as assemble


class RecoveredParentTests(unittest.TestCase):
    def setUp(self):
        self.member = "V-G01-A"
        self.contract = "a" * 64
        self.parent_sha = "b" * 64
        self.response = "The exact existing SFT response."
        self.asset = {
            "member_id": self.member,
            "adapter_directory_sha256": self.parent_sha,
        }
        self.parent = {
            "experiment": "VILLAGE-001C",
            "member_id": self.member,
            "member_contract_sha256": self.contract,
            "adapter": {"sha256": self.parent_sha},
            "evaluation": {"responses": [
                {"scenario_id": "T01", "response": self.response},
            ]},
            "member_probe_evaluation": {"responses": []},
        }
        self.records = {
            label: {
                "member_id": self.member,
                "member_contract_sha256": self.contract,
                "parent_adapter_sha256": self.parent_sha,
            }
            for label in ("feedback", "pairs", "preference", "comparison")
        }
        self.feedback = [{
            "member_id": self.member,
            "scenario_id": "T01",
            "parent_adapter_sha256": self.parent_sha,
            "provenance": {
                "member_contract_sha256": self.contract,
                "sft_scenario_response_sha256": hashlib.sha256(
                    self.response.encode()
                ).hexdigest(),
            },
        }]
        self.pairs = [{
            "member_id": self.member,
            "scenario_id": "T01",
            "source_parent_response": self.response,
            "provenance": {
                "member_contract_sha256": self.contract,
                "parent_adapter_sha256": self.parent_sha,
            },
        }]

    def validate(self):
        assemble.validate_recovered_parent(
            self.member, self.contract, self.asset, self.parent,
            self.records, self.feedback, self.pairs,
        )

    def test_current_parent_bound_lane_passes(self):
        self.validate()

    def test_internally_consistent_stale_lane_is_rejected(self):
        for row in self.records.values():
            row["parent_adapter_sha256"] = "c" * 64
        self.feedback[0]["parent_adapter_sha256"] = "c" * 64
        self.pairs[0]["provenance"]["parent_adapter_sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "current SFT parent"):
            self.validate()

    def test_current_release_and_result_must_agree(self):
        self.parent["adapter"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "result/release mismatch"):
            self.validate()

    def test_every_stage_requires_current_parent_and_contract(self):
        original = copy.deepcopy(self.records)
        for label in self.records:
            for field in ("parent_adapter_sha256", "member_contract_sha256", "member_id"):
                with self.subTest(label=label, field=field):
                    self.records = copy.deepcopy(original)
                    self.records[label][field] = "wrong"
                    with self.assertRaisesRegex(RuntimeError, "current SFT parent"):
                        self.validate()

    def test_feedback_binds_exact_current_source_response(self):
        self.parent["evaluation"]["responses"][0]["response"] += " changed"
        with self.assertRaisesRegex(RuntimeError, "feedback row"):
            self.validate()

    def test_feedback_row_parent_substitution_is_rejected(self):
        self.feedback[0]["parent_adapter_sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "feedback row"):
            self.validate()

    def test_pair_parent_or_source_response_substitution_is_rejected(self):
        original = copy.deepcopy(self.pairs)
        for field in ("parent", "response", "scenario"):
            with self.subTest(field=field):
                self.pairs = copy.deepcopy(original)
                if field == "parent":
                    self.pairs[0]["provenance"]["parent_adapter_sha256"] = "c" * 64
                elif field == "response":
                    self.pairs[0]["source_parent_response"] = "substituted"
                else:
                    self.pairs[0]["scenario_id"] = "P99"
                with self.assertRaisesRegex(RuntimeError, "pair row"):
                    self.validate()

    def test_stale_preflight_keeps_existing_outputs_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "final_release_tag": "village-001-preference-v1",
                "parent_release_tag": "village-001-sft-v1",
                "campaign_config": "campaign.json",
                "sft_evidence_root": "sft",
                "feedback": {"scenario_count": 24, "member_probe_count": 8, "workers_per_scenario": 3},
            }
            assemble.write_json(root / "config.json", config)
            assemble.write_json(root / "campaign.json", {"archive": {"dataset_root": "dataset"}})
            assemble.write_json(root / "dataset/manifest.json", {
                "stage": "VILLAGE-001B-member-corpora",
                "member_contract_sha256": self.contract,
            })
            assemble.write_json(root / "sft/adapter-release.json", {
                "assets": [self.asset], "release_tag": config["parent_release_tag"],
                "member_contract_sha256": self.contract,
            })
            assemble.write_json(root / f"sft/results/{self.member}.json", self.parent)
            stages = {
                "feedback": ("manifest.json", "sprocket-feedback"),
                "pairs": ("manifest.json", "preference-pairs"),
                "preference": ("result.json", "preference-continuation"),
                "comparison": ("evaluation.json", "before-after-evaluation"),
            }
            for label, (filename, stage) in stages.items():
                value = {
                    **self.records[label], "experiment": "VILLAGE-001D",
                    "stage": stage, "parent_adapter_sha256": "c" * 64,
                }
                assemble.write_json(root / label / self.member / filename, value)
            for label, name, rows in (
                ("feedback", "feedback.jsonl", self.feedback),
                ("pairs", "pairs.jsonl", self.pairs),
            ):
                (root / label / self.member / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
            for directory in ("evidence", "release"):
                (root / directory).mkdir()
                (root / directory / "keep.txt").write_bytes(b"existing authority\x00\xff")
            argv = [
                "assemble", "--feedback-artifacts", str(root / "feedback"),
                "--pair-artifacts", str(root / "pairs"),
                "--preference-artifacts", str(root / "preference"),
                "--comparison-artifacts", str(root / "comparison"),
                "--config", str(root / "config.json"),
                "--evidence-out", str(root / "evidence"),
                "--release-out", str(root / "release"),
            ]
            population = {
                "member_contract_sha256": self.contract,
                "members": [{"member_id": self.member}],
            }
            with mock.patch.object(assemble, "ROOT", root), mock.patch.object(
                assemble, "build_manifest", return_value=population
            ), mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(RuntimeError, "current SFT parent"):
                    assemble.main()
            for directory in ("evidence", "release"):
                self.assertEqual(
                    (root / directory / "keep.txt").read_bytes(),
                    b"existing authority\x00\xff",
                )


class ComparisonCustodyTests(unittest.TestCase):
    def source(self, root):
        source = root / "source"
        provider = source / "provider"
        provider.mkdir(parents=True)
        (source / "evaluation.json").write_text('{"member_id":"V-G01-A"}\n')
        (provider / "batch-request.json").write_bytes(b'{"prompt":"exact"}\n')
        (provider / "batch-provider.json").write_bytes(b'{"raw":"provider"}\n')
        (provider / "batch-content.txt").write_bytes(b'{"malformed":"unescaped "quote""}\r\n')
        (provider / "failed-attempt.bin").write_bytes(b"\x00\xff\x01")
        (source / "recovery.json").write_text('{"reused":true}\n')
        return source

    def test_complete_raw_tree_and_failure_bytes_are_copied_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            destination = root / "evidence/before-after-evidence/V-G01-A"
            assemble.copy_comparison_evidence(source, destination)
            self.assertEqual(assemble.directory_hash(source), assemble.directory_hash(destination))
            inventory = assemble.hash_inventory(root / "evidence", {"hashes.sha256"})
            for path in source.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(source)
                    self.assertEqual(path.read_bytes(), (destination / relative).read_bytes())
                    self.assertIn(f"{assemble.sha256_file(path)}  before-after-evidence/V-G01-A/{relative.as_posix()}", inventory)

    def test_summary_without_raw_requests_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            (source / "provider/batch-request.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "provider evidence missing"):
                assemble.copy_comparison_evidence(source, root / "output")
            self.assertFalse((root / "output").exists())

    def test_symlink_is_rejected_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            (root / "external").write_text("not artifact evidence")
            (source / "provider/link").symlink_to(root / "external")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                assemble.copy_comparison_evidence(source, root / "output")
            self.assertFalse((root / "output").exists())


class RecoveryReportTests(unittest.TestCase):
    def test_feedback_denominator_comes_from_actual_count(self):
        row = {
            "member_id": "V-G01-A", "valid_feedback": 0, "feedback_total": 96,
            "preference_margin_before": 0, "preference_margin_after": 1,
            "preference_accuracy_before": 0, "preference_accuracy_after": 1,
            "task_relevance_delta": 0, "behavior_fit_delta": 0,
            "sibling_accuracy_delta": 0, "qualified_for_population_tournament": True,
        }
        self.assertIn("| 0/96 |", assemble.member_report_row(row))
        self.assertNotIn("/72", assemble.member_report_row(row))

    def test_release_override_does_not_change_frozen_config(self):
        config = {"final_release_tag": "village-001-preference-v1"}
        original = copy.deepcopy(config)
        self.assertEqual(assemble.select_release_tag(config, None), original["final_release_tag"])
        self.assertEqual(assemble.select_release_tag(config, "village-recovery-40613393744"), "village-recovery-40613393744")
        self.assertEqual(config, original)

    def test_unsafe_release_tags_are_rejected(self):
        for tag in ("", "../main", "refs/heads/main", "-option", "two words", "tag\nother"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                assemble.select_release_tag({"final_release_tag": "default"}, tag)


if __name__ == "__main__":
    unittest.main()
