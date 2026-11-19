"""Offline adversarial custody tests for the fixed Stage E qualification gates."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import village_001e_assemble as assemble
from village_001e_generation_checkpoint import GenerationCheckpoint


class ProviderCustodyTests(unittest.TestCase):
    def make_lane(self, root):
        lane = root / "lane"
        provider = lane / "provider"
        provider.mkdir(parents=True)
        assemble.write_json(lane / "provider-identity.json", {"source_sha256": "a" * 64})
        for suffix in ("request.json", "provider.json", "attempts.json"):
            (provider / f"F01-first-{suffix}").write_text('{}\n')
        (provider / "F01-first-content.txt").write_bytes(b'malformed "quoted" bytes\r\n')
        (provider / "failed-transport-attempts.json").write_text('[{"status":502}]\n')
        files = {
            path.relative_to(lane).as_posix(): {"sha256": assemble.sha256_file(path), "bytes": path.stat().st_size}
            for path in lane.rglob("*") if path.is_file()
        }
        assemble.write_json(lane / "provider-evidence-hashes.json", {"schema_version": 1, "files": files})
        doc = {
            "provider_identity_sha256": assemble.sha256_file(lane / "provider-identity.json"),
            "provider_call_ids": ["F01-first"],
        }
        return lane, doc

    def test_complete_archive_including_failed_transport_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, doc = self.make_lane(Path(tmp))
            self.assertEqual(assemble.validate_provider_custody(lane, doc), {"source_sha256": "a" * 64})

    def test_changed_raw_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, doc = self.make_lane(Path(tmp))
            (lane / "provider/F01-first-content.txt").write_text('repaired rewrite')
            with self.assertRaisesRegex(RuntimeError, "provider inventory"):
                assemble.validate_provider_custody(lane, doc)

    def test_deleting_failed_transport_is_custody_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, doc = self.make_lane(Path(tmp))
            (lane / "provider/failed-transport-attempts.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "provider inventory"):
                assemble.validate_provider_custody(lane, doc)

    def test_summary_only_archive_rejected_even_with_matching_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, doc = self.make_lane(Path(tmp))
            path = lane / "provider/F01-first-provider.json"
            path.unlink()
            inventory_path = lane / "provider-evidence-hashes.json"
            inventory = json.loads(inventory_path.read_text())
            del inventory["files"]["provider/F01-first-provider.json"]
            assemble.write_json(inventory_path, inventory)
            with self.assertRaisesRegex(RuntimeError, "raw provider file missing"):
                assemble.validate_provider_custody(lane, doc)

    def test_identity_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, doc = self.make_lane(Path(tmp))
            assemble.write_json(lane / "provider-identity.json", {"source_sha256": "b" * 64})
            with self.assertRaisesRegex(RuntimeError, "provider identity SHA"):
                assemble.validate_provider_custody(lane, doc)

    def test_symlink_outside_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane, doc = self.make_lane(root)
            (root / "outside.txt").write_text('not evidence')
            (lane / "provider/link").symlink_to(root / "outside.txt")
            with self.assertRaisesRegex(RuntimeError, "nonregular"):
                assemble.validate_provider_custody(lane, doc)


class TournamentSourceTests(unittest.TestCase):
    def setUp(self):
        pred = {"predicted_lineage": "G01", "predicted_sibling": "A", "lineage_confidence": 2,
                "sibling_confidence": 2, "task_relevance": 1, "genericness": 0}
        self.row = {
            "response": "An awkward but retained failed phenotype.", "adapter_sha256": "a" * 64,
            "behavior_summary": "The subject waits rather than solving the task.",
            "raw_prediction": copy.deepcopy(pred), "behavior_prediction": copy.deepcopy(pred),
            "raw_sibling_prediction": {"predicted_sibling": "A"},
            "behavior_sibling_prediction": {"predicted_sibling": "B"},
        }

    def validate(self):
        assemble.validate_tournament_record(self.row, member={"member_id": "V-G01-A"},
            source_response="An awkward but retained failed phenotype.", adapter_sha="a" * 64, scenario_id="F01")

    def test_low_quality_record_remains_valid_evidence(self):
        self.validate()

    def test_favorable_rewrite_of_source_rejected(self):
        self.row["response"] = "A perfect task solution."
        with self.assertRaisesRegex(RuntimeError, "source response"):
            self.validate()

    def test_stale_adapter_rejected(self):
        self.row["adapter_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "adapter SHA"):
            self.validate()

    def test_nonfinite_boolean_and_out_of_range_scores_rejected(self):
        for value in (float("nan"), float("inf"), True, -1, 5, "4"):
            with self.subTest(value=value):
                self.row["raw_prediction"]["task_relevance"] = value
                with self.assertRaisesRegex(RuntimeError, "integer score"):
                    self.validate()


class EcologyIdentityTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "session_id": "E-P-G01", "kind": "same_lineage_pair", "scenario": {"id": "resource", "text": "fixed situation"},
            "members": ["V-G01-A", "V-G01-B"], "handles": {"M1": "V-G01-B", "M2": "V-G01-A"},
        }
        self.campaign = {"base_model": {"id": "frozen/model", "revision": "c" * 40}}
        self.assets = {member: {"adapter_directory_sha256": str(index) * 64} for index, member in enumerate(self.expected["members"])}
        self.session = {key: value for key, value in self.expected.items() if key != "handles"}
        self.session.update({"handle_member_mapping": self.expected["handles"], "public_handles": ["M1", "M2"],
            "rounds": 3, "base_model": "frozen/model", "base_revision": "c" * 40,
            "adapter_sha256": {key: value["adapter_directory_sha256"] for key, value in self.assets.items()},
            "turns": [{"round": r, "handle": h, "member_id": member, "text": "", "generated_tokens": 1}
                      for r in range(1, 4) for h, member in self.expected["handles"].items()]})
        self.judgment = {key: self.session[key] for key in ("session_id", "kind", "scenario")}
        self.judgment.update({"agents_received_hidden_lineage_prompts": False,
            "lineage_audit_received_expected_lineage_descriptors": False, "lineage_audit": None,
            "metrics": {key: 0 for key in assemble.ECOLOGY_METRICS},
            "agents": [{"handle": handle, "member_id": member, "distinctiveness": 0,
                        "generic_assistant_tendency": 4, "dominance": 0, "deference": 4}
                       for handle, member in self.expected["handles"].items()]})

    def validate(self):
        assemble.validate_ecology_identity(self.session, self.judgment, expected=self.expected,
            campaign=self.campaign, release_assets=self.assets, rounds=3, max_new_tokens=96)

    def test_unhelpful_empty_turns_preserved_as_evidence(self):
        self.validate()

    def test_all_sources_agreeing_on_wrong_session_rejected(self):
        self.session["scenario"] = self.judgment["scenario"] = {"id": "resource", "text": "new easier situation"}
        with self.assertRaisesRegex(RuntimeError, "ecology scenario"):
            self.validate()

    def test_same_members_swapped_handles_rejected(self):
        self.session["handle_member_mapping"] = {"M1": "V-G01-A", "M2": "V-G01-B"}
        with self.assertRaisesRegex(RuntimeError, "handle mapping"):
            self.validate()

    def test_missing_or_out_of_order_turns_rejected(self):
        original = copy.deepcopy(self.session["turns"])
        for turns in (original[:-1], list(reversed(original))):
            self.session["turns"] = turns
            with self.assertRaises(RuntimeError):
                self.validate()

    def test_duplicate_judgment_agent_rejected(self):
        self.judgment["agents"][1] = self.judgment["agents"][0]
        with self.assertRaisesRegex(RuntimeError, "judgment handle/member"):
            self.validate()

    def test_missing_ecology_metric_rejected(self):
        del self.judgment["metrics"]["information_hoarding"]
        with self.assertRaisesRegex(RuntimeError, "metric set"):
            self.validate()


class FinalSuiteBindingTests(unittest.TestCase):
    def test_suite_is_bound_to_exact_D_hashes_and_recipe(self):
        recipe_path = Path(__file__).resolve().parents[1] / "experiments/village-001/final-eval.json"
        recipe = json.loads(recipe_path.read_text())
        # Synthetic hashes deliberately do not create the campaign's final suite.
        kwargs = dict(recipe=recipe, recipe_sha=assemble.sha256_file(recipe_path), summary_sha="a" * 64,
                      release_sha="b" * 64, forbidden=set())
        suite = assemble.build_suite(recipe=recipe, recipe_sha256=kwargs["recipe_sha"],
            preference_summary_sha256=kwargs["summary_sha"], preference_release_sha256=kwargs["release_sha"], forbidden=set())
        assemble.validate_suite_binding(suite, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "exact frozen final suite"):
            assemble.validate_suite_binding(suite, **{**kwargs, "release_sha": "c" * 64})
        suite["scenarios"][0]["prompt"] += " A changed evaluation prompt."
        with self.assertRaisesRegex(RuntimeError, "exact frozen final suite"):
            assemble.validate_suite_binding(suite, **kwargs)


class NativeGenerationCustodyTests(unittest.TestCase):
    def make_lane(self, root):
        lane = root / "lane"
        cp = GenerationCheckpoint(out=lane, resume=None, identity={"sources": {"final_eval": "f" * 64}},
            keys=["F01", "F02"], terminal_name="result.json", validate_row=lambda *_: None)
        for key in ("F01", "F02"):
            cp.append(key, {"scenario_id": key, "response": "Retained imperfect response.",
                "input_token_ids": [1, 2], "generated_token_ids": [3], "generated_tokens": 1,
                "rendered_prompt": "A native prompt."})
        result = {"responses": cp.rows}
        cp.finish(result, "responses")
        return lane, result

    def validate(self, lane, result):
        return assemble.validate_generation_custody(lane, result, row_field="responses", keys=["F01", "F02"],
            maximum=160, sources={"final_eval": "f" * 64})

    def test_complete_native_custody_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, result = self.make_lane(Path(tmp))
            self.validate(lane, result)

    def test_terminal_rewrite_rejected_against_native_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, result = self.make_lane(Path(tmp))
            result["responses"][0]["response"] = "An improved but fabricated response."
            with self.assertRaisesRegex(RuntimeError, "generation row row"):
                self.validate(lane, result)

    def test_missing_native_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, result = self.make_lane(Path(tmp))
            (lane / "rows/001-F01.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "generation row names"):
                self.validate(lane, result)

    def test_corrupted_preceding_chain_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane, result = self.make_lane(Path(tmp))
            path = lane / "rows/002-F02.json"
            payload = json.loads(path.read_text())
            payload["previous_checkpoint_sha256"] = "a" * 64
            assemble.write_json(path, payload)
            with self.assertRaisesRegex(RuntimeError, "previous_checkpoint_sha256"):
                self.validate(lane, result)


if __name__ == "__main__":
    unittest.main()
