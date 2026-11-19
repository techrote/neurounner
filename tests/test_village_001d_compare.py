"""Offline recovery tests for source-bound VILLAGE-001D blind comparisons."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import village_001d_compare as compare


class ComparisonRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.provider = self.root / "provider"
        self.member = {"member_id": "V-G12-B", "lineage": "G12", "sibling": "B"}
        self.packet = {"persona_summary": "A synthetic local repairer."}
        self.model = "synthetic-critic"
        self.parent = [dict(scenario_id=f"T{i:02d}", family="synthetic",
                            prompt=f"Scenario {i}", response=f"Parent response {i}")
                       for i in range(1, 25)]
        self.child = [{**r, "response": r["response"].replace("Parent", "Child")}
                      for r in self.parent]
        self.cases = compare.build_cases(self.member, self.parent, self.child)

    def side(self, score=2):
        return {**{key: score for key in compare.SCORE_KEYS},
                "behavior_summary": "Makes a concrete local repair.",
                "sibling_prediction": "B", "reason": "A concrete rationale."}

    def reviews(self, cases, score=2):
        return [{"scenario_id": case["scenario_id"],
                 "X": self.side(score), "Y": self.side(score)} for case in cases]

    def archive(self, cases, label, raw=None, *, repair=False):
        payload = compare.request_payload(self.member, self.packet, cases,
                                          self.model, repair=repair)
        content = raw if raw is not None else json.dumps({"reviews": self.reviews(cases)})
        compare.write_json(self.provider / f"{label}-request.json", {
            "provider": "nvidia",
            "role": "before-after-critic-targeted-repair" if repair else "before-after-critic",
            "payload": payload,
        })
        compare.write_json(self.provider / f"{label}-provider.json", {
            "choices": [{"message": {"content": content}}]})
        (self.provider / f"{label}-content.txt").write_text(content + "\n")
        return content

    def label(self, index):
        return f"V-G12-B-before-after-{index:02d}-bolt"

    def replay(self, **kwargs):
        return compare.replay_provider_history(self.provider, self.member,
                                               self.packet, self.cases, self.model,
                                               **kwargs)

    def collect(self, **kwargs):
        return compare.collect_reviews(member=self.member, packet=self.packet,
                                       cases=self.cases, model_id=self.model,
                                       provider=self.provider,
                                       endpoint="https://invalid.example/v1", key="synthetic",
                                       attempts_limit=2, **kwargs)

    def test_partial_json_salvages_whole_reviews_without_repairing_bad_record(self):
        rows = self.reviews(self.cases[6:12])
        rows[3]["X"]["reason"] = 'Waiting for "other shoe to drop".'
        raw = json.dumps({"reviews": rows}, indent=2).replace('\\"other shoe to drop\\"',
                                                            '"other shoe to drop"')
        actual, problems = compare.parse_reviews_partial(raw, self.cases[6:12])
        self.assertEqual(set(actual), {"T07", "T08", "T09"})
        self.assertIn("review_missing:T10,T11,T12", problems)
        self.assertNotIn("T10", actual)
        legacy, _ = compare.parse_reviews_partial(raw, self.cases[6:12], salvage_partial=False)
        self.assertEqual(legacy, {})

    def test_partial_json_does_not_salvage_arbitrary_json_without_review_container(self):
        raw = json.dumps(self.reviews(self.cases[:1])[0], indent=2)
        self.assertEqual(compare.parse_reviews_partial(raw, self.cases[:1])[0], {})

    def test_broken_array_never_salvages_outside_or_nested_review_objects(self):
        row = self.reviews(self.cases[:1])[0]
        outside = json.dumps(row, indent=2)
        outside_wrapper = json.dumps({"reviews": [row]}, indent=2)
        for raw in (
            '{"reviews": [BROKEN]}\n' + outside + ',\n',
            '{"reviews": [BROKEN]}\n' + outside_wrapper,
            '{"reviews": BROKEN}\n' + outside_wrapper,
            '{"ignored": true, "reviews": [BROKEN]}\n' + outside_wrapper,
            'not JSON\n' + outside_wrapper,
            '{"reviews": [ {"broken": BROKEN, "nested": ' + outside + '} ]}',
            '{"reviews": [' + json.dumps({"nested": row}) + ', BROKEN]}',
            '{"reviews": [BROKEN, ' + outside + ']}',
        ):
            with self.subTest(raw=raw):
                parsed, _ = compare.parse_reviews_partial(raw, self.cases[:1])
                self.assertEqual(parsed, {})

    def test_prefix_salvage_stops_at_closure_or_ambiguous_truncation(self):
        first, second = self.reviews(self.cases[:2])
        a, b = json.dumps(first), json.dumps(second)
        raw = '{"reviews": [' + a + ', BROKEN]}\n' + b + ',\n'
        parsed, _ = compare.parse_reviews_partial(raw, self.cases[:2])
        self.assertEqual(set(parsed), {"T01"})
        for raw in ('{"reviews": [' + a, '{"reviews": [' + a[:-1],
                    '{"reviews": [' + a + ' ' + b + ']}'):
            with self.subTest(raw=raw):
                parsed, _ = compare.parse_reviews_partial(raw, self.cases[:2])
                self.assertEqual(parsed, {})

    def test_strict_side_thresholds_survive_salvage(self):
        for bad in (True, 5, -1, 1.0, "2"):
            row = self.reviews(self.cases[:1])[0]
            row["X"]["task_relevance"] = bad
            parsed, _ = compare.parse_reviews_partial(json.dumps({"reviews": [row]}), self.cases[:1])
            self.assertEqual(parsed, {})

    def test_legacy_transcripts_replay_with_exact_payload_binding(self):
        self.archive(self.cases[:6], self.label(1))
        reviews, requested, generation = self.replay()
        self.assertEqual(len(reviews), 6)
        self.assertEqual(requested, {self.label(1)})
        self.assertEqual(generation, 0)
        for field in ("prompt", "X", "Y"):
            with self.subTest(field=field):
                drifted = copy.deepcopy(self.cases)
                drifted[0][field] += " changed"
                with self.assertRaisesRegex(RuntimeError, "request/source mismatch"):
                    compare.replay_provider_history(self.provider, self.member, self.packet,
                                                    drifted, self.model)

    def test_replay_refuses_changed_model_or_content_custody(self):
        self.archive(self.cases[:6], self.label(1))
        with self.assertRaisesRegex(RuntimeError, "request/source mismatch"):
            compare.replay_provider_history(self.provider, self.member, self.packet,
                                            self.cases, "different-model")
        (self.provider / f"{self.label(1)}-content.txt").write_text('{"reviews": []}')
        with self.assertRaisesRegex(RuntimeError, "content/provider mismatch"):
            self.replay()

    def test_replay_preserves_first_valid_judgment(self):
        self.archive(self.cases[:6], self.label(1))
        changed = json.dumps({"reviews": self.reviews(self.cases[:1], score=4)})
        self.archive(self.cases[:1], self.label(1) + "-targeted-repair1", changed, repair=True)
        self.assertEqual(self.replay()[0]["T01"]["X"]["task_relevance"], 2)

    def test_complete_recovery_never_calls_provider(self):
        for index in range(4):
            self.archive(self.cases[index*6:(index+1)*6], self.label(index+1))
        with mock.patch.object(compare, "call_api", side_effect=AssertionError("must stay offline")):
            rows = self.collect(offline=True)
        self.assertEqual(len(rows), 24)
        receipt = json.loads((self.root / "recovery.json").read_text())
        self.assertEqual(receipt["new_provider_requests"], 0)
        self.assertEqual(receipt["recovered_reviews"], 24)

    def test_offline_partial_recovery_preserves_evidence_and_reports_missing(self):
        self.archive(self.cases[:6], self.label(1))
        original = {p.name: p.read_bytes() for p in self.provider.iterdir()}
        with mock.patch.object(compare, "call_api", side_effect=AssertionError("must stay offline")):
            with self.assertRaisesRegex(RuntimeError, "T07,T08"):
                self.collect(offline=True)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.provider.iterdir()})
        receipt = json.loads((self.root / "recovery.json").read_text())
        self.assertEqual(len(receipt["missing_scenario_ids"]), 18)

    def test_missing_only_repair_shrinks_batches_and_keeps_original_bytes(self):
        self.archive(self.cases[:6], self.label(1), '{"reviews": []}')
        for index in range(1,4):
            self.archive(self.cases[index*6:(index+1)*6], self.label(index+1))
        original = {p.name: p.read_bytes() for p in self.provider.iterdir()}
        requested = []
        def fake_api(**kwargs):
            cases = json.loads(kwargs["payload"]["messages"][1]["content"].rsplit("CASES:\n", 1)[1])
            requested.append([c["scenario_id"] for c in cases])
            raw = json.dumps({"reviews": self.reviews(cases) if len(cases) == 1 else []})
            self.archive(cases, kwargs["label"], raw, repair=True)
            return raw, {}, []
        with mock.patch.object(compare, "call_api", side_effect=fake_api):
            rows = self.collect()
        self.assertEqual([len(cases) for cases in requested], [3,3,2,2,2,1,1,1,1,1,1])
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(set(ids) <= {f"T{i:02d}" for i in range(1,7)} for ids in requested))
        for name, content in original.items():
            self.assertEqual((self.provider / name).read_bytes(), content)
        self.assertEqual(len(self.replay()[0]), 24)

    def test_repair_budget_is_bounded(self):
        self.archive(self.cases[:6], self.label(1), '{"reviews": []}')
        with mock.patch.object(compare, "call_api", return_value=('{"reviews": []}', {}, [])) as call:
            with self.assertRaisesRegex(ValueError, "incomplete after targeted repair"):
                self.collect()
        self.assertEqual(call.call_count, 11)
        for value in (0,4,True,1.0):
            with self.assertRaises(ValueError):
                compare.repair_batch_size(value)

    def test_resume_identity_drift_fails_before_copy_or_overwrite(self):
        source, out = self.root / "source", self.root / "out"
        identity = {"member_id": "V-G12-B", "child_sha": "a"}
        compare.write_json(source / "comparison-identity.json", identity)
        (source / "original.txt").write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            compare.prepare_output(out, source, {**identity, "child_sha": "b"})
        self.assertFalse(out.exists())
        compare.prepare_output(out, source, identity)
        self.assertEqual((out / "original.txt").read_text(), "keep")
        with self.assertRaisesRegex(RuntimeError, "output must be empty"):
            compare.prepare_output(out, None, identity)
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            compare.prepare_output(source, source, identity)

    def test_duplicate_response_ids_are_not_silently_overwritten(self):
        with self.assertRaisesRegex(RuntimeError, "count must be exactly 24"):
            compare.build_cases(self.member, self.parent + [self.parent[0]], self.child)

    def test_scoring_thresholds_and_orientation_remain_unchanged(self):
        rows = self.reviews(self.cases)
        for case,row in zip(self.cases,rows):
            child = "Y" if case["_orientation"] == "parent-x" else "X"
            row[child]["task_relevance"] = 1
        result = compare.build_evaluation(self.member, self.cases, rows, contract_sha="contract",
                                         parent_sha="parent", child_sha="child", model_id=self.model)
        self.assertEqual(result["delta"]["task_relevance"], -1)
        self.assertFalse(result["gates"]["qualified_for_population_tournament"])
        for case,row in zip(self.cases,rows):
            child = "Y" if case["_orientation"] == "parent-x" else "X"
            row[child]["task_relevance"] = 2
            row[child]["lineage_fit"] = 3
            row[child]["behavior_fit"] = 1
        result = compare.build_evaluation(self.member, self.cases, rows, contract_sha="contract",
                                         parent_sha="parent", child_sha="child", model_id=self.model)
        self.assertTrue(result["gates"]["style_only_regression_detected"])
        self.assertFalse(result["gates"]["qualified_for_population_tournament"])


if __name__ == "__main__":
    unittest.main()
