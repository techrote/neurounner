"""No-network regressions for blind F scoring, source custody and bounded replay."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stderr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import village_001f_judge as j
from village_001e_generation_checkpoint import GenerationCheckpoint
from village_001e_provider_checkpoint import prepare_output, write_once


def worker_score(value=2):
    return {key: (1 if key == "genericness" else value) for key in j.WORKER_SCORES}


def fingerprint(seed="a"):
    row = {"dtype": "torch.float32", "shape": [1], "sha256": seed * 64}
    return {"safetensors_file_sha256": seed * 64,
            "canonical_tensor_sha256": hashlib.sha256(j.json_bytes({"name": "fixture.weight", **row})).hexdigest(),
            "tensor_count": 1, "tensors": {"fixture.weight": row}}


def execution_result():
    return {"runner": {"cpu_model": "Fixture CPU", "cpu_count": 4, "platform": "Fixture Linux"},
        "packages": {"python": "3.12.3", **j.worker.RUNTIME},
        "recovery": {"resumed": False, "resumed_from_step": 0}}


def arm_order(member):
    return list(j.worker.MASKS if member in j.worker.HISTORICAL_FIRST else reversed(j.worker.MASKS))


def invocation_execution(member, arm):
    position = arm_order(member).index(arm) + 1
    return {"repository": "techrote/neurounner", "run_id": "123", "run_attempt": "1", "job": "train",
        "member_id": member, "paired_job_nonce": "fixture-job-" + member, "source_head": "a" * 40,
        "runner_name": "fixture-runner-" + member, "boot_id": "fixture-boot-" + member,
        "hostname": "fixture-host-" + member, **execution_result()["runner"],
        "packages": execution_result()["packages"], "process_id": 1000 + position,
        "started_at_utc": f"2026-11-19T00:00:0{position}+00:00", "started_monotonic_ns": position * 200 - 100}


def fresh_provenance(member, arm):
    execution = invocation_execution(member, arm)
    start = {"member_id": member, "mask_mode": arm, "arm_order": arm_order(member),
        "arm_position": arm_order(member).index(arm) + 1, "resume_requested": False,
        "source_kind": "fresh", "start_step": 0, "execution": execution}
    finish = {"status": "completed", "end_step": 120, "finished_monotonic_ns": execution["started_monotonic_ns"] + 100}
    return {"member_id": member, "mask_mode": arm, "invocations": [
        {"ordinal": 1, "start": start, "finish": finish, "checkpoints": [], "retained_end_step": 120}],
        "optimizer_segments": [{"invocation_ordinal": 1, "start_step": 0, "end_step": 120,
            "steps": 120, "execution": execution}], "unattributed_optimizer_segments": [],
        "zero_step_copies": [], "unfinished_invocations": [], "resume_invocations": [],
        "retained_optimizer_steps": 120, "files": {}}


def write_archive(kw, answer):
    out, label = kw["outdir"], kw["label"]
    write_once(out / f"{label}-request.json", {"provider": "nvidia", "role": kw["role"], "payload": kw["payload"]})
    write_once(out / f"{label}-provider.json", {"choices": [{"message": {"content": answer}}]})
    write_once(out / f"{label}-attempts.json", [{"status": 200, "attempt": 1}])
    (out / f"{label}-content.txt").write_text(answer + "\n")
    return answer, {}, []


def fixture_api(**kw):
    items = json.loads(kw["payload"]["messages"][1]["content"].rsplit("ITEMS:\n", 1)[1])
    rows = []
    for item in items:
        if "primary" in kw["role"]:
            row = {"id": item["id"], "A": worker_score(), "B": worker_score(), "winner": "tie", "reason": "Both attempt the concrete task."}
        elif "context" in kw["role"]:
            row = {"id": item["id"], **worker_score(), "reason": "A partly useful concrete action."}
        else:
            row = {"id": item["id"], **{key: 1 for key in j.SPROCKET_SCORES}, "unsupported_claims": False,
                   "source_quote": "", "reason": "References a visible but underspecified action."}
        rows.append(row)
    return write_archive(kw, json.dumps({"reviews": rows}))


def synthetic_population():
    suite = {"scenarios": [{"id": f"Q{i:02d}", "family": f"family-{(i-1)//2}", "prompt": f"Situation {i}."} for i in range(1, 13)]}
    wcfg = {"members": j.worker.PANEL, "primary_analysis": {"working_effect_threshold": .5,
        "minimum_members_with_positive_mean_gain": 4, "maximum_member_mean_relevance_regression": .5}}
    workers, training = {}, {}
    for member, stage in [(m, s) for m in wcfg["members"] for s in j.STAGES] + [(None, "BASE")]:
        rows = [{"scenario_id": s["id"], "family": s["family"], "prompt": s["prompt"], "response": f"A concrete action for {s['id']}."} for s in suite["scenarios"]]
        workers[(member, stage)] = {"rows": rows, "result": {"adapter_directory_sha256": "a" * 64, "adapter_weight_fingerprint": fingerprint()}}
        if stage in ("D_REPRODUCED", "D_CORRECTED"):
            result, historical = execution_result(), execution_result()
            arm = next(arm for arm, arm_stage in j.ARM_STAGE.items() if arm_stage == stage)
            training[(member, stage)] = {"result": result, "original_D_result": historical,
                "paired_arm_order": arm_order(member), "execution_provenance": fresh_provenance(member, arm),
                "receipt": {"adapter_directory_sha256": "a" * 64, "adapter_weight_fingerprint": fingerprint(),
                    "training_result_sha256": j.sha256_value(result), "original_D_reference_result_sha256": j.sha256_value(historical)}}
    sp, mapping, source_rows = [], {}, []
    for case in range(1, 7):
        for seed in range(1, 4):
            for arm in ("original", "compact_schema_first"):
                key = f"case-{case}-{seed}-{arm}"
                ident = "B-" + hashlib.sha256(key.encode()).hexdigest()[:14]
                private = {"key": key, "case_id": f"case-{case}", "worker": seed, "arm": arm}
                mapping[ident] = private
                sp.append({"id": ident, "situation": "A device has failed.", "villager_response": "Repair the device.", "feedback": "The repair action is underspecified."})
                source_rows.append({"id": ident, **private, "legacy_schema_valid": arm == "compact_schema_first",
                    "strict_json_schema_valid": arm == "compact_schema_first", "empty_response": False})
    blinded, private = j.make_blind_inputs(workers, wcfg, suite, {"items": sorted(sp, key=lambda r: r["id"]), "mapping": mapping})
    return workers, training, wcfg, blinded, private, source_rows


class PromptCoverageTests(unittest.TestCase):
    def rows(self, kind, ids):
        fields = {"primary": {"A": "", "B": ""}, "context": {"response": ""},
                  "sprocket": {"villager_response": "", "feedback": ""}}[kind]
        return [{"id": ident, "situation": "A device has failed.", **fields} for ident in ids]

    def test_all_families_initial_and_repair_name_exact_actual_ids_and_count(self):
        requested = {"primary": ["P037", "P052"], "context": ["R085", "R143"],
                     "sprocket": ["B-a97b", "B-f31c"]}
        for kind, ids in requested.items():
            for repair in (False, True):
                with self.subTest(kind=kind, repair=repair):
                    selected = ids[1:] if repair else ids
                    rows = self.rows(kind, selected)
                    instructions, serialized = j.prompt(kind, rows, repair).rsplit("\nITEMS:\n", 1)
                    self.assertEqual(json.loads(serialized), rows)
                    actual_ids = json.loads(instructions.split("\nRequired IDs: ", 1)[1].splitlines()[0])
                    self.assertEqual(actual_ids, selected)
                    self.assertIn(f"exactly {len(selected)} review objects", instructions)
                    self.assertIn("every listed ID exactly once", instructions)
                    self.assertIn("Never return an empty reviews array", instructions)
                    example, _ = json.JSONDecoder().raw_decode(instructions.split("Return JSON only: ", 1)[1])
                    self.assertEqual(example["reviews"][0]["id"], selected[0])
                    self.assertEqual("Missing-only repair:" in instructions, repair)

    def test_empty_text_items_remain_present_and_use_the_existing_rubric(self):
        for kind in ("primary", "context", "sprocket"):
            with self.subTest(kind=kind):
                rows = self.rows(kind, ["actual-empty-item"])
                instructions, serialized = j.prompt(kind, rows, False).rsplit("\nITEMS:\n", 1)
                self.assertEqual(json.loads(serialized), rows)
                self.assertIn("Do not skip an item because any response or feedback text is empty; grade it using the rubric above.", instructions)
                if kind == "sprocket":
                    self.assertIn("Empty feedback earns zero on all three metrics and unsupported_claims=false.", instructions)
                else:
                    self.assertTrue(instructions.startswith(j.WORKER_RUBRIC))

    def test_ambiguous_or_empty_id_sets_fail_before_building_a_request(self):
        for rows in ([], [{"id": "same"}, {"id": "same"}], [{"id": ""}], [{"id": None}], [{}]):
            with self.subTest(rows=rows), self.assertRaisesRegex(RuntimeError, "judge prompt requires"):
                j.prompt("context", rows, False)


class JudgeValidationTests(unittest.TestCase):
    def paired(self, a="same", b="same"):
        return {"id": "P001", "A": worker_score(), "B": worker_score(), "winner": "tie", "reason": "Both show the same action."}, {"P001": {"A": a, "B": b}}

    def test_identical_text_cannot_create_false_gain(self):
        row, items = self.paired()
        j.validate_reviews(json.dumps({"reviews": [row]}), ["P001"], kind="primary", items=items)
        row["B"]["task_relevance"] = 1
        with self.assertRaisesRegex(ValueError, "identical paired"):
            j.validate_reviews(json.dumps({"reviews": [row]}), ["P001"], kind="primary", items=items)

    def test_identical_text_cannot_receive_non_tie_preference(self):
        row, items = self.paired()
        row["winner"] = "A"
        with self.assertRaisesRegex(ValueError, "identical paired"):
            j.validate_reviews(json.dumps({"reviews": [row]}), ["P001"], kind="primary", items=items)

    def test_different_text_can_have_different_valid_scores(self):
        row, items = self.paired("Do useful work.", "Do nothing.")
        row["B"]["task_relevance"] = 0
        row["winner"] = "A"
        result = j.validate_reviews(json.dumps({"reviews": [row]}), ["P001"], kind="primary", items=items)
        self.assertEqual(result["P001"]["B"]["task_relevance"], 0)

    def test_boolean_nonfinite_and_out_of_range_scores_fail(self):
        for bad in (True, 5, -1, float("nan"), "3"):
            with self.subTest(value=bad):
                row, items = self.paired()
                row["A"]["groundedness"] = bad
                with self.assertRaises(ValueError):
                    j.validate_reviews(json.dumps({"reviews": [row]}), ["P001"], kind="primary", items=items)

    def test_maximum_grounding_requires_actual_source_quote(self):
        items = {"B-one": {"villager_response": "Repair the device.", "feedback": "Repair is a concrete act."}}
        row = {"id": "B-one", "response_grounding": 2, "scenario_grounding": 2, "corrective_usefulness": 1,
               "unsupported_claims": False, "source_quote": "invented action", "reason": "A concrete source-grounded observation."}
        with self.assertRaisesRegex(ValueError, "verbatim"):
            j.validate_reviews(json.dumps({"reviews": [row]}), ["B-one"], kind="sprocket", items=items)
        row["source_quote"] = "Repair the device."
        j.validate_reviews(json.dumps({"reviews": [row]}), ["B-one"], kind="sprocket", items=items)

    def test_empty_feedback_cannot_be_reported_as_grounded(self):
        items = {"B-one": {"villager_response": "Repair the device.", "feedback": ""}}
        row = {"id": "B-one", **{key: 1 for key in j.SPROCKET_SCORES}, "unsupported_claims": False,
               "source_quote": "", "reason": "Nothing is present to evaluate."}
        with self.assertRaisesRegex(ValueError, "empty feedback"):
            j.validate_reviews(json.dumps({"reviews": [row]}), ["B-one"], kind="sprocket", items=items)

    def test_weight_fingerprint_detects_tampered_tensor_metadata(self):
        value = fingerprint()
        j.validate_fingerprint(value)
        value["tensors"]["fixture.weight"]["shape"] = [2]
        with self.assertRaisesRegex(RuntimeError, "canonical adapter"):
            j.validate_fingerprint(value)


class BlindPlanAndAggregationTests(unittest.TestCase):
    def setUp(self):
        self.workers, self.training, self.wcfg, self.blinded, self.mapping, self.sp = synthetic_population()

    def ratings(self):
        return {"primary": {row["id"]: {"A": worker_score(), "B": worker_score(), "winner": "tie", "reason": "Same concrete action."} for row in self.blinded["primary"]},
                "context": {row["id"]: {**worker_score(), "reason": "A concrete action."} for row in self.blinded["context"]},
                "sprocket": {row["id"]: {**{key: 0 for key in j.SPROCKET_SCORES}, "unsupported_claims": False,
                    "source_quote": "", "reason": "No useful grounded criticism."} for row in self.blinded["sprocket"]}}

    def aggregate(self, ratings):
        return j.aggregate(ratings, self.mapping, self.blinded, self.workers, self.training, self.sp, self.wcfg)

    def test_plan_is_complete_and_public_inputs_omit_private_labels(self):
        self.assertEqual([len(self.blinded[k]) for k in ("primary", "context", "sprocket")], [72, 156, 36])
        text = json.dumps(self.blinded)
        for secret in [*j.worker.PANEL, *j.STAGES, "compact_schema_first", "historical_token_id_mask"]:
            self.assertNotIn(secret, text)
        self.assertEqual(len(self.mapping["primary"]), 72)

    def test_identical_scores_produce_no_mask_effect(self):
        result = self.aggregate(self.ratings())
        self.assertEqual(result["worker"]["mean_delta"]["task_relevance"], 0)
        self.assertFalse(result["worker"]["working_effect_signal"])
        self.assertEqual(result["worker"]["original_D_reproduced_exact_responses"], 72)
        self.assertEqual(result["worker"]["context_summary"]["BASE"]["responses"], 12)

    def test_paired_direction_uses_private_orientation_not_A_bias(self):
        ratings = self.ratings()
        for ident, row in ratings["primary"].items():
            corrected = "A" if self.mapping["primary"][ident]["A"] == "D_CORRECTED" else "B"
            row[corrected] = worker_score(3)
            row["B" if corrected == "A" else "A"] = worker_score(1)
            row["winner"] = corrected
        result = self.aggregate(ratings)
        self.assertEqual(result["worker"]["mean_delta"]["task_relevance"], 2)
        self.assertTrue(result["worker"]["working_effect_signal"])
        self.assertEqual(result["worker"]["positive_member_relevance_gains"], 6)

    def test_member_harm_blocks_working_signal_despite_positive_mean(self):
        ratings = self.ratings()
        for ident, row in ratings["primary"].items():
            private = self.mapping["primary"][ident]
            corrected = "A" if private["A"] == "D_CORRECTED" else "B"
            row[corrected] = worker_score(0 if private["member_id"] == j.worker.PANEL[0] else 4)
            row["B" if corrected == "A" else "A"] = worker_score(2)
        result = self.aggregate(ratings)
        self.assertGreater(result["worker"]["mean_delta"]["task_relevance"], .5)
        self.assertFalse(result["worker"]["working_effect_signal"])
        self.assertEqual(result["worker"]["classification"], "member_harm_exceeds_limit")

    def test_exact_half_point_threshold_uses_integer_case_deltas(self):
        for sums, expected in (((1, 1, 1, 2, 2, 29), True), ((1, 1, 1, 2, 2, 28), False)):
            with self.subTest(member_sums=sums):
                ratings = self.ratings()
                for member, total in zip(j.worker.PANEL, sums):
                    for ident in sorted(k for k, value in self.mapping["primary"].items() if value["member_id"] == member):
                        delta = min(total, 4)
                        total -= delta
                        corrected = "A" if self.mapping["primary"][ident]["A"] == "D_CORRECTED" else "B"
                        ratings["primary"][ident][corrected] = worker_score(delta)
                        ratings["primary"][ident]["B" if corrected == "A" else "A"] = worker_score(0)
                    self.assertEqual(total, 0)
                worker = self.aggregate(ratings)["worker"]
                self.assertEqual(worker["mean_delta"]["task_relevance"], sum(sums) / 72)
                self.assertIs(worker["working_effect_criteria"]["mean_gain"], expected)
                self.assertIs(worker["working_effect_signal"], expected)
                self.assertEqual(worker["positive_member_relevance_gains"], 6)
                self.assertEqual(worker["member_replication_units"], 6)

    def test_syntax_gain_alone_is_not_grounded_feedback_success(self):
        result = self.aggregate(self.ratings())
        self.assertEqual(result["sprocket"]["strict_syntax_count_gain"], 18)
        self.assertEqual(result["sprocket"]["strict_grounded_actionable_count_gain"], 0)
        self.assertEqual(result["sprocket"]["classification"], "syntax_gain_without_grounded_actionable_gain")
        self.assertEqual(len(result["sprocket"]["matched_case_seed_effects"]), 18)


class ExecutionDisclosureTests(unittest.TestCase):
    setUp = BlindPlanAndAggregationTests.setUp
    ratings = BlindPlanAndAggregationTests.ratings
    aggregate = BlindPlanAndAggregationTests.aggregate

    def test_matched_execution_retains_all_metadata_and_hashes(self):
        summary = self.aggregate(self.ratings())
        member = summary["worker"]["per_member"][0]
        execution = member["training_execution"]
        self.assertEqual(set(execution["records"]), {"D_ORIGINAL", "D_REPRODUCED", "D_CORRECTED"})
        for row in execution["records"].values():
            self.assertEqual(row["result_sha256"], j.sha256_value(execution_result()))
            self.assertEqual(row["packages"], execution_result()["packages"])
            self.assertEqual(row["recorded_result_host"], execution_result()["runner"])
            self.assertTrue(row["evidence_path"].startswith("sources/"))
        for comparison in ("paired_arms", "original_vs_reproduced"):
            row = execution[comparison]
            for key in ("cpu_model_match", "cpu_count_match", "platform_match", "recorded_hardware_match", "recorded_host_match", "full_package_metadata_match"):
                self.assertIs(row[key], True)
            self.assertEqual(row["differences"], {})
            self.assertEqual(row["unknown_fields"], [])
        self.assertEqual(member["attribution_limitations"], [])
        self.assertTrue(member["original_D_reproduced_tensor_hash_equal"])
        self.assertEqual(summary["worker"]["original_D_reproduced_exact_responses"], 72)
        report = j.report(summary)
        self.assertIn("only planned algorithmic intervention", report)
        self.assertIn("does not guarantee numerical determinism", report)
        self.assertEqual(summary["worker"]["paired_execution_classifications"], {"fresh_same_job_same_runner": 6})
        self.assertIn("python=3.12.3", report)
        self.assertNotIn("The recorded experimental difference", report)

    def test_mismatched_execution_is_disclosed_without_changing_quality_criteria(self):
        ratings = self.ratings()
        for ident, row in ratings["primary"].items():
            corrected = "A" if self.mapping["primary"][ident]["A"] == "D_CORRECTED" else "B"
            row[corrected], row["B" if corrected == "A" else "A"] = worker_score(3), worker_score(1)
        before = self.aggregate(ratings)
        lane = self.training[(j.worker.PANEL[0], "D_CORRECTED")]
        lane["result"]["runner"] = {"cpu_model": "Different CPU", "cpu_count": 8, "platform": "Different Linux"}
        lane["result"]["packages"]["python"] = "3.12.4"
        lane["result"]["packages"]["extra_library"] = "1.0"
        lane["result"]["recovery"] = {"resumed": True, "resumed_from_step": 30}
        lane["receipt"]["training_result_sha256"] = j.sha256_value(lane["result"])
        summary = self.aggregate(ratings)
        self.assertEqual(summary["worker"]["working_effect_criteria"], before["worker"]["working_effect_criteria"])
        self.assertTrue(summary["worker"]["working_effect_signal"])
        member = summary["worker"]["per_member"][0]
        execution = member["training_execution"]
        row = execution["paired_arms"]
        for key in ("cpu_model_match", "cpu_count_match", "platform_match", "recorded_hardware_match", "recorded_host_match"):
            self.assertIs(row[key], False)
        self.assertIs(row["full_package_metadata_match"], False)
        self.assertIs(row["package_matches"]["python"], False)
        self.assertEqual(set(row["differences"]), {"runner.cpu_model", "runner.cpu_count", "runner.platform", "packages.python", "packages.extra_library"})
        self.assertEqual(execution["records"]["D_CORRECTED"]["result_sha256"], lane["receipt"]["training_result_sha256"])
        limits = " ".join(member["attribution_limitations"])
        self.assertIn("cannot attribute the observed paired effect uniquely", limits)
        self.assertIn("every earlier execution segment", limits)
        self.assertTrue(member["original_D_reproduced_tensor_hash_equal"])
        self.assertEqual(member["original_D_reproduced_exact_responses"], 12)
        report = j.report(summary)
        for text in ("Different CPU", "Different Linux", "python=3.12.4", "extra_library=1.0", "True / 30", "do not exclude members"):
            self.assertIn(text, report)

    def test_historical_metadata_difference_is_distinct_from_paired_arm_difference(self):
        member_id = j.worker.PANEL[0]
        for stage in ("D_REPRODUCED", "D_CORRECTED"):
            lane = self.training[(member_id, stage)]
            lane["original_D_result"]["runner"]["cpu_model"] = "Historical CPU"
            lane["receipt"]["original_D_reference_result_sha256"] = j.sha256_value(lane["original_D_result"])
        member = self.aggregate(self.ratings())["worker"]["per_member"][0]
        self.assertIs(member["training_execution"]["paired_arms"]["recorded_host_match"], True)
        self.assertIs(member["training_execution"]["original_vs_reproduced"]["cpu_model_match"], False)
        self.assertTrue(member["original_D_reproduced_tensor_hash_equal"])
        self.assertEqual(len(member["attribution_limitations"]), 1)
        self.assertIn("metadata difference alone does not establish", member["attribution_limitations"][0])

    def test_missing_metadata_never_counts_as_hardware_match(self):
        for stage in ("D_REPRODUCED", "D_CORRECTED"):
            lane = self.training[(j.worker.PANEL[0], stage)]
            del lane["result"]["runner"]["cpu_model"]
            lane["receipt"]["training_result_sha256"] = j.sha256_value(lane["result"])
        summary = self.aggregate(self.ratings())
        member = summary["worker"]["per_member"][0]
        row = member["training_execution"]["paired_arms"]
        self.assertIsNone(row["cpu_model_match"])
        self.assertIsNone(row["recorded_hardware_match"])
        self.assertIn("runner.cpu_model", row["unknown_fields"])
        self.assertIn("missing fields do not establish", " ".join(member["attribution_limitations"]))
        self.assertIn("unknown", j.report(summary))

    def test_tensor_and_text_reproduction_checks_survive_matched_hardware(self):
        member_id = j.worker.PANEL[0]
        self.training[(member_id, "D_REPRODUCED")]["receipt"]["adapter_weight_fingerprint"] = fingerprint("b")
        self.training[(member_id, "D_REPRODUCED")]["receipt"]["adapter_directory_sha256"] = "b" * 64
        self.workers[(member_id, "D_ORIGINAL")]["rows"][0]["response"] = "A different historical response."
        summary = self.aggregate(self.ratings())
        member = summary["worker"]["per_member"][0]
        self.assertIs(member["training_execution"]["paired_arms"]["recorded_host_match"], True)
        for key in ("original_D_reproduced_tensor_hash_equal", "original_D_reproduced_safetensors_file_hash_equal", "original_D_reproduced_adapter_hash_equal"):
            self.assertIs(member[key], False)
        self.assertEqual(member["original_D_reproduced_exact_responses"], 11)
        self.assertEqual(summary["worker"]["original_D_reproduced_exact_responses"], 71)
        report = j.report(summary)
        self.assertIn("tensor fingerprint was not reproduced", report)
        self.assertIn("agree for 5/6 members", report)
        self.assertIn("11/12 exact Q responses", report)


class PairedExecutionProvenanceTests(unittest.TestCase):
    setUp = BlindPlanAndAggregationTests.setUp
    ratings = BlindPlanAndAggregationTests.ratings
    aggregate = BlindPlanAndAggregationTests.aggregate

    def proof(self, member=None):
        return j.paired_execution_provenance(member or j.worker.PANEL[0], self.training)

    def recovered_copy(self, member, stage):
        provenance = self.training[(member, stage)]["execution_provenance"]
        prior = provenance["invocations"][0]
        invocation = copy.deepcopy(prior)
        invocation["ordinal"] = 2
        invocation["start"].update(resume_requested=True, source_kind="complete_copy", start_step=120)
        invocation["start"]["execution"].update(run_id="456", paired_job_nonce="copy-only-job", boot_id="copy-only-boot")
        invocation["finish"]["finished_monotonic_ns"] = 900
        provenance["invocations"].append(invocation)
        provenance["zero_step_copies"] = [2]
        provenance["resume_invocations"] = [2]
        return provenance

    def test_fresh_pairs_prove_same_job_runner_distinct_processes_and_counterbalanced_order(self):
        first = []
        for member in j.worker.PANEL:
            proof = self.proof(member)
            for key in ("fresh_uninterrupted_pair", "same_job_for_optimizer_segments",
                "same_runner_instance_for_optimizer_segments", "distinct_fresh_processes", "counterbalanced_order_observed"):
                self.assertIs(proof[key], True)
            first.append(proof["predeclared_arm_order"][0])
            self.assertIs(proof["historical_D_full_execution_provenance_available"], False)
        self.assertEqual(first.count("historical_token_id_mask"), 3)
        self.assertEqual(first.count("length_derived_mask"), 3)

    def test_matching_cpu_strings_do_not_establish_fresh_pairing(self):
        member = j.worker.PANEL[0]
        for field, value in (("boot_id", "other-boot"), ("paired_job_nonce", "other-job"), ("run_id", None)):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.training)
                provenance = self.training[(member, "D_CORRECTED")]["execution_provenance"]
                provenance["optimizer_segments"][0]["execution"][field] = value
                with self.assertRaisesRegex(RuntimeError, "fresh paired arms must share"):
                    self.proof()
                self.training = saved

    def test_fresh_pair_requires_distinct_processes_and_predeclared_sequential_order(self):
        member = j.worker.PANEL[0]
        provenance = self.training[(member, "D_CORRECTED")]["execution_provenance"]
        execution = provenance["invocations"][0]["start"]["execution"]
        execution["process_id"] = 1001
        with self.assertRaisesRegex(RuntimeError, "distinct recorded processes"):
            self.proof()
        execution["process_id"] = 1002
        execution["started_monotonic_ns"] = 150
        with self.assertRaisesRegex(RuntimeError, "predeclared sequential order"):
            self.proof()

    def test_zero_step_copy_on_another_host_does_not_rewrite_optimizer_history(self):
        member = j.worker.PANEL[0]
        self.recovered_copy(member, "D_REPRODUCED")
        proof = self.proof()
        self.assertFalse(proof["fresh_uninterrupted_pair"])
        self.assertTrue(proof["same_job_and_runner_for_all_optimizer_segments"])
        self.assertEqual(proof["classification"], "recovered_same_job_same_runner")
        arm = proof["arms"]["D_REPRODUCED"]
        self.assertEqual(arm["zero_step_copies"], [2])
        self.assertEqual(len(arm["optimizer_segments"]), 1)
        self.assertEqual(arm["optimizer_segments"][0]["execution"]["run_id"], "123")
        self.assertIn("later zero-step copies", " ".join(self.aggregate(self.ratings())["worker"]["per_member"][0]["attribution_limitations"]))

    def test_recovered_mixed_job_pair_is_retained_and_qualified_without_score_changes(self):
        before = self.aggregate(self.ratings())
        member = j.worker.PANEL[0]
        self.recovered_copy(member, "D_REPRODUCED")
        corrected = self.training[(member, "D_CORRECTED")]["execution_provenance"]
        corrected["optimizer_segments"][0]["execution"].update(run_id="456", boot_id="second-boot", paired_job_nonce="second-job")
        proof = self.proof()
        self.assertEqual(proof["classification"], "recovered_mixed_job_or_runner")
        self.assertFalse(proof["same_job_and_runner_for_all_optimizer_segments"])
        after = self.aggregate(self.ratings())
        self.assertEqual(len(after["worker"]["per_member"]), 6)
        for key in ("mean_delta", "working_effect_criteria", "working_effect_signal", "member_replication_units"):
            self.assertEqual(after["worker"][key], before["worker"][key])
        self.assertIn("cross recorded jobs or runner instances", j.report(after))

    def test_interrupted_unattributed_steps_cannot_establish_full_pairing(self):
        member = j.worker.PANEL[0]
        provenance = self.training[(member, "D_REPRODUCED")]["execution_provenance"]
        provenance["invocations"][0]["finish"] = None
        provenance["invocations"][0]["retained_end_step"] = 90
        provenance["optimizer_segments"][0].update(end_step=90, steps=90)
        provenance["unfinished_invocations"] = [1]
        provenance["unattributed_optimizer_segments"] = [{"start_step": 90, "end_step": 120,
            "steps": 30, "reason": "interrupted provenance write"}]
        proof = self.proof()
        self.assertFalse(proof["fresh_uninterrupted_pair"])
        self.assertIsNone(proof["same_job_and_runner_for_all_optimizer_segments"])
        self.assertEqual(proof["classification"], "recovered_incomplete_execution_provenance")
        self.assertIn("provenance unknown", j.report(self.aggregate(self.ratings())))


class BudgetAndReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        prepare_output(self.out, None, {"fixture": "fixed diagnostic sources"})
        self.campaign = {"nvidia": {"endpoint": "https://invalid.example", "generation": {"attempts": 6}}}
        context = redirect_stderr(io.StringIO())
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def cp(self, out=None, offline=False, api=fixture_api):
        return j.DiagnosticCheckpoint(out or self.out, campaign=self.campaign, key="" if offline else "fixture",
                                      offline=offline, api=api, logical_request_limit=114)

    def test_full_50_call_protocol_replays_identically_offline(self):
        _, _, _, blinded, _, _ = synthetic_population()
        cfg = {"paired_max_tokens": 6144, "context_max_tokens": 4096, "sprocket_max_tokens": 4096}
        first = self.cp()
        ratings, refs = j.judge_inputs(first, blinded, cfg, {"id": "critic"})
        self.assertEqual(first.new_calls, 50)
        self.assertEqual(sum(label.startswith("f-worker-") for label in first.logical_labels), 44)
        self.assertEqual(sum(label.startswith("f-sprocket-") for label in first.logical_labels), 6)
        first.finish("judgments.json", {"ratings": ratings, "refs": refs})
        before = (self.out / "judgments.json").read_bytes()
        replay = self.root / "replay"
        prepare_output(replay, self.out, {"fixture": "fixed diagnostic sources"})
        second = self.cp(replay, offline=True, api=lambda **_: self.fail("offline provider call"))
        replayed, replay_refs = j.judge_inputs(second, blinded, cfg, {"id": "critic"})
        second.finish("judgments.json", {"ratings": replayed, "refs": replay_refs})
        self.assertEqual(second.new_calls, 0)
        self.assertEqual(before, (replay / "judgments.json").read_bytes())

    def test_worker_limit_cannot_consume_sprocket_reserve(self):
        cp = self.cp(api=lambda **_: self.fail("budget crossed"))
        cp.logical_labels = {f"f-worker-test-{i}" for i in range(88)}
        with self.assertRaisesRegex(RuntimeError, "worker logical request budget"):
            cp.request("f-worker-new", {}, "fixture")

    def test_sprocket_limit_cannot_consume_worker_reserve(self):
        cp = self.cp(api=lambda **_: self.fail("budget crossed"))
        cp.logical_labels = {f"f-sprocket-test-{i}" for i in range(26)}
        with self.assertRaisesRegex(RuntimeError, "sprocket logical request budget"):
            cp.request("f-sprocket-new", {}, "fixture")

    def test_overbudget_resume_is_rejected_before_any_call(self):
        raw = self.out / "provider"
        raw.mkdir()
        for index in range(89):
            write_once(raw / f"f-worker-fixture-{index}-request.json", {})
        with self.assertRaisesRegex(RuntimeError, "archived worker logical"):
            self.cp(api=lambda **_: self.fail("provider called"))

    def test_identical_text_inconsistency_repairs_only_missing_pair(self):
        calls = []
        blinded = {"primary": [{"id": f"P{i:03d}", "situation": "Test situation.", "A": "same", "B": "same"} for i in (1, 2)], "context": [], "sprocket": []}
        def api(**kw):
            items = json.loads(kw["payload"]["messages"][1]["content"].rsplit("ITEMS:\n", 1)[1])
            calls.append([r["id"] for r in items])
            rows = [{"id": item["id"], "A": worker_score(), "B": worker_score(), "winner": "tie", "reason": "Identical concrete action."} for item in items]
            if len(calls) == 1:
                rows[1]["A"]["task_relevance"] = 4
            return write_archive(kw, json.dumps({"reviews": rows}))
        cp = self.cp(api=api)
        ratings, _ = j.judge_inputs(cp, blinded, {"paired_max_tokens": 6144, "context_max_tokens": 4096, "sprocket_max_tokens": 4096}, {"id": "critic"})
        self.assertEqual(calls, [["P001", "P002"], ["P002"]])
        self.assertEqual(ratings["primary"]["P002"]["A"]["task_relevance"], 2)

    def test_missing_only_requests_use_actual_ids_in_every_family(self):
        rows = PromptCoverageTests().rows
        blinded = {kind: rows(kind, ids) for kind, ids in {
            "primary": ["P037", "P052"], "context": ["R085", "R143"],
            "sprocket": ["B-a97b", "B-f31c"]}.items()}
        calls = []
        def api(**kw):
            instructions, serialized = kw["payload"]["messages"][1]["content"].rsplit("\nITEMS:\n", 1)
            items = json.loads(serialized)
            ids = [item["id"] for item in items]
            self.assertEqual(json.loads(instructions.split("\nRequired IDs: ", 1)[1].splitlines()[0]), ids)
            example, _ = json.JSONDecoder().raw_decode(instructions.split("Return JSON only: ", 1)[1])
            self.assertEqual(example["reviews"][0]["id"], ids[0])
            self.assertIn(f"exactly {len(ids)} review objects", instructions)
            calls.append((kw["role"], ids, kw["payload"]["max_tokens"]))
            row = {**example["reviews"][0], "reason": "The supplied text is empty."}
            return write_archive(kw, json.dumps({"reviews": [row]}))
        cp = self.cp(api=api)
        ratings, _ = j.judge_inputs(cp, blinded,
            {"paired_max_tokens": 6144, "context_max_tokens": 4096, "sprocket_max_tokens": 4096}, {"id": "critic"})
        self.assertEqual(calls, [
            ("f-worker-primary", ["P037", "P052"], 6144), ("f-worker-primary-repair", ["P052"], 6144),
            ("f-worker-context", ["R085", "R143"], 4096), ("f-worker-context-repair", ["R143"], 4096),
            ("f-sprocket-grounding", ["B-a97b", "B-f31c"], 4096), ("f-sprocket-grounding-repair", ["B-f31c"], 4096)])
        self.assertEqual(cp.new_calls, 6)
        for kind, items in blinded.items():
            self.assertEqual(set(ratings[kind]), {item["id"] for item in items})


class NativeSourceTests(unittest.TestCase):
    def frozen_worker_fixture(self, root):
        cfg, wcfg, suite, campaign, pref = j.load_contract(j.DEFAULT_CONFIG)
        train_root, worker_root = root / "training", root / "workers"
        for member in wcfg["members"]:
            population, member_meta = j.worker.member_contract(campaign, member)
            parent = j.worker.source_result(member, "C")["adapter"]["sha256"]
            historical = j.worker.source_result(member, "D")["adapter"]["sha256"]
            for arm, stage in j.ARM_STAGE.items():
                out = train_root / member / arm
                identity = j.worker.run_identity(j.ROOT / cfg["worker_config"], wcfg,
                    kind="matched-preference-mask-training", member=member, stage=arm, adapter_sha=parent)
                identity.update(mask_mode=arm, member_seed=member_meta["seed"],
                    pair_dataset_sha256=j.sha256_file(j.worker.pair_path(member)), optimizer_steps=120,
                    training_config=pref["preference_training"], member_contract_sha256=population["member_contract_sha256"],
                    new_Q_reserve_used_for_training=False, original_E_F_suite_used_for_training=False,
                    post_training_legacy_T_generation_omitted_in_both_arms=True)
                write_once(out / "training-identity.json", identity)
                with patch.object(j.worker, "training_execution_metadata", return_value=invocation_execution(member, arm)):
                    invocation = j.worker.TrainingInvocation(out, wcfg, member, arm, resumed=False, complete=False)
                engine_sha = j.sha256_file(j.ROOT / "scripts/village_001d_preference_train.py")
                adapter = hashlib.sha256((member + stage).encode()).hexdigest()
                result = {"experiment": "VILLAGE-001F", "stage": "matched-mask-preference-continuation",
                    "member_id": member, "member_seed": member_meta["seed"], "mask_mode": arm,
                    "parent_adapter_sha256": parent, "preference_dataset_sha256": j.sha256_file(j.worker.pair_path(member)),
                    "training_config": pref["preference_training"], "training_identity_sha256": j.sha256_file(out / "training-identity.json"),
                    "new_Q_reserve_used_for_training": False, "training": {"optimizer_steps": 120, "pair_count": 60,
                        **{key: [float(index) / 100 for index in range(120)] for key in j.worker.TRACE_KEYS}},
                    "engine_source_sha256": engine_sha, "adapter": {"sha256": adapter},
                    **execution_result(), "evaluation": {"responses": []}}
                write_once(out / "result.json", result)
                write_once(out / "training.json", {"schema": "village-001f-matched-mask-training-v1",
                    "member_id": member, "mask_mode": arm, "training_identity_sha256": j.sha256_file(out / "training-identity.json"),
                    "training_result_sha256": j.sha256_file(out / "result.json"), "optimizer_steps": 120, "pair_count": 60,
                    "new_local_generations": 0, "original_C_adapter_sha256": parent, "original_D_adapter_sha256": historical,
                    "engine_source_sha256": engine_sha, "adapter_directory_sha256": adapter,
                    "original_D_reference_result_sha256": j.sha256_file(j.ROOT / "experiments/village-001/preference-evidence/results" / f"{member}.json"),
                    "adapter_weight_fingerprint": fingerprint("a" if stage == "D_REPRODUCED" else "b")})
                with patch.object(j.worker.time, "monotonic_ns", return_value=invocation.start["execution"]["started_monotonic_ns"] + 100):
                    invocation.finish("completed")
        training, _ = j.load_training(train_root, cfg, wcfg, campaign, pref)
        for member, stage in [(m, s) for m in wcfg["members"] for s in j.STAGES] + [(None, "BASE")]:
            if stage == "BASE":
                adapter, weight_fingerprint = None, None
            elif stage in ("C", "D_ORIGINAL"):
                adapter = j.worker.source_result(member, "C" if stage == "C" else "D")["adapter"]["sha256"]
                weight_fingerprint = fingerprint()
            else:
                adapter = training[(member, stage)]["receipt"]["adapter_directory_sha256"]
                weight_fingerprint = training[(member, stage)]["receipt"]["adapter_weight_fingerprint"]
            out = worker_root / stage / (member or "BASE")
            identity = j.worker.run_identity(j.ROOT / cfg["worker_config"], wcfg,
                kind="new-Q-native-responses", member=member, stage=stage, adapter_sha=adapter)
            cp = GenerationCheckpoint(out=out, resume=None, identity=identity, keys=[r["id"] for r in suite["scenarios"]],
                terminal_name="result.json", validate_row=lambda *_: None)
            for scenario in suite["scenarios"]:
                cp.append(scenario["id"], {"scenario_id": scenario["id"], "family": scenario["family"], "prompt": scenario["prompt"],
                    "response": "A fixed synthetic native response.", "input_token_ids": [1, 2], "generated_token_ids": [3],
                    "generated_tokens": 1, "rendered_prompt": "Native prompt.", "attention_mask": [1, 1]})
            cp.finish({"schema": "village-001f-diagnostic-responses-v1", "experiment": "VILLAGE-001F",
                "member_id": member, "stage": stage, "base_control": stage == "BASE", "new_Q_reserve_sha256": wcfg["eval_sha256"],
                "hidden_member_prompt_used": False, "adapter_directory_sha256": adapter,
                "adapter_weight_fingerprint": weight_fingerprint, "runtime": wcfg["runtime"], "responses": cp.rows}, "responses")
        return cfg, wcfg, suite, campaign, pref, train_root, worker_root, training

    def test_all_matched_training_and_native_worker_sources_validate_before_judging(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, wcfg, suite, campaign, pref, train_root, worker_root, training = self.frozen_worker_fixture(Path(tmp))
            workers, sources = j.load_workers(worker_root, training, cfg, wcfg, suite)
            self.assertEqual(len(training), 12)
            self.assertEqual(len(workers), 25)
            self.assertEqual(sum(len(lane["rows"]) for lane in workers.values()), 300)
            self.assertFalse(any(path.endswith((".bin", ".safetensors")) for path in sources))
            _, training_sources = j.load_training(train_root, cfg, wcfg, campaign, pref)
            self.assertEqual(len([key for key in training_sources if "/training-invocations/" in key]), 24)
            self.assertEqual(len([key for key in training_sources if key.startswith("training-references/")]), 6)
            for member in wcfg["members"]:
                self.assertTrue(j.paired_execution_provenance(member, training)["fresh_uninterrupted_pair"])
            lane = worker_root / "D_CORRECTED" / wcfg["members"][0] / "result.json"
            terminal = j.read(lane)
            terminal["adapter_weight_fingerprint"] = fingerprint("c")
            lane.write_bytes(j.json_bytes(terminal))
            with self.assertRaisesRegex(RuntimeError, "inferred weights differ"):
                j.load_workers(worker_root, training, cfg, wcfg, suite)

    def test_training_seed_drift_is_refused_even_with_rehashed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, wcfg, suite, campaign, pref, train_root, _, _ = self.frozen_worker_fixture(Path(tmp))
            lane = train_root / wcfg["members"][0] / "length_derived_mask"
            identity = j.read(lane / "training-identity.json")
            identity["member_seed"] += 1
            (lane / "training-identity.json").write_bytes(j.json_bytes(identity))
            receipt = j.read(lane / "training.json")
            receipt["training_identity_sha256"] = j.sha256_file(lane / "training-identity.json")
            (lane / "training.json").write_bytes(j.json_bytes(receipt))
            with self.assertRaisesRegex(RuntimeError, "source/seed/config drift"):
                j.load_training(train_root, cfg, wcfg, campaign, pref)

    def test_execution_metadata_remains_bound_to_validated_result_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, wcfg, _, campaign, pref, train_root, _, training = self.frozen_worker_fixture(Path(tmp))
            member = wcfg["members"][0]
            _, sources = j.load_training(train_root, cfg, wcfg, campaign, pref)
            disclosure = j.training_execution_disclosure(member, training)
            original = disclosure["records"]["D_ORIGINAL"]
            self.assertEqual(original["result_sha256"], j.sha256_file(sources[original["evidence_path"].removeprefix("sources/")]))
            self.assertEqual(original["recorded_result_host"], j.worker.source_result(member, "D")["runner"])
            lane = train_root / member / "length_derived_mask"
            result = j.read(lane / "result.json")
            result["runner"]["cpu_model"] = "A different recorded CPU"
            (lane / "result.json").write_bytes(j.json_bytes(result))
            with self.assertRaisesRegex(RuntimeError, "training result hash mismatch"):
                j.load_training(train_root, cfg, wcfg, campaign, pref)
            receipt = j.read(lane / "training.json")
            receipt["training_result_sha256"] = j.sha256_file(lane / "result.json")
            (lane / "training.json").write_bytes(j.json_bytes(receipt))
            with self.assertRaisesRegex(RuntimeError, "completed invocation differs from immutable training receipts"):
                j.load_training(train_root, cfg, wcfg, campaign, pref)

    def test_native_training_provenance_cannot_be_dropped_or_relabelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, wcfg, _, campaign, pref, train_root, _, _ = self.frozen_worker_fixture(Path(tmp))
            lane = train_root / wcfg["members"][0] / "length_derived_mask"
            start = lane / "training-invocations/invocation-0001-start.json"
            original = start.read_bytes()
            start.unlink()
            with self.assertRaisesRegex(RuntimeError, "lacks its immutable start"):
                j.load_training(train_root, cfg, wcfg, campaign, pref)
            value = json.loads(original)
            value["execution"]["paired_job_nonce"] = "a different job"
            start.write_bytes(j.json_bytes(value))
            with self.assertRaisesRegex(RuntimeError, "training completion provenance identity changed"):
                j.load_training(train_root, cfg, wcfg, campaign, pref)

    def sprocket_fixture(self, root):
        d = j.sprocket
        cfg, plan = d.build_plan(d.DEFAULT_CONFIG)
        d.prepare_output(root, None, plan)
        cases = {row["case_id"]: row for row in plan["cases"]}
        rows = []
        def archive(label, endpoint, payload, response, generation=False):
            d.write_json(root / "http" / f"{label}.request.json", {"endpoint": endpoint, "payload": payload, "generation": generation})
            d.write_json(root / "http" / f"{label}.response.json", response)
        for request in plan["requests"]:
            key, messages = request["key"], request["payload"]["messages"]
            rendered = "<|system|>" + messages[0]["content"] + "<|end|><|user|>" + messages[1]["content"] + "<|end|><|assistant|>"
            template, tokens, counted = {"prompt": rendered}, {"tokens": [0, 2, 100, 5, 3, 101, 5, 4]}, {"input_tokens": 8}
            archive(key + "-template", "/apply-template", request["payload"], template)
            archive(key + "-tokenize", "/tokenize", {"content": rendered, "add_special": True, "parse_special": True}, tokens)
            archive(key + "-count", "/v1/chat/completions/input_tokens", request["payload"], counted)
            wire = d.wire_record(request, template, tokens, counted, 2048, cfg["wire"]["gguf_chat_template_sha256"])
            d.write_json(root / "wire" / f"{key}.json", wire)
            response = {"choices": [{"message": {"content": "A malformed yet preserved critique."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2}, "__verbose": {"tokens": [31, 5],
                "prompt": "<|endoftext|>" + rendered, "truncated": False, "stop_type": "eos",
                "generation_settings": {"seed": request["seed"], "n_predict": 160, "temperature": 1.049999952, "top_p": .970000028}}}
            archive(key + "-generate", "/v1/chat/completions", request["payload"], response, generation=True)
            archive(key + "-actual-tokenize", "/tokenize", {"content": response["__verbose"]["prompt"], "add_special": False, "parse_special": True}, tokens)
            row = d.analyze_response(request, cases[request["case_id"]], wire, response, tokens)
            d.write_json(root / "rows" / f"{key}.json", row)
            rows.append(row)
        d.summarize(root, plan, rows)
        return cfg, plan

    def test_sprocket_source_reconstruction_rejects_rehashed_actual_prompt_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sprocket"
            _, plan = self.sprocket_fixture(root)
            cfg = j.read(j.DEFAULT_CONFIG)
            rows, public, _ = j.load_sprocket(root, cfg)
            self.assertEqual(len(rows), 36)
            self.assertTrue(all(row["feedback"] == "A malformed yet preserved critique." for row in public["items"]))
            request_path = root / "http" / f"{plan['requests'][0]['key']}-actual-tokenize.request.json"
            request = j.read(request_path)
            request["payload"]["content"] = "A substituted native prompt."
            request_path.write_bytes(j.sprocket.canonical(request))
            inventory = j.read(root / "evidence-hashes.json")
            inventory["files"][request_path.relative_to(root).as_posix()] = {"sha256": j.sha256_file(request_path), "bytes": request_path.stat().st_size}
            (root / "evidence-hashes.json").write_bytes(j.sprocket.canonical(inventory))
            with self.assertRaisesRegex(RuntimeError, "archived local request differs"):
                j.load_sprocket(root, cfg)

    def test_native_terminal_rewrite_cannot_substitute_a_better_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = {"scenarios": [{"id": f"Q{i:02d}", "family": "fixture", "prompt": f"Situation {i}."} for i in range(1, 13)]}
            identity = {"fixture": "native identity"}
            cp = GenerationCheckpoint(out=root / "lane", resume=None, identity=identity,
                keys=[r["id"] for r in suite["scenarios"]], terminal_name="result.json", validate_row=lambda *_: None)
            for scenario in suite["scenarios"]:
                cp.append(scenario["id"], {"scenario_id": scenario["id"], "family": "fixture", "prompt": scenario["prompt"],
                    "response": "The actual imperfect response.", "input_token_ids": [1, 2], "generated_token_ids": [3],
                    "generated_tokens": 1, "rendered_prompt": "Native prompt.", "attention_mask": [1, 1]})
            doc = {"responses": cp.rows}
            cp.finish(doc, "responses")
            j.native_rows(root / "lane", doc, identity, suite)
            doc["responses"][0]["response"] = "A substituted perfect answer."
            with self.assertRaisesRegex(RuntimeError, "checkpoint row"):
                j.native_rows(root / "lane", doc, identity, suite)


if __name__ == "__main__":
    unittest.main()
