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
            training[(member, stage)] = {"receipt": {"adapter_directory_sha256": "a" * 64, "adapter_weight_fingerprint": fingerprint()}}
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

    def test_syntax_gain_alone_is_not_grounded_feedback_success(self):
        result = self.aggregate(self.ratings())
        self.assertEqual(result["sprocket"]["strict_syntax_count_gain"], 18)
        self.assertEqual(result["sprocket"]["strict_grounded_actionable_count_gain"], 0)
        self.assertEqual(result["sprocket"]["classification"], "syntax_gain_without_grounded_actionable_gain")
        self.assertEqual(len(result["sprocket"]["matched_case_seed_effects"]), 18)


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
                engine_sha = j.sha256_file(j.ROOT / "scripts/village_001d_preference_train.py")
                adapter = hashlib.sha256((member + stage).encode()).hexdigest()
                result = {"experiment": "VILLAGE-001F", "stage": "matched-mask-preference-continuation",
                    "member_id": member, "member_seed": member_meta["seed"], "mask_mode": arm,
                    "parent_adapter_sha256": parent, "preference_dataset_sha256": j.sha256_file(j.worker.pair_path(member)),
                    "training_config": pref["preference_training"], "training_identity_sha256": j.sha256_file(out / "training-identity.json"),
                    "new_Q_reserve_used_for_training": False, "training": {"optimizer_steps": 120, "pair_count": 60},
                    "engine_source_sha256": engine_sha, "adapter": {"sha256": adapter},
                    "packages": wcfg["runtime"], "evaluation": {"responses": []}}
                write_once(out / "result.json", result)
                write_once(out / "training.json", {"schema": "village-001f-matched-mask-training-v1",
                    "member_id": member, "mask_mode": arm, "training_identity_sha256": j.sha256_file(out / "training-identity.json"),
                    "training_result_sha256": j.sha256_file(out / "result.json"), "optimizer_steps": 120, "pair_count": 60,
                    "new_local_generations": 0, "original_C_adapter_sha256": parent, "original_D_adapter_sha256": historical,
                    "engine_source_sha256": engine_sha, "adapter_directory_sha256": adapter,
                    "original_D_reference_result_sha256": j.sha256_file(j.ROOT / "experiments/village-001/preference-evidence/results" / f"{member}.json"),
                    "adapter_weight_fingerprint": fingerprint("a" if stage == "D_REPRODUCED" else "b")})
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
