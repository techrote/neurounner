"""Author-side secondary custody tests: inspect archive, never train or open finals."""
from __future__ import annotations

import copy
import json
import unittest

from scripts import detective_004a_custody_audit as audit
from scripts import detective_004a_protocol as protocol


class Detective004ACustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.b3 = protocol.load_contract()
        cls.rows = protocol.build_training(cls.cfg, cls.b3)

    def test_all_36_exact_model_visible_views_and_receipts(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                audit.audit_row(row, self.b3)

    def test_committed_archive_replay_without_final_or_training(self):
        result = audit.audit_archive(protocol.DATASET)
        self.assertEqual(result["rows_with_exact_model_visible_views"], 36)
        self.assertEqual(result["rows_with_replayed_assistant_char_spans"], 36)
        self.assertEqual(result["host_owned_result_continuations"], 6)
        self.assertFalse(result["real_tokenizer_labels_verified"])
        self.assertFalse(result["independent_protocol_approval"])
        self.assertEqual(result["optimizer_steps"], 0)

    def test_unexposed_view_change_refused_even_if_producer_accepted_it(self):
        row = copy.deepcopy(next(r for r in self.rows if r["kind"] == "tool_selection"))
        row["view"]["visible_assets"].append({
            "asset_id": "forged-out-of-band-asset", "excerpt": "not displayed to model"
        })
        # Old validator inspects view and assistant target, but not row view vs
        # the actual string inside the user turn. The secondary audit closes it.
        protocol.validate_example(row, self.b3)
        with self.assertRaisesRegex(protocol.ProtocolError, "not identical"):
            audit.audit_row(row, self.b3)

    def test_case_family_alias_refused(self):
        row = copy.deepcopy(self.rows[0])
        row["case_family"] = "cinder-span"
        with self.assertRaises(protocol.ProtocolError):
            audit.audit_row(row, self.b3)

    def test_host_receipt_is_not_the_user_turn(self):
        row = copy.deepcopy(next(r for r in self.rows if r["kind"] == "result_success"))
        row["view"]["player_request"] += " confirmed:amber-quay:5"
        with self.assertRaisesRegex(protocol.ProtocolError, "not identical"):
            audit.audit_row(row, self.b3)

    def test_host_result_tamper_rejected(self):
        row = copy.deepcopy(next(r for r in self.rows if r["kind"] == "result_denied"))
        row["turns"][3]["content"] = json.dumps(
            {"status": "ACCESS_DENIED", "evidence": "unadmitted record"}
        )
        with self.assertRaises(protocol.ProtocolError):
            audit.audit_row(row, self.b3)


    @classmethod
    def synthetic_gate_fixture(cls):
        """Unsealed unit-test tasks; no adapter, no posthash final material."""
        cfg, b3 = protocol.load_contract()
        seed = "0123456789abcdef" * 4
        rows = []
        extra = ("result_success", "result_denied", "restraint")
        for family, extra_kind in zip(protocol.EXPECTED_FINAL_FAMILIES, extra):
            case = protocol.family_data(family, seed)
            for index, op in enumerate(protocol.OPS):
                rows.append(protocol.make_example(
                    family, case, "tool_selection", index, op))
            rows.append(protocol.make_example(family, case, extra_kind, 3))
        for row in rows:
            row["rendered"], row["loss_char_spans"] = protocol.render_with_mask(
                row["turns"], b3)
        return rows

    def test_future_gate_guard_accepts_only_complete_synthetic_fixture(self):
        rows = self.synthetic_gate_fixture()
        audit.audit_future_gate_layout(rows, self.b3)
        # Unit-test outputs, not model generations or evidence of admission.
        trained = {row["id"]: row["turns"][-1]["content"] for row in rows}
        base = {row["id"]: "unsupported mock baseline" for row in rows}
        result = audit.guarded_admission_gate(rows, trained, base, self.b3)
        self.assertEqual(result["status"], "research_candidate")
        self.assertFalse(result["production_admission"])
        self.assertFalse(result["population_expansion_admitted"])

    def test_future_gate_guard_rejects_duplicate_replayed_tasks(self):
        rows = self.synthetic_gate_fixture()
        forged = copy.deepcopy(rows)
        forged[0] = copy.deepcopy(rows[4])
        trained = {row["id"]: row["turns"][-1]["content"] for row in forged}
        base = {row["id"]: "unsupported mock baseline" for row in forged}
        # The original frozen evaluator counts the repeated successful task.
        # Keep the generator/blob stable; the future guarded entry point refuses.
        self.assertEqual(
            protocol.admission_gate(forged, trained, base, self.b3)["status"],
            "research_candidate")
        with self.assertRaisesRegex(protocol.ProtocolError, "duplicate"):
            audit.guarded_admission_gate(forged, trained, base, self.b3)

    def test_future_gate_guard_rejects_category_and_visible_view_drift(self):
        rows = self.synthetic_gate_fixture()
        changed = copy.deepcopy(rows)
        changed[-1]["kind"] = "result_success"
        with self.assertRaisesRegex(protocol.ProtocolError, "denominator"):
            audit.audit_future_gate_layout(changed, self.b3)

        changed = copy.deepcopy(rows)
        changed[0]["view"]["visible_assets"].append({
            "asset_id": "not-visible-to-model", "excerpt": "fabricated"
        })
        # The producer's verifier still checks the detached view, not
        # that it matches the actual prompt. Guard checks both.
        protocol.validate_example(changed[0], self.b3)
        with self.assertRaisesRegex(protocol.ProtocolError, "user view differs"):
            audit.audit_future_gate_layout(changed, self.b3)

    def test_future_gate_guard_rejects_forged_receipts_and_seed_mix(self):
        rows = self.synthetic_gate_fixture()
        changed = copy.deepcopy(rows)
        success = next(r for r in changed if r["kind"] == "result_success")
        success["turns"][3]["content"] = protocol.canonical({
            "status": "OK", "marker": "confirmed:elsewhere:3",
            "excerpt": "Synthetic receipt confirmed:elsewhere:3"
        })
        success["rendered"], success["loss_char_spans"] = (
            protocol.render_with_mask(success["turns"], self.b3))
        with self.assertRaisesRegex(protocol.ProtocolError, "receipt changed"):
            audit.audit_future_gate_layout(changed, self.b3)

        changed = copy.deepcopy(rows)
        for row in changed:
            if row["case_family"] == protocol.EXPECTED_FINAL_FAMILIES[0]:
                row["view"]["case_id"] = "fiction-only-d004a-ravel-gate-aaaaaaaaaa"
                view = row["view"]
                row["turns"][1]["content"] = (
                    "Player request:\n" + view["player_request"] +
                    "\nCurrent fictional game view:\n" + protocol.canonical(view))
                row["rendered"], row["loss_char_spans"] = (
                    protocol.render_with_mask(row["turns"], self.b3))
        with self.assertRaisesRegex(protocol.ProtocolError, "seed prefixes"):
            audit.audit_future_gate_layout(changed, self.b3)


    def test_training_archive_native_inference_prefix_has_no_gold(self):
        for row in self.rows:
            with self.subTest(id=row["id"]):
                prompt = audit.model_input_prefix(row, self.b3)
                self.assertEqual(
                    prompt, row["rendered"][:row["loss_char_spans"][-1]["start"]])
                self.assertTrue(prompt.endswith("<|assistant|>"))
                self.assertNotIn(row["turns"][-1]["content"], prompt)
                self.assertEqual(
                    prompt, audit.native.render_text(row["turns"][:-1]))
                if row["kind"].startswith("result_"):
                    self.assertIn("<|tool_result|>", prompt)
                    self.assertIn("<|tool_call|>", prompt)
                else:
                    self.assertNotIn("<|tool_result|>", prompt)

    def test_unsealed_synthetic_future_prefix_excludes_target_answer(self):
        for row in self.synthetic_gate_fixture():
            with self.subTest(id=row["id"]):
                prefix = audit.model_input_prefix(row, self.b3)
                self.assertTrue(prefix.endswith("<|assistant|>"))
                self.assertNotIn(row["turns"][-1]["content"], prefix)
                self.assertEqual(prefix, row["rendered"][:row["loss_char_spans"][-1]["start"]])

    def test_inference_prefix_rejects_gold_leak_and_view_substitution(self):
        restraint = copy.deepcopy(
            next(r for r in self.rows if r["kind"] == "restraint"))
        # This is a synthetic adversarial test only. The frozen archive and
        # the 12-task posthash final remain untouched.
        restraint["view"]["player_request"] += " " + restraint["turns"][-1]["content"]
        view = restraint["view"]
        restraint["turns"][1]["content"] = (
            "Player request:\n" + view["player_request"] +
            "\nCurrent fictional game view:\n" + protocol.canonical(view))
        restraint["rendered"], restraint["loss_char_spans"] = (
            protocol.render_with_mask(restraint["turns"], self.b3))
        # Its fields and spans still pass the original producer validator.
        protocol.validate_example(restraint, self.b3)
        with self.assertRaisesRegex(protocol.ProtocolError, "gold assistant completion"):
            audit.model_input_prefix(restraint, self.b3)

        changed = copy.deepcopy(self.rows[0])
        changed["turns"][1]["content"] += " undisclosed-extra-text"
        changed["rendered"], changed["loss_char_spans"] = (
            protocol.render_with_mask(changed["turns"], self.b3))
        protocol.validate_example(changed, self.b3)
        with self.assertRaisesRegex(protocol.ProtocolError, "user view differs"):
            audit.model_input_prefix(changed, self.b3)



    def test_future_recipe_guard_rejects_coherent_asset_substitution(self):
        # Full self-consistent native rows used to pass the original shape
        # checks, while changing the preregistered expected answer.
        rows = self.synthetic_gate_fixture()
        family = protocol.EXPECTED_FINAL_FAMILIES[0]
        case = protocol.family_data(family, "0123456789abcdef" * 4)
        case["assets"][0]["asset_id"] += "-alternate"
        changed = protocol.make_example(
            family, case, "tool_selection", 0, protocol.OPS[0])
        changed["rendered"], changed["loss_char_spans"] = (
            protocol.render_with_mask(changed["turns"], self.b3))
        protocol.validate_example(changed, self.b3)
        audit.model_input_prefix(changed, self.b3)
        rows[0] = changed
        with self.assertRaisesRegex(
                protocol.ProtocolError, "deviates from frozen recipe"):
            audit.audit_future_gate_layout(rows, self.b3)

    def test_future_recipe_guard_rejects_coherent_restraint_label_swap(self):
        rows = self.synthetic_gate_fixture()
        changed = copy.deepcopy(rows[-1])
        self.assertEqual(changed["kind"], "restraint")
        changed["turns"][-1]["content"] = (
            "A generic harmless but different answer to this request.")
        changed["rendered"], changed["loss_char_spans"] = (
            protocol.render_with_mask(changed["turns"], self.b3))
        protocol.validate_example(changed, self.b3)
        audit.model_input_prefix(changed, self.b3)
        rows[-1] = changed
        with self.assertRaisesRegex(
                protocol.ProtocolError, "deviates from frozen recipe"):
            audit.audit_future_gate_layout(rows, self.b3)




    def test_guard_rejects_padded_or_endoftext_only_restraint(self):
        rows = self.synthetic_gate_fixture()
        restraint = next(row for row in rows if row["kind"] == "restraint")
        adapted = {row["id"]: row["turns"][-1]["content"] for row in rows}
        root = {row["id"]: "mock root baseline" for row in rows}
        for rogue in ("<|pad|>", "<|endoftext|>", "A sentence <think>"):
            with self.subTest(raw=rogue):
                # The frozen evaluator would count a synthetic control token
                # as valid no-tool restraint. Never rewrite the manifest-bound
                # protocol source to fix it after archive promotion.
                self.assertTrue(protocol.score_output(restraint, rogue, self.b3))
                contaminated = dict(adapted)
                contaminated[restraint["id"]] = rogue
                with self.assertRaisesRegex(
                        protocol.ProtocolError, "adapted unexpected native control"):
                    audit.guarded_admission_gate(rows, contaminated, root, self.b3)

    def test_guard_rejects_result_control_and_root_baseline_contamination(self):
        rows = self.synthetic_gate_fixture()
        success = next(row for row in rows if row["kind"] == "result_success")
        adapted = {row["id"]: row["turns"][-1]["content"] for row in rows}
        root = {row["id"]: "mock root baseline" for row in rows}
        leaked = adapted[success["id"]] + "<|endoftext|>"
        self.assertTrue(protocol.score_output(success, leaked, self.b3))
        changed = dict(adapted)
        changed[success["id"]] = leaked
        with self.assertRaisesRegex(
                protocol.ProtocolError, "adapted unexpected native control"):
            audit.guarded_admission_gate(rows, changed, root, self.b3)

        # Root must be checked too: a tainted control baseline can otherwise
        # distort the required >=4/9 research improvement comparison.
        root[success["id"]] = "mock root baseline<|pad|>"
        with self.assertRaisesRegex(
                protocol.ProtocolError, "root unexpected native control"):
            audit.guarded_admission_gate(rows, adapted, root, self.b3)

    def test_guard_accepts_single_terminal_yield_without_custody_claim(self):
        rows = self.synthetic_gate_fixture()
        adapted = {row["id"]: row["turns"][-1]["content"] + "<|end|>"
                   for row in rows}
        root = {row["id"]: "mock root baseline<|end|>" for row in rows}
        scored = audit.guarded_admission_gate(rows, adapted, root, self.b3)
        self.assertEqual(scored["status"], "research_candidate")
        self.assertFalse(scored["raw_output_execution_verified"])
        self.assertFalse(scored["independent_final_custody_verified"])
        self.assertFalse(scored["stage_b_evaluation_admitted"])
        self.assertFalse(scored["production_admission"])
        self.assertFalse(scored["population_expansion_admitted"])


    @classmethod
    def synthetic_untrusted_full_claim_fixture(cls, adapter_sha=None):
        """Synthetic NOT-AN-ADAPTER hash: no model, optimizer or real final."""
        adapter_sha = adapter_sha or ("c" * 64)
        cfg, b3 = protocol.load_contract()
        manifest_sha = protocol.verify_archive(protocol.DATASET, cfg, b3)
        seed = protocol.sha256_bytes(protocol.canonical({
            "recipe_sha": protocol.final_recipe_sha(cfg),
            "manifest_sha": manifest_sha,
            "adapter_dir_sha": adapter_sha,
            "salt": cfg["final_recipe"]["fixed_salt"],
        }).encode("utf-8"))
        tasks = []
        for family, extra in zip(
                protocol.EXPECTED_FINAL_FAMILIES,
                ("result_success", "result_denied", "restraint")):
            case = protocol.family_data(family, seed)
            for index, op in enumerate(protocol.OPS):
                tasks.append(protocol.make_example(
                    family, case, "tool_selection", index, op))
            tasks.append(protocol.make_example(family, case, extra, 3))
        for row in tasks:
            row["rendered"], row["loss_char_spans"] = protocol.render_with_mask(
                row["turns"], b3)
        claim = {"training_manifest_sha256": manifest_sha,
                 "adapter_dir_sha256": adapter_sha,
                 "post_hash_seed_sha256": seed}
        return tasks, claim

    def test_untrusted_full_posthash_claim_matches_frozen_recipe_only(self):
        tasks, claim = self.synthetic_untrusted_full_claim_fixture()
        result = audit.audit_untrusted_posthash_claim(tasks, claim, self.b3)
        self.assertEqual(result["status"], "claim_consistent_only")
        self.assertTrue(result["training_archive_verified"])
        self.assertTrue(result["full_seed_derivation_from_claims_verified"])
        self.assertFalse(result["adapter_source_and_optimizer_receipts_verified"])
        self.assertFalse(result["independent_posthash_final_custody_verified"])
        self.assertFalse(result["model_inputs_and_token_ids_verified"])
        self.assertFalse(result["training_admitted"])
        self.assertFalse(result["production_admission"])
        self.assertFalse(result["population_expansion_admitted"])

    def test_untrusted_claim_rejects_manifest_seed_and_prefix_substitution(self):
        tasks, claim = self.synthetic_untrusted_full_claim_fixture()
        changed = copy.deepcopy(claim)
        changed["training_manifest_sha256"] = "a" * 64
        with self.assertRaisesRegex(
                protocol.ProtocolError, "not the committed archive"):
            audit.audit_untrusted_posthash_claim(tasks, changed, self.b3)

        changed = copy.deepcopy(claim)
        changed["adapter_dir_sha256"] = "b" * 64
        with self.assertRaisesRegex(
                protocol.ProtocolError, "full posthash seed"):
            audit.audit_untrusted_posthash_claim(tasks, changed, self.b3)

        changed = copy.deepcopy(claim)
        changed["post_hash_seed_sha256"] = "d" * 64
        with self.assertRaisesRegex(
                protocol.ProtocolError, "full posthash seed"):
            audit.audit_untrusted_posthash_claim(tasks, changed, self.b3)

        changed = copy.deepcopy(claim)
        changed["post_hash_seed_sha256"] = "bad"
        with self.assertRaisesRegex(
                protocol.ProtocolError, "invalid full SHA-256"):
            audit.audit_untrusted_posthash_claim(tasks, changed, self.b3)

        # A coherent twelve-task record from another invented prefix passes
        # the old structural equality mirror, but not the full claimed seed.
        other, other_claim = self.synthetic_untrusted_full_claim_fixture(
            adapter_sha="e" * 64)
        audit.audit_future_gate_layout(other, self.b3)
        with self.assertRaisesRegex(
                protocol.ProtocolError, "future case suffix"):
            audit.audit_untrusted_posthash_claim(other, claim, self.b3)

        # This is deliberately NOT provenance: forged self-consistent claims
        # remain a structural match, never evidence of a trained adapter.
        still_untrusted = audit.audit_untrusted_posthash_claim(
            other, other_claim, self.b3)
        self.assertEqual(still_untrusted["status"], "claim_consistent_only")
        self.assertFalse(still_untrusted["adapter_source_and_optimizer_receipts_verified"])



if __name__ == "__main__":
    unittest.main()
