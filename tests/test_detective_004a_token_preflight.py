"""No-training token-mask preflight tests: artificial tokenizer only.

These tests CANNOT qualify the real root Sprocket tokenizer or authorize Stage B.
They exercise the exact pinned publisher role-by-role label semantics and
identify errors that character-span-only Stage-A checks cannot detect.
"""
from __future__ import annotations

import copy
import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import detective_004a_protocol as protocol
from scripts import detective_004a_token_preflight as preflight


class ToyTokenizer:
    """Deterministic non-model test double, NEVER a frozen-tokenizer substitute."""
    PIECES = re.compile(r"<\|[a-z_]+\|>|<think>|</think>|[^\s<]+|<|\s+")

    def __init__(self, spec):
        self.tokens = dict(spec)

    def token_to_id(self, marker):
        return self.tokens.get(marker)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("test tokenizer refuses implicit extra tokens")
        parts = self.PIECES.findall(text)
        if "".join(parts) != text:
            raise AssertionError("toy tokenization did not preserve input text")
        out = []
        for piece in parts:
            if piece in self.tokens:
                out.append(self.tokens[piece])
            else:
                out.append(1000 + int.from_bytes(
                    hashlib.sha256(piece.encode("utf-8")).digest()[:4], "big"))
        return SimpleNamespace(ids=out)


class Detective004ATokenPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.b3 = protocol.load_contract()
        cls.rows = protocol.build_training(cls.cfg, cls.b3)
        cls.toy = ToyTokenizer(cls.b3["special_token_ids"])

    def test_every_frozen_row_structurally_checks_with_artificial_tokenizer(self):
        report = preflight.inspect_archive_with_tokenizer(self.toy)
        self.assertEqual(report["rows_checked"], 36)
        self.assertEqual(len(report["ledger"]), 36)
        self.assertLessEqual(report["max_tokens_seen"], 512)
        self.assertEqual(report["max_allowed_tokens"], 512)
        self.assertFalse(report["real_frozen_tokenizer_identity_verified"])
        self.assertFalse(report["independent_stage_a_review_verified"])
        self.assertFalse(report["stage_b_training_admitted"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["model_inferences"], 0)
        self.assertFalse(report["population_expansion_admitted"])

    def test_source_native_assistant_only_mask_and_shift(self):
        for row in self.rows:
            with self.subTest(id=row["id"]):
                ids, mask, regions = preflight.source_role_ids(
                    row["turns"], self.toy, self.b3["special_token_ids"])
                self.assertEqual(len(ids), len(mask))
                self.assertTrue(mask[-1])
                self.assertEqual(ids[-1], 5)
                self.assertEqual(
                    sum(region["role"] == "assistant" for region in regions),
                    2 if row["kind"].startswith("result_") else 1)
                for region in regions:
                    self.assertFalse(mask[region["header"]])
                    self.assertEqual(
                        mask[region["end"] - 1],
                        region["role"] == "assistant")
                    if region["role"] == "tool":
                        self.assertEqual(ids[region["header"]], 7)
                        self.assertFalse(any(mask[region["start"]:region["end"]]))
                receipt = preflight.inspect_row(row, self.b3, self.toy)
                self.assertEqual(receipt["num_tokens"], len(ids))
                self.assertEqual(receipt["trained_tokens"], sum(mask))
                self.assertEqual(receipt["trained_tokens"],
                                 receipt["shifted_train_labels"])
                self.assertLess(receipt["prompt_tokens"], receipt["num_tokens"])
                self.assertEqual(len(receipt["token_ids_sha256"]), 64)
                self.assertEqual(len(receipt["train_mask_sha256"]), 64)

    def test_publisher_special_token_id_drift_refused(self):
        mutated = ToyTokenizer(self.b3["special_token_ids"])
        mutated.tokens["<|tool_call|>"] = 9006
        with self.assertRaisesRegex(protocol.ProtocolError, "special-token identity"):
            preflight.check_special_ids(mutated, self.b3["special_token_ids"])

    def test_source_runner_whole_wire_vs_publisher_turnwise_mismatch_refused(self):
        class IncompatibleWireTokenizer(ToyTokenizer):
            def encode(self, text, add_special_tokens=False):
                result = super().encode(text, add_special_tokens=add_special_tokens)
                if text.startswith("<|system|>") and "<|user|>" in text:
                    result.ids.append(987654321)  # malformed whole-wire encoding
                return result

        wrong = IncompatibleWireTokenizer(self.b3["special_token_ids"])
        with self.assertRaisesRegex(protocol.ProtocolError, "whole-wire B3 inference IDs"):
            preflight.inspect_row(self.rows[0], self.b3, wrong)

    def test_token_budget_refuses_without_truncating(self):
        class InflatedTokenizer(ToyTokenizer):
            def encode(self, text, add_special_tokens=False):
                result = super().encode(text, add_special_tokens=add_special_tokens)
                if text not in self.tokens:
                    result.ids *= 10
                return result

        wrong = InflatedTokenizer(self.b3["special_token_ids"])
        with self.assertRaisesRegex(protocol.ProtocolError, "exceeds 512 tokens"):
            preflight.inspect_row(self.rows[0], self.b3, wrong)

    def test_opt_in_raw_array_receipt_is_complete_but_not_real_model_evidence(self):
        row = next(r for r in self.rows if r["kind"] == "result_success")
        receipt = preflight.inspect_row(row, self.b3, self.toy, include_arrays=True)
        raw = receipt["raw_arrays"]
        self.assertEqual(set(raw), {
            "full_input_ids", "trained_token_mask", "causal_input_ids",
            "causal_target_labels", "model_only_prompt_ids", "model_only_prompt_utf8",
            "hf_forward_labels", "hf_effective_shift_labels"})
        self.assertEqual(len(raw["full_input_ids"]), len(raw["trained_token_mask"]))
        self.assertEqual(raw["causal_input_ids"], raw["full_input_ids"][:-1])
        self.assertEqual(raw["model_only_prompt_ids"][-1], 4)
        self.assertTrue(raw["trained_token_mask"][-1])
        self.assertEqual(raw["full_input_ids"][-1], 5)
        self.assertIn("<|tool_result|>", raw["model_only_prompt_utf8"])
        self.assertNotIn(row["turns"][-1]["content"], raw["model_only_prompt_utf8"])
        for item, digest in (
                ("full_input_ids", "token_ids_sha256"),
                ("trained_token_mask", "train_mask_sha256"),
                ("causal_target_labels", "shifted_labels_sha256"),
                ("model_only_prompt_ids", "prompt_ids_sha256"),
                ("hf_forward_labels", "hf_forward_labels_sha256"),
                ("hf_effective_shift_labels", "hf_effective_shift_labels_sha256")):
            value = protocol.canonical(raw[item]).encode("utf-8")
            self.assertEqual(hashlib.sha256(value).hexdigest(), receipt[digest])
        self.assertEqual(
            hashlib.sha256(raw["model_only_prompt_utf8"].encode("utf-8")).hexdigest(),
            receipt["prompt_utf8_sha256"])

    def test_non_local_or_wrong_hash_tokenizer_refused_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name, _ in preflight.LOCAL_TOKENIZER_FILES:
                (folder / name).write_bytes(b"not the pinned root tokenizer")
            with self.assertRaisesRegex(protocol.ProtocolError, "SHA drift"):
                preflight.inspect_local_frozen_tokenizer(folder)
            with self.assertRaisesRegex(protocol.ProtocolError, "local tokenizer directory"):
                preflight.inspect_local_frozen_tokenizer(folder / "not-found")


    def test_single_example_step_plan_covers_frozen_36_without_packing(self):
        plan = preflight.planned_single_example_steps(self.toy)
        steps = plan["steps"]
        self.assertEqual(len(steps), 36)
        self.assertEqual([item["step_1based"] for item in steps],
                         list(range(1, 37)))
        self.assertEqual([item["row_id"] for item in steps],
                         [row["id"] for row in self.rows])
        self.assertEqual(len({item["row_id"] for item in steps}), 36)
        self.assertTrue(plan["no_packing_or_truncation"])
        self.assertFalse(plan["real_frozen_tokenizer_identity_verified"])
        self.assertFalse(plan["optimizer_execution_verified"])
        self.assertFalse(plan["training_admitted"])
        self.assertTrue(all(0 < item["trained_tokens"] < item["num_tokens"] <= 512
                            for item in steps))

    def test_claimed_36_step_ledger_is_only_structural_evidence(self):
        steps = preflight.planned_single_example_steps(self.toy)["steps"]
        receipt = preflight.audit_claimed_training_steps(self.toy, steps)
        self.assertEqual(receipt["claimed_step_count"], 36)
        self.assertEqual(len(receipt["claimed_step_ledger_sha256"]), 64)
        self.assertTrue(receipt["structurally_exact_single_example_steps"])
        self.assertFalse(receipt["real_frozen_tokenizer_identity_verified"])
        self.assertFalse(receipt["actual_optimizer_batches_independently_attested"])
        self.assertFalse(receipt["adapter_ancestry_verified"])
        self.assertFalse(receipt["stage_b_training_admitted"])
        self.assertFalse(receipt["population_expansion_admitted"])

    def test_claimed_step_ledger_refuses_duplicates_order_drift_and_packing(self):
        original = preflight.planned_single_example_steps(self.toy)["steps"]
        for name, transform, message in (
                ("missing", lambda x: x.pop(), "exactly 36"),
                ("duplicate", lambda x: x.__setitem__(1, copy.deepcopy(x[0])),
                 "duplicate, missing or reordered"),
                ("reordered", lambda x: x.reverse(), "duplicate, missing or reordered"),
                ("mask drift", lambda x: x[0].update(train_mask_sha256="0" * 64),
                 "per-step"),
                ("label drift", lambda x: x[0].update(shifted_labels_sha256="f" * 64),
                 "per-step"),
                ("packed row", lambda x: x[0].update(num_tokens=x[0]["num_tokens"] + 17),
                 "per-step"),
                ("prompt drift", lambda x: x[0].update(prompt_ids_sha256="b" * 64),
                 "per-step"),
        ):
            with self.subTest(name=name):
                claim = copy.deepcopy(original)
                transform(claim)
                with self.assertRaisesRegex(protocol.ProtocolError, message):
                    preflight.audit_claimed_training_steps(self.toy, claim)



    def test_frozen_optimizer_preregistration_gap_is_explicit_and_blocking(self):
        gap = preflight.audit_optimizer_preregistration(self.cfg)
        self.assertEqual(
            gap["missing_preregistered_fields"],
            list(preflight.REQUIRED_OPTIMIZER_PREREG_FIELDS))
        self.assertFalse(gap["optimizer_execution_contract_complete"])
        self.assertEqual(gap["frozen_optimizer"], "AdamW")
        self.assertEqual(gap["frozen_learning_rate"], 0.0002)
        self.assertEqual(gap["frozen_optimizer_steps"], 36)
        self.assertEqual(gap["frozen_epochs"], 1)
        self.assertEqual(gap["frozen_rank"], 8)
        self.assertEqual(gap["frozen_target_modules"], ["q_proj", "v_proj"])
        self.assertFalse(gap["values_independently_reviewed"])
        self.assertFalse(gap["optimizer_execution_verified"])
        self.assertFalse(gap["stage_b_training_admitted"])
        self.assertFalse(gap["population_expansion_admitted"])

    def test_36_step_plan_carries_optimizer_preregistration_blocker(self):
        plan = preflight.planned_single_example_steps(self.toy)
        self.assertFalse(plan["optimizer_preregistration_complete"])
        self.assertEqual(
            plan["missing_optimizer_preregistered_fields"],
            list(preflight.REQUIRED_OPTIMIZER_PREREG_FIELDS))
        receipt = preflight.audit_claimed_training_steps(
            self.toy, copy.deepcopy(plan["steps"]))
        self.assertFalse(receipt["optimizer_preregistration_complete"])
        self.assertEqual(
            receipt["missing_optimizer_preregistered_fields"],
            list(preflight.REQUIRED_OPTIMIZER_PREREG_FIELDS))
        self.assertFalse(receipt["stage_b_training_admitted"])

    def test_field_presence_alone_never_authorizes_optimizer(self):
        cfg = copy.deepcopy(self.cfg)
        # Synthetic placeholders only: this test proves that even a structurally
        # complete field set is not an independent semantic review or authority.
        placeholders = {
            "training_seed": 0,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "adam_beta1": 0.0,
            "adam_beta2": 0.0,
            "adam_epsilon": 0.0,
            "weight_decay": 0.0,
            "lr_scheduler": "synthetic-test-only",
            "warmup_steps": 0,
        }
        cfg["future_training"].update(placeholders)
        gap = preflight.audit_optimizer_preregistration(cfg)
        self.assertEqual(gap["missing_preregistered_fields"], [])
        self.assertEqual(gap["invalid_claimed_fields"], ["adam_epsilon"])
        self.assertFalse(gap["claimed_field_set_structurally_valid"])
        self.assertFalse(gap["frozen_config_exact_match"])
        self.assertTrue(gap["untrusted_config_override"])
        self.assertFalse(gap["optimizer_execution_contract_complete"])
        self.assertFalse(gap["values_independently_reviewed"])
        self.assertFalse(gap["optimizer_execution_verified"])
        self.assertFalse(gap["adapter_ancestry_verified"])
        self.assertFalse(gap["stage_b_training_admitted"])
        self.assertFalse(gap["production_admission"])

    def test_valid_synthetic_parameter_values_cannot_spoof_frozen_preregistration(self):
        cfg = copy.deepcopy(self.cfg)
        # These are illustrative test-only values, NOT preregistered choices.
        cfg["future_training"].update({
            "training_seed": 14836,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.01,
            "lr_scheduler": "constant",
            "warmup_steps": 0,
        })
        gap = preflight.audit_optimizer_preregistration(cfg)
        self.assertEqual(gap["missing_preregistered_fields"], [])
        self.assertEqual(gap["invalid_claimed_fields"], [])
        self.assertTrue(gap["claimed_field_set_structurally_valid"])
        self.assertFalse(gap["frozen_config_exact_match"])
        self.assertFalse(gap["optimizer_execution_contract_complete"])
        self.assertFalse(gap["values_independently_reviewed"])
        self.assertFalse(gap["stage_b_training_admitted"])

    def test_invalid_synthetic_optimizer_claims_fail_closed(self):
        cfg = copy.deepcopy(self.cfg)
        values = {
            "training_seed": True,     # bool must not count as an integer seed
            "batch_size": 2,           # violates 36 single-example steps
            "gradient_accumulation_steps": 2,
            "adam_beta1": float("inf"),
            "adam_beta2": -0.5,
            "adam_epsilon": 0,
            "weight_decay": -0.01,
            "lr_scheduler": "",
            "warmup_steps": 37,
        }
        cfg["future_training"].update(values)
        gap = preflight.audit_optimizer_preregistration(cfg)
        self.assertEqual(gap["missing_preregistered_fields"], [])
        self.assertEqual(gap["invalid_claimed_fields"],
                         list(preflight.REQUIRED_OPTIMIZER_PREREG_FIELDS))
        self.assertFalse(gap["claimed_field_set_structurally_valid"])
        self.assertFalse(gap["optimizer_execution_contract_complete"])
        self.assertFalse(gap["stage_b_training_admitted"])



    def test_hf_llama_forward_labels_match_publisher_mask_after_one_shift(self):
        # Native GPT.forward has externally shifted labels/-1 ignore; the
        # source-exported LlamaForCausalLM uses full-width labels/-100 ignore
        # and shifts internally. This exercises arrays only, never a model.
        row = next(r for r in self.rows if r["kind"] == "result_success")
        receipt = preflight.inspect_row(row, self.b3, self.toy, include_arrays=True)
        raw = receipt["raw_arrays"]
        ids, mask, regions = preflight.source_role_ids(
            row["turns"], self.toy, self.b3["special_token_ids"])
        native_y = raw["causal_target_labels"]
        hf_labels = raw["hf_forward_labels"]
        hf_shifted = raw["hf_effective_shift_labels"]
        self.assertEqual(len(hf_labels), len(ids))
        self.assertEqual(len(hf_shifted), len(ids))
        self.assertEqual(hf_labels,
                         [tid if active else -100 for tid, active in zip(ids, mask)])
        self.assertEqual(
            hf_shifted, [(-100 if t == -1 else t) for t in native_y] + [-100])
        self.assertEqual(sum(t != -100 for t in hf_labels),
                         receipt["trained_tokens"])
        self.assertEqual(sum(t != -100 for t in hf_shifted),
                         receipt["shifted_train_labels"])
        self.assertEqual(hf_labels[0], -100)  # host header never supervised
        self.assertEqual(hf_labels[-1], 5)  # model yield trained
        self.assertEqual(receipt["publisher_ignore_index"], -1)
        self.assertEqual(receipt["hf_llama_ignore_index"], -100)
        self.assertTrue(receipt["hf_internal_shift_equivalence_checked"])
        self.assertEqual(hf_labels[regions[2]["content_start"]], 6)
        self.assertEqual(hf_labels[regions[3]["header"]], -100)
        self.assertEqual(hf_labels[regions[4]["header"]], -100)

    def test_untrusted_hf_label_claim_refuses_native_minus_one_and_double_shift(self):
        row = next(r for r in self.rows if r["kind"] == "tool_selection")
        ids, mask, _ = preflight.source_role_ids(
            row["turns"], self.toy, self.b3["special_token_ids"])
        native_y = [ids[i + 1] if mask[i + 1] else -1
                    for i in range(len(ids) - 1)]
        exact, effective = preflight.hf_forward_supervision(ids, mask, native_y)
        self.assertEqual(
            preflight.hf_forward_supervision(
                ids, mask, native_y, claimed_hf_labels=exact), (exact, effective))
        bad = [
            native_y + [-1],  # publisher's shifted/ignore=-1 arrays
            [-100] + native_y,  # double-shifted label vector, wrong ignore values
            [x if x != -100 else -1 for x in exact],  # native ignore value
            list(ids),  # accidental training on host/user text
        ]
        for labels in bad:
            with self.subTest(labels=labels[:4]):
                self.assertNotEqual(labels, exact)
                with self.assertRaisesRegex(
                        protocol.ProtocolError, "full-width, unshifted"):
                    preflight.hf_forward_supervision(
                        ids, mask, native_y, claimed_hf_labels=labels)
        wrong_native = native_y.copy()
        wrong_native[0] = 9999
        with self.assertRaisesRegex(
                protocol.ProtocolError, "publisher externally shifted"):
            preflight.hf_forward_supervision(ids, mask, wrong_native)

    def test_36_step_plan_binds_both_native_and_hf_loss_label_digests(self):
        plan = preflight.planned_single_example_steps(self.toy)
        self.assertFalse(plan["loss_backend_selected_and_independently_qualified"])
        self.assertEqual(len(plan["steps"]), 36)
        for step in plan["steps"]:
            self.assertEqual(len(step["shifted_labels_sha256"]), 64)
            self.assertEqual(len(step["hf_forward_labels_sha256"]), 64)
            self.assertEqual(len(step["hf_effective_shift_labels_sha256"]), 64)
        mutation = copy.deepcopy(plan["steps"])
        mutation[0]["hf_forward_labels_sha256"] = "0" * 64
        with self.assertRaisesRegex(protocol.ProtocolError, "per-step"):
            preflight.audit_claimed_training_steps(self.toy, mutation)


if __name__ == "__main__":
    unittest.main()
