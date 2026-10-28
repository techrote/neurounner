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
            "causal_target_labels", "model_only_prompt_ids", "model_only_prompt_utf8"})
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
                ("model_only_prompt_ids", "prompt_ids_sha256")):
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



if __name__ == "__main__":
    unittest.main()
