"""Model-free, network-free fail-closed checks for original VOID raw refusal audit."""
import copy
import hashlib
import json
import unittest
from raw_firstpass_audit import verify_raw, category, canonical_digest, independent_strict_json_valid


class OriginalRefusalAuditTest(unittest.TestCase):
    def setUp(self):
        self.raw = {"raw_text": "{}", "token_ids": [123, 125, 258],
                    "eos_terminated": True, "json_valid": True,
                    "program_valid": False, "first_pass_accepted": False,
                    "failure": "unknown component"}
        self.receipt = {"generated_tokens": 3, "eos_terminated": True,
                        "json_valid": True, "program_valid": False,
                        "first_pass_accepted": False, "compiler_accepted": False,
                        "completion_sha256": hashlib.sha256(b"{}").hexdigest(),
                        "operations_sha256": None, "native_execution": False,
                        "native_acknowledged_cells": None, "visual_quality_claim": False}

    def check(self, raw=None, receipt=None, text=b"{}", batch=b"[]\n"):
        return verify_raw(self.raw if raw is None else raw,
                          self.receipt if receipt is None else receipt, text, batch)

    def test_original_rejection(self):
        self.assertEqual(self.check()["category"], "json_valid_compiler_rejected")

    def test_malformed_utf8_refused(self):
        x = copy.deepcopy(self.raw)
        x["raw_text"] = "\ufffd"
        x["token_ids"] = [255, 258]
        with self.assertRaises((ValueError, UnicodeDecodeError)):
            self.check(raw=x, text="\ufffd".encode("utf8"))

    def test_forged_eos_refused(self):
        x = copy.deepcopy(self.raw)
        x["token_ids"] = [123, 125]
        with self.assertRaises(ValueError):
            self.check(raw=x)

    def test_special_token_injection_refused(self):
        x = copy.deepcopy(self.raw)
        x["token_ids"] = [123, 125, 257, 258]
        with self.assertRaises(ValueError):
            self.check(raw=x)

    def test_raw_text_change_refused(self):
        with self.assertRaises(ValueError):
            self.check(text=b"{ } ")

    def test_rejected_native_operations_refused(self):
        with self.assertRaises(ValueError):
            self.check(batch=b'["paint"]')

    def test_forged_native_ack_refused(self):
        r = copy.deepcopy(self.receipt)
        r["native_execution"] = True
        with self.assertRaises(ValueError):
            self.check(receipt=r)

    def test_json_compiler_chain_refused(self):
        x = copy.deepcopy(self.raw)
        x["program_valid"] = True
        with self.assertRaises(ValueError):
            self.check(raw=x)

    def test_duplicate_key_refusal_category(self):
        x = copy.deepcopy(self.raw)
        x.update(json_valid=False, failure="duplicate JSON key: kind")
        self.assertEqual(category(x), "duplicate_json_key_rejected")

    def test_forged_json_true_on_malformed_payload_refused(self):
        x = copy.deepcopy(self.raw)
        x["raw_text"], x["token_ids"] = "{", [123, 258]
        r = copy.deepcopy(self.receipt)
        r["generated_tokens"] = 2
        r["completion_sha256"] = hashlib.sha256(b"{").hexdigest()
        with self.assertRaisesRegex(ValueError, "independent strict parser"):
            self.check(raw=x, receipt=r, text=b"{")

    def test_forged_json_false_on_valid_payload_refused(self):
        x, r = copy.deepcopy(self.raw), copy.deepcopy(self.receipt)
        x["json_valid"] = r["json_valid"] = False
        with self.assertRaisesRegex(ValueError, "independent strict parser"):
            self.check(raw=x, receipt=r)

    def test_strict_json_duplicate_and_nonfinite_refused(self):
        for raw in ('{"k":1,"k":2}', '{"n":NaN}', '{"n":Infinity}'):
            self.assertFalse(independent_strict_json_valid(raw))
        self.assertTrue(independent_strict_json_valid('{"k":1,"n":[1,2]}'))

    def test_forged_compiler_acceptance_flag_refused(self):
        r = copy.deepcopy(self.receipt)
        r["compiler_accepted"] = True
        with self.assertRaisesRegex(ValueError, "compiled acceptance flag"):
            self.check(receipt=r)

    def test_forged_completion_digest_refused(self):
        r = copy.deepcopy(self.receipt)
        r["completion_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "completion SHA-256"):
            self.check(receipt=r)

    def test_forged_accepted_without_eos_refused(self):
        x, r = copy.deepcopy(self.raw), copy.deepcopy(self.receipt)
        x.update(token_ids=[123, 125], eos_terminated=False,
                 program_valid=True, first_pass_accepted=True)
        r.update(generated_tokens=2, eos_terminated=False, program_valid=True,
                 first_pass_accepted=True, compiler_accepted=True)
        with self.assertRaisesRegex(ValueError, "true EOS"):
            self.check(raw=x, receipt=r)

    def test_positive_compiled_operation_receipt(self):
        x, r = copy.deepcopy(self.raw), copy.deepcopy(self.receipt)
        x.update(program_valid=True, first_pass_accepted=True, failure=None)
        operations = [{"rect": [1, 2, 3, 4], "material": 58}]
        digest = canonical_digest(operations)
        r.update(program_valid=True, first_pass_accepted=True,
                 compiler_accepted=True, operations_sha256=digest)
        batch = json.dumps(operations).encode("utf-8")
        compiled = json.dumps({"operations": operations,
                               "operations_sha256": digest}).encode("utf-8")
        actual = verify_raw(x, r, b"{}", batch, compiled_bytes=compiled)
        self.assertTrue(actual["compiler_accepted"])
        with self.assertRaisesRegex(ValueError, "compiled/receipt SHA-256"):
            verify_raw(x, r, b"{}", batch, compiled_bytes=b'{"operations":[]}')

    def test_off_task_prose_category(self):
        x = copy.deepcopy(self.raw)
        x.update(json_valid=False, raw_text="\nQuestion: why?", failure="Expecting value")
        self.assertEqual(category(x), "off_task_non_json_text")


if __name__ == "__main__":
    unittest.main()
