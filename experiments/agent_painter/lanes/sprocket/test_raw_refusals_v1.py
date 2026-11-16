"""Model-free adversarial refusal taxonomy regressions; no DEVELOPMENT inputs loaded."""
from copy import deepcopy
import unittest

from raw_refusals_v1 import analyze, classify

GROUPS = (("greebles", "g1"), ("greebles", "g2"),
          ("pipework", "p1"), ("pipework", "p2"),
          ("architecture", "a1"), ("architecture", "a2"))

def fake(family, group, arm, mode="valid"):
    eos = mode != "no_eos"
    valid = mode in ("valid", "no_compile")
    json_ok = mode not in ("bad_json", "no_eos")
    ids = [111, 5] if eos else [111]
    compiled = mode == "valid"
    return {
        "arm": arm, "family": family, "structural_group": group,
        "prompt": "request " + group,
        "output_token_ids": ids, "output_tokens": len(ids),
        "raw": {
            "token_ids": ids[:], "raw_text": "{" if mode in ("bad_json", "no_eos") else "{}",
            "text_decoded": mode != "bad_decode", "eos_terminated": eos,
            "json_valid": json_ok or mode == "bad_decode",
            "program_valid": valid, "first_pass_accepted": valid and eos},
        "admitted_seed_programs": 2 if compiled else 0,
        "seed_results": [
            {"seed": seed, "first_pass_compiler_accepted": compiled,
             "native_execution": False, "native_acknowledged_cells": None,
             "native_png": None, "visual_quality_claim": False}
            for seed in (104729, 130363)]
    }

def rows(arm):
    modes = ("valid", "no_eos", "bad_json", "no_compile", "valid", "bad_decode")
    return [fake(f, g, arm, mode) for (f, g), mode in zip(GROUPS, modes)]

class RawRefusalRegression(unittest.TestCase):
    def setUp(self):
        self.early = {name: rows(name) for name in ("untuned", "step16")}
        self.late = {name: rows(name) for name in ("untuned", "step96")}

    def test_complete_original_denominators(self):
        report = analyze(self.early, self.late)
        self.assertEqual(report["unique_requests"], 6)
        self.assertEqual(report["arms"]["step96"]["seed_variants"], 12)
        self.assertEqual(report["arms"]["step96"]["compiled_seed_variants"], 4)
        self.assertEqual(report["new_native_paintings"], 0)
        self.assertFalse(report["final_opened"])

    def test_no_eos_refusal(self):
        self.assertEqual(classify(fake("greebles", "g", "step96", "no_eos"))["category"], "no_terminal_eos")

    def test_invalid_json_refusal(self):
        self.assertEqual(classify(fake("greebles", "g", "step96", "bad_json"))["category"], "invalid_json")

    def test_compiler_refusal_not_native_success(self):
        self.assertEqual(classify(fake("pipework", "p", "step96", "no_compile"))["category"], "compile_refusal")

    def test_altered_prompt_fails(self):
        self.late["step96"][0]["prompt"] += " new words"
        with self.assertRaisesRegex(ValueError, "different DEVELOPMENT"):
            analyze(self.early, self.late)

    def test_wrong_family_fails(self):
        self.late["step96"][0]["family"] = "isometric"
        with self.assertRaisesRegex(ValueError, "three families"):
            analyze(self.early, self.late)

    def test_missing_group_fails(self):
        self.late["step96"].pop()
        with self.assertRaisesRegex(ValueError, "six independent"):
            analyze(self.early, self.late)

    def test_raw_token_tamper_fails(self):
        row = fake("greebles", "g", "untuned")
        row["raw"]["token_ids"] = [999, 5]
        with self.assertRaisesRegex(ValueError, "raw/generated"):
            classify(row)

    def test_forged_eos_fails(self):
        row = fake("greebles", "g", "untuned")
        row["output_token_ids"][-1] = 11
        row["raw"]["token_ids"][-1] = 11
        with self.assertRaisesRegex(ValueError, "terminal EOS"):
            classify(row)


    def test_false_eos_flag_with_true_terminal_id_fails(self):
        row = fake("greebles", "g", "untuned", "no_eos")
        row["output_token_ids"] = [111, 5]
        row["raw"]["token_ids"] = [111, 5]
        row["output_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "terminal EOS flag"):
            classify(row)

    def test_json_flag_false_for_valid_raw_text_fails(self):
        row = fake("greebles", "g", "step96", "bad_json")
        row["raw"]["raw_text"] = "{}"
        with self.assertRaisesRegex(ValueError, "strict JSON flag"):
            classify(row)

    def test_json_flag_true_for_invalid_raw_text_fails(self):
        row = fake("greebles", "g", "step96")
        row["raw"]["raw_text"] = "{"
        with self.assertRaisesRegex(ValueError, "strict JSON flag"):
            classify(row)

    def test_duplicate_json_object_keys_refused(self):
        row = fake("greebles", "g", "step96")
        row["raw"]["raw_text"] = '{"x":1,"x":2}'
        with self.assertRaisesRegex(ValueError, "strict JSON flag"):
            classify(row)

    def test_nonfinite_json_constants_refused(self):
        row = fake("greebles", "g", "step96")
        row["raw"]["raw_text"] = '{"x":NaN}'
        with self.assertRaisesRegex(ValueError, "strict JSON flag"):
            classify(row)

    def test_nested_duplicate_json_keys_refused(self):
        row = fake("greebles", "g", "step96")
        row["raw"]["raw_text"] = '{"outer":{"a":1,"a":2}}'
        with self.assertRaisesRegex(ValueError, "strict JSON flag"):
            classify(row)

    def test_false_native_pass_fails(self):
        row = fake("architecture", "a", "step16")
        row["seed_results"][0]["native_execution"] = True
        with self.assertRaisesRegex(ValueError, "misrepresented"):
            classify(row)

    def test_wrong_compiler_denominator_fails(self):
        row = fake("pipework", "p", "step16", "no_compile")
        row["admitted_seed_programs"] = 2
        with self.assertRaisesRegex(ValueError, "unqualified compiler"):
            classify(row)

if __name__ == "__main__":
    unittest.main()
