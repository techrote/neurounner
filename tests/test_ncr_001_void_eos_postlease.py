"""Offline fail-closed void.1 post-lease EOS source-contract tests, no model."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001_void_eos_postlease import (
    check_contract, validate_causal_eos, CASES, LANES,
)


class VoidPostLeaseSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "experiments/ncr-001/protocol.json").read_text())
        cls.pins = json.loads(
            (ROOT / "experiments/ncr-001/source-pins.json").read_text())

    def test_pins_and_training_gate_still_closed(self):
        self.assertEqual(set(self.protocol["models"]), LANES)
        p = check_contract(self.protocol, self.pins)
        self.assertEqual(p["weight_bytes"], 360622744)
        self.assertFalse(self.protocol["train_authorized"])

    def test_falsified_training_admission_rejected(self):
        x = copy.deepcopy(self.protocol)
        x["train_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "training gate"):
            check_contract(x, self.pins)

    def test_wrong_void_model_or_sprocket_alias_rejected(self):
        for mutation in ("revision", "weight_file", "repo_id"):
            x = copy.deepcopy(self.pins)
            x["models"]["void1"][mutation if mutation != "weight_file"
                                   else "hf_weight_file"] = "wrong"
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                check_contract(self.protocol, x)

    def test_refuse_removed_fourth_lane(self):
        x = copy.deepcopy(self.protocol)
        del x["models"]["sprocket_instruct"]
        with self.assertRaisesRegex(RuntimeError, "four-lane"):
            check_contract(x, self.pins)

    def encode_fixture(self, prompt, completion):
        prefix = list(prompt.encode("utf-8"))
        suffix = list(completion.encode("utf-8")) + [258]
        return {"input_ids": prefix + suffix,
                "attention_mask": [1] * (len(prefix) + len(suffix)),
                "labels": [-100] * len(prefix) + suffix,
                "causal_shift_supervised_units": len(suffix)}

    def test_all_three_source_only_eos_cases(self):
        for name, prompt, completion in CASES:
            with self.subTest(name=name):
                x = self.encode_fixture(prompt, completion)
                witness = validate_causal_eos(x, prompt, completion)
                self.assertEqual(witness["supervised_after_shift"],
                                 len(completion.encode("utf-8")) + 1)

    def test_plain_eos_string_bytes_are_not_terminal_id(self):
        good = self.encode_fixture("x", "<eos>")
        self.assertEqual(good["labels"][1:],
                         [60, 101, 111, 115, 62, 258])
        bad = copy.deepcopy(good)
        bad["labels"][-1] = 62
        with self.assertRaises(RuntimeError):
            validate_causal_eos(bad, "x", "<eos>")

    def test_causal_shift_empty_context_and_false_count_refused(self):
        empty = {"input_ids": [258], "attention_mask": [1],
                 "labels": [258], "causal_shift_supervised_units": 1}
        with self.assertRaises(RuntimeError):
            validate_causal_eos(empty, "", "")
        good = self.encode_fixture("x", "")
        good["causal_shift_supervised_units"] = 0
        with self.assertRaises(RuntimeError):
            validate_causal_eos(good, "x", "")

    def test_mask_prompt_leak_and_byte_count_corruption_refused(self):
        good = self.encode_fixture("Q:", "π🔧")
        for key, index, value in (
            ("attention_mask", 0, 0),
            ("labels", 0, 81),
            ("input_ids", 2, 258),
        ):
            bad = copy.deepcopy(good)
            bad[key][index] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_causal_eos(bad, "Q:", "π🔧")

    def test_no_comparative_training_path(self):
        source = (ROOT / "scripts/ncr_001_void_eos_postlease.py").read_text()
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn(".backward(", source)
        self.assertIn("full_72_conformance_pass", source)
        self.assertIn("False", source)


if __name__ == "__main__":
    unittest.main()
