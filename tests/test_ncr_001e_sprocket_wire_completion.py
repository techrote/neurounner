"""No-model fail-closed regression tests for the bounded PR #102 completion."""
import copy
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ncr_001e_sprocket_publisher_wire as wire
import ncr_001e_sprocket_wire_diagnostic as diagnostic


class Tokenizer:
    def __init__(self, fault=None):
        self.fault = fault
        self.split = re.compile("(" + "|".join(re.escape(x) for x in wire.IDS) + ")")
    def encode(self, text, add_special_tokens=False):
        out = []
        for item in self.split.split(text):
            if item in wire.IDS:
                out.append(wire.IDS[item])
            else:
                out.extend(100 + ord(c) for c in item)
        if self.fault == "body" and text == "x":
            return [0]
        if self.fault == "joined" and text.startswith("<|user|>") and len(text) > 10:
            return out + [99]
        return out
    def convert_tokens_to_ids(self, marker):
        return wire.IDS[marker]
    def convert_ids_to_tokens(self, value):
        return next(k for k, v in wire.IDS.items() if v == value)


class Oracle:
    def render(self, rows):
        return [3, 220, 5], [False] * 3
    def render_prompt(self, rows):
        return [3, 220, 5, 4]


class CompletionTests(unittest.TestCase):
    def test_all_twelve_reserved_tokens_rejected_in_host_content(self):
        self.assertEqual(set(wire.IDS.values()), set(range(12)))
        for role in ("system", "user", "memory", "tool"):
            for marker in wire.IDS:
                with self.subTest(role=role, marker=marker), self.assertRaises(ValueError):
                    wire.render_text([diagnostic.turn(role, "x" + marker + "y")], variant="instruct")

    def test_eot_pad_and_unicode_rejected_inside_assistant(self):
        for text in ("x<|pad|>y", "x<|endoftext|>y", "bad\ud800"):
            with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                wire.render_text([diagnostic.turn("assistant", text), diagnostic.turn("user", "next")], variant="instruct")

    def test_json_overflow_nonfinite_duplicate_and_envelope_refusal(self):
        for body in ('{"name":"x","arguments":{"x":1e309}}',
                     '{"name":"x","arguments":{"x":-1e309}}',
                     '{"name":"x","arguments":{"x":NaN}}',
                     '{"name":"x","arguments":{"x":1,"x":2}}',
                     '{"name":" ","arguments":{}}',
                     '{"name":"x","arguments":{},"extra":1}'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                wire._valid_assistant("<|tool_call|>" + body)
        wire._valid_assistant('<|tool_call|>{"name":"x","arguments":{"x":1e3}}')

    def test_hidden_tokenizer_control_ids_and_join_drift_rejected(self):
        for fault in ("body", "joined"):
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                wire.render_ids_mask([diagnostic.turn("user", "x")], Tokenizer(fault), variant="instruct")

    def test_publisher_oracle_ids_and_mask_must_both_match(self):
        rows = [diagnostic.turn("user", "x")]
        ids, mask = diagnostic.parity(Oracle(), Tokenizer(), rows)
        self.assertEqual((ids, mask), ([3, 220, 5, 4], [False] * 4))
        bad = Oracle()
        bad.render = lambda _: ([3, 220, 5], [False, True, False])
        with self.assertRaisesRegex(RuntimeError, "masks"):
            diagnostic.parity(bad, Tokenizer(), rows)
        bad = Oracle()
        bad.render_prompt = lambda _: [3, 220, 5, 99]
        with self.assertRaisesRegex(RuntimeError, "IDs"):
            diagnostic.parity(bad, Tokenizer(), rows)

    def test_publisher_source_bytes_checked_before_execution(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "untrusted.py"
            path.write_text("raise AssertionError('must not execute')")
            with self.assertRaisesRegex(RuntimeError, "immutable bytes"):
                diagnostic.load_publisher_oracle(path, "unused", None)

    def test_fixed_plan_seven_cases_and_no_quality_threshold(self):
        plan = diagnostic.diagnostic_plan()
        self.assertEqual(len(plan["cases"]), 7)
        self.assertEqual(len({x["id"] for x in plan["cases"]}), 7)
        self.assertEqual(plan["max_new_units"], 64)
        self.assertEqual(plan["stop_ids"], [0, 5])
        self.assertEqual(len(plan["modes"]), 3)
        self.assertIsNone(plan["quality_acceptance_threshold"])
        self.assertIs(plan["train_authorized"], False)
        for case in plan["cases"]:
            wire.render_text(case["turns"], variant="instruct")
        wire.render_text(plan["rich_turns"], variant="instruct")

    def test_observations_are_literal_not_semantic_scores(self):
        case = {"literal": "42"}
        self.assertFalse(diagnostic.observe(case, "142")['expected_literal_present'])
        self.assertTrue(diagnostic.observe(case, "not 42")['expected_literal_present'])
        self.assertFalse(diagnostic.observe({}, '<|tool_call|>{bad}')['strict_tool_envelope'])
        self.assertTrue(diagnostic.observe({}, diagnostic.CALL)['strict_tool_envelope'])

    def test_chat_remains_distinct(self):
        with self.assertRaises(ValueError):
            wire.render_text([diagnostic.turn("user", "x")], variant="chat")


if __name__ == "__main__":
    unittest.main()
