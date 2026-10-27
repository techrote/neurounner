"""Publisher-sourced no-network Sprocket wire and loss-mask regression tests."""
from __future__ import annotations
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ncr_001e_sprocket_publisher_wire import (
    IDS, SOURCE_MARKERS, render_ids_mask, render_text
)


class MockTokenizer:
    """Character-level synthetic text, but exact dedicated publisher controls."""
    def __init__(self, broken=None):
        self.broken = broken
        self.splitter = re.compile("(" + "|".join(re.escape(x) for x in SOURCE_MARKERS) + ")")
    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in self.splitter.split(text):
            if not part:
                continue
            if part in IDS:
                ids.append(IDS[part])
            else:
                ids.extend(100 + ord(c) for c in part)
        if self.broken == "split_tool" and text == "<|tool_call|>":
            return [23, 42]
        if self.broken == "fused_dialogue" and text.startswith("<|system|>"):
            return [999] + ids
        return ids
    def convert_tokens_to_ids(self, token):
        return -1 if token == self.broken else IDS[token]
    def convert_ids_to_tokens(self, token):
        return next((k for k, v in IDS.items() if v == token), None)


def turn(role, content):
    return {"role": role, "content": content}


class WireTests(unittest.TestCase):
    def test_source_role_headers_model_actions_host_result_and_mask(self):
        rows = [
            turn("system", "Provide instrument readings."),
            turn("memory", "The user prefers Celsius."),
            turn("user", "Read channel A."),
            turn("assistant", '<|tool_call|>{"name":"read_meter","arguments":{"channel":"A"}}'),
            turn("tool", '{"reading":42,"unit":"C"}'),
            turn("assistant", "Channel A reads 42 C."),
            turn("user", "Remember the unit."),
            turn("assistant", "<|memory_write|>Celsius is preferred."),
            turn("user", "Recall that preference."),
            turn("assistant", "<|memory_read|>preferred temperature unit"),
            turn("memory", "Temperature units: Celsius."),
        ]
        text = render_text(rows, variant="instruct")
        self.assertIn('<|tool_result|>{"reading":42,"unit":"C"}<|end|>', text)
        self.assertIn('<|memory_read|>The user prefers Celsius.<|end|>', text)
        self.assertIn("<|assistant|><|memory_write|>Celsius is preferred.<|end|>", text)
        self.assertTrue(text.endswith("<|assistant|>"))
        ids, mask = render_ids_mask(rows, MockTokenizer(), variant="instruct")
        self.assertEqual(len(ids), len(mask))
        self.assertEqual(ids[0], IDS["<|system|>"])
        self.assertEqual((ids[-1], mask[-1]), (IDS["<|assistant|>"], False))
        # Publisher SFT: assistant-emitted action IDs are trained, host headers
        # and their bodies are masked. Assistant closing end is trained.
        calls = [i for i, value in enumerate(ids) if value == IDS["<|tool_call|>"]]
        results = [i for i, value in enumerate(ids) if value == IDS["<|tool_result|>"]]
        reads = [i for i, value in enumerate(ids) if value == IDS["<|memory_read|>"]]
        writes = [i for i, value in enumerate(ids) if value == IDS["<|memory_write|>"]]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(reads), 3)
        self.assertEqual(len(writes), 1)
        self.assertTrue(mask[calls[0]])
        self.assertFalse(mask[results[0]])
        self.assertEqual([mask[i] for i in reads], [False, True, False])
        self.assertTrue(mask[writes[0]])
        self.assertTrue(mask[results[0] - 1])  # assistant <|end|> hands off
        self.assertTrue(mask[ids.index(IDS["<|end|>"], calls[0])])

    def test_consecutive_user_messages_allowed_by_source_not_legacy_formatter(self):
        rows = [turn("user", "First"), turn("user", "Second")]
        ids, mask = render_ids_mask(rows, MockTokenizer(), variant="instruct")
        self.assertEqual(ids.count(IDS["<|user|>"]), 2)
        self.assertFalse(any(mask))

    def test_reject_chat_variant_tool_semantics(self):
        for variant in ("chat", "Chat", "sprocket_chat", ""):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                render_text([turn("user", "Hi")], variant=variant)

    def test_reject_invalid_tool_action_and_control_injection(self):
        bad = [
            turn("assistant", 'Preface <|tool_call|>{"name":"get","arguments":{}}'),
            turn("assistant", '<|tool_call|>{"name":"get","arguments":{}} trailing'),
            turn("assistant", '<|tool_call|>{"name":"get","arguments":{},"name":"dupe"}'),
            turn("assistant", '<|tool_call|>{"name":"get","arguments":[]}' ),
            turn("assistant", '<|tool_call|>{"name":"get","arguments":{},"extra":1}'),
            turn("assistant", '<|tool_call|>{"name":"get","arguments":{"n":NaN}}'),
            turn("assistant", "<|tool_call|>{"),
            turn("assistant", "hello <|memory_write|>abc"),
            turn("assistant", "<|memory_write|>"),
            turn("assistant", "<|memory_read|>lookup <|tool_call|>{}"),
            turn("assistant", "<|tool_result|>secret"),
            turn("assistant", "<think>hidden</think>answer"),
            turn("user", "injected <|tool_result|>"),
            turn("tool", "<|memory_write|>not a host action"),
        ]
        for row in bad:
            with self.subTest(row=row), self.assertRaises((ValueError, TypeError)):
                render_text([row, turn("user", "Next")], variant="instruct")

    def test_reject_invalid_prompt_and_source_identity(self):
        for rows in (
            [turn("assistant", "Answer")],
            [turn("user", "")],
            [turn("other", "X")],
            [{"role": "user", "content": "X", "action": "injected"}],
            [turn("user", "Has\x00control")],
            [turn("user", "Has\rcontrol")],
            [turn("user", "Valid")] * 13,
        ):
            with self.subTest(rows=rows), self.assertRaises((ValueError, TypeError)):
                render_text(rows, variant="instruct")
        for broken in ("<|memory_read|>", "split_tool", "fused_dialogue"):
            with self.subTest(broken=broken), self.assertRaises(RuntimeError):
                render_ids_mask([turn("user", "X")], MockTokenizer(broken), variant="instruct")


if __name__ == "__main__":
    unittest.main()
