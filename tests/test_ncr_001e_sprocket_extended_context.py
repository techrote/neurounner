"""Offline fail-closed NCR-001E 2048-context and raw-control-ID source guards.

Pure negative source tests. They do not execute or train any LLM or open a fixture.
"""
from __future__ import annotations
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001e_sprocket_extended_context import (
    LENGTHS, CONTROL_IDS, assert_fixed_context, assert_control_inventory,
)


def inventory():
    return {
        "controls": {
            marker: {"dedicated": True, "token_ids": [number],
                     "vocab_lookup": number}
            for marker, number in CONTROL_IDS.items()
        },
        "tool_and_memory_semantics_qualified": False,
    }


class ExtendedContextGuards(unittest.TestCase):
    def test_locked_context_512_1024_2048(self):
        self.assertEqual(LENGTHS, (512, 1024, 2048))
        assert_fixed_context(LENGTHS, 2048)
        for bad, limit in (
            ((512, 1024), 2048),
            ((512, 1024, 1536), 2048),
            ((512, 1024, 2048), 1024),
            ((512, 1024, 4096), 2048),
        ):
            with self.subTest(lengths=bad, limit=limit):
                with self.assertRaises(RuntimeError):
                    assert_fixed_context(bad, limit)

    def test_actual_special_token_identity_only(self):
        assert_control_inventory(inventory())
        for marker in CONTROL_IDS:
            for change in (
                {"dedicated": False},
                {"token_ids": [60, 101, 111, 115, 62]},
                {"vocab_lookup": 32001},
            ):
                with self.subTest(marker=marker, change=change):
                    corrupted = inventory()
                    corrupted["controls"][marker].update(change)
                    with self.assertRaises(RuntimeError):
                        assert_control_inventory(corrupted)

    def test_token_recognition_cannot_claim_tool_semantic_pass(self):
        corrupted = inventory()
        corrupted["tool_and_memory_semantics_qualified"] = True
        with self.assertRaises(RuntimeError):
            assert_control_inventory(corrupted)


if __name__ == "__main__":
    unittest.main()
