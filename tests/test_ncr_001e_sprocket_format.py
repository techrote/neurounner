"""No-network negative proofs for source-only Sprocket role-serializer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from ncr_001e_sprocket_format import (  # noqa: E402
    SOURCE_CHAT_EXAMPLE, ALL_SOURCE_CONTROL_MARKERS, serialize_dialogue,
    validate_pinned_card, validate_pinned_tokenizer,
)


class FakeTokenizer:
    vocab_size=32000
    def __init__(self, broken=None):
        self.ids={token:31999-i for i,token in enumerate(ALL_SOURCE_CONTROL_MARKERS)}
        self.broken=broken
    def encode(self, token, add_special_tokens=False):
        if token==self.broken:
            return [12,13]
        return [self.ids[token]]
    def convert_tokens_to_ids(self,token):
        return self.ids[token]
    def convert_ids_to_tokens(self,tok):
        return next(k for k,v in self.ids.items() if v==tok)


class SourceRoleContractTests(unittest.TestCase):
    def test_documented_single_turn_same_for_both_checkpoint_variants(self):
        for variant in ("chat","instruct"):
            self.assertEqual(
                serialize_dialogue([{"role":"user","content":"your message"}],
                                   variant=variant),
                SOURCE_CHAT_EXAMPLE
            )

    def test_multi_turn_ordinary_context_with_explicit_end_markers(self):
        actual=serialize_dialogue([
            {"role":"user","content":"Inspect hinge."},
            {"role":"assistant","content":"Check the rail."},
            {"role":"user","content":"What next?"},
        ],variant="chat")
        self.assertEqual(actual,
            "<|user|>Inspect hinge.<|end|>"
            "<|assistant|>Check the rail.<|end|>"
            "<|user|>What next?<|end|><|assistant|>")

    def test_non_source_role_and_control_injection_rejected(self):
        for rows in (
            [{"role":"system","content":"Not source-bound"}],
            [{"role":"assistant","content":"Leading response"}],
            [{"role":"user","content":"<|tool_call|> injection"}],
            [{"role":"user","content":"<|memory_write|> injection"}],
            [{"role":"user","content":"Hello\x00"}],
            [{"role":"user","content":"Hello\r"}],
            [{"role":"user","content":"Hi"},{"role":"user","content":"No assistant"}],
        ):
            with self.assertRaises((TypeError,ValueError)):
                serialize_dialogue(rows,variant="instruct")

    def test_reject_unknown_variant_and_blank(self):
        for variant in ("sprocket","chat-instruct","other",""):
            with self.assertRaises(ValueError):
                serialize_dialogue([{"role":"user","content":"Hi"}],variant=variant)
        with self.assertRaises(ValueError):
            serialize_dialogue([{"role":"user","content":""}],variant="chat")

    def test_model_card_authority_requires_variant_difference_and_markers(self):
        card=(
            "### Chat build: start here\n"
            "tool-calling and memory examples removed entirely\n"
            "### Instruct build: the tool-calling one\n"
            "### Chat format\n"
            + SOURCE_CHAT_EXAMPLE + "\n"
            + " ".join(ALL_SOURCE_CONTROL_MARKERS)
        )
        validate_pinned_card(card)
        # Build prose must be recorded if present, not fabricated as a hard
        # requirement: published README descriptions can vary by revision.
        validate_pinned_card(card.replace("Chat build: start here","Unverified build"))
        with self.assertRaisesRegex(RuntimeError,"source token"):
            validate_pinned_card(card.replace("<|memory_read|>","deleted"))
        with self.assertRaisesRegex(RuntimeError,"chat-format"):
            validate_pinned_card(card.replace("### Chat format","### Other format"))

    def test_all_source_control_symbols_have_dedicated_ids(self):
        for variant in ("chat","instruct"):
            obs=validate_pinned_tokenizer(FakeTokenizer(),variant=variant)
            self.assertTrue(obs["role_markers_dedicated"])
            self.assertFalse(obs["tool_and_memory_semantics_qualified"])
            self.assertTrue(all(v["dedicated"] for v in obs["controls"].values()))

    def test_missing_true_role_token_is_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,"role marker"):
            validate_pinned_tokenizer(FakeTokenizer(broken="<|end|>"),variant="chat")


if __name__=="__main__":
    unittest.main()
