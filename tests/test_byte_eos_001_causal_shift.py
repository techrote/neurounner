"""BYTE-EOS-001 causal-label shift regression; offline, no model, no optimizer.

A token can only be supervised in decoder-only causal CE when a preceding
input position exists. Test the *shifted* labels, not merely labels[-1].
This suite is intentionally separate from the archived #94 source receipt.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from byte_eos_001_void_labels import encode_supervised_byte_completion


class ByteTokenizer:
    """Synthetic byte codec, not a substitute for pinned ByteTokenizer runtime."""
    vocab_size = 259
    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=True):
        return list(text.encode("utf-8", errors="strict"))


class CausalLabelShiftTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()

    def test_refuse_empty_prompt_even_with_nonempty_completion(self):
        for completion in ("", "A", "π", "<eos>"):
            with self.subTest(completion=completion):
                with self.assertRaisesRegex(ValueError, "at least one prompt byte"):
                    encode_supervised_byte_completion(self.tokenizer, "", completion)

    def test_legacy_context_free_target_would_be_lost_by_causal_shift(self):
        # Original #94 code previously accepted input_ids=[EOS], labels=[EOS].
        # HF causal CE shifts the labels, leaving zero targets.
        legacy_labels = [258]
        self.assertEqual(sum(x != -100 for x in legacy_labels[1:]), 0)
        with self.assertRaises(ValueError):
            encode_supervised_byte_completion(self.tokenizer, "", "")

    def test_nonempty_prompt_full_completion_and_eos_survive_shift(self):
        prompt = "Q:"
        completion = "π🔧"
        a = encode_supervised_byte_completion(self.tokenizer, prompt, completion)
        prompt_ids = list(prompt.encode("utf-8"))
        target = list(completion.encode("utf-8")) + [258]
        self.assertEqual(a["labels"][len(prompt_ids):], target)
        self.assertEqual(a["input_ids"][-len(target):], target)
        self.assertEqual(a["labels"][1:], [-100] * (len(prompt_ids)-1) + target)
        self.assertEqual(a["causal_shift_supervised_units"], len(target))
        self.assertEqual(a["supervised_tokenizer_units"], len(target))
        self.assertTrue(a["terminal_eos_supervised"])
        self.assertFalse(a["historical_train002b_corrected"])

    def test_one_byte_prefix_and_empty_completion_supervise_actual_eos(self):
        a = encode_supervised_byte_completion(self.tokenizer, "x", "")
        self.assertEqual(a["input_ids"], [120, 258])
        self.assertEqual(a["labels"], [-100, 258])
        self.assertEqual(a["labels"][1:], [258])
        self.assertEqual(a["causal_shift_supervised_units"], 1)

    def test_one_byte_prefix_and_literal_eos_text_do_not_merge_tokens(self):
        a = encode_supervised_byte_completion(self.tokenizer, "x", "<eos>")
        self.assertEqual(a["labels"][1:], [60,101,111,115,62,258])
        self.assertEqual(a["causal_shift_supervised_units"], 6)

    def test_boundary_refuses_truncated_eos_not_just_truncated_text(self):
        a = encode_supervised_byte_completion(self.tokenizer, "x", "y", max_units=3)
        self.assertEqual(a["labels"][1:], [121, 258])
        with self.assertRaisesRegex(RuntimeError, "no truncation"):
            encode_supervised_byte_completion(self.tokenizer, "x", "y", max_units=2)

    def test_training_gate_and_source_variants_untouched(self):
        import json
        p = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertIs(p["train_authorized"], False)
        self.assertEqual(set(p["models"]),
                         {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        self.assertEqual(p["models"]["void1"]["tokenizer_reference"]["ids"]["eos"],258)


if __name__ == "__main__":
    unittest.main()
