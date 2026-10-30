"""BYTE-EOS-001 offline proof, no transformers, models or training."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from byte_eos_001_void_labels import (  # noqa: E402
    encode_supervised_byte_completion, compare_historical_literal_eos,
    check_tokenizer,
)


class ByteTokenizer:
    """Deliberate simple oracle, not a substitute for separate HF runtime check."""
    vocab_size = 259
    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    eos_token = "<eos>"

    def __init__(self, *, corrupt_special=False):
        self.corrupt_special = corrupt_special

    def encode(self, text, add_special_tokens=True):
        ids = list(text.encode("utf-8", errors="replace"))
        if self.corrupt_special and add_special_tokens:
            return [257, *ids, 258]
        return ids


class LabelConstructionTests(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()

    def test_source_special_token_contract(self):
        check_tokenizer(self.tok)
        self.assertEqual(self.tok.encode("<eos>"), [60, 101, 111, 115, 62])
        self.assertNotIn(258, self.tok.encode("<eos>"))

    def test_exact_assistant_supervision_and_terminal_eos(self):
        prompt = "Task: Explain a local defect.\nAnswer: "
        completion = '{"trigger":"friction","action":"inspect"}'
        x = encode_supervised_byte_completion(self.tok, prompt, completion)
        p = list(prompt.encode("utf-8"))
        c = list(completion.encode("utf-8"))
        self.assertEqual(x["input_ids"], p+c+[258])
        self.assertEqual(x["labels"], [-100]*len(p)+c+[258])
        self.assertEqual(x["attention_mask"], [1]*len(x["input_ids"]))
        self.assertEqual(x["supervised_tokenizer_units"], len(c)+1)
        self.assertTrue(x["terminal_eos_supervised"])
        self.assertFalse(x["historical_train002b_corrected"])

    def test_exact_unicode_multibyte_units_distinct_characters(self):
        prompt = "πΩ→🔧 "
        completion = "café 日本語"
        x = encode_supervised_byte_completion(self.tok,prompt,completion)
        self.assertEqual(x["sequence_tokenizer_units"],
                         len((prompt+completion).encode("utf-8"))+1)
        self.assertNotEqual(x["source_characters"], x["utf8_source_bytes"])
        self.assertEqual(x["input_ids"][-1],258)
        self.assertEqual(x["labels"][-1],258)
        self.assertEqual(x["supervised_tokenizer_units"],
                         len(completion.encode("utf-8"))+1)

    def test_literal_eos_text_remains_content_bytes_and_final_eos_is_id(self):
        x = encode_supervised_byte_completion(self.tok, "Hello ", "<eos>")
        self.assertEqual(x["input_ids"][-6:], [60,101,111,115,62,258])
        self.assertEqual(x["labels"][-6:], [60,101,111,115,62,258])

    def test_empty_completion_still_supervises_real_eos(self):
        x=encode_supervised_byte_completion(self.tok, "Prompt", "")
        self.assertEqual(x["labels"], [-100]*6+[258])
        self.assertEqual(x["supervised_tokenizer_units"],1)

    def test_max_unit_limit_never_truncates_real_eos(self):
        x=encode_supervised_byte_completion(self.tok,"abc","def",max_units=7)
        self.assertEqual(len(x["input_ids"]),7)
        with self.assertRaisesRegex(RuntimeError, "no truncation allowed"):
            encode_supervised_byte_completion(self.tok,"abc","def",max_units=6)
        with self.assertRaises(ValueError):
            encode_supervised_byte_completion(self.tok,"abc","def",max_units=1025)
        with self.assertRaises(ValueError):
            encode_supervised_byte_completion(self.tok,"abc","def",max_units=0)
        with self.assertRaises(ValueError):
            encode_supervised_byte_completion(self.tok,"abc","def",max_units=True)

    def test_invalid_python_unicode_is_rejected_not_silently_replaced(self):
        with self.assertRaises(UnicodeEncodeError):
            encode_supervised_byte_completion(self.tok,"ok","\ud800")

    def test_wrong_codec_or_accidental_special_insertion_rejected(self):
        class OtherTokenizer(ByteTokenizer): pass
        with self.assertRaisesRegex(ValueError, "Only verified"):
            encode_supervised_byte_completion(OtherTokenizer(),"a","b")
        with self.assertRaisesRegex(ValueError, "automatic BOS"):
            encode_supervised_byte_completion(ByteTokenizer(corrupt_special=True),"a","b")
        self.tok.eos_token_id = 7
        with self.assertRaisesRegex(ValueError,"Unexpected"):
            encode_supervised_byte_completion(self.tok,"a","b")

    def test_historical_targets_are_distinct_and_data_unchanged(self):
        p = "Task: "
        c = '{"action":"inspect"}'
        d = compare_historical_literal_eos(self.tok,p,c)
        self.assertEqual(d["legacy_suffix_ids"],[60,101,111,115,62])
        self.assertEqual(d["corrected_suffix_ids"],[258])
        self.assertFalse(d["legacy_had_actual_eos"])
        self.assertTrue(d["corrected_has_actual_eos"])
        self.assertEqual(d["units_saved"],4)
        self.assertFalse(d["TRAIN002b_retroactive_modification"])


if __name__ == "__main__":
    unittest.main()
