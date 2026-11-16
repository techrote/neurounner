#!/usr/bin/env python3
"""Real frozen W0 host refusal for an unmodified model digit+EOS7 choice."""
import unittest
import worm
import worm_train_w7_v1 as w7

class Tokenizer:
    eos_token_id = 7
    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return {(529, 7): "4", (529,): "4", (998, 7): "4\n",
                (999, 7): "x"}[tuple(ids)]

class RawChoiceCustody(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()

    def test_real_w0_host_sees_illegal_retrace_and_stops(self):
        d = w7.classify_first_pass(self.tok, [529, 7], ["STOP"])
        self.assertEqual(d["raw_text"], "4")
        self.assertEqual(d["token_ids"], [529, 7])
        self.assertEqual(d["decoded_action"], "RETRACE")
        self.assertEqual(d["raw_action"], "RETRACE")
        self.assertTrue(d["raw_syntactic_eos"])
        self.assertTrue(d["eos_terminated"])
        self.assertFalse(d["raw_legal"])
        self.assertFalse(d["valid_first_pass"])
        r = worm.run(3101, mode="callback", family="greebles", steps=1,
                     selector=lambda observation: d["raw_action"])
        event = r["events"][0]
        self.assertNotIn("RETRACE", event["observation"]["legal"])
        self.assertIn("STOP", event["observation"]["legal"])
        self.assertEqual(event["raw_choice"], "RETRACE")
        self.assertEqual(event["applied"], "STOP")
        self.assertFalse(event["raw_legal"])
        self.assertEqual(r["metrics"]["invalid_raw_choices"], 1)
        self.assertEqual(r["metrics"]["unique_trail_cells"], 1)

    def test_legal_retrace_remains_legal(self):
        d = w7.classify_first_pass(self.tok, [529, 7], ["RETRACE", "STOP"])
        self.assertEqual(d["decoded_action"], "RETRACE")
        self.assertEqual(d["raw_action"], "RETRACE")
        self.assertTrue(d["raw_legal"])
        self.assertTrue(d["valid_first_pass"])

    def test_incomplete_digit_is_refused_without_erasing_raw_decoding(self):
        d = w7.classify_first_pass(self.tok, [529], ["RETRACE"])
        self.assertEqual(d["decoded_action"], "RETRACE")
        self.assertEqual(d["raw_action"], "MODEL_REFUSED")
        self.assertFalse(d["eos_terminated"])
        self.assertFalse(d["raw_syntactic_eos"])
        self.assertTrue(d["raw_legal"])
        self.assertFalse(d["valid_first_pass"])

    def test_malformed_digit_is_refused_despite_eos7(self):
        d = w7.classify_first_pass(self.tok, [999, 7], ["STOP"])
        self.assertEqual(d["raw_text"], "x")
        self.assertIsNone(d["decoded_action"])
        self.assertEqual(d["raw_action"], "MODEL_REFUSED")
        self.assertTrue(d["eos_terminated"])
        self.assertFalse(d["raw_syntactic_eos"])

    def test_no_whitespace_repair(self):
        d = w7.classify_first_pass(self.tok, [998, 7], ["RETRACE"])
        self.assertEqual(d["raw_text"], "4\n")
        self.assertIsNone(d["decoded_action"])
        self.assertEqual(d["raw_action"], "MODEL_REFUSED")

if __name__ == "__main__":
    unittest.main()
