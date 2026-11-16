"""W10 zero-optimizer original-byte auditor adversarial source tests."""
import unittest
from w10_original_receipt_audit import raw,check
class W10SupportTests(unittest.TestCase):
    def sample(self):
        return dict(origin="W0-observed",expected="LEFT",legal=["LEFT","RIGHT","STOP"],
            raw_text="1",decoded_action="LEFT",token_ids=[526,7],
            raw_syntactic_eos=True,raw_legal=True,valid_first_pass=True,
            eos_terminated=True,raw_action="LEFT")
    def test_raw_true_digit_eos(self):
        self.assertEqual(raw(self.sample()),(True,True,True))
    def test_host_illegal_does_not_erase_true_eos(self):
        s=self.sample();s.update(expected="RIGHT",legal=["RIGHT","STOP"],raw_legal=False,valid_first_pass=False)
        self.assertEqual(raw(s),(True,False,False))
    def test_reject_fake_eos(self):
        s=self.sample();s["token_ids"]=[526,5]
        with self.assertRaises(ValueError):raw(s)
    def test_reject_token_label_swap(self):
        s=self.sample();s["token_ids"]=[527,7]
        with self.assertRaises(ValueError):raw(s)
    def test_reject_raw_repair(self):
        s=self.sample();s["raw_action"]="RIGHT"
        with self.assertRaises(ValueError):raw(s)
    def test_reject_syntax_legal_conflation(self):
        s=self.sample();s["raw_legal"]=False
        with self.assertRaises(ValueError):raw(s)
    def test_reject_invalid_source(self):
        s=self.sample();s["origin"]="unverified-native"
        with self.assertRaises(ValueError):raw(s)
    def test_bad_signature(self):
        with self.assertRaises(ValueError):check(False,"tampered archive")
if __name__=="__main__":unittest.main()
