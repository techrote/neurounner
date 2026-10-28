"""Offline adversarial checks of real-host causal-shift receipt contract."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from byte_eos_001_void_labels import encode_supervised_byte_completion
from byte_eos_001_frozen_qwen_loss import validate_shifted_record
class ByteTokenizer:
    vocab_size=259
    pad_token_id=256
    bos_token_id=257
    eos_token_id=258
    eos_token="<eos>"
    def encode(self,text,add_special_tokens=True):
        return list(text.encode("utf-8",errors="strict"))
class FrozenQwenLossContract(unittest.TestCase):
    def setUp(self):
        self.tok=ByteTokenizer()
    def test_single_prefix_empty_completion_targets_actual_eos(self):
        x=encode_supervised_byte_completion(self.tok,"x","")
        self.assertEqual(validate_shifted_record(x,"x","")["causal_supervised_units"],1)
        self.assertEqual(x["labels"],[-100,258])
    def test_unicode_and_literal_stop_are_plain_bytes(self):
        for text in ("π🔧","<eos>"):
            x=encode_supervised_byte_completion(self.tok,"Q:",text)
            self.assertEqual(validate_shifted_record(x,"Q:",text)["causal_supervised_units"],
                             len(text.encode("utf-8"))+1)
    def test_reject_missing_integer_eos(self):
        x=encode_supervised_byte_completion(self.tok,"Q:","ok")
        x["labels"][-1]=62
        with self.assertRaises(RuntimeError):validate_shifted_record(x,"Q:","ok")
    def test_reject_corrupted_mask_and_prompt_leak(self):
        x=encode_supervised_byte_completion(self.tok,"Q:","ok")
        for field,index,value in (("attention_mask",0,0),("labels",0,81)):
            y=copy.deepcopy(x)
            y[field][index]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                validate_shifted_record(y,"Q:","ok")
    def test_reject_falsified_post_shift_count(self):
        x=encode_supervised_byte_completion(self.tok,"Q:","")
        x["causal_shift_supervised_units"]=2
        with self.assertRaises(RuntimeError):validate_shifted_record(x,"Q:","")
    def test_protocol_no_training_four_identities(self):
        p=json.loads((ROOT/"experiments/ncr-001/protocol.json").read_text())
        self.assertIs(p["train_authorized"],False)
        self.assertEqual(set(p["models"]),{"smollm2","sprocket_chat","sprocket_instruct","void1"})
if __name__=="__main__":
    unittest.main()
