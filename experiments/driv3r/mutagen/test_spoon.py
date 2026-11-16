#!/usr/bin/env python3
"""Tiny fake GGUF regression: prove valid framing and anatomical-only spoon edits.

Does NOT count as trained or real d1 inference. Real donor is checked on Actions.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import spoon


def fake_gguf(path):
    def string(s):
        b=s.encode()
        return struct.pack("<Q",len(b))+b
    names=["blk.8.ffn_gate.weight","blk.1.shortconv.in_proj.weight",
           "blk.2.attn_q.weight"]+[f"unused.{i}.weight" for i in range(97)]
    raw=bytearray(b"GGUF")
    raw+=struct.pack("<IQQ",3,len(names),1)
    raw+=string("general.alignment")+struct.pack("<II",4,32)
    for i,name in enumerate(names):
        raw+=string(name)
        raw+=struct.pack("<IQIQ",1,32,8,i*64)
    while len(raw)%32:
        raw.append(0)
    for i in range(100):
        # 34 bytes Q8_0: fp16 scale + 32 signed int8 codes. Extra padding
        # fills the aligned start of the next fake tensor.
        raw+=struct.pack("<e",1.0)+bytes([12+i%7])*32+bytes(30)
    path.write_bytes(raw)
    return names


class SpoonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.dir=Path(self.temp.name)
        self.donor=self.dir/"donor.gguf"
        self.names=fake_gguf(self.donor)
        self.donor_hash=hashlib.sha256(self.donor.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def test_inventory_and_non_destructive_all_three_sculptures(self):
        overview=spoon.inspect(self.donor)
        self.assertEqual(overview["n_tensors"],100)
        self.assertEqual(overview["alignment"],32)
        self.assertEqual(overview["tensors"][0]["name"],self.names[0])
        for kind in spoon.SPECS:
            with self.subTest(kind=kind):
                path=self.dir/(kind+".gguf")
                rec=spoon.carve(self.donor,path,kind,
                               self.dir/(kind+".json"))
                self.assertEqual(rec["modified_tensor_count"],1)
                self.assertNotEqual(rec["candidate_sha256"],self.donor_hash)
                self.assertEqual(path.stat().st_size,self.donor.stat().st_size)
                self.assertEqual(rec["status"],"MUTATED_UNQUALIFIED_MODEL_INFERENCE_PENDING")
                self.assertEqual(spoon.inspect(path)["n_tensors"],100)
        self.assertEqual(hashlib.sha256(self.donor.read_bytes()).hexdigest(),self.donor_hash)

    def test_q8_scale_untouched_and_unselected_tensors_identical(self):
        meta=spoon.inspect(self.donor)
        path=self.dir/"scoop.gguf"
        spoon.carve(self.donor,path,"mlp_scoop",self.dir/"result.json")
        expected=self.donor.read_bytes()
        actual=path.read_bytes()
        first=meta["tensors"][0]["offset"]
        self.assertEqual(actual[first:first+2],expected[first:first+2])
        self.assertNotEqual(actual[first+2:first+34],expected[first+2:first+34])
        second=meta["tensors"][1]["offset"]
        self.assertEqual(actual[second:second+34],expected[second:second+34])

    def test_refuse_overwrite_and_nonsensical_model(self):
        path=self.dir/"dest.gguf"
        spoon.carve(self.donor,path,"mlp_scoop",self.dir/"result.json")
        with self.assertRaises(spoon.GGUFError):
            spoon.carve(self.donor,path,"mlp_scoop",self.dir/"again.json")
        (self.dir/"broken.gguf").write_bytes(b"GGUF")
        with self.assertRaises(spoon.GGUFError):
            spoon.inspect(self.dir/"broken.gguf")


if __name__=="__main__":
    unittest.main()
