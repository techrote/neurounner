#!/usr/bin/env python3
"""SPROCKET F4 fail-closed original artifact/re-sampling regression."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('f4',HERE/'worm_f3_posthoc_v1.py')
f4=importlib.util.module_from_spec(spec)
spec.loader.exec_module(f4)

class F4Test(unittest.TestCase):
    def test_sampling_boundaries(self):
        scores=dict.fromkeys(f4.ACTIONS,0.)
        for legal,draw in [([],0.1),(['STOP','STOP'],.1),(['TELEPORT'],.2),(['STRAIGHT'],1.0)]:
            with self.assertRaises(ValueError):f4.sample(scores,legal,draw)
        scores['LEFT']=float('nan')
        with self.assertRaises(ValueError):f4.sample(scores,['LEFT'],.1)
    def test_exact_original_if_present(self):
        path=os.environ.get('F3_ORIGINAL_ZIP')
        if not path:self.skipTest('Pass F3_ORIGINAL_ZIP to exercise original artifact')
        x=f4.inspect(path)
        self.assertEqual(x['crc_members'],65)
        self.assertEqual(x['verified_original_member_sha256'],62)
        self.assertEqual(x['original_model_forwards'],512)
        self.assertEqual(x['totals']['learned_differs_from_flat'],149)
        self.assertEqual(x['totals']['learned_differs_from_frozen_first'],33)
        self.assertEqual(x['totals']['STOP_deferred'],504)
        self.assertEqual(x['totals']['blocked_STRAIGHT_observed'],6)
    def test_tampered_artifacts_rejected_if_present(self):
        path=os.environ.get('F3_ORIGINAL_ZIP')
        if not path:self.skipTest('Original artifact is required for tamper test')
        with self.assertRaisesRegex(ValueError,'original outer SHA mismatch'):
            f4.inspect(path,expected_sha='0'*64)
        with tempfile.TemporaryDirectory() as temp:
            dst=Path(temp)/'tampered.zip'
            with zipfile.ZipFile(path) as original,zipfile.ZipFile(dst,'w') as modified:
                for info in original.infolist():
                    content=original.read(info.filename)
                    if info.filename.endswith('identity.json') and '/results/' in info.filename:
                        content=content.replace(b'"seed"',b'"fake"',1)
                    modified.writestr(info,content)
            with self.assertRaisesRegex(ValueError,'original member SHA mismatch'):
                f4.inspect(dst,expected_sha=hashlib.sha256(dst.read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
