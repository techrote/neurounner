#!/usr/bin/env python3
"""Positive + fail-closed F3 one-step original-evidence tests; no optimizer/model."""
import json, os, tempfile, unittest
from pathlib import Path
import f3_one_step_opportunity as one

class OneStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=Path(os.environ['SPROCKET_F3_ORIGINAL_ZIP'])
        cls.w0=one.load_w0(Path(os.environ['SPROCKET_F3_PINNED_W0']))
        cls.raw=cls.path.read_bytes()

    def test_original_512_matched_counterfactuals(self):
        r=one.analyze(self.raw,self.w0)
        self.assertEqual(r['population']['n'],512)
        self.assertEqual(r['differing_same_observation']['n'],149)
        self.assertEqual(r['learned_novel_gains_vs_flat'],44)
        self.assertEqual(r['learned_novel_losses_vs_flat'],10)
        self.assertEqual(r['different_but_same_novel_outcome'],95)
        self.assertEqual(r['population']['learned_new'],280)
        self.assertEqual(r['population']['flat_new'],246)
        self.assertTrue(r['not_closed_loop_counterfactual'])
        self.assertEqual(r['new_model_forwards'],0)
        self.assertEqual(set(r['action_counts']),set(one.ACTIONS))

    def test_exact_reproducible_report_and_14_original_host_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            r1=one.analyze(self.raw,self.w0,a)
            r2=one.analyze(self.raw,self.w0,b)
            self.assertEqual(r1,r2)
            self.assertEqual(r1['gallery_pngs'],14)
            self.assertEqual(sorted(x.name for x in a.iterdir()), sorted(x.name for x in b.iterdir()))
            for x in a.iterdir():
                self.assertEqual(x.read_bytes(),(b/x.name).read_bytes(),x.name)
            manifest=json.loads((a/'gallery-manifest.json').read_text())
            for item in manifest:
                self.assertEqual(one.sha((a/item['filename']).read_bytes()),item['sha256'])
                self.assertTrue((a/item['filename']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertEqual(json.loads((a/'report.json').read_text()),r1)
            self.assertEqual(len((a/'per_decision.jsonl').read_text().splitlines()),512)
            with self.assertRaises(FileExistsError):one.analyze(self.raw,self.w0,a)

    def test_forged_original_zip_fails_closed(self):
        altered=self.raw[:-1]+bytes([self.raw[-1]^1])
        with self.assertRaisesRegex(ValueError,'original F3 ZIP SHA mismatch'):
            one.analyze(altered,self.w0)

    def test_original_W0_file_pin_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'worm.py';p.write_text('ACTIONS=("STOP",)')
            with self.assertRaisesRegex(ValueError,'pinned W0 digest mismatch'):
                one.load_w0(p)

    def test_unknown_or_illegal_action_masks_rejected(self):
        with self.assertRaisesRegex(ValueError,'bad host mask/draw'):
            one.sampled(dict.fromkeys(one.ACTIONS,0),['LEFT','LEFT'],0.3)
        with self.assertRaisesRegex(ValueError,'bad host mask/draw'):
            one.sampled(dict.fromkeys(one.ACTIONS,0),['FORGE'],0.3)
        with self.assertRaisesRegex(ValueError,'invalid original logits'):
            one.sampled(dict.fromkeys(one.ACTIONS,float('nan')),['LEFT'],0.3)

    def test_legality_prng_is_one_to_one_and_no_STOP_conflation(self):
        a=dict.fromkeys(one.ACTIONS,0.)
        self.assertEqual(one.sampled(a,['STRAIGHT','LEFT','BRANCH'],0.05),'STRAIGHT')
        self.assertEqual(one.sampled(a,['STRAIGHT','LEFT','BRANCH'],0.5),'LEFT')
        self.assertEqual(one.sampled(a,['STRAIGHT','LEFT','BRANCH'],0.99),'BRANCH')
        self.assertNotEqual(one.sampled(a,['STOP'],0.5),one.sampled(a,['LEFT'],0.5))

if __name__=='__main__':unittest.main()
