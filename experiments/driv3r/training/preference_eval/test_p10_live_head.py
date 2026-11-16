"""P10 adversarial source, model-forward, legal-menu and STOP-prefix tests.
Requires P10_W16_SOURCE=<immutable producer ZIP> and installed CPU PyTorch;
source-only workflows skip explicitly, dedicated model job runs all tests.
"""
import os
import unittest
from pathlib import Path

SRC=os.getenv('P10_W16_SOURCE')
try:
    import torch
except ImportError:
    torch=None

@unittest.skipUnless(SRC and torch, 'P10 source original and CPU Torch required for model tests')
class P10ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import p10_live_head as p
        cls.p=p
        cls.data=Path(SRC).read_bytes()
        cls.z,cls.manifest,cls.w0,cls.head,cls.ck=p.open_original(cls.data)

    def test_pinned_archive_refuses_tampering(self):
        with self.assertRaises(ValueError):self.p.open_original(self.data[:-1]+b'0')

    def test_original_nine_real_head_replay_and_png_bytes(self):
        self.assertEqual(len(self.p.replay_original(self.z,self.manifest,self.w0,self.head)),9)

    def test_clock_ablation_changes_only_15_inputs(self):
        p=self.p
        obs=self.w0.run(2609,'random','greebles',1)['events'][0]['observation']
        obs['tick']=88
        a=p.model_features(obs,'intact');z=p.model_features(obs,'zeroed');f=p.model_features(obs,'frozen_t0')
        self.assertEqual(len(a),142)
        for other in (z,f):
            self.assertEqual(a[:107],other[:107]);self.assertEqual(a[122:],other[122:])
        self.assertEqual(z[107:121],[0.]*14)
        self.assertEqual(f[107:121],[0.]*7+[1.]*7)
        self.assertEqual(z[121],f[121])

    def test_stop_rng_is_independent_and_deterministic(self):
        u=self.p.deterministic_uniform
        self.assertNotEqual(u(2609,'greebles',89,'w11-CRN'),
                            u(2609,'greebles',89,'p10-independent-STOP-CRN-v1'))
        self.assertEqual(u(2609,'greebles',89,'w11-CRN'),u(2609,'greebles',89,'w11-CRN'))

    def test_actual_clock_early_stop_does_not_remap_movements(self):
        p=self.p
        a=p.rollout(self.w0,self.head,2609,'greebles','late','intact')
        b=p.rollout(self.w0,self.head,2609,'greebles','open','intact')
        pair=p.pair(a,b)
        self.assertEqual(pair['first_stop_tick'],89)
        self.assertGreater(pair['lost_novel'],0)
        self.assertTrue(all(e['raw_legal'] for e in b['events']))

    def test_frozen_t0_inference_no_early_stop(self):
        p=self.p
        a=p.rollout(self.w0,self.head,2609,'greebles','late','frozen_t0')
        b=p.rollout(self.w0,self.head,2609,'greebles','open','frozen_t0')
        self.assertFalse(p.pair(a,b)['early_stop'])
        self.assertEqual(len(b['head_decisions']),128)

    def test_detect_pre_stop_divergence(self):
        p=self.p
        a=p.rollout(self.w0,self.head,2609,'greebles','late','intact')
        b=p.rollout(self.w0,self.head,2609,'greebles','open','intact')
        b['events'][3]['applied']='STOP'
        with self.assertRaises(ValueError):p.pair(a,b)

if __name__=='__main__':unittest.main(verbosity=2)
