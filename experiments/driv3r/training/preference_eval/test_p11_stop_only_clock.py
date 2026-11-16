"""P11 adversarial source/clock/host-legality coupling checks.

Set P11_W16_SOURCE to the original immutable W16 Actions producer ZIP to
run model tests in a CPU-only environment; no synthetic neural impersonation.
"""
import os
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch=None
if torch is not None:
    import p11_stop_only_clock as p

SOURCE=os.environ.get('P11_W16_SOURCE')
@unittest.skipUnless(SOURCE and torch is not None,'requires immutable W16 original producer ZIP and CPU PyTorch')
class P11RealHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=Path(SOURCE).read_bytes()
        cls.z,cls.manifest,cls.w0,cls.head,cls.sha=p.src.open_original(cls.data)

    def test_original_model_replay_png_exact(self):
        self.assertEqual(len(p.src.replay_original(self.z,self.manifest,self.w0,self.head)),9)

    def test_checkpoint_rejects_mutated_source(self):
        with self.assertRaises(ValueError):p.src.open_original(self.data[:-1]+b'X')

    def test_critical_feature_indices_only(self):
        obs=self.w0.run(2609,'random','greebles',1)['events'][0]['observation']
        obs['tick']=89
        baseline=p.src.model_features(obs,'intact')
        self.assertEqual(len(baseline),142)
        for mode in p.MODES[1:]:
            full,used,n=p.get_logits(self.head,obs,mode)
            if mode!='full_zero_clock':
                self.assertEqual(used[:5],full[:5])
            self.assertEqual(n,2)
        original,_,_=p.get_logits(self.head,obs,'intact')
        self.assertEqual(original,p.src.scores(self.head,obs,'intact'))

    def test_unrecognised_mode_rejected(self):
        obs=self.w0.run(2609,'random','greebles',1)['events'][0]['observation']
        with self.assertRaises(ValueError):p.get_logits(self.head,obs,'not-a-model')

    def test_source_pinned_host_legal_actions(self):
        rec=p.rollout(self.w0,self.head,2609,'greebles','open','stop_no_periodics')
        self.assertEqual(rec['metrics']['invalid_raw_choices'],0)
        self.assertEqual(len(rec['events']),len(rec['head_decisions']))
        self.assertTrue(all(e['applied'] in e['observation']['legal'] for e in rec['events']))

    def test_stop_only_does_not_rewrite_nonstop_path(self):
        base=p.rollout(self.w0,self.head,2609,'greebles','late','intact')
        for mode in p.MODES[1:-1]:
            test=p.rollout(self.w0,self.head,2609,'greebles','late',mode)
            for a,b in zip(base['events'],test['events']):
                if 'STOP' in (a['applied'],b['applied']):break
                self.assertEqual(a['applied'],b['applied'])
                self.assertEqual(a['position_after'],b['position_after'])

    def test_stop_gate_crn_path_prefix(self):
        for mode in p.MODES:
            a=p.rollout(self.w0,self.head,2609,'greebles','late',mode)
            b=p.rollout(self.w0,self.head,2609,'greebles','open',mode)
            self.assertGreaterEqual(p.paired(a,b,mode)['lost_novel'],0)

    def test_adversarial_mutated_event_detected(self):
        a=p.rollout(self.w0,self.head,2609,'greebles','late','intact')
        b=p.rollout(self.w0,self.head,2609,'greebles','open','intact')
        self.assertTrue(p.paired(a,b,'intact')['pre_gated_stop'])
        b['events'][5]['applied']='RETRACE'
        with self.assertRaises(ValueError):p.paired(a,b,'intact')

    def test_independent_rng_and_temporal_spikes(self):
        obs=self.w0.run(2609,'random','greebles',1)['events'][0]['observation']
        self.assertNotEqual(p.src.deterministic_uniform(2609,'greebles',89,'w11-CRN'),
            p.src.deterministic_uniform(2609,'greebles',89,'p10-independent-STOP-CRN-v1'))
        for tick in (88,89,90,91):
            obs['tick']=tick
            full,eff,_=p.get_logits(self.head,obs,'stop_no_periodics')
            self.assertEqual(full[:5],eff[:5])

    def test_intact_positive_p10_control(self):
        a=p.rollout(self.w0,self.head,2609,'greebles','late','intact')
        b=p.src.rollout(self.w0,self.head,2609,'greebles','late','intact')
        self.assertEqual(a['trajectory'],b['trajectory'])
        self.assertEqual(a['metrics'],b['metrics'])
        self.assertEqual(self.w0.png_bytes(a),self.w0.png_bytes(b))

if __name__=='__main__':unittest.main(verbosity=2)
