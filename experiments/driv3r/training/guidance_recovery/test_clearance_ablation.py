"""Pure external synthetic-host controls; no native painting or shared changes."""
import copy
import unittest
import numpy as np
import torch
from experiment import ARMS, Scorer, sha
from positive_correction import CueGatedPolicy, candidates_fingerprint
from adversarial_cues import near_wall_false_world, decoy_true_world, source_hash
from clearance_ablation import (clearance,observe,EvidenceGatedPolicy,
    choose, construct,data,train,evaluate,summarize,FEATURES,
    long_range_decoy,evaluate_limits,style_rollout,summarize_style)

class ClearanceAblationTests(unittest.TestCase):
    def test_factual_optional_clearance_resolves_former_aliases(self):
        for kind in ('straight','corner'):
            a=near_wall_false_world(929,1,4,2,False,kind)[0]
            b=decoy_true_world(929,1,4,2,False,kind)[0]
            self.assertEqual(candidates_fingerprint(a,1),candidates_fingerprint(b,1))
            self.assertEqual(observe(a,1,False).tobytes(),observe(b,1,False).tobytes())
            xa,xb=observe(a,1,True),observe(b,1,True)
            self.assertFalse(np.array_equal(xa,xb))
            self.assertTrue(np.array_equal(xa[:,:25],xb[:,:25]))
            self.assertGreater(float(xa[:,25].max()),float(xb[:,25].max()))
            self.assertNotEqual(source_hash(a),source_hash(b))

    def test_new_observation_does_not_mutate_material_history_or_legality(self):
        for fam,w in construct(3929,-1,4,3,True).items():
            state=(source_hash(w),sha(sorted(w.protected)),sha(sorted(w.inked)),tuple(w.history))
            for a in ARMS:
                menu=candidates_fingerprint(w,a)
                self.assertEqual(observe(w,a).shape,(10,FEATURES))
                self.assertEqual(observe(w,a,False).shape,(10,FEATURES))
                self.assertEqual(candidates_fingerprint(w,a),menu)
                self.assertTrue(np.all(observe(w,a,False)[:,25:]==0))
                self.assertEqual(state,(source_hash(w),sha(sorted(w.protected)),sha(sorted(w.inked)),tuple(w.history)))

    def test_frozen_base_withheld_and_uncued_fallback_bit_exact(self):
        torch.manual_seed(97)
        old=CueGatedPolicy(Scorer()).eval()
        new=EvidenceGatedPolicy(copy.deepcopy(old)).eval()
        w=near_wall_false_world(123,1,4,2)[0]
        for a in ARMS:
            for cue in (False,True):
                v=copy.deepcopy(w)
                if cue:v.issue_recovery_cue(9)
                x=observe(v,a,True)
                y=observe(v,a,False)
                with torch.no_grad():
                    if not cue:self.assertTrue(torch.equal(new(torch.from_numpy(x)),old(torch.from_numpy(x[:,:25]))))
                    self.assertTrue(torch.equal(new(torch.from_numpy(y)),old(torch.from_numpy(y[:,:25]))))

    def test_paired_training_is_source_unique_and_seed_isolated(self):
        x,y,labels,sources=data(instances=9,seed=60217)
        self.assertEqual(x.shape,(54,10,FEATURES))
        self.assertTrue(all(abs(z.sum()-1)<1e-5 for z in y))
        self.assertEqual(len({(z['source'],z['family']) for z in sources}),54)
        self.assertFalse({z['seed'] for z in sources}&{920001+i*977 for i in range(32)})

    def test_gradient_replay_deterministic_and_neutral_fallback(self):
        torch.manual_seed(17)
        old=CueGatedPolicy(Scorer())
        x,y,*_=data(instances=6)
        a=EvidenceGatedPolicy(copy.deepcopy(old)); b=copy.deepcopy(a)
        a,steps,losses=train(a,x,y,epochs=2,seed=99)
        b,steps_b,loss_b=train(b,x,y,epochs=2,seed=99)
        self.assertEqual((steps,losses),(steps_b,loss_b))
        self.assertTrue(all(torch.equal(a.delta.state_dict()[k],b.delta.state_dict()[k]) for k in a.delta.state_dict()))
        w=construct(982,1,4,2,False)['straight_false'];w.issue_recovery_cue(7)
        with torch.no_grad():
            xx=torch.from_numpy(observe(w,1,False))
            self.assertTrue(torch.equal(a(xx),b(xx)))
            self.assertTrue(torch.equal(a(xx),a.prior(xx[:,:25])))

    def test_heldout_receipts_and_arms(self):
        torch.manual_seed(99)
        model=EvidenceGatedPolicy(CueGatedPolicy(Scorer())).eval()
        r=evaluate(model,seeds=2)
        self.assertEqual(len(r),2*3*6*2)
        self.assertEqual(len(summarize(r)),18)
        self.assertTrue(all(z['host_accepted'] for z in r))
        self.assertTrue(all(z['source'] and z['material_sha256'] and z['menu_sha256'] for z in r))

    def test_bounded_lookahead_cannot_resolve_long_hidden_trap(self):
        for kind in ('straight','corner'):
            for rot in (0,1,2,3):
                good,bad=long_range_decoy(1109,-1,rot,True,kind)
                self.assertNotEqual(source_hash(good),source_hash(bad))
                for arm in ARMS:
                    self.assertTrue(np.array_equal(observe(good,arm,True),observe(bad,arm,True)))

    def test_spoofed_evidence_is_a_disclosed_negative_control(self):
        torch.manual_seed(93)
        m=EvidenceGatedPolicy(CueGatedPolicy(Scorer())).eval()
        long_records,spoof=evaluate_limits(m,seeds=2)
        self.assertEqual(len(long_records),2*3*2*2)
        self.assertEqual(len(spoof),2*3*2*2)
        self.assertTrue(all(r['clean_observation_sha256']!=r['spoof_observation_sha256'] for r in spoof))

    def test_stochastic_transfers_keep_protected_material_and_host_gates(self):
        torch.manual_seed(94)
        m=EvidenceGatedPolicy(CueGatedPolicy(Scorer())).eval()
        records=style_rollout(m,seeds=3,turns=9)
        self.assertEqual(len(records),3*2*3*2)
        self.assertTrue(all(r['host_rejected']==0 and r['material_sha256'] for r in records))
        self.assertEqual(len(summarize_style(records)),6)

    def test_galleries_reconstruct_original_source_geometry(self):
        import json, tempfile
        from pathlib import Path
        from clearance_gallery import build
        torch.manual_seed(48)
        model=EvidenceGatedPolicy(CueGatedPolicy(Scorer())).eval()
        r=style_rollout(model,seeds=4,turns=5)
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.json';b=Path(d)/'b.json';out=Path(d)/'gallery.html'
            a.write_text(json.dumps(r));b.write_text(json.dumps(r))
            self.assertGreater(build(a,b,out,per_family=1),0)
            self.assertIn('source-reconstructed',out.read_text())

    def test_clearance_not_permission_or_native_undo(self):
        w=near_wall_false_world(231,1,4,2)[0];state=sha(sorted(w.inked))
        c=next(c for c in w.candidates(0) if c.action[0]=='INK' and c.legal)
        self.assertTrue(all(0<=v<=1 for v in clearance(w,c)))
        self.assertEqual(sha(sorted(w.inked)),state)
        bad=copy.deepcopy(c);bad.action=('INK',8,0);bad.target=(w.pos[0]+8,w.pos[1]);bad.legal=True
        self.assertFalse(w.apply(bad))
        self.assertEqual(sha(sorted(w.inked)),state)

if __name__=='__main__':unittest.main()
