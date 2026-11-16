"""Offline C7 regression and negative-control tests; no native/game imports."""
import copy, unittest
import numpy as np
import torch
from experiment import ARMS, Scorer, sha
from positive_correction import CueGatedPolicy, candidates_fingerprint
from adversarial_cues import pair_view, source_hash
from goal_switch_recovery import world, dataset, evaluate, personality_probe
from personality_gate import BacktrackOnlyGate, train, check_exact_noninterference, check_ink_rank_preservation

class PersonalityGateTests(unittest.TestCase):
    def build(self):
        torch.manual_seed(27)
        return BacktrackOnlyGate(CueGatedPolicy(Scorer())).eval()

    def test_no_cue_and_unmodified_goal_are_bit_identical(self):
        m=self.build()
        with torch.no_grad():
            for goal in ('reverse','ahead'):
                w=world(87,1,2,True,goal,True)
                for cue,changed in ((False,True),(True,False),(False,False)):
                    v=copy.deepcopy(w)
                    if cue:v.issue_recovery_cue(9)
                    v.goal_change=changed
                    for a in ARMS:
                        x=torch.tensor([c.features for c in v.candidates(a)],dtype=torch.float32)
                        self.assertTrue(torch.equal(m(x),m.prior(x)))

    def test_cue_may_change_only_backtrack_logit(self):
        m=self.build()
        with torch.no_grad():
            m.gate[-1].bias.fill_(7.)
            for outcome in ('reverse','ahead'):
                w=world(88,-1,3,True,outcome,True);w.issue_recovery_cue(9)
                x=torch.tensor([c.features for c in w.candidates(1.)],dtype=torch.float32)
                a=m.prior(x);b=m(x)
                self.assertTrue(torch.equal(a[:8],b[:8]))
                self.assertTrue(torch.equal(a[9:],b[9:]))
                self.assertNotEqual(float(a[8]),float(b[8]))

    def test_cue_paired_worlds_preserve_materials_and_menu(self):
        for shape in ('upper','lower'):
            w=world(121,-1,2,False,'reverse',True,shape)
            original=sha(sorted(w.protected))
            for a in ARMS:
                x,y=pair_view(w,a)
                self.assertEqual(source_hash(x),source_hash(y))
                self.assertEqual(candidates_fingerprint(x,a),candidates_fingerprint(y,a))
            self.assertEqual(original,sha(sorted(w.protected)))

    def test_deterministic_optimizer_and_frozen_prior(self):
        X,Y,_=dataset(n=8,mix_branches=True)
        torch.manual_seed(77);a=self.build()
        torch.manual_seed(77);b=self.build()
        a,sa,la=train(a,X,Y,epochs=3,seed=23)
        b,sb,lb=train(b,X,Y,epochs=3,seed=23)
        self.assertEqual((sa,la),(sb,lb))
        for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))
        self.assertGreater(sa,0)
        self.assertGreater(check_exact_noninterference(a,2),0)
        self.assertEqual(check_ink_rank_preservation(a,2)['max_score_diff'],0.)

    def test_training_sources_have_disjoint_world_identifiers(self):
        X,Y,rows=dataset(n=8,mix_branches=True)
        self.assertEqual(X.shape,(48,10,25))
        self.assertEqual(len(set(x['world_sha256'] for x in rows)),16)
        self.assertTrue(np.allclose(Y.sum(axis=-1),1))

    def test_protected_materials_remain_host_enforced(self):
        w=world(91,-1,2,False,'reverse',True)
        before=set(w.inked)
        forged=copy.deepcopy(w.candidates(1.)[0]);forged.action=('INK',6,5)
        forged.legal=True;forged.target=(w.pos[0]+6,w.pos[1]+5)
        self.assertFalse(w.apply(forged))
        self.assertEqual(w.inked,before)
        b=next(c for c in w.candidates(1.) if c.action[0]=='BACKTRACK')
        self.assertTrue(w.apply(b));self.assertEqual(w.inked,before)

    def test_heldout_receipts_still_use_host_admission(self):
        rows=evaluate(self.build(),n=2,branch=True,branch_shape='lower')
        self.assertEqual(len(rows),24)
        self.assertTrue(all(r['accepted'] and r['material_sha256'] for r in rows))
        for i in range(0,len(rows),2):
            self.assertEqual(rows[i]['menu_sha256'],rows[i+1]['menu_sha256'])

    def test_personality_probe_same_source_before_action(self):
        d=personality_probe(self.build(),seeds=2)
        self.assertEqual(len(d),18)
        self.assertTrue(all(0<=x['personality_action_divergence']<=2 for x in d.values()))

if __name__=='__main__':unittest.main()
