"""Offline C6 guards; no native interface or mutable production files."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from experiment import ARMS, Scorer, sha
from positive_correction import CueGatedPolicy, candidates_fingerprint
from adversarial_cues import source_hash, pair_view
from goal_switch_recovery import (world,GoalSwitchAdapter,choose,dataset,train,evaluate,
    summary,personality_probe,rollout_probe,rollout_summary)

class GoalSwitchTests(unittest.TestCase):
    def test_goal_flip_does_not_grant_permission_or_change_protected_material(self):
        for outcome in ('reverse','ahead'):
            for shape in ('upper','lower'):
                w=world(103,1,3,True,outcome,True,shape)
                mat=sha(sorted(w.protected));before=sha(sorted(w.inked))
                for arm in ARMS:
                    a,b=pair_view(w,arm)
                    self.assertEqual(source_hash(a),source_hash(b))
                    self.assertEqual(candidates_fingerprint(a,arm),candidates_fingerprint(b,arm))
                self.assertEqual(mat,sha(sorted(w.protected)))
                self.assertEqual(before,sha(sorted(w.inked)))

    def test_branch_shapes_have_real_legal_alternatives(self):
        for shape,y in (('upper',6),('lower',8)):
            for rot in range(4):
                w=world(102,1,rot,False,'ahead',True,shape)
                self.assertEqual(sum(c.legal and c.action[0]=='INK' and c.target not in w.history
                    for c in w.candidates(0)),2)

    def test_training_is_world_separated_across_outcomes(self):
        X,Y,r=dataset(n=8,mix_branches=True)
        self.assertEqual(X.shape,(48,10,25));self.assertEqual(Y.shape,(48,10))
        self.assertTrue(np.allclose(Y.sum(axis=1),1))
        self.assertEqual(len(set(z['world_sha256'] for z in r)),16)
        # Legal-menu fingerprints may repeat across distinct Worlds; they are not source IDs.
        self.assertGreater(len({z['menu_sha256'] for z in r}),1)

    def test_cue_goal_gate_exact_legacy_logits(self):
        torch.manual_seed(71)
        prior=CueGatedPolicy(Scorer()).eval()
        new=GoalSwitchAdapter(copy.deepcopy(prior)).eval()
        with torch.no_grad():
            for outcome in ('reverse','ahead'):
                for arm in ARMS:
                    w=world(111,-1,1,False,outcome,True)
                    for cue in (False,True):
                        v=copy.deepcopy(w)
                        if cue:v.issue_recovery_cue(9)
                        x=torch.tensor([c.features for c in v.candidates(arm)],dtype=torch.float32)
                        if not cue:
                            self.assertTrue(torch.equal(new(x),prior(x)))
                        v.goal_change=False
                        x=torch.tensor([c.features for c in v.candidates(arm)],dtype=torch.float32)
                        self.assertTrue(torch.equal(new(x),prior(x)))

    def test_optimizer_training_deterministic_repeated(self):
        torch.manual_seed(41)
        prior=CueGatedPolicy(Scorer()).eval()
        X,Y,_=dataset(n=8,mix_branches=True)
        torch.manual_seed(22);a=GoalSwitchAdapter(copy.deepcopy(prior))
        torch.manual_seed(22);b=GoalSwitchAdapter(copy.deepcopy(prior))
        a,sa,la=train(a,X,Y,epochs=3,seed=91)
        b,sb,lb=train(b,X,Y,epochs=3,seed=91)
        self.assertEqual((sa,la),(sb,lb))
        self.assertTrue(all(torch.equal(a.residual.state_dict()[k],b.residual.state_dict()[k]) for k in a.residual.state_dict()))
        # Verify exact policy passthrough AFTER nonzero optimizer updates.
        for outcome in ('reverse','ahead'):
            w=world(119,-1,1,True,outcome,True)
            for cue,change in ((False,True),(True,False)):
                v=copy.deepcopy(w)
                if cue:v.issue_recovery_cue(9)
                v.goal_change=change
                x=torch.tensor([z.features for z in v.candidates(.35)],dtype=torch.float32)
                with torch.no_grad():self.assertTrue(torch.equal(a(x),a.prior(x)))

    def test_heldout_pairs_share_materials_and_legal_menus(self):
        torch.manual_seed(7)
        model=GoalSwitchAdapter(CueGatedPolicy(Scorer())).eval()
        records=evaluate(model,n=3,branch=True,branch_shape='lower')
        self.assertEqual(len(records),3*2*3*2)
        for i in range(0,len(records),2):
            a,b=records[i:i+2]
            self.assertEqual(a['world_sha256'],b['world_sha256'])
            self.assertEqual(a['material_sha256'],b['material_sha256'])
            self.assertEqual(a['menu_sha256'],b['menu_sha256'])
            self.assertTrue(a['accepted'] and b['accepted'])
        self.assertEqual(len(summary(records)),12)

    def test_host_enforces_protection_and_real_undo_is_absent(self):
        w=world(117,1,0,False,'reverse')
        before=set(w.inked);p=copy.deepcopy(w.candidates(0)[0]);p.action=('INK',99,0);p.target=(106,7);p.legal=True
        self.assertFalse(w.apply(p));self.assertEqual(w.inked,before)
        c=next(c for c in w.candidates(0) if c.action[0]=='BACKTRACK')
        self.assertTrue(w.apply(c));self.assertEqual(w.inked,before)

    def test_rollout_receipts_and_source_reconstruction(self):
        torch.manual_seed(8)
        model=GoalSwitchAdapter(CueGatedPolicy(Scorer())).eval()
        records=rollout_probe(model,seeds=2,turns=5)
        self.assertEqual(len(records),3*2*3*2)
        self.assertTrue(all(r['host_rejected']==0 and r['material_sha256'] for r in records))
        self.assertEqual(len(rollout_summary(records)),18)
        from goal_switch_gallery import build
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=p/'a.json';b=p/'b.json';out=p/'g.html'
            import json
            a.write_text(json.dumps(records));b.write_text(json.dumps(records))
            self.assertEqual(build(a,b,out),18)
            self.assertIn('Source-reconstructed',out.read_text())

    def test_personality_comparison_is_same_state_not_diverged_trajectory(self):
        torch.manual_seed(7)
        model=GoalSwitchAdapter(CueGatedPolicy(Scorer())).eval()
        out=personality_probe(model,seeds=2)
        self.assertEqual(len(out),18)
        self.assertTrue(all(0<=v['personality_action_divergence']<=2 for v in out.values()))

if __name__=='__main__':unittest.main()
