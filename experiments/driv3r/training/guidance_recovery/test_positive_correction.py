"""Cheap offline adversarial checks for the bounded cue-only DRIV3R adapter."""
import copy
import hashlib
import unittest

import numpy as np
import torch
from experiment import ARMS, Candidate, Scorer, sha
from hard_recovery import false_cue_world
from positive_correction import (
    CueGatedPolicy, candidates_fingerprint, evaluate_pairs, greedy_choice,
    make_correction_data, summarize_pairs, train, warning_world,
    rollout_warnings, summarize_rollouts,
)


class PositiveCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(18)
        cls.base=Scorer().eval()

    def test_trap_is_legal_one_step_before_dead_end_for_all_rotations(self):
        for d in (2,3,4):
            for r in range(4):
                for mirrored in (False,True):
                    w,j,tip,_=warning_world(117,1,d,r,mirrored)
                    self.assertEqual(len(w.history),d-1)
                    self.assertTrue(w.allowed(tip))
                    self.assertNotIn(tip,w.inked)
                    self.assertGreater(w.distance(),w.distance(tip))
                    self.assertNotEqual(w.pos,j)
                    self.assertEqual(w.invalid_attempts,0)

    def test_cue_only_changes_observation_not_material_permissions(self):
        w,_,_,_=warning_world(81,-1,3,1,True)
        for arm in ARMS:
            before=copy.deepcopy(w);after=copy.deepcopy(w)
            after.issue_recovery_cue(8)
            self.assertEqual(candidates_fingerprint(before,arm),candidates_fingerprint(after,arm))
            a=before.candidates(arm);b=after.candidates(arm)
            for x,y in zip(a,b):
                self.assertEqual(x.legal,y.legal)
                self.assertEqual(x.action,y.action)
                self.assertEqual(x.target,y.target)
                self.assertEqual([z for j,z in enumerate(x.features) if j not in (16,23)],
                                 [z for j,z in enumerate(y.features) if j not in (16,23)])
            self.assertEqual(before.inked,after.inked)

    def test_forbidden_material_remains_forbidden_even_with_cue(self):
        w,_,_,_=warning_world(91,1,3,0,False)
        w.issue_recovery_cue(9)
        prohibited=next(iter(w.protected))
        forged=Candidate(('INK',1,0),prohibited,True,[1.]*25)
        self.assertFalse(w.apply(forged))
        self.assertEqual(w.inked,set())
        self.assertEqual(w.invalid_attempts,1)

    def test_recovery_is_cursor_only_and_does_not_erase_painted_material(self):
        w,_,tip,_=warning_world(95,1,3,0,False)
        ink=next(c for c in w.candidates(0) if c.action[0]=='INK' and c.target==tip)
        self.assertTrue(w.apply(ink))
        material_sha=sha(sorted(w.inked))
        back=next(c for c in w.candidates(0) if c.action[0]=='BACKTRACK')
        self.assertTrue(w.apply(back))
        self.assertEqual(sha(sorted(w.inked)),material_sha)
        self.assertIn(tip,w.inked)

    def test_train_and_eval_source_seeds_topologies_are_disjoint(self):
        x,y,labels,ids=make_correction_data(seed=51457,instances=12)
        self.assertEqual(x.shape[1:],(10,25))
        self.assertEqual(y.shape[1],10)
        self.assertEqual(len(labels),len(x))
        self.assertTrue(all(d in (2,3) and r in (0,1) for _,d,r,_,_,_ in ids))
        self.assertTrue(all(seed!=867100+i*811 for seed,_,_,_,_,_ in ids for i in range(24)))
        e=evaluate_pairs(self.base,seeds=4)
        self.assertTrue(all(r['rotation'] in (2,3) for r in e))
        self.assertEqual(len(e),4*3*2*2)

    def test_paired_data_reproducible_and_contains_opposite_labels(self):
        a,b,l,i=make_correction_data(instances=5)
        x,y,m,j=make_correction_data(instances=5)
        self.assertEqual(a.tobytes(),x.tobytes())
        self.assertEqual(b.tobytes(),y.tobytes())
        self.assertEqual(l,m);self.assertEqual(i,j)
        self.assertEqual(sum(z[0]=='unwarned-explore' for z in l),5)
        self.assertEqual(sum(z[0]=='warned-switch' for z in l),5)
        self.assertEqual(sum(z[0]=='reject-false-cue' for z in l),5)

    def test_neutral_adapter_starts_identical_to_base(self):
        torch.manual_seed(33)
        adapter=CueGatedPolicy(copy.deepcopy(self.base)).eval()
        w,_,_,_=warning_world(100,1,3,0,False)
        for flag in (False,True):
            if flag:w.issue_recovery_cue(9)
            x=torch.tensor([v.features for v in w.candidates(.35)],dtype=torch.float32)
            with torch.no_grad():self.assertTrue(torch.equal(self.base(x),adapter(x)))

    def test_no_cue_is_byte_identical_after_optimizer_updates(self):
        x,y,labels,_=make_correction_data(instances=10)
        base=copy.deepcopy(self.base)
        snapshot={k:v.clone() for k,v in base.state_dict().items()}
        trained,steps,losses,n=train(base,x,y,labels,epochs=2)
        self.assertGreater(steps,0)
        for name,p in base.state_dict().items():self.assertTrue(torch.equal(snapshot[name],p))
        w,_,_,_=warning_world(91,1,3,0,False)
        obs=torch.tensor([v.features for v in w.candidates(1.0)],dtype=torch.float32)
        with torch.no_grad():self.assertTrue(torch.equal(base(obs),trained(obs)))
        self.assertTrue(any(torch.any(v!=0) for v in trained.adapter[-1].state_dict().values()))

    def test_tiny_optimizer_is_reproducible_with_exact_model_weights(self):
        x,y,labels,_=make_correction_data(instances=8)
        a,_,_,_=train(copy.deepcopy(self.base),x,y,labels,epochs=2)
        b,_,_,_=train(copy.deepcopy(self.base),x,y,labels,epochs=2)
        self.assertTrue(all(torch.equal(a.adapter.state_dict()[k],b.adapter.state_dict()[k])
                            for k in a.adapter.state_dict()))

    def test_evaluator_keeps_source_and_legal_menu_hashes_matched(self):
        obs=evaluate_pairs(self.base,seeds=4)
        summary=summarize_pairs(obs)
        self.assertEqual(set(summary),{f'{a:.2f}/{f}' for a in ARMS for f in ('true_warning','false_alarm')})
        self.assertTrue(all(x['pairs']==4 for x in summary.values()))

    def test_rollout_has_no_native_undo_and_persists_bounded_traces(self):
        trace=rollout_warnings(self.base,seeds=2)
        self.assertEqual(len(trace),2*3*2)
        summary=summarize_rollouts(trace)
        self.assertTrue(all(s['host_rejected_actions']==0 for s in summary.values()))
        for t in trace:
            self.assertLessEqual(len(t['events']),17)
            self.assertTrue(all(event['ink_count']>=0 for event in t['events']))
            self.assertTrue(all(len(event['material_sha256'])==64 for event in t['events']))

    def test_wrongly_issued_cue_does_not_make_illegal_moves_legal(self):
        w=false_cue_world(55,-1);w.issue_recovery_cue(5)
        for arm in ARMS:
            choice,bad=greedy_choice(self.base,w,arm)
            self.assertTrue(choice.legal)
            self.assertFalse(bad or (choice.action[0]=='BACKTRACK' and not w.history))


if __name__=='__main__':unittest.main()
