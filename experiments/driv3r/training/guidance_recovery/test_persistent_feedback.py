"""C9 bounded-feedback checks; no game source or native runner required."""
import copy
import unittest
import numpy as np
import torch
from experiment import ARMS, sha
from adversarial_cues import source_hash
from topology_transfer import ALL_FAMILIES, TRAIN_FAMILIES, OOD_FAMILIES, build
from persistent_feedback import Context, PersistentGate, observe, state_after, data, optimize, rollout, check_noninterference


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(25,1)
    def forward(self,x):return self.layer(x).squeeze(-1)


class PersistentFeedbackTests(unittest.TestCase):
    def test_context_requires_both_host_events_and_expires(self):
        w=build(91,-1,'hairpin',0,False,'reverse')
        self.assertFalse(Context(w).active(w))
        w.issue_recovery_cue(12)
        c=Context(w,steps=3)
        self.assertTrue(c.active(w))
        self.assertFalse(Context(w,authorized=False).active(w))
        for _ in range(2):
            a=w.candidates(0)[8];self.assertTrue(w.apply(a));c.advance(w)
            self.assertTrue(c.active(w))
        self.assertTrue(w.apply(w.candidates(0)[8]));c.advance(w)
        self.assertFalse(c.active(w))
        with self.assertRaises(ValueError):Context(w,steps=13)
        with self.assertRaises(ValueError):Context(w,steps=0)

    def test_goal_change_revokes_context(self):
        w=build(97,1,'hook',0,False,'reverse');w.issue_recovery_cue(12)
        ctx=Context(w);self.assertTrue(ctx.active(w))
        w.goal=(1,2);ctx.advance(w)
        self.assertFalse(ctx.active(w))

    def test_sham_cue_without_goal_change_has_no_context(self):
        w=build(105,1,'staircase',2,False,'ahead');w.goal_change=False
        w.issue_recovery_cue(12)
        ctx=Context(w)
        self.assertFalse(ctx.active(w))
        cs,x=observe(w,1.,ctx.active(w))
        self.assertTrue(np.all(x[:,25]==0))
        self.assertTrue(np.array_equal(x[:,:25],np.asarray([z.features for z in cs],np.float32)))

    def test_host_replayed_states_are_legal_and_have_original_material(self):
        for family in ALL_FAMILIES:
          for outcome in ('reverse','ahead'):
           for k in range(4):
            a=build(201,-1,family,2,False,outcome)
            b,ctx=state_after(201,-1,family,2,False,outcome,k)
            self.assertEqual(sha(sorted(a.protected)),sha(sorted(b.protected)))
            self.assertTrue(a.inked<=b.inked)
            self.assertEqual(b.invalid_attempts,0)
            self.assertTrue(ctx.active(b))
            self.assertEqual(b.goal_change,k==0)

    def test_data_is_source_disjoint_and_contains_delayed_feedback(self):
        X,Y,sources=data(16)
        self.assertEqual(X.shape,(16*9*3,10,26))
        self.assertEqual(Y.shape,(16*9*3,))
        self.assertTrue(np.all(X[:,:,25]==1))
        self.assertEqual(int((X[:,8,18]==0).sum()),16*7*3)
        self.assertEqual(set(z['family'] for z in sources),set(TRAIN_FAMILIES))
        self.assertTrue(set(z['family'] for z in sources).isdisjoint(OOD_FAMILIES))
        self.assertEqual(int(Y.sum()),16*5*3)
        self.assertTrue(all(z['world'] and z['obs'] and z['menu'] for z in sources))

    def test_only_backtrack_residual_and_exact_fallback(self):
        torch.manual_seed(15)
        old=Dummy().eval();new=PersistentGate(copy.deepcopy(old))
        with torch.no_grad():new.delta[-1].bias.fill_(2.)
        result=check_noninterference(old,new,n=1)
        self.assertEqual(result['inactive_identical_cases'],1*4*2*3*3)
        self.assertEqual(result['active_other_scores_identical'],1*4*2*3*9)
        w=build(302,1,'zigzag',0,False,'ahead');w.issue_recovery_cue(12)
        _,x=observe(w,1.,True)
        with torch.no_grad():a=old(torch.from_numpy(x[:,:25]));b=new(torch.from_numpy(x))
        self.assertEqual(float(b[8]-a[8]),2.)
        self.assertTrue(torch.equal(a[:8],b[:8]))

    def test_optimizer_is_deterministic_actual_updates(self):
        X,Y,_=data(9)
        torch.manual_seed(48)
        p=Dummy();a=PersistentGate(copy.deepcopy(p));b=copy.deepcopy(a)
        a,sa,la=optimize(a,X,Y,epochs=2,seed=3)
        b,sb,lb=optimize(b,X,Y,epochs=2,seed=3)
        self.assertEqual((sa,la),(sb,lb))
        self.assertGreater(sa,0)
        for k,v in a.delta.state_dict().items():self.assertTrue(torch.equal(v,b.delta.state_dict()[k]))

    def test_stochastic_sham_and_fallback_equal_exact(self):
        torch.manual_seed(5)
        old=Dummy().eval(); new=PersistentGate(copy.deepcopy(old)).eval()
        rows=rollout({'C8':old,'C9':new},n=1,horizon=5)
        self.assertEqual(len(rows),4*1*2*3*2*2*2)
        for r in rows:self.assertEqual(r['host_rejected'],0)
        paired={}
        for r in rows:
            key=(r['family'],r['seed'],r['personality'],r['arm'],r['outcome'],r['cue'])
            paired.setdefault(key,{})[r['model']]=r
        self.assertEqual(len(paired)*2,len(rows))
        for group in paired.values():
            self.assertEqual(group['C8']['source_sha256'],group['C9']['source_sha256'])
            self.assertEqual([z['action'] for z in group['C8']['trace']],
                             [z['action'] for z in group['C9']['trace']])

    def test_forged_action_never_edits_material(self):
        w=build(201,1,'hairpin',0,False,'reverse');w.issue_recovery_cue(12)
        old=sha(sorted(w.protected));ink=set(w.inked)
        v=copy.deepcopy(w.candidates(0)[8]);v.action=('INK',100,0);v.legal=True
        self.assertFalse(w.apply(v))
        self.assertEqual(sha(sorted(w.protected)),old)
        self.assertEqual(w.inked,ink)

if __name__=='__main__':unittest.main()
