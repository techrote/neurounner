"""C10 external guidance trainer regression tests; no native/game execution."""
import copy, unittest
import numpy as np
import torch
from experiment import ARMS, sha
from topology_transfer import build, OOD_FAMILIES, TRAIN_FAMILIES
from persistent_feedback import state_after
from revocable_feedback import (HostFeedback, RevocableGate, observe, make_data,
    train, gate_checks, evaluate, summarize)

class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(26,1)
    def forward(self,x):return self.linear(x).squeeze(-1)

class RevocableFeedbackTests(unittest.TestCase):
    def test_host_revoke_clears_advice_not_ink_or_protected(self):
        w=build(42,1,'hairpin',0,False,'reverse')
        w.issue_recovery_cue(12)
        ctx=HostFeedback(w)
        old=(w.pos,list(w.history),set(w.inked),sha(sorted(w.protected)))
        self.assertTrue(ctx.active(w))
        ctx.revoke(w)
        self.assertFalse(ctx.active(w))
        self.assertEqual(w.recovery_cue_ttl,0)
        self.assertEqual(old,(w.pos,list(w.history),set(w.inked),sha(sorted(w.protected))))
        self.assertEqual(w.invalid_attempts,0)

    def test_no_cue_never_creates_context(self):
        w=build(45,-1,'hook',1,False,'ahead')
        ctx=HostFeedback(w)
        self.assertFalse(ctx.active(w))
        _,x=observe(w,.35,ctx)
        self.assertEqual(x.shape,(10,28))
        self.assertTrue(np.all(x[:,25:27]==0))
        self.assertTrue(np.array_equal(x[:,:25],np.asarray([c.features for c in w.candidates(.35)],np.float32)))

    def test_context_expires_and_revocation_cannot_be_overridden(self):
        w=build(48,-1,'hairpin',0,False,'reverse');w.issue_recovery_cue(12)
        ctx=HostFeedback(w,steps=2)
        self.assertTrue(ctx.active(w))
        for i in range(2):
            self.assertTrue(w.apply(w.candidates(0)[8]));ctx.advance(w)
        self.assertFalse(ctx.active(w))
        ctx.revoke(w)
        self.assertFalse(ctx.active(w))
        with self.assertRaises(ValueError):HostFeedback(w,strength=1.2)
        with self.assertRaises(ValueError):HostFeedback(w,strength=-0.1)

    def test_invalid_movement_remains_host_rejected_under_active_guidance(self):
        w=build(49,1,'zigzag',0,False,'reverse');w.issue_recovery_cue(12)
        fb=HostFeedback(w);_,x=observe(w,1.,fb)
        self.assertEqual(x[8,25],1)
        old=sha(sorted(w.protected)); ink=set(w.inked)
        bad=copy.deepcopy(w.candidates(1.)[8]);bad.action=('INK',4,0);bad.legal=True
        self.assertFalse(w.apply(bad))
        self.assertEqual(sha(sorted(w.protected)),old)
        self.assertEqual(w.inked,ink)
        self.assertEqual(w.invalid_attempts,1)

    def test_feedback_strength_not_label_leak(self):
        X,Y,r=make_data(8)
        self.assertEqual(X.shape,(8*2*4*2*3,10,28))
        self.assertEqual(Y.shape,(len(X),))
        self.assertTrue(np.all(X[:,8,25]==1))
        self.assertEqual(set(round(float(x),2) for x in np.unique(X[:,8,26])),{.55,1.})
        self.assertEqual(set(z['outcome'] for z in r),{'ahead','reverse'})
        self.assertEqual(set(z['family'] for z in r),set(TRAIN_FAMILIES))
        self.assertTrue(set(TRAIN_FAMILIES).isdisjoint(OOD_FAMILIES))
        for outcome in ('ahead','reverse'):
            subset={z['confidence'] for z in r if z['outcome']==outcome}
            self.assertEqual(subset,{.55,1.})

    def test_training_is_actual_optimizer_and_reproducible(self):
        torch.manual_seed(5); base=Dummy().eval()
        X,Y,_=make_data(8)
        a=RevocableGate(copy.deepcopy(base),width=8,scale=4.)
        b=copy.deepcopy(a)
        weights_before={k:v.clone() for k,v in a.delta.state_dict().items()}
        a,step1,l1=train(a,X,Y,epochs=2,seed=73,reverse_weight=2.)
        b,step2,l2=train(b,X,Y,epochs=2,seed=73,reverse_weight=2.)
        self.assertEqual((step1,l1),(step2,l2))
        self.assertGreater(step1,0)
        self.assertTrue(any(not torch.equal(v,a.delta.state_dict()[k]) for k,v in weights_before.items()))
        for k,v in a.delta.state_dict().items():self.assertTrue(torch.equal(v,b.delta.state_dict()[k]))

    def test_exact_cue_off_and_countermand_fallback(self):
        torch.manual_seed(5); old=Dummy().eval(); model=RevocableGate(copy.deepcopy(old))
        with torch.no_grad():model.delta[-1].bias.fill_(1.)
        c=gate_checks(old,model,n=2)
        self.assertEqual(c['inactive_identical_cases'],2*4*2*3*2)
        self.assertEqual(c['active_nonbacktrack_exact'],2*4*2*3*9)
        self.assertEqual(c['host_countermands_exercised'],2*4*2*3)

    def test_host_countermand_clears_after_live_cue_and_nonbacktrack_unchanged(self):
        torch.manual_seed(5); old=Dummy().eval(); new=RevocableGate(copy.deepcopy(old))
        with torch.no_grad():new.delta[-1].bias.fill_(1.)
        w=build(51,1,'staircase',1,False,'reverse');w.issue_recovery_cue(12)
        fb=HostFeedback(w)
        _,x=observe(w,1.,fb)
        with torch.no_grad():a=new(torch.from_numpy(x));b=old(torch.from_numpy(x[:,:26]))
        self.assertTrue(torch.equal(a[:8],b[:8]))
        self.assertAlmostEqual(float(a[8]-b[8]),float(new.scale*np.tanh(1.)),places=5)
        fb.revoke(w);_,x=observe(w,1.,fb)
        with torch.no_grad():a=new(torch.from_numpy(x));b=old(torch.from_numpy(x[:,:26]))
        self.assertTrue(torch.equal(a,b))

    def test_rollout_matches_pairs_legal_and_reproducible(self):
        torch.manual_seed(7); p=Dummy().eval(); q=RevocableGate(copy.deepcopy(p)).eval()
        x=evaluate({'C9':p,'C10':q},n=1,horizon=5,seed=123)
        y=evaluate({'C9':p,'C10':q},n=1,horizon=5,seed=123)
        self.assertEqual(x,y)
        self.assertEqual(len(x),4*1*2*3*2*3*2)
        self.assertEqual(sum(r['host_rejected'] for r in x),0)
        s=summarize(x)
        self.assertIn('C10/withheld/reverse/live',s)
        pairs={}
        for r in x:
            key=(r['family'],r['seed'],r['personality'],r['arm'],r['outcome'],r['cue_mode'])
            pairs.setdefault(key,{})[r['model']]=r
        self.assertEqual(len(pairs)*2,len(x))
        for v in pairs.values():
            self.assertEqual(v['C9']['source_sha256'],v['C10']['source_sha256'])
            self.assertEqual(v['C9']['material_sha256'],v['C10']['material_sha256'])

if __name__=='__main__':unittest.main()
