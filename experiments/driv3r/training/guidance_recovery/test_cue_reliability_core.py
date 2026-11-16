"""C15 pure model-side reproducibility and safety regression without binary fixtures."""
import copy,hashlib,unittest
import numpy as np
import torch
from torch import nn
from experiment import sha
from topology_transfer import build
from adversarial_cues import source_hash
from signed_feedback import HostAdvice,observe
from cue_reliability_core import ReliabilityGate,evidence,replay_data,fit

class FrozenSyntheticPrior(nn.Module):
    def forward(self,x):
        return x[...,7]*3. - x[...,22] + x[...,13]*.15

class CueReliabilityCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1501)
        self.prior=FrozenSyntheticPrior().eval()
        self.model=ReliabilityGate(self.prior)
        with torch.no_grad():self.model.delta[-1].bias.fill_(.6)
    def world(self,kind='reverse',cue='backtrack',arm=.35):
        w=build(150017,1,'hairpin',0,False,kind)
        ad=HostAdvice(w)
        if cue!='none':getattr(ad,'issue_'+cue)(w,strength=.7,ttl=3)
        cs,x=observe(w,arm,ad)
        return w,ad,cs,x
    def test_cue_off_bit_identity(self):
        _,_,_,x=self.world(cue='none')
        t=torch.tensor(x[None,:,:]);self.assertTrue(torch.equal(self.model(t),self.prior(t)))
    def test_revoked_cue_bit_identity(self):
        w,ad,_,_=self.world();ad.cancel(w)
        _,x=observe(w,1.,ad);t=torch.tensor(x[None,:,:])
        self.assertTrue(torch.equal(self.model(t),self.prior(t)))
    def test_only_backtrack_changes_under_live_signed_cue(self):
        for c in ('backtrack','continue'):
            _,ad,cs,x=self.world(cue=c)
            t=torch.tensor(x[None,:,:]);base=self.prior(t);got=self.model(t)
            self.assertTrue(torch.equal(base[:,:8],got[:,:8]))
            self.assertTrue(torch.equal(base[:,9:],got[:,9:]))
            self.assertTrue(cs[8].legal)
            self.assertLessEqual(abs(float((got[0,8]-base[0,8]).detach())),.7*1.5+1e-6)
    def test_illegal_backtrack_cannot_change(self):
        _,_,_,x=self.world();x[8,4]=0.
        t=torch.tensor(x[None,:,:]);self.assertTrue(torch.equal(self.model(t),self.prior(t)))
    def test_evidence_observable_and_bounded(self):
        _,_,_,x=self.world()
        z=evidence(torch.tensor(x[None,:,:]))
        self.assertEqual(z.shape,(1,5))
        self.assertTrue(torch.isfinite(z).all())
        self.assertGreaterEqual(float(z[0,4]),0.)
    def test_receipt_hash_replay(self):
        w,ad,cs,x=self.world()
        r={'seed':w.seed,'personality':w.personality,'family':'hairpin','rotation':0,'mirror':False,
           'outcome':'reverse','directive':'backtrack','strength':.7,'arm':.35,'advantage':.42,
           'world_sha256':source_hash(w),'protected_sha256':sha(sorted(w.protected)),
           'menu_sha256':sha([(c.action,c.target,c.legal) for c in cs]),
           'obs_sha256':hashlib.sha256(x.tobytes()).hexdigest()}
        X,Y,W,T,n=replay_data([r]);self.assertEqual(n,1)
        self.assertTrue(np.array_equal(X[0],x))
        self.assertEqual(float(T[0]),0.)
    def test_frozen_prior_with_actual_adamw_steps(self):
        _,_,_,x=self.world()
        X=np.repeat(x[None,:,:],8,axis=0).astype(np.float32)
        Y=np.full(8,.8,np.float32);W=np.ones(8,np.float32);T=np.zeros(8,np.float32)
        before=copy.deepcopy(self.prior.state_dict())
        out,steps,losses=fit(self.model,self.prior,X,Y,W,T,epochs=3)
        self.assertEqual(steps,3)
        self.assertEqual(len(losses),3)
        self.assertTrue(all(np.isfinite(v) for v in losses))
        self.assertEqual(before,self.prior.state_dict())

if __name__=='__main__':unittest.main()
