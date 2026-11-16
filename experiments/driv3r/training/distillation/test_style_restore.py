"""Source-only adversarial guards for style-rehearsal distillation."""
import unittest
import numpy as np
from experiments.driv3r.training.distillation.style_restore import make_targets,legal_softmax,select_score

class FakeModel:
    def scores(self,x):
        return np.zeros((len(x),6),dtype=np.float32)

class TestStyleRestore(unittest.TestCase):
    def data(self):
        x=np.zeros((2,100),dtype=np.float32)
        x[:,75]=1.;x[0,78]=1.;x[1,79]=1.
        y=np.zeros((2,6),dtype=np.float32)
        y[0,3]=1.;y[1,4]=1.
        return x,y
    def test_legal_mass(self):
        x,y=self.data()
        target,_=make_targets(x,y,FakeModel(),.45)
        self.assertTrue(np.allclose(target.sum(1),1))
        self.assertTrue(np.all(target[x[:,75:81]==0]==0))
    def test_negative_invalid_target(self):
        x,y=self.data();y[0,2]=.1;y[0,3]=.9
        with self.assertRaises(ValueError):make_targets(x,y,FakeModel(),.25)
    def test_missing_legal_action(self):
        x,y=self.data();x[0,75:81]=0
        with self.assertRaises(ValueError):make_targets(x,y,FakeModel(),.25)
    def test_no_nan(self):
        x,y=self.data();x[0,1]=np.nan
        with self.assertRaises(ValueError):make_targets(x,y,FakeModel(),.25)
    def test_bad_shape(self):
        x,y=self.data()
        with self.assertRaises(ValueError):make_targets(x[:,:99],y,FakeModel(),.2)
    def test_selection_penalizes_coverage_loss(self):
        base={k:{'cells':12.,'branches':12.} for k in ('normal','trap','squeeze')}
        repair={k:{'cells':30.,'branches':6.} for k in base}
        poor={k:{'cells':9.,'branches':20.} for k in base}
        good={k:{'cells':29.,'branches':10.} for k in base}
        self.assertGreater(select_score(good,base,repair)['score'],select_score(poor,base,repair)['score'])
if __name__=='__main__':unittest.main()
