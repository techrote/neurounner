import random
import unittest
from .world import ACTIONS,World,snapshots,N

class WorldTests(unittest.TestCase):
    def test_seeds_replay(self):
        a,b=World.make(19),World.make(19)
        self.assertEqual((a.prompt(),a.goal,a.walls),(b.prompt(),b.goal,b.walls))
        self.assertEqual(snapshots([12,14]),snapshots([12,14]))
    def test_menu_enforced(self):
        for seed in range(80):
            w=World.make(seed)
            for t in range(30):
                ds=w.destinations()
                self.assertTrue(set(ds).issubset(set(ACTIONS)))
                self.assertIn('STOP',ds)
                self.assertTrue(all(w.safe(v) for k,v in ds.items() if k!='STOP'))
                a=w.teacher(random.Random(seed+111*t))
                self.assertIn(a,ds)
                w.apply(a)
                if w.stopped:break
            self.assertTrue(w.painted.isdisjoint(w.protected|w.walls))
    def test_reject_illegal(self):
        w=World.make(42)
        with self.assertRaises(ValueError):w.apply('TELEPORT')
    def test_penup_no_paint(self):
        for seed in range(200):
            w=World.make(seed)
            if 'PENUP' in w.legal():
                n=len(w.painted); h=len(w.history)
                w.apply('PENUP')
                self.assertEqual(n,len(w.painted));self.assertEqual(h,len(w.history));return
        self.fail('missing penup scenario')
    def test_holdout_seeds(self):
        tr=snapshots(range(100,110))
        dev=snapshots(range(900,903))
        self.assertFalse({x[2] for x in tr} & {x[2] for x in dev})

if __name__=='__main__':unittest.main()
