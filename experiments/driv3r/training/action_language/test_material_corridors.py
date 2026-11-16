"""GAL-010 adversarial material-crop/host-authority tests."""
import tempfile
import unittest
from pathlib import Path
from material_corridors import (ARMS,KINDS,START,TRAIN_BASE,TEST_BASE,make_corridor,
 CorridorWorld,visible_material,vector,Model,train,episode,run,audit)
from lab import HostRefusal,INK,sha

class MaterialCorridorTests(unittest.TestCase):
    def test_induced_corridors_and_branching(self):
        for kind in KINDS:
            for seed in range(TRAIN_BASE,TRAIN_BASE+16):
                cells,_=make_corridor(seed,kind)
                self.assertIn(START,cells);self.assertEqual(33,len(cells))
                for at in cells:
                    n=sum(p in cells for p in ((at[0]+1,at[1]),(at[0]-1,at[1]),(at[0],at[1]+1),(at[0],at[1]-1)))
                    self.assertLessEqual(n,2 if kind=='path' else 4)
                if kind=='tree':self.assertGreaterEqual(sum(sum(p in cells for p in ((at[0]+1,at[1]),(at[0]-1,at[1]),(at[0],at[1]+1),(at[0],at[1]-1)))>=3 for at in cells),2)
    def test_determinism(self):
        self.assertEqual(make_corridor(TRAIN_BASE,'tree'),make_corridor(TRAIN_BASE,'tree'))
    def test_crop_contains_only_local_actual_material(self):
        w=CorridorWorld(TRAIN_BASE,'tree','jog','test')
        crop=visible_material(w,'material')
        self.assertTrue(all(abs(p[0]-w.cursor[0])+abs(p[1]-w.cursor[1])<=4 for p in crop))
        self.assertEqual(set(crop.values()),{0,3})
        self.assertFalse(visible_material(w,'occluded'))
    def test_distant_material_cannot_leak(self):
        w=CorridorWorld(TRAIN_BASE,'tree','jog','test')
        menu=w.legal();first={n:vector(w,n,p,'material') for n,p in menu.items()}
        distant=(40,40);w.material[w.idx(distant)]=5
        self.assertEqual(first,{n:vector(w,n,p,'material') for n,p in menu.items()})
    def test_no_hidden_goal_leak(self):
        w=CorridorWorld(TRAIN_BASE,'tree','lift','test')
        first={n:vector(w,n,p,'material') for n,p in w.legal().items()}
        w.goal=frozenset()
        self.assertEqual(first,{n:vector(w,n,p,'material') for n,p in w.legal().items()})
    def test_host_rejects_foreign_action_and_stale_lease(self):
        w=CorridorWorld(TRAIN_BASE,'tree','jog','test')
        menu,token=w.offer()
        with self.assertRaises(HostRefusal):w.apply('INK_E_40',token)
        with self.assertRaises(HostRefusal):w.apply('STOP','bad')
        self.assertEqual((menu,token),w.offer())
    def test_stop_never_masked(self):
        for arm in ARMS:
            w=CorridorWorld(TRAIN_BASE,'path',arm,'test')
            self.assertIn('STOP',w.legal())
            self.assertEqual(len(vector(w,'STOP',(),'material')),len(Model().weights))
    def test_optimizer_replays_and_updates(self):
        a,sa=train('lift','material',4);b,sb=train('lift','material',4)
        self.assertEqual(sa,sb);self.assertEqual(a.weights,b.weights)
        self.assertGreater(sa['updates'],0)
        self.assertTrue(any(abs(x)>1e-6 for x in a.weights))
        w,row=episode(a,'lift','material','tree',TEST_BASE)
        self.assertEqual(sha(w.trace),row['receipts_sha256'])
    def test_bounded_run_audit_and_tamper(self):
        with tempfile.TemporaryDirectory() as path:
            run(path,train_worlds=4,eval_worlds=4,gallery=0)
            result=audit(path)
            self.assertEqual(result['episodes_replayed'],6*2*2*4)
            f=Path(path)/'receipts.json';f.write_text(f.read_text()+' ')
            with self.assertRaises(ValueError):audit(path)

if __name__=='__main__':unittest.main()
