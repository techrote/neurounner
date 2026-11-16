"""P3 host-material, STOP, provenance, positive/negative controls."""
import copy, json, tempfile, unittest
from pathlib import Path
import recovery_p3 as p

class P3Tests(unittest.TestCase):
    def test_trap_has_real_legal_retrace_and_stop(self):
        w=p.make_world(9011,'zigzag');menu=p.allowed(w)
        self.assertIn('LIFT_W',menu);self.assertIn('STOP',menu)
        self.assertFalse(any(a.startswith('INK_') for a in menu))
        p.advance(w,'LIFT_W')
        self.assertFalse(w['ever_exited'])
        self.assertIn('INK_N',p.allowed(w))
        p.advance(w,'LIFT_W')
        self.assertTrue(w['ever_exited'])
    def test_stop_is_terminal(self):
        w=p.make_world(9000,'open');p.advance(w,'STOP')
        with self.assertRaises(ValueError):p.advance(w,'STOP')
        self.assertEqual(p.allowed(w),())
    def test_protected_material_and_obstacles_fail_closed(self):
        w=p.make_world(9003,'zigzag');w['pos']=(7,w['start'][1]-1)
        with self.assertRaises(ValueError):p.advance(w,'INK_N')
        with self.assertRaises(ValueError):p.advance(w,'LIFT_N')
        self.assertNotIn('INK_N',p.allowed(w))
        self.assertNotIn('LIFT_N',p.allowed(w))
    def test_receipt_strict_replay_catches_tamper(self):
        w=p.run_episode(9003,'zigzag','greedy',horizon=12)
        record=p.audited_record(w,'greedy')
        self.assertTrue(p.verify_receipt(record))
        for key in ('legal_menu','menu_sha256','action','new_material_cells'):
            bad=copy.deepcopy(record)
            bad['events'][0][key]='STOP' if key=='action' else 'tampered'
            with self.assertRaises((ValueError,KeyError)):p.verify_receipt(bad)
    def test_distinct_train_test_worlds_and_unseen_kind(self):
        train=[i for i in range(1000,1080) if i%4!=3]
        test=list(range(9000,9024))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertFalse(any(i%4==3 for i in train))
        self.assertTrue(any(i%4==3 for i in test))
    def test_real_adam_optimisation_and_ablation(self):
        pairs=p.make_training(range(1000,1012))
        self.assertEqual({y for _,y in pairs},{0,1})
        self.assertEqual(sum(y for _,y in pairs)*2,len(pairs))
        a=p.fit(pairs,steps=40);b=p.fit(pairs,steps=40)
        self.assertEqual(a,b)
        self.assertLess(a['loss'][-1][1],a['loss'][0][1])
        self.assertNotEqual(a['weights'],[0.]*len(p.FEATURES))
        no=p.fit(pairs,steps=40,disabled=(p.FEATURES.index('dist_exit'),p.FEATURES.index('xwest')))
        self.assertEqual(no['weights'][p.FEATURES.index('xwest')],0)
    def test_original_synthetic_material_gallery_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            report=p.main(d)
            self.assertEqual(report['human_votes'],0)
            self.assertEqual(report['native_receipts'],0)
            self.assertEqual(report['trained_cpu_steps'],480)
            self.assertEqual(len(report['rows']),24*6)
            self.assertEqual(len(list(Path(d).glob('p3-*.png'))),24*6)
            self.assertEqual(len(list(Path(d).glob('p3-*.json'))),24*6+3)
            for f in Path(d).glob('p3-*.json'):
                obj=json.loads(f.read_text())
                if 'events' in obj:self.assertTrue(p.verify_receipt(obj))

if __name__=='__main__':unittest.main()
