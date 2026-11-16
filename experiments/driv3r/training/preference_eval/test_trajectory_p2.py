"""Adversarial P2 source, chronological split and no-wall tests."""
import tempfile,unittest
from pathlib import Path
import trajectory_p2 as p

class ReceiptRankTests(unittest.TestCase):
    def setUp(self):self.groups=p.load()
    def test_full_original_source_cohorts(self):
        self.assertEqual(len(self.groups),8)
        self.assertEqual(sum(len(r) for r in self.groups.values()),32)
    def test_true_original_sha_and_no_human_votes(self):
        for world,rows in self.groups.items():
            self.assertEqual({r['role'] for r in rows},p.ROLES)
            self.assertEqual({len(r['trace_sha256']) for r in rows},{64})
    def test_malformed_input_fails_closed(self):
        data=(p.HERE/'turn_features_p2.csv').read_bytes()
        with tempfile.TemporaryDirectory() as d:
            x=Path(d)/'edited.csv';x.write_bytes(data.replace(b'lfm_w13,6111,learned',b'lfm_w13,6112,learned'))
            with self.assertRaises(ValueError):p.load(x)
    def test_no_future_input_leakage(self):
        self.assertFalse(any('future' in k for k in p.PREFIX))
        self.assertFalse(any('role' in k or 'cohort' in k for k in p.PREFIX))
    def test_no_wall_falsification_is_separate(self):
        for axis in p.OUTCOMES:
            allp,_=p.pairs(self.groups,axis,False)
            clean,_=p.pairs(self.groups,axis,True)
            self.assertGreater(len(allp),len(clean))
            self.assertTrue(all('wall' not in (a['role'],b['role']) for _,a,b,_ in clean))
    def test_optimizer_is_repeatable_and_real(self):
        pp,_=p.pairs(self.groups,'future_novelty',True)
        a=p.fit(pp,epochs=90);b=p.fit(pp,epochs=90)
        self.assertEqual(a['weights'],b['weights'])
        self.assertNotEqual(a['weights'],[0.]*len(p.PREFIX))
        self.assertLess(a['train_loss'][-1][1],a['train_loss'][0][1])
    def test_source_train_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            r=p.run(d)
            self.assertEqual(set(r['axes']),{'future_novelty-all','future_novelty-no_wall','future_stall_escape-all','future_stall_escape-no_wall'})
            self.assertEqual(r['owner_human_votes'],0)
            self.assertEqual(r['native_receipts'],0)
            self.assertEqual(len(list(Path(d).glob('*trained_head.json'))),4)

if __name__=='__main__':unittest.main()
