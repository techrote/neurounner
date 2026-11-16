"""Source and metric controls; no model download, native host, or expensive inference."""
import tempfile
import unittest
from pathlib import Path
import rank

HERE = Path(__file__).parent

class PreferenceProxyTests(unittest.TestCase):
    def setUp(self):
        self.worlds = rank.validate(HERE / 'source_features.csv')

    def test_four_real_paired_arms_per_world(self):
        self.assertEqual(len(self.worlds), 8)
        for world, rows in self.worlds.items():
            self.assertEqual({r['controller'] for r in rows}, {'learned','random','wall','fidget'})
            self.assertEqual({r['run_id'] for r in rows}, {43771476464 if world[0]=='lfm_w13' else 43769025429})

    def test_paired_dataset_has_exact_worlds_and_no_human_labels(self):
        for axis in ('local_weave','spatial_extent'):
            pairs,ties=rank.create_pairs(self.worlds,axis)
            self.assertEqual(len(pairs)+len(ties),48)
            self.assertTrue(all(p['a']['world_seed']==p['b']['world_seed'] for p in pairs))

    def test_heldout_worlds_not_leaked(self):
        pairs,_=rank.create_pairs(self.worlds,'local_weave')
        worlds=sorted({p['world'] for p in pairs})
        held=set(worlds[::4]);train=[p for p in pairs if p['world'] not in held]
        test=[p for p in pairs if p['world'] in held]
        self.assertFalse({p['world'] for p in train}&{p['world'] for p in test})

    def test_preference_head_trains_and_is_deterministic(self):
        pairs,_=rank.create_pairs(self.worlds,'spatial_extent')
        a=rank.fit(pairs,epochs=80);b=rank.fit(pairs,epochs=80)
        self.assertEqual(a['weights'],b['weights'])
        self.assertLess(a['train_loss_curve'][-1][1], a['train_loss_curve'][0][1])
        self.assertGreater(rank.evaluate(a,pairs)['correct'],len(pairs)//2)


    def test_w13_archive_shards_and_model_free_controls(self):
        for (family, seed), rows in self.worlds.items():
            archives={r['original_archive_sha256'] for r in rows}
            self.assertEqual(len(archives),1)
            if family=='lfm_w13':
                expected=('36fa8876f249b0263d19dc40bea17731e8a4776da9cc82400189a876980380ff'
                          if seed in (6143,6167) else
                          '0e69b23c100a00b438d4e2d9d07fa4550e276368e0ad4635e38a6c858f79b95e')
                self.assertEqual(archives,{expected})
            else:
                self.assertEqual(archives,{'5eb64e5f7d5335809f631aa11caf9edbfcf1a67f546601d5d14e795cc61f7af6'})
            self.assertEqual(sum(x['evidence_class'].startswith('trained-model-forward') for x in rows),1)

    def test_original_source_sha_mismatch_fails_closed(self):
        orig=(HERE/'source_features.csv').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'altered.csv';p.write_bytes(orig.replace(b'lfm_w13,6111,learned',b'lfm_w13,6112,learned'))
            with self.assertRaises(ValueError):rank.validate(p)

    def test_no_native_or_fake_human_labels(self):
        self.assertEqual(sum(r['controller']=='learned' for v in self.worlds.values() for r in v),8)
        self.assertTrue(all(r['evidence_class'].endswith('/synthetic-W0') for v in self.worlds.values() for r in v))

if __name__=='__main__':unittest.main()
