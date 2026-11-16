"""GAL-004 model-view intervention and independent replay guards."""
import json
from pathlib import Path
import tempfile
import unittest
from lab import FEATURES, HostRefusal, TinyPolicy
from observation_ablation import (MaskedWorld, MODES, audit, AuditRefusal,
                                    run_world, comparison, write_run)

class ObservationAblationTests(unittest.TestCase):
    def test_masks_do_not_change_original_host_legality(self):
        worlds=[MaskedWorld(160001,'pillars','sweep',mode) for mode in MODES]
        menus=[w.offer()[0] for w in worlds]
        self.assertTrue(all(x==menus[0] for x in menus))
        self.assertIn('STOP',menus[0])
        self.assertTrue(any(a.startswith('INK_') for a in menus[0]))
        for w in worlds:
            self.assertEqual(w.original,worlds[0].original)
            self.assertEqual(w.protected,worlds[0].protected)

    def test_model_feature_mask_does_not_hide_unmasked_features(self):
        baseline=MaskedWorld(160000,'empty','sweep','full')
        menu=baseline.legal();name=next(n for n in menu if n.startswith('INK_'))
        original=baseline.features(name,menu[name])
        for mode,dropped in MODES.items():
            world=MaskedWorld(160000,'empty','sweep',mode)
            actual=world.features(name,world.legal()[name])
            for i,k in enumerate(FEATURES):
                self.assertEqual(actual[i],0 if k in dropped else original[i])

    def test_constant_legacy_features_have_no_candidate_selection_power(self):
        world=MaskedWorld(160000,'empty','sweep','full')
        options=world.legal()
        for feature in ('near_ink','open'):
            values={world.features(action,path)[FEATURES.index(feature)]
                    for action,path in options.items()}
            self.assertEqual(len(values),1,feature)

    def test_protected_material_refused_with_blinded_view(self):
        world=MaskedWorld(160002,'gates','sweep','without_visit_memory')
        offered,token=world.offer();k=next(n for n in offered if n.startswith('INK_'))
        point=offered[k][0]
        world.material[world.idx(point)]=5;world.protected.add(point)
        with self.assertRaises(HostRefusal):world.apply(k,token)

    def test_stopped_policy_remains_stopped_under_all_masks(self):
        p=TinyPolicy();p.weights[FEATURES.index('stop')]=1000
        for mode in MODES:
            world,row=run_world(p,160000,'empty','sweep',mode)
            self.assertTrue(world.stopped);self.assertTrue(row['selected_STOP'])
            self.assertEqual(row['new_ink'],0);self.assertEqual(row['decisions'],1)
            self.assertIsNone(row['decision_to_32_ink'])

    def test_world_pair_refuses_foreign_material(self):
        a={'seed':1,'family':'empty','initial_material_sha256':'a',
           'decision_to_32_ink':5,'action_ids':['STOP']}
        b={**a,'initial_material_sha256':'b'}
        with self.assertRaises(ValueError):comparison([a],[b])

    def test_full_archive_replay_and_adversarial_action_forgery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=write_run(root,train_worlds=2,
                                             eval_worlds=5,gallery_worlds=1)
            self.assertEqual(report['source_sha256'].keys(),
                             {'lab.py','observation_ablation.py'})
            result=audit(root)
            self.assertEqual(result['audited_synthetic_episodes'],40)
            self.assertEqual(len(list((root/'gallery').glob('*.png'))),8)
            obj=json.loads((root/'traces.json').read_text())
            obj['sweep_without_visit_memory'][0][0]['action_id']='STOP'
            (root/'traces.json').write_text(json.dumps(obj,sort_keys=True)+'\n')
            with self.assertRaises(AuditRefusal):audit(root)

    def test_bounded_training_and_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):write_run(tmp,train_worlds=0)
            with self.assertRaises(ValueError):write_run(tmp,eval_worlds=4)
            with self.assertRaises(ValueError):write_run(tmp,gallery_worlds=6)

if __name__=='__main__':unittest.main()
