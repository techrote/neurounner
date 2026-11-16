"""Source-fenced, cheap DEVELOPMENT tests for waypoint rescue and legal controls."""
import json
import tempfile
import unittest
from pathlib import Path
from worm_sketch import canonical, digest
from worm_horizon_fixtures import SCENES, SEEDS
from worm_waypoint_study import frozen_state, run_case as prior_run_case
from worm_waypoint_recovery_study import ARMS, HORIZON, corpus, run_case, verify

class WaypointRecoveryTests(unittest.TestCase):
    def test_source_sham_and_previous_hold_arm_match(self):
        for scene in SCENES:
            for seed in SEEDS:
                old=prior_run_case(scene,seed,60,'hold12_8192')
                new=run_case(scene,seed,60,'hold12_8192')
                self.assertEqual(new['record']['path'],old['record']['path'])
                self.assertEqual(new['ink'],old['final_ink'])
                src=frozen_state(scene,seed,60)
                sham=run_case(scene,seed,60,'sham')
                self.assertEqual(sham['ink'],src['baseline']['ink'])
                self.assertEqual([tuple(p) for p in sham['record']['path']],src['baseline']['path'])

    def test_oracle_reaches_all_reachable_goals(self):
        for scene in SCENES:
            for seed in SEEDS:
                x=run_case(scene,seed,60,'oracle_geodesic')['record']
                self.assertTrue(x['metrics']['reached_goal'])
                self.assertLess(x['metrics']['first_reached_step'],HORIZON)
                self.assertEqual(x['metrics']['min_distance'],0)

    def test_gain_saturation_reveals_oracle_collapse(self):
        """High-gain steering must be labelled effectively hard, not 'soft'."""
        for scene in SCENES:
            for seed in SEEDS:
                oracle=run_case(scene,seed,60,'oracle_geodesic')['record']
                high=run_case(scene,seed,60,'full32_16384')['record']
                moderate=run_case(scene,seed,60,'full32_8192')['record']
                self.assertEqual(high['path'],oracle['path'])
                self.assertEqual(high['metrics']['material_sha256'],oracle['metrics']['material_sha256'])
                self.assertTrue(moderate['metrics']['reached_goal'])
                self.assertEqual(high['metrics']['geodesic_choices'],HORIZON)

    def test_safety_material_chain_and_shared_start(self):
        for scene in SCENES:
            for seed in SEEDS:
                initial=frozen_state(scene,seed,60)
                for arm in ARMS:
                    with self.subTest(scene=scene,seed=seed,arm=arm):
                        out=run_case(scene,seed,60,arm);r=out['record']
                        self.assertEqual(tuple(r['path'][0]),initial['pos'])
                        self.assertEqual(len(r['path']),HORIZON+1)
                        self.assertTrue(all(tuple(p) in out['allowed'] for p in r['path']))
                        self.assertEqual(r['events'][-1]['kind'],'UNDO_MATERIAL_EDIT_UNAVAILABLE')
                        self.assertFalse(r['events'][-1]['executed'])
                        self.assertFalse(r['native_replay_verified'])
                        self.assertLessEqual(r['metrics']['rescue_boost_steps'],12)
                        self.assertEqual(r['events'][0]['material_before_sha256'],r['input']['original_material_sha256'])
                        for a,b in zip(r['events'][:-2],r['events'][1:-1]):
                            self.assertEqual(a['material_after_sha256'],b['material_before_sha256'])
                        self.assertEqual(r['events'][-1]['material_after_sha256'],r['metrics']['material_sha256'])
                        self.assertTrue(all(e['native_receipt'] is None for e in r['events']))

    def test_unregistered_parameters_fail_closed(self):
        for args in [('open',23,60,'unknown'),('nonexistent',23,60,'sham'),
                     ('open',23,12,'sham'),('open',5000,60,'sham')]:
            with self.subTest(args=args),self.assertRaises(ValueError):run_case(*args)

    def test_corpus_replay_tamper_rehash_and_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'episodes';a=corpus(root)
            self.assertEqual(len(a['cases']),len(SCENES)*len(SEEDS)*len(ARMS))
            case=root/'baffled-seed23-tick60-rescue_hysteresis'
            self.assertTrue(verify(case))
            original=(case/'case.json').read_bytes();m_orig=(case/'manifest.json').read_bytes()
            try:
                data=json.loads(original);data['metrics']['min_distance']=-1
                (case/'case.json').write_bytes(canonical(data))
                m=json.loads(m_orig);m['artifacts']['case.json']=digest((case/'case.json').read_bytes())
                (case/'manifest.json').write_bytes(canonical(m))
                with self.assertRaises(ValueError):verify(case)
            finally:
                (case/'case.json').write_bytes(original)
                (case/'manifest.json').write_bytes(m_orig)
            self.assertEqual(a,corpus(root))
            self.assertTrue(verify(case))

if __name__ == '__main__':unittest.main()
