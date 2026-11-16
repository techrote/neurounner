"""Bounded DEVELOPMENT waypoint study: exact source, legality, sham and custody."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from worm_sketch import canonical
from worm_horizon_fixtures import SCENES, SEEDS
from worm_pulse_study import TICKS
from worm_waypoint_study import (ARMS, HORIZON, START_TICKS, corpus, frozen_state,
                                 run_case, verify)

class WaypointTests(unittest.TestCase):
    def test_input_provenance_and_reachable_goals(self):
        for scene in SCENES:
            for seed in SEEDS:
                for tick in START_TICKS:
                    with self.subTest(scene=scene,seed=seed,tick=tick):
                        f=frozen_state(scene,seed,tick)
                        self.assertIn(f['goal'],f['allowed'])
                        self.assertGreaterEqual(f['distances'][f['pos']],6)
                        self.assertLessEqual(f['distances'][f['pos']],20)

    def test_sham_exact_32_step_baseline_parity(self):
        for scene in SCENES:
            for seed in SEEDS:
                for tick in START_TICKS:
                    f=frozen_state(scene,seed,tick)
                    z=run_case(scene,seed,tick,'sham')
                    self.assertEqual([tuple(p) for p in z['record']['path']], f['baseline']['path'])
                    self.assertEqual(z['final_ink'], f['baseline']['ink'])
                    self.assertEqual(z['record']['metrics']['mismatched_cursor_steps'],0)
                    self.assertEqual(z['record']['metrics']['novel_ink_only'],0)

    def test_all_arms_obey_hard_masks_material_hashes_and_same_start(self):
        s=frozen_state('baffled',23,60)
        for arm in ARMS:
            z=run_case('baffled',23,60,arm)
            rec=z['record']
            self.assertEqual(tuple(rec['path'][0]),s['pos'])
            self.assertEqual(len(rec['path']),HORIZON+1)
            self.assertTrue(all(tuple(p) in z['allowed'] for p in rec['path']))
            self.assertEqual(rec['events'][-1]['kind'],'UNDO_MATERIAL_EDIT_UNAVAILABLE')
            self.assertFalse(rec['events'][-1]['executed'])
            self.assertTrue(all(e['native_receipt'] is None for e in rec['events']))
            self.assertTrue(all(e['model_decision'] is None for e in rec['events'][:-1]))
            self.assertEqual(rec['events'][-1]['material_after_sha256'], rec['metrics']['material_sha256'])
            self.assertFalse(rec['native_replay_verified'])
            self.assertEqual(rec['events'][0]['material_before_sha256'],rec['input']['baseline_start_material_sha256'])
            for a,b in zip(rec['events'][:-2],rec['events'][1:-1]):
                self.assertEqual(a['material_after_sha256'],b['material_before_sha256'])

    def test_unregistered_params_fail_closed(self):
        for args in [('open',23,12,'unknown'), ('not-a-scene',23,12,'sham'),
                     ('open',100,12,'sham'), ('open',23,99,'sham')]:
            with self.subTest(args=args),self.assertRaises(ValueError): run_case(*args)

    def test_corpus_replay_and_tamper_rehash_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            report=corpus(root)
            self.assertEqual(len(report['records']),len(SCENES)*len(SEEDS)*len(START_TICKS)*len(ARMS))
            selected=root/'baffled-seed23-tick60-hold12_4096'
            self.assertTrue(verify(selected))
            original=(selected/'case.json').read_bytes()
            try:
                x=json.loads(original);x['metrics']['reached_goal']=not x['metrics']['reached_goal']
                (selected/'case.json').write_bytes(canonical(x))
                m=json.loads((selected/'manifest.json').read_bytes())
                from worm_sketch import digest
                m['artifacts']['case.json']=digest((selected/'case.json').read_bytes())
                (selected/'manifest.json').write_bytes(canonical(m))
                with self.assertRaises(ValueError): verify(selected)
            finally:
                (selected/'case.json').write_bytes(original)
            # Fully rebuilding restores verified original bytes and source custody.
            self.assertEqual(report,corpus(root))
            self.assertTrue(verify(selected))

if __name__=='__main__':unittest.main()
