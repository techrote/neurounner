"""Offline source-frozen paired-controls and synthetic artifact replay tests."""
import json
import tempfile
import unittest
from pathlib import Path

from worm_horizon_study import ARMS, blobs, corpus, run_case, verify
from worm_horizon_fixtures import SCENES, SEEDS, fixture, preregistration
from worm_escape_study import run_one_escape
from worm_sketch import DIRS, canonical, digest


class HorizonStudyTests(unittest.TestCase):
    def test_old_baselines_exact_parity_on_legacy_scenes(self):
        for scene in ('open', 'slats', 'ring'):
            for seed in SEEDS:
                for arm, old in (('strong_x16', 'strong_x16'), ('short_escape', 'hysteresis'),
                                 ('always_novel', 'always_novel')):
                    a, b = run_case(scene, seed, arm), run_one_escape(scene, seed, old)
                    self.assertEqual(a['trail'], b['trail'], (scene, seed, arm))
                    self.assertEqual(a['material'], b['material'])
                    self.assertEqual(a['metrics']['escape_triggers'], b['metrics']['escape_triggers'])

    def test_sham_same_trajectory_and_material(self):
        for scene in SCENES:
            for seed in SEEDS:
                a, b = run_case(scene, seed, 'strong_x16'), run_case(scene, seed, 'sham_sustained')
                self.assertEqual(a['trail'], b['trail'])
                self.assertEqual(a['material'], b['material'])

    def test_actual_hard_geometry_and_material_event_chain(self):
        for scene in SCENES:
            for arm in ARMS:
                a = run_case(scene, 23, arm)
                allowed = a['permitted']
                paint = set()
                last = a['trail'][0]
                for e in a['events']:
                    self.assertIsNone(e.get('native_receipt'))
                    if e['kind'] == 'STEP':
                        target = tuple(e['to_cursor'])
                        self.assertIn(target, allowed)
                        self.assertEqual(abs(last[0]-target[0])+abs(last[1]-target[1]), 1)
                        self.assertIn([e['heading'], e['to_cursor']], e['legal_catalogue'])
                        self.assertEqual(e['material_before_sha256'], digest(canonical(sorted(map(list, paint)))))
                        if e['pen'] == 'down':
                            paint.add(target)
                        self.assertEqual(e['material_after_sha256'], digest(canonical(sorted(map(list, paint)))))
                        last = target
                    elif e['kind'].startswith('ESCAPE_'):
                        self.assertIs(e['material_changed'], False)
                    elif e['kind'] == 'UNDO_MATERIAL_EDIT_UNAVAILABLE':
                        self.assertIs(e['executed'], False)
                self.assertEqual(paint, a['material'])
                self.assertEqual(last, a['trail'][-1])

    def test_determinism_all_arms(self):
        for scene in SCENES:
            for seed in SEEDS:
                for arm in ARMS:
                    self.assertEqual(blobs(run_case(scene, seed, arm)), blobs(run_case(scene, seed, arm)))

    def test_frozen_original_has_same_hard_and_guide_across_arms(self):
        for scene in SCENES:
            for seed in SEEDS:
                orig = fixture(scene, seed)[0]
                for arm in ARMS:
                    self.assertEqual(run_case(scene, seed, arm)['spec']['input'], orig)

    def test_replay_rejects_content_and_rehashed_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            corpus(Path(d))
            path = Path(d) / 'canals-seed23-sustained_escape'
            self.assertTrue(verify(path))
            view = path / 'material.png'
            view.write_bytes(view.read_bytes() + b'tamper')
            with self.assertRaises(ValueError):
                verify(path)
            manifest_path = path / 'manifest.json'
            manifest = json.loads(manifest_path.read_bytes())
            manifest['artifacts']['material.png'] = digest(view.read_bytes())
            manifest_path.write_bytes(canonical(manifest))
            with self.assertRaises(ValueError):
                verify(path)

    def test_bounded_inputs_and_invalid_arms(self):
        for steps in (0, -1, 513, True, 1.2):
            with self.assertRaises(ValueError):
                run_case('open', 23, 'strong_x16', steps)
        with self.assertRaises(ValueError):
            run_case('open', 23, 'material_undo')
        with self.assertRaises(ValueError):
            run_case('unregistered', 23, 'strong_x16')
