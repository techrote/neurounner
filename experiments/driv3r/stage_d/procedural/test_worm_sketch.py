"""Offline line-worm replay and scene admission tests; no external dependencies."""
import json
from pathlib import Path
import tempfile
import unittest

from worm_sketch import (DIRS, POLICIES, canonical, default_specs, digest,
                         environment, field, recipe, run, simulate, replay)


class LineWormTest(unittest.TestCase):
    def test_preregistered_development_grid(self):
        specs = default_specs()
        self.assertEqual(len(specs), 18)
        self.assertEqual(len({(s['scene'], s['seed'], s['policy']) for s in specs}), 18)

    def test_cardinal_host_legal_and_pen_custody(self):
        for s in default_specs():
            r = simulate(s)
            ink = set()
            legal = environment(*s['canvas'], s['scene'])
            guide = field(*s['canvas'], s['seed'])
            self.assertEqual(r['permitted'], legal)
            for e in r['events']:
                self.assertIsNone(e.get('native_receipt'))
                if e['kind'] != 'STEP':
                    self.assertFalse(e.get('executed', False))
                    continue
                old, p = tuple(e['from_cursor']), tuple(e['to_cursor'])
                self.assertIn(p, legal)
                self.assertEqual(sum(abs(p[i]-old[i]) for i in (0,1)), 1)
                self.assertIn([e['heading'],list(p)], e['legal_catalogue'])
                self.assertEqual(e['guide_u16'], guide[p])
                self.assertEqual(e['material_before_sha256'], digest(canonical(sorted(map(list, ink)))))
                if e['pen'] == 'down':
                    ink.add(p)
                self.assertEqual(e['material_after_sha256'], digest(canonical(sorted(map(list, ink)))))
                self.assertEqual(e['changed'], e['material_before_sha256'] != e['material_after_sha256'])
            self.assertEqual(ink, r['material'])
            self.assertEqual(len(r['trail']) - 1, r['metrics']['steps'])

    def test_matched_baselines_and_descriptive_variance(self):
        for scene in ('open', 'slats', 'ring'):
            for seed in (23, 79):
                arms = {p:simulate(recipe(scene, seed, p)) for p in POLICIES}
                self.assertEqual(len({tuple(a['recipe']['origin']) for a in arms.values()}), 1)
                self.assertEqual(len({frozenset(a['permitted']) for a in arms.values()}), 1)
                self.assertEqual(len({tuple(sorted(a['guide'].items())) for a in arms.values()}), 1)
                self.assertGreater(arms['wayward']['metrics']['turns'], arms['stubborn']['metrics']['turns'])
                self.assertLess(arms['wayward']['metrics']['turns'], arms['random_legal']['metrics']['turns'])
                self.assertNotEqual(arms['wayward']['metrics']['trajectory_sha256'],
                                    arms['stubborn']['metrics']['trajectory_sha256'])

    def test_admission(self):
        valid=recipe('open',23,'wayward')
        cases = (dict(valid,steps=513),dict(valid,policy='unsafe'),
                 dict(valid,canvas=[16,16]),dict(valid,seed=-1),
                 dict(valid,origin=[0,0]),dict(valid,pen_rhythm={'period':0,'lifted':0}),
                 dict(valid,initial_heading=4))
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises((AssertionError,ValueError)):
                    simulate(bad)
        with self.assertRaises(ValueError):
            environment(80,80,'forged')

    def test_replay_byte_identity_and_manifest(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a=run(first,default_specs())
            b=run(second,default_specs())
            self.assertEqual(a,b)
            for case in a['cases']:
                for root in (first,second):
                    folder=Path(root)/case['case']
                    self.assertEqual(json.loads((folder/'manifest.json').read_bytes()),
                                     {k:v for k,v in case.items() if k!='case'})
                    self.assertTrue(replay(folder))
                    self.assertFalse(case['native_replay_verified'])
                    self.assertIsNone(case['native'])
                    self.assertIsNone(case['model'])
                    for filename, meta in case['artifacts'].items():
                        data=(folder/filename).read_bytes()
                        self.assertEqual(meta,{'bytes':len(data),'sha256':digest(data)})
            self.assertEqual((Path(first)/'index.json').read_bytes(),
                             (Path(second)/'index.json').read_bytes())

    def test_rehashed_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run(folder, [recipe('ring', 23, 'wayward', steps=50)])
            case = Path(folder) / 'ring-seed23-wayward'
            self.assertTrue(replay(case))
            material = case / 'material-after.png'
            original = material.read_bytes()
            mutated = original[:-1] + bytes([original[-1] ^ 1])
            material.write_bytes(mutated)
            manifest = json.loads((case / 'manifest.json').read_bytes())
            manifest['artifacts']['material-after.png'] = {
                'sha256': digest(mutated), 'bytes': len(mutated)}
            (case / 'manifest.json').write_bytes(canonical(manifest))
            with self.assertRaises(AssertionError):
                replay(case)

    def test_no_duplicate_case_output_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            example=recipe('open',23,'wayward',steps=20)
            with self.assertRaises(AssertionError):
                run(folder, [example,example])

    def test_hard_permission_persists_under_all_policies(self):
        for scene in ('slats','ring'):
            for policy in POLICIES:
                s=recipe(scene,23,policy)
                r=simulate(s)
                self.assertTrue(r['material'] <= r['permitted'])
                self.assertGreater(len(r['material']), 0)


if __name__=='__main__':
    unittest.main()
