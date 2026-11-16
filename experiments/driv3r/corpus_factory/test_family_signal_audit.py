#!/usr/bin/env python3
"""Qualified read-only analysis; no model/native or generator mutations."""
import json
import hashlib
import shutil
from pathlib import Path
import tempfile
import unittest
from corpus import build, audit, canonical
from family_signal_audit import analyze, render_signature

class FamilySignalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.dir.name) / 'source'
        build(cls.root, worlds_per_scenario=10, budget=24, png_limit=0)
    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_exact_three_families_and_source_immutability(self):
        before = (self.root / 'manifest.json').read_bytes()
        result = analyze(self.root)
        self.assertEqual(result['source_audit']['audit'], 'PASS')
        self.assertEqual(result['total']['worlds'], 60)
        self.assertEqual(result['total']['family_episodes'], 180)
        self.assertEqual(sum(result['total'][f'exact_render_variants_{n}'] for n in (1,2,3)), 60)
        self.assertEqual(result['native_game_records'], 0)
        self.assertEqual((self.root / 'manifest.json').read_bytes(), before)
        self.assertEqual(canonical(result), canonical(analyze(self.root)))

    def test_collapsed_scenarios_are_not_credited_as_style_diversity(self):
        result = analyze(self.root)
        for scenario in ('corridor','deceptive','protected','deadend'):
            self.assertEqual(result['by_scenario'][scenario]['exact_render_variants_1'],10)

    def test_rehashed_episode_forgery_refused(self):
        with tempfile.TemporaryDirectory() as td:
            changed = Path(td) / 'tampered'
            shutil.copytree(self.root, changed)
            name = 'train.episodes.jsonl'
            file = changed / name
            rows = [json.loads(line) for line in file.read_text().splitlines()]
            rows[0]['branch_points'] = [[3,3]]
            payload = ''.join(canonical(row)+'\n' for row in rows).encode()
            file.write_bytes(payload)
            mf = changed / 'manifest.json'
            m = json.loads(mf.read_text())
            m['files'][name]['sha256'] = hashlib.sha256(payload).hexdigest()
            m['files'][name]['bytes'] = len(payload)
            mf.write_text(canonical(m)+'\n')
            self.assertEqual(audit(changed)['audit'], 'PASS')
            with self.assertRaisesRegex(ValueError, 'pinned deterministic producer'):
                analyze(changed)

    def test_render_signature_ignores_order_only_not_branch_or_tip(self):
        ep = {'trail': [[7,7],[8,7],[7,7]], 'branch_points': [[7,7]]}
        same_image = {'trail': [[7,7],[7,7],[8,7],[7,7]], 'branch_points': [[7,7]]}
        other_branch = {'trail': [[7,7],[8,7],[7,7]], 'branch_points': []}
        other_tip = {'trail': [[7,7],[7,7],[8,7]], 'branch_points': [[7,7]]}
        self.assertEqual(render_signature(ep), render_signature(same_image))
        self.assertNotEqual(render_signature(ep), render_signature(other_branch))
        self.assertNotEqual(render_signature(ep), render_signature(other_tip))

if __name__ == '__main__':
    unittest.main()
