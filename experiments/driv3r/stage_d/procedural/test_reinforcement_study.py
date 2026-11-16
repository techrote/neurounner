"""Adversarial synthetic-only tests for post-freeze structural reinforcement."""
import json
from pathlib import Path
import unittest

from fragility_study import sha, encode
from reinforcement_study import (articulations, bounded_bypass,
    best_reinforcement, equal_edit_control, reinforce, trial, run_archive)

ARCHIVE = Path(__file__).parent / 'provisional-evidence.zip'

class ReinforcementTests(unittest.TestCase):
    def test_tarjan_large_grid_no_recursion(self):
        self.assertEqual(articulations({(x,y) for x in range(32) for y in range(32)}),set())

    def test_tarjan_chain_and_loop(self):
        chain = {(x, 2) for x in range(1, 6)}
        self.assertEqual(articulations(chain), {(x,2) for x in range(2,5)})
        loop = chain | {(1,3),(2,3),(3,3),(4,3),(5,3)}
        self.assertEqual(articulations(loop), set())

    def test_reinforces_line_with_permission(self):
        old = {(x, 2) for x in range(1, 6)}
        allowed = {(x,y) for x in range(8) for y in range(5)}
        final, plans, stop = reinforce(old, allowed, budget=9)
        self.assertTrue(plans)
        self.assertTrue(old <= final <= allowed)
        self.assertLess(len(articulations(final)), len(articulations(old)))
        self.assertLessEqual(len(final-old),9)
        for record in plans:
            route = record['route']
            self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1
                                for a,b in zip(route,route[1:])))
            self.assertEqual(record['before_material_sha256'] !=
                             record['after_material_sha256'],True)

    def test_enclosed_permission_refuses(self):
        initial = {(x,2) for x in range(1,6)}
        final,plans,why = reinforce(initial,set(initial),budget=24)
        self.assertEqual(final, initial)
        self.assertEqual(plans, [])
        self.assertEqual(why, 'NO_BUDGETED_ARTICULATION_BYPASS')

    def test_bypass_rejects_unpainted_endpoint(self):
        a={(1,2),(2,2),(3,2)}
        with self.assertRaises(ValueError):
            bounded_bypass(a,set(a),(2,2),(1,2),(3,3))

    def test_edit_budget_zero_preserves_material(self):
        old={(1,2),(2,2),(3,2)}
        permitted={(x,y) for x in range(5) for y in range(5)}
        final,plans,terminal=reinforce(old,permitted,budget=0)
        self.assertEqual((final,plans,terminal),(old,[],'EDIT_BUDGET_EXHAUSTED'))

    def test_determinism_and_control_denominator(self):
        old={(x,2) for x in range(1,6)}
        permitted={(x,y) for x in range(8) for y in range(5)}
        guidance=[(x*8191+y*13)%65536 for y in range(32) for x in range(32)]
        first=trial(old,old,permitted,guidance,7,budget=12)
        self.assertEqual(first,trial(old,old,permitted,guidance,7,budget=12))
        edits=[v['added_after_baseline'] for v in first['policies'].values()]
        self.assertEqual(len(set(edits)),1)
        for row in first['policies'].values():
            self.assertGreaterEqual(row['articulations_after'],0)

    def test_adversarial_permission(self):
        old={(x,2) for x in range(1,6)}
        permitted={(x,y) for x in range(8) for y in range(5)} - {(x,1) for x in range(8)}
        final,plans,stop=reinforce(old,permitted,budget=8)
        self.assertTrue(final <= permitted)
        self.assertTrue(all(set(map(tuple,p['edits']))<=permitted for p in plans))

    def test_bad_guidance_rejected(self):
        initial={(1,1),(2,1),(3,1)}
        with self.assertRaises(ValueError):
            trial(initial,initial,initial,[0],7)

    @unittest.skipUnless(ARCHIVE.exists(),'optional provisional source archive not present')
    def test_archive_rejects_guidance_tamper(self):
        import tempfile
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / 'mutated.zip'
            with zipfile.ZipFile(ARCHIVE) as source, zipfile.ZipFile(dest,'w') as sink:
                for name in source.namelist():
                    data = source.read(name)
                    if name.endswith('/guidance.u16le'):
                        data = bytes([data[0] ^ 1]) + data[1:]
                    sink.writestr(name, data)
            with self.assertRaisesRegex(ValueError, 'corrupted guidance'):
                run_archive(dest, budget=12)

    @unittest.skipUnless(ARCHIVE.exists(),'optional provisional source archive not present')
    def test_archive_custody_and_replay(self):
        a=run_archive(ARCHIVE, budget=12)
        self.assertEqual(len(a['cases']),8)
        self.assertFalse(a['input_policy_source_verified'])
        self.assertEqual(a['qualification'],'POST_FREEZE_EXPLORATORY_PROVISIONAL_ONLY')
        self.assertEqual(encode(a),encode(run_archive(ARCHIVE,budget=12)))
        self.assertTrue(any(c['policies']['reinforcement']['added_after_baseline']>0
                            for c in a['cases']))

if __name__=='__main__':
    unittest.main()
