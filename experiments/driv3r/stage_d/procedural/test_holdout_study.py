"""Exact-source holdout study replay and adversarial custody tests."""
import json
from pathlib import Path
import tempfile
import unittest
from holdout_study import (FROZEN_MANIFEST_SHA256, git_blob, png, run, sha, verify_sources)
from holdout_fixtures import shape, preregister
from fragility_study import decode_synthetic_rgba_png, encode

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'corpus' / 'holdout-prereg-v1.json'

class HoldoutStudyTests(unittest.TestCase):
    def test_manifest_and_source_pins(self):
        self.assertEqual(sha(MANIFEST.read_bytes()), FROZEN_MANIFEST_SHA256)
        verify_sources(ROOT)

    def test_png_reversible_and_masked(self):
        for entry in preregister()['cases']:
            a,b,_=shape(entry['family'],entry['seed'])
            self.assertEqual(decode_synthetic_rgba_png(png(a,b)),(a,b))

    def test_reference_source_modified_refused(self):
        with tempfile.TemporaryDirectory() as t:
            directory = Path(t)
            for name in ('holdout_fixtures.py','reinforcement_study.py','fragility_study.py'):
                (directory/name).write_bytes((ROOT/name).read_bytes())
            (directory/'reinforcement_study.py').write_bytes((directory/'reinforcement_study.py').read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'Git blob mismatch'):
                verify_sources(directory)

    def test_manifest_tampering_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'altered.json'
            data=json.loads(MANIFEST.read_bytes())
            data['budget_max_new_cells']+=1
            p.write_bytes(encode(data))
            with self.assertRaisesRegex(ValueError,'frozen input manifest'):
                run(p,ROOT)

    def test_full_holdout_replay_and_negative_control(self):
        x=run(MANIFEST,ROOT)
        self.assertEqual(len(x['cases']),12)
        self.assertEqual(encode(x),encode(run(MANIFEST,ROOT)))
        for c in x['cases']:
            n=c['actual_new_cells_per_arm']
            self.assertLessEqual(n,20)
            self.assertEqual({v['added_after_baseline'] for v in c['arms'].values()},{n})
        negative=[x for x in x['cases'] if x['id']=='disjoint-blocks'][0]
        self.assertEqual(negative['actual_new_cells_per_arm'],0)
        self.assertEqual(negative['terminal'],'NO_BUDGETED_ARTICULATION_BYPASS')

if __name__=='__main__':unittest.main()
