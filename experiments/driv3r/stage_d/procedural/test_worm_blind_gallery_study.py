"""All assessors' assets must be replayable, anonymous, and unbiased by labels."""
import json
import tempfile
import unittest
from pathlib import Path

from worm_blind_gallery_study import PULSES, plan, corpus, verify


class BlindGalleryTests(unittest.TestCase):
    def test_predeclared_cases_not_outcome_selected(self):
        tasks=plan()
        self.assertEqual(len(tasks),36)
        self.assertEqual({d for _,_,_,d in tasks},set(PULSES))
        self.assertEqual(len(set(tasks)),36)
        self.assertEqual(sum(delta==0 for *_,delta in tasks),12)

    def test_complete_source_replayed_gallery_and_balanced_sides(self):
        with tempfile.TemporaryDirectory() as t:
            r=corpus(Path(t)/'gallery')
            self.assertTrue(verify(Path(t)/'gallery'))
            self.assertEqual(len(r['assignments']),36)
            self.assertEqual(sum(a['baseline_side']=='left' for a in r['assignments']),18)
            self.assertTrue(all(a['image_sha256'] for a in r['assignments']))
            ballot=json.loads((Path(t)/'gallery/review/ballot.json').read_bytes())
            self.assertEqual(len(ballot['items']),36)
            self.assertTrue(all(i['preference']=='unrated' for i in ballot['items']))
            self.assertNotIn('pulse_u16', str(ballot))
            for a in r['assignments']:
                if a['pulse_u16'] == 0:
                    self.assertEqual(a['image_sha256']['left'],a['image_sha256']['right'])

    def test_rehashed_tamper_and_manifest_laundering_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'gallery'
            corpus(path)
            left=path/'review/item-001-left.png'
            left.write_bytes(left.read_bytes()+b'fake')
            r=json.loads((path/'custody/assignments.json').read_bytes())
            r['assignments'][0]['image_sha256']['left']='0'*64
            (path/'custody/assignments.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):verify(path)

    def test_clean_output_only(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'gallery'
            path.mkdir()
            (path/'old.txt').write_text('old')
            with self.assertRaises(ValueError):corpus(path)


if __name__=='__main__': unittest.main()
