"""Regress source custody, blind-rating admission, and all-case accounting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from worm_sketch import canonical
from worm_blind_gallery_study import corpus
from worm_blind_ballot_analysis import evaluate, write_analysis


class BallotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = tempfile.TemporaryDirectory()
        cls.root = Path(cls.h.name) / 'source'
        corpus(cls.root)
        cls.template = json.loads((cls.root/'review'/'ballot.json').read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls.h.cleanup()

    def rated(self, preference='tie', difference='none'):
        b = copy.deepcopy(self.template)
        b['status'] = 'RATED'
        for item in b['items']:
            item.update(preference=preference, difference=difference)
        return b

    def evaluate_ballot(self, b):
        path = Path(self.h.name)/'rated.json'
        path.write_bytes(canonical(b))
        return evaluate(self.root, path)

    def test_no_ratings_fabricated_and_partial_refused(self):
        with self.assertRaises(ValueError): self.evaluate_ballot(self.template)
        b = self.rated()
        b['items'][14]['preference'] = 'unrated'
        with self.assertRaises(ValueError): self.evaluate_ballot(b)

    def test_complete_ties_shams_and_all_arms(self):
        r = self.evaluate_ballot(self.rated())
        self.assertEqual(r['number_of_rated_items'], 36)
        self.assertEqual(sum(a['total'] for a in r['arms'].values()), 36)
        self.assertEqual(r['sham_catch']['false_difference'], 0)
        self.assertEqual(r['sham_catch']['forced_preference'], 0)
        self.assertEqual(r['all_identical_image_disagreements'], 0)
        self.assertEqual(r['arms']['0']['physically_identical_pairs'], 12)
        self.assertEqual(r['arms']['4096']['total'], 12)
        self.assertEqual(r['arms']['8192']['total'], 12)
        self.assertEqual(r, self.evaluate_ballot(self.rated()))

    def test_sham_preference_and_false_difference_disclosed_not_filtered(self):
        b = self.rated(preference='left', difference='clear')
        r = self.evaluate_ballot(b)
        self.assertEqual(r['sham_catch']['false_difference'], 12)
        self.assertEqual(r['sham_catch']['forced_preference'], 12)
        self.assertGreaterEqual(r['all_identical_image_disagreements'], 12)
        self.assertEqual(len(r['records']), 36)

    def test_order_protocol_and_injected_fields_refused(self):
        for edit in ('reverse','identity','inject','status'):
            b = self.rated()
            if edit == 'reverse': b['items'].reverse()
            if edit == 'identity': b['items'][0]['left'] = 'relabelled.png'
            if edit == 'inject': b['items'][0]['pulse_u16'] = 8192
            if edit == 'status': b['status'] = 'PARTIAL'
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                self.evaluate_ballot(b)

    def test_source_image_tampering_and_custody_laundering_refused(self):
        pic = self.root/'review/item-018-left.png'
        original = pic.read_bytes()
        try:
            pic.write_bytes(original+b'fake')
            with self.assertRaises(ValueError): self.evaluate_ballot(self.rated())
        finally:
            pic.write_bytes(original)
        path = self.root/'custody/assignments.json'
        original = path.read_bytes()
        try:
            data = json.loads(original)
            data['assignments'][0]['pulse_u16'] = 1234
            path.write_bytes(canonical(data))
            with self.assertRaises(ValueError): self.evaluate_ballot(self.rated())
        finally:
            path.write_bytes(original)

    def test_write_once_to_new_path(self):
        b = self.rated()
        filled = Path(self.h.name)/'filled.json'
        filled.write_bytes(canonical(b))
        destination = Path(self.h.name)/'results.json'
        write_analysis(self.root, filled, destination)
        self.assertEqual(json.loads(destination.read_bytes())['number_of_rated_items'], 36)
        with self.assertRaises(ValueError): write_analysis(self.root, filled, destination)


if __name__ == '__main__': unittest.main()
