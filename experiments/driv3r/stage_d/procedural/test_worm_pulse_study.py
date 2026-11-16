"""Exact-source shadow counterfactual and tamper rejection tests (offline)."""
import json
import tempfile
import unittest
from pathlib import Path

from worm_pulse_study import DELTAS, TICKS, artifact, case_id, corpus, probes, replay_case
from worm_horizon_fixtures import SCENES, SEEDS
from worm_sketch import canonical, digest


class PulseStudyTests(unittest.TestCase):
    def test_all_registered_probe_states_and_zero_delta_sham(self):
        total = 0
        for scene in SCENES:
            for seed in SEEDS:
                events = probes(scene, seed)
                self.assertEqual(len(events), len(TICKS) * len(DELTAS))
                for rec, ink, allowed in events:
                    total += 1
                    self.assertIn(tuple(rec['state']['cursor']), allowed)
                    self.assertIn(tuple(rec['chosen'][1]), allowed)
                    self.assertIn(tuple(rec['state']['target'][1]), allowed)
                    self.assertFalse(rec['material_changed'])
                    self.assertIsNone(rec['state']['native_receipt'])
                    if rec['delta_u16'] == 0:
                        self.assertFalse(rec['changed'])
                        self.assertEqual(rec['chosen'], rec['state']['baseline_choice'])
                    self.assertEqual(rec['pulsed_guide_u16'], min(65535, rec['state']['original_guide_u16'] + rec['delta_u16']))
                    self.assertIn(rec['delta_u16'], DELTAS)
        self.assertEqual(total, len(SCENES) * len(SEEDS) * len(TICKS) * len(DELTAS))

    def test_source_deterministic_bytes_and_genuine_baseline_state(self):
        first = probes('canals', 23)
        second = probes('canals', 23)
        for a, b in zip(first, second):
            self.assertEqual(artifact(*a), artifact(*b))
            self.assertEqual(a[0]['state'], b[0]['state'])
            self.assertEqual(case_id(a[0]), case_id(b[0]))

    def test_full_corpus_replays_and_rehash_cannot_launder_edit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = corpus(root)
            self.assertEqual(len(report['records']), 192)
            for rec in report['records']:
                self.assertTrue(replay_case(root / rec['case']))
            path = root / report['records'][0]['case']
            target = path / 'choice-overlay.png'
            target.write_bytes(target.read_bytes() + b'tamper')
            mpath = path / 'manifest.json'
            m = json.loads(mpath.read_bytes())
            m['artifacts']['choice-overlay.png'] = digest(target.read_bytes())
            mpath.write_bytes(canonical(m))
            with self.assertRaises(ValueError):
                replay_case(path)

    def test_registered_cases_only(self):
        with self.assertRaises(ValueError):
            probes('bad', 23)
        with self.assertRaises(ValueError):
            probes('open', 999)


if __name__ == '__main__':
    unittest.main()
