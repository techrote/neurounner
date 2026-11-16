"""Source-custodied 32-decision pulse/recovery comparison: no model/native API."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worm_pulse_horizon_study import (DELTAS, HORIZON, TICKS, SCENES, SEEDS,
                                      artifacts, baseline, corpus, run_case, verify)
from worm_sketch import canonical, digest


class PulseHorizonTests(unittest.TestCase):
    def test_zero_pulse_is_exact_prefix_matched_sham(self):
        for scene in SCENES:
            for seed in SEEDS:
                for tick in TICKS:
                    r, b, f=run_case(scene,seed,tick,0)
                    self.assertEqual(r['baseline_path'],r['fork_path'])
                    self.assertEqual(r['baseline_events'],r['fork_events'])
                    self.assertEqual(r['baseline_material_sha256'],r['fork_material_sha256'])
                    self.assertFalse(r['metrics']['initial_decision_changed'])
                    self.assertEqual(r['metrics']['novel_fork_only'],0)
                    self.assertEqual(r['metrics']['mismatched_cursor_ticks'],0)

    def test_only_first_step_pulsed_and_legality_and_no_native_history(self):
        for d in DELTAS:
            rec, b, fork = run_case('ring',23,60,d)
            self.assertEqual(len(rec['fork_events']),HORIZON)
            self.assertTrue(all(e['native_receipt'] is None and e['model_decision'] is None for e in rec['fork_events']))
            self.assertEqual(sum(e['first_step_pulse'] for e in rec['fork_events']),1)
            self.assertTrue(all(tuple(e['to_cursor']) in fork['allowed'] for e in rec['fork_events']))
            self.assertFalse(rec['material_undo_executed'])
            self.assertIsNone(rec['native_receipt'])
            if not rec['metrics']['initial_decision_changed']:
                self.assertEqual(rec['baseline_path'],rec['fork_path'])

    def test_frozen_original_baseline_reconstructed(self):
        for scene in SCENES:
            for seed in SEEDS:
                spec, allow, guide, source, states = baseline(scene,seed)
                self.assertEqual(len(states),source['metrics']['steps'])
                self.assertEqual(spec['version'],'d2.worm-horizon-inputs/1')

    def test_corpus_complete_replay_and_rehashed_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            report=corpus(root)
            self.assertEqual(len(report['cases']),len(SCENES)*len(SEEDS)*len(TICKS)*len(DELTAS))
            for item in report['cases']:
                self.assertTrue(verify(root/item['case']))
            case=root/report['cases'][0]['case']
            data=case/'difference.png'
            data.write_bytes(data.read_bytes()+b'evil')
            mpath=case/'manifest.json'
            m=json.loads(mpath.read_bytes())
            m['artifacts']['difference.png']=digest(data.read_bytes())
            mpath.write_bytes(canonical(m))
            with self.assertRaises(ValueError): verify(case)

    def test_only_registered_inputs(self):
        for args in [('open',23,60,999),('missing',23,60,2048),('open',999,60,0),('open',23,13,0)]:
            with self.assertRaises(ValueError):run_case(*args)


if __name__=='__main__': unittest.main()
