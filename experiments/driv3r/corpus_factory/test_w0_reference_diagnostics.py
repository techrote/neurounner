#!/usr/bin/env python3
"""Small self-contained fixture for read-only W0 reference diagnostics."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile
from w0_reference_diagnostics import analyze, canonical


class W0ReferenceDiagnosticsTests(unittest.TestCase):
    def _fixture(self, directory, forged=False):
        sources = []
        rows = []
        for arm in ('model-assisted', 'fidget', 'random', 'wall'):
            member = f'test/greebles-{arm}-9.json'
            origin_kind = 'model-assisted' if arm == 'model-assisted' else 'model-free-control'
            sources.append({'json_member': member, 'origin_kind': origin_kind,
                            'json_sha256': 'a' * 64, 'png_sha256': 'b' * 64,
                            'family': 'greebles', 'seed': 9, 'events': 2,
                            'observed_action_histogram': {'STRAIGHT': 2}})
            for tick in range(2):
                rows.append({'source_json_member': member,
                             'split': 'evaluation-reference-only',
                             'train_eligible': False,
                             'teacher_label': 'STRAIGHT' if (forged and tick == 1) else None,
                             'evidence_class': 'W0-SYNTHETIC-INFERENCE-EVALUATION-ONLY',
                             'source_zip_sha256': 'c' * 64,
                             'source_json_sha256': 'a' * 64,
                             'source_png_sha256': 'b' * 64,
                             'origin_kind': origin_kind, 'family': 'greebles',
                             'seed': 9, 'tick': tick,
                             'observed_applied_action_not_teacher': 'STRAIGHT',
                             'observation': {'tick': tick, 'x': 7 + tick, 'y': 48,
                                             'legal': ['STRAIGHT', 'STOP']},
                             'host_position_after': [8 + tick, 48]})
        payload = ''.join(canonical(row) + '\n' for row in rows).encode()
        manifest = {'schema': 'cybersand.game-corpus.w0-evaluation-export/1',
                    'split': 'evaluation-reference-only', 'teacher_labels': 0,
                    'train_examples': 0, 'native_game_records': 0,
                    'model_optimizer_steps': 0, 'archive_crc': 'PASS',
                    'host_legal_receipts': len(rows), 'source_zip_sha256': 'c' * 64,
                    'original_trace_count': len(sources), 'origin_traces': sources,
                    'files': {'w0-evaluation-decisions.jsonl': {
                        'sha256': hashlib.sha256(payload).hexdigest(),
                        'bytes': len(payload), 'rows': len(rows)}}}
        path = Path(directory) / 'source.zip'
        with ZipFile(path, 'w') as z:
            z.writestr('manifest.json', canonical(manifest))
            z.writestr('w0-evaluation-decisions.jsonl', payload)
        return path

    def test_read_only_matched_control_comparison(self):
        with TemporaryDirectory() as td:
            result = analyze(self._fixture(td))
            self.assertEqual(result['decision_receipts'], 8)
            self.assertEqual(result['world_count'], 1)
            self.assertEqual(result['trainer_eligible_records'], 0)
            self.assertEqual(result['teacher_labels'], 0)
            self.assertEqual(result['matched_worlds'][0]['model_unique_cells'], 3)

    def test_even_rehashed_teacher_label_is_rejected(self):
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'TRAIN quarantine'):
                analyze(self._fixture(td, forged=True))


if __name__ == '__main__':
    unittest.main()
