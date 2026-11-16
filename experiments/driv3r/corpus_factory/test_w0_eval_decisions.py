"""Offline tests of evaluation-only W0 host receipts; no game import."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from w0_eval_decisions import extract


class QuarantineTests(unittest.TestCase):
    def fixture(self, root, mutate=None):
        obs = {'tick': 0, 'x': 7, 'y': 48, 'family': 'greebles',
               'heading': 0, 'last_actions': [],
               'local_5x5': ['.....', '.....', '..@..', '.....', '.....'],
               'legal': ['STRAIGHT', 'STOP']}
        event = {'observation': obs, 'applied': 'STRAIGHT',
                 'raw_choice': 'STRAIGHT', 'raw_legal': True,
                 'position_after': [8, 48]}
        trace = {'contract': 'cybersand.worm.observation-action/0-experimental',
                 'size': 96, 'family': 'greebles', 'seed': 7, 'step_budget': 1,
                 'steps_executed': 1, 'trajectory': [[7, 48], [8, 48]],
                 'events': [event], 'mode': 'random'}
        if mutate:
            mutate(trace)
        name = 'fleet-0/results/a/original-png-and-json/test.json'
        png = name[:-5] + '.png'
        raw = json.dumps(trace).encode()
        pngraw = b'\x89PNG\r\n\x1a\nfixture'
        receipts = {
            'a/original-png-and-json/test.json': hashlib.sha256(raw).hexdigest(),
            'a/original-png-and-json/test.png': hashlib.sha256(pngraw).hexdigest()}
        archive = root / 'source.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(name, raw)
            z.writestr(png, pngraw)
            z.writestr('fleet-0/results/SHA256SUMS.json', json.dumps(receipts))
        return archive

    def test_emits_evaluation_only_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = extract(self.fixture(root), root / 'out')
            self.assertEqual(
                (data['train_examples'], data['teacher_labels'],
                 data['host_legal_receipts']), (0, 0, 1))
            self.assertEqual(data['sha256_member_receipts_checked'], 2)
            rec = json.loads((root / 'out' / 'w0-evaluation-decisions.jsonl').read_text())
            self.assertFalse(rec['train_eligible'])
            self.assertIsNone(rec['teacher_label'])
            self.assertEqual(rec['observed_applied_action_not_teacher'], 'STRAIGHT')

    def test_refuses_trajectory_divergence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = self.fixture(root, lambda t: t['trajectory'].__setitem__(0, [3, 5]))
            with self.assertRaisesRegex(ValueError, 'observation/receipt mismatch'):
                extract(archive, root / 'out')

    def test_refuses_host_illegal_action(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = self.fixture(
                root, lambda t: t['events'][0].__setitem__('applied', 'RETRACE'))
            with self.assertRaisesRegex(ValueError, 'observation/receipt mismatch'):
                extract(archive, root / 'out')

    def test_refuses_duplicate_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = self.fixture(root)
            extract(archive, root / 'out')
            with self.assertRaisesRegex(ValueError, 'existing output'):
                extract(archive, root / 'out')


if __name__ == '__main__':
    unittest.main()
