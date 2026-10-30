"""Standard-library-only result validation tests; no models or physical timings."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('runner_summary', ROOT / 'scripts/runner_001_summary.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def set_path(data, path, value):
    keys = path.split('.')
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def record(lane='ubuntu-24.04-x64', replicate=1):
    row = {'schema': 'neurounner.runner001.v2',
           'source': {'dataset_canonical_sha256': S.EXPECTED_CORPUS_SHA256,
                      'local_model': S.EXPECTED_MODEL, 'local_revision': S.EXPECTED_REVISION},
           'machine': {'lane': lane, 'replicate': replicate, 'logical_cpus': 4,
                       'memory_bytes': 16000000000, 'cpu_model': 'Synthetic test CPU',
                       'python': '3.12.0', 'runner_os': 'macOS' if lane == 'macos-15-arm64' else 'Linux',
                       'runner_arch': 'ARM64' if lane == 'macos-15-arm64' else 'X64',
                       'github_run_id': '123', 'github_run_attempt': '1', 'github_sha': 'a' * 40}}
    for path in S.METRIC_PATHS.values():
        set_path(row, path, 20.0)
    local = row['benchmarks']['local_model']
    local.update(model=S.EXPECTED_MODEL, revision=S.EXPECTED_REVISION,
                 versions={'torch': '2.14.1' if lane == 'macos-15-arm64' else '2.14.1+cpu',
                           'transformers': '4.57.6', 'peft': '0.21.2', 'huggingface_hub': '0.36.2',
                           'psutil': '7.2.2', 'safetensors': '0.8.0'})
    return row


class RunnerSummaryTests(unittest.TestCase):
    def batch(self):
        return [record(lane, rep) for lane in S.LANE_ORDER for rep in (1, 2, 3)]

    def aggregate(self, rows, succeeds):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, row in enumerate(rows):
                (root / f'runner001-test-{index}.json').write_text(json.dumps(row), encoding='utf-8')
            output, md = root / 'out/summary.json', root / 'out/summary.md'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/runner_001_summary.py'),
                                     '--input-dir', str(root), '--json-out', str(output), '--md-out', str(md)],
                                    capture_output=True, text=True)
            if succeeds:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(md.exists())
                return json.loads(output.read_text())
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists(), 'invalid records must not emit a summary')
            self.assertFalse(md.exists())

    def test_valid_nine_records(self):
        result = self.aggregate(self.batch(), True)
        self.assertEqual(result['schema'], 'neurounner.runner001.summary.v4')
        self.assertEqual(result['source_run'], {'id': '123', 'attempt': '1', 'sha': 'a' * 40})
        self.assertEqual(len(result['rows']), 9)
        self.assertTrue(all(a['replicates'] == 3 for a in result['aggregates']))
        self.assertTrue(all(a['metrics']['generation_tok_s']['median'] == 20.0 for a in result['aggregates']))

    def test_empty_benchmark_reproducer(self):
        rows = self.batch()
        for row in rows:
            row['benchmarks'] = {}
        self.aggregate(rows, False)

    def test_partial_missing_metric(self):
        rows = self.batch()
        del rows[0]['benchmarks']['local_model']['generation']
        self.aggregate(rows, False)

    def test_all_metrics_require_finite_positive_real_values(self):
        for path in S.METRIC_PATHS.values():
            for bad in (None, float('nan'), float('inf'), -float('inf'), 0, -1, True, '15'):
                with self.subTest(path=path, bad=bad):
                    data = record()
                    set_path(data, path, bad)
                    with self.assertRaises(ValueError):
                        S.validate_record(data)

    def test_nan_json_does_not_emit_summary(self):
        rows = self.batch()
        rows[0]['wall_seconds'] = float('nan')
        self.aggregate(rows, False)

    def test_model_and_revision_pins(self):
        for path in ('source.local_model', 'source.local_revision', 'benchmarks.local_model.model', 'benchmarks.local_model.revision'):
            data = record()
            set_path(data, path, 'different')
            with self.assertRaises(ValueError):
                S.validate_record(data)

    def test_wrong_uniform_corpus(self):
        rows = self.batch()
        for row in rows:
            row['source']['dataset_canonical_sha256'] = 'b' * 64
        self.aggregate(rows, False)

    def test_mixed_run_attempt_or_source(self):
        for key, value in (('github_run_id', '124'), ('github_run_attempt', '2'), ('github_sha', 'b' * 40)):
            rows = self.batch()
            rows[1]['machine'][key] = value
            self.aggregate(rows, False)

    def test_malformed_identity(self):
        for key, value in (('github_run_id', None), ('github_run_attempt', '0'), ('github_sha', 'invalid')):
            data = record()
            data['machine'][key] = value
            with self.assertRaises(ValueError):
                S.validate_record(data)

    def test_replica_coercion(self):
        for value in (True, '1', 1.2, 0, 4):
            with self.assertRaises(ValueError):
                S.validate_record(record(replicate=value))

    def test_missing_and_duplicate_replicas(self):
        self.aggregate(self.batch()[:-1], False)
        rows = self.batch()
        rows[-1] = copy.deepcopy(rows[-2])
        self.aggregate(rows, False)

    def test_unknown_lane_or_schema(self):
        for key, value in (('machine.lane', 'unknown'), ('schema', 'unknown')):
            data = record()
            set_path(data, key, value)
            with self.assertRaises(ValueError):
                S.validate_record(data)

    def test_missing_machine_metadata(self):
        for path in ('machine.logical_cpus', 'machine.memory_bytes', 'machine.cpu_model'):
            data = record()
            set_path(data, path, None)
            with self.assertRaises(ValueError):
                S.validate_record(data)

    def test_package_pin_drift(self):
        for key in record()['benchmarks']['local_model']['versions']:
            data = record()
            data['benchmarks']['local_model']['versions'][key] = 'changed'
            with self.assertRaises(ValueError):
                S.validate_record(data)

    def test_corpus_digest_matches_repository(self):
        data = json.loads((ROOT / 'experiments/train-001/kernel_compiler_dataset.json').read_bytes())
        raw = json.dumps(data, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), S.EXPECTED_CORPUS_SHA256)

    def test_physical_workflow_manual_only(self):
        source = (ROOT / '.github/workflows/runner-001.yml').read_text()
        self.assertIn("  benchmark:\n    if: ${{ github.event_name == 'workflow_dispatch' }}\n    needs: source-checks", source)
        job = source.split('  source-checks:', 1)[1].split('  benchmark:', 1)[0]
        self.assertIn('test_runner_001_summary.py', job)
        self.assertNotIn('pip install', job)
        self.assertNotIn('--local-model', job)


if __name__ == '__main__':
    unittest.main()
