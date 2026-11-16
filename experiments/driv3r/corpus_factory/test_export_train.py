#!/usr/bin/env python3
"""Qualification: TRAIN-only export, reproducibility and tamper resistance."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from corpus import build, canonical, audit, partition
from export_train import export, FIELDS, observed_fingerprint


class TrainExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        cls.corpus = cls.root / 'corpus'
        build(cls.corpus, worlds_per_scenario=10, budget=24, png_limit=0)
        assert audit(cls.corpus)['audit'] == 'PASS'

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def test_train_only_no_teacher_metadata_in_prompts(self):
        with tempfile.TemporaryDirectory() as td:
            exported = Path(td) / 'selected'
            report = export(self.corpus, exported, per_action_cap=7, per_world_cap=4)
            self.assertEqual(report['split'], 'train-only')
            self.assertEqual(report['native_game_records'], 0)
            self.assertEqual(report['model_optimizer_steps'], 0)
            examples = [json.loads(x) for x in (exported / 'train.messages.jsonl').read_text().splitlines()]
            references = [json.loads(x) for x in (exported / 'train.provenance.jsonl').read_text().splitlines()]
            self.assertEqual(len(examples), len(references))
            self.assertTrue(examples)
            train_groups = {json.loads(x)['world_group_id'] for x in (self.corpus / 'train.worlds.jsonl').read_text().splitlines()}
            dev_groups = {json.loads(x)['world_group_id'] for x in (self.corpus / 'dev.worlds.jsonl').read_text().splitlines()}
            heldout_groups = {json.loads(x)['world_group_id'] for x in (self.corpus / 'heldout.worlds.jsonl').read_text().splitlines()}
            self.assertFalse(train_groups & (dev_groups | heldout_groups))
            for item, ref in zip(examples, references):
                messages = item['messages']
                self.assertEqual([m['role'] for m in messages], ['system','user','assistant'])
                obs = json.loads(messages[1]['content'])
                self.assertEqual(set(obs), set(FIELDS))
                self.assertEqual(observed_fingerprint(obs), ref['input_sha256'])
                self.assertIn(messages[2]['content'], obs['legal'])
                self.assertIn(ref['world_group_id'], train_groups)
                self.assertNotIn(ref['world_group_id'], dev_groups | heldout_groups)
                for forbidden in ('candidate_scores','receipt','target_action','hard_negatives','world_group_id','record_id'):
                    self.assertNotIn(forbidden, messages[1]['content'])
            for name, item in report['files'].items():
                payload = (exported / name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item['sha256'])
                self.assertEqual(len(payload), item['bytes'])

    def test_byte_determinism_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / 'a', Path(td) / 'b'
            export(self.corpus, a, per_action_cap=5, per_world_cap=5, sampler_seed=17)
            export(self.corpus, b, per_action_cap=5, per_world_cap=5, sampler_seed=17)
            for filename in ('train.messages.jsonl','train.provenance.jsonl','export-manifest.json'):
                self.assertEqual((a / filename).read_bytes(), (b / filename).read_bytes())
            with self.assertRaises(FileExistsError):
                export(self.corpus, a, per_action_cap=5, per_world_cap=5)

    def test_tampered_observation_even_if_manifest_rehashed(self):
        with tempfile.TemporaryDirectory() as td:
            changed = Path(td) / 'forged'
            shutil.copytree(self.corpus, changed)
            file = changed / 'train.decisions.jsonl'
            rows = [json.loads(x) for x in file.read_text().splitlines()]
            rows[0]['observation']['heading'] = (rows[0]['observation']['heading'] + 1) % 4
            payload = ''.join(canonical(row)+'\n' for row in rows).encode()
            file.write_bytes(payload)
            mf = changed / 'manifest.json'
            manifest = json.loads(mf.read_text())
            manifest['files'][file.name]['sha256'] = hashlib.sha256(payload).hexdigest()
            manifest['files'][file.name]['bytes'] = len(payload)
            mf.write_text(canonical(manifest) + '\n')
            self.assertEqual(audit(changed)['audit'], 'PASS')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                export(changed, Path(td) / 'refused')

    def test_dev_or_model_derived_rows_cannot_enter_train(self):
        with tempfile.TemporaryDirectory() as td:
            changed = Path(td) / 'forged'
            shutil.copytree(self.corpus, changed)
            file = changed / 'train.decisions.jsonl'
            rows = [json.loads(x) for x in file.read_text().splitlines()]
            rows[0]['evidence_class'] = 'model-generated-policy-trace'
            payload = ''.join(canonical(row)+'\n' for row in rows).encode()
            file.write_bytes(payload)
            mf = changed / 'manifest.json'
            manifest = json.loads(mf.read_text())
            manifest['files'][file.name]['sha256'] = hashlib.sha256(payload).hexdigest()
            manifest['files'][file.name]['bytes'] = len(payload)
            mf.write_text(canonical(manifest) + '\n')
            self.assertEqual(audit(changed)['audit'], 'PASS')
            with self.assertRaisesRegex(ValueError, 'non-TRAIN or non-synthetic'):
                export(changed, Path(td) / 'refused')


    def _expect_rehashed_source_refusal(self, filename, mutate, expected):
        with tempfile.TemporaryDirectory() as td:
            changed = Path(td) / 'forged'
            shutil.copytree(self.corpus, changed)
            file = changed / filename
            rows = [json.loads(x) for x in file.read_text().splitlines()]
            mutate(rows)
            payload = ''.join(canonical(row) + '\n' for row in rows).encode()
            file.write_bytes(payload)
            mf = changed / 'manifest.json'
            manifest = json.loads(mf.read_text())
            manifest['files'][filename]['sha256'] = hashlib.sha256(payload).hexdigest()
            manifest['files'][filename]['bytes'] = len(payload)
            mf.write_text(canonical(manifest) + '\n')
            self.assertEqual(audit(changed)['audit'], 'PASS')
            with self.assertRaisesRegex(ValueError, expected):
                export(changed, Path(td) / 'refused')

    def test_rehashed_observable_position_tamper_refused(self):
        def corrupt(rows):
            rows[0]['observation']['x'] += 1
        self._expect_rehashed_source_refusal('train.decisions.jsonl', corrupt,
                                            'trail/receipt')

    def test_rehashed_observable_budget_tamper_refused(self):
        def corrupt(rows):
            rows[0]['observation']['budget_remaining'] += 2
        self._expect_rehashed_source_refusal('train.decisions.jsonl', corrupt,
                                            'position/heading/budget')

    def test_rehashed_dev_partition_disguised_as_train_refused(self):
        def corrupt(rows):
            candidate = 0
            while partition(f'partition-forgery:{candidate}') == 'train':
                candidate += 1
            rows[0]['world_group_id'] = f'partition-forgery:{candidate}'
        self._expect_rehashed_source_refusal('train.worlds.jsonl', corrupt,
                                            'world partition')

    def test_rehashed_material_observation_and_fingerprint_refused(self):
        # A forged manifest AND matching local-input fingerprint must not
        # turn a false material observation into a model training prompt.
        def corrupt(rows):
            row = rows[0]
            obs = row['observation']
            old = obs['local_material_5x5'][0][0]
            obs['local_material_5x5'][0][0] = 'metal' if old != 'metal' else 'sand'
            row['input_sha256'] = observed_fingerprint(obs)
        self._expect_rehashed_source_refusal(
            'train.decisions.jsonl', corrupt, 'reconstructed world/trail')

    def test_rehashed_symbolic_observation_and_fingerprint_refused(self):
        def corrupt(rows):
            row = rows[0]
            obs = row['observation']
            old = obs['local_5x5'][0][0]
            obs['local_5x5'][0] = ('#' if old != '#' else '.') + obs['local_5x5'][0][1:]
            row['input_sha256'] = observed_fingerprint(obs)
        self._expect_rehashed_source_refusal(
            'train.decisions.jsonl', corrupt, 'reconstructed world/trail')

    def test_rehashed_world_fixture_refused(self):
        def corrupt(rows):
            rows[0]['material_grid_sha256'] = '0' * 64
        self._expect_rehashed_source_refusal(
            'train.worlds.jsonl', corrupt, 'world fixture')

    def test_rehashed_heading_without_changing_trail_is_refused(self):
        # A valid-range but wrong heading passes local-grid and receipt checks;
        # it must not become an allegedly faithful TRAIN observation.
        def corrupt(rows):
            row = rows[0]
            obs = row['observation']
            obs['heading'] = (obs['heading'] + 1) % 4
            row['input_sha256'] = observed_fingerprint(obs)
        self._expect_rehashed_source_refusal(
            'train.decisions.jsonl', corrupt, 'deterministic producer replay')

    def test_rehashed_forged_host_legal_menu_is_refused(self):
        def corrupt(rows):
            row = rows[0]
            obs = row['observation']
            self.assertNotIn('RETRACE', obs['legal'])  # first tick: no history
            obs['legal'].append('RETRACE')
            row['input_sha256'] = observed_fingerprint(obs)
        self._expect_rehashed_source_refusal(
            'train.decisions.jsonl', corrupt, 'deterministic producer replay')

    def test_rehashed_forged_teacher_action_and_receipt_is_refused(self):
        # The old validator checked receipt self-consistency but did not
        # recompute which choice the seeded teacher actually selected.
        def corrupt(rows):
            row = rows[0]
            other = next(a for a in row['observation']['legal']
                         if a != row['target_action'])
            row['target_action'] = other
            row['receipt']['raw_choice'] = other
            row['receipt']['applied'] = other
        self._expect_rehashed_source_refusal(
            'train.decisions.jsonl', corrupt, 'deterministic producer replay')

    def test_rehashed_episode_branch_history_is_refused(self):
        def corrupt(rows):
            ep = rows[0]
            ep['branch_points'] = [[4, 4]] if ep['branch_points'] != [[4, 4]] else []
        self._expect_rehashed_source_refusal(
            'train.episodes.jsonl', corrupt, 'deterministic producer replay')

    def test_refuse_source_tree_output(self):
        with self.assertRaisesRegex(ValueError,'overwrite source'):
            export(self.corpus, self.corpus / 'new_export')


if __name__ == '__main__':
    unittest.main()
