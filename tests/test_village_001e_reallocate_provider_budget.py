import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import village_001e_provider_checkpoint as c
import village_001e_recover_provider_budget as frozen
import village_001e_reallocate_provider_budget as a


CAMPAIGN = {'nvidia': {'endpoint': 'https://invalid.example', 'generation': {'attempts': 6}}}
MODEL = {'id': 'critic'}
LABEL = 'F20-behavior-sibling-02'
ROLE = 'tournament-behavior-sibling-classifier'


def prompt(ids, repair):
    return 'Fixed rubric.\nITEMS:\n' + json.dumps([{'id': ident, 'text': 'Fixed source response.'} for ident in ids])


def collect(cp):
    return cp.collect(label=LABEL, expected=['R001', 'R002'], prompt_factory=prompt,
        system='fixed system', model=MODEL, max_tokens=4096, role=ROLE,
        parser=lambda raw, expected: json.loads(raw))


def archived_api(answer, calls=None):
    def api(**kw):
        if calls is not None:
            calls.append(kw)
        raw = answer(kw) if callable(answer) else answer
        out, label = kw['outdir'], kw['label']
        c.write_once(out / f'{label}-request.json', {'provider': 'nvidia', 'role': kw['role'], 'payload': kw['payload']})
        c.write_once(out / f'{label}-provider.json', {'choices': [{'message': {'content': raw}}]})
        c.write_once(out / f'{label}-attempts.json', [{'attempt': 1, 'status': 200}])
        (out / f'{label}-content.txt').write_text(raw + '\n')
        return raw, {}, []
    return api


class AllocationTests(unittest.TestCase):
    def setUp(self):
        quiet = contextlib.redirect_stderr(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(a, 'ROOT', self.root).start()
        self.provider_identity = {'logical_request_limit': 42, 'scenario_id': 'F20', 'fixed': 'identity'}
        self.original_source = self.root / 'original'
        c.prepare_output(self.original_source, None, self.provider_identity)
        initial_payload = {'model': MODEL['id'], 'stage': 0}
        self.old = {'kind': 'tournament', 'lane': 'F20', 'artifact_name': 'village001e-judge-tournament-F20',
            'terminal': 'scenario.json', 'status': 'unstarted', 'original_logical_limit': 42,
            'original_files': frozen.tree(self.original_source),
            'original_provider_identity_sha256': c.sha256_file(self.original_source / 'provider-identity.json'),
            'original_request_labels': [], 'original_logical_labels': [], 'original_logical_count': 0,
            'effective_logical_limit': 40, 'supplemental_logical_reservation': 40, 'artifact': None,
            'next_unarchived_request': {'logical_label': 'F20-stage-00', 'payload_sha256': c.sha256_value(initial_payload), 'role': 'fixture'}}
        self.original_manifest = {'lanes': [self.old], 'super_model': 'super', 'budget': {'maximum_total_logical_requests': 1081}}
        self.original_path = self.root / 'manifest.json'
        frozen.write_once(self.original_path, self.original_manifest)
        self.old_identity = frozen.recovery_identity(self.original_path, self.original_manifest, self.old)
        self.source = self.root / 'baseline'
        c.prepare_output(self.source, self.original_source, self.provider_identity)
        cls = frozen.checkpoint_factory(self.old, self.original_manifest, self.old_identity, offline=False, capture=[])
        cp = cls(self.source, campaign=CAMPAIGN, key='fixture', offline=False,
                 api=archived_api(lambda kw: '{"R001":"first valid"}' if kw['label'] == LABEL else '{}'), logical_request_limit=42)
        for number in range(39):
            cp.request(f'F20-stage-{number:02d}', {'model': MODEL['id'], 'stage': number}, 'fixture')
        with self.assertRaisesRegex(RuntimeError, 'whole-lane provider budget exhausted'):
            collect(cp)
        actual, logical = frozen.labels(self.source)
        repair_label = LABEL + '-missing-r1-b01'
        payload = c.payload_for(MODEL, prompt(['R002'], True), 'fixed system', 4096, repair=True)
        payload = frozen.amended_payload(repair_label, payload, set(), 'super')
        self.row = {key: self.old[key] for key in ('kind', 'lane', 'artifact_name', 'terminal', 'status')}
        self.row.update({'baseline_status': 'pending', 'baseline_source_path': 'lanes/' + self.old['artifact_name'],
            'baseline_files': frozen.tree(self.source), 'baseline_actual_labels': sorted(actual), 'baseline_logical_labels': sorted(logical),
            'baseline_logical_count': 40, 'previous_effective_logical_limit': 40, 'effective_logical_limit': 60,
            'remaining_logical_reservation': 20, 'next_unarchived_request': {
                'actual_label': repair_label, 'logical_label': repair_label, 'payload_sha256': c.sha256_value(payload),
                'role': ROLE + '-repair', 'model': MODEL['id'], 'max_tokens': 4096}})
        self.allocation = {'lanes': [self.row], 'original_manifest': {'path': 'manifest.json', 'sha256': c.sha256_file(self.original_path)},
            'prior_recovery_run': {'id': a.BASELINE_RUN}, 'prior_recovery_artifact': {'digest': 'sha256:' + a.BASELINE_ZIP_SHA256},
            'budget': {'maximum_total_logical_requests': 1059}}
        self.allocation_path = self.root / 'allocation.json'
        frozen.write_once(self.allocation_path, self.allocation)

    def checkpoint(self, source, out, api, offline=False):
        c.prepare_output(out, source, self.provider_identity)
        capture = []
        cls = a.checkpoint_factory(self.allocation, self.original_manifest, self.row, offline=offline, capture=capture)
        cp = cls(out, campaign=CAMPAIGN, key='' if offline else 'fixture', offline=offline, api=api, logical_request_limit=42)
        self.assertIs(capture[0], cp)
        return cp

    def replay_prefix(self, cp):
        for number in range(39):
            cp.request(f'F20-stage-{number:02d}', {'model': MODEL['id'], 'stage': number}, 'fixture')

    def test_extension_preserves_original_identities_and_first_valid_values(self):
        before = frozen.tree(self.source)
        calls = []
        live = self.root / 'live'
        cp = self.checkpoint(self.source, live, archived_api('{"R001":"replacement","R002":"completed"}', calls))
        self.replay_prefix(cp)
        values, _ = collect(cp)
        self.assertEqual(values, {'R001': 'first valid', 'R002': 'completed'})
        self.assertEqual(len(calls), 1)
        self.assertIn('exactly 1 classifications rows', calls[0]['payload']['messages'][1]['content'])
        cp.finish('scenario.json', values)
        self.assertTrue(all(frozen.tree(live).get(name) == value for name, value in before.items()))
        self.assertEqual((live / 'provider-identity.json').read_bytes(), (self.source / 'provider-identity.json').read_bytes())
        self.assertEqual((live / 'budget-recovery-identity.json').read_bytes(), (self.source / 'budget-recovery-identity.json').read_bytes())
        self.assertEqual(frozen.read(live / 'budget-recovery-identity.json')['effective_logical_limit'], 40)
        self.assertEqual(frozen.read(live / a.IDENTITY)['effective_logical_limit'], 60)
        a.validate_prior_source(live, self.allocation, self.row, original=self.original_manifest)
        replay = self.root / 'replay'
        cp2 = self.checkpoint(live, replay, lambda **kw: self.fail('offline replay called provider'), offline=True)
        self.replay_prefix(cp2)
        second, _ = collect(cp2)
        cp2.finish('scenario.json', second)
        self.assertEqual((live / 'scenario.json').read_bytes(), (replay / 'scenario.json').read_bytes())
        self.assertEqual(cp2.new_calls, 0)

    def test_new_cap_exact_boundary_and_persisted_exhaustion_across_resume(self):
        live = self.root / 'live'
        cp = self.checkpoint(self.source, live, archived_api('{"R002":"completed"}'))
        self.replay_prefix(cp)
        collect(cp)
        for number in range(19):
            cp.request(f'F20-later-{number:02d}', {'model': MODEL['id'], 'later': number}, 'fixture')
        self.assertEqual(len(cp.logical_labels), 60)
        resumed = self.root / 'resumed'
        cp2 = self.checkpoint(live, resumed, lambda **kw: self.fail('exhausted cap must prevent API call'))
        self.replay_prefix(cp2)
        collect(cp2)
        before = frozen.tree(resumed)
        with self.assertRaisesRegex(RuntimeError, 'whole-lane provider budget exhausted'):
            cp2.request('F20-one-too-many', {'model': MODEL['id']}, 'fixture')
        self.assertEqual(frozen.tree(resumed), before)
        self.assertEqual(cp2.new_calls, 0)

    def test_baseline_cannot_drop_or_rewrite_a_request_to_reset_spending(self):
        (self.source / 'provider/F20-stage-01-request.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'snapshot bytes changed or disappeared'):
            a.validate_prior_source(self.source, self.allocation, self.row, original=self.original_manifest)

    def test_post_baseline_calls_require_exact_extension_identity(self):
        c.write_once(self.source / 'provider/new-request.json', {'payload': {'model': 'critic'}})
        with self.assertRaisesRegex(RuntimeError, 'lacks allocation extension identity'):
            a.validate_prior_source(self.source, self.allocation, self.row, original=self.original_manifest)

    def test_wrong_extension_or_old_identity_rejected(self):
        c.write_once(self.source / a.IDENTITY, {'allocation_manifest_sha256': 'wrong'})
        with self.assertRaisesRegex(RuntimeError, 'another allocation extension'):
            a.validate_prior_source(self.source, self.allocation, self.row, original=self.original_manifest)
        (self.source / a.IDENTITY).unlink()
        old = self.source / 'budget-recovery-identity.json'
        value = frozen.read(old)
        value['effective_logical_limit'] = 60
        old.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'snapshot bytes changed|original recovery identity changed'):
            a.validate_prior_source(self.source, self.allocation, self.row, original=self.original_manifest)

    def test_first_new_payload_and_actual_transport_boundary_are_frozen(self):
        live = self.root / 'live'
        cp = self.checkpoint(self.source, live, lambda **kw: self.fail('wrong boundary must prevent API call'))
        self.replay_prefix(cp)
        with self.assertRaisesRegex(RuntimeError, 'frozen allocation pending boundary'):
            cp.request('F20-unrelated-new-work', {'model': 'critic'}, 'fixture')
        self.assertEqual(frozen.labels(live)[0], set(self.row['baseline_actual_labels']))

    def test_keyboard_interrupt_receipt_is_interrupted_not_completed(self):
        def interrupted_api(**kw):
            c.write_once(kw['outdir'] / (kw['label'] + '-request.json'),
                         {'provider': 'nvidia', 'role': kw['role'], 'payload': kw['payload']})
            raise KeyboardInterrupt('fixture cancellation')

        def invoke(config, kind, lane, source, out, cls, *, offline, quiet):
            c.prepare_output(out, source, self.provider_identity)
            cp = cls(out, campaign=CAMPAIGN, key='fixture', offline=False, api=interrupted_api, logical_request_limit=42)
            self.replay_prefix(cp)
            collect(cp)

        out = self.root / 'interrupted'
        with patch.object(frozen, 'invoke', side_effect=invoke), patch.dict(os.environ, {'GITHUB_RUN_ATTEMPT': '1'}):
            with self.assertRaises(KeyboardInterrupt):
                a.run_lane(self.allocation_path, self.allocation, self.original_manifest, self.root / 'config.json',
                           self.row, self.source, out, offline=False, quiet=True)
        receipt = frozen.read(out / 'allocation-extension-execution-001.json')
        self.assertEqual(receipt['status'], 'interrupted')
        self.assertFalse(receipt['terminal_verified'])
        self.assertEqual(receipt['new_logical_requests'], 1)
        self.assertEqual(receipt['new_transport_launches'], 1)
        self.assertEqual(receipt['logical_requests_archived'], 41)
        self.assertFalse((out / 'scenario.json').exists())
        resumed = self.root / 'after-interruption'
        calls = []
        cp = self.checkpoint(out, resumed, archived_api('{"R002":"completed"}', calls))
        self.replay_prefix(cp)
        collect(cp)
        self.assertEqual(len(cp.logical_labels), 41)
        self.assertEqual(calls[0]['label'], LABEL + '-missing-r1-b01-transport-resume-01')


class AllocationArithmeticTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for number in range(41):
            count = 30 if number == 40 else 15
            rows.append({'kind': 'tournament', 'lane': f'complete-{number}', 'baseline_status': 'completed',
                'baseline_logical_count': count, 'baseline_actual_labels': list(range(count)),
                'effective_logical_limit': count, 'remaining_logical_reservation': 0})
        for lane, count in [('F15', 40), ('F19', 3), ('F20', 0), ('F21', 0), ('F22', 0), ('F23', 0), ('F24', 0)]:
            cap = 69 if lane == 'F15' else 60
            row = {'kind': 'tournament', 'lane': lane, 'baseline_status': 'pending', 'baseline_logical_count': count,
                'baseline_actual_labels': list(range(count)), 'effective_logical_limit': cap,
                'remaining_logical_reservation': cap - count}
            if lane == 'F15':
                row['next_unarchived_request'] = {'logical_label': 'F15-behavior-sibling-02-missing-r1-b04'}
            rows.append(row)
        return rows

    def test_approved_allocation_retires_complete_and_stays_below_original_global_cap(self):
        budget = a.allocation_budget(self.rows())
        self.assertEqual(budget['completed_lanes'], 41)
        self.assertEqual(budget['completed_lane_logical_requests'], 630)
        self.assertEqual(budget['baseline_logical_requests_archived'], 673)
        self.assertEqual(budget['maximum_total_logical_requests'], 1059)
        self.assertEqual(budget['remaining_logical_requests_reserved'], 386)
        self.assertEqual(budget['unallocated_logical_slots'], 45)

    def test_global_overbooking_and_changed_structural_exception_fail_closed(self):
        rows = self.rows()
        rows.append({'kind': 'tournament', 'lane': 'extra', 'baseline_status': 'pending', 'baseline_logical_count': 0,
                     'baseline_actual_labels': [], 'effective_logical_limit': 60, 'remaining_logical_reservation': 60})
        with self.assertRaisesRegex(RuntimeError, 'global logical ceiling'):
            a.allocation_budget(rows)
        rows = self.rows()
        rows[41]['next_unarchived_request']['logical_label'] = 'wrong-phase'
        with self.assertRaisesRegex(RuntimeError, 'structural exception pending boundary'):
            a.allocation_budget(rows)
        rows = self.rows()
        rows[0]['effective_logical_limit'] += 1
        with self.assertRaisesRegex(RuntimeError, 'allocation cap'):
            a.allocation_budget(rows)

    def test_baseline_bundle_requires_exact_complete_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lane = root / 'inherited/lane'
            lane.mkdir(parents=True)
            (lane / 'request.json').write_text('fixed')
            files = frozen.tree(root)
            allocation = {'prior_bundle_tree_sha256': c.sha256_value(files), 'prior_bundle_file_count': len(files),
                'prior_bundle_file_bytes': sum(row['bytes'] for row in files.values()),
                'lanes': [{'baseline_source_path': 'inherited/lane', 'baseline_files': frozen.tree(lane)}]}
            a.validate_baseline_bundle(root, allocation)
            (lane / 'request.json').write_text('rewritten')
            with self.assertRaisesRegex(RuntimeError, 'exact frozen recovery bundle'):
                a.validate_baseline_bundle(root, allocation)


if __name__ == '__main__':
    unittest.main()
