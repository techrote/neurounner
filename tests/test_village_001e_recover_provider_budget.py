import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import village_001e_recover_provider_budget as r
import village_001e_provider_checkpoint as c
import village_001e_ecology_judge as ecology
from village_001e_ecology_judge import super_parts

CAMPAIGN = {'nvidia': {'endpoint': 'https://invalid.example', 'generation': {'attempts': 6}}}
SUPER = {'id': 'frozen-super'}
LABEL = 'E-P-G05-ecology-super-audit'
ROLE = 'ecology-adjudicator'


def archived_api(answer):
    def api(**kw):
        raw = answer(kw) if callable(answer) else answer
        out, label = kw['outdir'], kw['label']
        c.write_once(out / f'{label}-request.json', {'provider': 'nvidia', 'role': kw['role'], 'payload': kw['payload']})
        c.write_once(out / f'{label}-provider.json', {'choices': [{'message': {'content': raw}}]})
        c.write_once(out / f'{label}-attempts.json', [{'attempt': 1, 'status': 200}])
        (out / f'{label}-content.txt').write_text(raw + '\n')
        return raw, {}, []
    return api


def collect(cp):
    return cp.collect(label=LABEL, expected=['distinct_behavior_survives', 'generic_convergence_concerning', 'reason'],
        prompt_factory=lambda ids, repair: json.dumps(ids), system='fixed system', model=SUPER,
        max_tokens=2048, role=ROLE, parser=super_parts)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.original_identity = {'frozen': 'source', 'logical_request_limit': 3}
        c.prepare_output(self.source, None, self.original_identity)
        self.manifest = {'super_model': SUPER['id']}
        self.wrapper_identity = {'schema': 'fixture-recovery', 'original_identity_preserved': True}

    def initial_failure(self):
        def answer(kw):
            if kw['label'] == LABEL:
                return '{"distinct_behavior_survives":false,"generic_convergence_concerning":true,"reason":"bad "quote""}'
            return ''
        cp = c.Checkpoint(self.source, campaign=CAMPAIGN, key='fixture', offline=False, api=archived_api(answer), logical_request_limit=3)
        cp.request('bolt', {'model': 'critic'}, 'critic')
        with self.assertRaisesRegex(RuntimeError, 'whole-lane provider budget exhausted'):
            collect(cp)
        actual, logical = r.labels(self.source)
        next_payload = c.payload_for(SUPER, json.dumps(['reason']), 'fixed system', 2048, repair=True)
        next_payload = r.amended_payload(LABEL + '-missing-r2-b01', next_payload, logical, SUPER['id'])
        return {'original_files': r.tree(self.source), 'original_provider_identity_sha256': c.sha256_file(self.source / 'provider-identity.json'),
            'original_request_labels': sorted(actual), 'original_logical_labels': sorted(logical), 'original_logical_count': 3,
            'original_logical_limit': 3, 'effective_logical_limit': 6, 'supplemental_logical_reservation': 3,
            'next_unarchived_request': {'logical_label': LABEL + '-missing-r2-b01', 'payload_sha256': c.sha256_value(next_payload), 'role': ROLE + '-repair'}}

    def recovered_cp(self, row, out, api, offline=False):
        capture = []
        cls = r.checkpoint_factory(row, self.manifest, self.wrapper_identity, offline=offline, capture=capture)
        cp = cls(out, campaign=CAMPAIGN, key='' if offline else 'fixture', offline=offline, api=api, logical_request_limit=3)
        self.assertIs(capture[0], cp)
        return cp

    def test_only_new_super_missing_repair_gets_8192(self):
        payload = {'model': SUPER['id'], 'messages': ['unchanged'], 'temperature': 0.1, 'top_p': 0.9, 'max_tokens': 2048}
        old = {LABEL, LABEL + '-missing-r1-b01'}
        self.assertIs(r.amended_payload(LABEL + '-missing-r1-b01', payload, old, SUPER['id']), payload)
        self.assertIs(r.amended_payload(LABEL, payload, old, SUPER['id']), payload)
        amended = r.amended_payload(LABEL + '-missing-r2-b01', payload, old, SUPER['id'])
        self.assertEqual(amended, {**payload, 'max_tokens': 8192})
        self.assertEqual(payload['max_tokens'], 2048)
        critic = {**payload, 'model': 'critic'}
        self.assertIs(r.amended_payload('bolt-missing-r2-b01', critic, old, SUPER['id']), critic)
        self.assertIs(r.amended_payload('new-initial-audit', payload, old, SUPER['id']), payload)

    def test_new_sibling_repairs_append_only_actual_id_and_count_requirement(self):
        text = 'Frozen rubric and example {"id":"R001"}.\nITEMS:\n' + json.dumps([
            {'id': 'R013', 'correct_lineage': 'G01', 'text': 'Original behavior.'},
            {'id': 'R014', 'correct_lineage': 'G02', 'text': 'Other original behavior.'}]) + '\n'
        payload = {'model': 'critic', 'messages': [{'role': 'system', 'content': 'Frozen system'}, {'role': 'user', 'content': text}],
                   'temperature': 0.1, 'top_p': 0.9, 'max_tokens': 4096, 'stream': False}
        label = 'F01-behavior-sibling-02-missing-r2-b01'
        result = r.amended_payload(label, payload, set(), SUPER['id'])
        self.assertEqual({k: v for k, v in result.items() if k != 'messages'}, {k: v for k, v in payload.items() if k != 'messages'})
        self.assertEqual(result['messages'][0], payload['messages'][0])
        self.assertTrue(result['messages'][1]['content'].startswith(text))
        appended = result['messages'][1]['content'][len(text):]
        self.assertIn('exactly 2 classifications rows', appended)
        self.assertIn('["R013", "R014"]', appended)
        self.assertEqual(payload['messages'][1]['content'], text)
        self.assertIs(r.amended_payload(label, payload, {label}, SUPER['id']), payload)
        self.assertIs(r.amended_payload('F01-behavior-sibling-02', payload, set(), SUPER['id']), payload)

    def test_zero_call_proof_accepts_only_explicit_skip_or_no_runner_no_steps(self):
        queued = {'steps': [], 'conclusion': 'cancelled', 'runner_id': 0, 'runner_name': '', 'started_at': 'queue-created-timestamp'}
        proof = r.prove_unstarted(queued)
        self.assertEqual(proof['provider_calls'], 0)
        self.assertFalse(proof['zip_artifact_created'])
        skipped = {'steps': [{'name': r.PROVIDER_STEP, 'status': 'completed', 'conclusion': 'skipped'}]}
        self.assertEqual(r.prove_unstarted(skipped)['basis'], 'explicit-provider-step-skipped')
        attempted = {'steps': [{'name': r.PROVIDER_STEP, 'status': 'completed', 'conclusion': 'cancelled'}]}
        with self.assertRaisesRegex(RuntimeError, 'attempted provider step'):
            r.prove_unstarted(attempted)
        with self.assertRaisesRegex(RuntimeError, 'lacks proof'):
            r.prove_unstarted({**queued, 'runner_id': 7, 'runner_name': 'actual runner'})
        with self.assertRaisesRegex(RuntimeError, 'steps are unavailable'):
            r.prove_unstarted({})

    def test_first_valid_fields_old_bytes_and_8192_replay_are_preserved(self):
        row = self.initial_failure()
        out = self.root / 'live'
        c.prepare_output(out, self.source, self.original_identity)
        calls = []
        def answer(kw):
            calls.append(kw['payload'])
            self.assertEqual(kw['payload']['max_tokens'], 8192)
            self.assertEqual(kw['payload']['temperature'], 0.1)
            self.assertEqual(json.loads(kw['payload']['messages'][1]['content']), ['reason'])
            return '{"distinct_behavior_survives":true,"generic_convergence_concerning":false,"reason":"Concrete missing explanation."}'
        cp = self.recovered_cp(row, out, archived_api(answer))
        cp.request('bolt', {'model': 'critic'}, 'critic')
        result, _ = collect(cp)
        self.assertEqual(result, {'distinct_behavior_survives': False, 'generic_convergence_concerning': True, 'reason': 'Concrete missing explanation.'})
        cp.finish('judgment.json', result)
        r.preserve_original(out, row)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(cp.logical_labels), 4)
        replay = self.root / 'replay'
        c.prepare_output(replay, out, self.original_identity)
        cp2 = self.recovered_cp(row, replay, lambda **kw: self.fail('offline provider call'), offline=True)
        cp2.request('bolt', {'model': 'critic'}, 'critic')
        second, _ = collect(cp2)
        cp2.finish('judgment.json', second)
        self.assertEqual((out / 'judgment.json').read_bytes(), (replay / 'judgment.json').read_bytes())
        self.assertEqual(cp2.new_calls, 0)
        self.assertEqual(cp2.replayed_calls, 4)

    def test_effective_cap_exact_boundary_and_persisted_exhaustion(self):
        row = self.initial_failure()
        out = self.root / 'live'
        c.prepare_output(out, self.source, self.original_identity)
        cp = self.recovered_cp(row, out, archived_api('{}'))
        pending = c.payload_for(SUPER, json.dumps(['reason']), 'fixed system', 2048, repair=True)
        cp.request(LABEL + '-missing-r2-b01', pending, ROLE + '-repair')
        cp.request('other-one', {'model': 'critic'}, 'critic')
        cp.request('other-two', {'model': 'critic'}, 'critic')
        self.assertEqual(len(cp.logical_labels), 6)
        before = r.tree(out)
        resumed = self.root / 'resumed'
        c.prepare_output(resumed, out, self.original_identity)
        cp2 = self.recovered_cp(row, resumed, lambda **kw: self.fail('cap must stop call'))
        cp2.request(LABEL + '-missing-r2-b01', pending, ROLE + '-repair')
        with self.assertRaisesRegex(RuntimeError, 'whole-lane provider budget exhausted'):
            cp2.request('other-three', {'model': 'critic'}, 'critic')
        self.assertEqual(r.tree(resumed), before)
        self.assertEqual(cp2.new_calls, 0)

    def test_original_identity_or_raw_mutation_rejected_before_recovery(self):
        row = self.initial_failure()
        path = self.source / 'provider' / (LABEL + '-content.txt')
        path.write_text('rewritten original judgment')
        with self.assertRaisesRegex(RuntimeError, 'original provider archive bytes'):
            r.preserve_original(self.source, row)

    def test_unbound_supplemental_archive_cannot_be_used_as_resume(self):
        row = self.initial_failure()
        c.write_once(self.source / 'provider/new-request.json', {'provider': 'nvidia', 'payload': {'model': SUPER['id']}})
        with self.assertRaisesRegex(RuntimeError, 'supplemental provider history lacks'):
            r.preserve_original(self.source, row)

    def test_new_work_must_begin_at_frozen_probe_boundary(self):
        row = self.initial_failure()
        out = self.root / 'live'
        c.prepare_output(out, self.source, self.original_identity)
        cp = self.recovered_cp(row, out, lambda **kw: self.fail('wrong boundary must stop call'))
        with self.assertRaisesRegex(RuntimeError, 'preceded the frozen offline probe'):
            cp.request('unrelated-new-work', {'model': 'critic'}, 'critic')
        wrong = c.payload_for(SUPER, 'changed rubric', 'fixed system', 2048, repair=True)
        with self.assertRaisesRegex(RuntimeError, 'differs from frozen offline probe'):
            cp.request(LABEL + '-missing-r2-b01', wrong, ROLE + '-repair')

    def test_original_transport_resume_keeps_2048_payload(self):
        payload = c.payload_for(SUPER, 'same original request', 'fixed', 2048)
        raw = self.source / 'provider'
        c.write_once(raw / (LABEL + '-request.json'), {'provider': 'nvidia', 'role': ROLE, 'payload': payload})
        c.write_once(raw / (LABEL + '-attempts.json'), [{'status': 502}])
        row = {'original_files': r.tree(self.source), 'original_provider_identity_sha256': c.sha256_file(self.source / 'provider-identity.json'),
            'original_request_labels': [LABEL], 'original_logical_labels': [LABEL], 'original_logical_count': 1,
            'original_logical_limit': 3, 'effective_logical_limit': 4,
            'next_unarchived_request': {'logical_label': LABEL, 'payload_sha256': c.sha256_value(payload), 'role': ROLE}}
        out = self.root / 'live'
        c.prepare_output(out, self.source, self.original_identity)
        seen = []
        def answer(kw):
            seen.append(kw['payload']['max_tokens'])
            return '{}'
        cp = self.recovered_cp(row, out, archived_api(answer))
        _, actual = cp.request(LABEL, payload, ROLE)
        self.assertEqual(actual, LABEL + '-transport-resume-01')
        self.assertEqual(seen, [2048])
        self.assertEqual(len(cp.logical_labels), 1)
        cp.request('later-required-phase', {'model': 'critic', 'max_tokens': 4096}, 'critic')
        self.assertEqual(len(cp.logical_labels), 2)

    def test_actual_ecology_driver_wrapper_preserves_identity_and_replays_amendment(self):
        campaign = copy.deepcopy(CAMPAIGN)
        campaign['nvidia']['models'] = {'critic': {'id': 'critic'}, 'adjudicator': SUPER}
        c.write_once(self.root / 'campaign.json', campaign)
        c.write_once(self.root / 'release/adapter-release.json', {'frozen': True})
        config = self.root / 'config.json'
        c.write_once(config, {'campaign_config': 'campaign.json', 'preference_evidence_root': 'release'})
        session = {'experiment': 'VILLAGE-001E', 'stage': 'ecology-session', 'session_id': 'PAIR',
            'kind': 'same_lineage_pair', 'scenario': 'Specific situation.', 'rounds': 3, 'public_handles': ['M1', 'M2'],
            'handle_member_mapping': {'M1': 'V-G01-A', 'M2': 'V-G01-B'}, 'turns': []}
        c.write_once(self.root / 'generation/ecology/PAIR/session.json', session)
        bolt = {'metrics': {key: 1 for key in ecology.METRICS},
            'agents': [{'handle': h, 'distinctiveness': 1, 'generic_assistant_tendency': 1, 'dominance': 1, 'deference': 1,
                'behavior_summary': 'Concrete observed strategy.', 'reason': 'Specific observation.'} for h in ['M1', 'M2']],
            'overall_reason': 'Limited differentiation observed.'}
        def initial_answer(kw):
            if kw['role'] == 'ecology-critic':
                return json.dumps(bolt)
            if kw['role'] == 'ecology-adjudicator':
                return '{"distinct_behavior_survives":false,"generic_convergence_concerning":true,"reason":"bad "quote""}'
            return ''
        source = self.root / 'driver-source'
        with patch.object(ecology, 'ROOT', self.root), patch.object(r, 'ROOT', self.root), patch.object(ecology, 'validate_session_source'), patch.object(ecology, 'packet_for', return_value={'frozen': 'lineage'}), patch.dict(os.environ, {'NVIDIA_API_KEY': 'fixture', 'GITHUB_RUN_ATTEMPT': '1'}):
            argv = ['original', '--config', str(config), '--session', str(self.root / 'generation/ecology/PAIR'), '--out', str(source)]
            with patch.object(sys, 'argv', argv), patch.object(ecology, 'call_api', side_effect=archived_api(initial_answer)):
                with self.assertRaisesRegex(RuntimeError, 'whole-lane provider budget exhausted'):
                    ecology.main()
            state = r.probe(config, 'ecology', 'PAIR', source)
            self.assertEqual(state['next_unarchived_request']['max_tokens'], 8192)
            derived = self.root / 'proven-zero-provider-source'
            with patch.object(ecology, 'call_api', side_effect=AssertionError('zero-call derivation invoked provider')):
                start = r.probe(config, 'ecology', 'PAIR', None, materialize=derived)
            self.assertEqual(start['next_unarchived_request']['logical_label'], 'PAIR-ecology-bolt')
            self.assertEqual(set(r.tree(derived)), {'provider-identity.json'})
            self.assertEqual(r.labels(derived), (set(), set()))
            self.assertEqual((derived / 'provider-identity.json').read_bytes(), (source / 'provider-identity.json').read_bytes())
            actual, logical = r.labels(source)
            row = {'kind': 'ecology', 'lane': 'PAIR', 'terminal': 'judgment.json', **state,
                'original_files': r.tree(source), 'original_provider_identity_sha256': c.sha256_file(source / 'provider-identity.json'),
                'original_request_labels': sorted(actual), 'original_logical_labels': sorted(logical), 'original_logical_count': 3,
                'original_logical_limit': 3, 'effective_logical_limit': 6, 'supplemental_logical_reservation': 3,
                'artifact': {'digest': 'sha256:' + 'a' * 64}}
            manifest = {'super_model': SUPER['id'], 'budget': {'maximum_total_logical_requests': 6}}
            manifest_path = self.root / 'manifest.json'
            r.write_once(manifest_path, manifest)
            def recovered_answer(kw):
                self.assertEqual(kw['payload']['max_tokens'], 8192)
                return '{"reason":"Concrete completed explanation."}'
            live, replay = self.root / 'driver-recovered', self.root / 'driver-replay'
            with patch.object(ecology, 'call_api', side_effect=archived_api(recovered_answer)):
                r.run_lane(manifest_path, manifest, config, row, source, live, offline=False, quiet=True)
            with patch.object(ecology, 'call_api', side_effect=AssertionError('offline network forbidden')):
                r.run_lane(manifest_path, manifest, config, row, live, replay, offline=True, quiet=True)
            judgment = r.read(live / 'judgment.json')
            self.assertEqual(judgment['metrics'], bolt['metrics'])
            self.assertEqual(judgment['super_audit'], {'distinct_behavior_survives': False, 'generic_convergence_concerning': True, 'reason': 'Concrete completed explanation.'})
            self.assertEqual((source / 'provider-identity.json').read_bytes(), (live / 'provider-identity.json').read_bytes())
            self.assertEqual((live / 'judgment.json').read_bytes(), (replay / 'judgment.json').read_bytes())
            self.assertEqual(r.read(live / 'budget-recovery-execution-001.json')['new_logical_requests'], 1)
            self.assertEqual(r.read(replay / 'budget-recovery-execution-002.json')['new_provider_calls'], 0)

    def test_retired_allocations_keep_global_ceiling_and_reject_overbooking(self):
        lanes = []
        for base in r.expected_lanes(r.DEFAULT_CONFIG):
            lanes.append({**base, 'status': 'completed', 'original_logical_count': base['original_logical_limit'],
                          'supplemental_logical_reservation': 0, 'effective_logical_limit': base['original_logical_limit']})
        lanes[0].update(status='pending', supplemental_logical_reservation=3, effective_logical_limit=lanes[0]['original_logical_count'] + 3)
        with self.assertRaisesRegex(RuntimeError, 'exceed original global'):
            r.allocation(lanes)
        lanes[-1]['original_logical_count'] -= 3
        lanes[-1]['effective_logical_limit'] -= 3
        budget = r.allocation(lanes)
        self.assertEqual(budget['maximum_total_logical_requests'], 1104)
        self.assertEqual(budget['unused_original_allocations_retired'], 3)
        self.assertEqual(budget['supplemental_logical_requests_reserved'], 3)

    def test_started_and_unstarted_tournament_caps_depend_only_on_execution_state(self):
        rows = []
        for base in r.expected_lanes(r.DEFAULT_CONFIG):
            if base['kind'] == 'ecology':
                row = {**base, 'status': 'completed', 'original_logical_count': 1}
            elif base['lane'] in ('F01', 'F02'):
                row = {**base, 'status': 'pending', 'original_logical_count': 42}
            elif base['lane'] == 'F03':
                row = {**base, 'status': 'pending', 'original_logical_count': 2}
            else:
                row = {**base, 'status': 'unstarted', 'original_logical_count': 0}
            row['effective_logical_limit'] = r.effective_limit(row)
            row['supplemental_logical_reservation'] = row['effective_logical_limit'] - row['original_logical_count']
            rows.append(row)
        budget = r.allocation(rows)
        self.assertEqual(budget['provider_unstarted_lanes'], 21)
        self.assertEqual(budget['maximum_total_logical_requests'], 24 + 3 * 60 + 21 * 40)
        self.assertEqual({x['effective_logical_limit'] for x in rows if x['status'] == 'pending'}, {60})
        self.assertEqual({x['effective_logical_limit'] for x in rows if x['status'] == 'unstarted'}, {40})

    def test_manifest_refuses_in_progress_original_run_before_artifact_work(self):
        run, inventory = self.root / 'run.json', self.root / 'inventory.json'
        r.write_once(run, {'status': 'in_progress', 'path': r.ORIGINAL_WORKFLOW, 'head_branch': r.ORIGINAL_BRANCH})
        r.write_once(inventory, {'artifacts': []})
        args = SimpleNamespace(config=r.DEFAULT_CONFIG, run_record=run, artifact_inventory=inventory,
                               archives=self.root / 'absent-zips', artifacts=self.root / 'absent-artifacts')
        with self.assertRaisesRegex(RuntimeError, 'original run must be completed'):
            r.make_manifest(args)


if __name__ == '__main__':
    unittest.main()
