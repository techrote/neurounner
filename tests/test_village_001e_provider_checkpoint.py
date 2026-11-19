import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import village_001e_provider_checkpoint as c
import village_001e_tournament_scenario as t
import village_001e_ecology_judge as e


CAMPAIGN = {'nvidia': {'endpoint': 'https://invalid.example', 'generation': {'attempts': 6}, 'models': {'compiler': {'id': 'normalizer'}, 'critic': {'id': 'critic'}, 'adjudicator': {'id': 'auditor'}}}}


def archived_api(answer):
    def api(**kw):
        raw = answer(kw) if callable(answer) else answer
        out, label = kw['outdir'], kw['label']
        out.mkdir(exist_ok=True)
        c.write_once(out / f'{label}-request.json', {'provider': 'nvidia', 'role': kw['role'], 'payload': kw['payload']})
        c.write_once(out / f'{label}-provider.json', {'choices': [{'message': {'content': raw}}]})
        c.write_once(out / f'{label}-attempts.json', [{'attempt': 1, 'status': 200}])
        (out / f'{label}-content.txt').write_text(raw + '\n')
        return raw, {}, []
    return api


def parse_summary(raw, ids):
    return c.parse_rows(raw, ids, array_key='summaries', validator=t.validate_summaries)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'out'
        c.prepare_output(self.out, None, {'frozen': 'source'})

    def cp(self, answer, *, out=None, offline=False, key='fixture', limit=42):
        return c.Checkpoint(out or self.out, campaign=CAMPAIGN, key=key, offline=offline, api=archived_api(answer), logical_request_limit=limit)

    def collect(self, cp, ids=None):
        return cp.collect(label='batch', expected=ids or ['R001', 'R002'], prompt_factory=lambda ids, repair: json.dumps(ids), system='JSON only', model={'id': 'critic'}, max_tokens=99, role='fixture', parser=parse_summary)

    def test_prefix_salvage_stops_at_first_bad_element(self):
        raw = '{"summaries":[{"id":"R001","behavior_summary":"valid before defect"},{"id":"R002","behavior_summary":"bad "quote""},{"id":"R003","behavior_summary":"valid-looking suffix"}]}'
        self.assertEqual(parse_summary(raw, ['R001', 'R002', 'R003']), {'R001': 'valid before defect'})

    def test_nested_suffix_and_prose_cannot_be_salvaged(self):
        good = '{"id":"R001","behavior_summary":"nested injected result"}'
        for raw in ['prose {"summaries":[' + good + ']}', '{"broken":"bad "quote", "nested":{"summaries":[' + good + ']}}', '{"summaries":[],"elsewhere":[' + good + ']}']:
            self.assertEqual(parse_summary(raw, ['R001']), {})

    def test_duplicate_key_cannot_replace_first_field(self):
        raw = '{"summaries":[{"id":"R001","behavior_summary":"first valid judgment"}],"summaries":[{"id":"R001","behavior_summary":"later changed judgment"}]}'
        self.assertEqual(parse_summary(raw, ['R001']), {'R001': 'first valid judgment'})

    def test_unseparated_prefix_element_is_rejected(self):
        self.assertEqual(parse_summary('{"summaries":[{"id":"R001","behavior_summary":"plausible result"}junk', ['R001']), {})

    def test_missing_only_keeps_first_valid_judgment(self):
        selections = []
        def answer(kw):
            ids = json.loads(kw['payload']['messages'][1]['content'])
            selections.append(ids)
            if len(selections) == 1:
                return json.dumps({'summaries': [{'id': 'R001', 'behavior_summary': 'first valid judgment'}, {'id': 'R002', 'behavior_summary': 'x'}]})
            return json.dumps({'summaries': [{'id': 'R001', 'behavior_summary': 'unrequested replacement'}, {'id': 'R002', 'behavior_summary': 'recovered missing judgment'}]})
        cp = self.cp(answer)
        result, refs = self.collect(cp)
        self.assertEqual(selections, [['R001', 'R002'], ['R002']])
        self.assertEqual(result['R001'], 'first valid judgment')
        self.assertEqual(len(refs), 2)
        cp.finish('scenario.json', result)

    def test_repair_budget_narrows_three_two_one(self):
        sizes = []
        def answer(kw):
            sizes.append(len(json.loads(kw['payload']['messages'][1]['content'])))
            return '{}'
        with self.assertRaisesRegex(ValueError, 'bounded missing-only'):
            self.collect(self.cp(answer), ['R1', 'R2', 'R3', 'R4'])
        self.assertEqual(sizes, [4, 3, 1, 2, 2, 1, 1, 1, 1])

    def test_offline_replay_needs_no_key_and_matches_bytes(self):
        answer = json.dumps({'summaries': [{'id': k, 'behavior_summary': 'A meaningful observation.'} for k in ['R001', 'R002']]})
        cp = self.cp(answer)
        result, _ = self.collect(cp)
        cp.finish('scenario.json', result)
        original = (self.out / 'scenario.json').read_bytes()
        replay = self.root / 'replay'
        c.prepare_output(replay, self.out, {'frozen': 'source'})
        cp2 = self.cp(lambda _: self.fail('offline provider called'), out=replay, offline=True, key='')
        result2, _ = self.collect(cp2)
        cp2.finish('scenario.json', result2)
        self.assertEqual((replay / 'scenario.json').read_bytes(), original)
        self.assertEqual(cp2.new_calls, 0)
        self.assertEqual(cp2.replayed_calls, 1)
        self.assertEqual((replay / 'provider-evidence-hashes.json').read_bytes(), (self.out / 'provider-evidence-hashes.json').read_bytes())

    def test_offline_incomplete_does_not_call_provider(self):
        with self.assertRaisesRegex(RuntimeError, 'fresh provider work required'):
            self.collect(self.cp(lambda _: self.fail('offline API called'), offline=True))

    def test_request_payload_drift_rejected(self):
        cp = self.cp('{}')
        cp.request('request', {'model': 'one'}, 'role')
        with self.assertRaisesRegex(RuntimeError, 'request/source mismatch'):
            cp.request('request', {'model': 'two'}, 'role')

    def test_content_tamper_rejected(self):
        cp = self.cp('{}')
        cp.request('request', {'model': 'one'}, 'role')
        (self.out / 'provider/request-content.txt').write_text('tampered\n')
        with self.assertRaisesRegex(RuntimeError, 'content/provider mismatch'):
            cp.request('request', {'model': 'one'}, 'role')

    def test_terminal_tamper_rejected(self):
        cp = self.cp('{}')
        c.write_once(self.out / 'scenario.json', {'wrong': True})
        with self.assertRaisesRegex(RuntimeError, 'immutable checkpoint mismatch'):
            cp.finish('scenario.json', {'right': True})

    def test_source_identity_drift_rejected_before_copy(self):
        target = self.root / 'bad'
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            c.prepare_output(target, self.out, {'frozen': 'different'})
        self.assertFalse(target.exists())

    def test_symlink_resume_rejected(self):
        (self.out / 'link').symlink_to('/tmp')
        with self.assertRaisesRegex(RuntimeError, 'unsafe resume'):
            c.prepare_output(self.root / 'bad', self.out, {'frozen': 'source'})

    def test_nonempty_output_is_never_deleted(self):
        (self.out / 'preserved').write_text('keep')
        with self.assertRaisesRegex(RuntimeError, 'output must be empty'):
            c.prepare_output(self.out, None, {'frozen': 'source'})
        self.assertEqual((self.out / 'preserved').read_text(), 'keep')

    def test_transport_resume_uses_new_filename(self):
        payload = {'model': 'one'}
        raw = self.out / 'provider'
        raw.mkdir()
        c.write_once(raw / 'request-request.json', {'provider': 'nvidia', 'role': 'role', 'payload': payload})
        c.write_once(raw / 'request-attempts.json', [{'status': 502}])
        before = (raw / 'request-request.json').read_bytes()
        content, label = self.cp('{}').request('request', payload, 'role')
        self.assertEqual(label, 'request-transport-resume-01')
        self.assertEqual((raw / 'request-request.json').read_bytes(), before)

    def test_persisted_budget_exhausted_across_resume(self):
        cp = self.cp('{}', limit=2)
        cp.request('one', {'model': 'fixed'}, 'role')
        cp.request('two', {'model': 'fixed'}, 'role')
        cp.finish('scenario.json', {})
        before = {p.relative_to(self.out).as_posix(): p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        resumed = self.root / 'resumed'
        c.prepare_output(resumed, self.out, {'frozen': 'source'})
        cp2 = self.cp(lambda _: self.fail('budget must prevent new API work'), out=resumed, limit=2)
        cp2.request('one', {'model': 'fixed'}, 'role')
        cp2.request('two', {'model': 'fixed'}, 'role')
        with self.assertRaisesRegex(RuntimeError, 'whole-lane provider budget exhausted'):
            cp2.request('three', {'model': 'fixed'}, 'role')
        after = {p.relative_to(resumed).as_posix(): p.read_bytes() for p in resumed.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(cp2.new_calls, 0)

    def test_final_repair_at_exact_budget_boundary_succeeds(self):
        def answer(kw):
            ids = json.loads(kw['payload']['messages'][1]['content'])
            if kw['label'] == 'batch':
                ids = ['R001']
            return json.dumps({'summaries': [{'id': ident, 'behavior_summary': 'Valid fixed judgment.'} for ident in ids]})
        cp = self.cp(answer, limit=2)
        result, refs = self.collect(cp)
        cp.finish('scenario.json', result)
        self.assertEqual(set(result), {'R001', 'R002'})
        self.assertEqual(len(refs), 2)
        self.assertEqual(len(cp.logical_labels), 2)
        self.assertEqual(cp.new_calls, 2)

    def test_transport_resume_does_not_reset_or_double_count_logical_budget(self):
        raw = self.out / 'provider'
        raw.mkdir()
        payload = {'model': 'fixed'}
        c.write_once(raw / 'one-request.json', {'provider': 'nvidia', 'role': 'role', 'payload': payload})
        c.write_once(raw / 'one-attempts.json', [{'status': 502}])
        cp = self.cp('{}', limit=1)
        _, actual = cp.request('one', payload, 'role')
        self.assertEqual(actual, 'one-transport-resume-01')
        self.assertEqual(cp.logical_labels, {'one'})
        with self.assertRaisesRegex(RuntimeError, 'whole-lane provider budget exhausted'):
            cp.request('two', payload, 'role')


    def test_unconsumed_history_rejected(self):
        cp = self.cp('{}')
        c.write_once(cp.rawdir / 'unexpected-request.json', {})
        with self.assertRaisesRegex(RuntimeError, 'unconsumed provider history'):
            cp.finish('scenario.json', {})


class EcologyPartsTests(unittest.TestCase):
    def test_public_session_does_not_leak_internal_lineage_identifier(self):
        session = {'session_id': 'E-P-G01', 'kind': 'same_lineage_pair', 'scenario': {'id': 'repair', 'text': 'Repair the sorter.'}, 'rounds': 1, 'public_handles': ['M1'], 'handle_member_mapping': {'M1': 'V-G01-A'}, 'turns': [{'round': 1, 'handle': 'M1', 'member_id': 'V-G01-A', 'text': 'Inspect the cable.'}]}
        public = e.public_session(session)
        self.assertNotIn('session_id', public)
        self.assertNotIn('G01', json.dumps(public))
        self.assertNotIn('member_id', public['turns'][0])

    def test_partial_agents_and_metrics_retained(self):
        agent = {'handle': 'M1', 'distinctiveness': 2, 'generic_assistant_tendency': 1, 'dominance': 3, 'deference': 0, 'behavior_summary': 'Specific behavior.', 'reason': 'Concrete observation.'}
        obj = {'metrics': {k: 2 for k in e.METRICS}, 'agents': [agent], 'overall_reason': 'x'}
        parts = e.primary_parts(json.dumps(obj), ['metrics', 'agent:M1', 'agent:M2', 'overall_reason'], ['M1', 'M2'])
        self.assertEqual(set(parts), {'metrics', 'agent:M1'})

    def test_lineage_contamination_keeps_other_handle_valid(self):
        row = {'handle': 'M1', 'expected_lineage_fit': 1, 'cross_lineage_contamination': 3, 'distinction_survival': 0, 'contaminating_handle': 'M2', 'reason': 'Copied the other strategy.'}
        parts = e.lineage_parts(json.dumps({'agents': [row]}), ['agent:M1'], ['M1', 'M2'])
        self.assertEqual(parts['agent:M1']['contaminating_handle'], 'M2')
        row['contaminating_handle'] = 'unknown'
        self.assertEqual(e.lineage_parts(json.dumps({'agents': [row]}), ['agent:M1'], ['M1', 'M2']), {})

    def test_super_only_requested_fields_and_real_booleans(self):
        self.assertEqual(e.super_parts('{"distinct_behavior_survives":1,"generic_convergence_concerning":false,"reason":"Valid explanation."}', ['distinct_behavior_survives', 'reason']), {'reason': 'Valid explanation.'})


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'release').mkdir()
        c.write_once(self.root / 'campaign.json', CAMPAIGN)
        c.write_once(self.root / 'release/adapter-release.json', {'release_tag': 'frozen-test'})
        self.cfg = {'campaign_config': 'campaign.json', 'preference_evidence_root': 'release', 'final_eval_config': 'final.json', 'tournament': {'scenario_count': 1, 'normalizer_chunk_size': 12, 'classifier_chunk_size': 12}}
        c.write_once(self.root / 'config.json', self.cfg)
        c.write_once(self.root / 'final.json', {'scenarios': [{'id': 'F01'}]})

    def test_full_tournament_driver_then_offline_replay(self):
        scenario = {'scenario_id': 'F01', 'prompt': 'Test situation.', 'family': 'test', 'final_eval_config_sha256': c.sha256_file(self.root / 'final.json')}
        members = [{'member_id': f'V-G{(i-1)//3+1:02d}-{"ABC"[(i-1)%3]}', 'lineage': f'G{(i-1)//3+1:02d}', 'sibling': 'ABC'[(i-1)%3], 'response': 'Local response.', 'opaque_id': f'R{i:03d}', 'adapter_sha256': 'frozen'} for i in range(1, 37)]
        control = {'opaque_id': 'R037', 'response': 'Base response.', 'base_control': True}
        def answer(kw):
            prompt = kw['payload']['messages'][1]['content']
            items = json.loads(prompt.rsplit('RESPONSES:\n' if 'RESPONSES:\n' in prompt else 'ITEMS:\n', 1)[1])
            role = kw['role']
            if 'normalizer' in role:
                return json.dumps({'summaries': [{'id': x['id'], 'behavior_summary': 'Neutral behavior observed.'} for x in items]})
            if 'sibling-classifier' in role:
                return json.dumps({'classifications': [{'id': x['id'], 'predicted_sibling': 'A', 'sibling_confidence': 3, 'reason': 'Specific choice.'} for x in items]})
            return json.dumps({'classifications': [{'id': x['id'], 'predicted_lineage': 'G01', 'predicted_sibling': 'A', 'lineage_confidence': 3, 'sibling_confidence': 3, 'task_relevance': 2, 'genericness': 1, 'reason': 'Specific choice.'} for x in items]})
        baseargs = ['driver', '--config', str(self.root / 'config.json'), '--scenario', 'F01', '--responses-root', str(self.root)]
        with patch.object(t, 'ROOT', self.root), patch.object(t, 'load_cases', side_effect=lambda *args: copy.deepcopy((scenario, members, control))), patch.object(t, 'lineage_catalog', return_value={f'G{i:02d}': 'fixture' for i in range(1, 13)}), patch.object(t, 'source_response_hashes', return_value={m['member_id']: 'frozen' for m in members} | {'BASE': 'frozen'}), patch.object(t, 'call_api', side_effect=archived_api(answer)), patch.dict(os.environ, {'NVIDIA_API_KEY': 'fixture'}):
            with patch.object(sys, 'argv', baseargs + ['--out', str(self.root / 'live')]):
                t.main()
            with patch.object(sys, 'argv', baseargs + ['--resume', str(self.root / 'live'), '--out', str(self.root / 'replay'), '--offline']), patch.object(t, 'call_api', side_effect=AssertionError('offline API')):
                t.main()
        first = (self.root / 'live/scenario.json').read_bytes()
        self.assertEqual(first, (self.root / 'replay/scenario.json').read_bytes())
        self.assertEqual(len(json.loads(first)['provider_call_ids']), 18)

    def test_truncated_ecology_source_fails_before_any_provider_work(self):
        config_path = e.ROOT / 'experiments/village-001/tournament-config.json'
        cfg = json.loads(config_path.read_text())
        campaign = json.loads((e.ROOT / cfg['campaign_config']).read_text())
        release = json.loads((e.ROOT / cfg['preference_evidence_root'] / 'adapter-release.json').read_text())
        assets = {x['member_id']: x for x in release['assets']}
        plan = e.build_sessions(cfg)[0]
        session = {'experiment': 'VILLAGE-001E', 'stage': 'ecology-session', 'session_id': plan['session_id'], 'kind': plan['kind'], 'scenario': plan['scenario'], 'members': plan['members'], 'rounds': cfg['ecology']['rounds'], 'public_handles': sorted(plan['handles']), 'handle_member_mapping': plan['handles'], 'base_model': campaign['base_model']['id'], 'base_revision': campaign['base_model']['revision'], 'adapter_sha256': {m: assets[m]['adapter_directory_sha256'] for m in plan['members']}, 'turns': []}
        path = self.root / 'truncated-session.json'
        c.write_once(path, session)
        with patch.object(sys, 'argv', ['driver', '--config', str(config_path), '--session', str(path), '--out', str(self.root / 'blocked')]), patch.object(e, 'call_api', side_effect=AssertionError('provider must not run')):
            with self.assertRaisesRegex(RuntimeError, 'turn count mismatch'):
                e.main()
        self.assertFalse((self.root / 'blocked').exists())

    def test_ecology_driver_then_offline_replay(self):
        session = {'experiment': 'VILLAGE-001E', 'stage': 'ecology-session', 'session_id': 'PAIR01', 'kind': 'sibling_pair', 'scenario': 'Test situation.', 'rounds': 3, 'public_handles': ['M1', 'M2'], 'handle_member_mapping': {'M1': 'V-G01-A', 'M2': 'V-G01-B'}, 'turns': []}
        c.write_once(self.root / 'session.json', session)
        obj = {'metrics': {k: 2 for k in e.METRICS}, 'agents': [{'handle': h, 'distinctiveness': 2, 'generic_assistant_tendency': 1, 'dominance': 2, 'deference': 1, 'behavior_summary': 'Specific behavior.', 'reason': 'Concrete evidence.'} for h in ['M1', 'M2']], 'overall_reason': 'Distinct approaches observed.'}
        baseargs = ['driver', '--config', str(self.root / 'config.json'), '--session', str(self.root / 'session.json')]
        with patch.object(e, 'ROOT', self.root), patch.object(e, 'validate_session_source'), patch.object(e, 'packet_for', return_value={'fixture': 'lineage'}), patch.object(e, 'call_api', side_effect=archived_api(json.dumps(obj))), patch.dict(os.environ, {'NVIDIA_API_KEY': 'fixture'}):
            with patch.object(sys, 'argv', baseargs + ['--out', str(self.root / 'live')]):
                e.main()
            with patch.object(sys, 'argv', baseargs + ['--resume', str(self.root / 'live'), '--out', str(self.root / 'replay'), '--offline']), patch.object(e, 'call_api', side_effect=AssertionError('offline API')):
                e.main()
        self.assertEqual((self.root / 'live/judgment.json').read_bytes(), (self.root / 'replay/judgment.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
