import copy
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import village_001f_sprocket_diagnostic as d


class SprocketDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg, self.plan = d.build_plan(d.DEFAULT_CONFIG)

    def response_fixture(self, request, raw=None):
        raw = raw if raw is not None else json.dumps({'keep': ['checking the cable'], 'mutate': ['stop repeating'], 'failure_guess': 'misses the fault', 'genericness': 'low'})
        return {'choices': [{'message': {'content': raw}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2}, '__verbose': {'tokens': [31, 5], 'prompt': '<|user|>native<|assistant|>', 'truncated': False, 'stop_type': 'eos', 'generation_settings': {'seed': request['seed'], 'n_predict': 160, 'temperature': 1.049999952, 'top_p': 0.970000028}}}

    def archive_exchange(self, out, label, endpoint, payload, response, generation=False):
        d.write_json(out / 'http' / f'{label}.request.json', {'endpoint': endpoint, 'payload': payload, 'generation': generation})
        d.write_json(out / 'http' / f'{label}.response.json', response)

    def archive_wire(self, out, req):
        key = req['key']
        messages = req['payload']['messages']
        rendered = '<|system|>' + messages[0]['content'] + '<|end|><|user|>' + messages[1]['content'] + '<|end|><|assistant|>'
        template, tokens, counted = {'prompt': rendered}, {'tokens': [0, 2, 100, 5, 3, 101, 5, 4]}, {'input_tokens': 8}
        self.archive_exchange(out, key + '-template', '/apply-template', req['payload'], template)
        self.archive_exchange(out, key + '-tokenize', '/tokenize', {'content': rendered, 'add_special': True, 'parse_special': True}, tokens)
        self.archive_exchange(out, key + '-count', '/v1/chat/completions/input_tokens', req['payload'], counted)
        wire = d.wire_record(req, template, tokens, counted, 2048, self.cfg['wire']['gguf_chat_template_sha256'])
        d.write_json(out / 'wire' / f'{key}.json', wire)
        return wire

    def test_fixed_plan_has_only_six_tp_cases_and_36_generations(self):
        self.assertEqual(len(self.plan['cases']), 6)
        self.assertEqual(len(self.plan['requests']), 36)
        self.assertTrue(all(r['scenario_id'].startswith(('T', 'P')) for r in self.plan['cases']))
        self.assertFalse(any('final-eval' in key for key in self.plan['sources']))
        self.assertEqual(self.plan['cloud_requests'], 0)
        self.assertEqual(self.plan['optimizer_steps'], 0)
        self.assertEqual(sum(r['arm'] == 'original' for r in self.plan['requests']), 18)

    def test_paired_seeds_and_counterbalanced_arm_order(self):
        pairs = list(zip(self.plan['requests'][::2], self.plan['requests'][1::2]))
        self.assertEqual(sum(a['arm'] == 'original' for a, _ in pairs), 9)
        for a, b in pairs:
            self.assertEqual(a['case_id'], b['case_id'])
            self.assertEqual(a['seed'], b['seed'])
            self.assertEqual(a['payload']['messages'][1], b['payload']['messages'][1])
            for key in ['seed', 'max_tokens', 'temperature', 'top_p', 'stream']:
                self.assertEqual(a['payload'][key], b['payload'][key])
            self.assertNotIn('response_format', a['payload'])
            self.assertNotIn('grammar', a['payload'])
            self.assertNotIn('cache_prompt', a['payload'])

    def test_default_entry_point_is_plan_only(self):
        out = self.root / 'plan'
        with patch.object(sys, 'argv', ['diagnostic', '--out', str(out)]), patch.object(d, 'execute', side_effect=AssertionError('live work forbidden')):
            d.main()
        self.assertTrue((out / 'plan.json').is_file())
        self.assertFalse((out / 'rows').exists())

    def test_final_suite_source_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'final-suite source'):
            d.source_path('experiments/village-001/final-eval.json')

    def test_resume_identity_drift_is_refused(self):
        src = self.root / 'source'
        d.prepare_output(src, None, self.plan)
        altered = copy.deepcopy(self.plan)
        altered['requests'][0]['payload']['temperature'] = 0.1
        with self.assertRaisesRegex(RuntimeError, 'resume plan/source identity'):
            d.prepare_output(self.root / 'out', src, altered)
        self.assertTrue((src / 'plan.json').exists())

    def test_runtime_wire_rejects_lost_system_or_count_drift(self):
        req = self.plan['requests'][0]
        messages = req['payload']['messages']
        rendered = '<|system|>' + messages[0]['content'] + '<|end|><|user|>' + messages[1]['content'] + '<|end|><|assistant|>'
        self.assertEqual(d.validate_wire(req, rendered, {'tokens': [0, 2, 100, 5, 3, 101, 5, 4]}, {'input_tokens': 8}, 2048), [0, 2, 100, 5, 3, 101, 5, 4])
        with self.assertRaisesRegex(RuntimeError, 'lost system or user'):
            d.validate_wire(req, messages[1]['content'], {'tokens': [1]}, {'input_tokens': 1}, 2048)
        with self.assertRaisesRegex(RuntimeError, 'counts disagree'):
            d.validate_wire(req, rendered, {'tokens': [0, 2, 100, 5, 3, 101, 5, 4]}, {'input_tokens': 7}, 2048)
        with self.assertRaisesRegex(RuntimeError, 'exceeds the effective'):
            d.validate_wire(req, rendered, {'tokens': [0, 2, 100, 5, 3, 101, 5, 4]}, {'input_tokens': 8}, 167)

    def test_native_token_and_sampling_custody_is_required(self):
        req = self.plan['requests'][0]
        case = self.plan['cases'][0]
        wire = {'rendered_prompt': 'native', 'input_token_ids': [1, 2, 3]}
        response = self.response_fixture(req)
        result = d.analyze_response(req, case, wire, response, {'tokens': [1, 2, 3]})
        self.assertTrue(result['legacy_schema_valid'])
        self.assertTrue(result['strict_json_schema_valid'])
        self.assertIsNone(result['semantic_grounding_score'])
        self.assertEqual(result['generated_token_ids'], [31, 5])
        with self.assertRaisesRegex(RuntimeError, 'actual native prompt token IDs'):
            d.analyze_response(req, case, wire, response, {'tokens': [1, 2, 4]})
        response['__verbose']['generation_settings']['seed'] += 1
        with self.assertRaisesRegex(RuntimeError, 'native seed/token budget'):
            d.analyze_response(req, case, wire, response, {'tokens': [1, 2, 3]})

    def test_full_provider_output_preserved_even_when_schema_invalid(self):
        req, case = self.plan['requests'][0], self.plan['cases'][0]
        wire = {'rendered_prompt': 'native', 'input_token_ids': [1, 2, 3]}
        raw = '  free-form malfunction\n'
        response = self.response_fixture(req, raw)
        result = d.analyze_response(req, case, wire, response, {'tokens': [1, 2, 3]})
        self.assertEqual(result['raw_response'], raw)
        self.assertFalse(result['legacy_schema_valid'])
        self.assertIsNone(result['parsed'])

    def test_local_generation_is_never_reissued_after_ambiguous_failure(self):
        http = d.LocalHTTP(18950, self.root)
        payload = {'messages': []}
        path = self.root / 'http/case-generate.request.json'
        d.write_json(path, {'endpoint': '/v1/chat/completions', 'payload': payload, 'generation': True})
        before = path.read_bytes()
        with patch.object(d.urllib.request, 'urlopen', side_effect=AssertionError('must not reissue')):
            with self.assertRaisesRegex(RuntimeError, 'outcome is ambiguous'):
                http.request('case-generate', '/v1/chat/completions', payload, generation=True)
        self.assertEqual(path.read_bytes(), before)

    def test_local_archived_response_replayed_without_network(self):
        http = d.LocalHTTP(18950, self.root)
        payload = {'messages': []}
        d.write_json(self.root / 'http/case-generate.request.json', {'endpoint': '/v1/chat/completions', 'payload': payload, 'generation': True})
        d.immutable(self.root / 'http/case-generate.response.json', b'{"fixture":true}')
        with patch.object(d.urllib.request, 'urlopen', side_effect=AssertionError('must not invoke network')):
            self.assertEqual(http.request('case-generate', '/v1/chat/completions', payload, generation=True), {'fixture': True})

    def test_orphan_response_is_not_reissued_and_bad_receipt_is_rejected(self):
        http = d.LocalHTTP(18950, self.root)
        d.immutable(self.root / 'http/case-generate.response.json', b'{"fixture":true}')
        with patch.object(d.urllib.request, 'urlopen', side_effect=AssertionError('must not invoke network')):
            with self.assertRaisesRegex(RuntimeError, 'orphaned local response'):
                http.request('case-generate', '/v1/chat/completions', {}, generation=True)
            d.write_json(self.root / 'http/case-generate.request.json', {'endpoint': '/v1/chat/completions', 'payload': {}, 'generation': True})
            d.write_json(self.root / 'http/case-generate.exchange.json', {'http_status': 200, 'response_sha256': 'incorrect'})
            with self.assertRaisesRegex(RuntimeError, 'receipt/hash mismatch'):
                http.request('case-generate', '/v1/chat/completions', {}, generation=True)

    def test_offline_native_wire_reconstructs_exact_requests_and_rejects_count_drift(self):
        req = self.plan['requests'][0]
        wire = self.archive_wire(self.root, req)
        self.assertEqual(d.reconstruct_wire(self.root, req, self.cfg), wire)
        count_path = self.root / 'http' / f"{req['key']}-count.response.json"
        count_path.write_bytes(d.canonical({'input_tokens': 7}))
        with self.assertRaisesRegex(RuntimeError, 'counts disagree'):
            d.reconstruct_wire(self.root, req, self.cfg)
        count_path.write_bytes(d.canonical({'input_tokens': 8}))
        request_path = self.root / 'http' / f"{req['key']}-tokenize.request.json"
        record = json.loads(request_path.read_text())
        record['payload']['parse_special'] = False
        request_path.write_bytes(d.canonical(record))
        with self.assertRaisesRegex(RuntimeError, 'archived local request differs'):
            d.reconstruct_wire(self.root, req, self.cfg)

    def test_gguf_metadata_read_preserves_exact_template_and_special_tokens(self):
        def string(s):
            raw = s.encode()
            return struct.pack('<Q', len(raw)) + raw
        def kv(key, kind, value):
            return string(key) + struct.pack('<I', kind) + value
        data = b'GGUF' + struct.pack('<IQQ', 3, 0, 3)
        data += kv('general.architecture', 8, string('fixture'))
        data += kv('tokenizer.chat_template', 8, string('EXACT {{ messages }} TEMPLATE'))
        data += kv('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, 2) + string('<|user|>') + string('hello'))
        path = self.root / 'fixture.gguf'
        path.write_bytes(data)
        parsed = d.gguf_metadata(path)
        self.assertEqual(parsed['metadata']['tokenizer.chat_template'], 'EXACT {{ messages }} TEMPLATE')
        self.assertEqual(parsed['control_token_strings_by_id'], {'0': '<|user|>'})
        self.assertEqual(parsed['metadata_end_offset'], len(data))
        path.write_bytes(data[:-4])
        with self.assertRaisesRegex(RuntimeError, 'truncated GGUF'):
            d.gguf_metadata(path)

    def test_completed_synthetic_diagnostic_reconstructs_offline_and_stays_blind(self):
        out = self.root / 'complete'
        d.prepare_output(out, None, self.plan)
        cases = {x['case_id']: x for x in self.plan['cases']}
        rows = []
        for req in self.plan['requests']:
            key = req['key']
            wire = self.archive_wire(out, req)
            response = self.response_fixture(req)
            response['usage']['prompt_tokens'] = 8
            response['__verbose']['prompt'] = '<|endoftext|>' + wire['rendered_prompt']
            tokens = {'tokens': wire['input_token_ids']}
            self.archive_exchange(out, key + '-generate', '/v1/chat/completions', req['payload'], response, generation=True)
            self.archive_exchange(out, key + '-actual-tokenize', '/tokenize', {'content': response['__verbose']['prompt'], 'add_special': False, 'parse_special': True}, tokens)
            row = d.analyze_response(req, cases[req['case_id']], wire, response, tokens)
            d.write_json(out / 'rows' / f'{key}.json', row)
            rows.append(row)
        summary = d.summarize(out, self.plan, rows)
        with patch.object(d, 'verify_runtime', side_effect=AssertionError('offline must not start runtime')), patch.object(d.urllib.request, 'urlopen', side_effect=AssertionError('offline network')):
            self.assertEqual(d.execute(out, self.cfg, self.plan, self.root / 'none', 18950, offline=True), summary)
            self.assertEqual(d.execute(out, self.cfg, self.plan, self.root / 'none', 18950, offline=False), summary)
        review = json.loads((out / 'blinded-grounding-review.json').read_text())
        self.assertTrue(review['ratings_are_unfilled'])
        self.assertEqual(len(review['items']), 36)
        self.assertTrue(all('arm' not in row and 'worker' not in row and 'key' not in row for row in review['items']))
        self.assertTrue(all(row['response_grounding_0_to_2'] is None for row in review['items']))
        path = out / 'http' / f"{self.plan['requests'][0]['key']}-actual-tokenize.request.json"
        record = json.loads(path.read_text())
        record['payload']['content'] = 'substituted prompt'
        path.write_bytes(d.canonical(record))
        with self.assertRaisesRegex(RuntimeError, 'archived local request differs'):
            d.execute(out, self.cfg, self.plan, self.root / 'none', 18950, offline=True)


if __name__ == '__main__':
    unittest.main()
