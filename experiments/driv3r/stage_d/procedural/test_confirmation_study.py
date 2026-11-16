"""Offline D-2 source-pinned confirmation study and receipt-negative tests."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from confirmation_fixtures import CASE_SEEDS, fixture
from confirmation_study import (FROZEN_COMMIT, FROZEN_MANIFEST_SHA256,
                                audit_events, permitted_reachability, run, verify_frozen_input)
import pareto_study as p


class ConfirmationStudyTests(unittest.TestCase):
    def test_freeze_matches_on_disk(self):
        frozen=verify_frozen_input(Path(__file__).resolve().parent)
        self.assertEqual(len(frozen['cases']),8)
        self.assertEqual(len(FROZEN_COMMIT),40)
        self.assertEqual(len(FROZEN_MANIFEST_SHA256),64)

    def test_tampered_or_resorted_prereg_refused(self):
        src=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'corpus').mkdir()
            shutil.copy(src/'confirmation_fixtures.py',root/'confirmation_fixtures.py')
            original=(src/'corpus'/'confirmation-prereg-v1.json').read_bytes()
            (root/'corpus'/'confirmation-prereg-v1.json').write_bytes(original.replace(b'1401',b'1409'))
            with self.assertRaises(ValueError):verify_frozen_input(root)

    def test_reachable_is_not_equivalent_to_budget_feasible(self):
        allowed,initial,_=fixture('staggered-shelves',1404)
        self.assertTrue(permitted_reachability(allowed,initial)['all_initial_islands_globally_reachable'])
        allowed,initial,_=fixture('hard-separator',1408)
        self.assertFalse(permitted_reachability(allowed,initial)['all_initial_islands_globally_reachable'])

    def test_route_paint_event_reconstruction_and_tamper(self):
        allowed,initial,scalar=fixture('nested-courtyard',1402)
        r,after=p.run_bridge(allowed,initial,scalar,44,0,False,120)
        audit_events(initial,allowed,r,after)
        event=next(e for e in r['events'] if e['kind']=='SYNTHETIC_BRIDGE_PAINT')
        event['after']='0'*64
        with self.assertRaises(AssertionError):audit_events(initial,allowed,r,after)

    def test_full_cohort_repeats_byte_identically_and_no_fake_native(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            first,sha1=run(Path(a))
            second,sha2=run(Path(b))
            self.assertEqual((sha1,first),(sha2,second))
            self.assertEqual(len(first['cases']),len(CASE_SEEDS))
            self.assertTrue(first['all_registered_cases_included'])
            self.assertFalse(first['native_executed'])
            self.assertEqual(first['native_receipts'],0)
            self.assertFalse(first['model_inference'])
            self.assertEqual(first['visual_acceptance'],'NOT_PERFORMED')
            for case in first['cases']:
                a_case,b_case=Path(a)/case['id'],Path(b)/case['id']
                for file in a_case.iterdir():
                    self.assertEqual(file.read_bytes(),(b_case/file.name).read_bytes())
                for key,data in case['policies'].items():
                    self.assertEqual(data['new_cells'],data['matched_control']['new_cells'])
                    self.assertLessEqual(data['new_cells'],44)
                    self.assertEqual(data['native_receipts'],0)
            self.assertGreater(first['cases'][0]['policies']['weighted-0']['new_cells'],0)
            self.assertTrue(all(v['components']>1 for v in first['cases'][-1]['policies'].values()))


    def test_synthetic_cursor_teleport_is_not_hidden_by_unchanged_material_hashes(self):
        allowed, initial, scalar = fixture('nested-courtyard', 1402)
        report, after = p.run_bridge(allowed, initial, scalar, 44, 0, False, 120)
        audit_events(initial, allowed, report, after, scalar)
        move = next(e for e in report['events'] if e['kind'] == 'BRUSH_UP_ROUTE')
        move['from'] = [0, 0]
        report['events_sha256'] = p.sha(p.encoded(report['events']))
        with self.assertRaisesRegex(AssertionError, 'cursor teleport'):
            audit_events(initial, allowed, report, after, scalar)

    def test_unexecuted_plan_and_corrupt_scalar_labels_refused(self):
        allowed, initial, scalar = fixture('nested-courtyard', 1402)
        report, after = p.run_bridge(allowed, initial, scalar, 44, 0, False, 120)
        painted = next(e for e in report['events'] if e['kind'] == 'SYNTHETIC_BRIDGE_PAINT')
        painted['below_soft_threshold'] = not painted['below_soft_threshold']
        report['events_sha256'] = p.sha(p.encoded(report['events']))
        with self.assertRaisesRegex(AssertionError, 'guidance threshold'):
            audit_events(initial, allowed, report, after, scalar)
        fresh, after = p.run_bridge(allowed, initial, scalar, 44, 0, False, 120)
        fresh['events'] = fresh['events'][:-1]
        fresh['events_sha256'] = p.sha(p.encoded(fresh['events']))
        with self.assertRaises(AssertionError):
            audit_events(initial, allowed, fresh, after, scalar)

    def test_partial_selection_is_explicitly_not_full_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = run(Path(tmp), names=('nested-courtyard',))
            self.assertEqual(result['qualification'], 'PRE_REGISTERED_PARTIAL_SYNTHETIC_DIAGNOSTIC')
            self.assertEqual(result['selection_scope'], 'PARTIAL_SUBSET_DIAGNOSTIC')
            self.assertTrue(result['policy_source_verified'])
            self.assertFalse(result['complete_source_qualified_cohort'])
            self.assertFalse(result['all_registered_cases_included'])
            self.assertEqual(result['included_case_ids'], ['nested-courtyard'])
            self.assertEqual(len(result['omitted_case_ids']), len(CASE_SEEDS)-1)
            self.assertEqual([c['id'] for c in result['cases']], ['nested-courtyard'])

    def test_reject_ambiguous_or_unregistered_case_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            for selection in ((), ('nested-courtyard', 'nested-courtyard'),
                              ('does-not-exist',)):
                with self.subTest(selection=selection):
                    with self.assertRaises(ValueError):
                        run(Path(tmp), names=selection)

    def test_complete_cohort_keeps_full_preregistration_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = run(Path(tmp), names=(name for name, _ in CASE_SEEDS))
            self.assertEqual(result['qualification'], 'PRE_REGISTERED_SYNTHETIC_OFFLINE_ONLY')
            self.assertTrue(result['complete_source_qualified_cohort'])
            self.assertEqual(result['selection_scope'], 'FULL_PREREGISTERED_COHORT')
            self.assertEqual(result['omitted_case_ids'], [])

    def test_exact_policy_required_unless_provisional_shadow_explicit(self):
        import confirmation_study as study
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(study, 'POLICY_GIT_BLOB', '0' * 40):
                with self.assertRaisesRegex(ValueError, 'unqualified policy source'):
                    study.run(Path(tmp))

if __name__ == '__main__':unittest.main()
