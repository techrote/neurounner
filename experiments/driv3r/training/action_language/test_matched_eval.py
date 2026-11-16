"""GAL-002 external-synthetic controls and matched-world audit tests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lab import World, TinyPolicy, train
from audit_original import AuditRefusal, audit
from matched_eval import (measure, paired_target, summarise, evaluate_panel,
                          write_run, file_sha)


class MatchedResearchTests(unittest.TestCase):
    def test_single_policy_replay_is_byte_deterministic(self):
        p,u=train('sweep',3)
        w1,a=measure(p,'sweep',97001,'pillars')
        w2,b=measure(p,'sweep',97001,'pillars')
        self.assertEqual(a,b)
        self.assertEqual(w1.trace,w2.trace)
        self.assertEqual(w1.material,w2.material)

    def test_initial_world_identical_across_action_arms(self):
        policy=TinyPolicy()
        _,cell=measure(policy,'cell',97001,'pillars')
        _,sweep=measure(policy,'sweep',97001,'pillars')
        self.assertEqual(cell['initial_material_sha256'],sweep['initial_material_sha256'])
        self.assertNotEqual(cell['lease'],sweep['lease'])

    def test_stop_early_is_terminal_not_fabricated_continuation(self):
        p=TinyPolicy();p.weights[-2]=1000 # stop weight (FEATURES[-2])
        w,a=measure(p,'sweep',97000,'empty')
        self.assertTrue(a['stopped'])
        self.assertEqual(a['decisions'],1)
        self.assertFalse(a['actually_reached_decision']['16'])
        self.assertEqual(a['ink_by_decision_budget']['16'],0)
        self.assertEqual(a['decisions_to_ink_target']['32'],None)
        self.assertEqual(w.trace[0]['action_id'],'STOP')

    def test_no_phantom_target_success_after_early_stop(self):
        r={'seed':1,'family':'empty','host_receipts_sha256':'d',
           'new_ink':0,'decisions':1,'logical_moves':0,'stopped':True,
           'ink_by_decision_budget':{'16':0,'32':0,'64':0},
           'actually_reached_decision':{'16':False,'32':False,'64':False},
           'decisions_to_ink_target':{'32':None,'64':None}}
        s=summarise([r])
        self.assertEqual(s['target_reached_fraction']['32'],0)
        self.assertIsNone(s['mean_calls_if_target_reached']['32'])
        self.assertEqual(s['actually_reached_decision_fraction']['16'],0)

    def test_pairs_account_for_all_failures(self):
        def row(seed,calls):
            return {'seed':seed,'family':'empty',
                    'decisions_to_ink_target':{'32':calls,'64':None}}
        a=[row(1,10),row(2,20),row(3,None),row(4,None)]
        b=[row(1,7),row(2,None),row(3,21),row(4,None)]
        out=paired_target(a,b,32)
        self.assertEqual([out[k] for k in ('both_reached','first_only','second_only','neither_reached')],
                         [1,1,1,1])
        self.assertEqual(out['mean_call_saving_second_vs_first_if_both_reached'],3)

    def test_pairs_reject_nonmatched_worlds(self):
        with self.assertRaises(ValueError):paired_target(
            [{'seed':1,'family':'empty','decisions_to_ink_target':{'32':1}}],
            [{'seed':2,'family':'empty','decisions_to_ink_target':{'32':1}}],32)

    def test_panel_source_and_lease_isolation(self):
        policies={arm:TinyPolicy() for arm in
                  ('cell','sweep','lift','retrace','anchor','jog')}
        rows,stats=evaluate_panel(policies,count=5,gallery_per_arm=0)
        self.assertEqual(len(rows),8)
        self.assertEqual({len(v) for v in rows.values()},{5})
        for i in range(5):
            sources={rr[i]['initial_material_sha256'] for rr in rows.values()}
            leases={rr[i]['lease'] for rr in rows.values()}
            self.assertEqual(len(sources),1)
            # All independent counterfactual worlds carry different leases.
            self.assertEqual(len(leases),8)
        self.assertIn('same_sweep_policy_cell_vs_sweep',stats['paired_target_comparisons'])

    def test_heldout_bounds_and_no_training_overlap(self):
        p={a:TinyPolicy() for a in ('cell','sweep','lift','retrace','anchor','jog')}
        with self.assertRaises(ValueError):evaluate_panel(p,count=4)
        with self.assertRaises(ValueError):evaluate_panel(p,count=5,seed_base=10000)
        with self.assertRaises(ValueError):evaluate_panel(p,count=5,gallery_per_arm=9)

    def test_adversarial_archive_replay_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_run(root,training_worlds=1,heldout_worlds=5,gallery_per_arm=0)
            original=json.loads((root/'matched-traces.json').read_text())
            victim=original['cell_trained'][0][0]
            victim['action_id']='INK_N_4'  # excluded from one-cell catalog
            (root/'matched-traces.json').write_text(json.dumps(original,sort_keys=True)+'\n')
            with self.assertRaises(AuditRefusal):audit(root)
            # A malicious producer can alter summary hash too; replay still refuses.
            metrics=json.loads((root/'metrics.json').read_text())
            metrics['original_heldout_traces_sha256']=file_sha(root/'matched-traces.json')
            (root/'metrics.json').write_text(json.dumps(metrics,sort_keys=True)+'\n')
            with self.assertRaises(AuditRefusal):audit(root)

    def test_manifest_fences_weights_gallery_and_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            report=write_run(d,training_worlds=2,heldout_worlds=5,gallery_per_arm=1)
            self.assertTrue(report['truth']['synthetic_only'])
            self.assertEqual(report['heldout_worlds'],5)
            self.assertEqual(report['train_seed_range'],[20000,20001])
            self.assertEqual(report['heldout_seed_range'],[97000,97004])
            self.assertEqual(len(report['training']),6)
            root=Path(d)
            for line in (root/'SHA256SUMS').read_text().splitlines():
                digest,rel=line.split('  ',1)
                self.assertEqual(digest,file_sha(root/rel))
            self.assertEqual(file_sha(root/'matched-episodes.json'),
                             report['original_heldout_rows_sha256'])
            self.assertEqual(len(list((root/'gallery').glob('*.png'))),8)
            result=audit(root)
            self.assertEqual(result['replayed_episodes'],40)
            self.assertGreater(result['replayed_actions'],0)

if __name__=='__main__':unittest.main()
