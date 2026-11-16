"""P13 adversarial tests; no paid GPU, native CyberSand, or root-SLM inference."""
from __future__ import annotations
import copy
import io
import json
import os
from pathlib import Path
import unittest
import zipfile
import torch
import p13_goal_conditional_stop as p

SOURCE=Path(os.environ.get('P13_P11_ARTIFACT', str(Path(__file__).parent.parent/'immutable-P11-source'/'P11-hosted-43918309938.zip')))

class GoalPreferenceTests(unittest.TestCase):
    def test_exhaustive_goal_contract(self):
        self.assertEqual(len(p.GOALS),4)
        self.assertEqual(len(set(p.GOALS)),4)
        with self.assertRaises(ValueError):p.accomplished({},'unknown')

    def test_cover_boundary(self):
        s={'novel':47};self.assertFalse(p.accomplished(s,'coverage48'))
        s['novel']=48;self.assertTrue(p.accomplished(s,'coverage48'))

    def test_scaffold_both_requirements(self):
        s={'novel':96,'branches':9};self.assertFalse(p.accomplished(s,'sculpture96_branch10'))
        s['branches']=10;self.assertTrue(p.accomplished(s,'sculpture96_branch10'))
        s['novel']=95;self.assertFalse(p.accomplished(s,'sculpture96_branch10'))

    def test_recovery_is_not_retrace_alone(self):
        s={'retraces':7,'post_retrace_new':9}
        self.assertFalse(p.accomplished(s,'recovery6_new10'))
        s['post_retrace_new']=10;self.assertTrue(p.accomplished(s,'recovery6_new10'))
        s['retraces']=5;self.assertFalse(p.accomplished(s,'recovery6_new10'))

    def test_landmark_boundary(self):
        self.assertTrue(p.position_in_landmark([30,45]))
        self.assertTrue(p.position_in_landmark([35,51]))
        self.assertFalse(p.position_in_landmark([29,45]))
        self.assertFalse(p.position_in_landmark([35,52]))

    def test_goal_feature_identity_is_explicit(self):
        s={'tick':6,'novel':4,'branches':1,'retraces':0,'post_retrace_new':0,
         'landmark':False,'x':11,'y':48,'recent_n':4,'recent_b':1,'recent_r':0}
        f=[p.feat(s,g) for g in p.GOALS]
        self.assertTrue(all(len(x)==21 for x in f))
        self.assertEqual(len({tuple(x) for x in f}),4)
        self.assertTrue(all(sum(x[:4])==1 for x in f))

    @unittest.skipUnless(SOURCE.exists(),'P11 original immutable source not mounted')
    def test_source_mutation_is_rejected(self):
        content=SOURCE.read_bytes()
        p.base.open_pinned(content)
        with self.assertRaises(ValueError):p.base.open_pinned(content[:-1]+bytes([content[-1]^1]))

    @unittest.skipUnless(SOURCE.exists(),'P11 original immutable source not mounted')
    def test_future_truncation_does_not_change_prefix_features(self):
        archive,host,head,sha=p.base.open_pinned(SOURCE.read_bytes())
        rec,_,_=p.base.extract_trace(archive,'greebles',2711)
        states=p.make_states(rec)
        t=45
        past=copy.deepcopy(rec);past['events']=past['events'][:t]
        past['trajectory']=past['trajectory'][:t+1]
        second=p.make_states(past)
        self.assertEqual(states[:t+1],second)
        for goal in p.GOALS:
            self.assertEqual(p.feat(states[t],goal),p.feat(second[t],goal))

    @unittest.skipUnless(SOURCE.exists(),'P11 original immutable source not mounted')
    def test_live_legal_stop_safety_and_original_reproduction(self):
        z,host,head,ck=p.base.open_pinned(SOURCE.read_bytes())
        rec,_,_=p.base.extract_trace(z,'greebles',2789)
        torch.manual_seed(314)
        head_goal=torch.nn.Sequential(torch.nn.Linear(21,24),torch.nn.Tanh(),
                              torch.nn.Linear(24,12),torch.nn.Tanh(),torch.nn.Linear(12,1))
        for mode in ('baseline_late','oracle','trained_shielded'):
            r=p.live_rollout(host,head,2789,'greebles','recovery6_new10',mode,head_goal,rec)
            self.assertTrue(all(e['raw_legal'] for e in r['events']))
            self.assertEqual(r['metrics']['invalid_raw_choices'],0)
            if mode=='baseline_late':
                self.assertEqual(r['trajectory'],rec['trajectory'])
                self.assertEqual(host.png_bytes(r),z.read('rollouts/greebles-2789-intact-late.png'))
            if mode in ('oracle','trained_shielded'):
                self.assertFalse(r['metrics']['terminated_by_stop'] and not r['goal_completed'])

    @unittest.skipUnless(SOURCE.exists(),'P11 original immutable source not mounted')
    def test_cohort_isolation(self):
        self.assertTrue(set(p.TRAIN).isdisjoint(p.VALIDATION))
        self.assertTrue(set(p.TRAIN).isdisjoint(p.HOLDOUT))
        self.assertTrue(set(p.VALIDATION).isdisjoint(p.HOLDOUT))

if __name__=='__main__':unittest.main()
