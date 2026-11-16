"""Recovery-balanced curriculum tests; no structural heldout evaluation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace
import numpy as np
from . import recovery_balance_lab as lab, frontier_viability_lab as fv
from .world import Spec, World, specs


class RecoveryBalanceTests(unittest.TestCase):
    def test_train_states_recovery_and_host_menu(self):
        X,mask,Y,weights,meta=lab.hard_cases(592,36,1)
        self.assertEqual(X.shape[1:],(7,39))
        self.assertTrue(np.all(mask[np.arange(len(Y)),Y]))
        self.assertGreater(sum((X[np.arange(len(Y)),Y,1]+X[np.arange(len(Y)),Y,2])>0),2)
        self.assertTrue(np.all(weights>0))
        self.assertEqual(meta['split'],'train')

    def test_same_initial_state_and_control_arms(self):
        a=fv.Ranker(592,'local');b=fv.Ranker(592,'local')
        self.assertTrue(np.array_equal(a.w,b.w))
        self.assertTrue(np.array_equal(a.m,b.m))
        self.assertEqual(a.updates,b.updates)

    def test_data_is_development_only(self):
        sp=specs('heldout',1)[0]
        m=fv.Ranker(592,'local')
        with self.assertRaises(ValueError):lab.episode(sp,m,592,'uniform',128,True)
        with self.assertRaises(ValueError):lab.episode(specs('train',1)[0],m,592,'uniform',128,True)

    def test_forced_six_east_are_legal(self):
        sp=replace(specs('development',1)[0],challenge='trap_corridor')
        w=World.create(sp)
        for t in range(6):
            a,i=lab.intervene(w,'MOVE_N',t,592,True)
            self.assertEqual(a,'MOVE_E')
            self.assertIn(a,w.menu())
            w.apply(a)
        self.assertEqual(w.cursor,(54,48))
        self.assertIn('BACKTRACK',w.menu())
        self.assertIn('RELOCATE_START',w.menu())

    def test_recovery_balanced_sampling_is_really_different(self):
        X,M,Y,W,_=lab.dataset(592,36)
        self.assertTrue(np.any(W>1.))
        a=fv.Ranker(592,'local');b=fv.Ranker(592,'local')
        ua=lab.fit(a,X,M,Y,W,12,592,False)
        ub=lab.fit(b,X,M,Y,W,12,592,True)
        self.assertEqual(a.updates,b.updates)
        self.assertGreater(ub['sampling']['recovery'],ua['sampling']['recovery'])
        self.assertFalse(np.array_equal(a.w,b.w))

    def test_source_replay_with_probabilities(self):
        sp=replace(specs('development',1)[0],challenge='trap_corridor')
        m=fv.Ranker(592,'local')
        e=lab.episode(sp,m,592,'uniform',128,True)
        self.assertEqual(lab.check_episode(e,m,sp),e['metrics']['decisions'])
        self.assertEqual(e['rows'][:6][-1]['event']['cursor_after'],[54,48])
        self.assertEqual(e['metrics']['illegal_host_actions'],0)
        self.assertEqual(e['metrics']['protected_cell_edits'],0)
        self.assertEqual(e['native_receipts'],0)

    def test_replay_rejects_probability_forgery(self):
        sp=specs('development',1)[0]
        m=fv.Ranker(592,'local');e=lab.episode(sp,m,592,'uniform',128,False)
        k=next(iter(e['rows'][0]['probabilities']))
        e['rows'][0]['probabilities'][k]=0.99999999
        with self.assertRaises(ValueError):lab.check_episode(e,m,sp)

    def test_replay_rejects_illegal_edit(self):
        sp=specs('development',1)[0]
        m=fv.Ranker(592,'local');e=lab.episode(sp,m,592,'uniform',128,False)
        e['rows'][0]['event']['painted']=[[1,1]]
        with self.assertRaises(ValueError):lab.check_episode(e,m,sp)


if __name__=='__main__':unittest.main()
